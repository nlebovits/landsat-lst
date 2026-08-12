#!/usr/bin/env python3
"""Tier 1 smoke test: build GeoZarr multiscale overviews from an EXISTING tile.

Reuses an already-computed Phase 0 Icechunk composite (no STAC, no recompute) to
exercise the real write path on real 18001x18001 data. It decodes the stored uint16
LST back to Celsius, feeds it through ``write_zarr`` (which re-encodes + builds the
pyramid), commits to a scratch repo, then validates GeoZarr structure and physical
sanity at every level.

The key invariant under test: overview ``lst_p95`` must stay in a physical range
(-20..60 degC). If fill (DN=0 -> -50 degC) leaked into the block means, the overview
minimum would collapse toward -50 -- this catches that.

Usage:
    uv run python scripts/smoke_tier1_overviews.py
    LST_SRC=results/phase0/N25E080/icechunk LST_GROUP=2024/N25E080 \\
      uv run python scripts/smoke_tier1_overviews.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import icechunk as ic
import xarray as xr
import zarr

from landsat_lst.config import settings
from landsat_lst.encoding import LST_NODATA_FLOAT, LST_OFFSET, LST_SCALE
from landsat_lst.zarr_writer import write_zarr

SRC = os.environ.get("LST_SRC", "results/phase0/N60W150/icechunk")
GROUP = os.environ.get("LST_GROUP", "2024/N60W150")


def _decode_to_float_composite(ds: xr.Dataset) -> xr.Dataset:
    """Decode stored uint16 LST DN back to a float Celsius composite.

    DN=0 is the fill value; map it to the pipeline's -9999 nodata so build_overviews
    excludes it from the block means. qa_count passes through unchanged.
    """
    scale = ds["lst_p95"].attrs.get("lst_scale_factor", LST_SCALE)
    offset = ds["lst_p95"].attrs.get("lst_add_offset", LST_OFFSET)
    dn = ds["lst_p95"]
    celsius = (dn * scale + offset).where(dn > 0, LST_NODATA_FLOAT).astype("float32")
    return xr.Dataset(
        {"lst_p95": celsius, "qa_count": ds["qa_count"]},
        coords={"latitude": ds["latitude"], "longitude": ds["longitude"]},
    )


def _validate_pyramid(dst_repo, native_h: int, native_qa_max: int) -> bool:
    """Validate the written pyramid's GeoZarr structure and physical sanity.

    An overview is a mean of VALID native pixels, so its values must stay within the
    native [min, max] envelope. Fill (DN=0 -> -50 degC) leaking into the block means
    would push an overview minimum well below native min -- that is the failure this
    detects. (Exact per-block masking is proven by
    tests/unit/test_geozarr_multiscale.py::test_build_overviews_excludes_fill.)
    """
    rs = dst_repo.readonly_session("main")
    parent = zarr.open_group(rs.store, path=GROUP, mode="r")
    for key in ("multiscales", "proj:code", "spatial:transform", "spatial:shape"):
        assert key in parent.attrs, f"parent missing GeoZarr attr {key!r}"
    layout = parent.attrs["multiscales"]["layout"]
    level_names = [e["asset"] for e in layout]
    assert level_names == ["0", *[str(i + 1) for i in range(len(settings.pyramid_factors))]]
    print(f"  multiscales layout: {[(e['asset'], e['transform']['scale'][0]) for e in layout]}")

    ok = True
    tol = 0.5  # degC, for rounding at the uint16 re-encode step
    native_vmin = native_vmax = None
    for name in level_names:
        lvl = xr.open_zarr(rs.store, group=f"{GROUP}/{name}", consolidated=False)
        dn = lvl["lst_p95"]
        valid = (dn.where(dn > 0) * LST_SCALE + LST_OFFSET).compute()
        vmin, vmax = float(valid.min()), float(valid.max())
        n_valid = int((dn > 0).sum().compute())
        qa_max = int(lvl["qa_count"].max().compute())
        h = int(lvl.sizes["latitude"])
        coarser = "native" if name == "0" else f"{native_h // h}x"
        status = "OK"
        if name == "0":
            native_vmin, native_vmax = vmin, vmax
            if h != native_h:
                status, ok = "BAD-SHAPE", False
        elif vmin < native_vmin - tol or vmax > native_vmax + tol:
            status, ok = "FILL-LEAK", False
        if n_valid == 0:
            status, ok = "ALL-FILL", False
        if qa_max > native_qa_max + 1:
            status, ok = "QA-INFLATED", False
        print(
            f"  level {name} ({coarser}): shape={h}x{int(lvl.sizes['longitude'])} "
            f"valid={n_valid} lst=[{vmin:.1f},{vmax:.1f}]degC qa_max={qa_max} -> {status}"
        )
    return ok


def main() -> int:
    print(f"Tier 1 smoke: source={SRC} group={GROUP} factors={settings.pyramid_factors}")

    # --- Read existing native composite (no recompute) -----------------------
    src_repo = ic.Repository.open(ic.local_filesystem_storage(SRC))
    src_ds = xr.open_zarr(src_repo.readonly_session("main").store, group=GROUP, consolidated=False)
    native_h = int(src_ds.sizes["latitude"])
    native_fill_frac = float((src_ds["lst_p95"] == 0).mean().compute())
    native_qa_max = int(src_ds["qa_count"].max().compute())
    print(f"  native: {dict(src_ds.sizes)} fill_frac={native_fill_frac:.3f} qa_max={native_qa_max}")

    composite = _decode_to_float_composite(src_ds)

    # --- Write the multiscale pyramid into a fresh scratch repo --------------
    scratch = tempfile.mkdtemp(prefix="lst_tier1_")
    try:
        dst_repo = ic.Repository.open_or_create(ic.local_filesystem_storage(scratch))
        session = dst_repo.writable_session("main")
        write_zarr(composite, session, group=GROUP)
        commit_id = session.commit("tier1 overview smoke")
        print(f"  committed {commit_id[:12]} -> {scratch}")

        ancestry = list(dst_repo.ancestry(branch="main"))
        assert len(ancestry) == 2, f"expected one pyramid commit, got {len(ancestry) - 1}"

        ok = _validate_pyramid(dst_repo, native_h, native_qa_max)
        print("TIER 1 PASSED" if ok else "TIER 1 FAILED")
        return 0 if ok else 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
