"""Propagate candidate stack representations through the P95 and the encoder.

Issue #136 asks, for each intermediate on the composite path, what precision the
published ``lst_p95`` (uint16, 0.01 C per DN) can justify. This script answers the
one question analysis cannot settle alone: how often does a narrower stack
representation flip an encoded DN, on real Landsat DN and real QA?

It loads one 512 x 512 window of the retained S30W065 fixture (300 scenes,
memory-mapped ``.npy``), runs the shipped path as the baseline, and compares
three alternatives that differ *only* in the stack's representation:

- ``float64``: control. Shows that float32 versus float64 does not change the
  encoded product, so the float64 kernel output narrowed at ``pipeline.py`` is
  lossless.
- ``uint16_dn``: the stack stays in native DN units, and each scene's offset is
  rounded to a whole DN (0.00341802 C). The only error is that rounding, bounded
  by half a DN step per sample.
- ``int16_cc``: the debiased Celsius value rounded to 0.01 C, held as int16.
  Bounded by 0.005 C per sample.

The P95 kernel structure is the shipped one (sort, virtual index, numpy's two-branch
lerp in float64); the alternatives use a sentinel-first sort because an integer
array has no NaN. ``qa_count`` is recomputed from the sentinel and asserted equal.

Bounded on purpose: one window, in memory, about 5.4 GB peak RSS and 72 s on a laptop
(xarray materializes several full-stack temporaries). Never widen it to the
full 2,250-square fixture; the memory question belongs to the probe and the VM.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/experimental/precision_audit_p95.py \
        --fixture results/fixtures/S30W065_2021-2025_n300_f8 --row 800 --col 800
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.encoding import LST_MIN_TRUSTED_C, LST_SCALE, encode_lst_uint16
from landsat_lst.kernels import nanquantile_last
from landsat_lst.pipeline import _composite_graph
from landsat_lst.qa import apply_qa_mask, convert_to_celsius

# Landsat Collection 2 Level-2 ST_B10 scaling, as written in qa.convert_to_celsius.
DN_SCALE_K = 0.00341802
DN_OFFSET_K = 149.0
KELVIN_TO_C = 273.15
Q = 0.95
# Calibrated offset spread (docs/findings-destriping-and-multiyear.md, Pergamino).
OFFSET_STD_C = 5.71


def _load_window(fixture: Path, row: int, col: int, edge: int) -> xr.Dataset:
    meta = json.loads((fixture / "meta.json").read_text())
    arrays = {}
    for band in ("lwir11", "qa_pixel"):
        mapped = np.load(fixture / f"{band}.npy", mmap_mode="r")
        arrays[band] = (
            ("time", "latitude", "longitude"),
            np.ascontiguousarray(mapped[:, row : row + edge, col : col + edge]),
        )
    return xr.Dataset(
        arrays,
        coords={
            "time": pd.to_datetime(meta["times"]),
            "latitude": np.arange(edge, dtype=np.float64),
            "longitude": np.arange(edge, dtype=np.float64),
        },
    )


def _offsets(n: int, seed: int) -> np.ndarray:
    """One float32 offset per scene, as the cache hands a shard (offsets.py:229)."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, OFFSET_STD_C, n)
    return raw.astype(np.float32)


def _lerp_last(s: np.ndarray, n: np.ndarray, first_valid: np.ndarray, q: float) -> np.ndarray:
    """Shipped kernel's index arithmetic on a sorted array whose valid run starts at
    ``first_valid`` along the last axis (NaN-last sort: 0; sentinel-first sort: T - n).

    Values are lifted to float64 before the lerp, which is what the shipped kernel
    does implicitly through ``np.float64(q)``.
    """
    last = s.shape[-1] - 1
    h = (n - 1) * np.float64(q)
    lo = np.clip(np.floor(h).astype(np.int64), 0, last)
    hi = np.clip(lo + 1, 0, np.clip(n - 1, 0, last))
    # A pixel with no valid sample has first_valid == T; clip so the gather stays in
    # bounds. Its result is overwritten with NaN below, as the shipped kernel does.
    ia = np.clip(first_valid + lo, 0, last)
    ib = np.clip(first_valid + hi, 0, last)
    a = np.take_along_axis(s, ia[..., None], -1)[..., 0].astype(np.float64)
    b = np.take_along_axis(s, ib[..., None], -1)[..., 0].astype(np.float64)
    t = h - lo
    diff = b - a
    out = np.asarray(a + diff * t, dtype=np.float64)
    np.subtract(b, diff * (1 - t), out=out, where=t >= 0.5)
    out[n == 0] = np.nan
    return out


def _finish(p95_c: np.ndarray, n: np.ndarray, coords: dict) -> np.ndarray:
    """The tail of ``_composite_graph`` and ``_encode_native`` on a 2-D array."""
    p95 = np.where(n > 0, p95_c, settings.nodata)
    anomalous = (n > 0) & (p95 < LST_MIN_TRUSTED_C)
    p95 = np.where(~anomalous, p95, settings.nodata).astype(np.float32)
    da = xr.DataArray(p95, dims=("latitude", "longitude"), coords=coords)
    return encode_lst_uint16(da).values


def _month_counts(valid: np.ndarray, months: np.ndarray) -> np.ndarray:
    out = np.zeros((12, *valid.shape[1:]), dtype=np.int64)
    for m in range(1, 13):
        out[m - 1] = valid[months == m].sum(axis=0)
    return out.astype(np.uint8)


def _compare(name: str, dn: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> dict:
    d = dn.astype(np.int64) - ref.astype(np.int64)
    both = valid & (dn != 0) & (ref != 0)
    abs_d = np.abs(d[both])
    return {
        "candidate": name,
        "pixels_compared": int(both.sum()),
        "fill_disagreements": int(((dn == 0) != (ref == 0)).sum()),
        "identical": int((abs_d == 0).sum()),
        "one_dn": int((abs_d == 1).sum()),
        "more_than_one_dn": int((abs_d > 1).sum()),
        "max_abs_dn": int(abs_d.max()) if abs_d.size else 0,
        "flip_fraction": float((abs_d >= 1).mean()) if abs_d.size else 0.0,
        "mean_signed_dn": float(d[both].mean()) if both.any() else 0.0,
    }


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--row", type=int, default=800)
    ap.add_argument("--col", type=int, default=800)
    ap.add_argument("--edge", type=int, default=512)
    ap.add_argument("--seed", type=int, default=136)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.monotonic()
    ds = _load_window(args.fixture, args.row, args.col, args.edge)
    scenes = ds.sizes["time"]

    # ---- Shipped path, verbatim functions ---------------------------------
    lst = convert_to_celsius(apply_qa_mask(ds)["lwir11"])
    assert lst.dtype == np.float32, lst.dtype
    offset32 = _offsets(scenes, args.seed)
    keep = np.abs(offset32) <= settings.destripe_max_offset_c
    lst = lst.isel(time=np.flatnonzero(keep))
    offset32 = offset32[keep]
    off_da = xr.DataArray(offset32, dims=("time",), coords={"time": lst.time})
    debiased32 = lst - off_da  # normalization.debias_with_offsets, the subtraction
    assert debiased32.dtype == np.float32, debiased32.dtype

    composite = _composite_graph(debiased32)
    assert composite["lst_p95"].dtype == np.float32
    assert composite["qa_count"].dtype == np.uint8
    p95_base_c = composite["lst_p95"].values
    coords = {"latitude": composite.latitude, "longitude": composite.longitude}
    dn_base = encode_lst_uint16(composite["lst_p95"]).values
    qa_base = composite["qa_count"].values

    valid = np.isfinite(debiased32.values)  # (T, y, x), the shipped NaN pattern
    n = valid.sum(axis=0)
    months = lst.time.dt.month.values
    valid_out = n > 0

    # Kernel sanity: the shipped kernel on the shipped stack equals the graph output.
    x32 = np.moveaxis(debiased32.values, 0, -1)
    k = nanquantile_last(np.ascontiguousarray(x32), Q)
    assert np.array_equal(_finish(k, n, coords), dn_base), (
        "harness does not reproduce _composite_graph"
    )

    results = []

    # ---- Control: float64 stack --------------------------------------------
    deb64 = lst.values.astype(np.float64) - offset32.astype(np.float64)[:, None, None]
    x64 = np.ascontiguousarray(np.moveaxis(deb64, 0, -1))
    p95_64 = nanquantile_last(x64, Q)
    dn_64 = _finish(p95_64, n, coords)
    r = _compare("float64_stack", dn_64, dn_base, valid_out)
    r["max_abs_delta_c"] = float(np.nanmax(np.abs(p95_64 - p95_base_c.astype(np.float64))))
    r["bytes_per_element"] = 8
    results.append(r)

    # ---- Candidate: uint16 in native DN, offset rounded to whole DN ---------
    dn_raw = ds["lwir11"].values[keep].astype(np.int64)  # (T, y, x) source DN
    shift = np.rint(offset32.astype(np.float64) / DN_SCALE_K).astype(np.int64)
    shifted = dn_raw - shift[:, None, None]
    lo_v, hi_v = int(shifted[valid].min()), int(shifted[valid].max())
    fits_u16 = lo_v >= 1 and hi_v <= 65535
    stack_u16 = np.where(valid, shifted, 0).astype(np.uint16 if fits_u16 else np.int32)
    xs = np.ascontiguousarray(np.moveaxis(stack_u16, 0, -1))
    s = np.sort(xs, axis=-1)  # sentinel 0 sorts first
    first_valid = xs.shape[-1] - n
    p95_dn = _lerp_last(s, n, first_valid, Q)
    p95_u16_c = p95_dn * DN_SCALE_K + DN_OFFSET_K - KELVIN_TO_C
    dn_u16 = _finish(p95_u16_c, n, coords)
    qa_u16 = _month_counts(stack_u16 != 0, months)
    r = _compare("uint16_dn_stack", dn_u16, dn_base, valid_out)
    r["max_abs_delta_c"] = float(np.nanmax(np.abs(p95_u16_c - p95_base_c.astype(np.float64))))
    r["bytes_per_element"] = 2
    r["shifted_dn_range"] = [lo_v, hi_v]
    r["fits_uint16"] = fits_u16
    r["offset_rounding_max_c"] = float(np.abs(offset32 - shift * DN_SCALE_K).max())
    r["predicted_flip_fraction"] = float(np.abs(offset32 - shift * DN_SCALE_K).mean() / LST_SCALE)
    r["qa_count_equal"] = bool(np.array_equal(qa_u16, qa_base))
    results.append(r)

    # ---- Candidate: int16 at 0.01 C -----------------------------------------
    cc = np.rint(debiased32.values.astype(np.float64) * 100.0)
    stack_i16 = np.where(valid, cc, -32768).astype(np.int16)
    xs = np.ascontiguousarray(np.moveaxis(stack_i16, 0, -1))
    s = np.sort(xs, axis=-1)  # sentinel -32768 sorts first
    p95_cc = _lerp_last(s, n, first_valid, Q)
    p95_i16_c = p95_cc / 100.0
    dn_i16 = _finish(p95_i16_c, n, coords)
    qa_i16 = _month_counts(stack_i16 != -32768, months)
    r = _compare("int16_centicelsius_stack", dn_i16, dn_base, valid_out)
    r["max_abs_delta_c"] = float(np.nanmax(np.abs(p95_i16_c - p95_base_c.astype(np.float64))))
    r["bytes_per_element"] = 2
    r["predicted_flip_fraction"] = float(
        np.abs(debiased32.values[valid].astype(np.float64) - cc[valid] / 100.0).mean() / LST_SCALE
    )
    r["qa_count_equal"] = bool(np.array_equal(qa_i16, qa_base))
    results.append(r)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report = {
        "fixture": str(args.fixture),
        "window": {"row": args.row, "col": args.col, "edge": args.edge},
        "scenes_loaded": int(scenes),
        "scenes_kept": int(keep.sum()),
        "valid_samples": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "pixels_with_output": int(valid_out.sum()),
        "median_valid_per_pixel": float(np.median(n[valid_out])) if valid_out.any() else 0.0,
        "p95_celsius_range": [
            float(np.nanmin(p95_base_c[valid_out])),
            float(np.nanmax(p95_base_c[valid_out])),
        ],
        "seed": args.seed,
        "wall_s": round(time.monotonic() - t0, 1),
        "peak_rss_mb": round(peak_mb, 1),
        "numpy": np.__version__,
        "xarray": xr.__version__,
        "results": results,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
