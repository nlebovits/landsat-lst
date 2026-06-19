#!/usr/bin/env python3
"""Smoke test: full pipeline on a SMALL bbox -> GeoZarr multiscale icechunk -> COG.

Runs locally against Planetary Computer (no AWS egress) on a small slice of Pergamino,
Argentina -- cheap enough to confirm the whole chain works end to end without
recomputing a 5-degree tile:

    STAC query -> scene load -> annual P95 composite
      -> write_zarr (GeoZarr multiscale pyramid, Blosc, atomic icechunk commit)
      -> read native level back
      -> derive a Cloud-Optimized GeoTIFF (COG) from lst_p95

Uses the lower-level pipeline functions with a custom bbox (not process_tile, which is
fixed to 5-degree tiles). No land mask -- Pergamino is fully inland.

Usage:
    uv run python scripts/smoke_small_tile_cog.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import icechunk as ic
import pystac_client
import rasterio
import rioxarray  # noqa: F401 - .rio accessor
import xarray as xr
import zarr
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles

from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings

# Local rule: Planetary Computer endpoint (free, no egress). Set before any query.
settings.stac_url = STAC_PLANETARY_COMPUTER

from landsat_lst.azure_auth import enable_pc_azure_refresh  # noqa: E402
from landsat_lst.pipeline import compute_annual_composite, load_scenes  # noqa: E402
from landsat_lst.zarr_writer import LST_OFFSET, LST_SCALE, write_zarr  # noqa: E402

# A small (~0.2 deg, ~700px at 30m) slice of Pergamino, Argentina.
BBOX = (-60.60, -33.95, -60.40, -33.75)
YEAR = 2024
GROUP = f"{YEAR}/PERGAMINO_SMALL"
OUT_DIR = Path("output/smoke_small")
COG_PATH = OUT_DIR / "pergamino_small_lst_p95_cog.tif"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"endpoint={settings.stac_url}")
    print(f"bbox={BBOX} year={YEAR} factors={settings.pyramid_factors}")

    # --- STAC query + load + composite (Planetary Computer) ------------------
    catalog = pystac_client.Client.open(settings.stac_url)
    items = list(
        catalog.search(
            collections=[settings.collection],
            bbox=BBOX,
            datetime=f"{YEAR}-01-01/{YEAR}-12-31",
            query={
                "eo:cloud_cover": {"lt": settings.max_cloud_cover},
                "platform": {"in": ["landsat-8", "landsat-9"]},
            },
        ).items()
    )
    print(f"scenes: {len(items)}")
    assert items, "no scenes found for bbox/year"

    patch_url = enable_pc_azure_refresh(items)  # token-free /vsiaz hrefs for PC reads
    data = load_scenes(items, BBOX, patch_url=patch_url)
    composite = compute_annual_composite(data).compute()

    valid = composite["lst_p95"].where(composite["lst_p95"] != settings.nodata)
    print(
        f"composite: {dict(composite.sizes)} "
        f"lst=[{float(valid.min()):.1f},{float(valid.max()):.1f}]degC "
        f"qa_max={int(composite['qa_count'].max())}"
    )

    # --- Write GeoZarr multiscale pyramid to a local icechunk repo -----------
    repo = ic.Repository.open_or_create(ic.local_filesystem_storage(str(OUT_DIR / "icechunk")))
    session = repo.writable_session("main")
    write_zarr(composite, session, group=GROUP)
    commit_id = session.commit("small-tile smoke")
    print(f"icechunk commit: {commit_id[:12]}")

    rs = repo.readonly_session("main")
    parent = zarr.open_group(rs.store, path=GROUP, mode="r")
    levels = [e["asset"] for e in parent.attrs["multiscales"]["layout"]]
    print(f"multiscale levels: {levels}  (proj={parent.attrs['proj:code']})")

    # --- Derive a COG from native-resolution lst_p95 (rio-cogeo) -------------
    ds = xr.open_zarr(rs.store, group=f"{GROUP}/0")
    da = (
        ds["lst_p95"]
        .rio.write_crs("EPSG:4326")
        .rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        .rio.write_nodata(0)
    )

    scratch = Path(tempfile.mkdtemp(prefix="lst_cog_"))
    try:
        src_tif = scratch / "src.tif"
        da.rio.to_raster(src_tif)  # plain GeoTIFF source for cog_translate
        # Embed GDAL band scale/offset so viewers (QGIS, gdalinfo) auto-decode the
        # uint16 DN to Celsius: value = DN * scale + offset.
        with rasterio.open(src_tif, "r+") as src:
            src.scales = (LST_SCALE,)
            src.offsets = (LST_OFFSET,)
        cog_translate(
            str(src_tif),
            str(COG_PATH),
            cog_profiles.get("deflate"),
            overview_resampling="average",
            nodata=0,
            quiet=True,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    is_valid, errors, warnings = cog_validate(str(COG_PATH))
    print(f"wrote COG: {COG_PATH}  (decode: degC = DN*{LST_SCALE} + ({LST_OFFSET}))")
    print(f"cog_validate: valid={is_valid} errors={errors} warnings={warnings}")
    assert is_valid, f"output is not a valid COG: {errors}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
