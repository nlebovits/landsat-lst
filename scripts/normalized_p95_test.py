#!/usr/bin/env python3
"""Test: 1-year P95 with per-scene relative normalization (striping fix).

Levels each scene to a common per-pixel reference before taking P95, so the value
no longer jumps where the contributing scene-set changes (WRS footprint seams):

    ref(x,y)  = median over time of all scenes
    offset_i  = median over scene i's valid pixels of (scene_i - ref)   # per-scene bias
    scene_i'  = scene_i - offset_i                                       # leveled
    P95'      = 95th percentile over time of the leveled scenes

Writes results/decision/percentile_test/lst_p95_norm_2024_S30W065.tif for a direct
QGIS comparison against the un-normalized lst_p95_2024_S30W065.tif. Same 1deg AOI,
Planetary Computer, plain Dask.

Usage:
    LST_LOAD_CHUNK_SIZE=256 uv run python scripts/normalized_p95_test.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pystac_client
import structlog
import xarray as xr

from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings

settings.stac_url = STAC_PLANETARY_COMPUTER  # local rule: PC (free, no egress)

from landsat_lst.azure_auth import enable_pc_azure_refresh  # noqa: E402
from landsat_lst.cog import export_lst_cog  # noqa: E402
from landsat_lst.encoding import encode_lst_uint16  # noqa: E402
from landsat_lst.pipeline import load_scenes  # noqa: E402
from landsat_lst.qa import apply_qa_mask, convert_to_celsius  # noqa: E402

log = structlog.get_logger()

AOI_BBOX = (-61.1, -34.4, -60.1, -33.4)  # ~1deg around Pergamino
YEAR = 2024
OUT = Path("results/decision/percentile_test/lst_p95_norm_2024_S30W065.tif")


def main() -> int:
    from dask.distributed import Client, LocalCluster  # noqa: PLC0415

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cluster = LocalCluster(
        n_workers=settings.dask_workers,
        threads_per_worker=settings.dask_threads_per_worker,
        memory_limit=settings.dask_memory_limit,
        dashboard_address=":8787",
    )
    client = Client(cluster)
    log.info("dask_ready", chunk=settings.load_chunk_size)

    try:
        catalog = pystac_client.Client.open(settings.stac_url)
        items = list(
            catalog.search(
                collections=[settings.collection],
                bbox=AOI_BBOX,
                datetime=f"{YEAR}-01-01/{YEAR}-12-31",
                query={
                    "eo:cloud_cover": {"lt": settings.max_cloud_cover},
                    "platform": {"in": ["landsat-8", "landsat-9"]},
                },
            ).items()
        )
        log.info("scenes_found", n=len(items))
        if not items:
            msg = "no scenes found"
            raise ValueError(msg)

        patch_url = enable_pc_azure_refresh(items)
        data = load_scenes(items, AOI_BBOX, patch_url=patch_url)
        lst = convert_to_celsius(apply_qa_mask(data)["lwir11"])  # (time, lat, lon) degC
        # Drop physically impossible LST (~-124 degC from resampling near DN=0 fill;
        # see CLAUDE.md) so edge junk can't poison the reference or per-scene offsets.
        lst = lst.where((lst > -20) & (lst < 65))

        # --- Per-scene relative normalization -------------------------------
        ref = lst.median(dim="time", skipna=True)  # robust per-pixel reference
        anom = (lst - ref).median(dim=["latitude", "longitude"], skipna=True)  # (time,)
        # Only shift scenes with enough valid overlap; never let a NaN offset
        # delete a scene (fillna 0 = leave it unshifted).
        n_valid = lst.notnull().sum(dim=["latitude", "longitude"])
        offset = anom.where(n_valid > 500, 0.0).fillna(0.0)
        lst_norm = lst - offset

        valid = lst.notnull().sum(dim="time")
        p95 = lst_norm.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")
        p95 = p95.where(valid > 0, settings.nodata)

        log.info("computing")
        p95, offsets = xr.Dataset({"p95": p95}).compute(), offset.compute()
        p95 = p95["p95"]
        ov = offsets.values
        log.info(
            "per_scene_offsets_degC",
            n=int(np.isfinite(ov).sum()),
            std=round(float(np.nanstd(ov)), 2),
            min=round(float(np.nanmin(ov)), 2),
            max=round(float(np.nanmax(ov)), 2),
        )

        ds = xr.Dataset(
            {"lst_p95": encode_lst_uint16(p95)},
            coords={"latitude": p95.latitude, "longitude": p95.longitude},
        )
        export_lst_cog(ds, OUT)
        valid_c = p95.where(p95 != settings.nodata)
        log.info(
            "wrote_cog",
            path=str(OUT),
            degc_min=round(float(valid_c.min()), 1),
            degc_mean=round(float(valid_c.mean()), 1),
            degc_max=round(float(valid_c.max()), 1),
        )
    finally:
        client.close()
        cluster.close()

    print(f"\nNormalized P95 COG: {OUT}")
    print(
        "Compare in QGIS against: results/decision/lst_2024_S30W065.tif "
        "(baseline P95, full-window) or percentile_test/lst_p95_2024_S30W065.tif"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
