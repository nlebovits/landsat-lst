#!/usr/bin/env python3
"""Striping diagnostic: P50/P75/P90/P95 LST composites from one loaded stack.

Computes several percentiles of the SAME 1-year Pergamino AOI scene stack in a
single pass (the scene load dominates; extra quantiles are nearly free) and exports
each as a single-band LST COG so striping-vs-warmth can be compared in QGIS.

Purpose: decide whether the persistent WRS striping is estimator-tail variance
(shrinks at lower percentiles -> just pick a lower percentile) or genuine
per-scene acquisition bias (persists even at P50 -> needs bias normalization).

Runs locally against Planetary Computer. Plain Dask (no Frisky).

Usage:
    LST_LOAD_CHUNK_SIZE=256 uv run python scripts/percentile_striping_test.py
"""

from __future__ import annotations

from pathlib import Path

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
QUANTILES = [0.50, 0.75, 0.90, 0.95]
OUT_DIR = Path("results/decision/percentile_test")


def main() -> int:
    from dask.distributed import Client, LocalCluster  # noqa: PLC0415

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster = LocalCluster(
        n_workers=settings.dask_workers,
        threads_per_worker=settings.dask_threads_per_worker,
        memory_limit=settings.dask_memory_limit,
        dashboard_address=":8787",
    )
    client = Client(cluster)
    log.info("dask_ready", dashboard=":8787", chunk=settings.load_chunk_size)

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

        # All percentiles in one reduction over the shared (loaded) stack.
        pct = lst.quantile(QUANTILES, dim="time", skipna=True)
        valid = lst.notnull().sum(dim="time")
        pct = pct.where(valid > 0, settings.nodata)

        log.info("computing", quantiles=QUANTILES)
        pct = pct.compute()
        log.info("compute_done", size_gb=round(pct.nbytes / 1e9, 2))

        for q in QUANTILES:
            da = pct.sel(quantile=q).drop_vars("quantile")
            # Reuse the LST COG exporter (expects an "lst_p95"-named uint16 DN var).
            ds = xr.Dataset(
                {"lst_p95": encode_lst_uint16(da)},
                coords={"latitude": da.latitude, "longitude": da.longitude},
            )
            out = OUT_DIR / f"lst_p{int(q * 100)}_{YEAR}_S30W065.tif"
            export_lst_cog(ds, out)
            valid_c = da.where(da != settings.nodata)
            log.info(
                "wrote_cog",
                percentile=f"p{int(q * 100)}",
                path=str(out),
                degc_min=round(float(valid_c.min()), 1),
                degc_mean=round(float(valid_c.mean()), 1),
                degc_max=round(float(valid_c.max()), 1),
            )
    finally:
        client.close()
        cluster.close()

    print(f"\nPercentile COGs written to: {OUT_DIR}/")
    for q in QUANTILES:
        print(f"  lst_p{int(q * 100)}_{YEAR}_S30W065.tif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
