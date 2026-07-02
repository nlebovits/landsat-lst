#!/usr/bin/env python3
"""1-year P95 with improved QA masking (cirrus + dilated cloud + physical clamp).

Plain P95 on the ~1deg Pergamino AOI using the updated qa.py (dilated-cloud/cirrus
bits now masked; implausible LST clamped). No normalization -- absolute hot signal
preserved. Writes lst_p95_masked_2024_S30W065.tif for a QGIS comparison against the
old-masking baseline lst_p95_2024_S30W065.tif.

Usage:
    LST_LOAD_CHUNK_SIZE=256 uv run python scripts/masked_p95_test.py
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
from landsat_lst.pipeline import load_scenes  # noqa: E402
from landsat_lst.qa import apply_qa_mask, convert_to_celsius  # noqa: E402
from landsat_lst.zarr_writer import encode_lst_uint16  # noqa: E402

log = structlog.get_logger()

AOI_BBOX = (-61.1, -34.4, -60.1, -33.4)  # ~1deg around Pergamino
YEAR = 2024
# Name encodes the scene cloud-cover threshold (LST_MAX_CLOUD_COVER) so successive
# runs don't clobber each other: cc100 = no scene filter, cc70 = drop >70% cloud.
OUT = Path(
    f"results/decision/percentile_test/lst_p95_masked_cc{int(settings.max_cloud_cover)}"
    "_2024_S30W065.tif"
)


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
        # Improved masking (dilated cloud + cirrus) and physical clamp applied inside.
        lst = convert_to_celsius(apply_qa_mask(data)["lwir11"])

        valid = lst.notnull().sum(dim="time")
        p95 = lst.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")
        p95 = p95.where(valid > 0, settings.nodata)

        log.info("computing")
        p95 = p95.compute()

        ds = xr.Dataset(
            {"lst_p95": encode_lst_uint16(p95)},
            coords={"latitude": p95.latitude, "longitude": p95.longitude},
        )
        export_lst_cog(ds, OUT)
        vc = p95.where(p95 != settings.nodata)
        log.info(
            "wrote_cog",
            path=str(OUT),
            degc_min=round(float(vc.min()), 1),
            degc_mean=round(float(vc.mean()), 1),
            degc_max=round(float(vc.max()), 1),
        )
    finally:
        client.close()
        cluster.close()

    print(f"\nMasked P95 COG: {OUT}")
    print(
        "Compare in QGIS vs old masking: results/decision/percentile_test/lst_p95_2024_S30W065.tif"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
