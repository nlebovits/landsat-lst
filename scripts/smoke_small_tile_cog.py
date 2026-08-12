#!/usr/bin/env python3
"""Smoke test: full pipeline on a SMALL bbox -> COG pair.

Runs locally against Planetary Computer (no AWS egress) on a small slice of Pergamino,
Argentina -- cheap enough to confirm the whole chain works end to end without
recomputing a 5-degree tile:

    STAC query -> scene load -> annual P95 composite
      -> uint16 encoding
      -> cog_export (single-band LST COG + 12-band monthly QA COG)

Uses the lower-level pipeline functions with a custom bbox (not process_tile, which is
fixed to 5-degree tiles). No land mask -- Pergamino is fully inland.

Usage:
    uv run python scripts/smoke_small_tile_cog.py
"""

from __future__ import annotations

from pathlib import Path

import pystac_client
import rasterio
import xarray as xr

from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings

# Local rule: Planetary Computer endpoint (free, no egress). Set before any query.
settings.stac_url = STAC_PLANETARY_COMPUTER

from landsat_lst.azure_auth import enable_pc_azure_refresh  # noqa: E402
from landsat_lst.cog import cog_export  # noqa: E402
from landsat_lst.encoding import encode_lst_uint16  # noqa: E402
from landsat_lst.pipeline import compute_annual_composite, load_scenes  # noqa: E402

# A small (~0.2 deg, ~700px at 30m) slice of Pergamino, Argentina.
BBOX = (-60.60, -33.95, -60.40, -33.75)
YEAR = 2024
OUT_DIR = Path("output/smoke_small")
LST_COG = OUT_DIR / "pergamino_small_lst_p95.tif"
QA_COG = OUT_DIR / "pergamino_small_qa_count.tif"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"endpoint={settings.stac_url}")
    print(f"bbox={BBOX} year={YEAR}")

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

    # --- Encode to the published uint16 contract, then export both COGs ------
    native = xr.Dataset(
        {
            "lst_p95": encode_lst_uint16(composite["lst_p95"]),
            "qa_count": composite["qa_count"],
        }
    )
    cog_export(native, LST_COG, QA_COG)

    for path in (LST_COG, QA_COG):
        with rasterio.open(path) as src:
            print(
                f"{path.name}: {src.count}x{src.width}x{src.height} {src.dtypes[0]} "
                f"overviews={src.overviews(1)} nodata={src.nodata} "
                f"scales={src.scales} offsets={src.offsets}"
            )
    print(f"COGs written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
