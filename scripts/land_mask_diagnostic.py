#!/usr/bin/env python3
"""Diagnostic script to visualize land mask coverage with and without buffer.

Generates comparison rasters showing original NE 10m mask vs buffered mask,
useful for validating coastal coverage before running full tile processing.

Usage:
    uv run python scripts/land_mask_diagnostic.py --tile N40W075
    uv run python scripts/land_mask_diagnostic.py --tile N40W075 --buffer 50000
    uv run python scripts/land_mask_diagnostic.py --bbox -75,35,-70,40

Output:
    - mask_original.tif: Natural Earth 10m mask (no buffer)
    - mask_buffered.tif: Mask with coastal buffer applied
    - mask_diff.tif: Pixels gained by buffer (value=1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
import rasterio.transform

# rioxarray needed for .rio accessor
import rioxarray  # noqa: F401
import structlog
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.masks import COASTAL_BUFFER_METERS, NATURAL_EARTH_URL
from landsat_lst.tiling import parse_tile_name

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()


def load_raw_land_polygons() -> gpd.GeoDataFrame:
    """Load Natural Earth 10m land polygons without any buffer."""
    log.info("loading_land_polygons", source="Natural Earth 10m")
    land = gpd.read_file(NATURAL_EARTH_URL)
    return land.to_crs("EPSG:4326")


def buffer_polygons(gdf: gpd.GeoDataFrame, buffer_meters: int) -> gpd.GeoDataFrame:
    """Buffer polygons by specified distance in meters."""
    log.info("buffering_polygons", buffer_km=buffer_meters // 1000)
    projected = gdf.to_crs("EPSG:3857")
    projected["geometry"] = projected.geometry.buffer(buffer_meters)
    return projected.to_crs("EPSG:4326")


def rasterize_mask(
    gdf: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    resolution: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Rasterize polygons to a boolean mask."""
    west, south, east, north = bbox
    width = int((east - west) / resolution)
    height = int((north - south) / resolution)

    transform = rasterio.transform.from_bounds(west, south, east, north, width, height)

    mask = rasterio.features.rasterize(
        gdf.geometry,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )

    return mask, (height, width)


def export_mask(
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    resolution: float,
) -> None:
    """Export mask as COG."""
    west, south, east, north = bbox
    height, width = mask.shape

    lats = np.linspace(north - resolution / 2, south + resolution / 2, height)
    lons = np.linspace(west + resolution / 2, east - resolution / 2, width)

    da = xr.DataArray(
        mask,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
    )
    da = da.rio.write_crs("EPSG:4326")
    da.rio.to_raster(output_path, driver="COG", compress="DEFLATE")
    log.info("wrote_mask", path=str(output_path), size_mb=output_path.stat().st_size / 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize land mask coverage with and without buffer"
    )
    parser.add_argument(
        "--tile",
        type=str,
        help="Tile name (e.g., N40W075). Mutually exclusive with --bbox.",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        help="Bounding box as west,south,east,north. Mutually exclusive with --tile.",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=COASTAL_BUFFER_METERS,
        help=f"Buffer distance in meters (default: {COASTAL_BUFFER_METERS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mask-diagnostic"),
        help="Output directory for mask rasters",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=settings.source_resolution,
        help=f"Pixel resolution in degrees (default: {settings.source_resolution})",
    )

    args = parser.parse_args()

    if args.tile and args.bbox:
        parser.error("--tile and --bbox are mutually exclusive")
    if not args.tile and not args.bbox:
        parser.error("Either --tile or --bbox is required")

    if args.tile:
        tile = parse_tile_name(args.tile)
        bbox = tile.bbox
        tile_name = args.tile
    else:
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            parser.error("--bbox must be west,south,east,north")
        bbox = tuple(parts)
        tile_name = "custom"

    log.info(
        "starting_diagnostic",
        tile=tile_name,
        bbox=bbox,
        buffer_km=args.buffer // 1000,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    # Load and process
    land_raw = load_raw_land_polygons()
    land_buffered = buffer_polygons(land_raw, args.buffer)

    # Rasterize both
    log.info("rasterizing_masks", resolution=args.resolution)
    mask_orig, _ = rasterize_mask(land_raw, bbox, args.resolution)
    mask_buff, _ = rasterize_mask(land_buffered, bbox, args.resolution)

    # Calculate diff
    diff = mask_buff.astype(np.int8) - mask_orig.astype(np.int8)

    # Stats
    orig_pixels = int(mask_orig.sum())
    buff_pixels = int(mask_buff.sum())
    gained = int((diff > 0).sum())

    log.info(
        "coverage_stats",
        original_pixels=f"{orig_pixels:,}",
        buffered_pixels=f"{buff_pixels:,}",
        gained_pixels=f"{gained:,}",
        gained_pct=f"{100 * gained / orig_pixels:.1f}%" if orig_pixels > 0 else "N/A",
    )

    # Export
    export_mask(mask_orig, bbox, args.output / "mask_original.tif", args.resolution)
    export_mask(mask_buff, bbox, args.output / "mask_buffered.tif", args.resolution)
    export_mask(diff, bbox, args.output / "mask_diff.tif", args.resolution)

    print(f"\nOutput written to: {args.output}/")
    print("  - mask_original.tif: Natural Earth 10m (no buffer)")
    print(f"  - mask_buffered.tif: With {args.buffer // 1000}km buffer")
    print("  - mask_diff.tif: Pixels gained by buffer (value=1)")


if __name__ == "__main__":
    main()
