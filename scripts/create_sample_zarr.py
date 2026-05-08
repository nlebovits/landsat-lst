#!/usr/bin/env python3
"""Create sample Zarr for QGIS plugin validation (Issue #15).

Creates a synthetic LST Zarr with 2 tiles and uploads to Source Coop.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import fsspec
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS

# Constants from landsat_lst.cog
LST_SCALE = 0.01
LST_OFFSET = -50.0

# Tile specifications
TILES = ["N40W075", "N45W075"]
TILE_SIZE_DEG = 5.0
PIXELS_PER_TILE = 500
CHUNK_SIZE = 500

# Output paths
LOCAL_PATH = Path("/tmp/sample.zarr")
S3_BUCKET = "us-west-2.opendata.source.coop"
S3_PREFIX = "nlebovits/landsat-lst-test/sample.zarr"


def parse_tile_name(name: str) -> tuple[int, int]:
    """Parse tile name to (lat, lon) of SW corner."""
    match = re.match(r"^([NS])(\d{2})([EW])(\d{3})$", name)
    if not match:
        raise ValueError(f"Invalid tile name: {name}")

    lat_dir, lat_val, lon_dir, lon_val = match.groups()
    lat = int(lat_val) * (1 if lat_dir == "N" else -1)
    lon = int(lon_val) * (1 if lon_dir == "E" else -1)
    return lat, lon


def create_tile_coords(tile_name: str, n_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """Create lat/lon coordinate arrays for a tile.

    Latitude runs north-to-south (descending), longitude west-to-east (ascending).
    """
    lat_start, lon_start = parse_tile_name(tile_name)

    lat = np.linspace(lat_start + TILE_SIZE_DEG, lat_start, n_pixels, endpoint=False)
    lon = np.linspace(lon_start, lon_start + TILE_SIZE_DEG, n_pixels, endpoint=False)
    return lat, lon


def create_synthetic_lst(n_pixels: int, seed: int) -> np.ndarray:
    """Create synthetic LST data as uint16.

    Generates realistic urban heat patterns with:
    - Base temperature gradient (north cooler than south)
    - Urban heat island effect (center warmer)
    - Random noise
    """
    rng = np.random.default_rng(seed)

    y = np.arange(n_pixels)
    x = np.arange(n_pixels)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    base_temp = 25.0 + (yy / n_pixels) * 5.0

    center = n_pixels / 2
    dist_from_center = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    urban_heat = 3.0 * np.exp(-dist_from_center / (n_pixels / 4))

    noise = rng.normal(0, 1, (n_pixels, n_pixels))

    celsius = base_temp + urban_heat + noise

    dn = ((celsius - LST_OFFSET) / LST_SCALE).clip(1, 65535).astype(np.uint16)
    return dn


def create_synthetic_qa(n_pixels: int, seed: int) -> np.ndarray:
    """Create synthetic QA count data (0-365 observations)."""
    rng = np.random.default_rng(seed)
    return rng.integers(50, 300, size=(n_pixels, n_pixels), dtype=np.uint16)


def create_tile_dataset(tile_name: str, seed: int) -> xr.Dataset:
    """Create xarray Dataset for a single tile.

    Note: We store raw uint16 values and put scale_factor/add_offset in attrs
    for GDAL/QGIS to read. We DON'T use xarray's CF encoding because that
    auto-decodes on read, losing the uint16 storage format.
    """
    lat, lon = create_tile_coords(tile_name, PIXELS_PER_TILE)

    lst_p50 = create_synthetic_lst(PIXELS_PER_TILE, seed)
    lst_p95 = create_synthetic_lst(PIXELS_PER_TILE, seed + 1000)
    qa_count = create_synthetic_qa(PIXELS_PER_TILE, seed + 2000)

    ds = xr.Dataset(
        {
            "lst_p50": (["latitude", "longitude"], lst_p50),
            "lst_p95": (["latitude", "longitude"], lst_p95),
            "qa_count": (["latitude", "longitude"], qa_count),
        },
        coords={
            "latitude": lat,
            "longitude": lon,
            "time": np.datetime64("2023-01-01"),
        },
    )

    for var in ["lst_p50", "lst_p95"]:
        ds[var].attrs.update(
            {
                "lst_scale_factor": LST_SCALE,
                "lst_add_offset": LST_OFFSET,
                "units": "celsius",
                "long_name": f"Land Surface Temperature ({'median' if 'p50' in var else '95th percentile'})",
                "missing_value": 0,
                "dtype": "uint16",
            }
        )

    ds["qa_count"].attrs.update(
        {
            "units": "count",
            "long_name": "Number of valid observations",
            "missing_value": 0,
            "dtype": "uint16",
        }
    )

    crs = CRS.from_epsg(4326)
    ds.attrs["_CRS"] = crs.to_wkt()
    ds.attrs["crs"] = "EPSG:4326"
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["title"] = f"Landsat LST Sample - Tile {tile_name}"

    return ds


def create_sample_zarr() -> Path:
    """Create sample Zarr with multiple tiles as separate groups."""
    if LOCAL_PATH.exists():
        shutil.rmtree(LOCAL_PATH)

    print(f"Creating sample Zarr at {LOCAL_PATH}")

    for i, tile_name in enumerate(TILES):
        print(f"  Creating tile {tile_name}...")
        ds = create_tile_dataset(tile_name, seed=i * 10000)

        tile_path = LOCAL_PATH / tile_name
        encoding = {
            var: {"chunks": (CHUNK_SIZE, CHUNK_SIZE), "dtype": "uint16"}
            for var in ["lst_p50", "lst_p95", "qa_count"]
        }

        ds.to_zarr(tile_path, mode="w", encoding=encoding)
        print(f"    Saved to {tile_path}")

    print(f"\nSample Zarr created at {LOCAL_PATH}")
    return LOCAL_PATH


def verify_local_zarr(zarr_path: Path) -> bool:
    """Verify Zarr opens correctly with xarray and has proper CRS."""
    print("\nVerifying local Zarr...")

    for tile_name in TILES:
        tile_path = zarr_path / tile_name
        print(f"  Checking tile {tile_name}...")

        ds = xr.open_zarr(tile_path)

        assert "lst_p50" in ds, "Missing lst_p50 variable"
        assert "lst_p95" in ds, "Missing lst_p95 variable"
        assert "qa_count" in ds, "Missing qa_count variable"

        assert "latitude" in ds.coords, "Missing latitude coordinate"
        assert "longitude" in ds.coords, "Missing longitude coordinate"
        assert "time" in ds.coords, "Missing time coordinate"

        assert ds["lst_p50"].attrs.get("lst_scale_factor") == LST_SCALE
        assert ds["lst_p50"].attrs.get("lst_add_offset") == LST_OFFSET

        assert "_CRS" in ds.attrs, "Missing _CRS attribute"

        print(f"    Variables: {list(ds.data_vars)}")
        print(f"    Dimensions: {dict(ds.sizes)}")
        print(f"    dtype: {ds['lst_p50'].dtype} (raw zarr: uint16)")
        print(f"    CRS: {ds.attrs.get('crs')}")

        ds.close()

    print("Local Zarr verification passed!")
    print("  Note: xarray auto-converts uint16 to float32 on read; underlying zarr is uint16")
    return True


def upload_to_source_coop(zarr_path: Path) -> str:
    """Upload Zarr to Source Coop S3."""
    s3_url = f"s3://{S3_BUCKET}/{S3_PREFIX}"
    print(f"\nUploading to {s3_url}...")

    cmd = [
        "aws",
        "s3",
        "sync",
        str(zarr_path),
        s3_url,
        "--profile",
        "source-coop",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f"Upload failed: {result.stderr}")
        raise RuntimeError(f"S3 upload failed: {result.stderr}")

    print("Upload complete!")
    return s3_url


def verify_remote_zarr(s3_url: str) -> bool:
    """Verify Zarr is accessible from S3 via public anonymous access."""
    print(f"\nVerifying remote Zarr at {s3_url}...")

    for tile_name in TILES:
        tile_url = f"{s3_url}/{tile_name}"
        print(f"  Checking tile {tile_name}...")

        store = fsspec.get_mapper(tile_url, anon=True)
        ds = xr.open_zarr(store)

        print(f"    Variables: {list(ds.data_vars)}")
        print(f"    Dimensions: {dict(ds.sizes)}")
        print(f"    CRS: {ds.attrs.get('crs')}")

        assert ds["lst_p50"].attrs.get("lst_scale_factor") == LST_SCALE
        assert ds["lst_p50"].attrs.get("lst_add_offset") == LST_OFFSET

        ds.close()

    print("Remote Zarr verification passed!")
    return True


def main() -> None:
    """Create, verify, upload, and verify sample Zarr."""
    zarr_path = create_sample_zarr()

    verify_local_zarr(zarr_path)

    s3_url = upload_to_source_coop(zarr_path)

    verify_remote_zarr(s3_url)

    print("\n" + "=" * 60)
    print("SUCCESS: Sample Zarr created and uploaded!")
    print(f"Local:  {zarr_path}")
    print(f"Remote: {s3_url}")
    print("=" * 60)


if __name__ == "__main__":
    main()
