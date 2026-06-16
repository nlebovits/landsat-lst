"""Land and water masking utilities."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
import xarray as xr
from shapely.geometry import box

NATURAL_EARTH_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"


def load_land_polygons(cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    """Load Natural Earth 10m land polygons.

    Args:
        cache_dir: Optional directory to cache downloaded data.

    Returns:
        GeoDataFrame of land polygons in EPSG:4326.
    """
    if cache_dir:
        cache_path = cache_dir / "ne_10m_land.gpkg"
        if cache_path.exists():
            return gpd.read_file(cache_path)

    land = gpd.read_file(NATURAL_EARTH_URL)
    land = land.to_crs("EPSG:4326")

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        land.to_file(cache_path, driver="GPKG")

    return land


def get_land_mask_for_bbox(
    bbox: tuple[float, float, float, float],
    resolution: float,
    land_polygons: gpd.GeoDataFrame,
    *,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Create a land mask raster for a bounding box.

    Args:
        bbox: Bounding box as (west, south, east, north).
        resolution: Pixel resolution in degrees.
        land_polygons: GeoDataFrame of land polygons.
        target_shape: Optional (height, width) to match exactly. If provided,
            overrides resolution-based calculation to ensure alignment with
            the target raster (e.g., odc-stac output which may use different
            rounding).

    Returns:
        Boolean numpy array where True indicates land.
    """
    west, south, east, north = bbox

    if target_shape is not None:
        height, width = target_shape
    else:
        width = int((east - west) / resolution)
        height = int((north - south) / resolution)

    transform = rasterio.transform.from_bounds(west, south, east, north, width, height)

    bbox_geom = box(west, south, east, north)
    clipped = land_polygons.clip(bbox_geom)

    if clipped.empty:
        return np.zeros((height, width), dtype=bool)

    geometries = clipped.geometry.values
    mask = rasterio.features.rasterize(
        geometries,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )

    return mask.astype(bool)


def apply_land_mask(
    data: xr.DataArray,
    land_mask: np.ndarray,
) -> xr.DataArray:
    """Apply land mask to data, setting water pixels to NaN.

    Args:
        data: DataArray to mask.
        land_mask: Boolean array where True indicates land.

    Returns:
        Masked DataArray with water pixels as NaN.
    """
    mask_da = xr.DataArray(
        land_mask,
        dims=["y", "x"],
        coords={"y": data.y, "x": data.x},
    )
    return data.where(mask_da)
