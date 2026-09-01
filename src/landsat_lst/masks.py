"""Land and water masking utilities.

The land mask uses Natural Earth 10m land polygons with a 25km coastal buffer
to ensure coverage of barrier islands, marshes, estuaries, and other coastal
features that may be missing from the generalized NE polygons. The goal is to
exclude open ocean while erring on the side of inclusion for any potentially
inhabited or administered areas. See docs/findings-land-mask-buffer.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import rasterio.features
import xarray as xr
from shapely.geometry import box

if TYPE_CHECKING:
    from pathlib import Path

    from odc.geo.geobox import GeoBox

NATURAL_EARTH_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
COASTAL_BUFFER_METERS = 25_000  # 25km buffer for coastal features


def load_land_polygons(
    cache_dir: Path | None = None,
    *,
    buffer_meters: int = COASTAL_BUFFER_METERS,
) -> gpd.GeoDataFrame:
    """Load Natural Earth 10m land polygons with coastal buffer.

    The buffer ensures coverage of barrier islands, marshes, and coastal
    features that may be missing from the generalized NE 10m polygons.
    See docs/findings-land-mask-buffer.md for rationale.

    Args:
        cache_dir: Optional directory to cache downloaded data.
        buffer_meters: Buffer distance in meters (default 25km). Set to 0
            to disable buffering.

    Returns:
        GeoDataFrame of (buffered) land polygons in EPSG:4326.
    """
    cache_suffix = f"_buf{buffer_meters // 1000}km" if buffer_meters else ""
    cache_filename = f"ne_10m_land{cache_suffix}.gpkg"

    if cache_dir:
        cache_path = cache_dir / cache_filename
        if cache_path.exists():
            return gpd.read_file(cache_path)

    land = gpd.read_file(NATURAL_EARTH_URL)
    land = land.to_crs("EPSG:4326")

    if buffer_meters > 0:
        # Buffer in a projected CRS for accurate distance, then reproject back
        land_projected = land.to_crs("EPSG:3857")
        land_projected["geometry"] = land_projected.geometry.buffer(buffer_meters)
        land = land_projected.to_crs("EPSG:4326")

        # Buffering + reprojection to EPSG:4326 can produce self-intersecting
        # geometries near the antimeridian, which raise GEOS TopologyException
        # ("side location conflict") in downstream .clip(). Repair them so the
        # land set is valid for any tile. See issue #31.
        land["geometry"] = land.geometry.make_valid()

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
            overrides the resolution-based calculation, so the mask lines up
            with a grid that was built elsewhere.

    Returns:
        Boolean numpy array where True indicates land.
    """
    west, south, east, north = bbox

    if target_shape is not None:
        height, width = target_shape
    else:
        # Round rather than truncate. Neither 1/3600 nor 1/1200 is exactly
        # representable, so a 5-degree span divides to 17999.999999999996 on
        # the source grid and 5999.999999999999 on the delivered one, and
        # int() would silently drop a pixel row and column on either.
        width = round((east - west) / resolution)
        height = round((north - south) / resolution)

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


def get_land_mask_for_geobox(
    geobox: GeoBox,
    land_polygons: gpd.GeoDataFrame,
) -> np.ndarray:
    """Create a land mask on a geobox's own grid, exactly.

    The transform comes from ``geobox.transform``, not from
    ``rasterio.transform.from_bounds``. They agree to about fifteen digits and
    that is not enough: ``from_bounds`` divides the span by the pixel count, so
    a five-degree delivered tile gets a pixel size of ``5/6000`` rather than the
    grid's ``1/1200``, and a row band cut out of that tile gets
    ``(rows/1200)/rows``. (The same argument held one grid up, at ``5/18000``
    against ``1/3600``; this function serves both and cares about neither.)
    Those differ in the last bits, which is enough to move a polygon edge
    across a pixel centre and flip a pixel between a band's mask and the
    corresponding rows of the tile's. A shard's mask has to be the *slice* of
    the tile's mask, not a very good approximation of it -- otherwise the
    seams between bands carry a one-pixel land/ocean disagreement that no
    downstream check looks for. The geobox transform is the grid's own affine
    and slices exactly, because ``geobox[a:b, :]`` only moves its origin.

    Args:
        geobox: The ``odc.geo.geobox.GeoBox`` the data was loaded on. Imported
            for typing only: the rasterization itself needs nothing from
            ``odc.geo`` beyond the affine and the bounds.
        land_polygons: GeoDataFrame of land polygons in EPSG:4326.

    Returns:
        Boolean numpy array where True indicates land, shaped like the geobox.
    """
    height, width = int(geobox.shape[0]), int(geobox.shape[1])
    transform = geobox.transform
    left, bottom, right, top = (float(v) for v in geobox.boundingbox)

    clipped = land_polygons.clip(box(left, bottom, right, top))
    if clipped.empty:
        return np.zeros((height, width), dtype=bool)

    mask = rasterio.features.rasterize(
        clipped.geometry.values,
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
