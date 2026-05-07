"""Tiling utilities for global 5-degree grid."""

import math
from collections.abc import Iterator

from landsat_lst.config import settings
from landsat_lst.models import TileId


def generate_global_tiles(
    min_lat: float = settings.min_latitude,
    max_lat: float = settings.max_latitude,
    tile_size: float = settings.tile_size_degrees,
) -> Iterator[TileId]:
    """Generate all tile IDs for the global grid.

    Args:
        min_lat: Minimum latitude (default: -60).
        max_lat: Maximum latitude (default: 60).
        tile_size: Tile size in degrees (default: 5).

    Yields:
        TileId for each tile in the grid.
    """
    lat = int(max_lat)
    while lat > min_lat:
        lon = -180
        while lon < 180:
            yield TileId(lat=lat, lon=lon)
            lon += int(tile_size)
        lat -= int(tile_size)


def tile_from_point(lat: float, lon: float, tile_size: float = 5.0) -> TileId:
    """Get the tile containing a given point.

    Tiles use (south, north] convention: south-exclusive, north-inclusive.
    A point exactly on the northern boundary belongs to that tile.

    Args:
        lat: Latitude of point.
        lon: Longitude of point.
        tile_size: Tile size in degrees.

    Returns:
        TileId containing the point.
    """
    tile_lat = int(math.ceil(lat / tile_size) * tile_size)
    tile_lon = int(math.floor(lon / tile_size) * tile_size)

    return TileId(lat=tile_lat, lon=tile_lon)


def tiles_intersecting_bbox(
    bbox: tuple[float, float, float, float],
    tile_size: float = 5.0,
) -> Iterator[TileId]:
    """Get all tiles that intersect a bounding box.

    Args:
        bbox: Bounding box as (west, south, east, north).
        tile_size: Tile size in degrees.

    Yields:
        TileId for each intersecting tile.
    """
    west, south, east, north = bbox

    seen = set()
    lat = south
    while lat <= north:
        lon = west
        while lon <= east:
            tile = tile_from_point(lat, lon, tile_size)
            if tile.name not in seen:
                seen.add(tile.name)
                yield tile
            lon += tile_size
        lat += tile_size
