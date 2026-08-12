"""Synthetic COG pairs for catalog tests.

The catalog builder reads footprints, sizes, and band statistics out of COG
headers, so its tests need real COGs -- but only their headers matter. These
fixtures are 64x64 pixels stretched across a whole five-degree tile: kilobytes
on disk, exact on the grid the builder checks against, and carrying the same
embedded statistics, nodata, band descriptions, and scale/offset the production
export writes.

Shared rather than inlined because the thumbnail and item-mirror work reuses
the same tiles.
"""

from __future__ import annotations

import calendar
import struct
import zlib
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from landsat_lst.encoding import LST_FILL_VALUE, LST_OFFSET, LST_SCALE
from landsat_lst.tiling import parse_tile_name, tile_geobox

if TYPE_CHECKING:
    from pathlib import Path

#: Small enough that the whole raster fits in one internal tile, which is what
#: lets a valid COG skip overviews.
FIXTURE_SIZE = 64

_MONTHS = tuple(calendar.month_name[month] for month in range(1, 13))


def tile_bounds(tile: str) -> tuple[float, float, float, float]:
    """The exact grid extent of a tile, as the builder will expect to see it."""
    box = tile_geobox(parse_tile_name(tile)).boundingbox
    return (box.left, box.bottom, box.right, box.top)


def _stats_tags(values: np.ndarray, nodata: int | None) -> dict[str, str]:
    """The embedded statistics GDAL would write for one band."""
    valid = values if nodata is None else values[values != nodata]
    if valid.size == 0:
        valid = values
    tags = {
        "STATISTICS_MINIMUM": str(float(valid.min())),
        "STATISTICS_MAXIMUM": str(float(valid.max())),
        "STATISTICS_MEAN": str(float(valid.mean())),
        "STATISTICS_STDDEV": str(float(valid.std())),
    }
    if nodata is not None:
        # A band with nodata MUST declare its valid percent; a band without one
        # only should, and qa_count deliberately leaves it out.
        tags["STATISTICS_VALID_PERCENT"] = str(100.0 * valid.size / values.size)
    return tags


def _write_cog(
    path: Path,
    data: np.ndarray,
    bounds: tuple[float, float, float, float],
    *,
    nodata: int | None,
    descriptions: tuple[str, ...] | None = None,
    scales: tuple[float, ...] | None = None,
    offsets: tuple[float, ...] | None = None,
) -> Path:
    """Write one multi-band array as a validated COG with embedded statistics."""
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": data.dtype.name,
        "crs": "EPSG:4326",
        "transform": from_bounds(*bounds, width, height),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    source = path.parent / f".{path.name}.src.tif"
    source.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source, "w", **profile) as dst:
        dst.write(data)
        for index in range(1, count + 1):
            dst.update_tags(index, **_stats_tags(data[index - 1], nodata))
            if descriptions is not None:
                dst.set_band_description(index, descriptions[index - 1])
        if scales is not None:
            dst.scales = scales
        if offsets is not None:
            dst.offsets = offsets
    cog_translate(
        str(source),
        str(path),
        cog_profiles.get("deflate"),
        nodata=nodata,
        overview_level=0,
        forward_band_tags=True,  # carries the STATISTICS_* tags into the COG
        quiet=True,
    )
    source.unlink()
    return path


def _write_lst(path: Path, data: np.ndarray, bounds: tuple[float, float, float, float]) -> Path:
    """One percentile-composite COG, carrying the production encoding tags."""
    return _write_cog(
        path,
        data,
        bounds,
        nodata=LST_FILL_VALUE,
        descriptions=("lst_p95",),
        scales=(LST_SCALE,),
        offsets=(LST_OFFSET,),
    )


def write_lst_cog(path: Path, bounds: tuple[float, float, float, float], seed: int = 0) -> Path:
    """A single-band uint16 percentile composite, DN 0 as fill."""
    rng = np.random.default_rng(seed)
    celsius = rng.uniform(20.0, 45.0, size=(1, FIXTURE_SIZE, FIXTURE_SIZE))
    data = ((celsius - LST_OFFSET) / LST_SCALE).astype(np.uint16)
    data[0, 0, :4] = LST_FILL_VALUE  # a little fill, so valid percent is not 100
    return _write_lst(path, data, bounds)


def write_flat_lst_cog(
    path: Path, bounds: tuple[float, float, float, float], celsius: float | None
) -> Path:
    """One temperature everywhere, or nothing but fill when ``celsius`` is None.

    A composite with a single known value is what lets a thumbnail test name
    the colour a pixel must have, rather than re-deriving it from the raster it
    is supposed to be checking.
    """
    dn = LST_FILL_VALUE if celsius is None else round((celsius - LST_OFFSET) / LST_SCALE)
    data = np.full((1, FIXTURE_SIZE, FIXTURE_SIZE), dn, dtype=np.uint16)
    return _write_lst(path, data, bounds)


def write_qa_cog(path: Path, bounds: tuple[float, float, float, float], seed: int = 0) -> Path:
    """A twelve-band uint8 monthly observation count, no nodata."""
    rng = np.random.default_rng(seed + 1)
    data = rng.integers(0, 12, size=(12, FIXTURE_SIZE, FIXTURE_SIZE), dtype=np.uint8)
    return _write_cog(path, data, bounds, nodata=None, descriptions=_MONTHS)


def write_tile_pair(
    directory: Path, tile: str, window: str = "2021-2025", seed: int = 0
) -> tuple[Path, Path]:
    """Both COGs for one tile, named the way the pipeline names them."""
    directory.mkdir(parents=True, exist_ok=True)
    bounds = tile_bounds(tile)
    lst = write_lst_cog(directory / f"lst_p95_{window}_{tile}.tif", bounds, seed)
    qa = write_qa_cog(directory / f"qa_count_{window}_{tile}.tif", bounds, seed)
    return lst, qa


def write_source_tree(root: Path, tiles: tuple[str, ...], window: str = "2021-2025") -> Path:
    """A flat source directory holding a complete pair for each tile."""
    for seed, tile in enumerate(tiles):
        write_tile_pair(root, tile, window, seed)
    return root


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def write_thumbnail(path: Path, size: int = 8) -> Path:
    """A minimal valid greyscale PNG, so the collection can carry a thumbnail."""
    rows = b"".join(b"\x00" + bytes(range(size)) for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )
    return path
