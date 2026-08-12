"""Render the collection's preview image from the COGs it publishes.

Portolan requires every geospatial collection to carry a thumbnail, and the
only honest thumbnail for a global tiled dataset is the global mosaic: four
pixels to the degree across the whole publishable band, each tile dropped into
the cell its footprint claims. Coverage gaps stay transparent, so the preview
reports what the collection actually holds rather than implying a full globe.

The read is deliberately coarse. Asking rasterio for a 20x20 window of an
18000x18000 tile is served out of the COG's smallest overview, so the whole
mosaic costs a few kilobytes per tile whether the files are local, on
``/vsicurl``, or on S3.

The colormap and the rescale range come from the :class:`CatalogSpec`, which is
the same source the render extension reads in :mod:`landsat_lst.catalog.items`.
A preview coloured differently from the map a client renders would be a lie
about the data, so neither is allowed its own copy of the numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.image import imsave
from rasterio.windows import from_bounds

from landsat_lst.catalog.scan import TilePair
from landsat_lst.catalog.spec import DEFAULT_SPEC
from landsat_lst.encoding import LST_FILL_VALUE, LST_OFFSET, LST_SCALE

if TYPE_CHECKING:
    from collections.abc import Iterable

    from landsat_lst.catalog.spec import CatalogSpec

#: Mosaic resolution. Four pixels to the degree makes a five-degree tile a
#: 20x20 cell: enough for continental structure, small enough that the whole
#: preview is one screen-sized PNG.
PIXELS_PER_DEGREE = 4

#: Longitude span of the mosaic: the whole world.
LON_BOUNDS = (-180.0, 180.0)

#: Latitude span. Landsat thermal coverage worth compositing stops well before
#: the poles, and the dataset publishes tiles only between 60S and 60N, so the
#: preview shows that band rather than padding two empty polar strips.
LAT_BOUNDS = (-60.0, 60.0)

WIDTH = round((LON_BOUNDS[1] - LON_BOUNDS[0]) * PIXELS_PER_DEGREE)
HEIGHT = round((LAT_BOUNDS[1] - LAT_BOUNDS[0]) * PIXELS_PER_DEGREE)

#: Sidecar statistics never ship with the data, and listing a remote prefix on
#: every open would dominate the cost of a coarse read.
_READ_ENV = {"GDAL_PAM_ENABLED": "NO", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}

_Bounds = tuple[float, float, float, float]


def _uri_of(source: TilePair | str | Path) -> str:
    """The percentile composite behind one source, however it was named."""
    if isinstance(source, TilePair):
        return source.lst.file.uri
    return str(source)


def _clip(bounds: _Bounds) -> _Bounds | None:
    """The part of a footprint that falls inside the mosaic, in degrees."""
    west = max(bounds[0], LON_BOUNDS[0])
    south = max(bounds[1], LAT_BOUNDS[0])
    east = min(bounds[2], LON_BOUNDS[1])
    north = min(bounds[3], LAT_BOUNDS[1])
    if east <= west or north <= south:
        return None
    return (west, south, east, north)


def _cell(bounds: _Bounds) -> tuple[int, int, int, int]:
    """The ``(col0, row0, col1, row1)`` span a clipped footprint occupies.

    Rounded rather than truncated: a five-degree edge is a whole number of
    mosaic pixels, and ``int()`` on a float that lands a hair below the integer
    would shave a column off every second tile.
    """
    west, south, east, north = bounds
    col0 = round((west - LON_BOUNDS[0]) * PIXELS_PER_DEGREE)
    col1 = round((east - LON_BOUNDS[0]) * PIXELS_PER_DEGREE)
    row0 = round((LAT_BOUNDS[1] - north) * PIXELS_PER_DEGREE)
    row1 = round((LAT_BOUNDS[1] - south) * PIXELS_PER_DEGREE)
    return col0, row0, col1, row1


def _read_patch(
    src: rasterio.DatasetReader, bounds: _Bounds, width: int, height: int
) -> np.ndarray:
    """One tile decoded to Celsius at mosaic resolution, fill as ``NaN``.

    The decimated read is nearest-neighbour on purpose: averaging would blend
    the DN 0 fill into its neighbours and paint a warm halo of -50 C around
    every coastline.
    """
    window = from_bounds(*bounds, transform=src.transform)
    dn = src.read(1, window=window, out_shape=(height, width))
    celsius = dn.astype("float32") * LST_SCALE + LST_OFFSET
    return np.where(dn == LST_FILL_VALUE, np.nan, celsius)


def _mosaic(uris: list[str]) -> np.ndarray:
    """Every tile placed on the global grid, uncovered cells left ``NaN``."""
    grid = np.full((HEIGHT, WIDTH), np.nan, dtype="float32")
    with rasterio.Env(**_READ_ENV):
        for uri in uris:
            with rasterio.open(uri) as src:
                clipped = _clip(tuple(src.bounds))
                if clipped is None:
                    continue
                col0, row0, col1, row1 = _cell(clipped)
                if col1 <= col0 or row1 <= row0:
                    continue
                patch = _read_patch(src, clipped, col1 - col0, row1 - row0)
                cell = grid[row0:row1, col0:col1]
                np.copyto(cell, patch, where=~np.isnan(patch))
    return grid


def _colorize(grid: np.ndarray, spec: CatalogSpec) -> np.ndarray:
    """The mosaic as RGBA bytes, with everything unobserved fully transparent."""
    missing = np.isnan(grid)
    norm = Normalize(vmin=spec.rescale[0], vmax=spec.rescale[1], clip=True)
    rgba = colormaps[spec.colormap](norm(np.nan_to_num(grid)), bytes=True)
    rgba[..., 3] = np.where(missing, 0, 255)
    return rgba


def generate_thumbnail(
    sources: Iterable[TilePair | str | Path],
    out_png: str | Path,
    spec: CatalogSpec = DEFAULT_SPEC,
) -> Path:
    """Render the collection preview from the tiles it publishes.

    Args:
        sources: The scanned tiles, or paths and URIs of percentile-composite
            COGs. Anything rasterio can open works, including ``/vsicurl/``,
            ``/vsis3/``, and ``s3://`` locations.
        out_png: Where to write the PNG; parent directories are created.
        spec: Supplies the colormap and the rescale range, which the render
            extension declares from the same fields.

    Returns:
        The path written.
    """
    grid = _mosaic([_uri_of(source) for source in sources])
    path = Path(out_png)
    path.parent.mkdir(parents=True, exist_ok=True)
    imsave(path, _colorize(grid, spec))
    return path
