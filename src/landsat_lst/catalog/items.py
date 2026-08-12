"""Build one STAC item per tile, carrying both of the tile's assets.

The item's bounding box is read from the COG header rather than computed from
the tile name, then checked against the grid the pipeline claims to write. A
mismatch means the raster is not on the shared global grid, which is a defect
worth stopping the build for -- a catalog that reports a footprint the file
does not have is worse than no catalog.
"""

from __future__ import annotations

import calendar
import math
from typing import TYPE_CHECKING, Any

import pystac

from landsat_lst.catalog.spec import (
    COG_MEDIA_TYPE,
    FILE_EXTENSION_URI,
    LST_ASSET_KEY,
    QA_ASSET_KEY,
    RASTER_EXTENSION_URI,
    RENDER_EXTENSION_URI,
)
from landsat_lst.encoding import LST_FILL_VALUE, LST_OFFSET, LST_SCALE
from landsat_lst.tiling import parse_tile_name, tile_geobox

if TYPE_CHECKING:
    from landsat_lst.catalog.scan import BandStats, CogHeader, TilePair
    from landsat_lst.catalog.spec import CatalogSpec

#: A tile edge is a whole number of 1/3600-degree pixels, so agreement between
#: the header and the grid is exact up to float64 round-off in the transform.
_BBOX_TOLERANCE_DEG = 1e-9

_MONTHS = tuple(calendar.month_name[month] for month in range(1, 13))

#: Item properties that state the encoding contract in one place, so a reader
#: never has to infer the decode from band metadata. Required by issue #2.
_ENCODING_PROPERTIES: dict[str, Any] = {
    "lst:scale": LST_SCALE,
    "lst:offset": LST_OFFSET,
    "lst:units": "celsius",
    "lst:nodata": LST_FILL_VALUE,
}

#: Key of the one render this dataset publishes. The percentile composite is
#: the thing worth looking at; the monthly counts are evidence, not a map.
RENDER_KEY = "lst"

_LST_TITLE = "Land surface temperature, 95th percentile"
_QA_TITLE = "Valid observation count per calendar month"
_QA_DESCRIPTION = (
    "Number of cloud-free observations in this calendar month, pooled across "
    "the window and counted after de-striping. 0 means no valid observation."
)


class GridMismatchError(RuntimeError):
    """A COG's footprint disagrees with the tile grid it claims to be on."""


def _polygon(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    """The tile footprint as a counter-clockwise GeoJSON polygon."""
    west, south, east, north = bbox
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {"type": "Polygon", "coordinates": [ring]}


def _check_grid(tile: str, bbox: tuple[float, float, float, float]) -> None:
    """Raise unless the header footprint matches the tile's grid extent."""
    box = tile_geobox(parse_tile_name(tile)).boundingbox
    expected = (box.left, box.bottom, box.right, box.top)
    if any(
        not math.isclose(a, b, abs_tol=_BBOX_TOLERANCE_DEG)
        for a, b in zip(bbox, expected, strict=True)
    ):
        msg = (
            f"tile {tile}: COG bbox {bbox} disagrees with the global grid extent "
            f"{expected}; the raster is not on the shared 1/3600-degree grid"
        )
        raise GridMismatchError(msg)


def _statistics(stats: BandStats, scale: float, offset: float) -> dict[str, float]:
    """Band statistics rescaled into the units a reader sees.

    The offset shifts locations but not spread, so the standard deviation is
    scaled alone. Absent tags are omitted rather than guessed.
    """
    located = {
        field: getattr(stats, field) * scale + offset
        for field in ("minimum", "maximum", "mean")
        if getattr(stats, field) is not None
    }
    if stats.stddev is not None:
        located["stddev"] = stats.stddev * scale
    return located


# STAC 1.1 folded per-band raster metadata into the core ``bands`` array, and
# raster v2.0.0 followed: it defines only ``raster:``-prefixed fields and
# rejects any it does not define, so the nodata, data type, and statistics live
# on the band object itself while scale, offset, and sampling keep their prefix.
def _lst_bands(header: CogHeader) -> list[dict[str, Any]]:
    """The single decoded-temperature band of the percentile composite."""
    return [
        {
            "name": LST_ASSET_KEY,
            "description": _LST_TITLE,
            "data_type": header.data_type,
            "nodata": LST_FILL_VALUE,
            "unit": "celsius",
            "statistics": _statistics(header.bands[0], LST_SCALE, LST_OFFSET),
            "raster:scale": LST_SCALE,
            "raster:offset": LST_OFFSET,
            "raster:sampling": "area",
        }
    ]


def _qa_bands(header: CogHeader) -> list[dict[str, Any]]:
    """One band per calendar month, January first."""
    if len(header.bands) != len(_MONTHS):
        msg = (
            f"{QA_ASSET_KEY} must carry {len(_MONTHS)} monthly bands, "
            f"found {len(header.bands)} in {header.file.name}"
        )
        raise GridMismatchError(msg)
    return [
        {
            "name": f"{QA_ASSET_KEY}_{month.lower()}",
            "description": month,
            "data_type": header.data_type,
            "unit": "count",
            "statistics": _statistics(header.bands[index], 1.0, 0.0),
            "raster:sampling": "area",
        }
        for index, month in enumerate(_MONTHS)
    ]


# Render v2.0.0 keeps every rendering field inside a ``renders`` object -- on an
# item that object is ``properties.renders``, and the schema *requires* it of
# any item declaring the extension. There are no ``render:*`` asset fields to
# set: a render names the assets it draws instead, which is how this one points
# at the percentile composite and leaves the monthly counts alone.
def renders_for(spec: CatalogSpec) -> dict[str, Any]:
    """The one render a client should draw, in decoded Celsius.

    The collection declares the same object over its ``item_assets``, so a
    browser that never opens an item still knows how to draw the tiles.
    """
    low, high = spec.rescale
    return {
        RENDER_KEY: {
            "title": _LST_TITLE,
            "assets": [LST_ASSET_KEY],
            "colormap_name": spec.colormap,
            # The band declares raster:scale and raster:offset, so the range a
            # client stretches over is the temperature, not the stored DN.
            "rescale": [[low, high]],
        }
    }


def _asset(
    header: CogHeader, title: str, roles: list[str], bands: list[dict[str, Any]]
) -> pystac.Asset:
    """One COG asset, referenced relative to the item that declares it."""
    return pystac.Asset(
        href=f"./{header.file.name}",
        title=title,
        media_type=COG_MEDIA_TYPE,
        roles=roles,
        extra_fields={"file:size": header.file.size, "bands": bands},
    )


def build_item(pair: TilePair, spec: CatalogSpec) -> pystac.Item:
    """One item describing a tile's percentile composite and its QA counts."""
    _check_grid(pair.tile, pair.lst.bbox)
    item = pystac.Item(
        id=pair.tile,
        geometry=_polygon(pair.lst.bbox),
        bbox=list(pair.lst.bbox),
        datetime=None,
        properties={
            "title": f"{pair.tile} land surface temperature, {spec.window}",
            "start_datetime": spec.start_datetime,
            "end_datetime": spec.end_datetime,
            "renders": renders_for(spec),
            **_ENCODING_PROPERTIES,
        },
        stac_extensions=[FILE_EXTENSION_URI, RASTER_EXTENSION_URI, RENDER_EXTENSION_URI],
    )
    item.add_asset(LST_ASSET_KEY, _asset(pair.lst, _LST_TITLE, ["data"], _lst_bands(pair.lst)))
    item.add_asset(
        QA_ASSET_KEY, _asset(pair.qa, _QA_TITLE, ["data", "quality"], _qa_bands(pair.qa))
    )
    item.assets[QA_ASSET_KEY].description = _QA_DESCRIPTION
    return item
