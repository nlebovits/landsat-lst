"""Tiling utilities for global 5-degree grid.

Land Filtering Design Decision (2026-05-08)
-------------------------------------------
We use a hardcoded set of land tile names rather than runtime Natural Earth
lookups for the following reasons:

1. **Zero dependencies**: No geodatasets package or shapefile downloads required
2. **Zero I/O**: Instant frozenset lookup vs file parsing
3. **Reproducibility**: Same tiles every run, no network/file variability
4. **Sufficient precision**: 110m Natural Earth data is more than adequate for
   filtering 5° tiles (~550km). We're checking tile intersection, not precise
   coastlines.

The list was generated from Natural Earth 110m land polygons
(https://www.naturalearthdata.com/downloads/110m-physical-vectors/)
on 2026-05-08. Land boundaries change negligibly at this resolution.

Latitude bounds (±60°) rationale:
- Covers all significant population centers globally
- Excludes Antarctica (34+ tiles below -65°S with no inhabitants)
- 60°N includes northern Russia, Canada, Scandinavia (populated)
- 55°S includes Tierra del Fuego (Ushuaia, Punta Arenas - populated)
- Arctic cities (65-70°N) excluded as edge cases for urban heat analysis

Result: 700 land tiles vs 1,728 total = 59.5% reduction in processing.
"""

import math
from collections.abc import Iterator

from odc.geo.geobox import GeoBox

from landsat_lst.config import settings
from landsat_lst.models import TileId

# Land tiles within ±60° latitude bounds.
# Generated from Natural Earth 110m land polygons (2026-05-08).
# Sorted alphabetically for reproducibility and easy diffing.
LAND_TILES: frozenset[str] = frozenset(
    {
        "N00E005",
        "N00E010",
        "N00E015",
        "N00E020",
        "N00E025",
        "N00E030",
        "N00E035",
        "N00E040",
        "N00E095",
        "N00E100",
        "N00E105",
        "N00E110",
        "N00E115",
        "N00E120",
        "N00E125",
        "N00E130",
        "N00E135",
        "N00E140",
        "N00E145",
        "N00E150",
        "N00W040",
        "N00W045",
        "N00W050",
        "N00W055",
        "N00W060",
        "N00W065",
        "N00W070",
        "N00W075",
        "N00W080",
        "N00W085",
        "N05E005",
        "N05E010",
        "N05E015",
        "N05E020",
        "N05E025",
        "N05E030",
        "N05E035",
        "N05E040",
        "N05E045",
        "N05E095",
        "N05E100",
        "N05E105",
        "N05E110",
        "N05E115",
        "N05E120",
        "N05E125",
        "N05W005",
        "N05W010",
        "N05W050",
        "N05W055",
        "N05W060",
        "N05W065",
        "N05W070",
        "N05W075",
        "N05W080",
        "N05W085",
        "N10E000",
        "N10E005",
        "N10E010",
        "N10E015",
        "N10E020",
        "N10E025",
        "N10E030",
        "N10E035",
        "N10E040",
        "N10E045",
        "N10E050",
        "N10E075",
        "N10E080",
        "N10E095",
        "N10E100",
        "N10E105",
        "N10E110",
        "N10E115",
        "N10E120",
        "N10E125",
        "N10W005",
        "N10W010",
        "N10W015",
        "N10W055",
        "N10W060",
        "N10W065",
        "N10W070",
        "N10W075",
        "N10W080",
        "N10W085",
        "N10W090",
        "N15E000",
        "N15E005",
        "N15E010",
        "N15E015",
        "N15E020",
        "N15E025",
        "N15E030",
        "N15E035",
        "N15E040",
        "N15E045",
        "N15E050",
        "N15E070",
        "N15E075",
        "N15E080",
        "N15E095",
        "N15E100",
        "N15E105",
        "N15E115",
        "N15E120",
        "N15E125",
        "N15W005",
        "N15W010",
        "N15W015",
        "N15W020",
        "N15W065",
        "N15W070",
        "N15W075",
        "N15W080",
        "N15W085",
        "N15W090",
        "N15W095",
        "N20E000",
        "N20E005",
        "N20E010",
        "N20E015",
        "N20E020",
        "N20E025",
        "N20E030",
        "N20E035",
        "N20E040",
        "N20E045",
        "N20E050",
        "N20E055",
        "N20E070",
        "N20E075",
        "N20E080",
        "N20E085",
        "N20E090",
        "N20E095",
        "N20E100",
        "N20E105",
        "N20E110",
        "N20E115",
        "N20E120",
        "N20W005",
        "N20W010",
        "N20W015",
        "N20W020",
        "N20W070",
        "N20W075",
        "N20W080",
        "N20W085",
        "N20W090",
        "N20W095",
        "N20W100",
        "N20W105",
        "N20W110",
        "N20W155",
        "N20W160",
        "N25E000",
        "N25E005",
        "N25E010",
        "N25E015",
        "N25E020",
        "N25E025",
        "N25E030",
        "N25E035",
        "N25E040",
        "N25E045",
        "N25E050",
        "N25E055",
        "N25E065",
        "N25E070",
        "N25E075",
        "N25E080",
        "N25E085",
        "N25E090",
        "N25E095",
        "N25E100",
        "N25E105",
        "N25E110",
        "N25E115",
        "N25E120",
        "N25W005",
        "N25W010",
        "N25W015",
        "N25W020",
        "N25W075",
        "N25W080",
        "N25W085",
        "N25W090",
        "N25W095",
        "N25W100",
        "N25W105",
        "N25W110",
        "N25W115",
        "N25W160",
        "N30E000",
        "N30E005",
        "N30E010",
        "N30E015",
        "N30E020",
        "N30E025",
        "N30E030",
        "N30E035",
        "N30E040",
        "N30E045",
        "N30E050",
        "N30E055",
        "N30E060",
        "N30E065",
        "N30E070",
        "N30E075",
        "N30E080",
        "N30E085",
        "N30E090",
        "N30E095",
        "N30E100",
        "N30E105",
        "N30E110",
        "N30E115",
        "N30E120",
        "N30W005",
        "N30W010",
        "N30W015",
        "N30W080",
        "N30W085",
        "N30W090",
        "N30W095",
        "N30W100",
        "N30W105",
        "N30W110",
        "N30W115",
        "N30W120",
        "N35E000",
        "N35E005",
        "N35E010",
        "N35E015",
        "N35E020",
        "N35E025",
        "N35E030",
        "N35E035",
        "N35E040",
        "N35E045",
        "N35E050",
        "N35E055",
        "N35E060",
        "N35E065",
        "N35E070",
        "N35E075",
        "N35E080",
        "N35E085",
        "N35E090",
        "N35E095",
        "N35E100",
        "N35E105",
        "N35E110",
        "N35E115",
        "N35E120",
        "N35E125",
        "N35E130",
        "N35E135",
        "N35W005",
        "N35W010",
        "N35W080",
        "N35W085",
        "N35W090",
        "N35W095",
        "N35W100",
        "N35W105",
        "N35W110",
        "N35W115",
        "N35W120",
        "N35W125",
        "N40E000",
        "N40E005",
        "N40E010",
        "N40E015",
        "N40E020",
        "N40E025",
        "N40E030",
        "N40E035",
        "N40E040",
        "N40E045",
        "N40E050",
        "N40E055",
        "N40E060",
        "N40E065",
        "N40E070",
        "N40E075",
        "N40E080",
        "N40E085",
        "N40E090",
        "N40E095",
        "N40E100",
        "N40E105",
        "N40E110",
        "N40E115",
        "N40E120",
        "N40E125",
        "N40E130",
        "N40E135",
        "N40E140",
        "N40W005",
        "N40W010",
        "N40W075",
        "N40W080",
        "N40W085",
        "N40W090",
        "N40W095",
        "N40W100",
        "N40W105",
        "N40W110",
        "N40W115",
        "N40W120",
        "N40W125",
        "N45E000",
        "N45E005",
        "N45E010",
        "N45E015",
        "N45E020",
        "N45E025",
        "N45E030",
        "N45E035",
        "N45E040",
        "N45E045",
        "N45E050",
        "N45E055",
        "N45E060",
        "N45E065",
        "N45E070",
        "N45E075",
        "N45E080",
        "N45E085",
        "N45E090",
        "N45E095",
        "N45E100",
        "N45E105",
        "N45E110",
        "N45E115",
        "N45E120",
        "N45E125",
        "N45E130",
        "N45E135",
        "N45E140",
        "N45E145",
        "N45W005",
        "N45W010",
        "N45W065",
        "N45W070",
        "N45W075",
        "N45W080",
        "N45W085",
        "N45W090",
        "N45W095",
        "N45W100",
        "N45W105",
        "N45W110",
        "N45W115",
        "N45W120",
        "N45W125",
        "N50E000",
        "N50E005",
        "N50E010",
        "N50E015",
        "N50E020",
        "N50E025",
        "N50E030",
        "N50E035",
        "N50E040",
        "N50E045",
        "N50E050",
        "N50E055",
        "N50E060",
        "N50E065",
        "N50E070",
        "N50E075",
        "N50E080",
        "N50E085",
        "N50E090",
        "N50E095",
        "N50E100",
        "N50E105",
        "N50E110",
        "N50E115",
        "N50E120",
        "N50E125",
        "N50E130",
        "N50E135",
        "N50E140",
        "N50W005",
        "N50W010",
        "N50W055",
        "N50W060",
        "N50W065",
        "N50W070",
        "N50W075",
        "N50W080",
        "N50W085",
        "N50W090",
        "N50W095",
        "N50W100",
        "N50W105",
        "N50W110",
        "N50W115",
        "N50W120",
        "N50W125",
        "N50W130",
        "N55E000",
        "N55E005",
        "N55E010",
        "N55E015",
        "N55E020",
        "N55E025",
        "N55E030",
        "N55E035",
        "N55E040",
        "N55E045",
        "N55E050",
        "N55E055",
        "N55E060",
        "N55E065",
        "N55E070",
        "N55E075",
        "N55E080",
        "N55E085",
        "N55E090",
        "N55E095",
        "N55E100",
        "N55E105",
        "N55E110",
        "N55E115",
        "N55E120",
        "N55E125",
        "N55E130",
        "N55E135",
        "N55E140",
        "N55E155",
        "N55E160",
        "N55W005",
        "N55W010",
        "N55W060",
        "N55W065",
        "N55W070",
        "N55W075",
        "N55W080",
        "N55W085",
        "N55W090",
        "N55W095",
        "N55W100",
        "N55W105",
        "N55W110",
        "N55W115",
        "N55W120",
        "N55W125",
        "N55W130",
        "N55W135",
        "N55W165",
        "N60E005",
        "N60E010",
        "N60E015",
        "N60E020",
        "N60E025",
        "N60E030",
        "N60E035",
        "N60E040",
        "N60E045",
        "N60E050",
        "N60E055",
        "N60E060",
        "N60E065",
        "N60E070",
        "N60E075",
        "N60E080",
        "N60E085",
        "N60E090",
        "N60E095",
        "N60E100",
        "N60E105",
        "N60E110",
        "N60E115",
        "N60E120",
        "N60E125",
        "N60E130",
        "N60E135",
        "N60E140",
        "N60E145",
        "N60E150",
        "N60E155",
        "N60E160",
        "N60E165",
        "N60E170",
        "N60W005",
        "N60W010",
        "N60W060",
        "N60W065",
        "N60W070",
        "N60W075",
        "N60W080",
        "N60W085",
        "N60W090",
        "N60W095",
        "N60W100",
        "N60W105",
        "N60W110",
        "N60W115",
        "N60W120",
        "N60W125",
        "N60W130",
        "N60W135",
        "N60W140",
        "N60W145",
        "N60W150",
        "N60W155",
        "N60W160",
        "N60W165",
        "N60W170",
        "S05E010",
        "S05E015",
        "S05E020",
        "S05E025",
        "S05E030",
        "S05E035",
        "S05E100",
        "S05E105",
        "S05E110",
        "S05E115",
        "S05E120",
        "S05E125",
        "S05E130",
        "S05E135",
        "S05E140",
        "S05E145",
        "S05E150",
        "S05E155",
        "S05E160",
        "S05W035",
        "S05W040",
        "S05W045",
        "S05W050",
        "S05W055",
        "S05W060",
        "S05W065",
        "S05W070",
        "S05W075",
        "S05W080",
        "S05W085",
        "S10E010",
        "S10E015",
        "S10E020",
        "S10E025",
        "S10E030",
        "S10E035",
        "S10E040",
        "S10E045",
        "S10E050",
        "S10E115",
        "S10E120",
        "S10E125",
        "S10E130",
        "S10E135",
        "S10E140",
        "S10E145",
        "S10E150",
        "S10E160",
        "S10E165",
        "S10W040",
        "S10W045",
        "S10W050",
        "S10W055",
        "S10W060",
        "S10W065",
        "S10W070",
        "S10W075",
        "S10W080",
        "S15E010",
        "S15E015",
        "S15E020",
        "S15E025",
        "S15E030",
        "S15E035",
        "S15E040",
        "S15E045",
        "S15E050",
        "S15E115",
        "S15E120",
        "S15E125",
        "S15E130",
        "S15E135",
        "S15E140",
        "S15E145",
        "S15E165",
        "S15E175",
        "S15W040",
        "S15W045",
        "S15W050",
        "S15W055",
        "S15W060",
        "S15W065",
        "S15W070",
        "S15W075",
        "S15W080",
        "S15W180",
        "S20E010",
        "S20E015",
        "S20E020",
        "S20E025",
        "S20E030",
        "S20E035",
        "S20E040",
        "S20E045",
        "S20E110",
        "S20E115",
        "S20E120",
        "S20E125",
        "S20E130",
        "S20E135",
        "S20E140",
        "S20E145",
        "S20E150",
        "S20E160",
        "S20E165",
        "S20W045",
        "S20W050",
        "S20W055",
        "S20W060",
        "S20W065",
        "S20W070",
        "S20W075",
        "S25E010",
        "S25E015",
        "S25E020",
        "S25E025",
        "S25E030",
        "S25E040",
        "S25E045",
        "S25E110",
        "S25E115",
        "S25E120",
        "S25E125",
        "S25E130",
        "S25E135",
        "S25E140",
        "S25E145",
        "S25E150",
        "S25W050",
        "S25W055",
        "S25W060",
        "S25W065",
        "S25W070",
        "S25W075",
        "S30E015",
        "S30E020",
        "S30E025",
        "S30E030",
        "S30E110",
        "S30E115",
        "S30E120",
        "S30E125",
        "S30E130",
        "S30E135",
        "S30E140",
        "S30E145",
        "S30E150",
        "S30E170",
        "S30W055",
        "S30W060",
        "S30W065",
        "S30W070",
        "S30W075",
        "S35E115",
        "S35E135",
        "S35E140",
        "S35E145",
        "S35E150",
        "S35E170",
        "S35E175",
        "S35W060",
        "S35W065",
        "S35W070",
        "S35W075",
        "S40E140",
        "S40E145",
        "S40E165",
        "S40E170",
        "S40E175",
        "S40W065",
        "S40W070",
        "S40W075",
        "S45E065",
        "S45E070",
        "S45E165",
        "S45E170",
        "S45W070",
        "S45W075",
        "S45W080",
        "S50W060",
        "S50W065",
        "S50W070",
        "S50W075",
        "S50W080",
        "S55W070",
        "S55W075",
    }
)


def _global_geobox(pixels_per_degree: int) -> GeoBox:
    """The one global array at a given pixel density.

    Both the source grid and the delivered grid are cut from a construction of
    this shape, with the same origin and the same latitude bounds. That shared
    origin is what makes the delivered grid an *exact* aggregation of aligned
    source blocks rather than a resampling that happens to be close: an output
    cell covers source cells ``[3i, 3i+3)`` with no remainder anywhere on the
    globe. See ADR-017.
    """
    return GeoBox.from_bbox(
        (-180.0, settings.min_latitude, 180.0, settings.max_latitude),
        crs=settings.crs,
        resolution=1.0 / pixels_per_degree,
        tight=True,
    )


def global_geobox() -> GeoBox:
    """The single SOURCE grid every scene load is cut from.

    Scenes load and solar-day fuse here; the product is published on
    :func:`output_global_geobox`, an exact 3x coarsening of it.

    ``pixels_per_degree`` is an integer, so this comes out at exactly
    1,296,000 x 432,000 px and divides cleanly by every overview factor down to
    64x (20,250 x 6,750). A 5-degree tile is 18,000 px, which divides by 4 and
    16 but not by 64 -- the reason overviews belong to the global array rather
    than to a tile. See ADR-008.
    """
    return _global_geobox(settings.pixels_per_degree)


def output_global_geobox() -> GeoBox:
    """The single DELIVERED grid every published tile is cut from.

    At the V1 default of 1,200 px/deg this is 432,000 x 144,000 px, and a
    five-degree tile is 6,000 x 6,000. Nominal ~100 m: the spacing is
    geographic, so physical cell width varies with latitude.
    """
    return _global_geobox(settings.output_pixels_per_degree)


def geobox_for_bbox(
    bbox: tuple[float, float, float, float],
    resolution_factor: int = 1,
) -> GeoBox:
    """Cut the grid for ``bbox`` out of :func:`global_geobox`.

    Pixel origin comes from the global grid, not from ``bbox``, so adjacent
    areas share pixel edges exactly. Deriving the grid from the bbox instead is
    what left tile N40W075 overshooting its own eastern boundary by 0.484 px and
    sitting 0.14 px off its neighbour (ADR-008).

    Args:
        bbox: Bounding box as (west, south, east, north), in degrees.
        resolution_factor: Zoom out by this factor, for reads that need no
            spatial detail (per-scene offset estimation). Powers of two land on
            a stored COG overview.

    Returns:
        A GeoBox aligned to the global grid.

    Raises:
        ValueError: If ``bbox`` falls outside the global grid's latitude bounds.
    """
    geobox = _cut(bbox, settings.pixels_per_degree, global_geobox())
    return geobox.zoom_out(resolution_factor) if resolution_factor > 1 else geobox


def _cut(
    bbox: tuple[float, float, float, float],
    pixels_per_degree: int,
    grid: GeoBox,
) -> GeoBox:
    """Window ``bbox`` out of ``grid``, indexing in that grid's own pixels.

    Raises:
        ValueError: If ``bbox`` falls outside the global grid's latitude bounds.
    """
    west, south, east, north = bbox
    if south < settings.min_latitude or north > settings.max_latitude:
        msg = (
            f"bbox latitudes ({south}, {north}) fall outside the global grid "
            f"({settings.min_latitude}, {settings.max_latitude})"
        )
        raise ValueError(msg)

    col0 = round((west + 180.0) * pixels_per_degree)
    col1 = round((east + 180.0) * pixels_per_degree)
    # Latitude descends north-down, so row 0 is the northern edge of the grid.
    row0 = round((settings.max_latitude - north) * pixels_per_degree)
    row1 = round((settings.max_latitude - south) * pixels_per_degree)

    return grid[row0:row1, col0:col1]


def output_geobox_for_bbox(bbox: tuple[float, float, float, float]) -> GeoBox:
    """Cut the DELIVERED grid for ``bbox`` out of :func:`output_global_geobox`.

    Covers the identical extent and the identical row/column blocks as
    ``geobox_for_bbox(bbox).zoom_out(spatial_aggregation_factor)``, pinned that
    way in ``tests/unit/test_tiling.py``. The two transforms differ by one ULP
    and this form is the authority: ``zoom_out`` computes ``(1/3600) * 3 =
    0.0008333333333333333`` where the integer density gives ``1/1200 =
    0.0008333333333333334``. That is ADR-008's argument one grid down -- a
    density is an integer and a spacing is derived from it, never the reverse
    -- and it is why the delivered grid is cut here rather than zoomed into
    existence. Cut from the global array rather than computed by zooming a
    tile out, for the same reason
    :func:`geobox_for_bbox` is: a grid derived from an area's own bounds
    anchors to those bounds, and neighbouring areas then disagree by a fraction
    of a pixel (ADR-008).

    There is no ``resolution_factor`` here. Zooming the delivered grid out
    further would produce a grid the product is not published on, and every
    coarse read this project makes -- the offset estimator's -- is a coarsening
    of the *source* grid, which is a separate axis on purpose (ADR-017).
    """
    return _cut(bbox, settings.output_pixels_per_degree, output_global_geobox())


def tile_geobox(tile: TileId, resolution_factor: int = 1) -> GeoBox:
    """Cut a tile's SOURCE grid out of :func:`global_geobox`."""
    return geobox_for_bbox(tile.bbox, resolution_factor)


def output_tile_geobox(tile: TileId) -> GeoBox:
    """Cut a tile's DELIVERED grid out of :func:`output_global_geobox`."""
    return output_geobox_for_bbox(tile.bbox)


def generate_land_tiles(
    min_lat: float = settings.min_latitude,
    max_lat: float = settings.max_latitude,
    tile_size: float = settings.tile_size_degrees,
) -> Iterator[TileId]:
    """Generate tile IDs for land areas only.

    Filters the global grid to tiles that intersect land polygons, based on
    Natural Earth 110m data. This reduces processing from 1,728 tiles to 700
    (59.5% reduction).

    Args:
        min_lat: Minimum latitude (default: -60).
        max_lat: Maximum latitude (default: 60).
        tile_size: Tile size in degrees (default: 5).

    Yields:
        TileId for each tile that intersects land.

    Note:
        The land tile list is hardcoded for performance and reproducibility.
        See module docstring for rationale.
    """
    for tile in generate_global_tiles(min_lat, max_lat, tile_size):
        if tile.name in LAND_TILES:
            yield tile


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


def parse_tile_name(name: str) -> TileId:
    """Parse a tile name like 'N40W075' back to a TileId.

    Args:
        name: Tile name in format N40W075 or S10E030.

    Returns:
        TileId with lat/lon coordinates.

    Raises:
        ValueError: If name format is invalid.
    """
    import re  # noqa: PLC0415

    match = re.match(r"^([NS])(\d{2})([EW])(\d{3})$", name)
    if not match:
        msg = f"Invalid tile name format: {name}"
        raise ValueError(msg)

    lat_dir, lat_val, lon_dir, lon_val = match.groups()
    lat = int(lat_val) * (1 if lat_dir == "N" else -1)
    lon = int(lon_val) * (1 if lon_dir == "E" else -1)

    return TileId(lat=lat, lon=lon)


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
