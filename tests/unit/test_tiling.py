"""Unit tests for tiling utilities."""

import pytest

from landsat_lst.config import settings
from landsat_lst.models import TileId
from landsat_lst.tiling import (
    LAND_TILES,
    generate_global_tiles,
    generate_land_tiles,
    geobox_for_bbox,
    global_geobox,
    parse_tile_name,
    tile_from_point,
    tile_geobox,
    tiles_intersecting_bbox,
)


class TestGenerateGlobalTiles:
    def test_generates_tiles(self):
        tiles = list(generate_global_tiles())
        assert len(tiles) > 0

    def test_respects_latitude_bounds(self):
        tiles = list(generate_global_tiles(min_lat=-30, max_lat=30))

        for tile in tiles:
            assert tile.lat <= 30
            assert tile.lat - 5 >= -30

    def test_default_bounds_produce_expected_count(self):
        tiles = list(generate_global_tiles())
        lat_count = (60 - (-60)) // 5
        lon_count = 360 // 5
        expected = lat_count * lon_count
        assert len(tiles) == expected


class TestGenerateLandTiles:
    """Tests for land-filtered tile generation."""

    def test_returns_fewer_tiles_than_global(self):
        """Land filtering should significantly reduce tile count."""
        global_tiles = list(generate_global_tiles())
        land_tiles = list(generate_land_tiles())

        assert len(land_tiles) < len(global_tiles)
        assert len(land_tiles) == 700  # Known count from Natural Earth 110m

    def test_all_returned_tiles_are_in_land_set(self):
        """Every returned tile should be in the LAND_TILES set."""
        for tile in generate_land_tiles():
            assert tile.name in LAND_TILES

    def test_known_land_tiles_included(self):
        """Spot check: tiles over known land masses should be included."""
        land_tiles = {t.name for t in generate_land_tiles()}

        # Continental tiles that must be present
        assert "N40W075" in land_tiles  # New York area
        assert "N50E000" in land_tiles  # Western Europe
        assert "S35W060" in land_tiles  # Buenos Aires area
        assert "N35E135" in land_tiles  # Japan
        assert "S35E145" in land_tiles  # Melbourne area

    def test_ocean_tiles_excluded(self):
        """Spot check: tiles over open ocean should be excluded."""
        land_tiles = {t.name for t in generate_land_tiles()}

        # Mid-ocean tiles that must NOT be present
        assert "N30W030" not in land_tiles  # Mid-Atlantic
        assert "S30W090" not in land_tiles  # South Pacific
        assert "N00E160" not in land_tiles  # Central Pacific

    def test_respects_latitude_bounds(self):
        """Land tiles should respect the ±60° latitude bounds."""
        for tile in generate_land_tiles():
            assert tile.lat <= 60
            assert tile.lat - 5 >= -60


class TestLandTilesConstant:
    """Tests for the LAND_TILES constant itself."""

    def test_count_matches_expected(self):
        """Verify the hardcoded count is correct."""
        assert len(LAND_TILES) == 700

    def test_all_entries_are_valid_tile_names(self):
        """Every entry should match the tile naming convention."""
        import re

        pattern = re.compile(r"^[NS]\d{2}[EW]\d{3}$")
        for name in LAND_TILES:
            assert pattern.match(name), f"Invalid tile name: {name}"

    def test_no_tiles_outside_latitude_bounds(self):
        """No tiles should have NW corner outside ±60°."""
        for name in LAND_TILES:
            lat = int(name[1:3])
            if name[0] == "S":
                lat = -lat
            assert -55 <= lat <= 60, f"Tile {name} outside latitude bounds"


class TestTileFromPoint:
    def test_point_in_northern_hemisphere(self):
        tile = tile_from_point(lat=42.5, lon=-73.2)
        assert tile.lat == 45
        assert tile.lon == -75

    def test_point_on_tile_boundary(self):
        tile = tile_from_point(lat=40.0, lon=-75.0)
        assert tile.lat == 40
        assert tile.lon == -75

    def test_point_in_southern_hemisphere(self):
        tile = tile_from_point(lat=-33.9, lon=-60.5)
        assert tile.lat == -30
        assert tile.lon == -65


class TestTilesIntersectingBbox:
    def test_single_tile(self):
        bbox = (-74.0, 41.0, -73.0, 42.0)
        tiles = list(tiles_intersecting_bbox(bbox))
        assert len(tiles) == 1

    def test_multiple_tiles(self):
        bbox = (-76.0, 39.0, -71.0, 43.0)
        tiles = list(tiles_intersecting_bbox(bbox))
        assert len(tiles) >= 2

    def test_no_duplicates(self):
        bbox = (-80.0, 35.0, -70.0, 45.0)
        tiles = list(tiles_intersecting_bbox(bbox))
        names = [t.name for t in tiles]
        assert len(names) == len(set(names))


class TestParseTileName:
    """Tests for tile name parsing."""

    def test_parse_northern_western(self):
        """Should parse N40W075 correctly."""
        tile = parse_tile_name("N40W075")
        assert tile.lat == 40
        assert tile.lon == -75
        assert tile.name == "N40W075"

    def test_parse_northern_eastern(self):
        """Should parse N50E000 correctly."""
        tile = parse_tile_name("N50E000")
        assert tile.lat == 50
        assert tile.lon == 0

    def test_parse_southern_western(self):
        """Should parse S35W060 correctly."""
        tile = parse_tile_name("S35W060")
        assert tile.lat == -35
        assert tile.lon == -60

    def test_parse_southern_eastern(self):
        """Should parse S35E145 correctly."""
        tile = parse_tile_name("S35E145")
        assert tile.lat == -35
        assert tile.lon == 145

    def test_roundtrip_all_land_tiles(self):
        """Should roundtrip all land tiles through parse/name."""
        for name in LAND_TILES:
            tile = parse_tile_name(name)
            assert tile.name == name

    def test_invalid_format_raises(self):
        """Should raise ValueError for invalid format."""
        with pytest.raises(ValueError, match="Invalid tile name"):
            parse_tile_name("invalid")

    def test_wrong_length_raises(self):
        """Should raise ValueError for wrong length."""
        with pytest.raises(ValueError, match="Invalid tile name"):
            parse_tile_name("N40W75")  # Missing leading zero


class TestGlobalGeobox:
    """The shared grid every tile is cut from. See ADR-008."""

    def test_global_grid_is_exact(self):
        """Integer pixels-per-degree gives whole-pixel global dimensions."""
        assert global_geobox().shape == (432_000, 1_296_000)

    def test_divides_by_every_pyramid_factor(self):
        """Overviews of the global array never trim, at any configured factor."""
        height, width = global_geobox().shape
        for factor in settings.pyramid_factors:
            assert height % factor == 0, f"{height} not divisible by {factor}"
            assert width % factor == 0, f"{width} not divisible by {factor}"

    def test_tile_does_not_divide_by_coarsest_factor(self):
        """Why overviews belong to the global array rather than to a tile.

        18,000 = 2^4 * 3^2 * 5^3, so a 5-degree tile divides by 4 and 16 but
        not by 64. Coarsening per-tile would trim 16 px and shift block phase
        between neighbours; coarsening the global array does neither.
        """
        tile_px = int(settings.tile_size_degrees * settings.pixels_per_degree)
        assert tile_px == 18_000
        assert tile_px % 4 == 0
        assert tile_px % 16 == 0
        assert tile_px % 64 != 0


class TestTileGeobox:
    def test_tile_is_whole_pixels(self):
        assert tile_geobox(TileId(lat=40, lon=-75)).shape == (18_000, 18_000)

    def test_tile_bbox_is_exact(self):
        box = tile_geobox(TileId(lat=40, lon=-75)).boundingbox
        assert (box.left, box.bottom, box.right, box.top) == (-75.0, 35.0, -70.0, 40.0)

    def test_adjacent_tiles_are_contiguous(self):
        """One pixel step across a shared edge, with no gap and no overlap.

        The pre-ADR-008 grid overshot the shared edge by 0.484 px and sat
        0.14 px off its neighbour, because each tile anchored to its own bbox.
        """
        west = tile_geobox(TileId(lat=40, lon=-75))
        east = tile_geobox(TileId(lat=40, lon=-70))
        lon_west = west.coordinates["longitude"].values
        lon_east = east.coordinates["longitude"].values
        step = lon_west[1] - lon_west[0]
        assert (lon_east[0] - lon_west[-1]) == pytest.approx(step, rel=1e-9)

    def test_vertically_adjacent_tiles_are_contiguous(self):
        north = tile_geobox(TileId(lat=40, lon=-75))
        south = tile_geobox(TileId(lat=35, lon=-75))
        lat_north = north.coordinates["latitude"].values
        lat_south = south.coordinates["latitude"].values
        step = lat_north[1] - lat_north[0]
        assert (lat_south[0] - lat_north[-1]) == pytest.approx(step, rel=1e-9)

    def test_zoomed_out_tile_matches_offset_factor(self):
        """The offset-estimation grid stays whole-pixel at the default factor."""
        factor = settings.destripe_offset_resolution_factor
        expected = 18_000 // factor
        assert tile_geobox(TileId(lat=40, lon=-75), factor).shape == (expected, expected)

    def test_sub_tile_bbox_snaps_to_the_global_grid(self):
        """An arbitrary AOI still lands on global pixel edges, not its own."""
        box = geobox_for_bbox((-60.6, -34.0, -60.4, -33.8)).boundingbox
        assert (box.left, box.bottom, box.right, box.top) == pytest.approx(
            (-60.6, -34.0, -60.4, -33.8), abs=1e-9
        )

    def test_bbox_outside_latitude_bounds_raises(self):
        with pytest.raises(ValueError, match="outside the global grid"):
            geobox_for_bbox((-75.0, -80.0, -70.0, -75.0))
