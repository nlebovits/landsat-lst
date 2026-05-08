"""Unit tests for tiling utilities."""

import pytest

from landsat_lst.tiling import (
    LAND_TILES,
    generate_global_tiles,
    generate_land_tiles,
    parse_tile_name,
    tile_from_point,
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
