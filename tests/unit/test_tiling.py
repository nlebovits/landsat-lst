"""Unit tests for tiling utilities."""

from landsat_lst.tiling import (
    generate_global_tiles,
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
