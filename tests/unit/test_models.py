"""Unit tests for Pydantic models."""

import pytest

from landsat_lst.models import ProcessingJob, TileId, YearRange


class TestTileId:
    def test_tile_name_north_west(self):
        tile = TileId(lat=40, lon=-75)
        assert tile.name == "N40W075"

    def test_tile_name_south_east(self):
        tile = TileId(lat=-10, lon=30)
        assert tile.name == "S10E030"

    def test_tile_name_equator_prime_meridian(self):
        tile = TileId(lat=0, lon=0)
        assert tile.name == "N00E000"

    def test_tile_bbox(self):
        tile = TileId(lat=40, lon=-75)
        assert tile.bbox == (-75.0, 35.0, -70.0, 40.0)

    def test_tile_lat_validation(self):
        with pytest.raises(ValueError):
            TileId(lat=70, lon=0)

        with pytest.raises(ValueError):
            TileId(lat=-70, lon=0)

    def test_tile_lon_validation(self):
        with pytest.raises(ValueError):
            TileId(lat=0, lon=180)


class TestYearRange:
    def test_iteration(self):
        yr = YearRange(start=2021, end=2023)
        assert list(yr) == [2021, 2022, 2023]

    def test_contains(self):
        yr = YearRange(start=2021, end=2023)
        assert 2022 in yr
        assert 2020 not in yr

    def test_validation(self):
        with pytest.raises(ValueError):
            YearRange(start=2010, end=2023)


class TestProcessingJob:
    def test_datetime_range(self):
        job = ProcessingJob(tile=TileId(lat=40, lon=-75), year=2023)
        assert job.datetime_range == "2023-01-01/2023-12-31"

    def test_asset_filename(self):
        job = ProcessingJob(tile=TileId(lat=40, lon=-75), year=2023)
        assert job.asset_filename("lst_p95") == "lst_p95_2023_N40W075.tif"
        assert job.asset_filename("qa_count") == "qa_count_2023_N40W075.tif"

    def test_single_year_window_label(self):
        job = ProcessingJob(tile=TileId(lat=40, lon=-75), year=2024)
        assert job.window_label == "2024"

    def test_multi_year_datetime_range_and_label(self):
        job = ProcessingJob(tile=TileId(lat=-30, lon=-65), year=2020, end_year=2024)
        assert job.datetime_range == "2020-01-01/2024-12-31"
        assert job.window_label == "2020-2024"
        assert job.asset_filename("lst_p95") == "lst_p95_2020-2024_S30W065.tif"

    def test_end_year_equal_to_year_is_single(self):
        job = ProcessingJob(tile=TileId(lat=40, lon=-75), year=2022, end_year=2022)
        assert job.window_label == "2022"
        assert job.datetime_range == "2022-01-01/2022-12-31"

    def test_end_year_before_year_rejected(self):
        with pytest.raises(ValueError, match="end_year"):
            ProcessingJob(tile=TileId(lat=40, lon=-75), year=2024, end_year=2020)
