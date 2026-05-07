"""Integration tests for STAC queries."""

import pytest

from landsat_lst.models import ProcessingJob, TileId
from landsat_lst.pipeline import query_stac


@pytest.mark.integration
class TestStacQuery:
    def test_query_returns_items(self, pergamino_bbox: tuple[float, float, float, float]):
        """Test that we can query Earth Search and get results."""
        tile = TileId(lat=-30, lon=-65)
        job = ProcessingJob(tile=tile, year=2023)

        items = query_stac(job)

        assert len(items) > 0

    def test_items_have_required_assets(self, pergamino_bbox: tuple[float, float, float, float]):
        """Test that returned items have the bands we need."""
        tile = TileId(lat=-30, lon=-65)
        job = ProcessingJob(tile=tile, year=2023)

        items = query_stac(job)

        if items:
            item = items[0]
            assert "lwir11" in item.assets or "ST_B10" in item.assets
            assert "qa_pixel" in item.assets

    def test_items_are_landsat_8_or_9(self, pergamino_bbox: tuple[float, float, float, float]):
        """Test that we only get Landsat 8 and 9 scenes."""
        tile = TileId(lat=-30, lon=-65)
        job = ProcessingJob(tile=tile, year=2023)

        items = query_stac(job)

        for item in items:
            platform = item.properties.get("platform", "")
            assert platform in ["landsat-8", "landsat-9", "LANDSAT_8", "LANDSAT_9"]
