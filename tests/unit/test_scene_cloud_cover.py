"""Joining per-item ``eo:cloud_cover`` onto the solar-day axis a load produces.

``load_scenes`` groups by solar day, so a per-item property has no axis to sit
on until it is aggregated the same way. These tests pin the aggregation and,
more importantly, the guard: the join reproduces odc-stac's grouping rule, and a
reproduction that drifts must fail loudly rather than misalign silently.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pystac
import pytest
import xarray as xr

from landsat_lst.pipeline import scene_cloud_cover

BBOX = (-61.1, -34.4, -60.1, -33.4)


def _item(item_id: str, when: dt.datetime, cloud: float) -> pystac.Item:
    """A STAC item carrying only what the grouping and the join read."""
    item = pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [
                [
                    [BBOX[0], BBOX[1]],
                    [BBOX[2], BBOX[1]],
                    [BBOX[2], BBOX[3]],
                    [BBOX[0], BBOX[3]],
                    [BBOX[0], BBOX[1]],
                ]
            ],
        },
        bbox=list(BBOX),
        datetime=when,
        properties={"eo:cloud_cover": cloud, "proj:epsg": 4326},
    )
    item.add_asset("lwir11", pystac.Asset(href="http://example/a.tif", media_type="image/tiff"))
    return item


def _time(*dates: str) -> xr.DataArray:
    values = np.array(dates, dtype="datetime64[ns]")
    return xr.DataArray(values, dims=["time"], coords={"time": values})


@pytest.fixture
def two_days() -> list[pystac.Item]:
    """Two items on one solar day, one on another."""
    return [
        _item("a", dt.datetime(2021, 3, 1, 13, 50, tzinfo=dt.UTC), 10.0),
        _item("b", dt.datetime(2021, 3, 1, 13, 51, tzinfo=dt.UTC), 30.0),
        _item("c", dt.datetime(2021, 3, 9, 13, 50, tzinfo=dt.UTC), 80.0),
    ]


class TestSceneCloudCover:
    def test_items_sharing_a_solar_day_are_averaged(self, two_days):
        """Two items collapse to one scene, and its cover is their mean."""
        cover = scene_cloud_cover(two_days, BBOX, _time("2021-03-01T13:50", "2021-03-09T13:50"))
        assert cover.sizes["time"] == 2
        assert cover.values[0] == pytest.approx(20.0)
        assert cover.values[1] == pytest.approx(80.0)

    def test_result_sits_on_the_caller_time_axis(self, two_days):
        """The join must be indexable against the array it describes."""
        time = _time("2021-03-01T13:50", "2021-03-09T13:50")
        cover = scene_cloud_cover(two_days, BBOX, time)
        assert np.array_equal(cover.time.values, time.values)

    def test_a_time_axis_of_the_wrong_length_raises(self, two_days):
        """Silence here would misalign every downstream statistic."""
        with pytest.raises(ValueError, match="does not reproduce the loaded time axis"):
            scene_cloud_cover(two_days, BBOX, _time("2021-03-01T13:50"))

    def test_a_shifted_time_axis_raises(self, two_days):
        """Same length, wrong days: still a misalignment, still an error."""
        with pytest.raises(ValueError, match="does not reproduce the loaded time axis"):
            scene_cloud_cover(two_days, BBOX, _time("2021-03-02T13:50", "2021-03-10T13:50"))

    def test_the_resolution_factor_does_not_move_the_grouping(self, two_days):
        """A coarser offset grid shares the tile's centroid, so it shares scenes.

        The factor reaches the grouping only through the geobox centroid, which
        a change of resolution leaves where it was. Recorded because the offset
        pass and the composite must describe the same scene set for a per-scene
        offset to mean anything.
        """
        time = _time("2021-03-01T13:50", "2021-03-09T13:50")
        native = scene_cloud_cover(two_days, BBOX, time, 1)
        coarse = scene_cloud_cover(two_days, BBOX, time, 4)
        assert np.array_equal(native.values, coarse.values)
