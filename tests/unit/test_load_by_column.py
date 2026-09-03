"""Per-column loading: same array, one stac_load per column, a shared time axis or an error."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from landsat_lst import pipeline
from landsat_lst.config import settings

pytestmark = pytest.mark.unit

TIMES = np.array(["2021-01-01T10:00:00.123456789", "2021-01-17T10:00:00.5"], dtype="datetime64[ns]")


def _fake_stac_load(calls, *, times_for=None):
    def stac_load(items, *, bands, geobox, chunks, **kwargs):
        calls.append(geobox)
        ny, nx = geobox.shape
        times = times_for(len(calls)) if times_for else TIMES
        coords = {
            "time": times,
            "latitude": np.linspace(0, 1, ny),
            "longitude": np.linspace(0, 1, nx) + 10 * (len(calls) - 1),
        }
        data = {
            band: xr.DataArray(
                np.full((times.size, ny, nx), len(calls), dtype=np.uint16),
                dims=("time", "latitude", "longitude"),
                coords=coords,
            )
            for band in bands
        }
        return xr.Dataset(data)

    return stac_load


def _geobox(width: int) -> GeoBox:
    return GeoBox.from_bbox((0, 0, width / 3600, 512 / 3600), crs="EPSG:4326", shape=(512, width))


def test_per_column_loads_each_column_and_concatenates_along_longitude(monkeypatch):
    calls: list[GeoBox] = []
    monkeypatch.setattr(pipeline, "stac_load", _fake_stac_load(calls))
    monkeypatch.setattr(settings, "load_chunk_size", 1024)

    ds = pipeline.load_scenes([], (0, 0, 1, 1), geobox=_geobox(2560), per_column=True)

    assert [g.shape[1] for g in calls] == [1024, 1024, 512]
    assert ds["lwir11"].shape == (2, 512, 2560)
    assert np.array_equal(ds["time"].values, TIMES)
    # each column carries its own load's marker, in longitude order
    row = ds["lwir11"].isel(time=0, latitude=0).values
    assert list(row[[0, 1024, 2048]]) == [1, 2, 3]


def test_single_load_is_the_default(monkeypatch):
    calls: list[GeoBox] = []
    monkeypatch.setattr(pipeline, "stac_load", _fake_stac_load(calls))
    monkeypatch.setattr(settings, "load_chunk_size", 1024)

    ds = pipeline.load_scenes([], (0, 0, 1, 1), geobox=_geobox(2560))

    assert len(calls) == 1
    assert ds["lwir11"].shape == (2, 512, 2560)


def test_a_column_with_a_different_time_axis_is_refused(monkeypatch):
    calls: list[GeoBox] = []

    def times_for(call_index):
        return TIMES if call_index == 1 else TIMES[:1]

    monkeypatch.setattr(pipeline, "stac_load", _fake_stac_load(calls, times_for=times_for))
    monkeypatch.setattr(settings, "load_chunk_size", 1024)

    with pytest.raises(ValueError, match="do not share a time axis"):
        pipeline.load_scenes([], (0, 0, 1, 1), geobox=_geobox(2048), per_column=True)
