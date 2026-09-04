"""The GED gap mask through the production composite path.

Runs ``process_tile`` end to end with everything external stubbed -- the STAC
query, the scene load, the land mask -- while the GED mask itself is built
*for real* from synthetic AG100 granules.

The rule under test is a conjunction: a pixel is dropped only where the gap
geometry and the hot threshold agree. So the scene stack carries three probe
pixels, hot and cold, inside the gap footprint and outside it, and exactly
one of them may disappear. The geometry alone covers 11,664 pixels here; a
regression that drops the value gate fails on the count, not on a subtlety.

``qa_count`` is never masked, because zero observations is data and the count
layer stays the evidence behind every P95 value. The offset estimator is
untouched by construction here (destripe off) and by design everywhere: the
mask applies after ``compute_annual_composite`` returns, exactly like the
land mask's output-side application.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from landsat_lst import ged
from landsat_lst.config import settings
from landsat_lst.models import ProcessingJob, TileId
from landsat_lst.pipeline import process_tile
from landsat_lst.tiling import geobox_for_bbox

h5py = pytest.importorskip("h5py")

#: A 0.1-degree corner of tile N40W075: 360 x 360 native pixels, 10 x 10 GED
#: cells, all inside granule (40, -75).
BBOX = (-75.0, 39.9, -74.9, 40.0)

#: The gap cell and its expected buffered native footprint (buffer 1 cell):
#: cell (2, 3) -> native rows [36, 144), cols [72, 180).
GAP_CELL = (2, 3)
GAP_ROWS = slice(36, 144)
GAP_COLS = slice(72, 180)

#: Probe pixels, planted in every scene. Only the first satisfies both halves
#: of the rule, so only the first may be dropped.
HOT_IN_GAP = (40, 80)
COOL_IN_GAP = (40, 100)
HOT_OUTSIDE_GAP = (200, 200)

#: Well above settings.ged_gap_hot_threshold_c, and inside the composite's own
#: plausibility clamp (settings.lst_valid_max, 80 degC) so nothing else drops it.
HOT_C = 75.0


def _write_granule(ged_dir, lat_top, lon_west, numobs):
    lat = np.linspace(lat_top, lat_top - 1, 100)[:, None] * np.ones((1, 100))
    lon = np.ones((100, 1)) * np.linspace(lon_west, lon_west + 1, 100)[None, :]
    ged_dir.mkdir(parents=True, exist_ok=True)
    name = f"AG1km.v003.{lat_top:02d}.-{-lon_west:03d}.0010.h5"
    with h5py.File(ged_dir / name, "w") as f:
        f["Geolocation/Latitude"] = lat
        f["Geolocation/Longitude"] = lon
        f["Observations/NumObs"] = numobs.astype(np.int16)


@pytest.fixture
def stubbed_pipeline(tmp_path, monkeypatch):
    """process_tile with external I/O stubbed and a real synthetic GED source."""
    geobox = geobox_for_bbox(BBOX)
    height, width = int(geobox.shape[0]), int(geobox.shape[1])
    t = geobox.transform
    lons = t.c + t.a * (np.arange(width) + 0.5)
    lats = t.f + t.e * (np.arange(height) + 0.5)
    times = np.array(
        ["2024-06-01T13:45:12.482915", "2024-07-03T13:45:12.483052"], dtype="datetime64[ns]"
    )

    def load_scenes(items, bbox, **kwargs):
        rng = np.random.default_rng(11)
        celsius = 25.0 + rng.normal(0.0, 2.0, (len(times), height, width))
        # Planted in every scene, so each survives the P95 as itself.
        celsius[:, HOT_IN_GAP[0], HOT_IN_GAP[1]] = HOT_C
        celsius[:, HOT_OUTSIDE_GAP[0], HOT_OUTSIDE_GAP[1]] = HOT_C
        dn = ((celsius + 273.15) - 149.0) / 0.00341802
        return xr.Dataset(
            {
                "lwir11": (["time", "latitude", "longitude"], dn.astype(np.float32)),
                "qa_pixel": (
                    ["time", "latitude", "longitude"],
                    np.full((len(times), height, width), 21824, dtype="uint16"),
                ),
            },
            coords={"time": times, "latitude": lats, "longitude": lons},
        )

    def build_land_mask(geobox, latitude, longitude):
        return xr.DataArray(
            np.ones((latitude.size, longitude.size), dtype=bool),
            dims=["latitude", "longitude"],
            coords={"latitude": latitude, "longitude": longitude},
        )

    monkeypatch.setattr("landsat_lst.pipeline.resolve_items", lambda _job: [object(), object()])
    monkeypatch.setattr("landsat_lst.pipeline._patch_url_for", lambda _items: None)
    monkeypatch.setattr("landsat_lst.pipeline.geobox_for_bbox", lambda _bbox, _factor=1: geobox)
    monkeypatch.setattr("landsat_lst.pipeline.load_scenes", load_scenes)
    monkeypatch.setattr("landsat_lst.pipeline._build_land_mask", build_land_mask)
    monkeypatch.setattr("landsat_lst.pipeline.cache_for_items", lambda **_kwargs: None)
    monkeypatch.setattr(settings, "destripe", False)

    # Synthetic granules: the core with one gap cell, plus the margin ring the
    # buffered window reaches across the north and west granule boundaries.
    ged_dir = tmp_path / "aster_ged"
    numobs = np.full((100, 100), 7, dtype=np.int16)
    numobs[GAP_CELL] = 0
    _write_granule(ged_dir, 40, -75, numobs)
    for lat_top, lon_west in [(40, -76), (41, -75), (41, -76)]:
        _write_granule(ged_dir, lat_top, lon_west, np.full((100, 100), 7, dtype=np.int16))
    monkeypatch.setattr(settings, "ged_dir", ged_dir)
    monkeypatch.setattr(settings, "ged_artifact", tmp_path / "absent.npz")
    # The wheel now carries the production artifact, which outranks a
    # granule directory by design; these fixtures test the granule path.
    monkeypatch.setattr(ged, "packaged_artifact_path", lambda: None)

    return ProcessingJob(tile=TileId(lat=40, lon=-75), year=2024)


class TestGedMaskInComposite:
    def test_only_the_hot_pixel_inside_the_gap_is_dropped(self, stubbed_pipeline):
        composite = process_tile(stubbed_pipeline).compute()

        lst = composite["lst_p95"].values
        expected = np.zeros(lst.shape, dtype=bool)
        expected[HOT_IN_GAP] = True

        np.testing.assert_array_equal(np.isnan(lst), expected)
        assert int(np.isnan(lst).sum()) == 1

    def test_ordinary_data_inside_the_gap_survives(self, stubbed_pipeline):
        """The failure the geometric mask shipped with, pinned from the front.

        Every pixel of the gap footprint except the planted hot one keeps its
        value. Under the unconditional rule all 11,664 were lost.
        """
        lst = process_tile(stubbed_pipeline).compute()["lst_p95"].values

        footprint = lst[GAP_ROWS, GAP_COLS]
        assert int(np.count_nonzero(np.isnan(footprint))) == 1
        assert footprint.size == 108 * 108
        assert lst[COOL_IN_GAP] == pytest.approx(25.0, abs=6.0)

    def test_a_hot_pixel_outside_the_gap_survives(self, stubbed_pipeline):
        """The threshold masks nothing on its own. Only the conjunction does."""
        lst = process_tile(stubbed_pipeline).compute()["lst_p95"].values
        assert lst[HOT_OUTSIDE_GAP] == pytest.approx(HOT_C, abs=0.1)

    def test_qa_count_is_never_masked(self, stubbed_pipeline):
        # Every pixel of every month keeps its count, gap cells included. Two
        # scenes in two months over all-clear QA means those months carry 1
        # everywhere.
        qa = process_tile(stubbed_pipeline).compute()["qa_count"].values
        assert qa[:, GAP_ROWS, GAP_COLS].sum() == qa[:, 0:108, 0:1].sum() * 108
        assert int(qa.sum(axis=0).min()) == 2

    def test_raising_the_threshold_spares_the_hot_pixel_too(self, stubbed_pipeline, monkeypatch):
        """The threshold is the whole decision, so nothing is dropped above it."""
        monkeypatch.setattr(settings, "ged_gap_hot_threshold_c", 78.0)
        lst = process_tile(stubbed_pipeline).compute()["lst_p95"].values
        assert not np.isnan(lst).any()

    def test_toggle_off_is_a_noop(self, stubbed_pipeline, monkeypatch):
        monkeypatch.setattr(settings, "ged_gap_mask", False)
        composite = process_tile(stubbed_pipeline).compute()
        assert not np.isnan(composite["lst_p95"].values).any()
        assert int(composite["qa_count"].values.sum(axis=0).min()) == 2
