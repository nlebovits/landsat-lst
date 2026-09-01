"""The GED gap mask through the production composite path.

Runs ``process_tile`` end to end with everything external stubbed -- the STAC
query, the scene load, the land mask -- while the GED mask itself is built
*for real* from synthetic AG100 granules. One known gap cell must remove
exactly its buffered 36 x 36-pixel-per-cell footprint from ``lst_p95`` and
nothing from ``qa_count``: zero observations is data, and the count layer
stays the evidence behind every P95 value. The offset estimator is asserted
untouched by construction here (destripe off) and by design everywhere --
the mask applies after ``compute_annual_composite`` returns, exactly like
the land mask's output-side application.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.models import ProcessingJob, TileId
from landsat_lst.pipeline import process_tile
from landsat_lst.tiling import geobox_for_bbox, output_geobox_for_bbox

h5py = pytest.importorskip("h5py")

#: A 0.1-degree corner of tile N40W075: 360 x 360 source pixels, 120 x 120
#: delivered pixels, 10 x 10 GED cells, all inside granule (40, -75).
BBOX = (-75.0, 39.9, -74.9, 40.0)

#: Delivered pixels per 0.01-degree GED cell: 12 at 1,200 px/deg. Derived
#: rather than typed, because the mask is applied on the DELIVERED grid now
#: (ADR-017) and a hard-coded 36 would be a source-grid number quietly
#: asserting the mask had not moved with it.
PX_PER_GED_CELL = round(0.01 * settings.output_pixels_per_degree)

#: The gap cell and its expected buffered delivered footprint (buffer 1 cell):
#: cell (2, 3) spans cells [1, 4) x [2, 5) once buffered, so delivered rows
#: [12, 48) and cols [24, 60) -- a 36 x 36 square.
GAP_CELL = (2, 3)
GAP_EDGE = 3 * PX_PER_GED_CELL
GAP_ROWS = slice((GAP_CELL[0] - 1) * PX_PER_GED_CELL, (GAP_CELL[0] + 2) * PX_PER_GED_CELL)
GAP_COLS = slice((GAP_CELL[1] - 1) * PX_PER_GED_CELL, (GAP_CELL[1] + 2) * PX_PER_GED_CELL)


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
    # The delivered grid the composite comes back on, and the grid the GED gap
    # mask is now rasterized against. Real, not a toy: the mask under test maps
    # ~1 km GED cells onto whatever grid it is handed, and the point of this
    # test is that it lands on the same pixels the composite publishes.
    output_geobox = output_geobox_for_bbox(BBOX)
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
    monkeypatch.setattr("landsat_lst.pipeline.output_geobox_for_bbox", lambda _bbox: output_geobox)
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

    return ProcessingJob(tile=TileId(lat=40, lon=-75), year=2024)


class TestGedMaskInComposite:
    def test_gap_cell_removes_exact_pixels_from_lst_only(self, stubbed_pipeline):
        composite = process_tile(stubbed_pipeline).compute()

        lst = composite["lst_p95"].values
        expected = np.zeros(lst.shape, dtype=bool)
        expected[GAP_ROWS, GAP_COLS] = True

        np.testing.assert_array_equal(np.isnan(lst), expected)
        assert int(np.isnan(lst).sum()) == GAP_EDGE * GAP_EDGE

        # qa_count is untouched: every pixel of every month keeps its count,
        # gap cells included. Two scenes in two months over all-clear QA means
        # those months carry 1 everywhere.
        qa = composite["qa_count"].values
        assert qa[:, GAP_ROWS, GAP_COLS].sum() == qa[:, 0:GAP_EDGE, 0:1].sum() * GAP_EDGE
        assert int(qa.sum(axis=0).min()) == 2

    def test_toggle_off_is_a_noop(self, stubbed_pipeline, monkeypatch):
        monkeypatch.setattr(settings, "ged_gap_mask", False)
        composite = process_tile(stubbed_pipeline).compute()
        assert not np.isnan(composite["lst_p95"].values).any()
        assert int(composite["qa_count"].values.sum(axis=0).min()) == 2
