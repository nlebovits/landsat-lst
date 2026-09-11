"""The 2026-09-04 S30W065 failure: labels on the loaded axis, stack post-rejection.

Every one of 35 composite shards died with

    IndexError: Index is out of bounds for axis 0 with size 912

because WRS path labels were derived from the 1,031 solar-day steps the band
loaded, while ``_composite_graph`` receives the stack **after** de-striping has
dropped rejected scenes -- 912 of them. Positional indices addressed the wrong
axis.

These use the real cardinalities. The spatial dims stay tiny: the defect is on
the time axis, and a production-width grid here would build float64
intermediates that OOM a CI worker (CLAUDE.md).
"""

from __future__ import annotations

import dask.array as dsa
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from shapely.geometry import box

from landsat_lst import wrs
from landsat_lst.pipeline import _composite_graph

pytestmark = pytest.mark.integration

LOADED = 1031  # solar-day steps S30W065 loads
SURVIVING = 912  # what de-striping leaves, measured on the failed run
WEST, EAST = "228", "229"


def _times(n: int) -> np.ndarray:
    # Sub-second stamps on purpose: a whole-second axis hides the precision
    # class of bug this join has already paid for once (CLAUDE.md).
    base = np.datetime64("2021-01-01T14:02:11.123456789", "ns")
    return base + np.arange(n) * np.timedelta64(43200, "s")


def _grid(width: int, height: int) -> GeoBox:
    return GeoBox.from_bbox((0.0, 0.0, 10.0, 1.0), crs="EPSG:4326", shape=(height, width))


def _polygons() -> dict:
    return {WEST: box(-1.0, -1.0, 6.0, 2.0), EAST: box(4.0, -1.0, 11.0, 2.0)}


def _stack(values: np.ndarray, times: np.ndarray) -> xr.DataArray:
    n_t, n_y, n_x = values.shape
    return xr.DataArray(
        dsa.from_array(values, chunks=(n_t, n_y, n_x)),
        dims=["time", "latitude", "longitude"],
        coords={
            "time": times,
            "latitude": np.arange(n_y, dtype="float64"),
            "longitude": np.arange(n_x, dtype="float64"),
        },
    )


@pytest.fixture
def production_shaped():
    """1,031 labelled steps, 912 of which survive rejection."""
    rng = np.random.default_rng(0)
    n_y = n_x = 8
    loaded_times = _times(LOADED)
    labels = xr.DataArray(
        np.array([WEST if i % 2 == 0 else EAST for i in range(LOADED)], dtype=object),
        dims=["time"],
        coords={"time": loaded_times},
    )
    # De-striping keeps a non-contiguous subset, exactly as offset rejection does.
    keep = np.sort(rng.choice(LOADED, size=SURVIVING, replace=False))
    values = rng.uniform(10.0, 40.0, size=(SURVIVING, n_y, n_x)).astype(np.float32)
    survivors = _stack(values, loaded_times[keep])
    return labels, survivors, keep, values


def test_a_post_rejection_stack_composites_without_running_off_the_axis(production_shaped):
    labels, survivors, _keep, _values = production_shaped
    weights = wrs.path_weights(_grid(8, 8), _polygons())

    out = _composite_graph(survivors, path_of_step=labels, weights=weights)["lst_p95"]

    assert survivors.sizes["time"] == SURVIVING
    assert labels.sizes["time"] == LOADED
    assert out.compute().values.shape == (8, 8)


def test_each_path_gets_exactly_its_surviving_scenes(production_shaped):
    """The join must select by time value, so rejection thins each path."""
    labels, survivors, keep, _values = production_shaped
    aligned = labels.sel(time=survivors["time"]).values

    assert aligned.shape[0] == SURVIVING
    for path in (WEST, EAST):
        expected = sum(1 for i in keep if (WEST if i % 2 == 0 else EAST) == path)
        assert int((aligned == path).sum()) == expected
    # and the labels follow the survivors, not the loaded order
    assert list(aligned) == [WEST if i % 2 == 0 else EAST for i in keep]


def test_positional_labels_of_the_wrong_length_are_refused(production_shaped):
    """A bare array sized to the loaded axis must fail loudly, not silently."""
    labels, survivors, _keep, _values = production_shaped
    weights = wrs.path_weights(_grid(8, 8), _polygons())

    with pytest.raises(ValueError, match=r"1031 steps but the stack carries 912"):
        _composite_graph(survivors, path_of_step=labels.values, weights=weights)


def test_a_step_the_labels_do_not_cover_raises(production_shaped):
    """A survivor missing from the labels is a defect, not something to guess."""
    labels, survivors, _keep, _values = production_shaped
    weights = wrs.path_weights(_grid(8, 8), _polygons())
    thinned = labels.isel(time=slice(1, None))

    with pytest.raises(KeyError):
        _composite_graph(survivors, path_of_step=thinned, weights=weights)
