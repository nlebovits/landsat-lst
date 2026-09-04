"""WRS-path feathering: geometry weights and the per-path composite.

Grids here are deliberately tiny and built directly, never from a tile name.
A toy plan does not make a toy grid (CLAUDE.md): anything deriving its geobox
from a real tile gets the production 18,000-column grid and builds float64
intermediates that OOM a CI worker.
"""

from __future__ import annotations

import dask.array as dsa
import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from shapely.geometry import box

from landsat_lst import wrs
from landsat_lst.config import settings
from landsat_lst.pipeline import _composite_graph

pytestmark = pytest.mark.unit

WEST, EAST, BOTH = "001", "002", "003"


def grid(width: int = 60, height: int = 6) -> GeoBox:
    return GeoBox.from_bbox((0.0, 0.0, 10.0, 1.0), crs="EPSG:4326", shape=(height, width))


def two_paths() -> dict[str, object]:
    # overlap on x in [4, 6]
    return {WEST: box(-1.0, -1.0, 6.0, 2.0), EAST: box(4.0, -1.0, 11.0, 2.0)}


def midrow(a: np.ndarray) -> np.ndarray:
    return a[a.shape[0] // 2]


class TestWeights:
    def test_weights_sum_to_one_wherever_a_path_covers(self):
        w = wrs.path_weights(grid(), two_paths())
        total = w.weight.sum(axis=0)
        assert np.allclose(total[w.covered], 1.0, atol=1e-6)
        assert np.all(total[~w.covered] == 0.0)

    def test_a_single_path_pixel_keeps_all_of_its_weight(self):
        w = wrs.path_weights(grid(), two_paths())
        single = w.n_paths_at_pixel == 1
        assert single.any()
        assert np.allclose(w.weight.sum(axis=0)[single], 1.0, atol=1e-6)
        # and it is the covering path that holds it
        for j, _path in enumerate(w.paths):
            held = single & (w.weight[j] > 0)
            assert np.all(w.weight[j][held] == pytest.approx(1.0, abs=1e-6))

    def test_each_weight_runs_from_one_to_zero_across_the_overlap(self):
        w = wrs.path_weights(grid(), two_paths(), factor=1)
        west = midrow(w.weight[w.paths.index(WEST)])
        east = midrow(w.weight[w.paths.index(EAST)])
        overlap = np.flatnonzero(midrow(w.n_paths_at_pixel) == 2)
        assert overlap.size > 4
        # west path fades out going east, east path fades in
        assert west[overlap[0]] > west[overlap[-1]]
        assert east[overlap[0]] < east[overlap[-1]]
        assert west[overlap[0]] == pytest.approx(1.0, abs=0.15)
        assert west[overlap[-1]] == pytest.approx(0.0, abs=0.15)
        assert np.allclose(west[overlap] + east[overlap], 1.0, atol=1e-6)

    def test_the_ramp_is_monotone_and_has_no_step(self):
        w = wrs.path_weights(grid(width=200), two_paths(), factor=1)
        west = midrow(w.weight[w.paths.index(WEST)])
        overlap = np.flatnonzero(midrow(w.n_paths_at_pixel) == 2)
        seg = west[overlap]
        assert np.all(np.diff(seg) <= 1e-6), "weight must not rise going east"
        assert np.max(np.abs(np.diff(seg))) < 0.25, "no hard boundary inside the overlap"

    def test_three_covering_paths_blend_continuously(self):
        polys = {**two_paths(), BOTH: box(5.0, -1.0, 11.0, 2.0)}
        w = wrs.path_weights(grid(width=200), polys)
        triple = w.n_paths_at_pixel == 3
        assert triple.any(), "the fixture must exercise k=3"
        assert np.allclose(w.weight.sum(axis=0)[triple], 1.0, atol=1e-6)
        assert np.all(w.weight[:, triple] >= 0.0)

    def test_no_path_means_no_weight_and_no_coverage(self):
        w = wrs.path_weights(grid(), {WEST: box(-5.0, -5.0, -4.0, -4.0)})
        assert not w.covered.any()
        assert np.all(w.weight == 0.0)

    def test_a_grid_too_small_to_coarsen_uses_the_exact_ramp(self):
        """Below the floor the coarse grid would carry the ramp on a few cells."""
        small = grid(width=60, height=6)  # 6 // 8 == 0 coarse rows
        assert np.array_equal(
            wrs.path_weights(small, two_paths(), factor=8).weight,
            wrs.path_weights(small, two_paths(), factor=1).weight,
        )

    def test_a_production_shaped_band_takes_the_coarse_path(self):
        """The floor must not reach a real band. 512 rows sat exactly on it."""
        for rows in (512, 514, 500, 256, 128):
            assert min(rows // 8, 18000 // 8) >= wrs._MIN_COARSE_EDGE, (
                f"a {rows}-row band would fall back to the exact ramp"
            )

    def test_the_coarse_ramp_tracks_the_exact_one(self):
        """Interpolating the ramp must not move a weight materially."""
        g = grid(width=2048, height=512)
        exact = wrs.path_weights(g, two_paths(), factor=1)
        coarse = wrs.path_weights(g, two_paths(), factor=8)
        assert coarse.paths == exact.paths
        assert np.array_equal(coarse.n_paths_at_pixel, exact.n_paths_at_pixel)
        # containment is exact whatever the factor, so single-path pixels match
        single = exact.n_paths_at_pixel == 1
        assert np.array_equal(coarse.weight[:, single], exact.weight[:, single])
        cov = exact.covered
        assert np.allclose(coarse.weight.sum(axis=0)[cov], 1.0, atol=1e-5)
        d = np.abs(coarse.weight - exact.weight)[:, cov]
        assert d.max() < 0.02, f"coarse ramp moved a weight by {d.max():.4f}"

    def test_path_order_cannot_change_the_weights(self):
        polys = {**two_paths(), BOTH: box(5.0, -1.0, 11.0, 2.0)}
        a = wrs.path_weights(grid(width=120), polys)
        b = wrs.path_weights(grid(width=120), dict(reversed(list(polys.items()))))
        assert a.paths == b.paths
        assert np.array_equal(a.weight, b.weight)


def stack(values: np.ndarray, times, chunk: int = 3) -> xr.DataArray:
    n_t, n_y, n_x = values.shape
    return xr.DataArray(
        dsa.from_array(values, chunks=(n_t, n_y, chunk)),
        dims=["time", "latitude", "longitude"],
        coords={
            "time": np.asarray(times, dtype="datetime64[ns]"),
            "latitude": np.arange(n_y, dtype="float64"),
            "longitude": np.arange(n_x, dtype="float64"),
        },
    )


def times(n: int):
    return np.array([np.datetime64("2021-01-01") + np.timedelta64(8 * i, "D") for i in range(n)])


class TestCompositeGraph:
    def _fixture(self, n_t=8, n_y=6, n_x=60, seed=0):
        rng = np.random.default_rng(seed)
        vals = rng.uniform(10.0, 40.0, size=(n_t, n_y, n_x)).astype(np.float32)
        labels = np.array([WEST if i % 2 == 0 else EAST for i in range(n_t)], dtype=object)
        return vals, labels

    def test_without_paths_it_is_the_pooled_composite(self):
        vals, _ = self._fixture()
        lst = stack(vals, times(vals.shape[0]))
        assert "lst_p95" in _composite_graph(lst)

    def test_a_pixel_reached_by_one_path_is_bit_identical_to_that_path_alone(self):
        vals, labels = self._fixture()
        lst = stack(vals, times(vals.shape[0]))
        w = wrs.path_weights(grid(width=vals.shape[2], height=vals.shape[1]), two_paths())
        got = _composite_graph(lst, path_of_step=labels, weights=w)["lst_p95"].compute().values

        single = w.n_paths_at_pixel == 1
        for j, path in enumerate(w.paths):
            mine = single & (w.weight[j] > 0)
            if not mine.any():
                continue
            alone = stack(vals[labels == path], times((labels == path).sum()))
            only = _composite_graph(alone)["lst_p95"].compute().values
            assert np.array_equal(got[mine], only[mine]), f"path {path} region moved"

    def test_permuting_the_paths_leaves_the_result_bit_identical(self):
        vals, labels = self._fixture()
        lst = stack(vals, times(vals.shape[0]))
        g = grid(width=vals.shape[2], height=vals.shape[1])
        a = (
            _composite_graph(lst, path_of_step=labels, weights=wrs.path_weights(g, two_paths()))[
                "lst_p95"
            ]
            .compute()
            .values
        )
        flipped = dict(reversed(list(two_paths().items())))
        b = (
            _composite_graph(lst, path_of_step=labels, weights=wrs.path_weights(g, flipped))[
                "lst_p95"
            ]
            .compute()
            .values
        )
        assert np.array_equal(a, b)

    def test_nodata_is_preserved_where_nothing_was_observed(self):
        vals, labels = self._fixture()
        vals[:, :, :5] = np.nan
        lst = stack(vals, times(vals.shape[0]))
        w = wrs.path_weights(grid(width=vals.shape[2], height=vals.shape[1]), two_paths())
        out = _composite_graph(lst, path_of_step=labels, weights=w)["lst_p95"].compute().values
        assert np.all(out[:, :5] == settings.nodata)
        assert np.all(out[:, 20:30] != settings.nodata)

    def test_mixed_path_steps_are_excluded_from_every_reduction(self):
        vals, labels = self._fixture()
        poisoned = vals.copy()
        mixed = 3
        labels = labels.copy()
        labels[mixed] = wrs.MIXED_PATH
        poisoned[mixed] = 999.0  # would dominate any P95 that used it
        lst = stack(poisoned, times(poisoned.shape[0]))
        w = wrs.path_weights(grid(width=vals.shape[2], height=vals.shape[1]), two_paths())
        out = _composite_graph(lst, path_of_step=labels, weights=w)["lst_p95"].compute().values
        assert np.nanmax(out[out != settings.nodata]) < 100.0

    def test_the_pooled_baseline_rides_the_same_compute(self):
        vals, labels = self._fixture()
        lst = stack(vals, times(vals.shape[0]))
        w = wrs.path_weights(grid(width=vals.shape[2], height=vals.shape[1]), two_paths())
        ds = _composite_graph(lst, path_of_step=labels, weights=w, emit_pooled=True)
        assert "lst_p95_pooled" in ds
        pooled = _composite_graph(lst)["lst_p95"].compute().values
        assert np.array_equal(ds["lst_p95_pooled"].compute().values, pooled)


class TestSharedSource:
    """The per-path split must not add a source pass."""

    def _counted(self, vals, chunk=3):
        reads: list[int] = []

        def tally(block):
            reads.append(1)
            return block

        n_t, n_y, n_x = vals.shape
        arr = dsa.from_array(vals, chunks=(n_t, n_y, chunk))
        counted = arr.map_blocks(tally, dtype=arr.dtype, meta=np.array((), dtype=arr.dtype))
        da = xr.DataArray(
            counted,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": times(n_t),
                "latitude": np.arange(n_y, dtype="float64"),
                "longitude": np.arange(n_x, dtype="float64"),
            },
        )
        return da, reads, arr.npartitions

    def test_per_path_reductions_read_each_source_block_once(self):
        rng = np.random.default_rng(1)
        vals = rng.uniform(10.0, 40.0, size=(8, 6, 60)).astype(np.float32)
        labels = np.array([WEST if i % 2 == 0 else EAST for i in range(8)], dtype=object)
        lst, reads, blocks = self._counted(vals)
        w = wrs.path_weights(grid(width=60, height=6), two_paths())
        ds = _composite_graph(lst, path_of_step=labels, weights=w, emit_pooled=True)
        import dask

        dask.compute(ds["lst_p95"], ds["qa_count"], ds["lst_p95_pooled"])
        assert len(reads) == blocks, f"{len(reads) / blocks:.2f} passes over the sources"

    def test_separate_computes_still_cost_a_pass_each(self):
        """The negative control, so 1.0 above means something."""
        rng = np.random.default_rng(1)
        vals = rng.uniform(10.0, 40.0, size=(8, 6, 60)).astype(np.float32)
        labels = np.array([WEST if i % 2 == 0 else EAST for i in range(8)], dtype=object)
        lst, reads, blocks = self._counted(vals)
        w = wrs.path_weights(grid(width=60, height=6), two_paths())
        ds = _composite_graph(lst, path_of_step=labels, weights=w, emit_pooled=True)
        ds["lst_p95"].compute()
        ds["lst_p95_pooled"].compute()
        assert len(reads) == 2 * blocks
