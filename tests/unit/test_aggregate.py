"""The nominal ~100 m aggregation contract, one test per required property.

Issue #120's follow-up decision lists the evidence an implementation must
produce. Each class below answers one line of that list, and the docstrings say
which. Nothing here loads a pixel or touches a network.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from landsat_lst.aggregate import (
    AGGREGATION_VERSION,
    aggregate_to_output_grid,
    aligned_source_chunk,
    row_area_weights,
    source_row_weights,
    spatial_dims,
    support_fraction,
)
from landsat_lst.config import settings

pytestmark = pytest.mark.unit

FACTOR = 3
CELLS = FACTOR * FACTOR


def _stack(values: np.ndarray, *, lat0: float = 40.0, spacing: float = 1 / 3600) -> xr.DataArray:
    """A source-grid stack with real descending latitudes and one time step."""
    scenes, rows, cols = values.shape
    return xr.DataArray(
        values,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": np.array(
                [
                    np.datetime64("2021-06-01T13:45:12.482915") + np.timedelta64(i, "D")
                    for i in range(scenes)
                ]
            ),
            "latitude": lat0 - spacing * np.arange(rows),
            "longitude": -75.0 + spacing * np.arange(cols),
        },
    )


class TestAlignment:
    """#120: "exact 3x alignment and 6000 x 6000 tile geometry"."""

    def test_the_configured_factor_is_an_exact_three(self):
        assert settings.spatial_aggregation_factor == FACTOR
        assert settings.pixels_per_degree == FACTOR * settings.output_pixels_per_degree

    def test_a_block_of_nine_becomes_one_cell(self):
        out = aggregate_to_output_grid(_stack(np.ones((1, 9, 12), "float32")), factor=FACTOR)
        assert out.shape == (1, 3, 4)

    def test_a_ragged_edge_is_refused_rather_than_trimmed(self):
        """Trimming would put a partial block on the shared global grid."""
        with pytest.raises(ValueError, match="not a whole multiple of 3"):
            aggregate_to_output_grid(_stack(np.ones((1, 10, 9), "float32")), factor=FACTOR)

    def test_a_ragged_column_edge_is_refused_too(self):
        with pytest.raises(ValueError, match="not a whole multiple of 3"):
            aggregate_to_output_grid(_stack(np.ones((1, 9, 10), "float32")), factor=FACTOR)

    def test_factor_one_is_the_identity(self):
        stack = _stack(np.ones((1, 5, 5), "float32"))
        assert aggregate_to_output_grid(stack, factor=1) is stack


class TestValidAreaRule:
    """#120: "the 5/9 boundary, including masked/fill values never entering the mean"."""

    @staticmethod
    def _block_with_valid(n_valid: int, value: float = 10.0) -> xr.DataArray:
        """One 3x3 block with ``n_valid`` cells set and the rest NaN."""
        flat = np.full(CELLS, np.nan, dtype="float32")
        flat[:n_valid] = value
        return _stack(flat.reshape(1, FACTOR, FACTOR))

    @pytest.mark.parametrize("n_valid", range(CELLS + 1))
    def test_the_boundary_sits_exactly_at_five_of_nine(self, n_valid):
        out = aggregate_to_output_grid(
            self._block_with_valid(n_valid), factor=FACTOR, min_valid_cells=5
        )
        value = float(out.values[0, 0, 0])
        if n_valid >= 5:
            assert value == pytest.approx(10.0, abs=1e-3)
        else:
            assert np.isnan(value)

    def test_the_default_threshold_is_five(self):
        assert settings.min_valid_source_cells == 5
        default = aggregate_to_output_grid(self._block_with_valid(5), factor=FACTOR)
        explicit = aggregate_to_output_grid(
            self._block_with_valid(5), factor=FACTOR, min_valid_cells=5
        )
        assert np.array_equal(default.values, explicit.values, equal_nan=True)

    def test_masked_cells_never_pull_the_mean_toward_zero(self):
        """The failure a naive ``fillna(0).mean()`` would produce."""
        flat = np.full(CELLS, np.nan, dtype="float32")
        flat[:5] = 30.0
        out = aggregate_to_output_grid(_stack(flat.reshape(1, FACTOR, FACTOR)), factor=FACTOR)
        # Mean of the five valid cells, not 5 * 30 / 9 = 16.67.
        assert float(out.values[0, 0, 0]) == pytest.approx(30.0, abs=1e-3)

    def test_a_zero_valued_cell_still_counts_as_an_observation(self):
        """0 degrees C is data. Only NaN is absence, and the two must not merge."""
        flat = np.full(CELLS, np.nan, dtype="float32")
        flat[:4] = 9.0
        flat[4] = 0.0
        out = aggregate_to_output_grid(_stack(flat.reshape(1, FACTOR, FACTOR)), factor=FACTOR)
        assert float(out.values[0, 0, 0]) == pytest.approx(4 * 9.0 / 5, abs=1e-3)

    def test_a_threshold_of_zero_is_refused(self):
        """It would emit a temperature for a block with no observation behind it."""
        with pytest.raises(ValueError, match=r"outside 1\.\.9"):
            aggregate_to_output_grid(self._block_with_valid(0), factor=FACTOR, min_valid_cells=0)

    def test_a_threshold_above_the_block_is_refused(self):
        with pytest.raises(ValueError, match=r"outside 1\.\.9"):
            aggregate_to_output_grid(self._block_with_valid(9), factor=FACTOR, min_valid_cells=10)

    def test_support_fraction_reports_what_the_rule_thresholds(self):
        out = support_fraction(self._block_with_valid(5), factor=FACTOR)
        assert float(out.values[0, 0, 0]) == pytest.approx(5 / 9)


class TestAreaWeighting:
    """#120: "area-weighted aggregation on asymmetric fixtures"."""

    def test_weights_come_from_the_sine_of_the_latitude_edges(self):
        """Exact spherical band area, not the small-angle cosine form."""
        edges = np.array([60.0, 50.0, 40.0])
        expected = np.abs(np.diff(np.sin(np.deg2rad(edges))))
        assert row_area_weights(edges) == pytest.approx(expected)

    def test_rows_nearer_the_equator_weigh_more(self):
        """The direction of the effect, which a transposed weight would flip."""
        weights = source_row_weights(np.array([60.0, 59.0, 58.0]), -1.0)
        assert weights[0] < weights[1] < weights[2]

    def test_an_ascending_axis_gets_the_same_weights_in_its_own_order(self):
        descending = source_row_weights(np.array([60.0, 59.0, 58.0]), -1.0)
        ascending = source_row_weights(np.array([58.0, 59.0, 60.0]), 1.0)
        assert ascending == pytest.approx(descending[::-1])

    def test_the_row_variation_is_negligible_and_measured(self):
        """The claim the documentation makes, checked rather than asserted.

        At 60 degrees, the edge of the published latitude band and so the worst
        case, a block's three source rows of 1/3600 degree differ in area by
        1.7e-5 relative -- ``tan(60) * 2 / 3600`` in radians. The bound is
        stated as 2e-5 rather than the measured value so that a change in the
        factor or the grid fails here loudly instead of drifting.
        """
        weights = source_row_weights(60.0 - np.arange(3) / 3600, -1 / 3600)
        spread = (weights.max() - weights.min()) / weights.mean()
        assert spread == pytest.approx(1.68e-5, rel=0.02)
        assert spread < 2e-5

    def test_an_asymmetric_block_catches_a_transpose(self):
        """Row-varying values and column-varying values must not swap.

        A weight vector applied to the wrong axis would still produce a number,
        and on a symmetric fixture it would produce the *right* number. This
        block is asymmetric in both directions, so it cannot.
        """
        values = np.arange(CELLS, dtype="float32").reshape(1, FACTOR, FACTOR)
        out = aggregate_to_output_grid(_stack(values, lat0=60.0), factor=FACTOR)
        weights = source_row_weights(60.0 - np.arange(3) / 3600, -1 / 3600)
        expected = float((values[0] * weights[:, None]).sum() / (weights.sum() * FACTOR))
        assert float(out.values[0, 0, 0]) == pytest.approx(expected, rel=1e-6)

    def test_weighting_shifts_the_answer_off_the_plain_mean(self):
        """Proof the weights are applied at all, not silently dropped."""
        values = np.arange(CELLS, dtype="float32").reshape(1, FACTOR, FACTOR)
        weighted = float(
            aggregate_to_output_grid(_stack(values, lat0=60.0), factor=FACTOR).values[0, 0, 0]
        )
        assert weighted != float(values.mean())

    def test_a_stack_with_no_latitude_coordinate_falls_back_to_equal_weights(self):
        """A synthetic benchmark stack has no latitude to weight by."""
        values = np.arange(CELLS, dtype="float32").reshape(1, FACTOR, FACTOR)
        bare = xr.DataArray(values, dims=["time", "y", "x"])
        out = aggregate_to_output_grid(bare, factor=FACTOR)
        assert float(out.values[0, 0, 0]) == pytest.approx(float(values.mean()))


class TestDimensionNaming:
    """The reducer must not care whether the axes are lat/lon or y/x."""

    def test_it_finds_the_two_spatial_dims_whatever_they_are_called(self):
        assert spatial_dims(xr.DataArray(np.zeros((2, 3, 3)), dims=["time", "y", "x"])) == [
            "y",
            "x",
        ]
        assert spatial_dims(
            xr.DataArray(np.zeros((2, 3, 3)), dims=["time", "latitude", "longitude"])
        ) == ["latitude", "longitude"]

    def test_a_stack_with_three_spatial_dims_is_refused(self):
        odd = xr.DataArray(np.zeros((2, 3, 3, 3)), dims=["time", "band", "y", "x"])
        with pytest.raises(ValueError, match="exactly two spatial dims"):
            spatial_dims(odd)


class TestTimeAxis:
    """#120: one delivered observation per solar day, on the same time axis."""

    def test_the_time_coordinate_survives_untouched(self):
        stack = _stack(np.ones((4, 9, 9), "float32"))
        out = aggregate_to_output_grid(stack, factor=FACTOR)
        assert out.sizes["time"] == 4
        assert np.array_equal(out["time"].values, stack["time"].values)

    def test_each_scene_is_reduced_independently(self):
        values = np.stack(
            [np.full((FACTOR, FACTOR), 10.0), np.full((FACTOR, FACTOR), 20.0)]
        ).astype("float32")
        out = aggregate_to_output_grid(_stack(values), factor=FACTOR)
        assert float(out.values[0, 0, 0]) == pytest.approx(10.0, abs=1e-3)
        assert float(out.values[1, 0, 0]) == pytest.approx(20.0, abs=1e-3)


class TestCoordinateStamping:
    """Delivered labels come from the grid, never from averaged source labels."""

    def test_supplied_coordinates_replace_the_coarsened_means(self):
        stack = _stack(np.ones((1, 9, 9), "float32"))
        exact = np.array([1.0, 2.0, 3.0])
        out = aggregate_to_output_grid(
            stack, factor=FACTOR, coords={"latitude": exact, "longitude": exact}
        )
        assert np.array_equal(out["latitude"].values, exact)
        assert np.array_equal(out["longitude"].values, exact)

    def test_a_coordinate_of_the_wrong_length_is_refused(self):
        """It would mean the geobox is not the aggregation of the loaded one."""
        stack = _stack(np.ones((1, 9, 9), "float32"))
        with pytest.raises(ValueError, match="is not the aggregation"):
            aggregate_to_output_grid(stack, factor=FACTOR, coords={"latitude": np.zeros(4)})


class TestChunkAlignment:
    """Aligned chunks skip a rechunk; a misaligned one still gives the same answer."""

    def test_a_chunk_edge_rounds_up_to_a_whole_number_of_cells(self):
        assert aligned_source_chunk(512, FACTOR) == 513
        assert aligned_source_chunk(513, FACTOR) == 513
        assert aligned_source_chunk(256, FACTOR) == 258
        assert aligned_source_chunk(512, 1) == 512

    def test_the_answer_does_not_depend_on_the_caller_s_chunking(self):
        values = np.arange(4 * 9 * 9, dtype="float32").reshape(4, 9, 9)
        eager = aggregate_to_output_grid(_stack(values), factor=FACTOR)
        misaligned = aggregate_to_output_grid(
            _stack(values).chunk({"latitude": 4, "longitude": 4}), factor=FACTOR
        )
        np.testing.assert_allclose(eager.values, misaligned.compute().values, rtol=1e-6)

    def test_it_stays_lazy_over_a_dask_stack(self):
        chunked = _stack(np.ones((4, 9, 9), "float32")).chunk({"time": 2})
        assert aggregate_to_output_grid(chunked, factor=FACTOR).chunks is not None


class TestProvenance:
    """The contract a plan digest and a reader both need to see."""

    def test_the_result_records_the_rule_it_was_produced_under(self):
        out = aggregate_to_output_grid(_stack(np.ones((1, 9, 9), "float32")), factor=FACTOR)
        assert out.attrs["aggregation_factor"] == FACTOR
        assert out.attrs["min_valid_source_cells"] == settings.min_valid_source_cells
        assert out.attrs["aggregation_version"] == AGGREGATION_VERSION
