"""Unit tests for compute_annual_composite function.

These tests verify the P95 computation and the per-month QA climatology logic
independent of distributed execution context. P50 was removed per issue #22.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.pipeline import compute_annual_composite
from landsat_lst.zarr_writer import LST_OFFSET, LST_SCALE, encode_lst_uint16


def _monthly_times(year: int = 2024, per_month: int = 2) -> np.ndarray:
    """Datetime coords with ``per_month`` observations in each calendar month."""
    days = [5, 20][:per_month]
    stamps = [f"{year}-{m:02d}-{d:02d}" for m in range(1, 13) for d in days]
    return pd.to_datetime(stamps).values


class TestComputeAnnualComposite:
    """Tests for compute_annual_composite function."""

    @pytest.fixture
    def mock_landsat_data(self) -> xr.Dataset:
        """Mock Landsat data with 2 observations per calendar month (24 total).

        January (first 2 time steps) is clouded for the top half of the scene, so
        the per-month climatology should show January coverage 0 there and 2 for
        every other month / the bottom half.
        """
        np.random.seed(42)
        times = _monthly_times()
        n_time = len(times)  # 24
        n_y, n_x = 50, 50

        lwir_values = np.random.uniform(42000, 45000, (n_time, n_y, n_x)).astype(np.float32)

        qa_values = np.zeros((n_time, n_y, n_x), dtype=np.uint16)
        qa_values[0:2, :25, :] = 8  # Cloud bit on both January obs, top half

        return xr.Dataset(
            {
                "lwir11": (["time", "y", "x"], lwir_values),
                "qa_pixel": (["time", "y", "x"], qa_values),
            },
            coords={"time": times, "y": np.arange(n_y), "x": np.arange(n_x)},
        )

    def test_returns_expected_variables(self, mock_landsat_data):
        """Test that composite has lst_p95 and qa_count (no lst_p50)."""
        result = compute_annual_composite(mock_landsat_data)

        assert "lst_p95" in result.data_vars
        assert "qa_count" in result.data_vars
        assert "lst_p50" not in result.data_vars

    def test_qa_count_is_monthly_uint8(self, mock_landsat_data):
        """qa_count is a 12-month climatology stored as uint8."""
        result = compute_annual_composite(mock_landsat_data)

        qa = result["qa_count"]
        assert qa.dtype == np.uint8
        assert "month" in qa.dims
        assert qa.sizes["month"] == 12
        assert list(qa["month"].values) == list(range(1, 13))
        # Spatial dims preserved.
        assert qa.sizes["y"] == 50
        assert qa.sizes["x"] == 50

    def test_p95_in_reasonable_range(self, mock_landsat_data):
        """Test that P95 values are in reasonable temperature range."""
        result = compute_annual_composite(mock_landsat_data)

        p95 = result["lst_p95"].values
        valid_p95 = p95[p95 > -9000]

        assert valid_p95.min() > 15, f"P95 min {valid_p95.min()} too low"
        assert valid_p95.max() < 50, f"P95 max {valid_p95.max()} too high"
        assert valid_p95.mean() > 20, f"P95 mean {valid_p95.mean()} too low"
        assert valid_p95.mean() < 40, f"P95 mean {valid_p95.mean()} too high"

    def test_qa_count_reflects_monthly_masking(self, mock_landsat_data):
        """Per-month counts reflect the clouded January in the top half."""
        result = compute_annual_composite(mock_landsat_data)
        qa = result["qa_count"]

        # January (month=1): top half clouded -> 0, bottom half -> 2.
        jan = qa.sel(month=1).values
        assert jan[:25, :].max() == 0
        assert jan[25:, :].min() == 2

        # Every other month has both observations everywhere.
        feb_to_dec = qa.sel(month=slice(2, 12)).values
        assert feb_to_dec.min() == 2
        assert feb_to_dec.max() == 2

    def test_p95_not_all_same_value(self, mock_landsat_data):
        """Test that P95 has variation."""
        result = compute_annual_composite(mock_landsat_data)

        p95 = result["lst_p95"].values
        valid_p95 = p95[p95 > -9000]

        unique_count = len(np.unique(valid_p95))
        assert unique_count > 1, f"P95 has only {unique_count} unique values!"

    def test_p95_encoding_roundtrip(self, mock_landsat_data):
        """Test that P95 survives uint16 encoding roundtrip."""
        result = compute_annual_composite(mock_landsat_data)

        p95 = result["lst_p95"]
        encoded = encode_lst_uint16(p95)
        decoded = encoded.values * LST_SCALE + LST_OFFSET

        unique_encoded = np.unique(encoded.values[encoded.values > 0])
        assert len(unique_encoded) > 1, (
            f"Encoded P95 has only {len(unique_encoded)} unique non-zero values!"
        )

        valid_mask = p95.values > -9000
        np.testing.assert_array_almost_equal(
            decoded[valid_mask],
            p95.values[valid_mask],
            decimal=1,
        )

    def test_with_all_nan_pixel(self, mock_landsat_data):
        """A pixel clouded in every observation is nodata with zero monthly counts."""
        mock_landsat_data["qa_pixel"][:, 0, 0] = 8  # All cloud, all months

        result = compute_annual_composite(mock_landsat_data)

        assert result["lst_p95"].values[0, 0] == -9999.0
        # Every month has zero valid observations at that pixel.
        assert int(result["qa_count"].sel(month=slice(1, 12)).values[:, 0, 0].sum()) == 0


class TestFloorAnomalyGuard:
    """Guard against P95 values landing on the encoding floor (issue #24).

    DN 0 is fill and DN 1 decodes to -49.99 C. A hot-season P95 over land never
    reaches that, so a pixel with observations that still produces such a value
    is a failed retrieval and becomes nodata rather than a plausible-looking
    temperature.
    """

    @pytest.fixture(autouse=True)
    def _no_destripe(self, monkeypatch):
        """De-striping is orthogonal to the floor guard.

        These fixtures use a 5x5 grid, far below ``destripe_min_scene_pixels``,
        so every scene would be discarded as too sparse to estimate an offset
        from. De-striping has its own tests in test_destripe_normalization.py.
        """
        monkeypatch.setattr(settings, "destripe", False)

    # lwir DN 21698 -> -49.9858 C, inside the DN 0/1 band the guard rejects.
    LWIR_ON_FLOOR = 21698
    # lwir DN 21705 -> -49.9619 C, cold but encodes to DN 3 and must survive.
    LWIR_ABOVE_FLOOR = 21705

    @pytest.fixture
    def warm_scene(self) -> xr.Dataset:
        """Clear-sky scene, uniformly warm, with room to plant anomalies."""
        times = _monthly_times()
        n_time, n_y, n_x = len(times), 5, 5
        return xr.Dataset(
            {
                "lwir11": (
                    ["time", "y", "x"],
                    np.full((n_time, n_y, n_x), 44000.0, dtype=np.float32),
                ),
                "qa_pixel": (["time", "y", "x"], np.zeros((n_time, n_y, n_x), dtype=np.uint16)),
            },
            coords={"time": times, "y": np.arange(n_y), "x": np.arange(n_x)},
        )

    def test_floor_pixel_with_observations_becomes_nodata(self, warm_scene):
        """A pixel whose P95 sits on the encoding floor is flagged as missing."""
        warm_scene["lwir11"][:, 2, 2] = self.LWIR_ON_FLOOR

        result = compute_annual_composite(warm_scene)

        assert result["lst_p95"].values[2, 2] == -9999.0
        # The pixel did have valid observations; only the value was rejected.
        assert int(result["qa_count"].values[:, 2, 2].sum()) == len(warm_scene.time)

    def test_neighbors_of_floor_pixel_are_untouched(self, warm_scene):
        """The guard rejects one pixel without disturbing the surrounding data."""
        warm_scene["lwir11"][:, 2, 2] = self.LWIR_ON_FLOOR

        result = compute_annual_composite(warm_scene)

        neighborhood = result["lst_p95"].values[1:4, 1:4]
        surrounding = np.delete(neighborhood.ravel(), 4)  # drop the centre pixel
        assert np.all(surrounding > 0), f"guard leaked into neighbors: {surrounding}"

    def test_cold_but_encodable_pixel_survives(self, warm_scene):
        """A value above the floor is kept, so the guard is not a blanket cold cut."""
        warm_scene["lwir11"][:, 1, 1] = self.LWIR_ABOVE_FLOOR

        result = compute_annual_composite(warm_scene)

        assert result["lst_p95"].values[1, 1] == pytest.approx(-49.96, abs=0.01)

    def test_rejected_pixel_encodes_to_fill(self, warm_scene):
        """Composite and encoded output agree: the rejected pixel becomes DN 0."""
        warm_scene["lwir11"][:, 2, 2] = self.LWIR_ON_FLOOR

        result = compute_annual_composite(warm_scene)
        encoded = encode_lst_uint16(result["lst_p95"])

        assert encoded.values[2, 2] == 0
        assert encoded.values[0, 0] != 0

    def test_clean_scene_is_unaffected(self, warm_scene):
        """With no anomalies present the guard changes nothing."""
        result = compute_annual_composite(warm_scene)

        assert np.all(result["lst_p95"].values > 0)


class TestDaskComposite:
    """Tests for compute_annual_composite with Dask arrays."""

    @pytest.fixture
    def dask_landsat_data(self) -> xr.Dataset:
        """Create Dask-backed mock Landsat data with monthly datetime coords."""
        import dask.array as da

        np.random.seed(42)
        times = _monthly_times()
        n_time = len(times)
        n_y, n_x = 50, 50

        lwir_np = np.random.uniform(42000, 45000, (n_time, n_y, n_x)).astype(np.float32)
        qa_np = np.zeros((n_time, n_y, n_x), dtype=np.uint16)
        qa_np[0:2, :25, :] = 8  # Cloud in January, top half

        lwir_dask = da.from_array(lwir_np, chunks=(6, 25, 25))
        qa_dask = da.from_array(qa_np, chunks=(6, 25, 25))

        return xr.Dataset(
            {
                "lwir11": (["time", "y", "x"], lwir_dask),
                "qa_pixel": (["time", "y", "x"], qa_dask),
            },
            coords={"time": times, "y": np.arange(n_y), "x": np.arange(n_x)},
        )

    def test_dask_p95_has_variation(self, dask_landsat_data):
        """Test that P95 with Dask arrays has variation."""
        result = compute_annual_composite(dask_landsat_data)

        p95 = result["lst_p95"].compute().values
        valid_p95 = p95[p95 > -9000]

        unique_count = len(np.unique(valid_p95))
        assert unique_count > 1, f"Dask P95 has only {unique_count} unique values!"

    def test_dask_monthly_qa(self, dask_landsat_data):
        """Dask composite yields the 12-month uint8 qa_count."""
        result = compute_annual_composite(dask_landsat_data)

        qa = result["qa_count"].compute()
        assert qa.dtype == np.uint8
        assert qa.sizes["month"] == 12
        assert "lst_p50" not in result.data_vars
