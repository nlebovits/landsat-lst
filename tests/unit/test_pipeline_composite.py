"""Unit tests for compute_annual_composite function.

These tests verify the P95 computation logic independent of
distributed execution context. P50 was removed per issue #22.
"""

import numpy as np
import pytest
import xarray as xr

from landsat_lst.pipeline import compute_annual_composite
from landsat_lst.zarr_writer import LST_OFFSET, LST_SCALE, encode_lst_uint16


class TestComputeAnnualComposite:
    """Tests for compute_annual_composite function."""

    @pytest.fixture
    def mock_landsat_data(self) -> xr.Dataset:
        """Create mock Landsat data with realistic patterns.

        Returns Dataset with:
        - lwir11: thermal band with values that should produce 15-35°C
        - qa_pixel: QA band with some cloud/shadow flagged pixels
        """
        np.random.seed(42)
        n_time, n_y, n_x = 20, 50, 50

        # LWIR11 DN values that convert to ~20-30°C using Landsat C2 L2 scaling
        lwir_values = np.random.uniform(42000, 45000, (n_time, n_y, n_x)).astype(np.float32)

        # QA pixel: mostly clear, some clouds
        qa_values = np.zeros((n_time, n_y, n_x), dtype=np.uint16)
        # Add clouds in first 5 time steps for half the pixels
        qa_values[0:5, :25, :] = 8  # Cloud bit set

        return xr.Dataset(
            {
                "lwir11": (["time", "y", "x"], lwir_values),
                "qa_pixel": (["time", "y", "x"], qa_values),
            },
            coords={
                "time": np.arange(n_time),
                "y": np.arange(n_y),
                "x": np.arange(n_x),
            },
        )

    def test_returns_expected_variables(self, mock_landsat_data):
        """Test that composite has lst_p95 and qa_count (no lst_p50)."""
        result = compute_annual_composite(mock_landsat_data)

        assert "lst_p95" in result.data_vars
        assert "qa_count" in result.data_vars
        # P50 was removed per issue #22
        assert "lst_p50" not in result.data_vars

    def test_p95_in_reasonable_range(self, mock_landsat_data):
        """Test that P95 values are in reasonable temperature range."""
        result = compute_annual_composite(mock_landsat_data)

        p95 = result["lst_p95"].values
        # Filter out nodata
        valid_p95 = p95[p95 > -9000]

        # Should be in range of input temperatures (roughly 20-35°C for P95)
        assert valid_p95.min() > 15, f"P95 min {valid_p95.min()} too low"
        assert valid_p95.max() < 50, f"P95 max {valid_p95.max()} too high"
        assert valid_p95.mean() > 20, f"P95 mean {valid_p95.mean()} too low"
        assert valid_p95.mean() < 40, f"P95 mean {valid_p95.mean()} too high"

    def test_qa_count_reflects_masking(self, mock_landsat_data):
        """Test that qa_count reflects cloud masking."""
        result = compute_annual_composite(mock_landsat_data)

        qa_count = result["qa_count"].values

        # First 25 rows should have 15 valid (20 - 5 clouded)
        # Last 25 rows should have 20 valid (no clouds)
        assert qa_count[:25, :].max() <= 15
        assert qa_count[25:, :].min() >= 20

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

        # Encode to uint16
        encoded = encode_lst_uint16(p95)

        # Decode back
        decoded = encoded.values * LST_SCALE + LST_OFFSET

        # Should not all be the same
        unique_encoded = np.unique(encoded.values[encoded.values > 0])
        assert len(unique_encoded) > 1, (
            f"Encoded P95 has only {len(unique_encoded)} unique non-zero values!"
        )

        # Decoded should be close to original
        valid_mask = p95.values > -9000
        np.testing.assert_array_almost_equal(
            decoded[valid_mask],
            p95.values[valid_mask],
            decimal=1,  # Allow 0.1°C rounding error
        )

    def test_with_all_nan_pixel(self, mock_landsat_data):
        """Test handling of pixels with all NaN (completely clouded)."""
        # Make one pixel completely clouded
        mock_landsat_data["qa_pixel"][:, 0, 0] = 8  # All cloud

        result = compute_annual_composite(mock_landsat_data)

        # That pixel should have nodata value
        assert result["lst_p95"].values[0, 0] == -9999.0
        assert result["qa_count"].values[0, 0] == 0


class TestDaskComposite:
    """Tests for compute_annual_composite with Dask arrays."""

    @pytest.fixture
    def dask_landsat_data(self) -> xr.Dataset:
        """Create Dask-backed mock Landsat data."""
        import dask.array as da

        np.random.seed(42)
        n_time, n_y, n_x = 20, 50, 50

        lwir_np = np.random.uniform(42000, 45000, (n_time, n_y, n_x)).astype(np.float32)
        qa_np = np.zeros((n_time, n_y, n_x), dtype=np.uint16)
        qa_np[0:5, :25, :] = 8  # Cloud

        # Convert to chunked Dask arrays
        lwir_dask = da.from_array(lwir_np, chunks=(10, 25, 25))
        qa_dask = da.from_array(qa_np, chunks=(10, 25, 25))

        return xr.Dataset(
            {
                "lwir11": (["time", "y", "x"], lwir_dask),
                "qa_pixel": (["time", "y", "x"], qa_dask),
            },
            coords={
                "time": np.arange(n_time),
                "y": np.arange(n_y),
                "x": np.arange(n_x),
            },
        )

    def test_dask_p95_has_variation(self, dask_landsat_data):
        """Test that P95 with Dask arrays has variation."""
        result = compute_annual_composite(dask_landsat_data)

        # Force computation
        p95 = result["lst_p95"].compute().values
        valid_p95 = p95[p95 > -9000]

        unique_count = len(np.unique(valid_p95))
        assert unique_count > 1, f"Dask P95 has only {unique_count} unique values!"

    def test_dask_returns_only_p95(self, dask_landsat_data):
        """Test that Dask composite only has P95 (no P50)."""
        result = compute_annual_composite(dask_landsat_data)

        assert "lst_p95" in result.data_vars
        assert "qa_count" in result.data_vars
        assert "lst_p50" not in result.data_vars
