"""Unit tests for QA filtering functions."""

import numpy as np
import xarray as xr

from landsat_lst.qa import convert_to_celsius, create_qa_mask


class TestCreateQaMask:
    def test_clear_pixels_are_valid(self, mock_qa_pixel: xr.DataArray):
        mask = create_qa_mask(mock_qa_pixel)
        assert mask[5:, :].all()

    def test_cloud_pixels_are_masked(self, mock_qa_pixel: xr.DataArray):
        mask = create_qa_mask(mock_qa_pixel)
        assert not mask[0:2, :].any()

    def test_shadow_pixels_are_masked(self, mock_qa_pixel: xr.DataArray):
        mask = create_qa_mask(mock_qa_pixel)
        assert not mask[2:4, :].any()

    def test_snow_pixels_are_masked(self, mock_qa_pixel: xr.DataArray):
        mask = create_qa_mask(mock_qa_pixel)
        assert not mask[4:5, :].any()

    def test_combined_flags_are_masked(self):
        data = np.array([[8 | 16]], dtype=np.uint16)
        qa = xr.DataArray(data, dims=["y", "x"])
        mask = create_qa_mask(qa)
        assert not mask[0, 0]

    def test_dilated_cloud_pixels_are_masked(self):
        """Bit 1 (dilated cloud, near-cloud edge) must be masked."""
        qa = xr.DataArray(np.array([[1 << 1]], dtype=np.uint16), dims=["y", "x"])
        assert not create_qa_mask(qa)[0, 0]

    def test_cirrus_pixels_are_masked(self):
        """Bit 2 (cirrus / thin cloud) must be masked."""
        qa = xr.DataArray(np.array([[1 << 2]], dtype=np.uint16), dims=["y", "x"])
        assert not create_qa_mask(qa)[0, 0]

    def test_clear_bit6_only_is_valid(self):
        """A pixel with only the 'clear' bit (6) set stays usable."""
        qa = xr.DataArray(np.array([[1 << 6]], dtype=np.uint16), dims=["y", "x"])
        assert create_qa_mask(qa)[0, 0]


class TestConvertToCelsius:
    def test_known_conversion(self):
        data = np.array([[40000.0]], dtype=np.float32)
        lwir = xr.DataArray(data, dims=["y", "x"])

        celsius = convert_to_celsius(lwir)

        expected = 40000.0 * 0.00341802 + 149.0 - 273.15
        np.testing.assert_almost_equal(celsius.values[0, 0], expected, decimal=2)

    def test_output_is_reasonable(self, mock_lwir_band: xr.DataArray):
        celsius = convert_to_celsius(mock_lwir_band)
        assert (celsius > -50).all()
        assert (celsius < 80).all()

    def test_preserves_nan(self):
        data = np.array([[np.nan, 40000.0]], dtype=np.float32)
        lwir = xr.DataArray(data, dims=["y", "x"])

        celsius = convert_to_celsius(lwir)

        assert np.isnan(celsius.values[0, 0])
        assert not np.isnan(celsius.values[0, 1])

    def test_zero_fill_value_becomes_nan(self):
        """Zero is Landsat fill/nodata value - should become NaN, not -124°C."""
        data = np.array([[0, 40000.0]], dtype=np.float32)
        lwir = xr.DataArray(data, dims=["y", "x"])

        celsius = convert_to_celsius(lwir)

        assert np.isnan(celsius.values[0, 0]), "Fill value 0 should become NaN"
        assert not np.isnan(celsius.values[0, 1]), "Valid DN should convert normally"

    def test_implausible_values_clamped_to_nan(self):
        """Resampling/saturation junk outside the physical range becomes NaN.

        DN=100 -> ~-123.8°C (below min); DN=65535 -> ~99.9°C (above max);
        DN=40000 -> ~12.6°C stays valid.
        """
        data = np.array([[100.0, 65535.0, 40000.0]], dtype=np.float32)
        lwir = xr.DataArray(data, dims=["y", "x"])

        celsius = convert_to_celsius(lwir)

        assert np.isnan(celsius.values[0, 0]), "~-124°C junk should be dropped"
        assert np.isnan(celsius.values[0, 1]), "~100°C saturation junk should be dropped"
        assert not np.isnan(celsius.values[0, 2]), "In-range value should survive"
