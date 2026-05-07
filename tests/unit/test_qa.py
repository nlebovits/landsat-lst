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
