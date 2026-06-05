"""Integration tests for land mask and data quality.

Verifies that ocean pixels are correctly masked to NaN using Natural Earth
10m land polygons, and that LST output values are in valid ranges.
See issues #26, #30.
"""

import numpy as np
import pytest

from landsat_lst.config import settings
from landsat_lst.masks import get_land_mask_for_bbox, load_land_polygons
from landsat_lst.qa import convert_to_celsius


class TestLandMask:
    """Tests for land mask generation and application."""

    @pytest.fixture(scope="class")
    def land_polygons(self):
        """Load Natural Earth land polygons once for all tests."""
        return load_land_polygons()

    def test_ocean_pixels_masked_jersey_shore(self, land_polygons):
        """N40W075 tile: Atlantic Ocean east of Jersey Shore should be masked.

        The Jersey Shore coastline runs roughly along -74.4°W at 39°N.
        Points east of that (toward -74.0°W) are ocean and should be False.
        Points west (toward -75.0°W) are land and should be True.
        """
        # N40W075 bbox: west=-75, south=40, east=-70, north=45
        bbox = (-75.0, 40.0, -70.0, 45.0)

        mask = get_land_mask_for_bbox(bbox, settings.resolution, land_polygons)

        # Mask shape should match expected tile dimensions
        expected_width = int((bbox[2] - bbox[0]) / settings.resolution)
        expected_height = int((bbox[3] - bbox[1]) / settings.resolution)
        assert mask.shape == (expected_height, expected_width)

        # Convert lat/lon to pixel indices
        # Note: rasterio convention is north-down, so row 0 is north
        def latlon_to_pixel(lat, lon):
            col = int((lon - bbox[0]) / settings.resolution)
            row = int((bbox[3] - lat) / settings.resolution)  # north-down
            return row, col

        # Test point in Atlantic Ocean (should be False/water)
        # At 40.5°N, the NJ coast is ~-74°W, so -73.5°W is ocean
        ocean_lat, ocean_lon = 40.5, -73.5
        ocean_row, ocean_col = latlon_to_pixel(ocean_lat, ocean_lon)
        assert not mask[ocean_row, ocean_col], (
            f"Ocean point ({ocean_lat}, {ocean_lon}) should be masked (False)"
        )

        # Test point on land - central New Jersey (should be True/land)
        land_lat, land_lon = 40.5, -74.5
        land_row, land_col = latlon_to_pixel(land_lat, land_lon)
        assert mask[land_row, land_col], (
            f"Land point ({land_lat}, {land_lon}) should not be masked (True)"
        )

    def test_land_mask_dtype_is_bool(self, land_polygons):
        """Land mask should be boolean array."""
        bbox = (-75.0, 40.0, -70.0, 45.0)
        mask = get_land_mask_for_bbox(bbox, settings.resolution, land_polygons)
        assert mask.dtype == bool

    def test_coastal_tile_has_mixed_mask(self, land_polygons):
        """Coastal tiles should have both True and False values."""
        # N40W075 is a coastal tile
        bbox = (-75.0, 40.0, -70.0, 45.0)
        mask = get_land_mask_for_bbox(bbox, settings.resolution, land_polygons)

        land_pixels = np.sum(mask)
        water_pixels = np.sum(~mask)

        assert land_pixels > 0, "Coastal tile should have land pixels"
        assert water_pixels > 0, "Coastal tile should have water pixels"


class TestLSTValueRange:
    """Tests for LST output value validation. See issue #30."""

    def test_zero_fill_not_converted_to_temperature(self):
        """LWIR fill value (0) should become NaN, not -124°C.

        This catches the bug where unmasked fill values converted to
        unrealistic temperatures via the scale/offset formula.
        """
        import xarray as xr

        # Simulate raw LWIR data with fill values
        data = np.array([[0, 0, 45000], [45000, 0, 45000]], dtype=np.float32)
        lwir = xr.DataArray(data, dims=["y", "x"])

        celsius = convert_to_celsius(lwir)

        # No values should be below -50°C (physical impossibility for LST)
        valid = celsius.values[~np.isnan(celsius.values)]
        assert len(valid) > 0, "Should have some valid values"
        assert valid.min() > -50, (
            f"Min temp {valid.min():.1f}°C is unrealistic - likely unmasked fill value"
        )

    def test_valid_dn_produces_reasonable_temperature(self):
        """Valid Landsat DN values should produce Earth-like temperatures."""
        import xarray as xr

        # DN range for typical LST: ~30000-55000
        # 45000 DN → ~20°C (reasonable surface temp)
        data = np.array([[45000]], dtype=np.float32)
        lwir = xr.DataArray(data, dims=["y", "x"])

        celsius = convert_to_celsius(lwir)

        temp = float(celsius.values[0, 0])
        assert -30 < temp < 70, f"Temperature {temp}°C outside valid range"
