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


class TestMaskOrientation:
    """Tests to prevent mask coordinate alignment bugs."""

    @pytest.fixture(scope="class")
    def land_polygons(self):
        """Load Natural Earth land polygons."""
        return load_land_polygons()

    def test_odc_stac_latitude_is_descending(self):
        """Verify odc-stac loads data with north-to-south (descending) latitude.

        This is critical for mask alignment - if this changes, the land mask
        will be inverted (Manhattan becomes water, bays become land).
        """
        import planetary_computer as pc
        import pystac_client
        from odc.stac import stac_load

        from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings

        # Small bbox for fast test
        bbox = (-74.5, 40.0, -74.0, 40.5)

        catalog = pystac_client.Client.open(STAC_PLANETARY_COMPUTER, modifier=pc.sign_inplace)
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=bbox,
            datetime="2024-06-01/2024-06-30",
            query={"eo:cloud_cover": {"lt": 20}},
        )
        items = list(search.items())[:1]

        if not items:
            pytest.skip("No STAC items found for test")

        data = stac_load(
            items, bands=["lwir11"], crs="EPSG:4326", resolution=settings.resolution, bbox=bbox
        )

        lat = data.latitude.values
        assert lat[0] > lat[-1], (
            f"Expected descending latitude (north-to-south), "
            f"but got lat[0]={lat[0]:.4f}, lat[-1]={lat[-1]:.4f}. "
            f"If this changed, the land mask flip logic needs updating!"
        )

    def test_manhattan_is_land_raritan_bay_is_water(self, land_polygons):
        """Verify mask correctly identifies known land/water points.

        This catches mask inversion bugs like the flip error.
        """
        import rasterio.transform

        from landsat_lst.masks import get_land_mask_for_bbox

        bbox = (-75.0, 39.5, -73.5, 41.0)
        mask = get_land_mask_for_bbox(bbox, settings.resolution, land_polygons)

        west, south, east, north = bbox
        width = int((east - west) / settings.resolution)
        height = int((north - south) / settings.resolution)
        transform = rasterio.transform.from_bounds(west, south, east, north, width, height)

        # Manhattan - definitely land
        row, col = rasterio.transform.rowcol(transform, -74.0, 40.75)
        assert mask[row, col], "Manhattan should be land (True)"

        # Raritan Bay - definitely water
        row, col = rasterio.transform.rowcol(transform, -74.2, 40.45)
        assert not mask[row, col], "Raritan Bay should be water (False)"

        # Atlantic Ocean - definitely water
        row, col = rasterio.transform.rowcol(transform, -73.6, 40.5)
        assert not mask[row, col], "Atlantic Ocean should be water (False)"


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


@pytest.mark.slow
class TestDataQualityEndToEnd:
    """End-to-end data quality tests using real Landsat data.

    These tests verify the full pipeline produces reasonable output.
    Marked slow because they download real data from Planetary Computer.
    """

    @pytest.fixture(scope="class")
    def processed_coastal_tile(self):
        """Process a small coastal area for data QA tests."""
        import planetary_computer as pc
        import pystac_client
        import rasterio.features
        import xarray as xr
        from odc.stac import stac_load
        from shapely.geometry import box

        from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings
        from landsat_lst.masks import load_land_polygons
        from landsat_lst.qa import apply_qa_mask, convert_to_celsius

        # Small coastal bbox (Jersey Shore)
        bbox = (-74.5, 40.0, -74.0, 40.5)

        catalog = pystac_client.Client.open(STAC_PLANETARY_COMPUTER, modifier=pc.sign_inplace)
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=bbox,
            datetime="2024-06-01/2024-08-31",  # Summer
            query={
                "eo:cloud_cover": {"lt": 20},
                "platform": {"in": ["landsat-8", "landsat-9"]},
            },
        )
        items = list(search.items())[:10]

        if not items:
            pytest.skip("No STAC items found")

        data = stac_load(
            items,
            bands=["lwir11", "qa_pixel"],
            crs="EPSG:4326",
            resolution=settings.resolution,
            chunks={"time": 5, "latitude": 512, "longitude": 512},
            groupby="solar_day",
            bbox=bbox,
        )

        masked = apply_qa_mask(data)
        lst = convert_to_celsius(masked["lwir11"])
        lst_p95 = lst.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")

        # Apply land mask
        land_polygons = load_land_polygons()
        lat, lon = lst_p95.latitude.values, lst_p95.longitude.values
        height, width = len(lat), len(lon)
        west, south, east, north = bbox
        transform = rasterio.transform.from_bounds(west, south, east, north, width, height)
        clipped = land_polygons.clip(box(west, south, east, north))
        land_mask = rasterio.features.rasterize(
            clipped.geometry.values,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8,
        ).astype(bool)

        lst_p95_masked = lst_p95.where(
            xr.DataArray(
                land_mask,
                dims=["latitude", "longitude"],
                coords={"latitude": lat, "longitude": lon},
            )
        )

        return lst_p95_masked.compute(), bbox

    def test_no_unrealistic_cold_temperatures(self, processed_coastal_tile):
        """LST should never show -124°C (unmasked fill value)."""
        result, _ = processed_coastal_tile
        valid = result.values[~np.isnan(result.values)]

        assert valid.min() > -50, (
            f"Found unrealistically cold temperature: {valid.min():.1f}°C. "
            "This likely indicates unmasked fill values (DN=0 → -124°C)."
        )

    def test_no_unrealistic_hot_temperatures(self, processed_coastal_tile):
        """LST P95 should not exceed physically plausible values."""
        result, _ = processed_coastal_tile
        valid = result.values[~np.isnan(result.values)]

        # Extreme urban surfaces can reach 60-70°C, but >80°C is suspicious
        assert valid.max() < 80, (
            f"Found unrealistically hot temperature: {valid.max():.1f}°C. "
            "This may indicate data corruption or processing error."
        )

    def test_mean_temperature_is_plausible(self, processed_coastal_tile):
        """Mean summer LST should be reasonable for Mid-Atlantic US."""
        result, _ = processed_coastal_tile
        valid = result.values[~np.isnan(result.values)]
        mean_temp = valid.mean()

        # Summer P95 in NJ should be roughly 20-45°C
        assert 15 < mean_temp < 50, (
            f"Mean temperature {mean_temp:.1f}°C is outside expected range "
            "for summer in Mid-Atlantic US (expected 15-50°C)."
        )

    def test_has_both_land_and_water_pixels(self, processed_coastal_tile):
        """Coastal tile should have both valid (land) and NaN (water) pixels."""
        result, _ = processed_coastal_tile

        nan_count = np.isnan(result.values).sum()
        valid_count = (~np.isnan(result.values)).sum()
        total = result.values.size

        assert valid_count > 0, "Should have some valid land pixels"
        assert nan_count > 0, "Coastal tile should have some water (NaN) pixels"

        # Sanity check ratios
        land_pct = 100 * valid_count / total
        assert 10 < land_pct < 95, (
            f"Land coverage {land_pct:.1f}% is unusual for coastal tile (expected 10-95%)"
        )

    def test_geographic_bounds_match_request(self, processed_coastal_tile):
        """Output coordinates should match requested bbox."""
        result, bbox = processed_coastal_tile
        west, south, east, north = bbox

        lat = result.latitude.values
        lon = result.longitude.values

        # Allow small tolerance for pixel alignment
        tol = 0.01  # ~1km

        assert lat.min() >= south - tol, f"South bound exceeded: {lat.min()} < {south}"
        assert lat.max() <= north + tol, f"North bound exceeded: {lat.max()} > {north}"
        assert lon.min() >= west - tol, f"West bound exceeded: {lon.min()} < {west}"
        assert lon.max() <= east + tol, f"East bound exceeded: {lon.max()} > {east}"
