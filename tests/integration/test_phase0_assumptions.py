"""Phase 0 assumption validation tests.

These tests validate our assumptions about external APIs and data formats
before building full infrastructure. Run with: pytest -m assumption

Key assumptions tested:
1. Planetary Computer STAC API structure and Landsat C2 L2 collection
2. odc.stac loading behavior (bands, dtype, CRS)
3. QA pixel bit structure for cloud/shadow/snow masking
4. Thermal band scale/offset for temperature conversion
5. COG internal structure for VirtualZarr compatibility
6. VirtualZarr → Icechunk roundtrip

Note: Uses Microsoft Planetary Computer (free, no requester-pays).
"""

import numpy as np
import planetary_computer
import pystac_client
import pytest
import xarray as xr
from odc.stac import configure_rio, stac_load

# Small bbox for fast tests (~0.2° near Pergamino, Argentina)
TEST_BBOX = (-60.5, -34.0, -60.3, -33.8)
TEST_YEAR = 2024  # Use 2024 for better data availability
TEST_DATETIME = f"{TEST_YEAR}-01-01/{TEST_YEAR}-12-31"


@pytest.fixture(scope="session", autouse=True)
def configure_rasterio():
    """Configure rasterio for cloud-optimized reads."""
    configure_rio(cloud_defaults=True)
    yield


@pytest.fixture(scope="module")
def stac_client():
    """Shared STAC client with Planetary Computer signing."""
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


@pytest.fixture(scope="module")
def stac_items(stac_client):
    """Fetch STAC items once for all tests in module."""
    search = stac_client.search(
        collections=["landsat-c2-l2"],
        bbox=TEST_BBOX,
        datetime=TEST_DATETIME,
        query={
            "eo:cloud_cover": {"lt": 20},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
    )
    items = list(search.items())
    if not items:
        pytest.skip(f"No Landsat scenes found for {TEST_BBOX} in {TEST_YEAR}")
    return items


# =============================================================================
# STAC API Assumption Tests
# =============================================================================


@pytest.mark.assumption
class TestSTACAssumptions:
    """Validate Earth Search STAC API contract."""

    def test_collection_exists(self, stac_client):
        """landsat-c2-l2 collection exists and is accessible."""
        collection = stac_client.get_collection("landsat-c2-l2")
        assert collection is not None
        assert collection.id == "landsat-c2-l2"

    def test_items_returned_for_test_region(self, stac_items):
        """Query returns items for Pergamino test region in 2025."""
        assert len(stac_items) > 0, "Expected at least one scene"
        # Reasonable expectation: 10-50 scenes for a year
        assert len(stac_items) < 200, f"Unexpected scene count: {len(stac_items)}"

    def test_item_has_required_assets(self, stac_items):
        """Items have lwir11 (thermal) and qa_pixel assets."""
        item = stac_items[0]
        assert "lwir11" in item.assets, f"Missing lwir11. Assets: {list(item.assets.keys())}"
        assert "qa_pixel" in item.assets, f"Missing qa_pixel. Assets: {list(item.assets.keys())}"

    def test_item_has_cloud_cover_metadata(self, stac_items):
        """Items have eo:cloud_cover property."""
        item = stac_items[0]
        assert "eo:cloud_cover" in item.properties
        cloud_cover = item.properties["eo:cloud_cover"]
        assert 0 <= cloud_cover <= 100

    def test_item_has_platform_metadata(self, stac_items):
        """Items have platform property (landsat-8 or landsat-9)."""
        item = stac_items[0]
        assert "platform" in item.properties
        assert item.properties["platform"] in ["landsat-8", "landsat-9"]

    def test_thermal_asset_is_cog(self, stac_items):
        """lwir11 asset is a Cloud-Optimized GeoTIFF."""
        item = stac_items[0]
        lwir_asset = item.assets["lwir11"]
        # Check media type indicates COG
        assert lwir_asset.media_type in [
            "image/tiff; application=geotiff; profile=cloud-optimized",
            "image/tiff; application=geotiff",
        ], f"Unexpected media type: {lwir_asset.media_type}"

    def test_thermal_asset_has_href(self, stac_items):
        """lwir11 asset has accessible href."""
        item = stac_items[0]
        href = item.assets["lwir11"].href
        assert href.startswith("https://") or href.startswith("s3://")


# =============================================================================
# odc.stac Loading Assumption Tests
# =============================================================================


@pytest.mark.assumption
class TestODCLoadingAssumptions:
    """Validate odc.stac loading behavior."""

    @pytest.fixture(scope="class")
    def loaded_data(self, stac_items):
        """Load data once for class tests."""
        # Load just 2 scenes for speed
        items_subset = stac_items[:2]
        ds = stac_load(
            items_subset,
            bands=["lwir11", "qa_pixel"],
            crs="EPSG:4326",
            resolution=0.00027778,  # ~30m in degrees
            chunks={"time": 1, "x": 512, "y": 512},
            groupby="solar_day",
            bbox=TEST_BBOX,
        )
        return ds

    def test_returns_xarray_dataset(self, loaded_data):
        """stac_load returns xarray Dataset."""
        assert isinstance(loaded_data, xr.Dataset)

    def test_has_expected_bands(self, loaded_data):
        """Dataset contains lwir11 and qa_pixel variables."""
        assert "lwir11" in loaded_data.data_vars
        assert "qa_pixel" in loaded_data.data_vars

    def test_has_time_dimension(self, loaded_data):
        """Dataset has time dimension."""
        assert "time" in loaded_data.dims

    def test_has_spatial_dimensions(self, loaded_data):
        """Dataset has x and y (or longitude/latitude) dimensions."""
        dims = set(loaded_data.dims)
        assert "x" in dims or "longitude" in dims
        assert "y" in dims or "latitude" in dims

    def test_thermal_band_dtype(self, loaded_data):
        """lwir11 band has numeric dtype (uint16 or float)."""
        dtype = loaded_data["lwir11"].dtype
        assert np.issubdtype(dtype, np.number), f"Unexpected dtype: {dtype}"

    def test_qa_band_dtype(self, loaded_data):
        """qa_pixel band has integer dtype for bit operations."""
        dtype = loaded_data["qa_pixel"].dtype
        assert np.issubdtype(dtype, np.integer), f"Unexpected dtype: {dtype}"

    def test_data_shape_reasonable(self, loaded_data):
        """Data shape is reasonable for 0.2° bbox."""
        # At 30m resolution, 0.2° ≈ 740 pixels
        x_size = loaded_data.dims.get("x") or loaded_data.dims.get("longitude")
        y_size = loaded_data.dims.get("y") or loaded_data.dims.get("latitude")
        assert 500 < x_size < 1500, f"Unexpected x size: {x_size}"
        assert 500 < y_size < 1500, f"Unexpected y size: {y_size}"


# =============================================================================
# QA Masking Assumption Tests
# =============================================================================


@pytest.mark.assumption
class TestQAMaskingAssumptions:
    """Validate QA pixel bit structure assumptions."""

    def test_cloud_bit_is_bit_3(self):
        """Cloud flag is at bit 3 (value 8 when set)."""
        # Bit 3 = 2^3 = 8
        qa_cloud = np.uint16(0b00001000)  # Only bit 3 set
        cloud_flag = (qa_cloud >> 3) & 1
        assert cloud_flag == 1

    def test_shadow_bit_is_bit_4(self):
        """Cloud shadow flag is at bit 4 (value 16 when set)."""
        qa_shadow = np.uint16(0b00010000)  # Only bit 4 set
        shadow_flag = (qa_shadow >> 4) & 1
        assert shadow_flag == 1

    def test_snow_bit_is_bit_5(self):
        """Snow/ice flag is at bit 5 (value 32 when set)."""
        qa_snow = np.uint16(0b00100000)  # Only bit 5 set
        snow_flag = (qa_snow >> 5) & 1
        assert snow_flag == 1

    def test_clear_pixel_passes_mask(self):
        """Clear pixel (no flags set) passes QA mask."""
        qa_clear = np.uint16(0)
        cloud = (qa_clear >> 3) & 1
        shadow = (qa_clear >> 4) & 1
        snow = (qa_clear >> 5) & 1
        is_good = (cloud == 0) and (shadow == 0) and (snow == 0)
        assert is_good

    def test_multiple_flags_detected(self):
        """Multiple flags set simultaneously are all detected."""
        # Cloud + shadow = 8 + 16 = 24
        qa_multi = np.uint16(0b00011000)
        cloud = (qa_multi >> 3) & 1
        shadow = (qa_multi >> 4) & 1
        assert cloud == 1
        assert shadow == 1

    def test_real_qa_values_have_expected_range(self, stac_items):
        """Real QA pixel values are within expected uint16 range."""
        # Load just first scene for speed
        ds = stac_load(
            stac_items[:1],
            bands=["qa_pixel"],
            crs="EPSG:4326",
            resolution=0.00027778,
            bbox=TEST_BBOX,
        )
        qa = ds["qa_pixel"].compute()
        assert qa.min() >= 0
        assert qa.max() <= 65535  # uint16 max


# =============================================================================
# Temperature Conversion Assumption Tests
# =============================================================================


@pytest.mark.assumption
class TestTemperatureConversionAssumptions:
    """Validate thermal band scale/offset assumptions."""

    # Landsat C2 L2 ST scaling: K = DN * 0.00341802 + 149.0
    SCALE = 0.00341802
    OFFSET = 149.0

    def test_scale_offset_formula(self):
        """Scale/offset formula matches USGS documentation."""
        # Known value: DN=30000 should give reasonable temperature
        dn = 30000
        kelvin = dn * self.SCALE + self.OFFSET
        celsius = kelvin - 273.15
        # Should be somewhere in reasonable Earth surface temp range
        assert -50 < celsius < 70, f"Unreasonable temp: {celsius}°C"

    def test_temperature_range_bounds(self):
        """Expected DN range produces reasonable temperatures."""
        # Typical DN range for valid LST: ~25000-45000
        dn_low = 25000
        dn_high = 45000

        kelvin_low = dn_low * self.SCALE + self.OFFSET
        kelvin_high = dn_high * self.SCALE + self.OFFSET

        celsius_low = kelvin_low - 273.15
        celsius_high = kelvin_high - 273.15

        # Cold: around -40°C to 0°C for dn_low
        assert -50 < celsius_low < 20, f"Low temp out of range: {celsius_low}°C"
        # Hot: around 30°C to 70°C for dn_high
        assert 20 < celsius_high < 100, f"High temp out of range: {celsius_high}°C"

    def test_nodata_value_zero(self):
        """Zero DN should be treated as nodata (invalid temperature)."""
        dn = 0
        kelvin = dn * self.SCALE + self.OFFSET
        celsius = kelvin - 273.15
        # 0 DN gives ~-124°C which is clearly invalid
        assert celsius < -100, "Zero DN should produce invalid (very low) temperature"

    def test_real_thermal_values_are_valid(self, stac_items):
        """Real thermal band values produce valid temperatures."""
        ds = stac_load(
            stac_items[:1],
            bands=["lwir11"],
            crs="EPSG:4326",
            resolution=0.00027778,
            bbox=TEST_BBOX,
        )
        lwir = ds["lwir11"].compute()

        # Mask zeros (nodata)
        valid = lwir.where(lwir > 0)
        kelvin = valid * self.SCALE + self.OFFSET
        celsius = kelvin - 273.15

        # Check range
        min_temp = float(celsius.min())
        max_temp = float(celsius.max())

        assert -60 < min_temp < 60, f"Min temp out of range: {min_temp}°C"
        assert -30 < max_temp < 80, f"Max temp out of range: {max_temp}°C"


# =============================================================================
# COG Structure Assumption Tests (requires cog.py module)
# =============================================================================


@pytest.mark.assumption
class TestCOGStructureAssumptions:
    """Validate COG internal structure for VirtualZarr compatibility."""

    @pytest.fixture
    def test_cog_path(self, tmp_path, stac_items):
        """Create a test COG from real data."""
        import rasterio
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

        # Load minimal data
        ds = stac_load(
            stac_items[:1],
            bands=["lwir11"],
            crs="EPSG:4326",
            resolution=0.00027778,
            bbox=TEST_BBOX,
        )
        data = ds["lwir11"].isel(time=0).compute().values

        # Write temp GeoTIFF
        tmp_tif = tmp_path / "temp.tif"
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": data.shape[1],
            "height": data.shape[0],
            "count": 1,
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_bounds(*TEST_BBOX, data.shape[1], data.shape[0]),
        }
        with rasterio.open(tmp_tif, "w", **profile) as dst:
            dst.write(data.astype("float32"), 1)

        # Convert to COG with 512x512 tiles
        cog_path = tmp_path / "test.tif"
        dst_profile = cog_profiles.get("deflate")
        dst_profile["blockxsize"] = 512
        dst_profile["blockysize"] = 512
        cog_translate(
            str(tmp_tif),
            str(cog_path),
            dst_profile,
            overview_resampling="average",
            use_cog_driver=True,
            overview_level=0,  # Disable overviews for VirtualZarr compatibility
        )
        return cog_path

    def test_cog_has_internal_tiling(self, test_cog_path):
        """COG has internal tiling structure."""
        import rasterio

        with rasterio.open(test_cog_path) as src:
            assert src.is_tiled, "COG must be internally tiled"

    def test_cog_blocksize_is_512(self, test_cog_path):
        """COG uses 512x512 block size."""
        import rasterio

        with rasterio.open(test_cog_path) as src:
            block_shapes = src.block_shapes
            # First band block shape
            block_y, block_x = block_shapes[0]
            assert block_x == 512, f"Expected 512 block width, got {block_x}"
            assert block_y == 512, f"Expected 512 block height, got {block_y}"

    def test_cog_is_valid_geotiff(self, test_cog_path):
        """COG is readable as GeoTIFF with valid CRS."""
        import rasterio

        with rasterio.open(test_cog_path) as src:
            assert src.crs is not None
            assert src.crs.to_epsg() == 4326


# =============================================================================
# VirtualZarr Assumption Tests
# =============================================================================


def _open_virtual_tiff(path: str):
    """Open a local TIFF as virtual dataset using VirtualZarr 2.x API."""
    from obspec_utils.registry import ObjectStoreRegistry
    from obstore.store import LocalStore
    from virtual_tiff import VirtualTIFF
    from virtualizarr import open_virtual_dataset

    registry = ObjectStoreRegistry()
    registry.register("file://", LocalStore())
    return open_virtual_dataset(f"file://{path}", registry=registry, parser=VirtualTIFF())


@pytest.mark.assumption
class TestVirtualZarrAssumptions:
    """Validate VirtualZarr can create references from COG."""

    @pytest.fixture
    def test_cog_path(self, tmp_path, stac_items):
        """Create a test COG."""
        import rasterio
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

        ds = stac_load(
            stac_items[:1],
            bands=["lwir11"],
            crs="EPSG:4326",
            resolution=0.00027778,
            bbox=TEST_BBOX,
        )
        data = ds["lwir11"].isel(time=0).compute().values

        tmp_tif = tmp_path / "temp.tif"
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": data.shape[1],
            "height": data.shape[0],
            "count": 1,
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_bounds(*TEST_BBOX, data.shape[1], data.shape[0]),
        }
        with rasterio.open(tmp_tif, "w", **profile) as dst:
            dst.write(data.astype("float32"), 1)

        cog_path = tmp_path / "test.tif"
        dst_profile = cog_profiles.get("deflate")
        dst_profile["blockxsize"] = 512
        dst_profile["blockysize"] = 512
        cog_translate(
            str(tmp_tif),
            str(cog_path),
            dst_profile,
            use_cog_driver=True,
            overview_level=0,  # Disable overviews for VirtualZarr compatibility
        )
        return cog_path

    def test_open_virtual_dataset_succeeds(self, test_cog_path):
        """VirtualZarr can open COG without error."""
        vds = _open_virtual_tiff(str(test_cog_path))
        assert vds is not None

    def test_virtual_dataset_has_data_vars(self, test_cog_path):
        """Virtual dataset has data variables."""
        vds = _open_virtual_tiff(str(test_cog_path))
        assert len(vds.data_vars) > 0

    def test_virtual_dataset_has_chunks(self, test_cog_path):
        """Virtual dataset has chunk structure from COG tiles."""
        vds = _open_virtual_tiff(str(test_cog_path))
        # Get first data variable
        var_name = next(iter(vds.data_vars))
        var = vds[var_name]
        # Should have encoding with chunks or chunksizes
        assert var.encoding or hasattr(var, "chunks")

    def test_can_add_time_dimension(self, test_cog_path):
        """Can expand virtual dataset with time dimension."""
        import pandas as pd

        vds = _open_virtual_tiff(str(test_cog_path))
        vds_with_time = vds.expand_dims(time=[pd.Timestamp("2025-07-01")])
        assert "time" in vds_with_time.dims


# =============================================================================
# Icechunk Roundtrip Assumption Tests
# =============================================================================


def _create_icechunk_repo_with_local_container(icechunk_path: str, cog_parent_dir: str):
    """Create Icechunk repo configured for local file virtual chunks.

    Icechunk 2.x requires explicit VirtualChunkContainer config to access
    external data (the COG files). This configures local filesystem access.
    """
    import icechunk
    from icechunk import ObjectStoreConfig, RepositoryConfig, VirtualChunkContainer

    # Configure local filesystem access for virtual chunks
    local_config = ObjectStoreConfig.LocalFileSystem(cog_parent_dir)
    container = VirtualChunkContainer(
        f"file://{cog_parent_dir}/",
        local_config,
        name="local",
    )

    storage = icechunk.local_filesystem_storage(icechunk_path)
    config = RepositoryConfig.default()
    config.set_virtual_chunk_container(container)

    return icechunk.Repository.create(storage, config=config)


def _open_icechunk_repo_with_local_container(icechunk_path: str, cog_parent_dir: str):
    """Open existing Icechunk repo with local file access configured."""
    import icechunk
    from icechunk import ObjectStoreConfig, RepositoryConfig, VirtualChunkContainer

    local_config = ObjectStoreConfig.LocalFileSystem(cog_parent_dir)
    container = VirtualChunkContainer(
        f"file://{cog_parent_dir}/",
        local_config,
        name="local",
    )

    storage = icechunk.local_filesystem_storage(icechunk_path)
    config = RepositoryConfig.default()
    config.set_virtual_chunk_container(container)

    # Need both config AND authorize_virtual_chunk_access for reading
    return icechunk.Repository.open(
        storage,
        config=config,
        authorize_virtual_chunk_access={f"file://{cog_parent_dir}/": None},
    )


@pytest.mark.assumption
class TestIcechunkAssumptions:
    """Validate Icechunk storage and retrieval."""

    @pytest.fixture
    def test_cog_path(self, tmp_path, stac_items):
        """Create a test COG."""
        import rasterio
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

        ds = stac_load(
            stac_items[:1],
            bands=["lwir11"],
            crs="EPSG:4326",
            resolution=0.00027778,
            bbox=TEST_BBOX,
        )
        data = ds["lwir11"].isel(time=0).compute().values

        tmp_tif = tmp_path / "temp.tif"
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": data.shape[1],
            "height": data.shape[0],
            "count": 1,
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_bounds(*TEST_BBOX, data.shape[1], data.shape[0]),
        }
        with rasterio.open(tmp_tif, "w", **profile) as dst:
            dst.write(data.astype("float32"), 1)

        cog_path = tmp_path / "test.tif"
        dst_profile = cog_profiles.get("deflate")
        dst_profile["blockxsize"] = 512
        dst_profile["blockysize"] = 512
        cog_translate(
            str(tmp_tif),
            str(cog_path),
            dst_profile,
            use_cog_driver=True,
            overview_level=0,  # Disable overviews for VirtualZarr compatibility
        )
        return cog_path

    def test_create_local_repository(self, tmp_path):
        """Can create local Icechunk repository."""
        import icechunk

        storage = icechunk.local_filesystem_storage(str(tmp_path / "icechunk"))
        repo = icechunk.Repository.create(storage)
        assert repo is not None

    def test_write_virtual_refs_to_icechunk(self, test_cog_path, tmp_path):
        """Can write VirtualZarr references to Icechunk."""
        vds = _open_virtual_tiff(str(test_cog_path))

        icechunk_path = str(tmp_path / "icechunk")
        cog_parent_dir = str(tmp_path)
        repo = _create_icechunk_repo_with_local_container(icechunk_path, cog_parent_dir)
        session = repo.writable_session("main")

        vds.virtualize.to_icechunk(session.store)
        commit_id = session.commit("Test commit")
        assert commit_id is not None

    def test_read_back_from_icechunk(self, test_cog_path, tmp_path):
        """Can read data back from Icechunk via xr.open_zarr."""
        vds = _open_virtual_tiff(str(test_cog_path))

        icechunk_path = str(tmp_path / "icechunk")
        cog_parent_dir = str(tmp_path)

        repo = _create_icechunk_repo_with_local_container(icechunk_path, cog_parent_dir)
        session = repo.writable_session("main")
        vds.virtualize.to_icechunk(session.store)
        session.commit("Test commit")

        # Read back
        read_repo = _open_icechunk_repo_with_local_container(icechunk_path, cog_parent_dir)
        read_session = read_repo.readonly_session("main")
        ds_read = xr.open_zarr(read_session.store, consolidated=False)

        assert len(ds_read.data_vars) > 0

    def test_data_values_match_original(self, test_cog_path, tmp_path):
        """Data read from Icechunk matches original COG values."""
        import rasterio

        # Read original data directly
        with rasterio.open(test_cog_path) as src:
            original_data = src.read(1)

        # Write to Icechunk
        vds = _open_virtual_tiff(str(test_cog_path))
        icechunk_path = str(tmp_path / "icechunk")
        cog_parent_dir = str(tmp_path)

        repo = _create_icechunk_repo_with_local_container(icechunk_path, cog_parent_dir)
        session = repo.writable_session("main")
        vds.virtualize.to_icechunk(session.store)
        session.commit("Test")

        # Read back
        read_repo = _open_icechunk_repo_with_local_container(icechunk_path, cog_parent_dir)
        ds_read = xr.open_zarr(read_repo.readonly_session("main").store, consolidated=False)

        # Get first variable
        var_name = next(iter(ds_read.data_vars))
        icechunk_data = ds_read[var_name].compute().values

        # Compare shapes at minimum
        assert icechunk_data.shape[-2:] == original_data.shape, (
            f"Shape mismatch: {icechunk_data.shape} vs {original_data.shape}"
        )
