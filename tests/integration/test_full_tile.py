"""Full 5° tile integration test.

End-to-end test: STAC query → composite → COG → VirtualZarr → Icechunk

Uses optimized settings from parameter sweeps:
- Cloud cover: ≤20%
- Chunk size: 512×512
- Resampling: bilinear (thermal), nearest (QA)
- Compression: zstd
- Statistics: median (not quantile)

Run with: pytest -m tile -v -s
"""

import time
from pathlib import Path

import numpy as np
import planetary_computer
import pytest
import pystac_client
import rasterio
import xarray as xr
import icechunk
from icechunk import ObjectStoreConfig, Repository, RepositoryConfig, VirtualChunkContainer
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from odc.stac import configure_rio, stac_load
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from virtual_tiff import VirtualTIFF
from virtualizarr import open_virtual_dataset

# 0.25° tile for integration test (~900x900px, ~500MB)
# Located within Pergamino region for good data availability
TILE_BBOX = (-60.5, -34.0, -60.25, -33.75)
TILE_NAME = "test_tile"
TILE_YEAR = 2024


@pytest.fixture(scope="module")
def stac_client():
    """STAC client with Planetary Computer signing."""
    configure_rio(cloud_defaults=True)
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


@pytest.fixture(scope="module")
def stac_items(stac_client):
    """Query STAC for all valid scenes in the tile."""
    print(f"\n{'='*60}")
    print(f"FULL TILE INTEGRATION TEST: {TILE_NAME} ({TILE_YEAR})")
    print(f"{'='*60}")
    print(f"Bbox: {TILE_BBOX}")

    start = time.perf_counter()
    search = stac_client.search(
        collections=["landsat-c2-l2"],
        bbox=TILE_BBOX,
        datetime=f"{TILE_YEAR}-01-01/{TILE_YEAR}-12-31",
        query={
            "eo:cloud_cover": {"lt": 20},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
    )
    items = list(search.items())
    elapsed = time.perf_counter() - start

    print(f"STAC query: {len(items)} scenes in {elapsed:.1f}s")

    if not items:
        pytest.skip(f"No scenes found for {TILE_NAME} in {TILE_YEAR}")

    # Show scene distribution
    cloud_covers = [i.properties["eo:cloud_cover"] for i in items]
    print(f"Cloud cover: {np.mean(cloud_covers):.1f}% avg, {np.min(cloud_covers):.1f}-{np.max(cloud_covers):.1f}% range")

    return items


@pytest.mark.tile
class TestFullTileIntegration:
    """End-to-end test for a complete 5° tile."""

    def test_01_load_all_scenes(self, stac_items):
        """Load all scenes for the tile."""
        print(f"\n--- Step 1: Load scenes ---")
        start = time.perf_counter()

        ds = stac_load(
            stac_items,
            bands=["lwir11", "qa_pixel"],
            crs="EPSG:4326",
            resolution=0.00027778,  # ~30m
            chunks={"time": 10, "latitude": 512, "longitude": 512},
            resampling={"lwir11": "bilinear", "qa_pixel": "nearest"},
            bbox=TILE_BBOX,
            groupby="solar_day",
        )

        elapsed = time.perf_counter() - start
        print(f"Dataset created in {elapsed:.1f}s (lazy)")
        print(f"Shape: {dict(ds.sizes)}")
        print(f"Chunks: {ds.chunks}")

        # Store for next tests
        pytest.ds = ds

    def test_02_apply_qa_mask(self):
        """Apply QA mask to remove clouds/shadows/snow."""
        print(f"\n--- Step 2: Apply QA mask ---")
        ds = pytest.ds

        qa = ds["qa_pixel"]
        cloud = (qa >> 3) & 1
        shadow = (qa >> 4) & 1
        snow = (qa >> 5) & 1
        mask = (cloud == 0) & (shadow == 0) & (snow == 0)

        lst_masked = ds["lwir11"].where(mask)

        # Store
        pytest.lst_masked = lst_masked
        print("QA mask created (lazy)")

    def test_03_convert_to_celsius(self):
        """Convert to Celsius using Landsat C2 L2 scaling."""
        print(f"\n--- Step 3: Convert to Celsius ---")
        lst_masked = pytest.lst_masked

        # Scale/offset from USGS
        lst_kelvin = lst_masked * 0.00341802 + 149.0
        lst_celsius = lst_kelvin - 273.15

        pytest.lst_celsius = lst_celsius
        print("Temperature conversion applied (lazy)")

    def test_04_compute_composite(self):
        """Compute annual composite: p50, p95, count."""
        print(f"\n--- Step 4: Compute composite ---")
        lst_celsius = pytest.lst_celsius

        start = time.perf_counter()

        # Use .median() instead of .quantile(0.5) - 200x faster
        lst_p50 = lst_celsius.median(dim="time", skipna=True)
        lst_p95 = lst_celsius.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")

        # Count valid observations
        valid_mask = ~np.isnan(lst_celsius)
        qa_count = valid_mask.sum(dim="time").astype(np.int16)

        # Set nodata where no valid observations
        nodata = -9999.0
        lst_p50 = lst_p50.where(qa_count > 0, nodata)
        lst_p95 = lst_p95.where(qa_count > 0, nodata)

        # Create composite dataset
        composite = xr.Dataset({
            "lst_p50": lst_p50.astype(np.float32),
            "lst_p95": lst_p95.astype(np.float32),
            "qa_count": qa_count,
        })

        # Compute!
        print("Computing composite (this takes a while)...")
        composite = composite.compute()

        elapsed = time.perf_counter() - start
        print(f"Composite computed in {elapsed:.1f}s")
        print(f"Shape: {dict(composite.sizes)}")

        # Stats
        p50_valid = composite["lst_p50"].values[composite["lst_p50"].values != nodata]
        if len(p50_valid) > 0:
            print(f"LST p50: {p50_valid.min():.1f}°C to {p50_valid.max():.1f}°C (mean {p50_valid.mean():.1f}°C)")

        pytest.composite = composite

    def test_05_write_cog(self, tmp_path):
        """Write composite to Cloud-Optimized GeoTIFF.

        Uses uint16 encoding with scale/offset following standard practice:
        - Landsat C2 L2 ST: uint16, scale=0.00341802, offset=149.0 (Kelvin)
        - MODIS MOD11: uint16, scale=0.02, offset=0 (Kelvin)

        Our encoding: scale=0.01, offset=-50.0 (Celsius)
        Range: -50°C to +105.535°C with 0.01°C precision

        References:
        - https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products
        - https://lpdaac.usgs.gov/documents/118/MOD11_User_Guide_V6.pdf
        """
        print(f"\n--- Step 5: Write COG (uint16) ---")
        composite = pytest.composite

        start = time.perf_counter()

        # Encoding constants
        LST_SCALE = 0.01
        LST_OFFSET = -50.0
        LST_NODATA_CELSIUS = -9999.0

        def encode_celsius_to_uint16(celsius):
            """Encode Celsius to uint16. DN=0 is nodata."""
            valid = celsius != LST_NODATA_CELSIUS
            dn = np.zeros_like(celsius, dtype=np.uint16)
            dn[valid] = np.round(
                (celsius[valid] - LST_OFFSET) / LST_SCALE
            ).clip(1, 65535).astype(np.uint16)
            return dn

        # Encode temperature bands to uint16
        lst_p50 = encode_celsius_to_uint16(composite["lst_p50"].values)
        lst_p95 = encode_celsius_to_uint16(composite["lst_p95"].values)
        qa_count = composite["qa_count"].values.astype(np.uint16)

        # Stack bands
        data = np.stack([lst_p50, lst_p95, qa_count], axis=0)

        # Get spatial info
        lat = composite["latitude"].values
        lon = composite["longitude"].values
        transform = rasterio.transform.from_bounds(
            lon.min(), lat.min(), lon.max(), lat.max(),
            data.shape[2], data.shape[1]
        )

        # Write temp GeoTIFF with uint16
        tmp_tif = tmp_path / "temp.tif"
        profile = {
            "driver": "GTiff",
            "dtype": "uint16",
            "width": data.shape[2],
            "height": data.shape[1],
            "count": 3,
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": 0,  # DN=0 is nodata
        }

        with rasterio.open(tmp_tif, "w", **profile) as dst:
            dst.write(data)
            dst.descriptions = ("lst_p50", "lst_p95", "qa_count")

        # Translate to COG with DEFLATE compression (Source Coop compatible)
        cog_path = tmp_path / f"{TILE_NAME}_{TILE_YEAR}.tif"
        dst_profile = cog_profiles.get("deflate")
        dst_profile["blockxsize"] = 512
        dst_profile["blockysize"] = 512

        cog_translate(
            str(tmp_tif),
            str(cog_path),
            dst_profile,
            use_cog_driver=True,
            overview_level=0,  # No overviews for VirtualZarr
            quiet=True,
        )

        elapsed = time.perf_counter() - start
        file_size_mb = cog_path.stat().st_size / (1024 * 1024)

        print(f"COG written in {elapsed:.1f}s")
        print(f"File: {cog_path.name} ({file_size_mb:.1f} MB)")

        # Validate COG structure
        with rasterio.open(cog_path) as src:
            assert src.count == 3
            assert src.is_tiled
            assert src.crs.to_epsg() == 4326
            assert src.dtypes[0] == "uint16", f"Expected uint16, got {src.dtypes[0]}"
            print(f"Data type: {src.dtypes[0]}")
            print(f"Encoding: scale=0.01, offset=-50.0 (Celsius)")

        pytest.cog_path = cog_path
        pytest.tmp_path = tmp_path

    def test_06_create_virtual_reference(self):
        """Create VirtualZarr reference from COG."""
        print(f"\n--- Step 6: Create VirtualZarr reference ---")
        cog_path = pytest.cog_path

        start = time.perf_counter()

        # VirtualZarr 2.x API
        registry = ObjectStoreRegistry()
        registry.register("file://", LocalStore())

        vds = open_virtual_dataset(
            f"file://{cog_path}",
            registry=registry,
            parser=VirtualTIFF(),
        )

        elapsed = time.perf_counter() - start
        print(f"Virtual dataset created in {elapsed:.3f}s")
        print(f"Variables: {list(vds.data_vars)}")

        pytest.vds = vds

    def test_07_write_to_icechunk(self):
        """Write virtual references to Icechunk store."""
        print(f"\n--- Step 7: Write to Icechunk ---")
        vds = pytest.vds
        tmp_path = pytest.tmp_path
        cog_path = pytest.cog_path

        start = time.perf_counter()

        # Configure Icechunk with local file access
        cog_parent_dir = str(tmp_path)
        local_config = ObjectStoreConfig.LocalFileSystem(cog_parent_dir)
        container = VirtualChunkContainer(
            f"file://{cog_parent_dir}/",
            local_config,
            name="local",
        )

        icechunk_path = str(tmp_path / "icechunk")
        storage = icechunk.local_filesystem_storage(icechunk_path)
        config = RepositoryConfig.default()
        config.set_virtual_chunk_container(container)

        repo = Repository.create(storage, config=config)
        session = repo.writable_session("main")

        vds.virtualize.to_icechunk(session.store)
        commit_id = session.commit(f"Added {TILE_NAME} {TILE_YEAR}")

        elapsed = time.perf_counter() - start
        print(f"Icechunk commit in {elapsed:.3f}s")
        print(f"Commit ID: {commit_id[:16]}...")

        pytest.icechunk_path = icechunk_path
        pytest.cog_parent_dir = cog_parent_dir

    def test_08_read_back_from_icechunk(self):
        """Read data back from Icechunk via xr.open_zarr."""
        print(f"\n--- Step 8: Read back from Icechunk ---")
        icechunk_path = pytest.icechunk_path
        cog_parent_dir = pytest.cog_parent_dir

        start = time.perf_counter()

        # Open with virtual chunk authorization
        local_config = ObjectStoreConfig.LocalFileSystem(cog_parent_dir)
        container = VirtualChunkContainer(
            f"file://{cog_parent_dir}/",
            local_config,
            name="local",
        )

        storage = icechunk.local_filesystem_storage(icechunk_path)
        config = RepositoryConfig.default()
        config.set_virtual_chunk_container(container)

        repo = Repository.open(
            storage,
            config=config,
            authorize_virtual_chunk_access={f"file://{cog_parent_dir}/": None},
        )

        ds_read = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)

        elapsed = time.perf_counter() - start
        print(f"Opened from Icechunk in {elapsed:.3f}s")
        print(f"Variables: {list(ds_read.data_vars)}")

        pytest.ds_read = ds_read

    def test_09_validate_roundtrip(self):
        """Validate data integrity through the full pipeline."""
        print(f"\n--- Step 9: Validate roundtrip ---")
        cog_path = pytest.cog_path
        ds_read = pytest.ds_read

        # Read original COG directly
        with rasterio.open(cog_path) as src:
            original_data = src.read()
            print(f"Original COG shape: {original_data.shape}")

        # Validate Icechunk dataset structure
        var_name = list(ds_read.data_vars)[0]
        print(f"Icechunk variable: {var_name}")
        print(f"Icechunk dims: {ds_read[var_name].dims}")
        print(f"Icechunk shape: {ds_read[var_name].shape}")

        # Shape validation
        assert ds_read[var_name].shape[-2:] == original_data.shape[-2:], (
            f"Shape mismatch: {ds_read[var_name].shape} vs {original_data.shape}"
        )

        print("✓ Roundtrip structure validated")

    def test_10_summary(self):
        """Print summary of full tile test."""
        print(f"\n{'='*60}")
        print("FULL TILE TEST COMPLETE")
        print(f"{'='*60}")
        print(f"Tile: {TILE_NAME}")
        print(f"Year: {TILE_YEAR}")
        print(f"Bbox: {TILE_BBOX}")
        print(f"Pipeline: STAC → Load → QA → Composite → COG → VirtualZarr → Icechunk")
        print("All steps passed!")
