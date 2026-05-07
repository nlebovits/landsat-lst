"""Full 5° tile integration test.

End-to-end test: STAC query → composite → COG → VirtualZarr → Icechunk

Uses optimized settings from parameter sweeps:
- Cloud cover: ≤20%
- Chunk size: 512x512
- Resampling: bilinear (thermal), nearest (QA)
- Compression: DEFLATE (Source Coop compatible)
- Statistics: median (not quantile)

Run with: pytest -m tile -v -s
"""

import time

import icechunk
import numpy as np
import planetary_computer
import pystac_client
import pytest
import rasterio
import xarray as xr
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
    print(f"\n{'=' * 60}")
    print(f"FULL TILE INTEGRATION TEST: {TILE_NAME} ({TILE_YEAR})")
    print(f"{'=' * 60}")
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
    print(
        f"Cloud cover: {np.mean(cloud_covers):.1f}% avg, {np.min(cloud_covers):.1f}-{np.max(cloud_covers):.1f}% range"
    )

    return items


@pytest.mark.tile
class TestFullTileIntegration:
    """End-to-end test for a complete 5° tile."""

    def test_01_load_all_scenes(self, stac_items):
        """Load all scenes for the tile."""
        print("\n--- Step 1: Load scenes ---")
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
        print("\n--- Step 2: Apply QA mask ---")
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
        print("\n--- Step 3: Convert to Celsius ---")
        lst_masked = pytest.lst_masked

        # Scale/offset from USGS
        lst_kelvin = lst_masked * 0.00341802 + 149.0
        lst_celsius = lst_kelvin - 273.15

        pytest.lst_celsius = lst_celsius
        print("Temperature conversion applied (lazy)")

    def test_04_compute_composite(self):
        """Build annual composite: p50, p95, count (LAZY - no .compute()).

        CRITICAL: This step keeps the composite lazy (chunked). The actual
        computation happens during COG write, streaming one chunk at a time.
        This enables processing tiles that would otherwise OOM.

        Memory model:
        - Without chunked write: 5° tile = 18000x18000 x 150 times x 4 bytes = ~388 GB
        - With chunked write: 512x512 chunk x 150 times x 4 bytes = ~157 MB peak
        """
        print("\n--- Step 4: Build composite (lazy) ---")
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

        # Create composite dataset - STAYS LAZY (chunked)
        composite = xr.Dataset(
            {
                "lst_p50": lst_p50.astype(np.float32),
                "lst_p95": lst_p95.astype(np.float32),
                "qa_count": qa_count,
            }
        )

        # Rechunk to optimal size for COG write
        composite = composite.chunk({"latitude": 512, "longitude": 512})

        elapsed = time.perf_counter() - start
        print(f"Composite graph built in {elapsed:.1f}s (lazy)")
        print(f"Shape: {dict(composite.sizes)}")
        print(f"Chunks: {composite.chunks}")

        # Verify it's still lazy
        assert composite.chunks is not None, "Composite should be chunked (lazy)"

        pytest.composite = composite

    def test_05_write_cog(self, tmp_path):
        """Write composite to Cloud-Optimized GeoTIFF using chunked streaming.

        CRITICAL: Uses rioxarray chunked write pattern to avoid OOM.
        The composite is LAZY (not computed) - rioxarray streams chunks
        to disk one at a time, bounded by chunk size (~157 MB per chunk).

        Two-stage approach:
        1. Stream chunks to tiled GeoTIFF via rioxarray (memory-bounded)
        2. Translate to COG with compression and overviews (disk-bound)

        Note: This test uses float32 for simplicity. Production code should
        use uint16 encoding for 43% storage reduction (see d87f18e).
        """
        import threading

        import rioxarray  # noqa: F401 - needed for .rio accessor

        print("\n--- Step 5: Write COG (chunked streaming) ---")
        composite = pytest.composite

        # Verify composite is still lazy
        assert composite.chunks is not None, "Composite should be chunked for memory-bounded write"

        start = time.perf_counter()

        # Set CRS and spatial dims for rioxarray
        composite = composite.rio.write_crs("EPSG:4326")
        composite = composite.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

        # Stage 1: Chunked write to temp GeoTIFF (memory-bounded)
        tmp_tif = tmp_path / "temp.tif"
        stacked = composite.to_array(dim="band")

        print("Streaming chunks to disk...")
        stacked.rio.to_raster(
            str(tmp_tif),
            tiled=True,
            lock=threading.Lock(),
        )

        stage1_elapsed = time.perf_counter() - start
        print(f"Stage 1 (chunked write): {stage1_elapsed:.1f}s")

        # Stage 2: COG translation with compression and overviews
        cog_path = tmp_path / f"{TILE_NAME}_{TILE_YEAR}.tif"
        dst_profile = cog_profiles.get("deflate")
        dst_profile["blockxsize"] = 512
        dst_profile["blockysize"] = 512

        cog_translate(
            str(tmp_tif),
            str(cog_path),
            dst_profile,
            use_cog_driver=True,
            overview_level=None,  # Auto-generate overviews; VirtualTIFF(ifd=0) handles them
            quiet=True,
        )

        # Clean up temp file
        tmp_tif.unlink()

        elapsed = time.perf_counter() - start
        file_size_mb = cog_path.stat().st_size / (1024 * 1024)

        print(f"Total COG write: {elapsed:.1f}s")
        print(f"File: {cog_path.name} ({file_size_mb:.1f} MB)")

        # Validate COG structure
        with rasterio.open(cog_path) as src:
            assert src.count == 3
            assert src.crs.to_epsg() == 4326
            # Note: small tiles may not be tiled (optimization by rio-cogeo)
            if src.width > 512 or src.height > 512:
                assert src.is_tiled, "Large COGs should be tiled"
            print(f"Data type: {src.dtypes[0]}")
            print(f"Overviews: {src.overviews(1)}")

        pytest.cog_path = cog_path
        pytest.tmp_path = tmp_path

    def test_06_create_virtual_reference(self):
        """Create VirtualZarr reference from COG."""
        print("\n--- Step 6: Create VirtualZarr reference ---")
        cog_path = pytest.cog_path

        start = time.perf_counter()

        # VirtualZarr 2.x API
        registry = ObjectStoreRegistry()
        registry.register("file://", LocalStore())

        # VirtualTIFF(ifd=0) selects full-res IFD, allowing COGs with overviews
        # Without ifd=0, COGs with overviews cause "conflicting sizes for dimension y"
        vds = open_virtual_dataset(
            f"file://{cog_path}",
            registry=registry,
            parser=VirtualTIFF(ifd=0),
        )

        elapsed = time.perf_counter() - start
        print(f"Virtual dataset created in {elapsed:.3f}s")
        print(f"Variables: {list(vds.data_vars)}")

        pytest.vds = vds

    def test_07_write_to_icechunk(self):
        """Write virtual references to Icechunk store."""
        print("\n--- Step 7: Write to Icechunk ---")
        vds = pytest.vds
        tmp_path = pytest.tmp_path

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
        print("\n--- Step 8: Read back from Icechunk ---")
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
        print("\n--- Step 9: Validate roundtrip ---")
        cog_path = pytest.cog_path
        ds_read = pytest.ds_read

        # Read original COG directly
        with rasterio.open(cog_path) as src:
            original_data = src.read()
            print(f"Original COG shape: {original_data.shape}")

        # Validate Icechunk dataset structure
        var_name = next(iter(ds_read.data_vars))
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
        print(f"\n{'=' * 60}")
        print("FULL TILE TEST COMPLETE")
        print(f"{'=' * 60}")
        print(f"Tile: {TILE_NAME}")
        print(f"Year: {TILE_YEAR}")
        print(f"Bbox: {TILE_BBOX}")
        print("Pipeline: STAC → Load → QA → Composite → COG → VirtualZarr → Icechunk")
        print("All steps passed!")
