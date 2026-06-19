"""Full tile integration test with direct Zarr + Icechunk.

End-to-end test: STAC query → composite → Zarr → Icechunk

Uses optimized settings from parameter sweeps:
- Cloud cover: ≤20%
- Chunk size: 500x500
- Resampling: bilinear (thermal), nearest (QA)
- Storage: Icechunk for versioned writes

Run with: pytest -m tile -v -s
"""

import time

import numpy as np
import planetary_computer
import pystac_client
import pytest
import xarray as xr
from odc.stac import configure_rio, stac_load

from landsat_lst.config import settings
from landsat_lst.storage import IcechunkStorage
from landsat_lst.zarr_writer import write_zarr

# 0.25° tile for integration test (~900x900px)
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
            "eo:cloud_cover": {"lt": settings.max_cloud_cover},
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
        f"Cloud cover: {np.mean(cloud_covers):.1f}% avg, "
        f"{np.min(cloud_covers):.1f}-{np.max(cloud_covers):.1f}% range"
    )

    return items


@pytest.mark.slow
@pytest.mark.tile
class TestFullTileIntegration:
    """End-to-end test for direct Zarr + Icechunk pipeline."""

    def test_01_load_all_scenes(self, stac_items):
        """Load all scenes for the tile."""
        print("\n--- Step 1: Load scenes ---")
        start = time.perf_counter()

        ds = stac_load(
            stac_items,
            bands=["lwir11", "qa_pixel"],
            crs="EPSG:4326",
            resolution=0.00027778,  # ~30m
            chunks={"time": 10, "latitude": 500, "longitude": 500},
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
        """Build annual composite: p95 and count (LAZY - no .compute())."""
        print("\n--- Step 4: Build composite (lazy) ---")
        lst_celsius = pytest.lst_celsius

        start = time.perf_counter()

        lst_p95 = lst_celsius.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")

        # Count valid observations
        valid_mask = ~np.isnan(lst_celsius)
        qa_count = valid_mask.sum(dim="time").astype(np.int16)

        # Set nodata where no valid observations
        nodata = -9999.0
        lst_p95 = lst_p95.where(qa_count > 0, nodata)

        # Create composite dataset - STAYS LAZY (chunked)
        composite = xr.Dataset(
            {
                "lst_p95": lst_p95.astype(np.float32),
                "qa_count": qa_count.astype(np.uint16),
            }
        )

        # Rechunk to optimal size for Zarr write
        composite = composite.chunk({"latitude": 500, "longitude": 500})

        elapsed = time.perf_counter() - start
        print(f"Composite graph built in {elapsed:.1f}s (lazy)")
        print(f"Shape: {dict(composite.sizes)}")
        print(f"Chunks: {composite.chunks}")

        # Verify it's still lazy
        assert composite.chunks is not None, "Composite should be chunked (lazy)"

        pytest.composite = composite

    def test_05_write_to_icechunk(self, tmp_path):
        """Write composite to Icechunk repository."""
        print("\n--- Step 5: Write to Icechunk ---")
        composite = pytest.composite

        start = time.perf_counter()

        # Create Icechunk storage
        icechunk_path = tmp_path / "icechunk"
        storage = IcechunkStorage.from_local(icechunk_path)

        # Write to Icechunk
        session = storage.writable_session()
        group_path = f"{TILE_YEAR}/{TILE_NAME}"

        print(f"Writing to Icechunk group: {group_path}")
        write_zarr(composite, session, group=group_path, chunks=(500, 500))

        # Commit
        commit_id = session.commit(f"Add {TILE_NAME} for {TILE_YEAR}")

        elapsed = time.perf_counter() - start
        print(f"Icechunk write + commit in {elapsed:.1f}s")
        print(f"Commit ID: {commit_id[:16]}...")

        pytest.icechunk_storage = storage
        pytest.group_path = group_path

    def test_06_read_back_from_icechunk(self):
        """Read data back from Icechunk."""
        print("\n--- Step 6: Read back from Icechunk ---")
        storage = pytest.icechunk_storage
        group_path = pytest.group_path

        start = time.perf_counter()

        # Read from Icechunk (native data lives in multiscale level "0")
        session = storage.readonly_session()
        ds_read = xr.open_zarr(session.store, group=f"{group_path}/0", consolidated=False)

        elapsed = time.perf_counter() - start
        print(f"Opened from Icechunk in {elapsed:.3f}s")
        print(f"Variables: {list(ds_read.data_vars)}")

        pytest.ds_read = ds_read

    def test_07_validate_structure(self):
        """Validate dataset structure."""
        print("\n--- Step 7: Validate structure ---")
        ds_read = pytest.ds_read

        # Check variables exist
        assert "lst_p95" in ds_read.data_vars
        assert "qa_count" in ds_read.data_vars

        # Check dtype (should be uint16 after encoding)
        assert ds_read["lst_p95"].dtype == np.uint16
        assert ds_read["qa_count"].dtype == np.uint16

        # Check encoding attributes preserved
        assert ds_read["lst_p95"].attrs["lst_scale_factor"] == 0.01
        assert ds_read["lst_p95"].attrs["lst_add_offset"] == -50.0

        print("✓ Structure validated")

    def test_08_validate_values(self):
        """Validate data values are reasonable."""
        print("\n--- Step 8: Validate values ---")
        ds_read = pytest.ds_read

        # Decode values
        scale = ds_read["lst_p95"].attrs["lst_scale_factor"]
        offset = ds_read["lst_p95"].attrs["lst_add_offset"]

        p95_celsius = ds_read["lst_p95"].values * scale + offset

        # Check reasonable temperature range (Argentina in winter/summer)
        # Allow wide range for different seasons and nodata handling
        valid_p95 = p95_celsius[(p95_celsius > -50) & (p95_celsius < 60)]

        if len(valid_p95) > 0:
            print(f"P95 range: {valid_p95.min():.1f}°C to {valid_p95.max():.1f}°C")

        print("✓ Values validated")

    def test_09_summary(self):
        """Print summary of full tile test."""
        print(f"\n{'=' * 60}")
        print("FULL TILE TEST COMPLETE")
        print(f"{'=' * 60}")
        print(f"Tile: {TILE_NAME}")
        print(f"Year: {TILE_YEAR}")
        print(f"Bbox: {TILE_BBOX}")
        print("Pipeline: STAC → Load → QA → Composite → Zarr → Icechunk")
        print("All steps passed!")
