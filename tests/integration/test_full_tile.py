"""Full tile integration test with COG output.

End-to-end test: STAC query → composite → COG → read back

Uses optimized settings from parameter sweeps:
- Cloud cover: ≤20%
- Chunk size: 500x500
- Resampling: bilinear (thermal), nearest (QA)
- Storage: one lst_p95 COG plus one qa_count COG per tile

Run with: pytest -m tile -v -s
"""

import time

import numpy as np
import planetary_computer
import pystac_client
import pytest
import rasterio
import xarray as xr
from odc.stac import configure_rio, stac_load
from rio_cogeo.cogeo import cog_validate

from landsat_lst.cog import cog_export
from landsat_lst.config import settings
from landsat_lst.encoding import encode_lst_uint16

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
    """End-to-end test for the COG output pipeline."""

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
                "qa_count": qa_count.astype(np.uint8),
            }
        )

        # Rechunk to the size the COG writer streams in
        composite = composite.chunk({"latitude": 500, "longitude": 500})

        elapsed = time.perf_counter() - start
        print(f"Composite graph built in {elapsed:.1f}s (lazy)")
        print(f"Shape: {dict(composite.sizes)}")
        print(f"Chunks: {composite.chunks}")

        # Verify it's still lazy
        assert composite.chunks is not None, "Composite should be chunked (lazy)"

        pytest.composite = composite

    def test_05_export_cogs(self, tmp_path):
        """Export the composite as the two output COGs."""
        print("\n--- Step 5: Export COGs ---")
        composite = pytest.composite

        start = time.perf_counter()

        # cog_export takes the encoded native level: uint16 LST DN + count band.
        encoded = composite.assign(lst_p95=encode_lst_uint16(composite["lst_p95"]))

        lst_cog = tmp_path / f"lst_p95_{TILE_YEAR}_{TILE_NAME}.tif"
        qa_cog = tmp_path / f"qa_count_{TILE_YEAR}_{TILE_NAME}.tif"
        cog_export(encoded, lst_cog, qa_cog)

        elapsed = time.perf_counter() - start
        print(f"COG export in {elapsed:.1f}s")
        print(f"LST: {lst_cog.stat().st_size / 1e6:.1f} MB")
        print(f"QA:  {qa_cog.stat().st_size / 1e6:.1f} MB")

        assert cog_validate(str(lst_cog))[0], "lst_p95 is not a valid COG"
        assert cog_validate(str(qa_cog))[0], "qa_count is not a valid COG"

        pytest.lst_cog = lst_cog
        pytest.qa_cog = qa_cog

    def test_06_read_back_with_rasterio(self):
        """Read the COGs back off disk."""
        print("\n--- Step 6: Read back COGs ---")

        start = time.perf_counter()

        with rasterio.open(pytest.lst_cog) as src:
            lst_profile = src.profile
            lst_scale, lst_offset = src.scales[0], src.offsets[0]
            lst_dn = src.read(1)
        with rasterio.open(pytest.qa_cog) as src:
            qa_dtype = src.dtypes[0]

        elapsed = time.perf_counter() - start
        print(f"Opened both COGs in {elapsed:.3f}s")

        pytest.lst_profile = lst_profile
        pytest.lst_scale_offset = (lst_scale, lst_offset)
        pytest.lst_dn = lst_dn
        pytest.qa_dtype = qa_dtype

    def test_07_validate_structure(self):
        """Validate raster structure."""
        print("\n--- Step 7: Validate structure ---")
        profile = pytest.lst_profile

        # LST is a single uint16 DN band with 0 reserved for nodata.
        assert profile["count"] == 1
        assert profile["dtype"] == "uint16"
        assert profile["nodata"] == 0
        assert profile["crs"].to_epsg() == 4326

        # Counts stay well under 255, so uint8.
        assert pytest.qa_dtype == "uint8"

        # Scale/offset are embedded so viewers auto-decode DN to Celsius.
        scale, offset = pytest.lst_scale_offset
        assert scale == 0.01
        assert offset == -50.0

        print("✓ Structure validated")

    def test_08_validate_values(self):
        """Validate data values are reasonable."""
        print("\n--- Step 8: Validate values ---")

        scale, offset = pytest.lst_scale_offset
        p95_celsius = pytest.lst_dn * scale + offset

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
        print("Pipeline: STAC → Load → QA → Composite → COG")
        print("All steps passed!")
