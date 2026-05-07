"""Parameter sweep tests for optimizing pipeline settings.

These tests explore different parameter combinations to find optimal settings
for cloud cover thresholds, chunk sizes, resampling methods, and COG profiles.

Run with: pytest -m sweep -v
"""

import time

import numpy as np
import planetary_computer
import pystac_client
import pytest
from odc.stac import configure_rio, stac_load

# Small bbox for fast sweeps (~0.1° near Pergamino)
SWEEP_BBOX = (-60.45, -33.95, -60.35, -33.85)
SWEEP_YEAR = 2024
SWEEP_DATETIME = f"{SWEEP_YEAR}-01-01/{SWEEP_YEAR}-12-31"


@pytest.fixture(scope="module")
def stac_client():
    """Shared STAC client with Planetary Computer signing."""
    configure_rio(cloud_defaults=True)
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


# =============================================================================
# Cloud Cover Threshold Sweep
# =============================================================================


@pytest.mark.sweep
class TestCloudCoverSweep:
    """Find optimal cloud cover threshold for scene availability vs quality."""

    @pytest.mark.parametrize("max_cloud", [10, 20, 30, 40, 50])
    def test_scene_count_by_cloud_threshold(self, stac_client, max_cloud):
        """Count available scenes at different cloud cover thresholds."""
        search = stac_client.search(
            collections=["landsat-c2-l2"],
            bbox=SWEEP_BBOX,
            datetime=SWEEP_DATETIME,
            query={
                "eo:cloud_cover": {"lt": max_cloud},
                "platform": {"in": ["landsat-8", "landsat-9"]},
            },
        )
        items = list(search.items())
        scene_count = len(items)

        # Record for analysis
        print(f"\nCloud <= {max_cloud}%: {scene_count} scenes")

        # Basic sanity checks
        assert scene_count >= 0
        # Higher threshold should yield more scenes (not strict due to query variance)

    def test_cloud_cover_distribution(self, stac_client):
        """Analyze cloud cover distribution to inform threshold choice."""
        search = stac_client.search(
            collections=["landsat-c2-l2"],
            bbox=SWEEP_BBOX,
            datetime=SWEEP_DATETIME,
            query={
                "eo:cloud_cover": {"lt": 100},  # All scenes
                "platform": {"in": ["landsat-8", "landsat-9"]},
            },
        )
        items = list(search.items())

        cloud_covers = [item.properties["eo:cloud_cover"] for item in items]

        if cloud_covers:
            print(f"\nCloud cover distribution (n={len(cloud_covers)}):")
            print(f"  Min: {min(cloud_covers):.1f}%")
            print(f"  Max: {max(cloud_covers):.1f}%")
            print(f"  Mean: {np.mean(cloud_covers):.1f}%")
            print(f"  Median: {np.median(cloud_covers):.1f}%")

            # Count by bucket
            buckets = [0, 10, 20, 30, 40, 50, 100]
            for i in range(len(buckets) - 1):
                count = sum(1 for cc in cloud_covers if buckets[i] <= cc < buckets[i + 1])
                print(f"  {buckets[i]}-{buckets[i + 1]}%: {count} scenes")


# =============================================================================
# Chunk Size Sweep
# =============================================================================


@pytest.mark.sweep
class TestChunkSizeSweep:
    """Find optimal chunk size for memory efficiency and read performance."""

    @pytest.fixture(scope="class")
    def stac_items(self, stac_client):
        """Get a few scenes for chunk testing."""
        search = stac_client.search(
            collections=["landsat-c2-l2"],
            bbox=SWEEP_BBOX,
            datetime=SWEEP_DATETIME,
            query={
                "eo:cloud_cover": {"lt": 20},
                "platform": {"in": ["landsat-8", "landsat-9"]},
            },
        )
        items = list(search.items())[:3]  # Just 3 scenes for speed
        if not items:
            pytest.skip("No scenes available")
        return items

    @pytest.mark.parametrize("chunk_size", [256, 512, 1024, 2048])
    def test_load_time_by_chunk_size(self, stac_items, chunk_size):
        """Measure load time for different chunk sizes."""
        start = time.perf_counter()

        ds = stac_load(
            stac_items,
            bands=["lwir11"],
            crs="EPSG:4326",
            resolution=0.00027778,
            chunks={"time": 1, "x": chunk_size, "y": chunk_size},
            bbox=SWEEP_BBOX,
        )

        # Force compute a small subset (dims are latitude/longitude, not x/y)
        subset = ds["lwir11"].isel(time=0, latitude=slice(0, 100), longitude=slice(0, 100))
        _ = subset.compute()

        elapsed = time.perf_counter() - start

        print(f"\nChunk {chunk_size}x{chunk_size}: {elapsed:.2f}s")

        # Record chunk info
        assert ds.chunks is not None

    def test_memory_estimate_by_chunk_size(self, stac_items):
        """Estimate memory usage for different chunk sizes."""
        print("\nMemory estimates per chunk (float32):")
        for chunk_size in [256, 512, 1024, 2048]:
            # float32 = 4 bytes, 2 bands (lwir11 + qa_pixel)
            bytes_per_chunk = chunk_size * chunk_size * 4 * 2
            mb_per_chunk = bytes_per_chunk / (1024 * 1024)
            print(f"  {chunk_size}x{chunk_size}: {mb_per_chunk:.1f} MB/chunk")


# =============================================================================
# Resampling Method Sweep
# =============================================================================


@pytest.mark.sweep
class TestResamplingSweep:
    """Compare resampling methods for quality and speed."""

    @pytest.fixture(scope="class")
    def stac_items(self, stac_client):
        """Get scenes for resampling tests."""
        search = stac_client.search(
            collections=["landsat-c2-l2"],
            bbox=SWEEP_BBOX,
            datetime=SWEEP_DATETIME,
            query={
                "eo:cloud_cover": {"lt": 20},
                "platform": {"in": ["landsat-8", "landsat-9"]},
            },
        )
        items = list(search.items())[:2]
        if not items:
            pytest.skip("No scenes available")
        return items

    @pytest.mark.parametrize(
        "resampling",
        ["nearest", "bilinear", "cubic", "average"],
    )
    def test_resampling_quality(self, stac_items, resampling):
        """Compare resampling methods on thermal data."""
        start = time.perf_counter()

        ds = stac_load(
            stac_items,
            bands=["lwir11"],
            crs="EPSG:4326",
            resolution=0.00027778,
            chunks={"time": 1, "x": 512, "y": 512},
            resampling={"lwir11": resampling},
            bbox=SWEEP_BBOX,
        )

        data = ds["lwir11"].isel(time=0).compute()
        elapsed = time.perf_counter() - start

        # Compute stats
        valid = data.where(data > 0)
        stats = {
            "min": float(valid.min()),
            "max": float(valid.max()),
            "mean": float(valid.mean()),
            "std": float(valid.std()),
        }

        print(f"\n{resampling}: {elapsed:.2f}s")
        print(f"  Range: {stats['min']:.0f} - {stats['max']:.0f}")
        print(f"  Mean: {stats['mean']:.1f} ± {stats['std']:.1f}")


# =============================================================================
# COG Compression Sweep
# =============================================================================


@pytest.mark.sweep
class TestCOGCompressionSweep:
    """Compare COG compression profiles for size vs read speed."""

    @pytest.fixture
    def test_data(self):
        """Generate realistic test data."""
        # Simulate LST data: ~300K (27°C) with noise
        rng = np.random.default_rng(42)
        data = 300 + rng.normal(0, 5, (1000, 1000)).astype(np.float32)
        return data

    @pytest.mark.parametrize(
        "compression",
        ["deflate", "lzw", "zstd", "webp", "none"],
    )
    def test_compression_size_and_speed(self, test_data, tmp_path, compression):
        """Compare compression profiles."""
        import rasterio
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles

        # Write temp TIFF
        tmp_tif = tmp_path / "temp.tif"
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": test_data.shape[1],
            "height": test_data.shape[0],
            "count": 1,
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_bounds(-60, -34, -59, -33, 1000, 1000),
        }
        with rasterio.open(tmp_tif, "w", **profile) as dst:
            dst.write(test_data, 1)

        # Translate to COG
        cog_path = tmp_path / f"test_{compression}.tif"

        if compression == "none":
            dst_profile = {"driver": "GTiff", "compress": None}
        elif compression == "webp":
            # WebP only works with uint8, skip for float32
            pytest.skip("WebP requires uint8 data")
        else:
            dst_profile = cog_profiles.get(compression)

        dst_profile["blockxsize"] = 512
        dst_profile["blockysize"] = 512

        start = time.perf_counter()
        cog_translate(
            str(tmp_tif),
            str(cog_path),
            dst_profile,
            use_cog_driver=True,
            overview_level=0,
            quiet=True,
        )
        write_time = time.perf_counter() - start

        # Measure file size
        file_size_mb = cog_path.stat().st_size / (1024 * 1024)

        # Measure read time
        start = time.perf_counter()
        with rasterio.open(cog_path) as src:
            _ = src.read(1)
        read_time = time.perf_counter() - start

        print(f"\n{compression}:")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Write: {write_time:.2f}s")
        print(f"  Read: {read_time:.3f}s")

        # Basic validation
        assert cog_path.exists()
        assert file_size_mb > 0


# =============================================================================
# Composite Statistics Sweep
# =============================================================================


@pytest.mark.sweep
class TestCompositeStatsSweep:
    """Compare different composite statistics (median, mean, percentiles)."""

    @pytest.fixture(scope="class")
    def loaded_data(self, stac_client):
        """Load data for composite testing."""
        search = stac_client.search(
            collections=["landsat-c2-l2"],
            bbox=SWEEP_BBOX,
            datetime=SWEEP_DATETIME,
            query={
                "eo:cloud_cover": {"lt": 20},
                "platform": {"in": ["landsat-8", "landsat-9"]},
            },
        )
        items = list(search.items())[:5]  # 5 scenes for composite
        if len(items) < 3:
            pytest.skip("Need at least 3 scenes for composite testing")

        ds = stac_load(
            items,
            bands=["lwir11", "qa_pixel"],
            crs="EPSG:4326",
            resolution=0.00027778,
            chunks={"time": 1, "x": 512, "y": 512},
            bbox=SWEEP_BBOX,
        )
        return ds.compute()

    @pytest.mark.parametrize(
        "stat_method",
        ["mean", "median", "p50", "p75", "p95"],
    )
    def test_composite_statistics(self, loaded_data, stat_method):
        """Compare different composite statistics."""
        # Apply QA mask
        qa = loaded_data["qa_pixel"]
        cloud = (qa >> 3) & 1
        shadow = (qa >> 4) & 1
        mask = (cloud == 0) & (shadow == 0)

        lst = loaded_data["lwir11"].where(mask)

        # Apply temperature conversion
        lst_celsius = lst * 0.00341802 + 149.0 - 273.15

        start = time.perf_counter()

        if stat_method == "mean":
            result = lst_celsius.mean(dim="time", skipna=True)
        elif stat_method == "median":
            result = lst_celsius.median(dim="time", skipna=True)
        elif stat_method == "p50":
            result = lst_celsius.quantile(0.5, dim="time", skipna=True)
        elif stat_method == "p75":
            result = lst_celsius.quantile(0.75, dim="time", skipna=True)
        elif stat_method == "p95":
            result = lst_celsius.quantile(0.95, dim="time", skipna=True)

        # Compute
        result_values = result.values
        elapsed = time.perf_counter() - start

        # Stats
        valid = result_values[~np.isnan(result_values)]
        if len(valid) > 0:
            print(f"\n{stat_method}: {elapsed:.3f}s")
            print(f"  Range: {valid.min():.1f}°C - {valid.max():.1f}°C")
            print(f"  Mean: {valid.mean():.1f}°C")
        else:
            print(f"\n{stat_method}: No valid data")
