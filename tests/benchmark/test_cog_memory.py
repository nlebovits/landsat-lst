"""Memory profiling for chunked COG write pattern.

Validates that the rioxarray chunked write stays within expected
memory bounds regardless of total array size.

Run with: pytest tests/benchmark/test_cog_memory.py -v -s --benchmark
"""

import threading
import tracemalloc

import numpy as np
import pytest
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles


def get_peak_memory_mb() -> float:
    """Get peak memory usage in MB from tracemalloc."""
    _, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


def create_lazy_composite(
    shape: tuple[int, int],
    chunk_size: int = 512,
) -> xr.Dataset:
    """Create a chunked Dataset simulating LST composite."""
    rng = np.random.default_rng(42)

    ds = xr.Dataset(
        {
            "lst_p50": (["latitude", "longitude"], rng.uniform(10, 40, shape).astype(np.float32)),
            "lst_p95": (["latitude", "longitude"], rng.uniform(30, 50, shape).astype(np.float32)),
            "qa_count": (["latitude", "longitude"], rng.integers(0, 100, shape).astype(np.int16)),
        },
        coords={
            "latitude": np.linspace(-34.0, -33.0, shape[0]),
            "longitude": np.linspace(-61.0, -60.0, shape[1]),
        },
    )
    return ds.chunk({"latitude": chunk_size, "longitude": chunk_size})


@pytest.mark.benchmark
class TestCOGMemoryProfile:
    """Memory profiling tests for chunked COG writes."""

    def test_memory_scales_with_chunk_not_total(self, tmp_path):
        """Verify peak memory is bounded by chunk size, not total array size.

        Creates a 4096x4096 array (64 tiles at 512x512) and verifies that
        peak memory stays well under what a full materialization would require.

        Expected memory:
        - Full materialization: 4096x4096 x 3 bands x 4 bytes = ~201 MB
        - Chunked write: 512x512 x 3 bands x 4 bytes = ~3 MB per chunk + overhead
        - Expected peak: < 50 MB (generous buffer for Dask overhead)
        """
        shape = (4096, 4096)
        chunk_size = 512

        # Memory for full array (what we're avoiding)
        full_memory_mb = (shape[0] * shape[1] * 3 * 4) / (1024 * 1024)
        print(f"\nFull array memory: {full_memory_mb:.1f} MB")

        # Memory per chunk
        chunk_memory_mb = (chunk_size * chunk_size * 3 * 4) / (1024 * 1024)
        print(f"Single chunk memory: {chunk_memory_mb:.1f} MB")

        # Create lazy dataset
        ds = create_lazy_composite(shape, chunk_size)
        ds = ds.rio.write_crs("EPSG:4326")
        ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

        # Start memory tracking
        tracemalloc.start()

        # Write with chunked streaming
        tmp_tif = tmp_path / "memory_test.tif"
        stacked = ds.to_array(dim="band")
        stacked.rio.to_raster(
            str(tmp_tif),
            tiled=True,
            lock=threading.Lock(),
        )

        peak_mb = get_peak_memory_mb()
        tracemalloc.stop()

        print(f"Peak memory during write: {peak_mb:.1f} MB")
        print(f"Memory ratio (peak/full): {peak_mb / full_memory_mb:.2%}")

        # Verify we stayed well under full materialization
        # Allow generous headroom for Dask task graph overhead
        max_allowed_mb = full_memory_mb * 0.5  # Should be <50% of full
        assert peak_mb < max_allowed_mb, (
            f"Peak memory {peak_mb:.1f} MB exceeded {max_allowed_mb:.1f} MB "
            f"(50% of full array). Chunked streaming may not be working."
        )

        # Verify output is valid
        assert tmp_tif.exists()
        with rasterio.open(tmp_tif) as src:
            assert src.count == 3
            assert src.width == shape[1]
            assert src.height == shape[0]

    def test_memory_profile_with_cog_translation(self, tmp_path):
        """Profile full pipeline including COG translation.

        Tests that the two-stage approach (chunked write + COG translate)
        stays memory-bounded even though COG translation reads the file.
        """
        shape = (2048, 2048)
        chunk_size = 512

        ds = create_lazy_composite(shape, chunk_size)
        ds = ds.rio.write_crs("EPSG:4326")
        ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

        tracemalloc.start()

        # Stage 1: Chunked write
        tmp_tif = tmp_path / "temp.tif"
        stacked = ds.to_array(dim="band")
        stacked.rio.to_raster(str(tmp_tif), tiled=True, lock=threading.Lock())

        peak_after_write = get_peak_memory_mb()

        # Stage 2: COG translation
        cog_path = tmp_path / "output.tif"
        dst_profile = cog_profiles.get("deflate")
        dst_profile["blockxsize"] = 512
        dst_profile["blockysize"] = 512

        cog_translate(
            str(tmp_tif),
            str(cog_path),
            dst_profile,
            use_cog_driver=True,
            quiet=True,
        )

        peak_after_translate = get_peak_memory_mb()
        tracemalloc.stop()

        print(f"\nPeak after chunked write: {peak_after_write:.1f} MB")
        print(f"Peak after COG translate: {peak_after_translate:.1f} MB")

        # Both stages should be memory-bounded
        full_memory_mb = (shape[0] * shape[1] * 3 * 4) / (1024 * 1024)
        assert peak_after_translate < full_memory_mb, (
            f"Peak {peak_after_translate:.1f} MB exceeded full array size {full_memory_mb:.1f} MB"
        )

    def test_compare_chunked_vs_computed(self, tmp_path):
        """Compare memory usage: chunked write vs pre-computed write.

        WARNING: This test intentionally triggers high memory usage
        to demonstrate the difference. Uses smaller array to avoid OOM.
        """
        shape = (1024, 1024)  # Small enough to compute safely
        chunk_size = 256

        ds = create_lazy_composite(shape, chunk_size)
        ds = ds.rio.write_crs("EPSG:4326")
        ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

        # Method 1: Chunked write (memory-bounded)
        tracemalloc.start()
        tmp_chunked = tmp_path / "chunked.tif"
        stacked = ds.to_array(dim="band")
        stacked.rio.to_raster(str(tmp_chunked), tiled=True, lock=threading.Lock())
        chunked_peak = get_peak_memory_mb()
        tracemalloc.stop()

        # Method 2: Compute then write (full materialization)
        tracemalloc.start()
        tmp_computed = tmp_path / "computed.tif"
        stacked_computed = ds.to_array(dim="band").compute()  # Full materialization
        stacked_computed.rio.to_raster(str(tmp_computed), tiled=True)
        computed_peak = get_peak_memory_mb()
        tracemalloc.stop()

        print(f"\nChunked write peak: {chunked_peak:.1f} MB")
        print(f"Computed write peak: {computed_peak:.1f} MB")
        print(f"Memory savings: {(1 - chunked_peak / computed_peak) * 100:.0f}%")

        # Chunked should use significantly less memory
        assert chunked_peak < computed_peak, (
            "Chunked write should use less memory than computed write"
        )
