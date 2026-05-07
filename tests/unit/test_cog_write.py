"""Unit tests for chunked COG write pattern.

Tests the memory-bounded rioxarray write pattern that enables
large tile processing without OOM.
"""

import threading

import numpy as np
import pytest
import rasterio
import xarray as xr
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from landsat_lst.cog import write_cog


class TestChunkedCOGWrite:
    """Test chunked COG write pattern for memory efficiency."""

    @pytest.fixture
    def lazy_composite(self) -> xr.DataArray:
        """Create a lazy (chunked) DataArray simulating LST composite."""
        # Small test: 1024x1024 with 512x512 chunks (4 chunks total)
        shape = (1024, 1024)
        chunks = (512, 512)

        # Create lazy array via dask
        data = np.random.default_rng(42).uniform(10.0, 40.0, shape).astype(np.float32)
        da = xr.DataArray(
            data,
            dims=["latitude", "longitude"],
            coords={
                "latitude": np.linspace(-34.0, -33.0, shape[0]),
                "longitude": np.linspace(-61.0, -60.0, shape[1]),
            },
            name="lst_p50",
        )
        # Chunk it to make it lazy
        return da.chunk({"latitude": chunks[0], "longitude": chunks[1]})

    def test_rioxarray_chunked_write(self, lazy_composite, tmp_path):
        """Test that rioxarray writes chunks without materializing full array."""
        import rioxarray  # noqa: F401 - needed for .rio accessor

        da = lazy_composite
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

        # Verify it's still lazy (not computed)
        assert da.chunks is not None, "DataArray should be chunked/lazy"

        # Write with rioxarray - this should stream chunks
        tmp_tif = tmp_path / "chunked.tif"
        da.rio.to_raster(
            str(tmp_tif),
            tiled=True,
            lock=threading.Lock(),
        )

        # Verify output
        assert tmp_tif.exists()
        with rasterio.open(tmp_tif) as src:
            assert src.is_tiled
            assert src.width == 1024
            assert src.height == 1024
            # Check data integrity
            read_data = src.read(1)
            np.testing.assert_allclose(read_data, da.values, rtol=1e-5)

    def test_cog_translation_after_chunked_write(self, lazy_composite, tmp_path):
        """Test full pipeline: chunked write → COG translation."""
        import rioxarray  # noqa: F401

        da = lazy_composite
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

        # Stage 1: Chunked write to temp tif
        tmp_tif = tmp_path / "temp.tif"
        da.rio.to_raster(
            str(tmp_tif),
            tiled=True,
            lock=threading.Lock(),
        )

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

        # Verify COG structure
        with rasterio.open(cog_path) as src:
            assert src.is_tiled
            block_shapes = src.block_shapes
            assert block_shapes[0] == (512, 512), f"Expected 512x512 blocks, got {block_shapes[0]}"

    def test_multiband_chunked_write(self, tmp_path):
        """Test chunked write with multiple bands (lst_p50, lst_p95, qa_count)."""
        import rioxarray  # noqa: F401

        shape = (512, 512)
        chunks = (256, 256)
        rng = np.random.default_rng(42)

        # Create 3-band dataset
        ds = xr.Dataset(
            {
                "lst_p50": (
                    ["latitude", "longitude"],
                    rng.uniform(10, 40, shape).astype(np.float32),
                ),
                "lst_p95": (
                    ["latitude", "longitude"],
                    rng.uniform(30, 50, shape).astype(np.float32),
                ),
                "qa_count": (
                    ["latitude", "longitude"],
                    rng.integers(0, 100, shape).astype(np.int16),
                ),
            },
            coords={
                "latitude": np.linspace(-34.0, -33.5, shape[0]),
                "longitude": np.linspace(-61.0, -60.5, shape[1]),
            },
        )
        ds = ds.chunk({"latitude": chunks[0], "longitude": chunks[1]})
        ds = ds.rio.write_crs("EPSG:4326")
        ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

        # Write each band separately (rioxarray Dataset.to_raster writes bands)
        tmp_tif = tmp_path / "multiband.tif"

        # Stack into DataArray for multi-band write
        stacked = ds.to_array(dim="band")
        stacked.rio.to_raster(
            str(tmp_tif),
            tiled=True,
            lock=threading.Lock(),
        )

        with rasterio.open(tmp_tif) as src:
            assert src.count == 3
            assert src.is_tiled


class TestWriteCogFunction:
    """Test the write_cog function from landsat_lst.cog module."""

    @pytest.fixture
    def composite_dataset(self) -> xr.Dataset:
        """Create a chunked Dataset simulating LST composite output.

        Uses 2048x2048 to ensure multiple 512x512 tiles are created.
        Smaller images get optimized to non-tiled format by rio-cogeo.
        """
        shape = (2048, 2048)
        chunks = (512, 512)
        rng = np.random.default_rng(42)

        ds = xr.Dataset(
            {
                "lst_p50": (
                    ["latitude", "longitude"],
                    rng.uniform(10, 40, shape).astype(np.float32),
                ),
                "lst_p95": (
                    ["latitude", "longitude"],
                    rng.uniform(30, 50, shape).astype(np.float32),
                ),
                "qa_count": (
                    ["latitude", "longitude"],
                    rng.integers(0, 100, shape).astype(np.int16),
                ),
            },
            coords={
                "latitude": np.linspace(-34.0, -33.0, shape[0]),
                "longitude": np.linspace(-61.0, -60.0, shape[1]),
            },
        )
        return ds.chunk({"latitude": chunks[0], "longitude": chunks[1]})

    def test_write_cog_basic(self, composite_dataset, tmp_path):
        """Test basic COG write with default settings."""
        output_path = tmp_path / "test_cog.tif"
        result = write_cog(composite_dataset, output_path)

        assert result == output_path
        assert output_path.exists()

        with rasterio.open(output_path) as src:
            assert src.count == 3
            assert src.is_tiled
            assert src.crs.to_epsg() == 4326

    def test_write_cog_with_overviews(self, composite_dataset, tmp_path):
        """Test COG write with overviews enabled."""
        output_path = tmp_path / "test_cog_overviews.tif"
        write_cog(composite_dataset, output_path, add_overviews=True)

        with rasterio.open(output_path) as src:
            # Should have overviews
            assert len(src.overviews(1)) > 0

    def test_write_cog_without_overviews(self, composite_dataset, tmp_path):
        """Test COG write without overviews."""
        output_path = tmp_path / "test_cog_no_overviews.tif"
        write_cog(composite_dataset, output_path, add_overviews=False)

        with rasterio.open(output_path) as src:
            # Should have no overviews
            assert len(src.overviews(1)) == 0

    def test_write_cog_rejects_non_chunked(self, tmp_path):
        """Test that write_cog rejects non-chunked (computed) arrays."""
        shape = (256, 256)
        ds = xr.Dataset(
            {"lst_p50": (["latitude", "longitude"], np.zeros(shape, dtype=np.float32))},
            coords={
                "latitude": np.linspace(-34.0, -33.5, shape[0]),
                "longitude": np.linspace(-61.0, -60.5, shape[1]),
            },
        )
        # NOT chunked - should raise

        with pytest.raises(ValueError, match="must be chunked"):
            write_cog(ds, tmp_path / "should_fail.tif")

    def test_write_cog_cleans_up_temp_file(self, composite_dataset, tmp_path):
        """Test that temporary file is cleaned up after COG translation."""
        output_path = tmp_path / "test_cog.tif"
        tmp_tif = output_path.with_suffix(".tmp.tif")

        write_cog(composite_dataset, output_path)

        assert output_path.exists()
        assert not tmp_tif.exists(), "Temp file should be cleaned up"
