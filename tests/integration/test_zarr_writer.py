"""Integration tests for Zarr writing.

Tests the direct Zarr write pipeline:
1. Create test composite with known data
2. Write to Zarr store
3. Verify xarray access reads correct values
"""

import numpy as np
import pytest
import xarray as xr

from landsat_lst.storage import LocalStorage
from landsat_lst.zarr_writer import (
    DEFAULT_CHUNKS,
    LST_OFFSET,
    LST_SCALE,
    encode_lst_uint16,
    write_zarr,
)


def _create_test_composite(tile_name: str, n_pixels: int = 100) -> xr.Dataset:
    """Create a minimal test composite with known values."""
    from landsat_lst.tiling import parse_tile_name

    tile_id = parse_tile_name(tile_name)

    lat = np.linspace(tile_id.lat + 5, tile_id.lat, n_pixels, endpoint=False)
    lon = np.linspace(tile_id.lon, tile_id.lon + 5, n_pixels, endpoint=False)

    # Known values for verification
    lst_p50_celsius = 25.0
    lst_p95_celsius = 35.0
    qa_count_value = 50

    lst_p50 = xr.DataArray(
        np.full((n_pixels, n_pixels), lst_p50_celsius, dtype=np.float32),
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
    )
    lst_p95 = xr.DataArray(
        np.full((n_pixels, n_pixels), lst_p95_celsius, dtype=np.float32),
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
    )
    qa_count = xr.DataArray(
        np.full((n_pixels, n_pixels), qa_count_value, dtype=np.uint16),
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
    )

    return xr.Dataset(
        {
            "lst_p50": lst_p50.chunk({"latitude": 50, "longitude": 50}),
            "lst_p95": lst_p95.chunk({"latitude": 50, "longitude": 50}),
            "qa_count": qa_count.chunk({"latitude": 50, "longitude": 50}),
        }
    )


@pytest.mark.integration
def test_encode_lst_uint16():
    """Test uint16 encoding preserves values correctly."""
    celsius_values = np.array([0.0, 25.0, 35.0, 50.0, -10.0])
    data = xr.DataArray(celsius_values, dims=["x"])

    encoded = encode_lst_uint16(data)

    assert encoded.dtype == np.uint16

    # Decode and verify
    decoded = encoded.values * LST_SCALE + LST_OFFSET
    np.testing.assert_array_almost_equal(decoded, celsius_values, decimal=2)


@pytest.mark.integration
def test_encode_lst_uint16_nodata():
    """Test that nodata values are encoded as 0."""
    data = xr.DataArray(
        np.array([25.0, -9999.0, np.nan, 35.0]),
        dims=["x"],
    )

    encoded = encode_lst_uint16(data)

    assert encoded.values[0] != 0  # Valid value
    assert encoded.values[1] == 0  # -9999.0 -> 0
    assert encoded.values[2] == 0  # NaN -> 0
    assert encoded.values[3] != 0  # Valid value


@pytest.mark.integration
def test_write_zarr_creates_store(tmp_path):
    """Test that write_zarr creates a valid Zarr store."""
    from pathlib import Path

    storage = LocalStorage(tmp_path)
    composite = _create_test_composite("N40W075")

    zarr_path = storage.zarr_path(2023, "N40W075")
    result_path = write_zarr(composite, zarr_path, chunks=(50, 50))

    result_path = Path(result_path)
    assert result_path.exists()
    assert (result_path / ".zmetadata").exists() or (result_path / "zarr.json").exists()


@pytest.mark.integration
def test_write_zarr_roundtrip(tmp_path):
    """Test that written Zarr can be read back correctly."""
    storage = LocalStorage(tmp_path)
    composite = _create_test_composite("N40W075")

    zarr_path = storage.zarr_path(2023, "N40W075")
    write_zarr(composite, zarr_path, chunks=(50, 50))

    # Read back
    ds = xr.open_zarr(zarr_path)

    assert "lst_p50" in ds.data_vars
    assert "lst_p95" in ds.data_vars
    assert "qa_count" in ds.data_vars

    # Verify uint16 encoding
    assert ds["lst_p50"].dtype == np.uint16
    assert ds["lst_p95"].dtype == np.uint16
    assert ds["qa_count"].dtype == np.uint16


@pytest.mark.integration
def test_write_zarr_value_roundtrip(tmp_path):
    """Test that values survive encoding/decoding roundtrip."""
    storage = LocalStorage(tmp_path)
    composite = _create_test_composite("N40W075")

    zarr_path = storage.zarr_path(2023, "N40W075")
    write_zarr(composite, zarr_path, chunks=(50, 50))

    ds = xr.open_zarr(zarr_path)

    # Decode and check values
    scale = ds["lst_p50"].attrs["lst_scale_factor"]
    offset = ds["lst_p50"].attrs["lst_add_offset"]

    decoded_p50 = ds["lst_p50"].values * scale + offset
    decoded_p95 = ds["lst_p95"].values * scale + offset

    # Original values were 25.0 and 35.0
    np.testing.assert_array_almost_equal(decoded_p50, 25.0, decimal=2)
    np.testing.assert_array_almost_equal(decoded_p95, 35.0, decimal=2)


@pytest.mark.integration
def test_write_zarr_has_crs(tmp_path):
    """Test that written Zarr has CRS metadata."""
    storage = LocalStorage(tmp_path)
    composite = _create_test_composite("N40W075")

    zarr_path = storage.zarr_path(2023, "N40W075")
    write_zarr(composite, zarr_path, chunks=(50, 50))

    ds = xr.open_zarr(zarr_path)

    assert "_CRS" in ds.attrs
    assert "EPSG" in ds.attrs["crs"] or "4326" in ds.attrs["_CRS"]


@pytest.mark.integration
def test_write_zarr_variable_attrs(tmp_path):
    """Test that variables have correct attributes."""
    storage = LocalStorage(tmp_path)
    composite = _create_test_composite("N40W075")

    zarr_path = storage.zarr_path(2023, "N40W075")
    write_zarr(composite, zarr_path, chunks=(50, 50))

    ds = xr.open_zarr(zarr_path)

    # LST bands have non-CF encoding attrs
    assert ds["lst_p50"].attrs["lst_scale_factor"] == LST_SCALE
    assert ds["lst_p50"].attrs["lst_add_offset"] == LST_OFFSET

    # QA count has units
    assert ds["qa_count"].attrs["units"] == "count"


@pytest.mark.integration
def test_storage_zarr_exists(tmp_path):
    """Test that storage correctly detects Zarr existence."""
    storage = LocalStorage(tmp_path)
    composite = _create_test_composite("N40W075")

    assert not storage.zarr_exists(2023, "N40W075")

    zarr_path = storage.zarr_path(2023, "N40W075")
    write_zarr(composite, zarr_path, chunks=(50, 50))

    assert storage.zarr_exists(2023, "N40W075")


@pytest.mark.integration
def test_write_zarr_default_chunks():
    """Test that DEFAULT_CHUNKS is set correctly."""
    assert DEFAULT_CHUNKS == (500, 500)


@pytest.mark.integration
def test_write_zarr_to_icechunk_session(tmp_path):
    """Test writing to Icechunk session."""
    from landsat_lst.storage import IcechunkStorage

    # Create Icechunk storage
    storage = IcechunkStorage.from_local(tmp_path / "icechunk")
    composite = _create_test_composite("N40W075")

    # Get writable session and write
    session = storage.writable_session()
    group_path = write_zarr(composite, session, group="2023/N40W075", chunks=(50, 50))

    # Commit
    commit_id = session.commit("test commit")

    assert group_path == "2023/N40W075"
    assert commit_id is not None

    # Read back and verify
    read_session = storage.readonly_session()
    ds = xr.open_zarr(read_session.store, group="2023/N40W075", consolidated=False)

    assert "lst_p50" in ds.data_vars
    assert ds["lst_p50"].dtype == np.uint16


@pytest.mark.integration
def test_write_zarr_icechunk_roundtrip_values(tmp_path):
    """Test that values survive Icechunk roundtrip."""
    from landsat_lst.storage import IcechunkStorage

    storage = IcechunkStorage.from_local(tmp_path / "icechunk")
    composite = _create_test_composite("N40W075")

    session = storage.writable_session()
    write_zarr(composite, session, group="2023/N40W075", chunks=(50, 50))
    session.commit("test")

    # Read back
    ds = xr.open_zarr(storage.readonly_session().store, group="2023/N40W075", consolidated=False)

    # Decode and verify values (original: 25.0 and 35.0 Celsius)
    scale = ds["lst_p50"].attrs["lst_scale_factor"]
    offset = ds["lst_p50"].attrs["lst_add_offset"]

    decoded_p50 = ds["lst_p50"].values * scale + offset
    decoded_p95 = ds["lst_p95"].values * scale + offset

    np.testing.assert_array_almost_equal(decoded_p50, 25.0, decimal=2)
    np.testing.assert_array_almost_equal(decoded_p95, 35.0, decimal=2)


@pytest.mark.integration
def test_icechunk_storage_zarr_exists(tmp_path):
    """Test IcechunkStorage.zarr_exists() works correctly."""
    from landsat_lst.storage import IcechunkStorage

    storage = IcechunkStorage.from_local(tmp_path / "icechunk")

    # Initially doesn't exist
    assert storage.zarr_exists(2023, "N40W075") is False

    # Write data
    composite = _create_test_composite("N40W075")
    session = storage.writable_session()
    write_zarr(composite, session, group="2023/N40W075", chunks=(50, 50))
    session.commit("test")

    # Now exists
    assert storage.zarr_exists(2023, "N40W075") is True

    # Other tile still doesn't exist
    assert storage.zarr_exists(2023, "N40W070") is False
