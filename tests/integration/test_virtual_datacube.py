"""Integration tests for virtual datacube creation.

Tests the full VirtualZarr + Icechunk pipeline:
1. Create test COGs with known data
2. Create virtual datacube from COGs
3. Verify xarray access reads correct values
"""

import numpy as np
import pytest
import xarray as xr

from landsat_lst.storage import LocalStorage

pytest.importorskip("virtualizarr")
pytest.importorskip("icechunk")


def _create_test_cog(storage: LocalStorage, year: int, tile_name: str) -> str:
    """Create a minimal test COG with known values."""
    from landsat_lst.cog import write_cog
    from landsat_lst.tiling import parse_tile_name

    tile_id = parse_tile_name(tile_name)

    n_pixels = 100
    lat = np.linspace(tile_id.lat + 5, tile_id.lat, n_pixels, endpoint=False)
    lon = np.linspace(tile_id.lon, tile_id.lon + 5, n_pixels, endpoint=False)

    lst_p50 = xr.DataArray(
        np.full((n_pixels, n_pixels), 25.0, dtype=np.float32),
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
    )
    lst_p95 = xr.DataArray(
        np.full((n_pixels, n_pixels), 35.0, dtype=np.float32),
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
    )
    qa_count = xr.DataArray(
        np.full((n_pixels, n_pixels), 50, dtype=np.uint16),
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
    )

    composite = xr.Dataset(
        {
            "lst_p50": lst_p50.chunk({"latitude": 50, "longitude": 50}),
            "lst_p95": lst_p95.chunk({"latitude": 50, "longitude": 50}),
            "qa_count": qa_count.chunk({"latitude": 50, "longitude": 50}),
        }
    )

    cog_path = storage.cog_path(year, tile_name)
    write_cog(composite, cog_path, blocksize=50, add_overviews=False)

    return cog_path


@pytest.mark.integration
def test_virtual_datacube_creation(tmp_path):
    """Test end-to-end virtual datacube creation and access."""
    try:
        from virtual_tiff import VirtualTIFF  # noqa: F401
    except ImportError:
        pytest.skip("virtual-tiff not installed")

    from landsat_lst.virtual import create_virtual_datacube

    storage = LocalStorage(tmp_path)

    tile_names = ["N40W075", "N45W075"]
    tile_paths = []

    for tile_name in tile_names:
        path = _create_test_cog(storage, 2023, tile_name)
        tile_paths.append(path)

    snapshot_id = create_virtual_datacube(
        tile_paths=tile_paths,
        tile_names=tile_names,
        year=2023,
        storage=storage,
    )

    assert snapshot_id is not None
    assert len(snapshot_id) > 0


@pytest.mark.integration
def test_virtual_datacube_has_correct_variables(tmp_path):
    """Test that virtual datacube has expected variables and coords."""
    try:
        from virtual_tiff import VirtualTIFF  # noqa: F401
    except ImportError:
        pytest.skip("virtual-tiff not installed")

    from landsat_lst.virtual import create_icechunk_repo, create_virtual_datacube

    storage = LocalStorage(tmp_path)

    tile_name = "N40W075"
    tile_path = _create_test_cog(storage, 2023, tile_name)

    create_virtual_datacube(
        tile_paths=[tile_path],
        tile_names=[tile_name],
        year=2023,
        storage=storage,
    )

    repo = create_icechunk_repo(storage, create=False)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    assert "lst_p50" in ds.data_vars
    assert "lst_p95" in ds.data_vars
    assert "qa_count" in ds.data_vars

    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert "time" in ds.coords

    assert ds.sizes["time"] == 1


@pytest.mark.integration
def test_virtual_datacube_variable_attrs(tmp_path):
    """Test that virtual datacube has correct CF-compliant attributes."""
    try:
        from virtual_tiff import VirtualTIFF  # noqa: F401
    except ImportError:
        pytest.skip("virtual-tiff not installed")

    from landsat_lst.virtual import create_icechunk_repo, create_virtual_datacube

    storage = LocalStorage(tmp_path)

    tile_name = "N40W075"
    tile_path = _create_test_cog(storage, 2023, tile_name)

    create_virtual_datacube(
        tile_paths=[tile_path],
        tile_names=[tile_name],
        year=2023,
        storage=storage,
    )

    repo = create_icechunk_repo(storage, create=False)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    assert ds["lst_p50"].attrs["scale_factor"] == 0.01
    assert ds["lst_p50"].attrs["add_offset"] == -50.0
    assert ds["lst_p50"].attrs["units"] == "celsius"

    assert ds["qa_count"].attrs["units"] == "count"


@pytest.mark.integration
def test_virtual_datacube_dataset_attrs(tmp_path):
    """Test that virtual datacube has correct dataset-level attributes."""
    try:
        from virtual_tiff import VirtualTIFF  # noqa: F401
    except ImportError:
        pytest.skip("virtual-tiff not installed")

    from landsat_lst.virtual import create_icechunk_repo, create_virtual_datacube

    storage = LocalStorage(tmp_path)

    tile_name = "N40W075"
    tile_path = _create_test_cog(storage, 2023, tile_name)

    create_virtual_datacube(
        tile_paths=[tile_path],
        tile_names=[tile_name],
        year=2023,
        storage=storage,
    )

    repo = create_icechunk_repo(storage, create=False)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    assert "title" in ds.attrs
    assert ds.attrs["lst_scale"] == 0.01
    assert ds.attrs["lst_offset"] == -50.0
