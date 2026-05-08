#!/usr/bin/env python
"""Spike: Test VirtualZarr + Icechunk integration for Issue #10.

This script validates:
1. VirtualZarr TIFF reader with IFD handling for COGs with overviews
2. Per-tile Icechunk commits with VirtualChunkContainer
3. Virtual datacube access via xarray
4. Tile concatenation strategy

Run with: uv run python scripts/spike_virtualzarr_icechunk.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import icechunk as ic
import numpy as np
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtual_tiff import VirtualTIFF
from virtualizarr import open_virtual_dataset


def create_test_cog(output_path: Path, tile_name: str, lat_offset: float = 0.0) -> Path:
    """Create a small test COG with synthetic LST data.

    Args:
        output_path: Directory to write COG
        tile_name: Name for the COG file
        lat_offset: Offset for latitude coordinates (for creating adjacent tiles)

    Note:
        Array size (128) must be evenly divisible by chunk size (64) for
        VirtualZarr concatenation to work. 128/64 = 2 chunks per dimension.
    """
    from landsat_lst.cog import write_cog  # noqa: PLC0415

    # Use 128x128 to be evenly divisible by blocksize=64
    # This is required for VirtualZarr concatenation
    size = 128
    lat = np.linspace(40.0 + lat_offset, 40.5 + lat_offset, size)
    lon = np.linspace(-75.0, -74.5, size)

    # Synthetic LST values (20-35°C range)
    np.random.seed(hash(tile_name) % 2**32)
    lst_data = np.random.uniform(20.0, 35.0, (size, size)).astype(np.float32)

    # Add some nodata
    lst_data[0:5, 0:5] = -9999.0

    ds = xr.Dataset(
        {
            "lst_p50": (["latitude", "longitude"], lst_data),
            "lst_p95": (["latitude", "longitude"], lst_data + 5.0),
            "qa_count": (["latitude", "longitude"], np.full((size, size), 50, dtype=np.uint16)),
        },
        coords={"latitude": lat, "longitude": lon},
    )

    # Chunk size must divide evenly into array size
    ds = ds.chunk({"latitude": 64, "longitude": 64})

    cog_path = output_path / f"{tile_name}.tif"
    write_cog(ds, cog_path, blocksize=64, add_overviews=True)
    print(f"✓ Created COG: {cog_path} ({cog_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Size: {size}x{size}, Lat range: {lat[0]:.2f} to {lat[-1]:.2f}")
    return cog_path


def test_virtualzarr_tiff_reader(cog_path: Path, registry: ObjectStoreRegistry) -> xr.Dataset:
    """Test VirtualZarr's TIFF reader with IFD handling."""
    print("\n--- Testing VirtualZarr TIFF Reader ---")

    # Test 1: Basic open_virtual_dataset with VirtualTIFF parser (no IFD specified)
    # NOTE: This fails for COGs with overviews due to dimension conflicts
    print("Testing open_virtual_dataset with VirtualTIFF() parser (no ifd)...")
    try:
        vds = open_virtual_dataset(
            f"file://{cog_path}",
            parser=VirtualTIFF(),
            registry=registry,
        )
        print(f"✓ Opened virtual dataset: {list(vds.data_vars)}")
        print(f"  Dims: {dict(vds.sizes)}")
        print(f"  Coords: {list(vds.coords)}")
    except Exception as e:
        print(
            f"✗ VirtualTIFF() without ifd failed (expected for COGs with overviews): {type(e).__name__}"
        )
        vds = None

    # Test 2: With IFD=0 for full resolution (COGs with overviews)
    # THIS IS THE REQUIRED PATTERN for COGs with overviews
    print("\nTesting with VirtualTIFF(ifd=0) for full resolution...")
    try:
        vds_ifd = open_virtual_dataset(
            f"file://{cog_path}",
            parser=VirtualTIFF(ifd=0),  # IFD 0 = full resolution
            registry=registry,
        )
        print(f"✓ IFD=0 selection worked: dims={dict(vds_ifd.sizes)}")
        vds = vds_ifd  # Use IFD=0 version
    except Exception as e:
        print(f"✗ IFD=0 selection failed: {e}")

    # Test 3: Inspect chunk manifest structure
    if vds is not None:
        print("\nInspecting chunk manifest...")
        for var_name in list(vds.data_vars)[:1]:  # Just first var
            var = vds[var_name]
            if hasattr(var.data, "manifest"):
                manifest = var.data.manifest
                print(f"  {var_name}: {len(manifest)} chunks")
                # Show first chunk reference
                for key in list(manifest.keys())[:1]:
                    ref = manifest[key]
                    # Handle both dict and object formats
                    if isinstance(ref, dict):
                        print(f"    Chunk {key}: {ref}")
                    else:
                        print(f"    Chunk {key}: offset={ref.offset}, length={ref.length}")
            else:
                print(f"  {var_name}: No manifest (type={type(var.data)})")

    return vds


def test_spatial_concatenation(
    cog_paths: list[Path], tmp_dir: Path, registry: ObjectStoreRegistry
) -> ic.Repository:
    """Test spatial tile concatenation then commit to Icechunk.

    For SPATIAL tiles (different lat/lon regions), we must concatenate
    all tiles into a single virtual dataset BEFORE writing to Icechunk.
    This is because each tile covers different coordinates.

    For TEMPORAL data (same tile, different years), we can use append_dim.
    """
    print("\n--- Testing Spatial Tile Concatenation ---")

    # Create Icechunk repository with VirtualChunkContainer
    icechunk_path = tmp_dir / "icechunk_store"
    storage = ic.local_filesystem_storage(str(icechunk_path))

    config = ic.config.RepositoryConfig.default()
    container_prefix = f"file://{tmp_dir}/"
    config.set_virtual_chunk_container(
        ic.virtual.VirtualChunkContainer(
            container_prefix,
            ic.storage.local_filesystem_store(str(tmp_dir)),
        )
    )

    # For virtual chunks, we need to authorize access to the container
    # For local files, we pass None which uses default/anonymous access
    credentials = ic.credentials.containers_credentials(
        {container_prefix: None}  # None = use environment/anonymous
    )

    print(f"Creating Icechunk repo with container: {container_prefix}")
    repo = ic.Repository.create(
        storage,
        config=config,
        authorize_virtual_chunk_access=credentials,
    )

    # Step 1: Open all tiles as virtual datasets
    print("\nOpening virtual datasets for all tiles...")
    vds_list = []
    for cog_path in cog_paths:
        vds = open_virtual_dataset(
            f"file://{cog_path}",
            parser=VirtualTIFF(ifd=0),
            registry=registry,
        )
        vds_list.append(vds)
        print(f"  ✓ {cog_path.stem}: dims={dict(vds.sizes)}")

    # Step 2: Combine using coordinates
    # For spatial tiles with different coordinates, use combine_by_coords
    # This aligns tiles based on their y/x coordinate values
    print("\nCombining tiles by coordinates...")
    try:
        combined = xr.combine_by_coords(vds_list, combine_attrs="override")
        print(f"✓ Combined dataset: dims={dict(combined.sizes)}")
    except Exception as e:
        print(f"✗ combine_by_coords failed: {e}")
        print("\nTrying manual concatenation along y dimension...")
        # Alternative: concatenate along spatial dimension
        combined = xr.concat(vds_list, dim="y", combine_attrs="override")
        print(f"✓ Combined dataset (y-concat): dims={dict(combined.sizes)}")

    # Step 3: Write to Icechunk
    print("\nWriting to Icechunk...")
    session = repo.writable_session("main")
    combined.vz.to_icechunk(session.store)
    snapshot_id = session.commit("Add all spatial tiles")
    print(f"✓ Committed: {snapshot_id[:12]}...")

    return repo


def test_temporal_append(
    cog_paths: list[Path], tmp_dir: Path, registry: ObjectStoreRegistry
) -> ic.Repository:
    """Test per-year commits with append_dim for temporal data.

    This pattern is for adding years incrementally (same tile, different years).
    Each commit appends along the time dimension.
    """
    import pandas as pd  # noqa: PLC0415

    print("\n--- Testing Temporal Append (Per-Year Commits) ---")

    # Create Icechunk repository
    icechunk_path = tmp_dir / "icechunk_temporal"
    storage = ic.local_filesystem_storage(str(icechunk_path))

    config = ic.config.RepositoryConfig.default()
    container_prefix = f"file://{tmp_dir}/"
    config.set_virtual_chunk_container(
        ic.virtual.VirtualChunkContainer(
            container_prefix,
            ic.storage.local_filesystem_store(str(tmp_dir)),
        )
    )

    print(f"Creating Icechunk repo with container: {container_prefix}")
    repo = ic.Repository.create(storage, config)

    # Simulate per-year commits (using same COGs but different years)
    years = [2023, 2024]
    for i, (cog_path, year) in enumerate(zip(cog_paths, years, strict=True)):
        print(f"\nCommitting year {year} (tile: {cog_path.stem})...")

        vds = open_virtual_dataset(
            f"file://{cog_path}",
            parser=VirtualTIFF(ifd=0),
            registry=registry,
        )

        # Add time coordinate
        vds = vds.expand_dims(time=[pd.Timestamp(f"{year}-01-01")])
        print(f"  Dims with time: {dict(vds.sizes)}")

        session = repo.writable_session("main")

        if i == 0:
            # First write: create the dataset
            vds.vz.to_icechunk(session.store)
        else:
            # Subsequent writes: append along time dimension
            try:
                vds.vz.to_icechunk(session.store, append_dim="time")
            except Exception as e:
                print(f"  ✗ append_dim failed: {e}")
                print("  Note: append_dim may require matching spatial dims")
                break

        snapshot_id = session.commit(f"Add year {year}")
        print(f"  ✓ Committed: {snapshot_id[:12]}...")

    return repo


def test_virtual_datacube_access(_repo: ic.Repository, tmp_dir: Path) -> None:
    """Test accessing the virtual datacube via xarray."""
    print("\n--- Testing Virtual Datacube Access ---")

    # For virtual chunks, we need to pass credentials when opening
    # For local files, we pass None which uses default/anonymous access
    credentials = ic.credentials.containers_credentials(
        {f"file://{tmp_dir}/": None}  # None = use environment/anonymous
    )

    # Re-open repository with credentials for virtual chunk access
    storage = ic.local_filesystem_storage(str(tmp_dir / "icechunk_store"))
    repo_with_creds = ic.Repository.open(
        storage,
        authorize_virtual_chunk_access=credentials,
    )
    session = repo_with_creds.readonly_session("main")

    print("Opening with xr.open_zarr...")
    try:
        ds = xr.open_zarr(session.store, consolidated=False)
        print("✓ Opened virtual datacube")
        print(f"  Variables: {list(ds.data_vars)}")
        print(f"  Dimensions: {dict(ds.sizes)}")
        print(f"  Coordinates: {list(ds.coords)}")

        # Test data access (should trigger byte-range reads from COG)
        print("\nTesting data access (triggers byte-range reads from COG)...")
        for var_name in list(ds.data_vars)[:1]:
            var = ds[var_name]
            # Read a small slice
            sample = var.isel({dim: slice(0, 10) for dim in var.dims}).values
            print(f"  {var_name}: read {sample.shape} values, mean={np.nanmean(sample):.2f}")

        print("\n✓ Virtual datacube access working!")

    except Exception as e:
        print(f"✗ Datacube access failed: {e}")
        import traceback  # noqa: PLC0415

        traceback.print_exc()


def main():
    """Run the full spike test."""
    print("=" * 60)
    print("VirtualZarr + Icechunk Spike Test for Issue #10")
    print("=" * 60)

    # Create temp directory for test artifacts
    tmp_dir = Path(tempfile.mkdtemp(prefix="landsat_lst_spike_"))
    print(f"\nWorking directory: {tmp_dir}")

    # Create ObjectStoreRegistry for local files
    # This maps file:// URLs to a LocalStore
    store = LocalStore(str(tmp_dir))
    registry = ObjectStoreRegistry({f"file://{tmp_dir}/": store})
    print(f"Created registry for: file://{tmp_dir}/")

    try:
        # Task 1: Create test COGs with different coordinate ranges
        print("\n" + "=" * 60)
        print("TASK 1: Create Test COGs (Adjacent Tiles)")
        print("=" * 60)
        cog_paths = []
        # Create two tiles that are adjacent in latitude
        # N40W075: lat 40.0-40.5
        # N41W075: lat 40.5-41.0 (offset by 0.5)
        tiles = [("N40W075", 0.0), ("N41W075", 0.5)]
        for tile_name, lat_offset in tiles:
            cog_path = create_test_cog(tmp_dir, tile_name, lat_offset=lat_offset)
            cog_paths.append(cog_path)

        # Task 2: Test VirtualZarr TIFF reader
        print("\n" + "=" * 60)
        print("TASK 2: VirtualZarr TIFF Reader")
        print("=" * 60)
        _vds = test_virtualzarr_tiff_reader(cog_paths[0], registry)

        # Task 3: Test spatial concatenation (correct approach for tiles)
        print("\n" + "=" * 60)
        print("TASK 3: Spatial Tile Concatenation")
        print("=" * 60)
        repo = test_spatial_concatenation(cog_paths, tmp_dir, registry)

        # Task 4: Test virtual datacube access
        print("\n" + "=" * 60)
        print("TASK 4: Virtual Datacube Access")
        print("=" * 60)
        test_virtual_datacube_access(repo, tmp_dir)

        # Task 5: Note on temporal append
        # Skipping because VirtualZarr has a bug with expand_dims when
        # the source TIFF has a transpose codec. This affects adding
        # time dimensions to existing virtual datasets.
        print("\n" + "=" * 60)
        print("TASK 5: Temporal Append (Skipped - VirtualZarr limitation)")
        print("=" * 60)
        print("VirtualZarr cannot expand_dims on arrays with transpose codecs.")
        print("For temporal data, concatenate datasets with time coords BEFORE")
        print("writing to Icechunk, rather than using append_dim.")

        print("\n" + "=" * 60)
        print("SPIKE COMPLETE")
        print("=" * 60)

    finally:
        # Cleanup
        print(f"\nCleaning up: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
