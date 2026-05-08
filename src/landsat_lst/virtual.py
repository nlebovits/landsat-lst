"""Virtual Zarr datacube creation over Cloud-Optimized GeoTIFFs.

This module provides functions to create a virtual Zarr datacube from COGs
using VirtualZarr and Icechunk. The virtual datacube enables efficient
xarray access to the entire dataset without duplicating data:

    import xarray as xr
    ds = xr.open_zarr("icechunk://source.coop/radiant-earth/landsat-lst")
    ds.lst_p50.sel(latitude=slice(45, 40), longitude=slice(-75, -70)).mean()

Architecture:
- COGs remain the source of truth (stored on S3)
- Icechunk stores virtual references (byte ranges into COGs)
- xarray reads data via byte-range requests

See ADR-002 for design rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from virtualizarr import open_virtual_dataset

from landsat_lst.cog import LST_OFFSET, LST_SCALE
from landsat_lst.tiling import parse_tile_name

if TYPE_CHECKING:
    from landsat_lst.storage import StorageBackend

try:
    from obspec_utils.registry import ObjectStoreRegistry
    from virtual_tiff import VirtualTIFF

    HAS_VIRTUAL_TIFF = True
except ImportError:
    HAS_VIRTUAL_TIFF = False
    ObjectStoreRegistry: Any = None
    VirtualTIFF: Any = None

import icechunk as ic


def create_registry(storage: StorageBackend) -> ObjectStoreRegistry:
    """Create ObjectStoreRegistry for COG access.

    Args:
        storage: Storage backend for COG location.

    Returns:
        ObjectStoreRegistry configured for the storage backend.

    Raises:
        ImportError: If virtual-tiff is not installed.
    """
    if not HAS_VIRTUAL_TIFF:
        msg = "virtual-tiff is required for virtualization: pip install virtual-tiff"
        raise ImportError(msg)

    registry = ObjectStoreRegistry()
    store, prefix = storage.virtual_chunk_store_and_prefix()
    registry.register(prefix, store)
    return registry


def open_tile_virtual(
    cog_url: str,
    tile_name: str,
    registry: ObjectStoreRegistry,
) -> xr.Dataset:
    """Open a single COG as virtual dataset with proper coords and names.

    Args:
        cog_url: URL or path to the COG file.
        tile_name: Tile name in format N40W075.
        registry: ObjectStoreRegistry for COG access.

    Returns:
        Virtual xarray Dataset with named variables and coordinates.
    """
    vds = open_virtual_dataset(
        cog_url,
        parser=VirtualTIFF(ifd=0),
        registry=registry,
    )

    vds = vds.rename({"0": "lst_p50", "1": "lst_p95", "2": "qa_count"})
    vds = assign_tile_coords(vds, tile_name)
    vds = _add_variable_attrs(vds)

    return vds


def assign_tile_coords(vds: xr.Dataset, tile_name: str) -> xr.Dataset:
    """Add lat/lon coordinates based on tile name.

    Args:
        vds: Virtual dataset with y/x dimensions.
        tile_name: Tile name in format N40W075.

    Returns:
        Dataset with latitude and longitude coordinates assigned.

    Note:
        Tiles are 5 x 5 degrees. Latitude runs north-to-south (descending),
        longitude runs west-to-east (ascending).
    """
    tile_id = parse_tile_name(tile_name)
    lat_start = tile_id.lat
    lon_start = tile_id.lon

    n_lat, n_lon = vds.sizes["y"], vds.sizes["x"]

    lat = np.linspace(lat_start + 5, lat_start, n_lat, endpoint=False)
    lon = np.linspace(lon_start, lon_start + 5, n_lon, endpoint=False)

    return vds.assign_coords(latitude=("y", lat), longitude=("x", lon))


def _add_variable_attrs(vds: xr.Dataset) -> xr.Dataset:
    """Add CF-compliant attributes to variables.

    Args:
        vds: Virtual dataset with LST variables.

    Returns:
        Dataset with scale_factor, add_offset, units, and long_name attrs.
    """
    vds["lst_p50"].attrs.update(
        {
            "scale_factor": LST_SCALE,
            "add_offset": LST_OFFSET,
            "units": "celsius",
            "long_name": "Land Surface Temperature (median)",
        }
    )

    vds["lst_p95"].attrs.update(
        {
            "scale_factor": LST_SCALE,
            "add_offset": LST_OFFSET,
            "units": "celsius",
            "long_name": "Land Surface Temperature (95th percentile)",
        }
    )

    vds["qa_count"].attrs.update(
        {
            "units": "count",
            "long_name": "Valid observation count",
        }
    )

    return vds


def create_icechunk_repo(
    storage: StorageBackend,
    *,
    create: bool = True,
) -> ic.Repository:
    """Create or open Icechunk repository with virtual chunk access.

    Args:
        storage: Storage backend for Icechunk location.
        create: If True, create new repo. If False, open existing.

    Returns:
        Icechunk Repository configured for virtual chunk access.
    """
    icechunk_storage = storage.icechunk_storage()
    container_prefix = storage.cog_container_prefix()

    config = ic.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        ic.VirtualChunkContainer(
            url_prefix=container_prefix,
            store=storage.virtual_chunk_store(),
            name=container_prefix,
        )
    )

    if create:
        return ic.Repository.create(
            icechunk_storage,
            config=config,
        )
    return ic.Repository.open(
        icechunk_storage,
        config=config,
    )


def create_virtual_datacube(
    tile_paths: list[str],
    tile_names: list[str],
    year: int,
    storage: StorageBackend,
) -> str:
    """Create virtual datacube from COGs for a single year.

    Args:
        tile_paths: List of COG URLs/paths.
        tile_names: List of tile names (e.g., ["N40W075", "N41W075"]).
        year: Year for the time coordinate.
        storage: Storage backend for Icechunk and COG access.

    Returns:
        Icechunk snapshot ID for the commit.

    Raises:
        ValueError: If tile_paths and tile_names have different lengths.
    """
    if len(tile_paths) != len(tile_names):
        msg = f"tile_paths ({len(tile_paths)}) and tile_names ({len(tile_names)}) must have same length"
        raise ValueError(msg)

    registry = create_registry(storage)

    vds_list = [
        open_tile_virtual(path, name, registry)
        for path, name in zip(tile_paths, tile_names, strict=True)
    ]

    combined = xr.concat(vds_list, dim="y", combine_attrs="override")

    combined = combined.assign_coords(time=pd.Timestamp(f"{year}-01-01"))
    combined = xr.concat([combined], dim="time")

    combined.attrs["title"] = "Landsat Land Surface Temperature Annual Composites"
    combined.attrs["institution"] = "Radiant Earth"
    combined.attrs["source"] = "Landsat Collection 2 Level-2"
    combined.attrs["lst_scale"] = LST_SCALE
    combined.attrs["lst_offset"] = LST_OFFSET

    repo = create_icechunk_repo(storage, create=True)

    session = repo.writable_session("main")
    combined.virtualizarr.to_icechunk(session.store)
    snapshot_id = session.commit(f"Add {year} virtual datacube ({len(tile_paths)} tiles)")

    return snapshot_id


def append_year_to_datacube(
    tile_paths: list[str],
    tile_names: list[str],
    year: int,
    storage: StorageBackend,
) -> str:
    """Append a new year to existing virtual datacube.

    Args:
        tile_paths: List of COG URLs/paths for the new year.
        tile_names: List of tile names.
        year: Year to append.
        storage: Storage backend.

    Returns:
        Icechunk snapshot ID for the commit.
    """
    registry = create_registry(storage)

    vds_list = [
        open_tile_virtual(path, name, registry)
        for path, name in zip(tile_paths, tile_names, strict=True)
    ]

    combined = xr.concat(vds_list, dim="y", combine_attrs="override")
    combined = combined.assign_coords(time=pd.Timestamp(f"{year}-01-01"))
    combined = xr.concat([combined], dim="time")

    repo = create_icechunk_repo(storage, create=False)

    session = repo.writable_session("main")
    existing = xr.open_zarr(session.store, consolidated=False)
    appended = xr.concat([existing, combined], dim="time")
    appended.virtualizarr.to_icechunk(session.store)
    snapshot_id = session.commit(f"Append {year} ({len(tile_paths)} tiles)")

    return snapshot_id
