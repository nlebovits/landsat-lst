"""Zarr writing with uint16 encoding for LST composites.

This module provides Zarr writes for LST composite data, supporting both:
- Plain Zarr stores (local filesystem or S3)
- Icechunk sessions (versioned storage with commits)

Memory model:
- Chunked writes via xarray/zarr (memory-bounded)
- No intermediate files required

Encoding (LST bands only):
- Scale: 0.01, Offset: -50.0
- Decode: celsius = dn * 0.01 + (-50.0)
- Fill value: 0 (uint16)

See ADR-003 for architecture rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import rioxarray  # noqa: F401 - needed for .rio accessor
import xarray as xr
from pyproj import CRS

if TYPE_CHECKING:
    import icechunk as ic

# Encoding constants for LST bands (lst_p50, lst_p95)
LST_SCALE: float = 0.01
LST_OFFSET: float = -50.0
LST_NODATA_FLOAT: float = -9999.0
LST_FILL_VALUE: int = 0

# Zarr chunking (500x500 divides 18,500 evenly = 37 chunks)
DEFAULT_CHUNKS: tuple[int, int] = (500, 500)

# Type alias for output target
OutputTarget = Union[Path, str, "ic.Session"]


def encode_lst_uint16(data: xr.DataArray) -> xr.DataArray:
    """Encode LST float values to uint16 with scale/offset.

    Formula: dn = (celsius - offset) / scale
    Decode:  celsius = dn * scale + offset

    Args:
        data: LST values in Celsius (float32), nodata=-9999.0

    Returns:
        Encoded uint16 values, fill_value=0
    """
    # Convert celsius to DN: dn = (celsius - offset) / scale
    dn = (data - LST_OFFSET) / LST_SCALE

    # Clamp to valid uint16 range (1-65535, reserve 0 for fill value)
    dn = dn.clip(1, 65535)

    # Set nodata pixels to fill value (0)
    dn = xr.where(data == LST_NODATA_FLOAT, LST_FILL_VALUE, dn)
    dn = xr.where(np.isnan(data), LST_FILL_VALUE, dn)

    return dn.astype(np.uint16)


def _add_zarr_metadata(ds: xr.Dataset) -> xr.Dataset:
    """Add metadata attributes for Zarr/GDAL compatibility.

    Uses non-CF attribute names (lst_scale_factor, lst_add_offset) to
    prevent xarray from auto-decoding on read. Standard CF names
    (scale_factor, add_offset) are consumed by xarray and stripped.

    Adds _CRS attribute with WKT for GDAL Zarr driver compatibility.
    """
    # Dataset-level attributes
    ds.attrs["_CRS"] = CRS.from_epsg(4326).to_wkt()
    ds.attrs["crs"] = "EPSG:4326"
    ds.attrs["title"] = "Landsat LST Annual Composite"
    ds.attrs["institution"] = "Radiant Earth"

    # LST band attributes (non-CF names to preserve on read)
    for var in ["lst_p50", "lst_p95"]:
        if var in ds:
            ds[var].attrs.update(
                {
                    "lst_scale_factor": LST_SCALE,
                    "lst_add_offset": LST_OFFSET,
                    "units": "DN (decode: celsius = dn * 0.01 + (-50.0))",
                    "long_name": "Land Surface Temperature"
                    if var == "lst_p50"
                    else "Land Surface Temperature 95th Percentile",
                    "valid_min": 1,
                    "valid_max": 65535,
                }
            )

    # QA count attributes
    if "qa_count" in ds:
        ds["qa_count"].attrs.update(
            {
                "units": "count",
                "long_name": "Number of valid observations",
            }
        )

    return ds


def write_zarr(
    composite: xr.Dataset,
    output: OutputTarget,
    *,
    chunks: tuple[int, int] = DEFAULT_CHUNKS,
    group: str | None = None,
) -> str:
    """Write composite Dataset to Zarr store with uint16 encoding.

    Supports two output modes:
    1. Path/str: Write to plain Zarr store (local or S3)
    2. Icechunk Session: Write to session.store with group path

    Args:
        composite: Dataset with lst_p50, lst_p95, qa_count variables.
            LST variables should be float32 in Celsius, nodata=-9999.0.
            Must have CRS and spatial dims set.
        output: Output path (Path/str) OR Icechunk Session.
        chunks: Chunk size for spatial dimensions (default 500x500).
        group: Zarr group path (required when output is Icechunk Session).

    Returns:
        Path/URL to written Zarr store, or group path for Icechunk.

    Raises:
        ValueError: If composite is missing required variables.
        ValueError: If group is not provided for Icechunk session.
    """
    # Validate required variables
    required = {"lst_p50", "lst_p95", "qa_count"}
    missing = required - set(composite.data_vars)
    if missing:
        msg = f"Composite missing required variables: {missing}"
        raise ValueError(msg)

    # Ensure CRS and spatial dims are set
    composite = composite.rio.write_crs("EPSG:4326")
    composite = composite.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

    # Encode LST bands to uint16, qa_count stays uint16
    encoded = xr.Dataset(
        {
            "lst_p50": encode_lst_uint16(composite["lst_p50"]),
            "lst_p95": encode_lst_uint16(composite["lst_p95"]),
            "qa_count": composite["qa_count"].astype(np.uint16),
        },
        coords=composite.coords,
        attrs=composite.attrs,
    )

    # Preserve CRS and spatial dims
    encoded = encoded.rio.write_crs("EPSG:4326")
    encoded = encoded.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

    # Add metadata attributes
    encoded = _add_zarr_metadata(encoded)

    # Rechunk for optimal Zarr access
    chunk_dict = {"latitude": chunks[0], "longitude": chunks[1]}
    encoded = encoded.chunk(chunk_dict)

    # Set up encoding
    encoding = {}
    for var in encoded.data_vars:
        encoding[var] = {
            "chunks": chunks,
        }

    # Write to appropriate target
    if isinstance(output, (Path, str)):
        # Write to plain Zarr path
        output_path = Path(output) if isinstance(output, str) else output
        encoded.to_zarr(
            str(output_path),
            mode="w",
            consolidated=True,
            encoding=encoding,
        )
        return str(output_path)
    else:
        # Icechunk session - must use icechunk.xarray.to_icechunk for Dask arrays
        # Regular to_zarr() fails because sessions can't be pickled to workers
        if group is None:
            msg = "group parameter required when writing to Icechunk session"
            raise ValueError(msg)

        from icechunk.xarray import to_icechunk  # noqa: PLC0415

        to_icechunk(
            encoded,
            output,  # type: ignore[arg-type] - Session type
            group=group,
            mode="w",
        )
        return group
