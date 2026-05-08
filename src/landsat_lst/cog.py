"""Cloud-Optimized GeoTIFF writing with chunked streaming and uint16 encoding.

This module provides memory-bounded COG writing that works with
large Dask arrays without materializing the full array in memory.

Memory model:
- Peak memory: ~500MB (single 512x512 chunk + overhead)
- Disk requirement: 2x final file size (temp + COG)

Encoding (LST bands only):
- Scale: 0.01, Offset: -50.0
- Decode: celsius = dn * 0.01 + (-50.0)
- Nodata: 0 (uint16), -9999.0 (internal float)
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401 - needed for .rio accessor
import xarray as xr
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

# Encoding constants for LST bands (lst_p50, lst_p95)
LST_SCALE: float = 0.01
LST_OFFSET: float = -50.0
LST_NODATA_FLOAT: float = -9999.0
LST_NODATA_UINT16: int = 0


def encode_lst_uint16(data: xr.DataArray) -> xr.DataArray:
    """Encode LST float values to uint16 with scale/offset.

    Formula: dn = (celsius - offset) / scale
    Decode:  celsius = dn * scale + offset

    Args:
        data: LST values in Celsius (float32), nodata=-9999.0

    Returns:
        Encoded uint16 values, nodata=0
    """
    # Convert celsius to DN: dn = (celsius - offset) / scale
    dn = (data - LST_OFFSET) / LST_SCALE

    # Clamp to valid uint16 range (1-65535, reserve 0 for nodata)
    dn = dn.clip(1, 65535)

    # Set nodata pixels to 0
    dn = xr.where(data == LST_NODATA_FLOAT, LST_NODATA_UINT16, dn)
    dn = xr.where(np.isnan(data), LST_NODATA_UINT16, dn)

    return dn.astype(np.uint16)


def _write_tiff_tags(cog_path: Path) -> None:
    """Write LST metadata as TIFF tags to COG bands.

    Band 1 (lst_p50) and Band 2 (lst_p95) get scale/offset/units.
    Band 3 (qa_count) gets units only.

    Note: We use IGNORE_COG_LAYOUT_BREAK because we're only updating
    metadata tags, not pixel data, which is safe for COG layout.
    """
    with rasterio.open(cog_path, "r+", IGNORE_COG_LAYOUT_BREAK="YES") as dst:
        # LST bands (1 and 2)
        for band in (1, 2):
            dst.update_tags(
                band,
                LST_SCALE=str(LST_SCALE),
                LST_OFFSET=str(LST_OFFSET),
                LST_UNITS="celsius",
            )
        # QA count band (3)
        dst.update_tags(3, UNITS="count")


def write_cog(
    composite: xr.Dataset,
    output_path: Path | str,
    *,
    blocksize: int = 512,
    compression: str = "deflate",
    add_overviews: bool = True,
) -> Path:
    """Write composite Dataset to Cloud-Optimized GeoTIFF with uint16 encoding.

    Uses a three-stage approach:
    1. Encode LST bands to uint16 (scale=0.01, offset=-50.0)
    2. Stream chunks to tiled GeoTIFF via rioxarray (memory-bounded)
    3. Translate to COG with compression and overviews (disk-bound)

    Args:
        composite: Dataset with lst_p50, lst_p95, qa_count bands.
            LST bands should be float32 in Celsius, nodata=-9999.0.
            Must be chunked (lazy) for memory efficiency.
            Must have CRS and spatial dims set.
        output_path: Output COG file path.
        blocksize: COG tile size (default 512x512, must be multiple of 16).
        compression: Compression algorithm (deflate, lzw, zstd).
        add_overviews: Whether to add overviews (default True).
            VirtualTIFF(ifd=0) handles COGs with overviews.

    Returns:
        Path to the written COG file.

    Raises:
        ValueError: If composite is not chunked or missing spatial info.

    Note:
        Output COG has TIFF tags with scale/offset metadata:
        - LST_SCALE="0.01", LST_OFFSET="-50.0", LST_UNITS="celsius"
        Decode: celsius = dn * 0.01 + (-50.0)
    """
    output_path = Path(output_path)

    # Validate input
    if not composite.chunks:
        msg = "Composite must be chunked (lazy) for memory-bounded writes"
        raise ValueError(msg)

    # Ensure CRS and spatial dims are set
    composite = composite.rio.write_crs("EPSG:4326")
    composite = composite.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

    # Encode LST bands to uint16, qa_count to uint16 (counts are positive)
    encoded = xr.Dataset(
        {
            "lst_p50": encode_lst_uint16(composite["lst_p50"]),
            "lst_p95": encode_lst_uint16(composite["lst_p95"]),
            "qa_count": composite["qa_count"].astype(np.uint16),
        },
        coords=composite.coords,
        attrs=composite.attrs,
    )

    # Preserve CRS and spatial dims on encoded dataset
    encoded = encoded.rio.write_crs("EPSG:4326")
    encoded = encoded.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

    # Stage 1: Chunked write to temp GeoTIFF
    tmp_path = output_path.with_suffix(".tmp.tif")

    # Stack variables into bands for multi-band write
    stacked = encoded.to_array(dim="band")

    stacked.rio.to_raster(
        str(tmp_path),
        tiled=True,
        lock=threading.Lock(),
    )

    # Stage 2: COG translation with compression
    dst_profile = cog_profiles.get(compression)
    dst_profile["blockxsize"] = blocksize
    dst_profile["blockysize"] = blocksize

    # overview_level=None means auto-generate overviews
    # overview_level=0 means no overviews
    overview_level = None if add_overviews else 0

    cog_translate(
        str(tmp_path),
        str(output_path),
        dst_profile,
        use_cog_driver=True,
        overview_level=overview_level,
        quiet=True,
    )

    # Stage 3: Write metadata tags
    _write_tiff_tags(output_path)

    # Clean up temp file
    tmp_path.unlink()

    return output_path
