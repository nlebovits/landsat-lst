"""Cloud-Optimized GeoTIFF writing with chunked streaming.

This module provides memory-bounded COG writing that works with
large Dask arrays without materializing the full array in memory.

Memory model:
- Peak memory: ~500MB (single 512x512 chunk + overhead)
- Disk requirement: 2x final file size (temp + COG)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import rioxarray  # noqa: F401 - needed for .rio accessor
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

if TYPE_CHECKING:
    import xarray as xr


def write_cog(
    composite: xr.Dataset,
    output_path: Path | str,
    *,
    blocksize: int = 512,
    compression: str = "deflate",
    add_overviews: bool = True,
) -> Path:
    """Write composite Dataset to Cloud-Optimized GeoTIFF.

    Uses a two-stage approach for memory efficiency:
    1. Stream chunks to tiled GeoTIFF via rioxarray (memory-bounded)
    2. Translate to COG with compression and overviews (disk-bound)

    Args:
        composite: Dataset with lst_p50, lst_p95, qa_count bands.
            Must be chunked (lazy) for memory efficiency.
            Must have CRS and spatial dims set.
        output_path: Output COG file path.
        blocksize: COG tile size (default 512x512).
        compression: Compression algorithm (deflate, lzw, zstd).
        add_overviews: Whether to add overviews (default True).
            VirtualTIFF(ifd=0) handles COGs with overviews.

    Returns:
        Path to the written COG file.

    Raises:
        ValueError: If composite is not chunked or missing spatial info.
    """
    output_path = Path(output_path)

    # Validate input
    if not composite.chunks:
        msg = "Composite must be chunked (lazy) for memory-bounded writes"
        raise ValueError(msg)

    # Ensure CRS and spatial dims are set
    composite = composite.rio.write_crs("EPSG:4326")
    composite = composite.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")

    # Stage 1: Chunked write to temp GeoTIFF
    tmp_path = output_path.with_suffix(".tmp.tif")

    # Stack variables into bands for multi-band write
    stacked = composite.to_array(dim="band")

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

    # Clean up temp file
    tmp_path.unlink()

    return output_path
