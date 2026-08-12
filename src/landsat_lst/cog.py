"""Cloud-Optimized GeoTIFF (COG) export for LST composites.

Derives QGIS-ready COGs from a native-resolution composite level:

- ``lst_p95``  -> single-band uint16 COG. GDAL band scale/offset are embedded so
  viewers (QGIS, ``gdalinfo``) auto-decode DN to Celsius: ``degC = DN*0.01 - 50``.
  Fill value 0 is written as nodata.
- ``qa_count`` -> 12-band uint8 COG, one band per calendar month (Jan..Dec), band
  descriptions set accordingly. **No nodata** is set: a value of 0 is meaningful
  (no valid observations that month) and must stay visible for gap diagnosis.

The input is the decoded native level (``uint16`` LST DN + ``uint8`` per-month
counts), e.g. ``xr.open_zarr(store, group=f"{group}/0")``.

Three properties of the writer are load-bearing and easy to regress:

**The intermediate GeoTIFF is streamed, never materialized.** ``rio.to_raster``
only routes a dask array through ``dask.array.store`` when ``lock`` is truthy
(``rioxarray/raster_writer.py:315``); with the default ``lock=None`` it calls
``.values`` and pulls the whole array into RAM — about 3.9 GB for a production
12-band QA tile. ``_write_intermediate`` therefore always passes a real lock,
and compresses the intermediate so a worker's scratch disk survives too.

**Exact per-band statistics are embedded in the TIFF itself.** The Portolan
validator (PTL-DAT-009/010) requires the five ``STATISTICS_*`` keys and reads
them with PAM disabled, so a ``.aux.xml`` sidecar does not count. They are
computed in one windowed pass over the intermediate and written as band tags,
which ``cog_translate`` forwards **only** when ``forward_band_tags=True``
(default ``False`` silently drops them; ``rio_cogeo/cogeo.py:397``). Per GDAL
convention the statistics describe raw DN, not decoded Celsius, and they are
exact — hence no ``STATISTICS_APPROXIMATE`` tag.

**Overviews use ``average`` on both products, including LST.** That is not an
oversight about nodata: it was measured. A half-fill/half-constant fixture
translated through ``cog_translate(nodata=0, overview_resampling="average")``
decodes to exactly the constant temperature at every overview level, and
all-fill regions stay at DN 0 — GDAL's average resampling skips nodata rather
than averaging it in. ``nearest`` is actively worse here, since it can latch
onto the fill pixel and report -50 C. ``tests/unit/test_cog_stats.py`` pins this
behaviour, so a GDAL upgrade that changed it would fail the suite rather than
quietly cool the pyramid. For ``qa_count`` averaging is the plain correct
semantic anyway: zeros are genuine observation counts.
"""

from __future__ import annotations

import calendar
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
import rioxarray  # noqa: F401 - needed for .rio accessor
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles

from landsat_lst.encoding import LST_FILL_VALUE, LST_OFFSET, LST_SCALE

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import xarray as xr

# Month names for qa_count band descriptions (index 1..12).
_MONTH_NAMES = tuple(calendar.month_name[m] for m in range(1, 13))

# Matches the blocking of ``cog_profiles.get("deflate")``, so the intermediate's
# windows line up with the output's tiles.
_BLOCKSIZE = 512

# Empirically verified to honour nodata for both products; see module docstring.
_OVERVIEW_RESAMPLING = "average"

_STATISTIC_KEYS = (
    "STATISTICS_MINIMUM",
    "STATISTICS_MAXIMUM",
    "STATISTICS_MEAN",
    "STATISTICS_STDDEV",
    "STATISTICS_VALID_PERCENT",
)


def _prep(da: xr.DataArray) -> xr.DataArray:
    """Attach CRS + spatial dims so rioxarray can write a georeferenced raster."""
    return da.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")


@contextmanager
def _scratch_tif(prefix: str) -> Iterator[Path]:
    """Yield a path for the intermediate GeoTIFF, removing its directory after."""
    scratch = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield scratch / "src.tif"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _write_intermediate(da: xr.DataArray, src_tif: Path) -> None:
    """Write the pre-COG GeoTIFF, streaming block by block for dask inputs.

    The lock is what selects the streaming path: without it rioxarray materializes
    the full array in memory before writing anything.
    """
    da.rio.to_raster(
        src_tif,
        lock=threading.Lock(),
        tiled=True,
        blockxsize=_BLOCKSIZE,
        blockysize=_BLOCKSIZE,
        compress="deflate",
    )


def _dataset_tags(attrs: Mapping[str, Any]) -> dict[str, str]:
    """Render dataset attrs (tile, year, window, scene_count, ...) as TIFF tags."""
    return {str(key): str(value) for key, value in attrs.items() if value is not None}


def _statistics_tags(
    *,
    valid: float,
    total: int,
    total_sum: float,
    total_sumsq: float,
    minimum: float | None,
    maximum: float | None,
) -> dict[str, str]:
    """Format accumulated moments as GDAL ``STATISTICS_*`` tag values."""
    if valid == 0 or minimum is None or maximum is None:
        # An all-nodata band has no statistics to report, but the keys are
        # mandatory, so record the emptiness explicitly via valid_percent.
        return dict.fromkeys(_STATISTIC_KEYS, "0")
    mean = total_sum / valid
    # Guard against a tiny negative from floating-point cancellation.
    variance = max(total_sumsq / valid - mean * mean, 0.0)
    values = (minimum, maximum, mean, variance**0.5, 100.0 * valid / total)
    return {key: repr(float(value)) for key, value in zip(_STATISTIC_KEYS, values, strict=True)}


def _band_statistics(
    src: rasterio.io.DatasetWriter, bidx: int, nodata: float | None
) -> dict[str, str]:
    """Compute exact statistics for one band in a single windowed pass.

    Accumulators are float64 so a full 18,000 x 18,000 uint16 band sums without
    loss. Statistics are of raw DN, per GDAL convention.
    """
    valid = total_sum = total_sumsq = 0.0
    total = 0
    minimum: float | None = None
    maximum: float | None = None
    for _, window in src.block_windows(1):
        block = src.read(bidx, window=window).astype(np.float64)
        total += block.size
        if nodata is not None:
            block = block[block != nodata]
        if block.size == 0:
            continue
        valid += block.size
        total_sum += float(block.sum())
        total_sumsq += float(np.square(block).sum())
        block_min, block_max = float(block.min()), float(block.max())
        minimum = block_min if minimum is None else min(minimum, block_min)
        maximum = block_max if maximum is None else max(maximum, block_max)
    return _statistics_tags(
        valid=valid,
        total=total,
        total_sum=total_sum,
        total_sumsq=total_sumsq,
        minimum=minimum,
        maximum=maximum,
    )


def _embed_statistics(src: rasterio.io.DatasetWriter, nodata: float | None) -> None:
    """Attach exact ``STATISTICS_*`` band tags to an open, already-written raster."""
    for bidx in range(1, src.count + 1):
        src.update_tags(bidx, **_band_statistics(src, bidx, nodata))


def _to_cog(src_tif: Path, cog_path: Path, *, nodata: float | None) -> None:
    """Translate a plain GeoTIFF to a validated deflate COG with average overviews."""
    cog_path.parent.mkdir(parents=True, exist_ok=True)
    cog_translate(
        str(src_tif),
        str(cog_path),
        cog_profiles.get("deflate"),
        overview_resampling=_OVERVIEW_RESAMPLING,
        nodata=nodata,
        # Without this the STATISTICS_* band tags are dropped on the floor.
        forward_band_tags=True,
        quiet=True,
    )
    is_valid, errors, _warnings = cog_validate(str(cog_path))
    if not is_valid:
        msg = f"output is not a valid COG: {cog_path} errors={errors}"
        raise RuntimeError(msg)


def export_lst_cog(native: xr.Dataset, cog_path: Path) -> Path:
    """Write the single-band ``lst_p95`` COG (uint16 DN with scale/offset)."""
    da = _prep(native["lst_p95"]).rio.write_nodata(LST_FILL_VALUE)
    with _scratch_tif("lst_cog_") as src_tif:
        _write_intermediate(da, src_tif)
        with rasterio.open(src_tif, "r+") as src:
            # Embed GDAL band scale/offset so viewers auto-decode DN -> Celsius.
            src.scales = (LST_SCALE,)
            src.offsets = (LST_OFFSET,)
            src.update_tags(**_dataset_tags(native.attrs))
            _embed_statistics(src, nodata=LST_FILL_VALUE)
        _to_cog(src_tif, cog_path, nodata=LST_FILL_VALUE)
    return cog_path


def export_qa_cog(native: xr.Dataset, cog_path: Path) -> Path:
    """Write the 12-band ``qa_count`` COG (uint8, one band per calendar month)."""
    qa = _prep(native["qa_count"])
    if "month" in qa.dims:
        qa = qa.transpose("month", "latitude", "longitude")
    with _scratch_tif("qa_cog_") as src_tif:
        _write_intermediate(qa, src_tif)  # 3D -> one band per month
        with rasterio.open(src_tif, "r+") as src:
            # Label each band with its month so QGIS shows Jan..Dec, not Band 1..12.
            for band_idx in range(1, src.count + 1):
                src.set_band_description(band_idx, _MONTH_NAMES[band_idx - 1])
            src.update_tags(**_dataset_tags(native.attrs))
            # Every pixel counts: 0 observations is data, not absence of data.
            _embed_statistics(src, nodata=None)
        # No nodata: 0 = "no valid obs this month" is the signal we want to see.
        _to_cog(src_tif, cog_path, nodata=None)
    return cog_path


def cog_export(native: xr.Dataset, lst_cog: Path, qa_cog: Path) -> tuple[Path, Path]:
    """Export both LST and per-month QA COGs from a native composite level.

    Args:
        native: Decoded native-resolution level with ``lst_p95`` (uint16 DN) and
            ``qa_count`` (uint8, dims ``(month, latitude, longitude)``). Any
            dataset attrs (``tile``, ``year``, ``window``, ``scene_count``) are
            carried into both COGs as dataset-level TIFF tags.
        lst_cog: Output path for the single-band LST COG.
        qa_cog: Output path for the 12-band monthly QA COG.

    Returns:
        ``(lst_cog, qa_cog)`` paths written.
    """
    return export_lst_cog(native, lst_cog), export_qa_cog(native, qa_cog)
