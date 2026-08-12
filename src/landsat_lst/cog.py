"""Cloud-Optimized GeoTIFF (COG) export for LST composites.

Derives QGIS-ready COGs from a native-resolution composite level:

- ``lst_p95``  -> single-band uint16 COG. GDAL band scale/offset are embedded so
  viewers (QGIS, ``gdalinfo``) auto-decode DN to Celsius: ``degC = DN*0.01 - 50``.
  Fill value 0 is written as nodata.
- ``qa_count`` -> 12-band uint8 COG, one band per calendar month (Jan..Dec), band
  descriptions set accordingly. **No nodata** is set: a value of 0 is meaningful
  (no valid observations that month) and must stay visible for gap diagnosis.

This promotes the one-off logic in ``scripts/smoke_small_tile_cog.py`` into a
reusable function. The input is the decoded native level (``uint16`` LST DN +
``uint8`` per-month counts), e.g. ``xr.open_zarr(store, group=f"{group}/0")``.
"""

from __future__ import annotations

import calendar
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import rasterio
import rioxarray  # noqa: F401 - needed for .rio accessor
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles

from landsat_lst.encoding import LST_OFFSET, LST_SCALE

if TYPE_CHECKING:
    import xarray as xr

# Month names for qa_count band descriptions (index 1..12).
_MONTH_NAMES = tuple(calendar.month_name[m] for m in range(1, 13))


def _prep(da: xr.DataArray) -> xr.DataArray:
    """Attach CRS + spatial dims so rioxarray can write a georeferenced raster."""
    return da.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")


def _to_cog(src_tif: Path, cog_path: Path, *, nodata: float | None) -> None:
    """Translate a plain GeoTIFF to a validated deflate COG with average overviews."""
    cog_path.parent.mkdir(parents=True, exist_ok=True)
    cog_translate(
        str(src_tif),
        str(cog_path),
        cog_profiles.get("deflate"),
        overview_resampling="average",
        nodata=nodata,
        quiet=True,
    )
    is_valid, errors, _warnings = cog_validate(str(cog_path))
    if not is_valid:
        msg = f"output is not a valid COG: {cog_path} errors={errors}"
        raise RuntimeError(msg)


def export_lst_cog(native: xr.Dataset, cog_path: Path) -> Path:
    """Write the single-band ``lst_p95`` COG (uint16 DN with scale/offset)."""
    da = _prep(native["lst_p95"]).rio.write_nodata(0)
    scratch = Path(tempfile.mkdtemp(prefix="lst_cog_"))
    try:
        src_tif = scratch / "src.tif"
        da.rio.to_raster(src_tif)
        # Embed GDAL band scale/offset so viewers auto-decode DN -> Celsius.
        with rasterio.open(src_tif, "r+") as src:
            src.scales = (LST_SCALE,)
            src.offsets = (LST_OFFSET,)
        _to_cog(src_tif, cog_path, nodata=0)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return cog_path


def export_qa_cog(native: xr.Dataset, cog_path: Path) -> Path:
    """Write the 12-band ``qa_count`` COG (uint8, one band per calendar month)."""
    qa = _prep(native["qa_count"])
    if "month" in qa.dims:
        qa = qa.transpose("month", "latitude", "longitude")
    scratch = Path(tempfile.mkdtemp(prefix="qa_cog_"))
    try:
        src_tif = scratch / "src.tif"
        qa.rio.to_raster(src_tif)  # 3D -> one band per month
        # Label each band with its month so QGIS shows Jan..Dec, not Band 1..12.
        with rasterio.open(src_tif, "r+") as src:
            for band_idx in range(1, src.count + 1):
                src.set_band_description(band_idx, _MONTH_NAMES[band_idx - 1])
        # No nodata: 0 = "no valid obs this month" is the signal we want to see.
        _to_cog(src_tif, cog_path, nodata=None)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return cog_path


def cog_export(native: xr.Dataset, lst_cog: Path, qa_cog: Path) -> tuple[Path, Path]:
    """Export both LST and per-month QA COGs from a native composite level.

    Args:
        native: Decoded native-resolution level with ``lst_p95`` (uint16 DN) and
            ``qa_count`` (uint8, dims ``(month, latitude, longitude)``).
        lst_cog: Output path for the single-band LST COG.
        qa_cog: Output path for the 12-band monthly QA COG.

    Returns:
        ``(lst_cog, qa_cog)`` paths written.
    """
    return export_lst_cog(native, lst_cog), export_qa_cog(native, qa_cog)
