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

Output is a GeoZarr multiscale pyramid: native resolution in level group ``0`` plus
coarsened overview groups, with GeoZarr proj/spatial/multiscales metadata on the parent.
See ADR-003 (direct Zarr + Icechunk) and ADR-004 (GeoZarr multiscale overviews).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import rioxarray  # noqa: F401 - needed for .rio accessor
import xarray as xr
import zarr
from pyproj import CRS
from zarr.codecs import BloscCname, BloscCodec

from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    import icechunk as ic

# Permanent UUID of the GeoZarr multiscales convention (zarr-conventions/multiscales)
MULTISCALES_UUID = "d35379db-88df-4056-af3a-620245f8e347"

# Encoding constants for LST bands (lst_p95)
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
    ds.attrs["title"] = "Landsat LST P95 Composite"
    ds.attrs["institution"] = "Radiant Earth"

    # LST band attributes (non-CF names to preserve on read)
    if "lst_p95" in ds:
        ds["lst_p95"].attrs.update(
            {
                "lst_scale_factor": LST_SCALE,
                "lst_add_offset": LST_OFFSET,
                "units": "DN (decode: celsius = dn * 0.01 + (-50.0))",
                "long_name": "Land Surface Temperature 95th Percentile",
                "valid_min": 1,
                "valid_max": 65535,
            }
        )

    # QA count attributes. qa_count is a 12-month climatology: month M holds the
    # valid-observation count for calendar month M pooled across the window.
    if "qa_count" in ds:
        ds["qa_count"].attrs.update(
            {
                "units": "count",
                "long_name": "Number of valid observations per calendar month",
            }
        )

    # Month coordinate (1..12) for the qa_count climatology.
    if "month" in ds.coords:
        ds["month"].attrs.update(
            {
                "long_name": "calendar month",
                "units": "month_of_year",
            }
        )

    return ds


def _spatial_transform(ds: xr.Dataset) -> list[float]:
    """Affine transform in GeoZarr/rasterio order ``[a, b, c, d, e, f]``.

    Derived from the dataset's own latitude/longitude coordinates via rioxarray,
    so it is correct for both native and coarsened (overview) levels. Requires
    CRS and spatial dims to have been set on ``ds``.
    """
    affine = ds.rio.transform()
    return [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f]


def _geozarr_spatial_attrs(ds: xr.Dataset) -> dict:
    """GeoZarr ``proj`` + ``spatial`` convention attributes for a georeferenced group.

    These are what web viewers (xpublish-tiles, icechunk-js, deck.gl-zarr) read to
    interpret CRS and pixel geolocation. Written alongside the existing GDAL-style
    ``_CRS``/GeoTransform attrs (which GDAL readers use) rather than replacing them.
    """
    return {
        "proj:code": "EPSG:4326",
        "spatial:dimensions": ["latitude", "longitude"],
        "spatial:transform": _spatial_transform(ds),
        "spatial:shape": [int(ds.sizes["latitude"]), int(ds.sizes["longitude"])],
    }


def build_overviews(
    composite: xr.Dataset, factors: Sequence[int]
) -> list[tuple[str, int, xr.Dataset]]:
    """Build multiscale levels (native + coarsened overviews) from a float composite.

    Each overview is derived from the **native** resolution (not the previous level)
    so the block mean is exact rather than a weighted mean-of-means. ``lst_p95`` is
    averaged with fill (``-9999``) and ocean (``NaN``) excluded -- critical, since
    averaging fill DN=0 would otherwise drag overview temperatures toward -50 degC.
    ``qa_count`` is averaged as a float observation count.

    Args:
        composite: Float dataset with ``lst_p95`` (Celsius, nodata=-9999/NaN) and
            ``qa_count``. CRS/spatial dims need not be set yet.
        factors: Downsample factors for overview levels (e.g. ``[4, 16, 64]``).

    Returns:
        Ordered ``[(level_name, scale_factor, level_dataset), ...]`` where level
        ``"0"`` is native (factor 1). ``level_dataset`` keeps float ``lst_p95``
        (NaN for missing) -- encoding to uint16 happens downstream, per level.
    """
    # Treat -9999 nodata as NaN so it is excluded from the block means.
    lst = composite["lst_p95"].where(composite["lst_p95"] != LST_NODATA_FLOAT)
    qa = composite["qa_count"].astype(np.float32)
    src_attrs = dict(composite.attrs)
    native = xr.Dataset({"lst_p95": lst, "qa_count": qa}, coords=composite.coords, attrs=src_attrs)

    levels: list[tuple[str, int, xr.Dataset]] = [("0", 1, native)]
    for i, factor in enumerate(factors, start=1):
        coarsen = native.coarsen(latitude=factor, longitude=factor, boundary="trim")
        coarsened = coarsen.mean(skipna=True)  # ty: ignore[unresolved-attribute]
        coarsened.attrs = src_attrs
        levels.append((str(i), factor, coarsened))
    return levels


def _multiscales_attr(levels: list[tuple[str, int, xr.Dataset]]) -> dict:
    """GeoZarr ``multiscales`` convention layout for the pyramid (set on parent group).

    ``transform.scale`` is the cumulative factor relative to native; every overview
    is ``derived_from`` native (``"0"``), matching :func:`build_overviews`.
    """
    layout = []
    for name, factor, _ds in levels:
        entry: dict = {
            "asset": name,
            "transform": {
                "scale": [float(factor), float(factor)],
                "translation": [0.0, 0.0],
            },
        }
        if factor != 1:
            entry["derived_from"] = "0"
        layout.append(entry)
    return {
        "zarr_conventions": [{"name": "multiscales", "uuid": MULTISCALES_UUID}],
        "multiscales": {"layout": layout, "resampling_method": "average"},
    }


def _encode_level(level: xr.Dataset) -> xr.Dataset:
    """Encode one float level to uint16 with CRS, GDAL, and GeoZarr metadata.

    ``qa_count`` is a per-month climatology ``(month, latitude, longitude)``; it
    arrives as a float mean (from coarsening) and is rounded back to an integer
    observation count stored as ``uint8`` (counts stay well under 255).
    """
    qa = level["qa_count"].round().fillna(0).astype(np.uint8)
    if "month" in qa.dims:
        qa = qa.transpose("month", "latitude", "longitude")
    encoded = xr.Dataset(
        {
            "lst_p95": encode_lst_uint16(level["lst_p95"]),
            "qa_count": qa,
        },
        coords=level.coords,
        attrs=dict(level.attrs),
    )
    encoded = encoded.rio.write_crs("EPSG:4326")
    encoded = encoded.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    encoded = _add_zarr_metadata(encoded)
    # GeoZarr proj/spatial on the level group (written as group attrs by the writer).
    encoded.attrs.update(_geozarr_spatial_attrs(encoded))
    return encoded


def _level_chunks(level: xr.Dataset, chunks: tuple[int, int]) -> tuple[int, int]:
    """Clamp the default chunk size to the level's shape (overviews are small)."""
    ny = int(level.sizes["latitude"])
    nx = int(level.sizes["longitude"])
    return (min(chunks[0], ny), min(chunks[1], nx))


def _chunk_and_encoding(
    encoded: xr.Dataset, lch: tuple[int, int], compressors: tuple
) -> tuple[xr.Dataset, dict]:
    """Chunk a level and build its per-variable Zarr encoding.

    ``qa_count`` may carry a leading ``month`` dim (12-month climatology); it is
    kept in a single chunk and gets a 3-tuple chunk spec, while 2D variables use
    the ``(lat, lon)`` spatial chunk.
    """
    chunk_spec: dict[str, int] = {"latitude": lch[0], "longitude": lch[1]}
    if "month" in encoded.dims:
        chunk_spec["month"] = -1
    encoded = encoded.chunk(chunk_spec)

    encoding = {}
    for var in encoded.data_vars:
        da = encoded[var]
        var_chunks: tuple[int, ...] = (
            (int(da.sizes["month"]), lch[0], lch[1]) if "month" in da.dims else lch
        )
        encoding[var] = {
            "chunks": var_chunks,
            **({"compressors": compressors} if compressors else {}),
        }
    return encoded, encoding


def _compressors() -> tuple:
    """Blosc compressor tuple from settings, or empty if compression is disabled."""
    if settings.compression_level <= 0:
        return ()
    return (
        BloscCodec(
            cname=BloscCname(settings.compression_codec),
            clevel=settings.compression_level,
        ),
    )


def write_zarr(
    composite: xr.Dataset,
    output: OutputTarget,
    *,
    chunks: tuple[int, int] = DEFAULT_CHUNKS,
    group: str | None = None,
    storage_options: dict | None = None,
    factors: Sequence[int] | None = None,
) -> str:
    """Write composite as a GeoZarr multiscale pyramid with uint16 encoding.

    Writes native resolution as level group ``0`` plus one coarsened overview group
    per entry in ``factors`` (e.g. ``1``=4x, ``2``=16x). The parent group carries the
    GeoZarr ``multiscales`` layout and ``proj``/``spatial`` conventions; each level
    group carries its own ``proj``/``spatial`` attrs and GDAL ``_CRS``. All arrays are
    Blosc-compressed (see ``settings.compression_*``).

    Supports two output modes:
    1. Path/str: Write to plain Zarr store (local or S3).
    2. Icechunk Session: Write to ``session.store`` under ``group`` (caller commits).
       For Icechunk, every level + the parent metadata land in the session uncommitted,
       so the caller's single ``session.commit()`` makes the whole pyramid atomic.

    Args:
        composite: Dataset with ``lst_p95`` (float32 Celsius, nodata=-9999.0) and
            ``qa_count`` variables.
        output: Output path (Path/str) OR Icechunk Session.
        chunks: Spatial chunk size, clamped per level (default 500x500).
        group: Zarr group path (required when output is an Icechunk Session).
        storage_options: Optional fsspec options for S3 (plain Zarr path only).
        factors: Overview downsample factors. Defaults to ``settings.pyramid_factors``.

    Returns:
        The store path/URL (plain Zarr) or the parent group path (Icechunk).

    Raises:
        ValueError: If ``composite`` is missing required variables, or if ``group`` is
            not provided for an Icechunk session.
    """
    # Validate required variables
    required = {"lst_p95", "qa_count"}
    missing = required - set(composite.data_vars)
    if missing:
        msg = f"Composite missing required variables: {missing}"
        raise ValueError(msg)

    if factors is None:
        factors = settings.pyramid_factors

    levels = build_overviews(composite, factors)
    compressors = _compressors()

    # Branch on output type up front so the writer/store handles are well-typed.
    if isinstance(output, (Path, str)):
        is_icechunk = False
        output_str = str(output)
        attr_store = output_str
        parent_path = ""
    else:
        if group is None:
            msg = "group parameter required when writing to Icechunk session"
            raise ValueError(msg)
        # to_icechunk (not to_zarr) is required for Dask arrays: sessions can't be
        # pickled to workers.
        from icechunk.xarray import to_icechunk  # noqa: PLC0415

        is_icechunk = True
        attr_store = output.store
        parent_path = group
        output_str = group  # parent group path is the return value for Icechunk

    native_encoded: xr.Dataset | None = None
    for name, _factor, level in levels:
        encoded = _encode_level(level)
        lch = _level_chunks(encoded, chunks)
        encoded, encoding = _chunk_and_encoding(encoded, lch, compressors)
        if name == "0":
            native_encoded = encoded

        if is_icechunk:
            to_icechunk(
                encoded,
                output,  # ty: ignore[invalid-argument-type] - narrowed to Session above
                group=f"{group}/{name}",
                mode="w",
                encoding=encoding,
            )
        else:
            encoded.to_zarr(
                f"{output_str}/{name}",
                mode="w",
                consolidated=True,
                encoding=encoding,
                storage_options=storage_options,
            )

    # Stamp the parent group with the GeoZarr multiscales layout + native proj/spatial.
    # native_encoded is always set: level "0" is the first entry from build_overviews.
    assert native_encoded is not None
    if is_icechunk:
        parent = zarr.open_group(attr_store, path=parent_path, mode="a")
    else:
        parent = zarr.open_group(
            attr_store, path=parent_path, mode="a", storage_options=storage_options
        )
    parent.attrs.update(_multiscales_attr(levels))
    parent.attrs.update(_geozarr_spatial_attrs(native_encoded))
    parent.attrs["title"] = "Landsat LST P95 Composite"
    parent.attrs["institution"] = "Radiant Earth"

    return output_str
