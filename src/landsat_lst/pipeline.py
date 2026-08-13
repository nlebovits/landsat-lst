"""Main ETL pipeline for Landsat LST composites."""

import os
from collections.abc import Callable

import numpy as np
import pystac_client
import structlog
import xarray as xr
from odc.stac import stac_load

from landsat_lst.config import settings
from landsat_lst.encoding import LST_MIN_TRUSTED_C
from landsat_lst.masks import get_land_mask_for_bbox, load_land_polygons
from landsat_lst.models import ProcessingJob
from landsat_lst.normalization import offset_diagnostics, seasonal_debias
from landsat_lst.progress import report_phase
from landsat_lst.qa import apply_qa_mask, convert_to_celsius
from landsat_lst.tiling import geobox_for_bbox

log = structlog.get_logger()

# Planetary Computer URL prefix for conditional signing
_PC_URL_PREFIX = "https://planetarycomputer.microsoft.com"


def _is_planetary_computer() -> bool:
    """Check if using Planetary Computer endpoint."""
    return settings.stac_url.startswith(_PC_URL_PREFIX)


def _configure_requester_pays() -> None:
    """Configure GDAL/rasterio for AWS requester-pays buckets.

    Earth Search serves Landsat data from the usgs-landsat bucket which
    requires requester-pays. These env vars must be set before rasterio
    opens any files.
    """
    os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
    os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "NO")


def query_stac(job: ProcessingJob) -> list:
    """Query STAC catalog for Landsat scenes.

    For Planetary Computer: items are returned UNSIGNED. Azure access is
    handled at read time via refreshable GDAL config (``/vsiaz/`` paths plus a
    timer-refreshed ``AZURE_STORAGE_SAS_TOKEN``); see
    :func:`landsat_lst.azure_auth.enable_pc_azure_refresh`. Baking SAS tokens
    into URLs (``pc.sign_inplace``) is avoided because those tokens expire
    ~45min after the query and would fail mid-compute on long-running tiles.
    For Earth Search: configures requester-pays for S3 access.

    Args:
        job: Processing job with tile and year info.

    Returns:
        List of STAC items matching the query.
    """
    if not _is_planetary_computer():
        _configure_requester_pays()

    catalog = pystac_client.Client.open(settings.stac_url)

    search = catalog.search(
        collections=[settings.collection],
        bbox=job.tile.bbox,
        datetime=job.datetime_range,
        query={
            "eo:cloud_cover": {"lt": settings.max_cloud_cover},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
    )

    return list(search.items())


def load_scenes(
    items: list,
    bbox: tuple[float, float, float, float],
    patch_url: Callable[[str], str] | None = None,
    *,
    fail_on_error: bool = True,
    resolution_factor: int = 1,
) -> xr.Dataset:
    """Load Landsat scenes as an xarray Dataset.

    Args:
        items: List of STAC items to load.
        bbox: Bounding box for spatial subsetting.
        patch_url: Optional per-asset URL transform applied by odc-stac before
            building the load graph. Used for Planetary Computer to rewrite blob
            hrefs to token-free ``/vsiaz/`` paths (auth via GDAL config).
        fail_on_error: If True (default), odc-stac aborts the entire load when
            any single scene read fails. If False, a failed read is filled with
            nodata and the load continues. Set False for large multi-scene
            composites (e.g. multi-year windows), where a transient Azure/S3
            read blip is near-certain across hundreds of scenes and losing a
            few observations is negligible against a P95 over the full stack.
        resolution_factor: Load at ``resolution * factor`` instead of native.
            The source COGs carry internal overviews at [2, 4, 8, 16, 32, 64],
            so a coarser request is served from an overview and reads roughly
            ``factor**2`` fewer bytes. Use powers of two to land on a stored
            level. Intended for per-scene statistics that need no spatial
            detail; never for the composite itself.

    Returns:
        Dataset with thermal and QA bands.
    """
    csize = settings.load_chunk_size

    # Averaging only helps the thermal band, and only when coarsening. qa_pixel
    # is a bitfield and must be sampled, never interpolated -- averaging bit
    # flags yields values that decode to nonsense. (USGS built these overviews
    # with nearest/mode, verified by confirming a decimated read introduces no
    # values absent from the full-resolution vocabulary.) "nearest" for both is
    # odc-stac's own default, so the factor=1 path is unchanged.
    resampling: dict[str, str] | str = (
        {"lwir11": "average", "qa_pixel": "nearest"} if resolution_factor > 1 else "nearest"
    )

    # An explicit geobox rather than crs/resolution/bbox: odc-stac would anchor
    # the grid to this bbox, which is what left neighbouring tiles misregistered
    # by a fraction of a pixel. See ADR-008.
    return stac_load(
        items,
        bands=["lwir11", "qa_pixel"],
        geobox=geobox_for_bbox(bbox, resolution_factor),
        chunks={"time": 10, "latitude": csize, "longitude": csize},
        groupby="solar_day",
        patch_url=patch_url,
        fail_on_error=fail_on_error,
        resampling=resampling,
    )


def _build_land_mask(
    bbox: tuple[float, float, float, float],
    latitude: xr.DataArray,
    longitude: xr.DataArray,
) -> xr.DataArray:
    """Rasterize the Natural Earth land mask onto a grid's exact coordinates.

    ``target_shape`` comes from the loaded array rather than from the bbox, so
    the mask matches whatever grid the caller actually has, including the
    zoomed-out grid used for offset estimation.

    Both rasterio and odc-stac use north-down (descending latitude), so the
    rasterized array needs no flip.
    """
    land_polygons = load_land_polygons()
    land_mask = get_land_mask_for_bbox(
        bbox,
        settings.resolution,
        land_polygons,
        target_shape=(len(latitude), len(longitude)),
    )
    return xr.DataArray(
        land_mask,
        dims=["latitude", "longitude"],
        coords={"latitude": latitude, "longitude": longitude},
    )


def compute_annual_composite(
    data: xr.Dataset,
    *,
    land_mask: xr.DataArray | None = None,
    offset_source: xr.Dataset | None = None,
    offset_land_mask: xr.DataArray | None = None,
) -> xr.Dataset:
    """Compute an LST P95 composite with a per-month observation count.

    The P95 is pooled across *every* time step in ``data``. For a multi-year
    window (``ProcessingJob.end_year`` set) this pools all scenes across all
    years -- the correct way to build a multi-year percentile (never average
    per-year P95s: percentile-of-percentiles is wrong).

    ``qa_count`` is a **12-month climatology**: for month M it is the number of
    valid (cloud-free, QA-passing) observations in month M pooled across all
    years in the window. Dims ``(month, latitude, longitude)``, always 12
    months (missing months filled 0), dtype ``uint8`` (counts stay well under
    255 even for a 5-year window). It counts only observations that reach the
    composite, so scenes dropped by de-striping are excluded.

    When ``settings.destripe`` is on, each scene is shifted by a single
    scene-wide offset before compositing and scenes whose offset is implausible
    are discarded. See ``normalization.seasonal_debias`` and issue #46.

    WARNING: Land masking happens only when ``land_mask`` is supplied. For
    production output that excludes ocean pixels, use process_tile() instead,
    which always supplies one. See issue #26.

    Args:
        data: Dataset with thermal and QA bands across time.
        land_mask: Optional boolean land mask on ``(latitude, longitude)``.
            Applied before de-biasing so per-scene offsets are estimated over
            land only; ocean is thermally stable and would otherwise damp the
            estimate on coastal tiles.

    Returns:
        Dataset with ``lst_p95`` ``(latitude, longitude)`` and ``qa_count``
        ``(month, latitude, longitude)`` bands.

    Note:
        P50 (median) was removed per stakeholder feedback - hot season temps
        (P95) are what matter for urban heat applications. See issue #22.
    """
    masked = apply_qa_mask(data)

    lst = convert_to_celsius(masked["lwir11"])

    if land_mask is not None:
        lst = lst.where(land_mask)

    scenes_kept = None
    if settings.destripe:
        # The offset is one scalar per scene, so it can be estimated from a
        # coarse stack read off the source COGs' overviews. That cuts bytes
        # read, which post-load subsampling cannot do: dask must materialize a
        # whole chunk before discarding most of it.
        source = None
        if offset_source is not None:
            source = convert_to_celsius(apply_qa_mask(offset_source)["lwir11"])
            if offset_land_mask is not None:
                source = source.where(offset_land_mask)

        # Estimating the offsets is the first real compute of the tile, and on a
        # five-year window it runs for many minutes, so the watcher hears about
        # it before it starts rather than after.
        report_phase("destriping")
        lst, offset, keep = seasonal_debias(
            lst,
            max_offset_c=settings.destripe_max_offset_c,
            min_scene_pixels=settings.destripe_min_scene_pixels,
            min_offset_samples=settings.destripe_min_offset_samples,
            offset_source=source,
        )
        diagnostics = offset_diagnostics(offset, keep)
        log.info("destripe_offsets_degC", **diagnostics)
        scenes_kept = int(diagnostics["n_kept"])

    report_phase("compositing", scenes_kept=scenes_kept)

    # notnull() (not ~np.isnan) so the result stays a typed xarray DataArray.
    valid_mask = lst.notnull()

    # Per-calendar-month climatology of valid observations. groupby pools every
    # year in the window into its month bucket; reindex guarantees all 12 months.
    qa_count = (
        valid_mask.groupby("time.month")
        .sum()
        .reindex(month=range(1, 13), fill_value=0)
        .astype(np.uint8)
    )

    # Total valid obs per pixel (across the whole window) gates the P95 fill.
    total_valid = valid_mask.sum(dim="time")

    lst_p95 = lst.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")
    lst_p95 = lst_p95.where(total_valid > 0, settings.nodata)

    # Guard against a pixel that has observations yet still produces a P95 on
    # the encoding floor (DN 0 or DN 1). Writing it would resurface the
    # isolated -49.99 C anomalies of issue #24, so flag it as missing here and
    # keep the composite consistent with the encoded output. Gating on
    # total_valid keeps the existing nodata sentinel distinguishable from a
    # genuine bad retrieval.
    anomalous = (total_valid > 0) & (lst_p95 < LST_MIN_TRUSTED_C)
    lst_p95 = lst_p95.where(~anomalous, settings.nodata)

    return xr.Dataset(
        {
            "lst_p95": lst_p95.astype(np.float32),
            "qa_count": qa_count,
        }
    )


def process_tile(job: ProcessingJob) -> xr.Dataset:
    """Process a single tile for the job's window (single- or multi-year).

    Runs the full production pipeline: STAC query over ``job.datetime_range``,
    QA-masked scene load, pooled-P95 composite with a 12-month ``qa_count``
    climatology, and Natural Earth land masking.

    Args:
        job: Processing job specification (``year`` plus optional ``end_year``).

    Returns:
        Dataset with the LST P95 composite (``lst_p95``) and per-month
        ``qa_count`` for the job's window.
    """
    report_phase("stac_query")
    items = query_stac(job)

    if not items:
        msg = f"No scenes found for {job.tile.name} in {job.year}"
        raise ValueError(msg)

    report_phase("loading", scenes_found=len(items))

    # Planetary Computer: set up refreshable Azure SAS auth (local + Dask
    # workers) and rewrite asset hrefs to token-free /vsiaz/ paths so a
    # long-running compute never reads with an expired token. No-op for AWS.
    patch_url = None
    if _is_planetary_computer():
        from landsat_lst.azure_auth import enable_pc_azure_refresh  # noqa: PLC0415

        patch_url = enable_pc_azure_refresh(items)

    # A 5-year window pulls ~1900 scenes, so at least one transient read failure
    # is near-certain. Fill that scene with nodata rather than aborting the load;
    # a handful of dropped observations costs almost nothing against a P95 over
    # the full stack. The coverage check below is what makes this safe.
    data = load_scenes(items, job.tile.bbox, patch_url=patch_url, fail_on_error=False)

    # Built before the composite so de-striping estimates each scene's offset
    # over land only. Ocean is thermally stable and would damp the estimate on
    # coastal tiles.
    land_mask_da = _build_land_mask(job.tile.bbox, data.latitude, data.longitude)

    # A coarse second load for the de-striping offsets. Reading from the source
    # overviews costs ~factor**2 fewer bytes than a second native-resolution
    # pass, and the offset is a per-scene scalar that gains nothing from detail.
    offset_source = None
    offset_land_mask = None
    factor = settings.destripe_offset_resolution_factor
    if settings.destripe and factor > 1:
        offset_source = load_scenes(
            items,
            job.tile.bbox,
            patch_url=patch_url,
            fail_on_error=False,
            resolution_factor=factor,
        )
        offset_land_mask = _build_land_mask(
            job.tile.bbox, offset_source.latitude, offset_source.longitude
        )

    composite = compute_annual_composite(
        data,
        land_mask=land_mask_da,
        offset_source=offset_source,
        offset_land_mask=offset_land_mask,
    )

    # Silent nodata fill is otherwise undetectable: a low median or a high zero
    # fraction means reads failed en masse rather than occasionally.
    obs = composite["qa_count"].sum(dim="month").values
    log.info(
        "valid_coverage_obs_per_pixel",
        tile=job.tile.name,
        window=job.window_label,
        min=int(obs.min()),
        median=int(np.median(obs)),
        max=int(obs.max()),
        zero_frac=round(float((obs == 0).mean()), 3),
    )

    composite["lst_p95"] = composite["lst_p95"].where(land_mask_da)
    # Zero out ocean in the per-month counts too (broadcasts over the month dim);
    # keeps qa_count uint8 and makes ocean compress away.
    composite["qa_count"] = composite["qa_count"].where(land_mask_da, 0).astype(np.uint8)

    composite.attrs["tile"] = job.tile.name
    composite.attrs["year"] = job.year
    composite.attrs["window"] = job.window_label
    composite.attrs["scene_count"] = len(items)

    return composite
