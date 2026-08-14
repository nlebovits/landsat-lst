"""Main ETL pipeline for Landsat LST composites."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pystac_client
import structlog
import xarray as xr
from odc.stac import stac_load
from pystac_client.stac_api_io import StacApiIO
from urllib3.util.retry import Retry

from landsat_lst.config import settings
from landsat_lst.encoding import LST_MIN_TRUSTED_C
from landsat_lst.masks import get_land_mask_for_bbox, load_land_polygons
from landsat_lst.normalization import (
    offset_diagnostics,
    rejection_floor,
    scene_keep_mask,
    scene_offsets,
    seasonal_debias,
)
from landsat_lst.offsets import OffsetCache, OffsetKey, cache_for_items
from landsat_lst.progress import report_phase, timed_section
from landsat_lst.qa import apply_qa_mask, convert_to_celsius
from landsat_lst.tiling import geobox_for_bbox

if TYPE_CHECKING:
    from collections.abc import Callable

    from landsat_lst.models import ProcessingJob
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

# Planetary Computer URL prefix for conditional signing
_PC_URL_PREFIX = "https://planetarycomputer.microsoft.com"

#: Scenes per time chunk when loading. Named rather than inlined because
#: :func:`landsat_lst.profiling.synthetic_dataset` has to reproduce it exactly:
#: a synthetic stack chunked differently from the real one builds a different
#: graph, and the whole value of planning against it is that it does not.
TIME_CHUNK = 10


#: Statuses worth trying again. 429 is throttling; the 5xx codes are the ones a
#: load balancer or a gateway returns while the API behind it is briefly
#: unwell. A 4xx other than 429 says the request itself is wrong, and repeating
#: it would only spend the retry budget.
_RETRY_STATUSES = (429, 500, 502, 503, 504)


def _stac_retry() -> Retry:
    """Build the retry policy for STAC requests.

    ``pystac_client`` already mounts an adapter, but it builds one from a plain
    integer, and ``Retry.from_int`` leaves ``status_forcelist`` empty, the
    backoff at zero, and ``allowed_methods`` restricted to idempotent verbs. So
    a 500 was never retried, a search is a POST and was not retried either, and
    nothing waited between attempts. That is how one transient HTTP 500 ended a
    tile at second 10 of a five-hour budget on 2026-08-14.

    ``allowed_methods=None`` retries every verb, which is what lets the POST
    search retry. A STAC search is a read, so replaying it is safe. The jitter
    keeps concurrent VMs from retrying in lockstep through a single outage.
    """
    return Retry(
        total=settings.stac_retries,
        backoff_factor=settings.stac_retry_backoff_s,
        status_forcelist=_RETRY_STATUSES,
        allowed_methods=None,
        respect_retry_after_header=True,
        backoff_jitter=0.5,
    )


def open_catalog() -> pystac_client.Client:
    """Open the configured STAC catalog with a retry policy that covers 5xx.

    Retrying here rather than through ``settings.coiled_retries`` is deliberate:
    a VM restart costs minutes and destroys everything the tile had computed,
    while an HTTP retry costs seconds and keeps it. A genuine outage still
    exhausts the budget in well under a minute and fails the tile, rather than
    holding a VM for ``settings.coiled_job_timeout``.
    """
    return pystac_client.Client.open(
        settings.stac_url,
        stac_io=StacApiIO(max_retries=_stac_retry(), timeout=settings.stac_timeout_s),
    )


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


def _sample_scenes(items: list, max_scenes: int) -> list:
    """Keep at most ``max_scenes``, spread evenly over the window by date.

    Evenly rather than the first N, because de-striping estimates each scene's
    offset against a per-pixel *monthly* climatology: a sample drawn from one
    end of the window would leave most months with no reference at all, and the
    run would exercise a rejection path rather than the pipeline.

    Sampled output is not the product. It exists so the machinery can be
    exercised at real tile geometry in minutes -- the same chunking, the same
    reprojection, the same COG write -- instead of hours.
    """
    if len(items) <= max_scenes:
        return items

    ordered = sorted(items, key=lambda item: item.datetime)
    step = len(ordered) / max_scenes
    sampled = [ordered[int(i * step)] for i in range(max_scenes)]

    log.warning(
        "scene_sample_applied",
        kept=len(sampled),
        available=len(items),
        note="composite is a sample, not the product",
    )
    return sampled


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

    catalog = open_catalog()

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
        chunks={"time": TIME_CHUNK, "latitude": csize, "longitude": csize},
        groupby="solar_day",
        patch_url=patch_url,
        fail_on_error=fail_on_error,
        resampling=resampling,
    )


def scene_cloud_cover(
    items: list,
    bbox: tuple[float, float, float, float],
    time: xr.DataArray,
    resolution_factor: int = 1,
) -> xr.DataArray:
    """Aggregate each STAC item's ``eo:cloud_cover`` onto a loaded time axis.

    ``load_scenes`` groups items by solar day, so several items collapse into
    one time step and there is no item-indexed axis to join a per-item property
    against. This reproduces that grouping and returns the mean cloud cover of
    the items behind each step.

    Replicating a grouping rule invites silent drift, so ``time`` is required
    rather than optional: the timestamps derived here are checked against the
    axis the caller actually loaded, and a mismatch raises. Skipping that check
    would let a changed upstream rule misalign the join, and every statistic
    drawn from it would then be wrong rather than absent.

    Args:
        items: The same STAC items handed to :func:`load_scenes`.
        bbox: The same bounding box.
        time: The ``time`` coordinate of the array :func:`load_scenes` returned.
        resolution_factor: The same factor. It reaches the grouping through the
            geobox centroid, which sets the solar-time shift.

    Returns:
        Cloud cover percentage per solar-day scene, on ``time``.

    Raises:
        ValueError: If the reproduced grouping does not match ``time``.
    """
    from odc.stac._mdtools import parse_items  # noqa: PLC0415
    from odc.stac._stac_load import _extract_timestamps, _group_items  # noqa: PLC0415

    parsed = list(parse_items(items))
    gbox = geobox_for_bbox(bbox, resolution_factor)
    ((mid_lon, _),) = gbox.extent.centroid.to_crs("epsg:4326").points

    grouped = _group_items(items, parsed, "solar_day", mid_lon)
    stamps = np.array(
        _extract_timestamps([[parsed[i] for i in g] for g in grouped]), dtype="datetime64[ns]"
    )

    loaded = time.values.astype("datetime64[ns]")
    if stamps.shape != loaded.shape or not np.array_equal(stamps, loaded):
        msg = (
            f"solar-day grouping does not reproduce the loaded time axis "
            f"({stamps.size} scenes derived vs {loaded.size} loaded). "
            "odc-stac's grouping rule has changed; scene_cloud_cover must follow it."
        )
        raise ValueError(msg)

    cover = [
        float(np.mean([items[i].properties["eo:cloud_cover"] for i in group])) for group in grouped
    ]
    return xr.DataArray(
        np.array(cover, dtype="float64"),
        dims=["time"],
        coords={"time": time.values},
        name="eo_cloud_cover",
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
    offset_cache: OffsetCache | None = None,
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
        offset_source: Optional coarse stack to estimate the offsets from.
        offset_land_mask: Land mask on ``offset_source``'s grid.
        offset_cache: Optional :class:`~landsat_lst.offsets.OffsetCache`. A hit
            replaces the tile's longest compute with a kilobyte read. Only the
            estimate is cached, never the rejection, so a cap sweep re-reads one
            record per candidate. See issue #77 item 2.

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
        # it before it starts rather than after. With a warm cache it is a
        # kilobyte read instead, and the phase passes in under a second.
        with timed_section("destriping"):
            lst, offset, keep = seasonal_debias(
                lst,
                max_offset_c=settings.destripe_max_offset_c,
                min_scene_pixels=settings.destripe_min_scene_pixels,
                min_offset_samples=settings.destripe_min_offset_samples,
                offset_source=source,
                cache=offset_cache,
            )
        diagnostics = offset_diagnostics(offset, keep)
        log.info("destripe_offsets_degC", **diagnostics)
        scenes_kept = int(diagnostics["n_kept"])

    # Everything below is graph construction: single-threaded Python that
    # allocates a task object per block per scene and computes nothing. At
    # production geometry it is minutes, it runs no dask graph so it publishes
    # no task fraction, and under the old single ``compositing`` label it was
    # indistinguishable from a wedged compute. See issue #77 item 4.
    with timed_section("composite_graph", scenes_kept=scenes_kept):
        return _composite_graph(lst)


def _composite_graph(lst: xr.DataArray) -> xr.Dataset:
    """Build the lazy P95 and monthly-count expressions. Computes nothing.

    Both outputs are deliberately built on one time-contiguous view of the
    stack. ``quantile`` needs the whole time series per pixel and inserts that
    rechunk itself; ``groupby("time.month").sum()`` does not and would otherwise
    consume the 10-scene chunks straight from the load. Two differently chunked
    consumers means every source block is materialized twice, and when
    :func:`~landsat_lst.cog.cog_export` then asks for both in one compute the
    scheduler has no block-by-block order that satisfies them together -- it
    fans out and holds the whole stack. Measured on a 4096 x 4096 x 120 synthetic
    tile: three sequential passes cost 1.30 GB peak, fusing the two export
    writes without this line cost 1.0 pass but **10.88 GB**, and fusing them
    with it costs 1.0 pass at 1.60 GB. The rechunk itself adds nothing to the
    memory floor, since the P95 already forced it.
    """
    if lst.chunks is not None:
        lst = lst.chunk({"time": -1})

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


def _patch_url_for(items: list) -> Callable[[str], str] | None:
    """Rewrite asset hrefs for Planetary Computer, or leave them alone on AWS.

    Sets up refreshable Azure SAS auth (local plus dask workers) and points the
    hrefs at token-free ``/vsiaz/`` paths, so a compute that outlives the
    45-minute token never reads with an expired one. No-op against Earth Search.
    """
    if not _is_planetary_computer():
        return None

    from landsat_lst.azure_auth import enable_pc_azure_refresh  # noqa: PLC0415

    return enable_pc_azure_refresh(items)


@dataclass(frozen=True)
class OffsetEstimate:
    """What one tile-window's offset pass produced, and what it cost."""

    key: OffsetKey
    scenes: int
    diagnostics: dict[str, float]
    cached: bool
    duration_s: float

    @property
    def rejected_frac(self) -> float:
        """Share of scenes de-striping would discard, as a fraction."""
        return float(self.diagnostics.get("rejected_frac", 0.0))


def compute_tile_offsets(
    job: ProcessingJob,
    *,
    use_offset_cache: bool = True,
    refresh: bool = False,
    storage: StorageBackend | None = None,
) -> OffsetEstimate:
    """Estimate and persist one tile-window's per-scene offsets, and stop there.

    The offset pass is the longest compute in a tile and the one whose result is
    a few kilobytes. Running it on its own warms the cache that
    :func:`process_tile` reads, so the composite that follows starts halfway;
    it is also the cheapest way to see a tile's rejection fraction, which is the
    number ``destripe_max_offset_c`` was calibrated against and the one worth
    checking on a climate unlike the mid-latitude cropland it was fitted on.

    Only the native stack is skipped. Everything the estimate depends on -- the
    scene set, the coarse load, the land mask, the QA mask, the clamp -- runs
    exactly as it does in a full tile, so the offsets this writes are the
    offsets a full tile would have computed.

    Args:
        job: Tile and window to estimate for.
        use_offset_cache: ``False`` neither reads nor writes the cache, which
            makes this a pure measurement with no side effect.
        refresh: Skip the lookup but still write, replacing whatever was stored.
        storage: Backend the cache lives in. Defaults to the configured one.

    Returns:
        :class:`OffsetEstimate` with the rejection diagnostics and whether the
        answer came from cache.
    """
    started = time.monotonic()

    with timed_section("stac_query"):
        items = query_stac(job)
    if not items:
        msg = f"No scenes found for {job.tile.name} in {job.window_label}"
        raise ValueError(msg)
    if job.max_scenes is not None:
        items = _sample_scenes(items, job.max_scenes)

    factor = settings.destripe_offset_resolution_factor
    cache = cache_for_items(
        tile=job.tile.name,
        window=job.window_label,
        items=items,
        factor=factor,
        storage=storage,
        enabled=use_offset_cache,
        read=not refresh,
    )

    report_phase("loading", scenes_found=len(items))
    patch_url = _patch_url_for(items)
    source = load_scenes(
        items,
        job.tile.bbox,
        patch_url=patch_url,
        fail_on_error=False,
        resolution_factor=factor,
    )

    with timed_section("land_mask"):
        land = _build_land_mask(job.tile.bbox, source.latitude, source.longitude)

    lst = convert_to_celsius(apply_qa_mask(source)["lwir11"]).where(land)

    with timed_section("destriping", scenes_found=len(items)):
        offset, n_valid = scene_offsets(lst, cache=cache)

    keep = scene_keep_mask(
        offset,
        n_valid,
        max_offset_c=settings.destripe_max_offset_c,
        floor=rejection_floor(offset_source_given=factor > 1),
    )
    diagnostics = offset_diagnostics(offset, keep)
    log.info("destripe_offsets_degC", tile=job.tile.name, **diagnostics)

    return OffsetEstimate(
        key=cache.key,
        scenes=len(items),
        diagnostics=diagnostics,
        cached=bool(cache.last_read_hit),
        duration_s=time.monotonic() - started,
    )


def process_tile(
    job: ProcessingJob,
    *,
    use_offset_cache: bool = True,
    storage: StorageBackend | None = None,
) -> xr.Dataset:
    """Process a single tile for the job's window (single- or multi-year).

    Runs the full production pipeline: STAC query over ``job.datetime_range``,
    QA-masked scene load, pooled-P95 composite with a 12-month ``qa_count``
    climatology, and Natural Earth land masking.

    Args:
        job: Processing job specification (``year`` plus optional ``end_year``).
        use_offset_cache: Read and write the per-scene offset cache. Turn it off
            (``--no-offset-cache``) to force a recompute, which is what you want
            when validating a change to the estimator itself rather than to
            anything downstream of it.
        storage: Backend the offset cache lives in. Defaults to the configured
            one. The composite itself is returned in memory either way.

    Returns:
        Dataset with the LST P95 composite (``lst_p95``) and per-month
        ``qa_count`` for the job's window.
    """
    with timed_section("stac_query"):
        items = query_stac(job)

    if not items:
        msg = f"No scenes found for {job.tile.name} in {job.year}"
        raise ValueError(msg)

    if job.max_scenes is not None:
        items = _sample_scenes(items, job.max_scenes)

    report_phase("loading", scenes_found=len(items))

    patch_url = _patch_url_for(items)

    # A 5-year window pulls ~1900 scenes, so at least one transient read failure
    # is near-certain. Fill that scene with nodata rather than aborting the load;
    # a handful of dropped observations costs almost nothing against a P95 over
    # the full stack. The coverage line the exporter logs is what makes this
    # safe: it is where a run that filled wholesale rather than occasionally
    # shows up. See `cog._log_coverage`.
    data = load_scenes(items, job.tile.bbox, patch_url=patch_url, fail_on_error=False)

    # Built before the composite so de-striping estimates each scene's offset
    # over land only. Ocean is thermally stable and would damp the estimate on
    # coastal tiles. Rasterizing Natural Earth over an 18,000 px grid runs no
    # dask graph and is not free, so it gets its own phase rather than hiding
    # inside `loading`.
    with timed_section("land_mask"):
        land_mask_da = _build_land_mask(job.tile.bbox, data.latitude, data.longitude)

    # A coarse second load for the de-striping offsets. Reading from the source
    # overviews costs ~factor**2 fewer bytes than a second native-resolution
    # pass, and the offset is a per-scene scalar that gains nothing from detail.
    offset_source = None
    offset_land_mask = None
    factor = settings.destripe_offset_resolution_factor
    if settings.destripe and factor > 1:
        # Timed in its own right. Building this graph is single-threaded Python
        # over every scene in the window, and sitting untimed between two
        # `land_mask` sections it billed its minutes to the mask and published
        # the same phase name twice with a silence in the middle.
        with timed_section("offset_load", scenes_found=len(items)):
            offset_source = load_scenes(
                items,
                job.tile.bbox,
                patch_url=patch_url,
                fail_on_error=False,
                resolution_factor=factor,
            )
        with timed_section("land_mask"):
            offset_land_mask = _build_land_mask(
                job.tile.bbox, offset_source.latitude, offset_source.longitude
            )

    composite = compute_annual_composite(
        data,
        land_mask=land_mask_da,
        offset_source=offset_source,
        offset_land_mask=offset_land_mask,
        offset_cache=cache_for_items(
            tile=job.tile.name,
            window=job.window_label,
            items=items,
            factor=factor,
            storage=storage,
            enabled=use_offset_cache,
        ),
    )

    # No coverage reduction here. Silent nodata fill still has to be caught --
    # `fail_on_error=False` above makes a low median or a high zero fraction the
    # only sign that reads failed en masse -- but this used to be an eager
    # `.values` on `qa_count`, a full pass over the native stack for four
    # numbers that were logged and thrown away. Measured at 1.0x a full pass on
    # a synthetic tile, one of three. The same four numbers now come off the
    # written QA raster during the statistics walk the exporter already runs
    # (`cog._log_coverage`, same `valid_coverage_obs_per_pixel` key). See #80.
    composite["lst_p95"] = composite["lst_p95"].where(land_mask_da)
    # Zero out ocean in the per-month counts too (broadcasts over the month dim);
    # keeps qa_count uint8 and makes ocean compress away.
    composite["qa_count"] = composite["qa_count"].where(land_mask_da, 0).astype(np.uint8)

    composite.attrs["tile"] = job.tile.name
    composite.attrs["year"] = job.year
    composite.attrs["window"] = job.window_label
    composite.attrs["scene_count"] = len(items)

    return composite
