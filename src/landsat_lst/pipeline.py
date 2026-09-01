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

from landsat_lst.aggregate import aggregate_to_output_grid, aligned_source_chunk
from landsat_lst.config import settings
from landsat_lst.encoding import LST_MIN_TRUSTED_C
from landsat_lst.ged import gap_mask_for_geobox
from landsat_lst.kernels import nanquantile_last
from landsat_lst.masks import get_land_mask_for_geobox, load_land_polygons
from landsat_lst.normalization import (
    debias_with_offsets,
    offset_diagnostics,
    rejection_floor,
    scene_keep_mask,
    scene_offsets,
)
from landsat_lst.offsets import OffsetCache, OffsetKey, cache_for_items
from landsat_lst.progress import report_phase, timed_section
from landsat_lst.qa import apply_qa_mask, convert_to_celsius
from landsat_lst.tiling import geobox_for_bbox, output_geobox_for_bbox

if TYPE_CHECKING:
    from collections.abc import Callable

    from odc.geo.geobox import GeoBox

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
    # Same cloud defaults the Planetary Computer path already sets
    # (azure_auth.py). Measured effect on AWS is within run-to-run noise
    # (the 2026-08 A/B's every arm matched its baseline_repeat), so this is
    # hygiene, not a speedup: it pins sane open/read behavior for the
    # request concurrency the unit reads now run at.
    from odc.stac import configure_rio  # noqa: PLC0415

    configure_rio(cloud_defaults=True)


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


def resolve_items(job: ProcessingJob) -> list:
    """The scene set one job runs over: query, then sample if asked.

    Every entry point into the tile path used to open with these seven lines,
    and a sharded tile needs them exactly once for the whole tile rather than
    once per shard. Two shards resolving their own scene set from a live
    catalog can disagree -- STAC is not frozen, and ``_sample_scenes`` is only
    deterministic given identical input -- and a tile assembled from two scene
    sets is striped in a way no downstream check inspects.

    Args:
        job: Tile and window to resolve.

    Returns:
        The STAC items to load, sampled when ``job.max_scenes`` is set.

    Raises:
        ValueError: If the query returned nothing.
    """
    with timed_section("stac_query"):
        items = query_stac(job)

    if not items:
        msg = f"No scenes found for {job.tile.name} in {job.window_label}"
        raise ValueError(msg)

    if job.max_scenes is not None:
        items = _sample_scenes(items, job.max_scenes)

    return items


def items_to_dicts(items: list) -> list[dict]:
    """Serialize resolved STAC items for a shard to read back.

    Whole items rather than the ids alone: a shard has to *load* these, which
    needs the asset hrefs and the properties ``odc-stac`` groups on, and
    re-fetching them by id would put the catalog back in the per-shard path
    that :func:`resolve_items` exists to keep it out of. (``fixture.py`` stores
    only ids because a fixture's pixels are already on disk and the ids are
    there for attribution.)
    """
    return [item.to_dict() for item in items]


def items_from_dicts(payload: list[dict]) -> list:
    """Rebuild items serialized by :func:`items_to_dicts`."""
    import pystac  # noqa: PLC0415

    return [pystac.Item.from_dict(entry) for entry in payload]


def load_scenes(
    items: list,
    bbox: tuple[float, float, float, float],
    patch_url: Callable[[str], str] | None = None,
    *,
    fail_on_error: bool = True,
    resolution_factor: int = 1,
    geobox: GeoBox | None = None,
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
        geobox: Load onto this grid instead of the whole ``bbox``. Used by a
            row-band shard, which loads a slice of the tile's geobox: slicing
            the tile's grid rather than deriving one from the band's own bounds
            is what keeps the band's pixels *the tile's* pixels (ADR-008), and
            ``bbox`` is then unused. Row slices only -- a column slice moves
            the geobox centroid longitude, which is what ``odc-stac`` derives
            its ``solar_day`` shift from, so two column bands can group the
            same items onto different time axes. See :mod:`landsat_lst.shards`.

    Returns:
        Dataset with thermal and QA bands.
    """
    # A coarse load (factor > 1) is the offset pass, which has no rechunk
    # term and takes the probe-measured larger request size; the native load
    # feeds the composite, whose single-time-chunk rechunk caps it at 512.
    # See the two fields' docstrings in config.py.
    # The native chunk is rounded up to a whole number of delivered cells (512
    # -> 513 at factor 3). ``coarsen`` on a straddling chunk has to rechunk the
    # stack first, and it picks an uneven split; aligning the request costs one
    # extra source column per chunk and skips the shuffle entirely. The coarse
    # offset path is never aggregated, so it keeps its own chunk untouched.
    csize = (
        settings.load_chunk_size_offsets
        if resolution_factor > 1
        else aligned_source_chunk(settings.load_chunk_size)
    )

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
        geobox=geobox if geobox is not None else geobox_for_bbox(bbox, resolution_factor),
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


def _output_coords(geobox: GeoBox | None) -> dict[str, np.ndarray] | None:
    """Delivered-grid coordinate values for the aggregator to stamp on.

    ``None`` when no geobox is given, which is the bare-Dataset path a unit
    test takes. Every production caller passes one, so every published pixel is
    labelled from the grid rather than from an average of source labels.
    """
    if geobox is None:
        return None
    return {name: coord.values for name, coord in geobox.coords.items()}


def geobox_coords(geobox: GeoBox) -> tuple[xr.DataArray, xr.DataArray]:
    """The ``(latitude, longitude)`` cell centres of a geobox, as coordinates.

    The delivered grid carries no loaded array to take coordinates from -- the
    stack is loaded on the source grid and only becomes delivered-grid data
    after :func:`~landsat_lst.aggregate.aggregate_to_output_grid` reduces it.
    So the output-side masks take their coordinates from the geobox's own
    affine, which is the same authority ``get_land_mask_for_geobox`` and
    ``gap_mask_for_geobox`` rasterize against.
    """
    coords = geobox.coords
    latitude = xr.DataArray(coords["latitude"].values, dims=["latitude"])
    longitude = xr.DataArray(coords["longitude"].values, dims=["longitude"])
    return latitude, longitude


def _build_land_mask(
    geobox: GeoBox,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
) -> xr.DataArray:
    """Rasterize the Natural Earth land mask onto a grid's exact coordinates.

    Rasterized against the geobox's own affine rather than against a transform
    rebuilt from its bounds. The two agree to about fifteen digits, which is
    not enough: a row band's mask has to be the exact slice of the tile's mask
    or the seam between two bands carries a one-pixel land/ocean disagreement,
    and offsets estimated over land would then be estimated over a slightly
    different set of pixels per band. See
    :func:`landsat_lst.masks.get_land_mask_for_geobox`.

    Both rasterio and odc-stac use north-down (descending latitude), so the
    rasterized array needs no flip.
    """
    land_polygons = load_land_polygons()
    land_mask = get_land_mask_for_geobox(geobox, land_polygons)
    return xr.DataArray(
        land_mask,
        dims=["latitude", "longitude"],
        coords={"latitude": latitude, "longitude": longitude},
    )


def _build_ged_gap_mask(
    geobox: GeoBox,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
) -> xr.DataArray:
    """Build the ASTER GED emissivity-gap mask on the grid's exact coordinates.

    True where a pixel must be dropped from the LST output: its ~1 km GED
    cell has NumObs == 0, or sits within ``settings.ged_gap_buffer_cells`` of
    one. Applied to the composite *output* only, mirroring the land mask's
    output-side application -- never to offset estimation, and never to
    ``qa_count``. See :mod:`landsat_lst.ged` and
    docs/findings-aster-ged-gaps.md.
    """
    gap = gap_mask_for_geobox(geobox)
    return xr.DataArray(
        gap,
        dims=["latitude", "longitude"],
        coords={"latitude": latitude, "longitude": longitude},
    )


def compute_annual_composite(
    data: xr.Dataset,
    *,
    land_mask: xr.DataArray | None = None,
    offset_source: xr.Dataset | None = None,
    offset_land_mask: xr.DataArray | None = None,
    source_land_mask: xr.DataArray | None = None,
    offset_cache: OffsetCache | None = None,
    offsets: tuple[xr.DataArray, xr.DataArray] | None = None,
    output_geobox: GeoBox | None = None,
) -> xr.Dataset:
    """Compute an LST P95 composite with a per-month observation count.

    The P95 is pooled across *every* time step in ``data``. For a multi-year
    window (``ProcessingJob.end_year`` set) this pools all scenes across all
    years -- the correct way to build a multi-year percentile (never average
    per-year P95s: percentile-of-percentiles is wrong).

    **Grids.** ``data`` arrives on the source grid, already fused into one
    observation per solar day. Everything from
    :func:`~landsat_lst.aggregate.aggregate_to_output_grid` onward -- the land
    mask, the correction, the percentile, the counts, and the returned Dataset
    -- is on the delivered nominal ~100 m grid. ``land_mask`` must therefore be
    on the *delivered* grid, and ``offset_source`` on the offset grid, which is
    a coarsening of the source grid and independent of both (ADR-017).

    ``qa_count`` is a **12-month climatology of delivered observations**: for
    month M it is the number of nominal ~100 m solar-day observations in month
    M that met the valid-area rule, pooled across all years in the window. It
    is not a sum of source-cell counts, and it never exceeds the number of
    solar days. Dims ``(month, latitude, longitude)``, always 12 months
    (missing months filled 0), dtype ``uint8`` (counts stay well under 255 even
    for a 5-year window). It counts only observations that reach the composite,
    so scenes dropped by de-striping are excluded.

    When ``settings.destripe`` is on, each scene is shifted by a single
    scene-wide offset before compositing and scenes whose offset is implausible
    are discarded. See ``normalization.seasonal_debias`` and issue #46.

    WARNING: Land masking happens only when ``land_mask`` is supplied. For
    production output that excludes ocean pixels, use process_tile() instead,
    which always supplies one. See issue #26.

    Args:
        data: Dataset with thermal and QA bands across time.
        land_mask: Optional boolean land mask on the **delivered** grid's
            ``(latitude, longitude)``. Applied after aggregation and before
            de-biasing, so per-scene offsets estimated on this path are
            estimated over land only; ocean is thermally stable and would
            otherwise damp the estimate on coastal tiles.
        offset_source: Optional coarse stack to estimate the offsets from.
        offset_land_mask: Land mask on ``offset_source``'s grid.
        source_land_mask: Land mask on the **source** grid, used only when no
            ``offset_source`` is given -- the offset factor is then 1, the
            estimator reads the source stack itself, and this is what keeps it
            estimating over land. Never applied to the delivered output; that
            is ``land_mask``'s job.
        offset_cache: Optional :class:`~landsat_lst.offsets.OffsetCache`. A hit
            replaces the tile's longest compute with a kilobyte read. Only the
            estimate is cached, never the rejection, so a cap sweep re-reads one
            record per candidate. See issue #77 item 2.
        offsets: A ready ``(offset, n_valid)`` pair, which skips estimation
            entirely -- ``offset_source``, ``offset_land_mask``, and
            ``offset_cache`` are then all unused, since there is nothing left
            to estimate or to cache. This is the seam a row-band shard runs
            through: the offsets are estimated once for the whole tile and
            every band applies *the same* scalars, which is what makes the
            bands concatenate into the tile's composite. Estimating per band
            would give each band its own reference climatology and its own
            correction, i.e. a horizontal seam at every band boundary. The
            pair must be aligned by time coordinate, not position; see
            :func:`~landsat_lst.normalization.debias_with_offsets`.
        output_geobox: The delivered grid this stack aggregates onto, used to
            label the result from the grid definition rather than from averaged
            source labels. Every production caller passes one; omitting it is
            for a bare Dataset with no mask to align against. See
            :func:`~landsat_lst.aggregate.aggregate_to_output_grid`.

    Returns:
        Dataset with ``lst_p95`` ``(latitude, longitude)`` and ``qa_count``
        ``(month, latitude, longitude)`` bands.

    Note:
        P50 (median) was removed per stakeholder feedback - hot season temps
        (P95) are what matter for urban heat applications. See issue #22.
    """
    # Steps 2 and 3 of the V1 order, in this order and no other (issue #120).
    # ``data`` arrives already fused into one observation per solar day, on the
    # source grid, by odc-stac's ``groupby="solar_day"``. QA, the DN=0 fill
    # test, the Collection 2 scaling, and the plausibility clamp all run on the
    # source cells, so a cell dropped by any of them is invisible to the
    # reducer rather than averaged in as a number.
    masked = apply_qa_mask(data)

    lst = convert_to_celsius(masked["lwir11"])

    # The estimate is made on the SOURCE side and applied on the delivered
    # side. That split is deliberate and is the whole of ADR-017's rule 7: the
    # offset grid is a coarsening of the source grid, its accuracy bound was
    # calibrated there (docs/findings-offset-subsampling.md), and
    # ``destripe_min_scene_pixels`` counts pixels in it. Estimating after
    # aggregation would silently move all three because the *output* moved. The
    # correction still applies after aggregation, as the V1 order requires, and
    # costs nothing to defer: subtracting one scalar per scene commutes exactly
    # with an area-weighted mean over that scene's cells.
    estimate = offsets
    if settings.destripe and estimate is None:
        # The offset is one scalar per scene, so it can be estimated from a
        # coarse stack read off the source COGs' overviews. That cuts bytes
        # read, which post-load subsampling cannot do: dask must materialize a
        # whole chunk before discarding most of it.
        if offset_source is not None:
            estimation_source = convert_to_celsius(apply_qa_mask(offset_source)["lwir11"])
            estimation_mask = offset_land_mask
        else:
            estimation_source = lst
            estimation_mask = source_land_mask
        if estimation_mask is not None:
            estimation_source = estimation_source.where(estimation_mask)

        # Estimating the offsets is the first real compute of the tile, and on a
        # five-year window it runs for many minutes, so the watcher hears about
        # it before it starts rather than after. With a warm cache it is a
        # kilobyte read instead, and the phase passes in under a second.
        with timed_section("destriping"):
            estimate = scene_offsets(
                estimation_source,
                cache=offset_cache,
                # The mask on whichever grid the offsets are estimated on, so
                # phase A can skip blocks that hold no land at all. On a
                # coastal tile most blocks are ocean; the skip is free work
                # reduction and value-identical (an all-NaN block's medians
                # are NaN either way).
                land_mask=estimation_mask,
            )

    # Aggregate every masked solar-day observation onto the delivered grid
    # *before* the correction and the percentile. Computing a P95 on the source
    # grid and coarsening the result afterwards is a different statistic and
    # saves none of the percentile work; the V1 decision rules it out
    # explicitly. See :mod:`landsat_lst.aggregate` and ADR-017.
    with timed_section("aggregating"):
        lst = aggregate_to_output_grid(lst, coords=_output_coords(output_geobox))

    # On the delivered grid from here down, which is the grid ``land_mask``
    # must already be on. Land is an output-side policy (step 7): it is applied
    # here rather than before aggregation so that a coastal cell's support is
    # decided by QA alone, and a 4-of-9-land cell is not silently starved of
    # support by a mask that has nothing to say about cloud. Applying it ahead
    # of de-biasing still keeps the factor-1 estimating path estimating over
    # land only, which is what it is here for.
    if land_mask is not None:
        lst = lst.where(land_mask)

    scenes_kept = None
    if settings.destripe:
        if estimate is None:  # pragma: no cover - unreachable; the branch above sets it
            msg = (
                "de-striping is on but no offset estimate exists. Either pass "
                "``offsets``, or leave the estimating branch above to build one."
            )
            raise ValueError(msg)
        # Rejection and subtraction, on the delivered stack. The floor follows
        # the grid the estimate was made on, which the configured factor names
        # -- read from the setting rather than from the presence of a coarse
        # stack a shard process never loaded.
        with timed_section("destriping"):
            lst, offset, keep = debias_with_offsets(
                lst,
                *estimate,
                max_offset_c=settings.destripe_max_offset_c,
                min_scene_pixels=settings.destripe_min_scene_pixels,
                min_offset_samples=settings.destripe_min_offset_samples,
                offset_source_given=settings.destripe_offset_resolution_factor > 1,
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

    # Per-calendar-month climatology of valid observations. ``lst`` is already
    # on the delivered grid, so a "valid observation" here is one nominal
    # ~100 m solar-day cell that met the valid-area rule -- never a count of
    # source cells, and never more than the number of solar days in the month.
    # The two populations agree by construction: the same NaN pattern gates the
    # percentile and the counts. groupby pools every year in the window into
    # its month bucket; reindex guarantees all 12 months.
    qa_count = (
        valid_mask.groupby("time.month")
        .sum()
        .reindex(month=range(1, 13), fill_value=0)
        .astype(np.uint8)
    )

    # Total valid obs per pixel (across the whole window) gates the P95 fill.
    total_valid = valid_mask.sum(dim="time")

    # Vectorized sort-based P95 instead of ``lst.quantile``. On the production
    # window xarray's path lands in ``np.nanquantile``'s per-pixel
    # ``apply_along_axis`` loop: dask's own vectorized escape hatch
    # (``_custom_nanquantile``) bails back to numpy above 1,000 elements on
    # the reduced axis, and the window holds ~2,930 scenes. That loop holds
    # the GIL, which is why 4 threads measured 1.06x one thread here.
    # ``kernels.nanquantile_last`` is one GIL-releasing ``np.sort`` and
    # reproduces the shipped values bit for bit after the float32 cast below
    # (pinned by tests/unit/test_kernels.py). apply_ufunc moves time to the
    # last axis per block; the single-chunk time rechunk above guarantees the
    # core dimension is whole.
    lst_p95 = xr.apply_ufunc(
        nanquantile_last,
        lst,
        input_core_dims=[["time"]],
        kwargs={"q": 0.95},
        dask="parallelized",
        output_dtypes=[np.float64],
    )
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

    items = resolve_items(job)

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
    coarse_geobox = geobox_for_bbox(job.tile.bbox, factor)
    source = load_scenes(
        items,
        job.tile.bbox,
        patch_url=patch_url,
        fail_on_error=False,
        geobox=coarse_geobox,
    )

    with timed_section("land_mask"):
        land = _build_land_mask(coarse_geobox, source.latitude, source.longitude)

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
    items = resolve_items(job)

    report_phase("loading", scenes_found=len(items))

    patch_url = _patch_url_for(items)
    # Two grids, named apart. Scenes load and solar-day fuse on the source
    # grid; every mask and every published pixel lives on the delivered one.
    native_geobox = geobox_for_bbox(job.tile.bbox)
    output_geobox = output_geobox_for_bbox(job.tile.bbox)

    # A 5-year window pulls ~1900 scenes, so at least one transient read failure
    # is near-certain. Fill that scene with nodata rather than aborting the load;
    # a handful of dropped observations costs almost nothing against a P95 over
    # the full stack. The coverage line the exporter logs is what makes this
    # safe: it is where a run that filled wholesale rather than occasionally
    # shows up. See `cog._log_coverage`.
    data = load_scenes(
        items, job.tile.bbox, patch_url=patch_url, fail_on_error=False, geobox=native_geobox
    )

    # On the delivered grid, because that is where the composite lives from the
    # aggregation onward. Built before the composite so de-striping estimates
    # each scene's offset over land only. Ocean is thermally stable and would
    # damp the estimate on coastal tiles. Rasterizing Natural Earth runs no dask
    # graph and is not free, so it gets its own phase rather than hiding inside
    # `loading`.
    with timed_section("land_mask"):
        land_mask_da = _build_land_mask(output_geobox, *geobox_coords(output_geobox))

    # A coarse second load for the de-striping offsets. Reading from the source
    # overviews costs ~factor**2 fewer bytes than a second native-resolution
    # pass, and the offset is a per-scene scalar that gains nothing from detail.
    offset_source = None
    offset_land_mask = None
    source_land_mask = None
    factor = settings.destripe_offset_resolution_factor
    if settings.destripe and factor == 1:
        # Factor 1 means the offset grid *is* the source grid, so the estimator
        # reads the native stack and needs a native-grid land mask to estimate
        # over land only. Not a production configuration -- the default factor
        # is 2 -- but the alternative is an estimate quietly taken over ocean.
        with timed_section("land_mask"):
            source_land_mask = _build_land_mask(native_geobox, data.latitude, data.longitude)
    if settings.destripe and factor > 1:
        # Timed in its own right. Building this graph is single-threaded Python
        # over every scene in the window, and sitting untimed between two
        # `land_mask` sections it billed its minutes to the mask and published
        # the same phase name twice with a silence in the middle.
        coarse_geobox = geobox_for_bbox(job.tile.bbox, factor)
        with timed_section("offset_load", scenes_found=len(items)):
            offset_source = load_scenes(
                items,
                job.tile.bbox,
                patch_url=patch_url,
                fail_on_error=False,
                geobox=coarse_geobox,
            )
        with timed_section("land_mask"):
            offset_land_mask = _build_land_mask(
                coarse_geobox, offset_source.latitude, offset_source.longitude
            )

    composite = compute_annual_composite(
        data,
        land_mask=land_mask_da,
        offset_source=offset_source,
        offset_land_mask=offset_land_mask,
        source_land_mask=source_land_mask,
        output_geobox=output_geobox,
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

    # ASTER GED emissivity-gap mask, output-side like the land mask and only
    # on the LST band: qa_count keeps counting over gap cells because zero-or-
    # more observations there is still the evidence layer, and the offsets
    # above were deliberately estimated without this mask (a per-scene median
    # over a whole tile does not care about 0.86% of pixels, and keeping the
    # mask out of the estimator keeps every cached offset record valid).
    if settings.ged_gap_mask:
        with timed_section("ged_gap_mask"):
            gap_mask_da = _build_ged_gap_mask(output_geobox, *geobox_coords(output_geobox))
        composite["lst_p95"] = composite["lst_p95"].where(~gap_mask_da)

    composite.attrs["tile"] = job.tile.name
    composite.attrs["year"] = job.year
    composite.attrs["window"] = job.window_label
    composite.attrs["scene_count"] = len(items)

    return composite
