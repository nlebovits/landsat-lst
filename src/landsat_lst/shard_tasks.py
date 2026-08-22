"""What one shard of one tile does, once, on one VM.

Every function here is a whole batch task: it reads the plan, works out which
slice of the tile is its own from its index, checks whether that slice's
artifacts already exist, and either exits or produces them. Nothing here knows
about Coiled, and nothing here polls: sequencing between stages belongs to
:mod:`landsat_lst.shard_driver`, and completion is bytes in the bucket.

Three rules hold across all of them, and each closes a failure this project has
already paid for somewhere else:

- **Every task checks its own output first.** The output key is a pure function
  of the index (:mod:`landsat_lst.shards`), so a shard that finds its artifact
  present exits immediately. That is what makes a barrier resubmission safe:
  the driver cannot tell a slow shard from a dead one, so it resubmits indexes
  that may still be running, and a second writer must be a no-op rather than a
  race.
- **Every task verifies the plan digest against its own settings.** Two
  processes deriving the block list from settings that drifted apart would each
  be internally consistent and jointly wrong, and the merge would not notice.
  :meth:`landsat_lst.shards.TilePlan.from_dict` raises on drift; nothing here
  catches it.
- **Instrumentation never fails a shard.** Heartbeats, logs, and attempt
  numbers are best-effort, as they are for a whole tile (ADR-014).

The composite stage loads at :attr:`~landsat_lst.config.Settings.shard_composite_chunk`
rather than ``load_chunk_size``. A whole-tile composite stops at 512 because the
single-time-chunk rechunk holds ``chunk**2 * scenes * 4 B``; a row band holds a
fraction of the rows, so the same per-task working set buys the larger request
the 2026-08-21 probe measured as the throughput lever. Because the plan digest
covers ``load_chunk_size``, :func:`apply_shard_settings` applies the override in
*every* shard process and in the planner, so all of them hash the same number.
The stages that read coarsely are unaffected either way: they load through
``load_chunk_size_offsets``.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog
import xarray as xr

from landsat_lst import shards
from landsat_lst.config import settings
from landsat_lst.logging_config import configure_logging
from landsat_lst.models import ProcessingJob
from landsat_lst.normalization import _io_block_edge, _scene_batches, climatology_by_blocks
from landsat_lst.normalization import offsets_by_scene as _offsets_by_scene
from landsat_lst.offsets import (
    OffsetCache,
    OffsetKey,
    merge_scene_partials,
    partial_payload,
)
from landsat_lst.progress import TileHeartbeat, capture_task_log, report_phase, timed_section
from landsat_lst.qa import apply_qa_mask, convert_to_celsius
from landsat_lst.storage import PRODUCTS, get_storage
from landsat_lst.tiling import geobox_for_bbox, parse_tile_name

if TYPE_CHECKING:
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()


def apply_shard_settings() -> None:
    """Pin the settings every shard of a tile must agree on.

    Called by the planner *and* by every shard, so the plan digest -- which
    covers ``load_chunk_size`` -- is computed from the same number everywhere.
    Setting it only in the composite shard would make that shard refuse a plan
    its own planner had cut.
    """
    settings.load_chunk_size = settings.shard_composite_chunk


def job_for_window(tile: str, window: str) -> ProcessingJob:
    """Rebuild the job a window label came from.

    ``ProcessingJob.window_label`` is invertible -- ``"2021-2025-sample300"``
    is exactly ``year=2021, end_year=2025, max_scenes=300`` -- so the plan does
    not carry a second copy of the job beside the label every storage key is
    already built from. Two records of one fact would eventually disagree.

    Raises:
        ValueError: If ``window`` is not a label this project produces.
    """
    body, _, sample = window.partition("-sample")
    max_scenes = int(sample) if sample else None
    start, _, end = body.partition("-")
    if not start.isdigit() or (end and not end.isdigit()):
        msg = f"{window!r} is not a window label"
        raise ValueError(msg)
    return ProcessingJob(
        tile=parse_tile_name(tile),
        year=int(start),
        end_year=int(end) if end else None,
        max_scenes=max_scenes,
    )


def _time_coord(plan: shards.TilePlan) -> xr.DataArray:
    """The tile's full time axis, as the plan froze it.

    The axis every per-scene answer is joined on: the offset partials merge
    against it and the cached record is read back on it. A band's own stack may
    be a strict subset of it (``odc-stac`` drops a step whose scenes miss the
    band's rows), which is exactly why the join is by coordinate value.
    """
    times = pd.to_datetime(plan.scene_times).values
    return xr.DataArray(times, dims=["time"], coords={"time": times})


def _offset_key(plan: shards.TilePlan) -> OffsetKey:
    """The ordinary ADR-012 cache key for this tile-window's scene set.

    Deliberately the *same* key a single-VM tile would write. The merged
    offsets are not a shard artifact in a shard-shaped location: they are the
    tile's offsets, and a later whole-tile run over the same scenes should find
    them and skip the pass.
    """
    return OffsetKey.build(
        tile=plan.tile,
        window=plan.window,
        factor=plan.offset_factor,
        scene_ids=plan.scene_ids,
    )


class ShardContext:
    """The plan, the items, and the job, resolved once for one shard."""

    def __init__(
        self,
        *,
        plan: shards.TilePlan,
        items: list,
        job: ProcessingJob,
        root: str,
        storage: StorageBackend,
    ) -> None:
        self.plan = plan
        self.items = items
        self.job = job
        self.root = root
        self.storage = storage

    @property
    def tile(self) -> str:
        return self.plan.tile

    def keys(self, prefix: str = "") -> set[str]:
        """Every key this tile's shards have published under ``prefix``."""
        return set(self.storage.list_prefix(f"{self.root}/{prefix}"))


def load_context(run_id: str, tile: str, *, storage: StorageBackend | None = None) -> ShardContext:
    """Read the plan and the frozen item list, refusing a plan cut elsewhere.

    Raises:
        FileNotFoundError: If the planner has not run for this tile.
        ValueError: If the plan was cut under a different configuration.
    """
    apply_shard_settings()
    storage = storage or get_storage()
    root = shards.shard_root(run_id, tile)

    raw_plan = storage.read_text(shards.plan_key(root))
    raw_items = storage.read_text(shards.items_key(root))
    if raw_plan is None or raw_items is None:
        msg = (
            f"no plan for {tile} in run {run_id!r} at {shards.plan_key(root)}; "
            "the resolve stage has not published"
        )
        raise FileNotFoundError(msg)

    from landsat_lst.pipeline import items_from_dicts  # noqa: PLC0415

    # Raises on digest drift. Not caught: a shard that computed against
    # different settings would merge into a tile nothing inspects.
    plan = shards.TilePlan.from_dict(json.loads(raw_plan))
    return ShardContext(
        plan=plan,
        items=items_from_dicts(json.loads(raw_items)),
        job=job_for_window(tile, plan.window),
        root=root,
        storage=storage,
    )


# --------------------------------------------------------------------------
# Stage 0: resolve
# --------------------------------------------------------------------------


def resolve_tile_plan(
    job: ProcessingJob,
    run_id: str,
    *,
    storage: StorageBackend | None = None,
) -> shards.TilePlan:
    """Query the catalog once for the whole tile and freeze where it is cut.

    This is the only process that touches a live catalog. Two shards resolving
    their own scene set can disagree -- STAC is not frozen -- and a tile
    assembled from two scene sets is striped in a way nothing downstream
    inspects, so the items are serialized here and read back by every shard.

    Everything else the plan holds is derived from the same coarse stack the
    offset pass will load, rather than re-derived per shard: the block edge,
    the block spans and which of them hold land, the scene batches, and the row
    bands. Building those graphs computes nothing.

    Idempotent: a plan already published for this run and tile is read back and
    returned, which also re-checks its digest.

    Returns:
        The :class:`~landsat_lst.shards.TilePlan`, also written to storage.
    """
    apply_shard_settings()
    storage = storage or get_storage()
    root = shards.shard_root(run_id, job.tile.name)

    existing = storage.read_text(shards.plan_key(root))
    if existing is not None and storage.read_text(shards.items_key(root)) is not None:
        log.info("shard_plan_exists", tile=job.tile.name, run_id=run_id)
        return shards.TilePlan.from_dict(json.loads(existing))

    from landsat_lst.pipeline import (  # noqa: PLC0415
        _build_land_mask,
        _patch_url_for,
        items_to_dicts,
        load_scenes,
        resolve_items,
    )

    items = resolve_items(job)
    report_phase("loading", scenes_found=len(items))

    patch_url = _patch_url_for(items)
    factor = settings.destripe_offset_resolution_factor
    coarse_geobox = geobox_for_bbox(job.tile.bbox, factor)
    native_geobox = geobox_for_bbox(job.tile.bbox)

    source = load_scenes(
        items, job.tile.bbox, patch_url=patch_url, fail_on_error=False, geobox=coarse_geobox
    )
    lst = convert_to_celsius(apply_qa_mask(source)["lwir11"])

    with timed_section("land_mask"):
        land = _build_land_mask(coarse_geobox, source.latitude, source.longitude)
    land_values = np.asarray(land.values, dtype=bool)

    coarse_shape = (int(source.sizes["latitude"]), int(source.sizes["longitude"]))
    native_shape = (int(native_geobox.shape[0]), int(native_geobox.shape[1]))

    block_edge = _io_block_edge(lst, settings.destripe_unit_memory_gb)
    blocks = shards.block_spans(coarse_shape, block_edge)
    block_has_land = [bool(land_values[y0:y1, x0:x1].any()) for y0, y1, x0, x1 in blocks]
    scene_batches = _scene_batches(lst, settings.destripe_scene_batch)

    ref_shards, scene_shards, band_shards = shards.stage_shard_counts(
        blocks=len(blocks),
        scene_batches=len(scene_batches),
        block_rows=native_shape[0] // settings.cog_blocksize,
    )

    from landsat_lst.offsets import _times_iso  # noqa: PLC0415

    plan = shards.TilePlan(
        tile=job.tile.name,
        window=job.window_label,
        scene_ids=[str(item.id) for item in items],
        scene_times=_times_iso(source.time),
        offset_factor=factor,
        coarse_shape=coarse_shape,
        native_shape=native_shape,
        block_edge=block_edge,
        blocks=blocks,
        block_has_land=block_has_land,
        scene_batches=scene_batches,
        bands=shards.band_edges(native_shape[0], band_shards, settings.cog_blocksize),
        ref_shards=ref_shards,
        scene_shards=scene_shards,
        band_shards=band_shards,
    )

    # Items first. A reader that finds the plan and no items would see a tile it
    # is allowed to start shards for and cannot load.
    storage.write_text(shards.items_key(root), json.dumps(items_to_dicts(items)))
    storage.write_text(shards.plan_key(root), json.dumps(plan.to_dict(), indent=2))
    log.info(
        "shard_plan_written",
        tile=job.tile.name,
        run_id=run_id,
        scenes=len(items),
        blocks=len(blocks),
        ref_shards=ref_shards,
        scene_shards=scene_shards,
        band_shards=band_shards,
        digest=plan.digest,
    )
    return plan


# --------------------------------------------------------------------------
# Stage A: climatology blocks
# --------------------------------------------------------------------------


def climatology_group(plan: shards.TilePlan, index: int) -> tuple[int, list[shards.Span]]:
    """The blocks one phase-A shard owns, and the global index of the first.

    Balanced on land rather than on block count: a block with no land pixel is
    filled with NaN and never read, so an equal-count split on a coastal tile
    would hand one shard a scene-deep read and another almost nothing. The
    groups are contiguous, so one integer offset locates the whole group in the
    plan's block list.
    """
    groups = shards.balance_by_land(plan.blocks, plan.block_has_land, plan.ref_shards)
    start = sum(len(g) for g in groups[:index])
    return start, groups[index]


def _coarse_stack(ctx: ShardContext) -> tuple[xr.DataArray, xr.DataArray]:
    """The land-masked Celsius stack both offset phases reduce, plus its mask."""
    from landsat_lst.pipeline import _build_land_mask, _patch_url_for, load_scenes  # noqa: PLC0415

    patch_url = _patch_url_for(ctx.items)
    geobox = geobox_for_bbox(ctx.job.tile.bbox, ctx.plan.offset_factor)
    report_phase("loading", scenes_found=len(ctx.items))
    source = load_scenes(
        ctx.items, ctx.job.tile.bbox, patch_url=patch_url, fail_on_error=False, geobox=geobox
    )
    with timed_section("land_mask"):
        land = _build_land_mask(geobox, source.latitude, source.longitude)
    return convert_to_celsius(apply_qa_mask(source)["lwir11"]).where(land), land


def run_climatology_shard(
    run_id: str,
    tile: str,
    index: int,
    *,
    storage: StorageBackend | None = None,
) -> list[str]:
    """Reduce this shard's blocks of the 12-month climatology and publish them.

    Each block is uploaded as raw ``.npy``, uncompressed, so the phase-B shards
    can memory-map them into an assembled ``ref`` slice by slice. A block the
    plan marks as holding no land is published as a zero-byte marker instead:
    ``climatology_by_blocks`` fills it with NaN without reading it, and
    uploading ``12 x block^2 x 4 B`` of NaN would cost more than the block cost
    to produce.

    Returns:
        The keys written, which is empty when they all already existed.
    """
    ctx = load_context(run_id, tile, storage=storage)
    start, group = climatology_group(ctx.plan, index)

    wanted = {}
    for offset, span in enumerate(group):
        gi = start + offset
        key = (
            shards.ref_block_key(ctx.root, gi)
            if ctx.plan.block_has_land[gi]
            else shards.ref_marker_key(ctx.root, gi)
        )
        wanted[gi] = (key, span)

    published = ctx.keys("offsets/ref/")
    if all(key in published for key, _ in wanted.values()):
        log.info("shard_skipped", stage="climatology", tile=tile, index=index, blocks=len(group))
        return []

    lst, land = _coarse_stack(ctx)
    with timed_section("destripe_climatology", blocks_total=len(group)):
        ref, _months = climatology_by_blocks(
            lst, block=ctx.plan.block_edge, land_mask=land, spans=group
        )

    written: list[str] = []
    report_phase("uploading", blocks_total=len(group))
    scratch = Path(tempfile.mkdtemp(prefix="lst_shard_ref_"))
    try:
        for gi, (key, (y0, y1, x0, x1)) in wanted.items():
            if not ctx.plan.block_has_land[gi]:
                ctx.storage.write_text(key, "", content_type="application/octet-stream")
            else:
                local = scratch / f"b{gi:04d}.npy"
                np.save(local, ref[:, y0:y1, x0:x1])
                ctx.storage.upload(local, key)
                local.unlink()
            written.append(key)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    log.info("shard_done", stage="climatology", tile=tile, index=index, blocks=len(written))
    return written


# --------------------------------------------------------------------------
# Stage B: per-scene offsets
# --------------------------------------------------------------------------


def offsets_group(plan: shards.TilePlan, index: int) -> list[tuple[int, int]]:
    """The scene ranges one phase-B shard owns.

    A plain contiguous split: every scene costs the same spatial median, so
    there is nothing to balance, and contiguity keeps a shard's reads inside as
    few source time chunks as possible.
    """
    return shards.partition(plan.scene_batches, plan.scene_shards)[index]


def _assemble_ref(ctx: ShardContext, months: np.ndarray, dtype: Any) -> np.ndarray:
    """Rebuild the whole climatology from the phase-A blocks.

    Every phase-B shard needs the whole thing: a scene's offset is the spatial
    median of its anomaly against the reference, over the tile's full
    footprint. At production geometry this is ``12 x 9000^2 x 4 B = 3.89 GB``,
    which is why the blocks are stored uncompressed -- ``np.load`` writes them
    straight into their slice.

    Raises:
        ValueError: If any block is missing. A merge that filled a gap with NaN
            would turn a lost shard into a quietly different estimate.
    """
    height, width = ctx.plan.coarse_shape
    ref = np.empty((months.size, height, width), dtype=dtype)

    published = ctx.keys("offsets/ref/")
    scratch = Path(tempfile.mkdtemp(prefix="lst_shard_refin_"))
    try:
        missing: list[int] = []
        for gi, (y0, y1, x0, x1) in enumerate(ctx.plan.blocks):
            if not ctx.plan.block_has_land[gi]:
                if shards.ref_marker_key(ctx.root, gi) not in published:
                    missing.append(gi)
                    continue
                ref[:, y0:y1, x0:x1] = np.nan
                continue
            key = shards.ref_block_key(ctx.root, gi)
            local = scratch / f"b{gi:04d}.npy"
            if not ctx.storage.download(key, local):
                missing.append(gi)
                continue
            ref[:, y0:y1, x0:x1] = np.load(local)
            local.unlink()
        if missing:
            msg = (
                f"{len(missing)} of {len(ctx.plan.blocks)} climatology blocks are "
                f"missing (first {missing[0]}); a phase-A shard did not publish"
            )
            raise ValueError(msg)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return ref


def run_offsets_shard(
    run_id: str,
    tile: str,
    index: int,
    *,
    storage: StorageBackend | None = None,
) -> str | None:
    """Estimate this shard's scenes' offsets against the assembled climatology.

    One key per shard, naming the scene range it covers, so a listing shows
    coverage without opening anything and a re-run replaces exactly its own
    partial. The merge joins on time coordinates rather than on those ranges.

    Returns:
        The key written, or ``None`` when it already existed.
    """
    ctx = load_context(run_id, tile, storage=storage)
    group = offsets_group(ctx.plan, index)
    key = shards.scene_partial_key(ctx.root, group[0][0], group[-1][1])

    if key in ctx.keys("offsets/scene/"):
        log.info("shard_skipped", stage="offsets", tile=tile, index=index)
        return None

    lst, _land = _coarse_stack(ctx)
    months = np.unique(lst.time.dt.month.values.astype(np.int16))

    with timed_section("destripe_climatology_merge", blocks_total=len(ctx.plan.blocks)):
        ref = _assemble_ref(ctx, months, np.dtype(lst.dtype))

    scenes = sum(stop - start for start, stop in group)
    with timed_section("destripe_offsets", scenes_total=scenes):
        offset, n_valid = _offsets_by_scene(lst, ref, months, batches=group)

    ctx.storage.write_text(key, json.dumps(partial_payload(offset, n_valid)))
    log.info("shard_done", stage="offsets", tile=tile, index=index, scenes=scenes, key=key)
    return key


def merge_offsets(
    run_id: str,
    tile: str,
    *,
    storage: StorageBackend | None = None,
) -> OffsetKey:
    """Assemble the tile's offsets from the phase-B partials and cache them.

    Runs **in the driver**, not on a VM. Its whole input is a few hundred
    kilobytes of JSON and its whole output is ~600 floats; a VM would spend
    more time booting than working.

    What it writes is the ordinary ADR-012 cache record at the canonical
    ``_offsets/`` key -- not a shard artifact in a shard-shaped place. That is
    the seam: the composite shards read it back exactly as a single-VM tile
    would, and the rejection (``scene_keep_mask``, ``rejection_floor``) is
    applied tile-wide and identically by every one of them, because only the
    estimate is ever cached.

    Idempotent: an existing record covering the planned axis is left alone.

    Raises:
        ValueError: If any scene has no partial. A missing shard is ordinary
            (a preempted VM writes nothing) and a merge that emitted NaN for it
            would turn that into a silently thinner composite.
    """
    ctx = load_context(run_id, tile, storage=storage)
    time_coord = _time_coord(ctx.plan)
    key = _offset_key(ctx.plan)
    cache = OffsetCache(storage=ctx.storage, key=key)

    if cache.read(time_coord) is not None:
        log.info("shard_offsets_merge_skipped", tile=tile, key=key.storage_key)
        return key

    prefix = f"{ctx.root}/offsets/scene/"
    partials = []
    for partial_key in sorted(ctx.storage.list_prefix(prefix)):
        raw = ctx.storage.read_text(partial_key)
        if raw is not None:
            partials.append(json.loads(raw))

    offset, n_valid = merge_scene_partials(partials, time_coord)
    # Written through the cache rather than around it, so a whole-tile run over
    # the same scenes finds these and skips its offset pass.
    cache.write(offset, n_valid)
    log.info(
        "shard_offsets_merged",
        tile=tile,
        partials=len(partials),
        scenes=int(offset.sizes["time"]),
        key=key.storage_key,
    )
    return key


# --------------------------------------------------------------------------
# Stage C: composite row bands
# --------------------------------------------------------------------------


def _tile_offsets(ctx: ShardContext) -> tuple[xr.DataArray, xr.DataArray]:
    """The merged estimate, on the tile's full axis rather than the band's.

    Read on the *planned* axis, never on the band's own: a spatial subset can
    lose a time step, and the cache compares the stored times for equality, so
    reading on a thinned axis would miss and silently re-estimate per band --
    a different reference climatology and a seam at every boundary.
    ``debias_with_offsets`` then joins the estimate to the band by coordinate.
    """
    cache = OffsetCache(storage=ctx.storage, key=_offset_key(ctx.plan))
    hit = cache.read(_time_coord(ctx.plan))
    if hit is None:
        msg = (
            f"no merged offsets for {ctx.tile} at {_offset_key(ctx.plan).storage_key}; "
            "the offsets stage has not been merged"
        )
        raise FileNotFoundError(msg)
    return hit


def run_composite_shard(
    run_id: str,
    tile: str,
    index: int,
    *,
    storage: StorageBackend | None = None,
) -> list[str]:
    """Composite one row band of the tile and publish both products' slabs.

    The band loads onto a *slice of the tile's geobox* rather than a grid
    derived from its own bounds, and masks against that same geobox's affine,
    so its pixels are the tile's pixels down to the last bit (ADR-008). Both
    products are written in one ``dask.compute``, so ADR-013's single native
    pass holds inside a shard as it does inside a whole tile.

    The slabs are plain tiled GeoTIFFs, never COGs: overviews belong to the
    assembled tile.

    Returns:
        The keys written, empty when they already existed.
    """
    from landsat_lst.cog import lst_product, qa_product, write_intermediates  # noqa: PLC0415
    from landsat_lst.job import _encode_native  # noqa: PLC0415
    from landsat_lst.pipeline import (  # noqa: PLC0415
        _build_land_mask,
        _patch_url_for,
        compute_annual_composite,
        load_scenes,
    )

    ctx = load_context(run_id, tile, storage=storage)
    start, stop = ctx.plan.bands[index]
    keys = {product: shards.band_key(ctx.root, product, index) for product in PRODUCTS}

    published = ctx.keys("composite/")
    if all(key in published for key in keys.values()):
        log.info("shard_skipped", stage="composite", tile=tile, index=index)
        return []

    offsets = _tile_offsets(ctx) if settings.destripe else None

    geobox = geobox_for_bbox(ctx.job.tile.bbox)[start:stop, :]
    patch_url = _patch_url_for(ctx.items)
    report_phase("loading", scenes_found=len(ctx.items))
    data = load_scenes(
        ctx.items, ctx.job.tile.bbox, patch_url=patch_url, fail_on_error=False, geobox=geobox
    )
    with timed_section("land_mask"):
        land = _build_land_mask(geobox, data.latitude, data.longitude)

    composite = compute_annual_composite(data, land_mask=land, offsets=offsets)
    # The same two lines process_tile applies, for the same reason: ocean must
    # be nodata in the LST band and zero in the counts, and a band that skipped
    # them would differ from the whole tile exactly along its own rows.
    composite["lst_p95"] = composite["lst_p95"].where(land)
    composite["qa_count"] = composite["qa_count"].where(land, 0).astype(np.uint8)
    composite.attrs.update(_tile_attrs(ctx.plan))

    native = _encode_native(composite)
    scratch = Path(tempfile.mkdtemp(prefix="lst_shard_band_"))
    try:
        paths = {product: scratch / f"{product}.tif" for product in PRODUCTS}
        products = [
            lst_product(native, paths["lst_p95"]),
            qa_product(native, paths["qa_count"]),
        ]
        with timed_section("exporting", scenes_found=len(ctx.items)):
            write_intermediates(
                [(p.da, path) for p, path in zip(products, paths.values(), strict=True)]
            )
        report_phase("uploading")
        for product, path in paths.items():
            ctx.storage.upload(path, keys[product])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    log.info("shard_done", stage="composite", tile=tile, index=index, rows=(start, stop))
    return list(keys.values())


# --------------------------------------------------------------------------
# Stage E: export merge
# --------------------------------------------------------------------------


def _tile_attrs(plan: shards.TilePlan) -> dict[str, Any]:
    """The dataset attrs a whole-tile run stamps onto both COGs."""
    job = job_for_window(plan.tile, plan.window)
    return {
        "tile": plan.tile,
        "year": job.year,
        "window": plan.window,
        "scene_count": len(plan.scene_ids),
    }


def _placeholder_native() -> xr.Dataset:
    """A one-pixel stand-in, purely so the two product descriptions can be built.

    :func:`~landsat_lst.cog.finish_product` reads a :class:`~landsat_lst.cog.Product`'s
    nodata, its band descriptions, and its output path; it never touches the
    array, because the raster it finishes was assembled from bands rather than
    computed. Building the products from a placeholder rather than
    reconstructing their ``describe`` callbacks here is deliberate: the band
    descriptions, the scale/offset pair, and the statistics are what a sharded
    tile has to match a single-VM one on, and a second implementation of them
    would drift on one without failing anything.
    """
    coords = {"latitude": [0.0], "longitude": [0.0]}
    return xr.Dataset(
        {
            "lst_p95": xr.DataArray(
                np.zeros((1, 1), dtype=np.uint16), dims=["latitude", "longitude"], coords=coords
            ),
            "qa_count": xr.DataArray(
                np.zeros((12, 1, 1), dtype=np.uint8),
                dims=["month", "latitude", "longitude"],
                coords={"month": np.arange(1, 13), **coords},
            ),
        }
    )


def run_export_merge(
    run_id: str,
    tile: str,
    *,
    storage: StorageBackend | None = None,
    force: bool = False,
) -> list[str]:
    """Stitch the row bands into the tile's two COGs and publish them.

    The merge is a windowed block copy, not a concatenation: band boundaries
    are multiples of the COG block height, so every write lands on a
    destination block edge and nothing larger than a block is resident. The COG
    tail is then :func:`~landsat_lst.cog.finish_product` verbatim -- the same
    statistics, tags, and overviews a single-VM tile would have written.

    Completion is unchanged: both assets at the canonical
    :meth:`~landsat_lst.storage.StorageBackend.cog_key` keys.

    Returns:
        The COG keys written, empty when the tile was already complete.

    Raises:
        FileNotFoundError: If any band slab is missing.
    """
    from landsat_lst.cog import (  # noqa: PLC0415
        finish_product,
        lst_product,
        merge_bands,
        qa_product,
    )

    ctx = load_context(run_id, tile, storage=storage)
    window = ctx.plan.window

    if not force and ctx.storage.cog_exists(window, tile):
        log.info("shard_skipped", stage="export", tile=tile)
        return []

    attrs = _tile_attrs(ctx.plan)
    placeholder = _placeholder_native()
    bands = ctx.plan.bands
    written: list[str] = []

    scratch = Path(tempfile.mkdtemp(prefix="lst_shard_merge_"))
    try:
        for product in PRODUCTS:
            report_phase("exporting")
            with timed_section("exporting", blocks_total=len(bands)):
                slabs = []
                for index in range(len(bands)):
                    key = shards.band_key(ctx.root, product, index)
                    local = scratch / f"{product}_band{index:03d}.tif"
                    if not ctx.storage.download(key, local):
                        msg = f"band slab {key} is missing; a composite shard did not publish"
                        raise FileNotFoundError(msg)
                    slabs.append(local)

                merged = merge_bands(slabs, scratch / f"{product}_merged.tif", bands)
                for slab in slabs:
                    slab.unlink(missing_ok=True)

                builder = lst_product if product == "lst_p95" else qa_product
                cog = finish_product(
                    merged, builder(placeholder, scratch / f"{product}.tif"), attrs
                )
                merged.unlink(missing_ok=True)

            report_phase("uploading")
            key = ctx.storage.cog_key(window, tile, product)
            ctx.storage.upload(cog, key)
            cog.unlink(missing_ok=True)
            written.append(key)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    log.info("shard_done", stage="export", tile=tile, keys=written)
    return written


# --------------------------------------------------------------------------
# Task entry point
# --------------------------------------------------------------------------


def run_shard(
    stage: str,
    run_id: str,
    tile: str,
    index: int,
    *,
    job: ProcessingJob | None = None,
    storage: StorageBackend | None = None,
) -> Any:
    """Run one shard as a batch task, with its log and its heartbeat.

    The instrumentation is the whole reason this wrapper exists. A batch task
    never registers with a dask scheduler, its stdout never reaches
    ``coiled logs``, and the exit code Coiled records is the tee wrapper's
    (ADR-014), so a shard is only visible through what it publishes: a state
    object it rewrites every ``heartbeat_interval_s``, and its own stdout and
    stderr uploaded on the way out either way. Both go under this tile's shard
    prefix rather than under ``_runs/``, where ``runs.classify`` would read them
    as tile attempts.

    Args:
        stage: One of :data:`landsat_lst.shards.STAGES`.
        run_id: Run token.
        tile: Tile name.
        index: Which shard of the stage. Always 0 for ``resolve`` and
            ``export``, which are single tasks; passed anyway so every stage has
            one command shape.
        job: The job to resolve, required by ``resolve`` and unused elsewhere
            (every other stage reads the window from the plan).
        storage: Backend. Defaults to the configured one.

    Returns:
        Whatever the stage's function returns.
    """
    configure_logging()
    apply_shard_settings()
    storage = storage or get_storage()
    root = shards.shard_root(run_id, tile)

    # Once per process. Asking twice would number the log above the state
    # object, since the log uploads last.
    attempt = shards.resolve_shard_attempt(storage, root, stage, index)
    heartbeat_job = job or _heartbeat_job(storage, root, tile)

    with ExitStack() as stack:
        stack.enter_context(
            capture_task_log(
                run_id=run_id,
                tile=f"{tile}.{stage}.{index:04d}",
                storage=storage,
                attempt=attempt,
                key=shards.shard_log_key(root, stage, index, attempt),
            )
        )
        stack.enter_context(
            TileHeartbeat(
                run_id=run_id,
                job=heartbeat_job,
                storage=storage,
                attempt=attempt,
                key=shards.shard_state_key(root, stage, index, attempt),
            )
        )
        report_phase(f"shard_{stage}")
        return _dispatch(stage, run_id, tile, index, job=job, storage=storage)


def _heartbeat_job(storage: StorageBackend, root: str, tile: str) -> ProcessingJob:
    """A job for the heartbeat's identity fields, best-effort.

    The heartbeat needs a window before the stage's own plan read happens, and
    losing observability must never fail a shard, so a plan that cannot be read
    yields a placeholder rather than an exception. The stage that follows will
    raise on the same missing plan with a message about the plan.
    """
    try:
        raw = storage.read_text(shards.plan_key(root))
        if raw is not None:
            return job_for_window(tile, str(json.loads(raw)["window"]))
    except Exception as e:  # pragma: no cover - instrumentation never fails a shard
        log.warning("shard_heartbeat_job_failed", tile=tile, error=str(e))

    from landsat_lst.job import DEFAULT_WINDOW  # noqa: PLC0415

    return ProcessingJob(
        tile=parse_tile_name(tile), year=DEFAULT_WINDOW[0], end_year=DEFAULT_WINDOW[1]
    )


def _dispatch(
    stage: str,
    run_id: str,
    tile: str,
    index: int,
    *,
    job: ProcessingJob | None,
    storage: StorageBackend | None,
) -> Any:
    if stage == "resolve":
        if job is None:
            msg = "the resolve stage needs the job it is resolving"
            raise ValueError(msg)
        return resolve_tile_plan(job, run_id, storage=storage)
    if stage == "climatology":
        return run_climatology_shard(run_id, tile, index, storage=storage)
    if stage == "offsets":
        return run_offsets_shard(run_id, tile, index, storage=storage)
    if stage == "composite":
        return run_composite_shard(run_id, tile, index, storage=storage)
    if stage == "export":
        return run_export_merge(run_id, tile, storage=storage)
    msg = f"unknown shard stage {stage!r}; expected one of {shards.STAGES}"
    raise ValueError(msg)
