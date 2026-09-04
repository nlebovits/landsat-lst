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
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog
import xarray as xr

from landsat_lst import offsets, shards
from landsat_lst.config import settings
from landsat_lst.exectrace import exec_trace
from landsat_lst.logging_config import configure_logging
from landsat_lst.models import ProcessingJob
from landsat_lst.normalization import (
    _io_block_edge,
    _read_values,
    _scene_batches,
    climatology_by_blocks,
)
from landsat_lst.normalization import offsets_by_scene as _offsets_by_scene
from landsat_lst.offsets import (
    OffsetCache,
    OffsetKey,
    merge_scene_partials,
    partial_payload,
)
from landsat_lst.profiling import PROFILE_COMPOSITE, profile_compute
from landsat_lst.progress import TileHeartbeat, capture_task_log, report_phase, timed_section
from landsat_lst.qa import (
    DN_SENTINEL,
    apply_qa_mask,
    celsius_stack,
    convert_to_celsius,
    dn_stack,
)
from landsat_lst.staging import (
    CoarseStage,
    StageKey,
    stage_batches,
    staged_batch_reader,
    staging_block_reader,
)
from landsat_lst.storage import PRODUCTS, get_storage
from landsat_lst.tiling import geobox_for_bbox, parse_tile_name

if TYPE_CHECKING:
    from collections.abc import Sequence

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
        run_id: str,
        root: str,
        storage: StorageBackend,
    ) -> None:
        self.plan = plan
        self.items = items
        self.job = job
        self.run_id = run_id
        self.root = root
        self.storage = storage

    @property
    def tile(self) -> str:
        return self.plan.tile

    def keys(self, prefix: str = "") -> set[str]:
        """Every key this tile's shards have published under ``prefix``."""
        return set(self.storage.list_prefix(f"{self.root}/{prefix}"))


def _item_time_values(items: list) -> list[np.datetime64]:
    """Every item's nominal timestamp, naive UTC, as ``odc-stac`` resolves it.

    ``odc-stac`` sets each group's time coordinate to
    ``group[0].nominal_datetime.replace(tzinfo=None)``, and a group's
    representative is always *some* item -- so whatever the solar-day grouping
    decided, the value it chose is in this list. That is what makes recovering
    a truncated axis possible without re-implementing the grouping, which is
    version-coupled and which nothing here would notice getting wrong.

    ``nominal_datetime`` falls back to ``start_datetime`` and then
    ``end_datetime``; this mirrors that order.
    """
    values: list[np.datetime64] = []
    for item in items:
        stamp = getattr(item, "datetime", None)
        if stamp is None:
            props = getattr(item, "properties", {}) or {}
            raw = props.get("start_datetime") or props.get("end_datetime")
            if raw is None:
                continue
            stamp = raw
        moment = pd.Timestamp(stamp)
        if moment.tzinfo is not None:
            moment = moment.tz_convert("UTC").tz_localize(None)
        values.append(np.datetime64(moment.to_datetime64()))
    return values


def upgrade_legacy_scene_times(plan: shards.TilePlan, items: list) -> shards.TilePlan:
    """Restore the sub-second component a pre-2026-08-22 planner truncated away.

    Plans written before the nanosecond fix froze ``scene_times`` at second
    precision. :func:`_time_coord` rebuilds the offset axis from those strings,
    and a composite shard then joins it against a stack loaded at full
    precision -- which raises exactly the error the record-side fix was supposed
    to have ended. The S30W065 plan is one of these, and a resumed run reads it
    rather than writing a new one, so the record fix alone left the whole chain
    blocked.

    The recovery is derived, not guessed, and the derivation is exact for a
    reason worth writing down. ``items.json`` holds one entry per *scene* while
    the axis holds one per *solar-day group*, so several items routinely fall
    inside one second -- adjacent WRS rows of one overpass are seconds apart.
    Where they do, the group's representative is the **earliest** of them:
    ``odc-stac`` sorts each group by ``(group_key, nominal_datetime, id)`` and
    takes ``group[0]``. Items within one second are necessarily on the same
    date and therefore in the same solar-day group, so the earliest candidate
    in a second *is* that group's timestamp. Taking the minimum is a proof, not
    a preference.

    What stays a hard error is ambiguity the items cannot resolve: two entries
    in the stored axis that truncate to the same second. The plan then fits
    more than one real axis, and there is nothing to read it from -- the same
    discipline as :func:`landsat_lst.offsets._truncation_of`, which treats an
    ambiguous truncation as a miss rather than a reading. So is a stamp no item
    matches at all, which means the plan and the item list disagree.

    The digest is untouched, and cannot move: it covers the scene *ids* and the
    settings, never the stamps (see :meth:`landsat_lst.shards.TilePlan.digest`).
    A legacy plan therefore still verifies against a current process.

    Args:
        plan: The plan as stored.
        items: The tile's resolved STAC items, from ``items.json``.

    Returns:
        The plan unchanged when its stamps already carry a fraction, else a
        copy whose ``scene_times`` are full precision.

    Raises:
        ValueError: If two entries in the stored axis truncate to the same
            second, or if a stored stamp matches no item at all.
    """
    stored = list(plan.scene_times)
    if not stored or any("." in stamp for stamp in stored):
        return plan
    if len(set(stored)) != len(stored):
        duplicate = next(s for s in stored if stored.count(s) > 1)
        msg = (
            f"the plan for {plan.tile} truncates ambiguously at {duplicate}: two of "
            "its time steps fall inside that second, so the stored axis fits more "
            "than one real one. Re-plan the tile rather than guessing which."
        )
        raise ValueError(msg)

    candidates: dict[str, set[np.datetime64]] = {}
    for value in _item_time_values(items):
        second = str(np.datetime_as_string(value, unit=offsets.LEGACY_TIME_UNIT))
        candidates.setdefault(second, set()).add(value)

    upgraded: list[np.datetime64] = []
    for stamp in stored:
        matches = candidates.get(stamp, set())
        if not matches:
            msg = (
                f"the plan for {plan.tile} carries {stamp}, which no item in "
                "items.json matches; the plan and the item list disagree"
            )
            raise ValueError(msg)
        # The earliest, which is the group's representative -- see the note
        # above on why several items inside one second are ordinary and why
        # ``min`` is exact rather than a preference.
        upgraded.append(min(matches))

    times = xr.DataArray(np.array(upgraded, dtype="datetime64[ns]"), dims=["time"])
    log.info(
        "offset_plan_legacy_precision",
        tile=plan.tile,
        window=plan.window,
        scenes=len(stored),
        note="plan stamps stored at second precision; recovered from items.json",
    )
    return replace(plan, scene_times=offsets._times_iso(times))


def load_context(run_id: str, tile: str, *, storage: StorageBackend | None = None) -> ShardContext:
    """Read the plan and the frozen item list, refusing a plan cut elsewhere.

    Raises:
        FileNotFoundError: If the planner has not run for this tile.
        ValueError: If the plan was cut under a different configuration, or if
            a legacy plan's truncated time axis cannot be recovered.
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
    items = items_from_dicts(json.loads(raw_items))
    # After the digest check, and with the items in hand: a plan written before
    # the nanosecond fix carries a truncated time axis that no join can match.
    plan = upgrade_legacy_scene_times(plan, items)
    return ShardContext(
        plan=plan,
        items=items,
        job=job_for_window(tile, plan.window),
        run_id=run_id,
        root=root,
        storage=storage,
    )


# --------------------------------------------------------------------------
# Stage 0: resolve
# --------------------------------------------------------------------------


def wait_for_key(
    storage: StorageBackend,
    key: str,
    *,
    timeout_s: float,
    what: str,
    poll_s: float | None = None,
) -> bool:
    """Poll for one object, in a process that is already booted.

    The consolidation's whole premise: a booted VM waiting is cheap, and a
    fleet booting again is not. Offsets-side shards computed ~6 minutes each
    while their stages held fleets ~30, so every wait here replaces a boot.

    Bounded, and loud when it expires. An unbounded wait would turn a stage
    whose predecessor died into a fleet idling until ``coiled_job_timeout`` --
    the same "fails as a hang" shape the backend mismatch had.

    Returns:
        Whether the key appeared before the deadline.
    """
    poll = settings.shard_unit_poll_s if poll_s is None else poll_s
    deadline = time.time() + timeout_s
    waited = 0.0
    while True:
        if storage.read_text(key) is not None:
            if waited:
                log.info("shard_wait_satisfied", what=what, key=key, waited_s=round(waited, 1))
            return True
        if time.time() >= deadline:
            log.warning("shard_wait_expired", what=what, key=key, timeout_s=timeout_s)
            return False
        time.sleep(poll)
        waited += poll


def resolve_tile_plan(
    job: ProcessingJob,
    run_id: str,
    *,
    units: int | None = None,
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
    block_weights = shards.block_scene_weights(
        blocks, block_has_land, _footprints(items), coarse_geobox.affine
    )
    scene_batches = _scene_batches(lst, settings.destripe_scene_batch)

    ref_shards, scene_shards, band_shards = shards.stage_shard_counts(
        blocks=len(blocks),
        scene_batches=len(scene_batches),
        block_rows=native_shape[0] // settings.cog_blocksize,
        # The offsets-side widths are the fused fleet's width, which the driver
        # fixed before this plan existed. See stage_shard_counts.
        units=units,
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
        block_weights=block_weights,
    )

    # Items first. A reader that finds the plan and no items would see a tile it
    # is allowed to start shards for and cannot load.
    storage.write_text(shards.items_key(root), json.dumps(items_to_dicts(items)))
    storage.write_text(shards.plan_key(root), json.dumps(plan.to_dict(), indent=2))
    group_weights = [sum(block_weights[i] for i in idx) for idx in _group_indexes(plan)]
    log.info(
        "shard_plan_written",
        tile=job.tile.name,
        run_id=run_id,
        scenes=len(items),
        blocks=len(blocks),
        ref_shards=ref_shards,
        scene_shards=scene_shards,
        band_shards=band_shards,
        block_weight_min=min(block_weights),
        block_weight_max=max(block_weights),
        group_weight_min=min(group_weights),
        group_weight_max=max(group_weights),
        digest=plan.digest,
    )
    return plan


def _footprints(items: Sequence[Any]) -> list[Any]:
    """Scene footprints as shapely geometries, ``None`` where an item has none."""
    from shapely.geometry import box, shape  # noqa: PLC0415

    out: list[Any] = []
    for item in items:
        geometry = getattr(item, "geometry", None)
        if geometry is not None:
            out.append(shape(geometry))
            continue
        bbox = getattr(item, "bbox", None)
        out.append(None if bbox is None else box(*bbox))
    return out


def _group_indexes(plan: shards.TilePlan) -> list[range]:
    """Global block indexes of each phase-A group, in group order."""
    out: list[range] = []
    start = 0
    for group in shards.climatology_groups(plan):
        out.append(range(start, start + len(group)))
        start += len(group)
    return out


# --------------------------------------------------------------------------
# Stage A: climatology blocks
# --------------------------------------------------------------------------


def climatology_group(plan: shards.TilePlan, index: int) -> tuple[int, list[shards.Span]]:
    """The blocks one phase-A shard owns, and the global index of the first.

    Balanced on what each block reads rather than on block count: the scene
    footprints crossing it when the plan stores them, the land flag when it
    does not (:func:`landsat_lst.shards.climatology_groups`). A block with no
    land pixel is filled with NaN and never read, so an equal-count split on
    a coastal tile would hand one shard a scene-deep read and another almost
    nothing. The groups are contiguous, so one integer offset locates the
    whole group in the plan's block list.
    """
    groups = shards.climatology_groups(plan)
    start = sum(len(g) for g in groups[:index])
    return start, groups[index]


def _coarse_stack(ctx: ShardContext) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """The land-masked stack both offset phases reduce, its mask, and its DN.

    The Celsius array is built *through* the DN rather than beside it, so the
    two cannot drift: ``celsius_stack(dn_stack(...))`` is bit-identical to the
    ``convert_to_celsius(apply_qa_mask(...))`` this replaced, which
    ``tests/unit/test_dn_stack.py`` pins. The DN is what the staging path
    (issue #125) publishes; a run with staging off never materializes it.

    Returns:
        ``(lst, land, dn)``. ``lst`` is the float32 Celsius stack the estimator
        reads, ``land`` its mask, and ``dn`` the ``uint16`` carrier, all lazy.
    """
    from landsat_lst.pipeline import _build_land_mask, _patch_url_for, load_scenes  # noqa: PLC0415

    patch_url = _patch_url_for(ctx.items)
    geobox = geobox_for_bbox(ctx.job.tile.bbox, ctx.plan.offset_factor)
    report_phase("loading", scenes_found=len(ctx.items))
    source = load_scenes(
        ctx.items, ctx.job.tile.bbox, patch_url=patch_url, fail_on_error=False, geobox=geobox
    )
    with timed_section("land_mask"):
        land = _build_land_mask(geobox, source.latitude, source.longitude)
    dn = dn_stack(source).where(land, DN_SENTINEL)
    if dn.dtype != np.uint16:  # pragma: no cover - xarray keeps uint16 for an int fill
        dn = dn.astype(np.uint16)
    return celsius_stack(dn), land, dn


def coarse_stage(ctx: ShardContext) -> CoarseStage:
    """This tile-window's stage, keyed by the offsets record it feeds.

    Both phases build it the same way from the plan, so neither has to be told
    where the other put things, and a stage written under a different scene
    set, factor, clamp, or ``offsets.ALGORITHM_VERSION`` is at a prefix this
    one never lists.
    """
    return CoarseStage(ctx.storage, StageKey.from_offset_key(ctx.root, _offset_key(ctx.plan)))


def _sweep_stage(ctx: ShardContext) -> int:
    """Delete this tile's coarse stage. Never raises.

    Called when the offsets record lands and when the driver gives a tile up.
    A stage that outlives its tile is an object under the run prefix that a
    later listing reads as finished work, which is the failure shape
    ``runs.classify`` already warns about one level up.
    """
    if not settings.destripe_stage_coarse:
        return 0
    try:
        return coarse_stage(ctx).cleanup()
    except Exception as exc:  # instrumentation never fails a tile
        log.warning("coarse_stage_cleanup_failed", tile=ctx.tile, error=repr(exc)[:200])
        return 0


def sweep_coarse_stage(run_id: str, tile: str, *, storage: StorageBackend | None = None) -> int:
    """Sweep one tile's stage from outside a shard, for a tile that failed."""
    try:
        return _sweep_stage(load_context(run_id, tile, storage=storage))
    except Exception as exc:  # a tile with no plan has no stage
        log.warning("coarse_stage_sweep_failed", tile=tile, error=repr(exc)[:200])
        return 0


def run_climatology_shard(
    run_id: str,
    tile: str,
    index: int,
    *,
    storage: StorageBackend | None = None,
    ctx: ShardContext | None = None,
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
    ctx = ctx or load_context(run_id, tile, storage=storage)
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

    lst, land, dn = _coarse_stack(ctx)

    # The staging seam (issue #125). The reader decodes the sources once, in
    # bounded scene groups, publishes the DN it decoded, and hands back the
    # float32 block phase A was going to read anyway. Only blocks this shard
    # actually reduces are staged, so a land-free block is never staged and
    # never read -- phase B reconstructs it as the all-NaN it already was.
    block_reader = None
    stage = None
    if settings.destripe_stage_coarse:
        stage = coarse_stage(ctx)
        index_of = {span: start + offset for offset, span in enumerate(group)}
        block_reader = staging_block_reader(
            dn,
            stage,
            block_index=lambda span: index_of[span],
            batches=stage_batches(dn),
            read_values=_read_values,
        )

    with timed_section("destripe_climatology", blocks_total=len(group)):
        ref, _months = climatology_by_blocks(
            lst,
            block=ctx.plan.block_edge,
            land_mask=land,
            spans=group,
            block_reader=block_reader,
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
    ctx: ShardContext | None = None,
) -> str | None:
    """Estimate this shard's scenes' offsets against the assembled climatology.

    One key per shard, naming the scene range it covers, so a listing shows
    coverage without opening anything and a re-run replaces exactly its own
    partial. The merge joins on time coordinates rather than on those ranges.

    Returns:
        The key written, or ``None`` when it already existed.
    """
    ctx = ctx or load_context(run_id, tile, storage=storage)
    group = offsets_group(ctx.plan, index)
    key = shards.scene_partial_key(ctx.root, group[0][0], group[-1][1])

    if key in ctx.keys("offsets/scene/"):
        log.info("shard_skipped", stage="offsets", tile=tile, index=index)
        return None

    lst, _land, dn = _coarse_stack(ctx)
    months = np.unique(lst.time.dt.month.values.astype(np.int16))

    with timed_section("destripe_climatology_merge", blocks_total=len(ctx.plan.blocks)):
        ref = _assemble_ref(ctx, months, np.dtype(lst.dtype))

    # The staging seam (issue #125): rebuild each scene batch from what phase A
    # already decoded rather than reading every Landsat source a second time.
    batch_reader = None
    if settings.destripe_stage_coarse:
        batch_reader = staged_batch_reader(
            coarse_stage(ctx),
            blocks=list(ctx.plan.blocks),
            block_has_land=list(ctx.plan.block_has_land),
            batches=stage_batches(dn),
            shape=tuple(ctx.plan.coarse_shape),
        )

    scenes = sum(stop - start for start, stop in group)
    with timed_section("destripe_offsets", scenes_total=scenes):
        offset, n_valid = _offsets_by_scene(
            lst, ref, months, batches=group, batch_reader=batch_reader
        )

    ctx.storage.write_text(key, json.dumps(partial_payload(offset, n_valid)))
    log.info("shard_done", stage="offsets", tile=tile, index=index, scenes=scenes, key=key)
    return key


def _block_keys(plan: shards.TilePlan, root: str) -> list[str]:
    """Every phase-A artifact the tile expects, land block or ocean marker."""
    return [
        shards.ref_block_key(root, i) if has_land else shards.ref_marker_key(root, i)
        for i, has_land in enumerate(plan.block_has_land)
    ]


def wait_for_blocks(ctx: ShardContext, *, timeout_s: float | None = None) -> bool:
    """Poll until every peer's climatology block has landed.

    The in-process phase-A barrier. Phase B measures each scene against the
    *whole* climatology, so it cannot start on a partial one -- but the process
    holding that requirement is already booted, and a second fleet exists only
    to re-establish that fact. Waiting here costs time; a separate stage costs
    time *and* a boot per shard.

    One listing per poll rather than one read per block: a production tile has
    324 of these.

    Returns:
        Whether every block appeared before the deadline.
    """
    limit = settings.shard_block_wait_s if timeout_s is None else timeout_s
    wanted = set(_block_keys(ctx.plan, ctx.root))
    deadline = time.time() + limit
    while True:
        present = ctx.keys("offsets/ref/")
        missing = wanted - present
        report_phase(
            "shard_barrier_wait",
            blocks_done=len(wanted) - len(missing),
            blocks_total=len(wanted),
        )
        if not missing:
            return True
        if time.time() >= deadline:
            log.warning(
                "shard_block_barrier_expired",
                tile=ctx.tile,
                missing=len(missing),
                total=len(wanted),
                first=sorted(missing)[0],
            )
            return False
        time.sleep(settings.shard_unit_poll_s)


def run_offsets_stage(
    run_id: str,
    tile: str,
    index: int,
    *,
    job: ProcessingJob | None = None,
    units: int | None = None,
    storage: StorageBackend | None = None,
) -> str | None:
    """The whole offsets side of a tile, on one VM, in one boot.

    Resolve, climatology, phase-A barrier, per-scene offsets -- four things
    that used to be three fleets and are now four sub-phases of one task. The
    measurement that forced it: an offsets-side shard computed for about six
    minutes while its stage held a fleet for about thirty. Boots and queueing
    dominated, and every barrier between those stages was paid for twice, once
    in the driver's poll and once in the next fleet's boot.

    What each shard does, by index:

    - **Shard 0 resolves.** It runs the one STAC query and publishes
      ``items.json`` and ``plan.json``. Exactly one process may do this: two
      would resolve two scene sets from a live catalog and the tile would be
      assembled from both.
    - **Every shard waits for that plan**, bounded. This is the barrier that
      used to be a whole stage boundary.
    - **Every shard reduces its climatology blocks**, then waits at the
      in-process phase-A barrier for its peers'.
    - **Every shard estimates its scenes' offsets.**

    A shard whose index falls past a phase's clamped shard count has nothing to
    do in that phase and says so rather than failing: the fleet's width is
    fixed before the plan exists, so it can exceed the work the tile holds.

    Every sub-phase still checks its own outputs first, so a retried fused task
    skips what it already finished -- the resolve is a plan read, the blocks it
    published are skipped, and it goes straight back to where it died.

    Returns:
        The scene-partial key this shard wrote, or ``None`` when it had none to
        write or it already existed.

    Raises:
        RuntimeError: If the plan or the phase-A barrier never arrives.
    """
    storage = storage or get_storage()
    root = shards.shard_root(run_id, tile)

    # Only when there is no plan yet. A retry of shard 0, and every shard of a
    # resumed run, finds one already there and must not need the job to reach
    # its own work -- a resume rebuilds the tile from the plan precisely so it
    # never resolves a second scene set.
    if index == 0 and storage.read_text(shards.plan_key(root)) is None:
        if job is None:
            msg = (
                f"no plan for {tile} and no job to resolve one from; shard 0 of the "
                "offsets stage carries the window, and this invocation did not"
            )
            raise ValueError(msg)
        with timed_section("shard_resolve"):
            resolve_tile_plan(job, run_id, units=units, storage=storage)

    with timed_section("shard_plan_wait"):
        if not wait_for_key(
            storage,
            shards.plan_key(root),
            timeout_s=settings.shard_plan_wait_s,
            what="tile plan",
        ):
            msg = (
                f"no plan for {tile} at {shards.plan_key(root)} after "
                f"{settings.shard_plan_wait_s}s; shard 0 never published one"
            )
            raise RuntimeError(msg)

    ctx = load_context(run_id, tile, storage=storage)

    if index < ctx.plan.ref_shards:
        run_climatology_shard(run_id, tile, index, storage=storage, ctx=ctx)
    else:
        log.info(
            "shard_phase_skipped",
            stage="climatology",
            tile=tile,
            index=index,
            shards=ctx.plan.ref_shards,
            note="fleet is wider than the work",
        )

    with timed_section("shard_barrier_wait"):
        if not wait_for_blocks(ctx):
            msg = (
                f"the phase-A climatology for {tile} is incomplete after "
                f"{settings.shard_block_wait_s}s; a peer never published its blocks"
            )
            raise RuntimeError(msg)

    if index >= ctx.plan.scene_shards:
        log.info(
            "shard_phase_skipped",
            stage="offsets",
            tile=tile,
            index=index,
            shards=ctx.plan.scene_shards,
            note="fleet is wider than the work",
        )
        return None

    return run_offsets_shard(run_id, tile, index, storage=storage, ctx=ctx)


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
    # The record exists, so the stage has nothing left to answer for. Sweeping
    # here rather than in a shard because only the driver knows every peer is
    # done reading, and best-effort because losing 475 GB of scratch to a
    # lifecycle rule costs less than failing a tile that already succeeded.
    _sweep_stage(ctx)
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

    **Waits rather than refuses.** The driver starts the composite fleet while
    phase B is still running (``shard_composite_overlap``), precisely so these
    VMs boot on somebody else's time. A shard that refused on arrival would
    burn the boot the overlap exists to save, so it polls to
    ``shard_offsets_record_wait_s`` instead -- and then fails loudly, because
    waiting forever is the hang shape this project keeps paying for.
    """
    key = _offset_key(ctx.plan)
    with timed_section("shard_offsets_wait"):
        found = wait_for_key(
            ctx.storage,
            key.storage_key,
            timeout_s=settings.shard_offsets_record_wait_s,
            what="merged offsets",
        )
    if not found:
        msg = (
            f"no merged offsets for {ctx.tile} at {key.storage_key} after "
            f"{settings.shard_offsets_record_wait_s}s; the offsets stage never merged"
        )
        raise FileNotFoundError(msg)

    cache = OffsetCache(storage=ctx.storage, key=key)
    hit = cache.read(_time_coord(ctx.plan))
    if hit is None:
        msg = (
            f"the merged offsets for {ctx.tile} at {key.storage_key} do not cover "
            "the planned time axis"
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
    from landsat_lst.cog import (  # noqa: PLC0415
        lst_product,
        qa_product,
        write_intermediates_bounded,
    )
    from landsat_lst.job import _encode_native, _thread_cap  # noqa: PLC0415
    from landsat_lst.pipeline import (  # noqa: PLC0415
        _build_ged_gap_mask,
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
        ctx.items,
        ctx.job.tile.bbox,
        patch_url=patch_url,
        fail_on_error=False,
        geobox=geobox,
        per_column=settings.shard_composite_per_column,
    )
    with timed_section("land_mask"):
        land = _build_land_mask(geobox, data.latitude, data.longitude)

    composite = compute_annual_composite(data, land_mask=land, offsets=offsets)
    # The same two lines process_tile applies, for the same reason: ocean must
    # be nodata in the LST band and zero in the counts, and a band that skipped
    # them would differ from the whole tile exactly along its own rows.
    composite["lst_p95"] = composite["lst_p95"].where(land)
    composite["qa_count"] = composite["qa_count"].where(land, 0).astype(np.uint8)
    # The GED gap mask a whole tile applies, on the band's slice of the tile's
    # grid -- gap_mask_for_geobox reads the geobox's own affine, so a band's
    # mask is the exact slice of the tile's and the seams stay invisible
    # (ADR-008). LST only; qa_count stays the evidence layer.
    if settings.ged_gap_mask:
        with timed_section("ged_gap_mask"):
            gap = _build_ged_gap_mask(geobox, data.latitude, data.longitude)
        composite["lst_p95"] = composite["lst_p95"].where(~gap)
    composite.attrs.update(_tile_attrs(ctx.plan))

    native = _encode_native(composite)
    scratch = Path(tempfile.mkdtemp(prefix="lst_shard_band_"))
    try:
        paths = {product: scratch / f"{product}.tif" for product in PRODUCTS}
        products = [
            lst_product(native, paths["lst_p95"]),
            qa_product(native, paths["qa_count"]),
        ]
        # settings.dask_max_threads (LST_DASK_MAX_THREADS) bounds the threaded
        # scheduler here exactly as process_tile_job bounds a whole tile; None
        # leaves dask's CPU-count pool, which is what every production shard
        # has run on so far.
        with (
            _thread_cap(),
            timed_section("exporting", scenes_found=len(ctx.items)),
            profile_compute(PROFILE_COMPOSITE),
            exec_trace(
                storage=ctx.storage,
                stem=shards.unit_trace_prefix(run_id, "composite", tile, index),
            ),
        ):
            write_intermediates_bounded(
                [(p.da, path) for p, path in zip(products, paths.values(), strict=True)],
                longitude_group=2 * settings.load_chunk_size,
            )
        report_phase("uploading")
        for product, path in paths.items():
            ctx.storage.upload(path, keys[product])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    log.info("shard_done", stage="composite", tile=tile, index=index, rows=(start, stop))

    claim_export(ctx, index)
    return list(keys.values())


def claim_export(ctx: ShardContext, index: int) -> bool:
    """Run the export here if this worker wrote the last band.

    The export is one task at the end of a wide stage. Submitting it as its own
    fleet costs a whole VM boot -- and a queue wait, and a plan read -- to do a
    merge that the worker which just finished the last band is already booted
    and warm for.

    **The claim is not a lock.** The export is idempotent at the canonical COG
    keys, so two workers racing produce the same two objects: a lost race costs
    duplicated work, never a corrupted tile. That is why this is a plain write
    with no compare-and-set. S3 offers none, and synthesizing one out of
    listings would add a failure mode to save a few minutes of one VM. The
    claim makes duplication rare; the driver's fallback covers a claim that is
    written and never executed, because the claiming VM was preempted.

    Never fails the shard. Its bands are already published, which is what the
    shard was for; an export that does not happen here happens in the fallback.

    Returns:
        Whether this worker ran the export.
    """
    try:
        bands = {
            shards.band_key(ctx.root, product, i)
            for product in PRODUCTS
            for i in range(len(ctx.plan.bands))
        }
        if bands - ctx.keys("composite/"):
            return False

        claim = shards.export_claim_key(ctx.root)
        if ctx.storage.read_text(claim) is not None:
            log.info("shard_export_already_claimed", tile=ctx.tile, index=index)
            return False

        ctx.storage.write_text(
            claim,
            json.dumps(
                {
                    "tile": ctx.tile,
                    "claimed_by_band": index,
                    "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            ),
        )
        log.info("shard_export_claimed", tile=ctx.tile, index=index)
    except Exception as e:  # pragma: no cover - claiming never fails a shard
        log.warning("shard_export_claim_failed", tile=ctx.tile, index=index, error=str(e))
        return False

    try:
        run_export_merge(ctx.run_id, ctx.tile, storage=ctx.storage)
    except Exception as e:
        # The driver's fallback exists for exactly this. Failing the shard now
        # would also fail its bands, which are already in the bucket and fine.
        log.warning("shard_export_failed", tile=ctx.tile, index=index, error=str(e))
        return False
    return True


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
    units: int | None = None,
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
        job: The job to resolve. Required by ``resolve`` and by shard 0 of the
            fused ``offsets`` stage, which resolves before it reduces; unused
            elsewhere, since every other stage reads the window from the plan.
        units: The fused offsets fleet's width, which the plan is cut to. Only
            the shard that writes the plan reads it.
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
                profile_key=lambda label: shards.shard_profile_key(
                    root, stage, index, attempt, label
                ),
            )
        )
        report_phase(f"shard_{stage}")
        return _dispatch(stage, run_id, tile, index, job=job, units=units, storage=storage)


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
    units: int | None,
    storage: StorageBackend | None,
) -> Any:
    if stage == "resolve":
        if job is None:
            msg = "the resolve stage needs the job it is resolving"
            raise ValueError(msg)
        return resolve_tile_plan(job, run_id, units=units, storage=storage)
    if stage == "climatology":
        return run_climatology_shard(run_id, tile, index, storage=storage)
    if stage == "offsets":
        # Fused: resolve (shard 0), climatology, phase-A barrier, offsets. One
        # fleet, one boot. See run_offsets_stage.
        return run_offsets_stage(run_id, tile, index, job=job, units=units, storage=storage)
    if stage == "composite":
        return run_composite_shard(run_id, tile, index, storage=storage)
    if stage == "export":
        return run_export_merge(run_id, tile, storage=storage)
    msg = f"unknown shard stage {stage!r}; expected one of {shards.STAGES}"
    raise ValueError(msg)
