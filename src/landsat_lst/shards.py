"""The arithmetic and the key grammar a sharded tile is cut along.

A tile is hours of work with no intra-tile checkpoint: a failure at minute 170
costs all 170 (ADR-015). Stage 3 cuts one tile into work that several processes
can hold at once, and this module owns the two things that decision reduces to
once the compute is set aside -- *where* the cuts fall, and *what the pieces are
called*. Both are pure functions here: no storage, no dask, no network, so the
cut geometry can be pinned by a unit test rather than inferred from a run.

**Row bands only, never column bands.** ``odc-stac`` groups scenes by
``solar_day``, and the solar-time shift it applies comes from the geobox
*centroid longitude*. Two column bands of one tile have different centroid
longitudes, so they can group the same items into different time steps -- a
different time axis per shard, and offsets that no longer line up with the
scenes they were estimated for. Cutting along rows leaves the centroid
longitude fixed and the grouping identical in every band. Nothing in this
module offers a column split, and nothing should add one.

Three properties hold the seams together and each is pinned in
``tests/unit/test_shards.py``:

- :func:`block_spans` is the *same* construction ``climatology_by_blocks`` runs,
  lifted rather than copied. A second spelling of the block loop would let a
  shard reduce a different set of pixels than the whole-tile path does.
- :func:`band_edges` cuts on multiples of the COG block size, so a band's rows
  land on a destination block boundary and the merge is a block-aligned copy
  rather than a read-modify-write. The last band absorbs the ragged remainder,
  which is mandatory rather than tidy: 18,000 = 512 x 35 + 80.
- :class:`TilePlan` carries a :attr:`~TilePlan.digest` over every setting that
  changes what a shard computes, so a shard started under a different
  configuration refuses the plan instead of contributing an incompatible piece
  to the merge. The same reasoning as
  :class:`landsat_lst.offsets.OffsetKey`, over a wider set of inputs: a plan
  also fixes the chunking the pieces are cut on.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Any

import structlog

from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: The stages one tile runs through, in the only order they can run in. Named
#: here rather than in the driver so a shard task, a submission, and a key all
#: spell a stage the same way.
STAGES: tuple[str, ...] = ("resolve", "climatology", "offsets", "composite", "export")

#: Prefix for per-shard intermediates. A sibling of the collection directories
#: and deliberately **disjoint** from :data:`landsat_lst.storage.RUN_RECORD_PREFIX`:
#: ``runs.classify`` reads every key under the run prefix as a tile artifact and
#: would file ``plan.json`` as a tile named ``plan``. Shard objects therefore
#: never live under ``_runs/``. See ``tests/unit/test_runs.py``.
SHARD_PREFIX = "_shards"

#: Hex characters kept from a plan digest. 16 is 64 bits, matching
#: :data:`landsat_lst.offsets._DIGEST_CHARS`, and keeps the value readable in a
#: bucket listing.
_DIGEST_CHARS = 16

#: A span of one climatology block: ``(y0, y1, x0, x1)``, half-open.
Span = tuple[int, int, int, int]

#: A half-open row band of the output grid: ``(row_start, row_stop)``.
Band = tuple[int, int]


def shard_root(run_id: str, tile_name: str) -> str:
    """Everything one tile's shards publish in one run.

    Args:
        run_id: Run token the shards belong to.
        tile_name: Tile name (``"N40W075"``).

    Returns:
        ``_shards/{run_id}/{tile}``
    """
    return f"{SHARD_PREFIX}/{run_id}/{tile_name}"


def plan_key(root: str) -> str:
    """The :class:`TilePlan` every shard of this tile reads before it starts."""
    return f"{root}/plan.json"


def items_key(root: str) -> str:
    """The resolved STAC items, serialized once by the planner.

    Written rather than re-queried per shard. Re-querying would hand each shard
    its own answer from a live catalog, and two shards that disagree about the
    scene set produce pieces that cannot be merged -- the same failure the
    offset cache's scene-id digest exists to catch, arriving instead as a silent
    seam in the middle of a tile.
    """
    return f"{root}/items.json"


def ref_block_key(root: str, index: int) -> str:
    """One phase-A climatology block, as raw ``.npy`` bytes.

    Uncompressed for the same reason the accuracy fixture is
    (:mod:`landsat_lst.fixture`): the merge memory-maps these and writes them
    into the assembled ``ref`` slice by slice, which compression would force
    into a materialized array.
    """
    return f"{root}/offsets/ref/b{index:04d}.npy"


def ref_marker_key(root: str, index: int) -> str:
    """Marker standing in for an all-NaN climatology block.

    ``climatology_by_blocks`` fills a block with no land pixel with NaN without
    reading it, so uploading ``12 x block^2 x 4 B`` of NaN would cost more than
    the block cost to produce. A zero-byte marker says the same thing, and a
    coastal tile is mostly markers.
    """
    return f"{ref_block_key(root, index)}.nan"


def scene_partial_key(root: str, start: int, stop: int) -> str:
    """One phase-B partial: the offsets for scenes ``[start, stop)``.

    The range is in the key so a listing shows coverage without opening
    anything, and so a re-run of one range replaces exactly its own partial.
    """
    return f"{root}/offsets/scene/s{start:06d}-{stop:06d}.json"


def band_key(root: str, product: str, index: int) -> str:
    """One row band's slab of one product, as a plain GeoTIFF.

    Never a COG at this level: overviews describe the assembled tile, and a
    pyramid built over a 512-row slab would be resampled across a boundary that
    does not exist in the output. See :func:`landsat_lst.cog.merge_bands`.
    """
    return f"{root}/composite/{product}/band{index:03d}.tif"


def shard_state_key(root: str, stage: str, index: int, attempt: int) -> str:
    """One shard's state object, keyed by stage, index, and attempt.

    Keyed by attempt for the reason ``runs.py`` documents at length: retries
    that share a key erase the evidence of the attempt that got furthest.
    """
    return f"{root}/state/{stage}.{index:04d}.{attempt}.json"


def shard_log_key(root: str, stage: str, index: int, attempt: int) -> str:
    """One shard's stdout and stderr, beside its state object."""
    return f"{root}/state/{stage}.{index:04d}.{attempt}.log"


def shard_profile_key(root: str, stage: str, index: int, attempt: int, label: str) -> str:
    """One shard profile beside its state object and log under ``_shards/``."""
    return f"{root}/state/{stage}.{index:04d}.{attempt}.{label}.profile.json"


def export_claim_key(root: str) -> str:
    """One composite shard's claim on running the export itself.

    The export is a single task at the end of a wide stage, and submitting it
    as its own fleet costs a whole VM boot to do a merge that the last
    composite worker is already booted for. So that worker claims it.

    **This is not a lock, and first-writer-wins is not needed.** The export is
    idempotent at the canonical COG keys: two workers running it concurrently
    produce the same two objects, so a lost race costs duplicated work, never a
    corrupted tile. The claim exists to make that duplication rare, not
    impossible, which is why it is a plain write with no compare-and-set --
    S3 offers none, and building one out of listings would add a failure mode
    to save a few minutes of one VM.
    """
    return f"{root}/state/export.claim.json"


def stage_submission_key(root: str, stage: str, submission_round: int) -> str:
    """One driver's record that it started this stage, and when.

    The only thing that tells a *second* driver that a stage is already in
    flight. Without it, a resume that arrives while the first driver's shards
    are still booting sees zero artifacts, concludes nothing has started, and
    submits a duplicate array -- which Coiled refuses outright when the cluster
    name collides, and which would otherwise pay twice for the same blocks.

    Deliberately not a Coiled API call: the bucket is the only thing both
    drivers and every test can read.
    """
    return f"{root}/state/{stage}.submission.{submission_round:03d}.json"


def stage_submission_prefix(root: str, stage: str) -> str:
    """Every submission record one stage has accumulated, across drivers.

    ``{stage}.submission.`` cannot collide with a shard's own artifacts: those
    carry a four-digit index where this carries the literal word.
    """
    return f"{root}/state/{stage}.submission."


def fleet_root(run_id: str) -> str:
    """Everything a *multi-tile* run publishes that is not about one tile.

    A sibling of every :func:`shard_root`, never a parent of one: the per-tile
    roots stay exactly where they were so nothing about a tile's artifacts
    changes when it is driven as part of a fleet. Still under
    :data:`SHARD_PREFIX`, so ``runs.classify`` never sees it.
    """
    return f"{SHARD_PREFIX}/{run_id}"


def fleet_manifest_key(run_id: str) -> str:
    """The tiles in this run, and each one's job parameters.

    The one piece of a fleet run that cannot be recovered by listing. A
    listing shows the tiles that got far enough to write something, which is
    precisely not the set a resume has to drive -- a tile whose first wave was
    preempted before it booted has published nothing at all and would silently
    drop out of the run. So the roster is written once, before anything is
    submitted, and read back by ``resume_fleet``.

    The VMs read it too: a task's token names its tile, and the window and
    scene cap for that tile come from here rather than from the command line.
    That is what lets one array carry tiles whose windows differ, instead of
    forbidding the case.
    """
    return f"{fleet_root(run_id)}/fleet.json"


def fleet_submission_key(run_id: str, stage: str, wave: int) -> str:
    """One wave's record: which units one array carried, across tiles.

    Distinct from :func:`stage_submission_key`, which stays per tile and keeps
    doing its per-tile jobs (adoption, and a round budget counted across
    drivers). This one answers a question only the fleet asks: how many VMs are
    still in flight, and against which deadline.
    """
    return f"{fleet_root(run_id)}/state/{stage}.wave.{wave:04d}.json"


def fleet_submission_prefix(run_id: str, stage: str) -> str:
    """Every wave one stage has run in this fleet."""
    return f"{fleet_root(run_id)}/state/{stage}.wave."


def unit_timing_prefix(run_id: str) -> str:
    """Where one run's per-unit timings live.

    Under :data:`SHARD_PREFIX` so ``runs.classify`` never reads one as a tile
    attempt, and deliberately **outside** ``fleet_root(run_id)``: the driver
    lists that prefix on every poll, and a per-unit object per unit would put
    tens of thousands of keys into a listing whose only job is to answer a
    question about barriers. Nothing in the barrier path reads these; the cost
    model does, once, afterwards.
    """
    return f"{SHARD_PREFIX}/timings/{run_id}/"


def unit_timing_key(run_id: str, stage: str, tile: str, index: int) -> str:
    """One work unit's start and end, as the VM that ran it saw them.

    Per-wave stamps bound billed VM time from above, and that is all they do: a
    worker between units and a worker running one look identical from the
    bucket. Idle is billed time minus boot minus what the units actually ran
    for, so the durations have to come from the units.
    """
    return f"{unit_timing_prefix(run_id)}{stage}.{tile}.{index:04d}.json"


def unit_trace_prefix(run_id: str, stage: str, tile: str, index: int) -> str:
    """Stem shared by one work unit's execution-trace artifacts."""
    return f"{unit_timing_prefix(run_id)}{stage}.{tile}.{index:04d}"


def fleet_unit_token(tile: str, index: int) -> str:
    """One task's whole input: which tile, and which shard of it.

    ``coiled.batch_run`` maps over a list of strings, and in the single-tile
    path that string is the shard index alone because the tile is baked into
    the command. A consolidated array carries units from many tiles through one
    command, so the tile has to travel in the value.

    A colon separates them because a tile name never contains one
    (:func:`landsat_lst.tiling.parse_tile_name`) and neither does a decimal
    index, which makes the split unambiguous rather than merely conventional.
    """
    return f"{tile}:{index}"


def parse_fleet_unit(token: str) -> tuple[str, int]:
    """``"N40W075:3"`` back into ``("N40W075", 3)``.

    Raises:
        ValueError: If the token is not exactly one tile and one integer. A
            token that parsed loosely would let a task compute the wrong slab
            of the wrong tile and publish it under a key that looks correct.
    """
    tile, _, raw = token.strip().partition(":")
    if not tile or not raw.isdigit():
        msg = f"malformed fleet unit token {token!r}; expected '<tile>:<index>'"
        raise ValueError(msg)
    return tile, int(raw)


def shard_attempt_prefix(root: str, stage: str, index: int) -> str:
    """Everything one shard of one stage has published across its attempts.

    The trailing dot is load-bearing for the same reason it is in
    ``runs.tile_artifact_prefix``: without it, shard 1 would collect shard 10's
    artifacts.
    """
    return f"{root}/state/{stage}.{index:04d}."


def resolve_shard_attempt(storage: StorageBackend, root: str, stage: str, index: int) -> int:
    """The number this shard should key its artifacts under.

    One more than the highest attempt that already left one, counting state
    objects *and* logs, because a VM preempted before it published state still
    uploaded a log on the way out. The same reasoning as
    ``runs.resolve_attempt``, over this module's key grammar rather than that
    one's; the two grammars are deliberately disjoint (see :data:`SHARD_PREFIX`)
    so neither module's classifier can be handed the other's keys.

    Resolve it **once per process**. Asking twice would number the log higher
    than the state object, since the log uploads last.

    A listing that fails returns 1: instrumentation never fails a shard.
    """
    prefix = shard_attempt_prefix(root, stage, index)
    try:
        listed = storage.list_prefix(prefix)
    except Exception as e:
        log.warning(
            "shard_attempt_listing_failed", root=root, stage=stage, index=index, error=str(e)
        )
        return 1

    highest = 0
    for key in listed:
        token = key[len(prefix) :].split(".")[0]
        if token.isdigit():
            highest = max(highest, int(token))
    return highest + 1


def stage_shard_counts(
    *, blocks: int, scene_batches: int, block_rows: int, units: int | None = None
) -> tuple[int, int, int]:
    """How wide each stage's fleet is, given what there is to divide.

    Configuration wins where it is set. Where it is 0 the width comes from
    :func:`landsat_lst.projection.tile_projection`, which turns the probe's
    measured per-VM rates into the VM count that fits a phase inside its share
    of the sixty-minute tile. That is a projection, not a measurement, and it is
    the same projection ``landsat-lst plan`` prints -- the point is that the
    fleet is priced before it is started, never discovered from a run.

    Every width is clamped to the work available. A shard with no block is a VM
    that boots, reads a plan, finds nothing to do, and bills a minute.

    Args:
        blocks: Phase-A climatology blocks in the tile.
        scene_batches: Phase-B scene ranges in the window.
        block_rows: Whole COG block rows in the output grid, which bounds the
            number of row bands :func:`band_edges` can cut.
        units: The fused offsets fleet's width, which the driver fixed *before*
            this plan existed -- it had to, since it starts that fleet and only
            then does shard 0 write the plan. Both offsets-side counts follow
            it (still clamped to the work available), so shard *i* of the fleet
            owns group *i* of each phase. A shard whose index falls past a
            clamped count simply has nothing to do in that phase. Ignoring this
            and re-deriving the width here would let the two processes disagree
            about how many partials the tile expects, and the stage would wait
            forever for one nobody was asked to write.

    Returns:
        ``(ref_shards, scene_shards, band_shards)``, each at least 1.
    """
    from landsat_lst.projection import tile_projection  # noqa: PLC0415

    projected = tile_projection()
    auto_offsets = max(1, round(projected.n_vms_offsets))
    auto_composite = max(1, round(projected.n_vms_composite))

    def _pick(configured: int, auto: int, available: int) -> int:
        return max(1, min(configured or auto, available))

    return (
        _pick(units or settings.shard_climatology_vms, auto_offsets, blocks),
        _pick(units or settings.shard_offset_vms, auto_offsets, scene_batches),
        _pick(settings.shard_composite_vms, auto_composite, block_rows),
    )


def offsets_fleet_units() -> int:
    """How wide the fused offsets fleet is, decided without a plan.

    The driver starts that fleet before any plan exists, so this cannot depend
    on the tile's geometry. It is passed to the planner (``--units``) rather
    than recomputed there: two processes deriving one number from settings that
    drifted apart would each be internally consistent and jointly wrong, and
    the symptom would be a stage waiting forever for a partial nobody owns.
    """
    from landsat_lst.projection import tile_projection  # noqa: PLC0415

    if settings.shard_offset_vms:
        return settings.shard_offset_vms
    return max(1, round(tile_projection().n_vms_offsets))


def block_spans(shape: tuple[int, int], block: int) -> list[Span]:
    """Y-major ``(y0, y1, x0, x1)`` spans tiling ``shape`` in ``block`` squares.

    The single definition of the phase-A block loop:
    :func:`landsat_lst.normalization.climatology_by_blocks` calls this rather
    than inlining it, so a shard asked to compute "blocks 12 through 23" indexes
    into the same list the whole-tile path walks. Two spellings of this
    construction would be two different sets of pixels wearing one set of
    indices.

    Edge blocks are ragged, never padded: the eastern and southern spans are
    clipped to ``shape``.

    Args:
        shape: ``(height, width)`` of the grid to tile.
        block: Block edge in pixels.

    Returns:
        Every span, y-major (all of row-band 0's blocks, then row-band 1's).

    Raises:
        ValueError: If ``block`` is not positive.
    """
    if block <= 0:
        msg = f"block edge must be positive, got {block}"
        raise ValueError(msg)

    height, width = shape
    return [
        (y0, min(y0 + block, height), x0, min(x0 + block, width))
        for y0 in range(0, height, block)
        for x0 in range(0, width, block)
    ]


def band_edges(height: int, n_bands: int, chunk: int) -> list[Band]:
    """Cut ``height`` rows into ``n_bands`` half-open bands on ``chunk`` edges.

    Every boundary is a multiple of ``chunk`` so that a band written on its own
    starts on a destination block row, which is what makes the merge a windowed
    copy instead of a read-modify-write of a straddling block.

    The last band absorbs the ragged remainder because the production grid
    leaves one: 18,000 rows is 512 x 35 + 80, so the final band is 80 rows
    shorter than a whole number of blocks. A cut rule that assumed an exact
    division would be wrong on every tile this project writes.

    Args:
        height: Rows in the output grid.
        n_bands: How many bands to cut. Must not exceed the number of
            ``chunk``-sized rows available, or a band would be empty.
        chunk: Block height every boundary must be a multiple of.

    Returns:
        ``[(start, stop), ...]`` covering ``[0, height)`` exactly, in order,
        with no overlap and no gap.

    Raises:
        ValueError: If any argument is non-positive, or if ``n_bands`` exceeds
            the number of chunk rows in ``height``.
    """
    if height <= 0 or chunk <= 0 or n_bands <= 0:
        msg = f"height, n_bands, and chunk must all be positive: {height}, {n_bands}, {chunk}"
        raise ValueError(msg)

    n_chunks = -(-height // chunk)  # ceil
    if n_bands > n_chunks:
        msg = (
            f"cannot cut {height} rows into {n_bands} bands on {chunk}-row "
            f"boundaries: only {n_chunks} chunk rows exist, so at least one "
            f"band would be empty"
        )
        raise ValueError(msg)

    base, extra = divmod(n_chunks, n_bands)
    out: list[Band] = []
    start = 0
    for i in range(n_bands):
        # The remainder goes to the *first* bands, so the last band is the one
        # carrying the short final chunk rather than a short chunk plus a whole
        # extra one.
        stop = min(start + (base + (1 if i < extra else 0)) * chunk, height)
        out.append((start, stop))
        start = stop
    return out


def partition(seq: Sequence[Any], n: int) -> list[list[Any]]:
    """Split ``seq`` into ``n`` contiguous, near-equal runs.

    Contiguous rather than round-robin: a shard's work should be an index range
    it can name in a key, and neighbouring blocks share source chunks, so
    handing one shard a contiguous run costs fewer reads than scattering it.

    The remainder goes to the earliest groups, so the split is a pure function
    of ``(len(seq), n)`` and two processes computing it independently agree.

    Args:
        seq: What to split.
        n: How many groups. Must not exceed ``len(seq)``.

    Returns:
        ``n`` non-empty lists whose concatenation is ``seq``.

    Raises:
        ValueError: If ``n`` is non-positive or larger than ``len(seq)``.
    """
    if n <= 0 or n > len(seq):
        msg = f"cannot split {len(seq)} items into {n} groups"
        raise ValueError(msg)

    base, extra = divmod(len(seq), n)
    out: list[list[Any]] = []
    start = 0
    for i in range(n):
        stop = start + base + (1 if i < extra else 0)
        out.append(list(seq[start:stop]))
        start = stop
    return out


def balance_by_weight(spans: Sequence[Span], weights: Sequence[float], n: int) -> list[list[Span]]:
    """Split ``spans`` into ``n`` contiguous groups holding similar total weight.

    An equal split of the *blocks* is not an equal split of the *work*. What a
    phase-A shard reads for a block is the scenes whose footprints cross it,
    and on S30W065 that count ran from 198 to 821 per block: an equal-count
    split handed one shard 4,191 footprint intersections and another 1,798,
    and every shard then waited for the heaviest at the in-process barrier
    (#133). Splitting on cumulative weight makes the groups alike in what they
    read while staying contiguous.

    Deterministic minimax partitioning: among every possible contiguous split,
    choose one whose heaviest group has the least total weight. Every group is
    left at least one span. With no weight anywhere this degenerates to
    :func:`partition`'s shape.

    Args:
        spans: Blocks in the order :func:`block_spans` produced them.
        weights: Non-negative cost of each span. A zero is a span that costs
            nothing to own, such as a block with no land pixel.
        n: How many groups.

    Returns:
        ``n`` non-empty contiguous groups whose concatenation is ``spans``.

    Raises:
        ValueError: If the inputs disagree in length, a weight is negative or
            non-finite, or ``n`` is out of range.
    """
    if len(spans) != len(weights):
        msg = f"spans and weights disagree: {len(spans)} vs {len(weights)}"
        raise ValueError(msg)
    if n <= 0 or n > len(spans):
        msg = f"cannot split {len(spans)} spans into {n} groups"
        raise ValueError(msg)
    numeric = [float(w) for w in weights]
    if any(not isfinite(w) or w < 0 for w in numeric):
        msg = "block weights must be finite and non-negative"
        raise ValueError(msg)

    total = sum(numeric)
    if total == 0:
        return partition(spans, n)

    # ``best[k][end]`` is the lightest possible maximum group weight when the
    # first ``end`` spans are split into ``k`` non-empty contiguous groups.
    # The production plan has tens of blocks, so the O(n * blocks**2) dynamic
    # program is tiny beside resolving thousands of scene footprints.
    prefix = [0.0]
    for weight in numeric:
        prefix.append(prefix[-1] + weight)

    count = len(spans)
    best = [[float("inf")] * (count + 1) for _ in range(n + 1)]
    cuts = [[-1] * (count + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for groups in range(1, n + 1):
        for end in range(groups, count + 1):
            for start in range(groups - 1, end):
                cost = max(best[groups - 1][start], prefix[end] - prefix[start])
                if cost < best[groups][end]:
                    best[groups][end] = cost
                    cuts[groups][end] = start

    out: list[list[Span]] = []
    end = count
    for groups in range(n, 0, -1):
        start = cuts[groups][end]
        out.append(list(spans[start:end]))
        end = start
    out.reverse()
    return out


def balance_by_land(spans: Sequence[Span], has_land: Sequence[bool], n: int) -> list[list[Span]]:
    """Split ``spans`` into ``n`` contiguous groups holding similar land counts.

    This is the legacy split a plan without block weights gets. A block with
    no land pixel is filled with NaN and never read
    (:func:`landsat_lst.normalization.climatology_by_blocks`), so on a coastal
    tile an equal-count split can hand one shard every scene it must read and
    another almost nothing.

    Args:
        spans: Blocks in the order :func:`block_spans` produced them.
        has_land: Whether each span holds at least one land pixel.
        n: How many groups.

    Returns:
        ``n`` non-empty contiguous groups whose concatenation is ``spans``.

    Raises:
        ValueError: If the inputs disagree in length, or ``n`` is out of range.
    """
    if len(spans) != len(has_land):
        msg = f"spans and has_land disagree: {len(spans)} vs {len(has_land)}"
        raise ValueError(msg)
    if n <= 0 or n > len(spans):
        msg = f"cannot split {len(spans)} spans into {n} groups"
        raise ValueError(msg)

    land = [1 if flag else 0 for flag in has_land]
    total = sum(land)
    if total == 0:
        return partition(spans, n)

    out: list[list[Span]] = []
    start = 0
    seen = 0
    for k in range(n):
        remaining = n - k - 1
        if remaining == 0:
            out.append(list(spans[start:]))
            break
        target = total * (k + 1) / n
        ceiling = len(spans) - remaining
        end = start + 1
        acc = seen + land[start]
        while end < ceiling and acc < target:
            acc += land[end]
            end += 1
        out.append(list(spans[start:end]))
        seen = acc
        start = end
    return out


def block_scene_weights(
    blocks: Sequence[Span],
    has_land: Sequence[bool],
    footprints: Sequence[Any],
    affine: Any,
) -> list[int]:
    """How many scene footprints cross each block, with land-free blocks at zero.

    The weight :func:`balance_by_weight` splits phase A on. Computed once by
    the planner and stored in the plan, never on a VM: a shard and the driver
    re-deriving it from a catalog that is not frozen could disagree, and a
    disagreement about the split is a block nobody owns.

    Args:
        blocks: Spans in the order :func:`block_spans` produced them, as pixel
            indexes into the grid ``affine`` describes.
        has_land: Whether each block holds at least one land pixel. A block
            without one weighs zero whatever crosses it, because the reducer
            never reads it.
        footprints: Scene footprints as shapely geometries in the grid's CRS,
            or ``None`` for a scene without one, which is skipped.
        affine: The grid's pixel-to-CRS transform, ``affine * (col, row)``.

    Returns:
        One non-negative integer per block.

    Raises:
        ValueError: If ``blocks`` and ``has_land`` disagree in length.
    """
    from shapely.geometry import box  # noqa: PLC0415
    from shapely.strtree import STRtree  # noqa: PLC0415

    if len(blocks) != len(has_land):
        msg = f"blocks and has_land disagree: {len(blocks)} vs {len(has_land)}"
        raise ValueError(msg)

    geoms = [g for g in footprints if g is not None and not g.is_empty]
    if not geoms:
        return [0] * len(blocks)
    tree = STRtree(geoms)

    weights: list[int] = []
    for (y0, y1, x0, x1), land in zip(blocks, has_land, strict=True):
        if not land:
            weights.append(0)
            continue
        ax, ay = affine * (x0, y0)
        bx, by = affine * (x1, y1)
        cell = box(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
        weights.append(len(tree.query(cell, predicate="intersects")))
    return weights


def climatology_groups(plan: TilePlan) -> list[list[Span]]:
    """The phase-A split every reader of ``plan`` agrees on.

    The one place that decides which weights the split uses: the plan's
    stored scene weights when it carries them, the land flags when it does
    not (a plan written before weights existed). The planner, each shard, and
    the budget all come through here, so a plan splits the same way in every
    process that reads it.
    """
    if plan.block_weights is not None:
        return balance_by_weight(plan.blocks, plan.block_weights, plan.ref_shards)
    return balance_by_land(plan.blocks, plan.block_has_land, plan.ref_shards)


@dataclass(frozen=True)
class TilePlan:
    """Where one tile is cut, and under which configuration it was cut there.

    Written once by a planner and read by every shard. The shards never
    re-derive any of it: two processes deriving a block list from settings that
    have drifted apart would each be internally consistent and jointly wrong,
    and the merge would not notice.

    Every field is plain JSON so the record survives a round trip through
    storage with no custom codec, and so a human debugging a stalled run can
    read the cut out of a bucket.
    """

    tile: str
    window: str
    #: STAC item ids, in load order. The same list
    #: :class:`landsat_lst.offsets.OffsetKey` hashes.
    scene_ids: list[str]
    #: Loaded time coordinate as ISO strings. Frozen here because phase B's
    #: partials are keyed on it and the merge verifies coverage against it.
    scene_times: list[str]
    offset_factor: int
    #: ``(height, width)`` of the grid offsets are estimated on.
    coarse_shape: tuple[int, int]
    #: ``(height, width)`` of the output grid.
    native_shape: tuple[int, int]
    block_edge: int
    blocks: list[Span]
    block_has_land: list[bool]
    #: Half-open scene ranges, as ``normalization._scene_batches`` cut them.
    scene_batches: list[tuple[int, int]]
    bands: list[Band]
    #: How many processes each stage is split across.
    ref_shards: int = 1
    scene_shards: int = 1
    band_shards: int = 1
    #: Scene footprints crossing each block, from :func:`block_scene_weights`,
    #: which :func:`climatology_groups` splits phase A on. ``None`` on a plan
    #: written before weights existed, which then splits on land flags. Not
    #: in the digest: the weights decide who reduces a block, never what the
    #: block reduces to.
    block_weights: list[int] | None = None
    #: Digest of the configuration the plan was cut under, filled in by
    #: :meth:`to_dict` and checked by :meth:`from_dict`.
    _digest: str | None = field(default=None, repr=False, compare=False)

    @property
    def digest(self) -> str:
        """Hash of every setting that changes what a shard computes.

        Mirrors :meth:`landsat_lst.offsets.OffsetKey.build` -- the offset
        resolution factor and the plausibility clamp, which together decide
        which pixels reach the median, plus
        :data:`landsat_lst.offsets.ALGORITHM_VERSION` for the code changes a
        hash cannot see. It adds the three chunking settings, because a plan
        also fixes the shape of the pieces: a shard that loaded with a
        different ``TIME_CHUNK`` would cut its scene batches on boundaries the
        planner's ranges do not share, and pay an extra read of every straddled
        chunk for pieces that still merge.
        """
        from landsat_lst.offsets import ALGORITHM_VERSION  # noqa: PLC0415
        from landsat_lst.pipeline import TIME_CHUNK  # noqa: PLC0415

        material = "\n".join(
            [
                f"tile={self.tile}",
                f"window={self.window}",
                f"offset_factor={self.offset_factor}",
                f"destripe_offset_resolution_factor={settings.destripe_offset_resolution_factor}",
                f"lst_valid_min={settings.lst_valid_min}",
                f"lst_valid_max={settings.lst_valid_max}",
                f"time_chunk={TIME_CHUNK}",
                f"load_chunk_size={settings.load_chunk_size}",
                f"load_chunk_size_offsets={settings.load_chunk_size_offsets}",
                f"algorithm_version={ALGORITHM_VERSION}",
                *sorted(self.scene_ids),
            ]
        )
        return hashlib.sha256(material.encode()).hexdigest()[:_DIGEST_CHARS]

    def to_dict(self) -> dict[str, Any]:
        """Render as JSON-safe primitives, stamping the current digest in."""
        payload = asdict(self)
        payload.pop("_digest")
        payload["coarse_shape"] = list(self.coarse_shape)
        payload["native_shape"] = list(self.native_shape)
        payload["blocks"] = [list(span) for span in self.blocks]
        payload["scene_batches"] = [list(span) for span in self.scene_batches]
        payload["bands"] = [list(span) for span in self.bands]
        payload["digest"] = self.digest
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TilePlan:
        """Rebuild a plan, refusing one cut under a different configuration.

        The refusal is the point. A shard that silently accepted a plan whose
        digest no longer matches its own settings would compute a piece that
        fits the merge's shape and not its meaning -- offsets estimated at a
        different factor, or a clamp that admitted different pixels -- and
        nothing downstream inspects a band closely enough to catch it.

        Raises:
            ValueError: If the stored digest disagrees with this process's.
        """
        stored = payload.get("digest")
        plan = cls(
            tile=payload["tile"],
            window=payload["window"],
            scene_ids=list(payload["scene_ids"]),
            scene_times=list(payload["scene_times"]),
            offset_factor=int(payload["offset_factor"]),
            coarse_shape=tuple(payload["coarse_shape"]),  # type: ignore[arg-type]
            native_shape=tuple(payload["native_shape"]),  # type: ignore[arg-type]
            block_edge=int(payload["block_edge"]),
            blocks=[tuple(span) for span in payload["blocks"]],  # type: ignore[misc]
            block_has_land=[bool(flag) for flag in payload["block_has_land"]],
            scene_batches=[tuple(span) for span in payload["scene_batches"]],  # type: ignore[misc]
            bands=[tuple(span) for span in payload["bands"]],  # type: ignore[misc]
            ref_shards=int(payload.get("ref_shards", 1)),
            scene_shards=int(payload.get("scene_shards", 1)),
            band_shards=int(payload.get("band_shards", 1)),
            block_weights=(
                None
                if payload.get("block_weights") is None
                else [int(w) for w in payload["block_weights"]]
            ),
            _digest=stored,
        )
        if stored is not None and stored != plan.digest:
            msg = (
                f"plan for {plan.tile} was cut under a different configuration "
                f"(digest {stored}, this process computes {plan.digest}). "
                "Re-plan the tile rather than contributing a shard to it."
            )
            raise ValueError(msg)
        return plan
