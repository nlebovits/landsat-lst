"""Sequencing one tile's shards, from a laptop, against S3 barriers.

``coiled.batch_run`` has no dependency mechanism: it starts an array of tasks
and tells you nothing about when they finish that is worth acting on. The exit
code it records is the tee wrapper's, task stdout never reaches ``coiled logs``,
and a task can exit non-zero after its artifact landed. So the ordering between
stages -- resolve, then the climatology, then the per-scene offsets, then the
composite bands, then the export -- is a poll loop here, and the only question
it ever asks is whether a key exists.

**Completion is bytes in the bucket.** A shard is done when its artifact is
listed, never when a task exits, and the artifact key is a pure function of the
shard index (:mod:`landsat_lst.shards`). That one rule is what makes the whole
thing restartable: the driver holds no state a crash could lose, and
:func:`resume_tile` reconstructs the position from a single listing of the
tile's shard prefix.

The merge between the offset stages runs **in this process**. Its input is a few
hundred kilobytes of JSON and its output is ~600 floats; a VM would spend longer
booting than working, and the record it writes is the ordinary ADR-012 offset
cache, which is the seam the composite shards read back through.

Failure is bounded rather than retried. When a stage's barrier expires the
driver resubmits *only the indexes that are still missing*, as a fresh small
array, at most ``settings.shard_barrier_rounds`` submissions in total; after
that the tile fails naming the keys that never appeared. Shards are idempotent,
so a resubmitted index that was merely slow finds its own artifact and exits.
There is no on-Coiled driver here: that needs a token that outlives the run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from landsat_lst import shard_tasks, shards
from landsat_lst.batch import submit_shard_stage
from landsat_lst.config import settings
from landsat_lst.storage import PRODUCTS, get_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from landsat_lst.models import ProcessingJob
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: Signature of the stage submitter, so a test can pass one that writes the
#: artifacts synchronously instead of starting a cluster.
Submitter = Callable[..., object]


class ShardStageFailed(RuntimeError):
    """A stage still had missing shards after its last permitted submission."""

    def __init__(self, stage: str, missing: Sequence[str]) -> None:
        self.stage = stage
        self.missing = list(missing)
        listed = ", ".join(self.missing[:5])
        more = "" if len(self.missing) <= 5 else f" (+{len(self.missing) - 5} more)"
        super().__init__(
            f"stage {stage!r} left {len(self.missing)} shard artifacts unwritten "
            f"after {settings.shard_barrier_rounds} submissions: {listed}{more}"
        )


@dataclass
class StageOutcome:
    """What one stage cost and how many times it had to be started."""

    stage: str
    shards: int
    #: Shards whose artifacts were already present when the stage was reached.
    already_done: int
    submissions: int
    wall_s: float
    cluster_ids: list[int | None] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "shards": self.shards,
            "already_done": self.already_done,
            "submissions": self.submissions,
            "wall_s": round(self.wall_s, 1),
            "cluster_ids": self.cluster_ids,
        }


@dataclass
class TileRunSummary:
    """One tile's whole sharded run, stage by stage."""

    run_id: str
    tile: str
    window: str
    stages: list[StageOutcome] = field(default_factory=list)
    completed: bool = False

    @property
    def wall_s(self) -> float:
        return sum(stage.wall_s for stage in self.stages)

    @property
    def resubmissions(self) -> int:
        """Extra submissions beyond the first for each stage that ran."""
        return sum(max(0, stage.submissions - 1) for stage in self.stages)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "tile": self.tile,
            "window": self.window,
            "completed": self.completed,
            "wall_s": round(self.wall_s, 1),
            "resubmissions": self.resubmissions,
            "stages": [stage.as_dict() for stage in self.stages],
        }


def shard_run_id(job: ProcessingJob) -> str:
    """A run token for one sharded tile.

    Carries the tile as well as the window, unlike a fleet run's: the shard
    prefix is per tile and a token shared by two tiles would put two plans in
    one place.
    """
    return f"shard-{job.tile.name}-{job.window_label}-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"


def _expected_keys(plan: shards.TilePlan, stage: str, root: str) -> dict[int, list[str]]:
    """The artifacts each shard of ``stage`` must leave, by index.

    The single definition of "done" for a stage. Both the barrier and the resume
    read it, so neither can develop its own opinion about what a finished shard
    looks like.
    """
    if stage == "climatology":
        out: dict[int, list[str]] = {}
        for index in range(plan.ref_shards):
            start, group = shard_tasks.climatology_group(plan, index)
            out[index] = [
                shards.ref_block_key(root, start + offset)
                if plan.block_has_land[start + offset]
                else shards.ref_marker_key(root, start + offset)
                for offset in range(len(group))
            ]
        return out
    if stage == "offsets":
        out = {}
        for index in range(plan.scene_shards):
            group = shard_tasks.offsets_group(plan, index)
            out[index] = [shards.scene_partial_key(root, group[0][0], group[-1][1])]
        return out
    if stage == "composite":
        return {
            index: [shards.band_key(root, product, index) for product in PRODUCTS]
            for index in range(len(plan.bands))
        }
    msg = f"stage {stage!r} has no per-shard artifacts"
    raise ValueError(msg)


def _missing(storage: StorageBackend, prefix: str, expected: dict[int, list[str]]) -> list[int]:
    """Which shard indexes have not published all of their artifacts.

    One listing per call, not one head request per key: a composite stage on a
    wide tile is dozens of keys and this runs every poll.
    """
    present = set(storage.list_prefix(prefix))
    return sorted(i for i, keys in expected.items() if not all(k in present for k in keys))


def _await_stage(
    *,
    stage: str,
    run_id: str,
    tile: str,
    storage: StorageBackend,
    prefix: str,
    expected: dict[int, list[str]],
    submit: Submitter,
    job: ProcessingJob | None = None,
) -> StageOutcome:
    """Submit whatever is missing, poll until it lands, resubmit once, then fail.

    Checks before it submits and before it sleeps, so a stage whose artifacts
    are all present costs one listing and starts no cluster -- which is what
    makes a resumed tile skip the stages it already paid for.
    """
    started = time.monotonic()
    outcome = StageOutcome(
        stage=stage,
        shards=len(expected),
        already_done=len(expected) - len(_missing(storage, prefix, expected)),
        submissions=0,
        wall_s=0.0,
    )

    for _round in range(settings.shard_barrier_rounds):
        missing = _missing(storage, prefix, expected)
        if not missing:
            break

        submission = submit(stage=stage, run_id=run_id, tile=tile, indexes=missing, job=job)
        outcome.submissions += 1
        outcome.cluster_ids.append(getattr(submission, "cluster_id", None))
        log.info(
            "shard_stage_open",
            run_id=run_id,
            tile=tile,
            stage=stage,
            submitted=len(missing),
            submission=outcome.submissions,
        )

        deadline = time.monotonic() + settings.shard_barrier_timeout_s
        while True:
            missing = _missing(storage, prefix, expected)
            if not missing:
                break
            if time.monotonic() >= deadline:
                log.warning(
                    "shard_stage_barrier_expired",
                    run_id=run_id,
                    tile=tile,
                    stage=stage,
                    missing=len(missing),
                )
                break
            time.sleep(settings.shard_driver_poll_s)
        if not missing:
            break

    missing = _missing(storage, prefix, expected)
    outcome.wall_s = time.monotonic() - started
    if missing:
        keys = [key for i in missing for key in expected[i]]
        raise ShardStageFailed(stage, keys)

    log.info(
        "shard_stage_done",
        run_id=run_id,
        tile=tile,
        stage=stage,
        shards=outcome.shards,
        submissions=outcome.submissions,
        wall_s=round(outcome.wall_s, 1),
    )
    return outcome


def _await_single(
    *,
    stage: str,
    run_id: str,
    tile: str,
    storage: StorageBackend,
    prefix: str,
    keys: Sequence[str],
    submit: Submitter,
    job: ProcessingJob | None = None,
) -> StageOutcome:
    """The one-task stages (``resolve``, ``export``), through the same barrier."""
    return _await_stage(
        stage=stage,
        run_id=run_id,
        tile=tile,
        storage=storage,
        prefix=prefix,
        expected={0: list(keys)},
        submit=submit,
        job=job,
    )


def drive_tile(
    job: ProcessingJob,
    *,
    run_id: str | None = None,
    storage: StorageBackend | None = None,
    submit: Submitter | None = None,
) -> TileRunSummary:
    """Run one tile across a fleet of shards, and return when the COGs exist.

    Stages run in the only order they can: the plan freezes the scene set every
    later stage reads; the climatology is the reference the per-scene offsets
    are measured against; the offsets are the scalars every band applies; the
    bands are what the export stitches. Each is a barrier, and the barrier is a
    listing rather than a task status.

    The driver holds nothing a crash could lose. Killing it and running
    :func:`resume_tile` with the same run id picks up at whichever stage the
    bucket says is unfinished.

    Args:
        job: The tile and window to build.
        run_id: Run token; generated from the tile, window, and a UTC timestamp
            when omitted. Print it: it is the only thing ``resume_tile`` needs.
        storage: Backend the shards publish to. Defaults to the configured one.
        submit: Stage submitter, for tests. Defaults to
            :func:`landsat_lst.batch.submit_shard_stage`.

    Returns:
        The :class:`TileRunSummary`.

    Raises:
        ShardStageFailed: If a stage exhausted its submissions with shards
            still missing.
    """
    storage = storage or get_storage()
    submit = submit or submit_shard_stage
    run_id = run_id or shard_run_id(job)
    return _drive(
        run_id=run_id,
        tile=job.tile.name,
        job=job,
        storage=storage,
        submit=submit,
    )


def resume_tile(
    run_id: str,
    tile: str,
    *,
    storage: StorageBackend | None = None,
    submit: Submitter | None = None,
) -> TileRunSummary:
    """Continue a run whose driver was killed, from the bucket alone.

    Takes no job: the plan carries the window and
    ``ProcessingJob.window_label`` is invertible, so the tile a resume rebuilds
    is the tile the run started. A resume before the plan exists is the one
    case that cannot work, and it says so rather than resolving a second scene
    set that might differ from the first.

    Raises:
        FileNotFoundError: If the run published no plan for this tile.
    """
    storage = storage or get_storage()
    submit = submit or submit_shard_stage
    root = shards.shard_root(run_id, tile)

    if storage.read_text(shards.plan_key(root)) is None:
        msg = (
            f"run {run_id!r} published no plan for {tile} at {shards.plan_key(root)}; "
            "there is nothing to resume -- start it with `landsat-lst shard process`"
        )
        raise FileNotFoundError(msg)

    return _drive(run_id=run_id, tile=tile, job=None, storage=storage, submit=submit)


def _drive(
    *,
    run_id: str,
    tile: str,
    job: ProcessingJob | None,
    storage: StorageBackend,
    submit: Submitter,
) -> TileRunSummary:
    """The stage sequence, shared by a fresh run and a resumed one."""
    root = shards.shard_root(run_id, tile)
    summary = TileRunSummary(run_id=run_id, tile=tile, window=job.window_label if job else "")

    summary.stages.append(
        _await_single(
            stage="resolve",
            run_id=run_id,
            tile=tile,
            storage=storage,
            prefix=f"{root}/",
            keys=[shards.plan_key(root), shards.items_key(root)],
            submit=submit,
            job=job,
        )
    )

    ctx = shard_tasks.load_context(run_id, tile, storage=storage)
    plan = ctx.plan
    summary.window = plan.window
    log.info(
        "shard_plan_read",
        run_id=run_id,
        tile=tile,
        scenes=len(plan.scene_ids),
        ref_shards=plan.ref_shards,
        scene_shards=plan.scene_shards,
        band_shards=len(plan.bands),
        digest=plan.digest,
    )

    for stage, prefix in (
        ("climatology", f"{root}/offsets/ref/"),
        ("offsets", f"{root}/offsets/scene/"),
    ):
        summary.stages.append(
            _await_stage(
                stage=stage,
                run_id=run_id,
                tile=tile,
                storage=storage,
                prefix=prefix,
                expected=_expected_keys(plan, stage, root),
                submit=submit,
            )
        )

    # In the driver, not on a VM: a kilobyte of JSON in, 600 floats out.
    merged = time.monotonic()
    key = shard_tasks.merge_offsets(run_id, tile, storage=storage)
    summary.stages.append(
        StageOutcome(
            stage="merge_offsets",
            shards=1,
            already_done=0,
            submissions=0,
            wall_s=time.monotonic() - merged,
        )
    )
    log.info("shard_offsets_ready", run_id=run_id, tile=tile, key=key.storage_key)

    summary.stages.append(
        _await_stage(
            stage="composite",
            run_id=run_id,
            tile=tile,
            storage=storage,
            prefix=f"{root}/composite/",
            expected=_expected_keys(plan, "composite", root),
            submit=submit,
        )
    )

    summary.stages.append(
        _await_single(
            stage="export",
            run_id=run_id,
            tile=tile,
            storage=storage,
            prefix=storage.cog_key(plan.window, tile, PRODUCTS[0]).rsplit("/", 1)[0] + "/",
            keys=[storage.cog_key(plan.window, tile, product) for product in PRODUCTS],
            submit=submit,
        )
    )

    # The tile's completion criterion is unchanged: both assets, at the
    # canonical keys, listed by the same check every other path uses.
    summary.completed = storage.cog_exists(plan.window, tile)
    log.info(
        "shard_tile_done",
        run_id=run_id,
        tile=tile,
        completed=summary.completed,
        wall_s=round(summary.wall_s, 1),
        resubmissions=summary.resubmissions,
    )
    return summary
