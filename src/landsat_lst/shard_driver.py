"""Sequencing one tile's shards, from a laptop, against S3 barriers.

``coiled.batch_run`` has no dependency mechanism: it starts an array of tasks
and tells you nothing about when they finish that is worth acting on. The exit
code it records is the tee wrapper's, task stdout never reaches ``coiled logs``,
and a task can exit non-zero after its artifact landed. So the ordering between
stages is a poll loop here, and the only question it ever asks is whether a key
exists.

**Two fleets, not five.** The offsets side is one fused task type: shard 0
resolves, every shard waits for that plan, reduces its climatology blocks,
waits at an in-process phase-A barrier, and estimates its scenes' offsets. An
offsets-side shard computed for about six minutes while its stage held a fleet
for about thirty, so the boundaries between those phases are now waits inside a
booted process rather than boots of new ones. The composite fleet starts from
inside the offsets barrier, as soon as phase B is demonstrably producing, and
the export is claimed by whichever composite worker writes the last band --
leaving the driver a fallback rather than a submission. See ADR-016.

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

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import ceil
from typing import TYPE_CHECKING

import structlog

from landsat_lst import shard_tasks, shards
from landsat_lst.batch import stage_cluster_name, submit_shard_stage
from landsat_lst.config import settings
from landsat_lst.storage import PRODUCTS, S3Storage, get_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from landsat_lst.models import ProcessingJob
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: Signature of the stage submitter, so a test can pass one that writes the
#: artifacts synchronously instead of starting a cluster.
Submitter = Callable[..., object]


class ShardBackendMismatch(RuntimeError):
    """The driver would poll one storage while its VMs wrote another.

    Not a warning and not an override. Silently switching the backend under a
    caller who asked for local storage would put COGs somewhere they did not
    ask for; polling on regardless is what actually happened, and it is worse.
    """


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
    #: Rounds this driver watched instead of starting, because another driver's
    #: submission record for the stage was still fresh.
    adopted: int = 0
    #: Highest round this stage reached, across every driver and including the
    #: round started outside the barrier (the fused fleet, and the composite
    #: fleet's overlapped start). ``submissions`` counts only what *this*
    #: driver sent, which is why the two differ and why the retry accounting
    #: reads this one.
    rounds: int = 0
    cluster_ids: list[int | None] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "shards": self.shards,
            "already_done": self.already_done,
            "submissions": self.submissions,
            "adopted": self.adopted,
            "rounds": self.rounds,
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
        """Rounds beyond the first, for each stage that ran.

        Counted from ``rounds`` rather than ``submissions``: a stage's first
        round is started outside its barrier now (the fused fleet before the
        plan exists, the composite fleet from inside the offsets barrier), so
        this driver's own submission count is one short of the stage's history.
        """
        return sum(max(0, stage.rounds - 1) for stage in self.stages)

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


def require_shared_storage(storage: StorageBackend, submit: Submitter | None) -> None:
    """Refuse to drive Coiled work against storage the VMs cannot see.

    The barrier's whole premise is that the driver and the shards read and
    write **one** namespace. Run with the default ``storage_backend=local`` and
    they do not: the shards inherit ``LST_STORAGE_BACKEND=s3`` from
    ``_worker_environ`` and publish to the bucket, while the driver lists a
    directory on the laptop that nothing will ever write to. Observed on
    ``S30W065`` -- ``plan.json`` was on S3 within 3.5 minutes and the resolve
    barrier still never closed, because the driver was looking somewhere else
    entirely. A barrier that cannot see its artifacts fails as a hang, which is
    the most expensive shape a failure can take.

    Checked only when the driver would actually start Coiled work. A caller
    that injects its own submitter -- every test in this repo -- is driving
    something local on purpose, and both halves then share ``LocalStorage``.

    Args:
        storage: The backend the driver will poll.
        submit: The submitter it will use. ``None`` means the default, so a
            caller that wants to check *before* it has built anything (the CLI,
            which must not print a resume hint for a run it cannot start) can
            pass nothing.

    Raises:
        ShardBackendMismatch: If Coiled work is about to be submitted against
            a non-S3 backend.
    """
    coiled_bound = submit is None or submit is submit_shard_stage
    if not coiled_bound or isinstance(storage, S3Storage):
        return

    msg = (
        f"the shard driver submits Coiled work, whose VMs always write S3, but "
        f"settings.storage_backend is {settings.storage_backend!r} -- the driver "
        f"would poll {type(storage).__name__} for artifacts that land in the "
        "bucket, and every stage barrier would hang. Re-run with "
        "LST_STORAGE_BACKEND=s3 (and LST_S3_BUCKET set)."
    )
    raise ShardBackendMismatch(msg)


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


def _submission_records(storage: StorageBackend, root: str, stage: str) -> list[dict]:
    """Every driver's record of having started this stage, oldest round first.

    Best-effort, like everything else that only exists to be read: a listing or
    a body that cannot be parsed yields no record, and the caller then behaves
    as it did before this existed -- it submits. Failing a tile because a
    bookkeeping object was unreadable would be worse than the duplicate
    submission the object exists to prevent.
    """
    out: list[dict] = []
    try:
        keys = sorted(storage.list_prefix(shards.stage_submission_prefix(root, stage)))
    except Exception as e:
        log.warning("shard_submission_listing_failed", root=root, stage=stage, error=str(e))
        return out

    for key in keys:
        try:
            raw = storage.read_text(key)
            if raw is not None:
                out.append(json.loads(raw))
        except (ValueError, TypeError) as e:
            log.warning("shard_submission_malformed", key=key, error=str(e))
    return sorted(out, key=lambda record: int(record.get("round", 0)))


def _record_submission(
    storage: StorageBackend,
    root: str,
    stage: str,
    *,
    submission_round: int,
    indexes: Sequence[int],
    run_id: str,
    tile: str,
    cluster_name: str,
    cluster_id: int | None = None,
) -> None:
    """Publish the fact that this stage was started, before it is started.

    Written *before* the submission, not after. A driver that dies between the
    two leaves a record for a cluster that never ran, which costs the next
    driver one barrier timeout of waiting; a driver that died the other way
    round would leave a live cluster no record mentions, and the next driver
    would collide with it -- which is the failure this whole mechanism exists
    to remove. One wasted wait beats one refused submission and a stage's worth
    of duplicated reads.
    """
    payload = {
        "run_id": run_id,
        "tile": tile,
        "stage": stage,
        "round": submission_round,
        "indexes": list(indexes),
        "cluster_name": cluster_name,
        "cluster_id": cluster_id,
        # Wall clock, not monotonic: the reader is a different process, often
        # on a different machine.
        "submitted_at": time.time(),
        "submitted_at_iso": datetime.now(tz=UTC).isoformat(),
    }
    try:
        storage.write_text(
            shards.stage_submission_key(root, stage, submission_round),
            json.dumps(payload, indent=2),
        )
    except Exception as e:  # pragma: no cover - instrumentation never fails a stage
        log.warning("shard_submission_record_failed", stage=stage, error=str(e))


def _watch(
    *,
    storage: StorageBackend,
    prefix: str,
    expected: dict[int, list[str]],
    deadline: float,
    run_id: str,
    tile: str,
    stage: str,
    on_poll: Callable[[], None] | None = None,
) -> list[int]:
    """Poll until every artifact has landed or the wall-clock deadline passes.

    Returns what is still missing, which is empty exactly when the stage is
    done. Checks before it sleeps, so a stage that finished while the caller
    was deciding costs no poll interval.

    ``on_poll`` runs after each check. It is how the *next* stage gets started
    while this one is still going: the offsets barrier calls a hook that starts
    the composite fleet as soon as phase B is demonstrably producing, so those
    VMs boot on time this stage was going to spend anyway. Best-effort -- an
    overlap that fails to start is a slower tile, not a broken one, and it must
    never take down the barrier it is riding on.
    """
    while True:
        missing = _missing(storage, prefix, expected)
        if not missing:
            return missing
        if on_poll is not None:
            try:
                on_poll()
            except Exception as e:
                log.warning("shard_overlap_hook_failed", run_id=run_id, stage=stage, error=str(e))
        if time.time() >= deadline:
            log.warning(
                "shard_stage_barrier_expired",
                run_id=run_id,
                tile=tile,
                stage=stage,
                missing=len(missing),
            )
            return missing
        time.sleep(settings.shard_driver_poll_s)


def _start_round(
    *,
    stage: str,
    run_id: str,
    tile: str,
    root: str,
    storage: StorageBackend,
    indexes: Sequence[int],
    submission_round: int,
    submit: Submitter,
    job: ProcessingJob | None = None,
    units: int | None = None,
) -> int | None:
    """Record the submission, then make it. Returns the cluster id, if any."""
    cluster_name = stage_cluster_name(run_id, tile, stage, submission_round)
    _record_submission(
        storage,
        root,
        stage,
        submission_round=submission_round,
        indexes=indexes,
        run_id=run_id,
        tile=tile,
        cluster_name=cluster_name,
    )
    submission = submit(
        stage=stage,
        run_id=run_id,
        tile=tile,
        indexes=indexes,
        job=job,
        units=units,
        submission_round=submission_round,
    )
    cluster_id = getattr(submission, "cluster_id", None)
    _record_submission(
        storage,
        root,
        stage,
        submission_round=submission_round,
        indexes=indexes,
        run_id=run_id,
        tile=tile,
        cluster_name=getattr(submission, "name", cluster_name),
        cluster_id=cluster_id,
    )
    log.info(
        "shard_stage_open",
        run_id=run_id,
        tile=tile,
        stage=stage,
        submitted=len(indexes),
        round=submission_round,
        cluster_name=cluster_name,
    )
    return cluster_id


def ensure_started(
    *,
    stage: str,
    run_id: str,
    tile: str,
    root: str,
    storage: StorageBackend,
    indexes: Sequence[int],
    submit: Submitter,
    job: ProcessingJob | None = None,
    units: int | None = None,
) -> bool:
    """Start a stage's first round unless somebody already has. No waiting.

    Two callers need to start a fleet without then blocking on it. The fused
    offsets stage is started *before* its expected artifacts are even knowable
    (they come from a plan its own shard 0 writes), and the composite fleet is
    started mid-way through the offsets barrier so its boot overlaps that
    stage's tail. Both then fall through to the ordinary
    :func:`_await_stage`, which sees the fresh submission record and **adopts**
    it rather than submitting again -- the same machinery that keeps two
    drivers from colliding.

    Returns:
        Whether this call started anything.
    """
    records = _submission_records(storage, root, stage)
    if records:
        return False
    _start_round(
        stage=stage,
        run_id=run_id,
        tile=tile,
        root=root,
        storage=storage,
        indexes=indexes,
        submission_round=1,
        submit=submit,
        job=job,
        units=units,
    )
    return True


def _await_stage(
    *,
    stage: str,
    run_id: str,
    tile: str,
    root: str,
    storage: StorageBackend,
    prefix: str,
    expected: dict[int, list[str]],
    submit: Submitter,
    job: ProcessingJob | None = None,
    units: int | None = None,
    on_poll: Callable[[], None] | None = None,
) -> StageOutcome:
    """Get this stage finished: adopt what is running, start what is not.

    Three states, distinguished by one listing and one small record:

    - Every artifact present. Return, having started nothing. This is what
      makes a resumed tile skip the stages it already paid for.
    - Artifacts missing, but a submission record younger than
      ``shard_barrier_timeout_s`` exists. Somebody's cluster is still in
      flight, so **watch it**. A second driver that submitted here instead
      would be refused by Coiled outright (the cluster name is already taken)
      and, if it were not, would pay a second time for the same blocks. Shards
      still booting publish nothing, so the artifacts alone cannot tell this
      state from the next one.
    - Artifacts missing and no fresh record. Start a new round covering only
      the missing indexes, bounded by ``shard_barrier_rounds`` counted across
      *all* drivers rather than per driver -- otherwise each resume would grant
      the stage a fresh budget.
    """
    started = time.monotonic()
    outcome = StageOutcome(
        stage=stage,
        shards=len(expected),
        already_done=len(expected) - len(_missing(storage, prefix, expected)),
        submissions=0,
        wall_s=0.0,
    )
    timeout = settings.shard_barrier_timeout_s

    # Bounded so the loop cannot spin on a clock that moves backwards: every
    # pass either finishes, adopts once, or burns one of the stage's rounds.
    for _pass in range(2 * settings.shard_barrier_rounds + 2):
        missing = _missing(storage, prefix, expected)
        if not missing:
            break

        records = _submission_records(storage, root, stage)
        latest = records[-1] if records else None

        if latest is not None and time.time() < float(latest["submitted_at"]) + timeout:
            outcome.adopted += 1
            log.info(
                "shard_stage_adopted",
                run_id=run_id,
                tile=tile,
                stage=stage,
                round=latest.get("round"),
                cluster_name=latest.get("cluster_name"),
                missing=len(missing),
            )
            missing = _watch(
                storage=storage,
                prefix=prefix,
                expected=expected,
                deadline=float(latest["submitted_at"]) + timeout,
                run_id=run_id,
                tile=tile,
                stage=stage,
                on_poll=on_poll,
            )
            if not missing:
                break
            continue

        next_round = int(latest["round"]) + 1 if latest else 1
        if next_round > settings.shard_barrier_rounds:
            break

        cluster_id = _start_round(
            stage=stage,
            run_id=run_id,
            tile=tile,
            root=root,
            storage=storage,
            indexes=missing,
            submission_round=next_round,
            submit=submit,
            job=job,
            units=units,
        )
        outcome.submissions += 1
        outcome.cluster_ids.append(cluster_id)

        missing = _watch(
            storage=storage,
            prefix=prefix,
            expected=expected,
            deadline=time.time() + timeout,
            run_id=run_id,
            tile=tile,
            stage=stage,
            on_poll=on_poll,
        )
        if not missing:
            break

    records = _submission_records(storage, root, stage)
    outcome.rounds = int(records[-1]["round"]) if records else 0
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
    root: str,
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
        root=root,
        storage=storage,
        prefix=prefix,
        expected={0: list(keys)},
        submit=submit,
        job=job,
    )


def _read_plan(
    run_id: str, tile: str, root: str, storage: StorageBackend
) -> shards.TilePlan | None:
    """The plan if shard 0 has already published it, else ``None``."""
    if storage.read_text(shards.plan_key(root)) is None:
        return None
    return shard_tasks.load_context(run_id, tile, storage=storage).plan


def _wait_for_plan(run_id: str, tile: str, root: str, storage: StorageBackend) -> shards.TilePlan:
    """Wait for shard 0 of the fused fleet to publish the plan.

    The driver cannot know what the offsets stage's artifacts are called until
    this exists -- the scene ranges come from the plan -- so this is the one
    barrier it takes before it has an expected-key map. Bounded by
    ``shard_plan_wait_s`` plus the fleet's own boot allowance, and loud on
    expiry: a plan that never arrives means shard 0 died, and waiting on it
    forever is the hang shape this project keeps paying for.

    Raises:
        ShardStageFailed: If no plan appears in time.
    """
    deadline = time.time() + settings.shard_plan_wait_s + settings.shard_barrier_timeout_s
    while True:
        if storage.read_text(shards.plan_key(root)) is not None:
            return shard_tasks.load_context(run_id, tile, storage=storage).plan
        if time.time() >= deadline:
            raise ShardStageFailed("offsets", [shards.plan_key(root)])
        time.sleep(settings.shard_driver_poll_s)


def _overlap_ready(storage: StorageBackend, root: str, plan: shards.TilePlan) -> bool:
    """Whether phase B has produced enough to justify booting the composite.

    The trigger is *evidence*, not a timer: at least one scene partial means
    the offsets stage is running and producing, which is what separates
    overlapping from gambling a fleet's worth of boot on a stage that may be
    about to fail. ``shard_composite_overlap`` raises the bar to a fraction of
    the partials; 1.0 turns the overlap off entirely.
    """
    present = set(storage.list_prefix(f"{root}/offsets/scene/"))
    done = sum(
        1
        for index in range(plan.scene_shards)
        if all(k in present for k in _expected_keys(plan, "offsets", root)[index])
    )
    if done == 0:
        return False
    needed = max(1, ceil(settings.shard_composite_overlap * plan.scene_shards))
    return done >= needed


def _await_export(
    *,
    run_id: str,
    tile: str,
    root: str,
    storage: StorageBackend,
    plan: shards.TilePlan,
    submit: Submitter,
) -> StageOutcome:
    """Wait for a composite worker to run the export; submit one only if none does.

    The last composite worker to write a band claims the export and runs it
    there, which saves a whole VM boot and a queue wait for a merge whose
    inputs that worker just produced. So the driver's first move is to wait, not
    to submit.

    The fallback is for the claim that is written and never executed -- the
    claiming VM preempted between the two. After
    ``shard_export_claim_fallback_s`` with the bands all present and no COGs,
    the driver submits the export stage exactly as it always did.
    """
    started = time.monotonic()
    cog_keys = [storage.cog_key(plan.window, tile, product) for product in PRODUCTS]
    prefix = cog_keys[0].rsplit("/", 1)[0] + "/"
    expected = {0: cog_keys}

    deadline = time.time() + settings.shard_export_claim_fallback_s
    while _missing(storage, prefix, expected):
        if time.time() >= deadline:
            log.warning(
                "shard_export_claim_unfulfilled",
                run_id=run_id,
                tile=tile,
                waited_s=settings.shard_export_claim_fallback_s,
                note="submitting the export stage",
            )
            outcome = _await_single(
                stage="export",
                run_id=run_id,
                tile=tile,
                root=root,
                storage=storage,
                prefix=prefix,
                keys=cog_keys,
                submit=submit,
            )
            outcome.wall_s += time.monotonic() - started
            return outcome
        time.sleep(settings.shard_driver_poll_s)

    log.info("shard_export_claimed_by_worker", run_id=run_id, tile=tile)
    return StageOutcome(
        stage="export",
        shards=1,
        already_done=1,
        submissions=0,
        wall_s=time.monotonic() - started,
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
        ShardBackendMismatch: If the configured backend is not S3 while Coiled
            work is about to be submitted.
        ShardStageFailed: If a stage exhausted its submissions with shards
            still missing.
    """
    storage = storage or get_storage()
    submit = submit or submit_shard_stage
    require_shared_storage(storage, submit)
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
        ShardBackendMismatch: If the configured backend is not S3 while Coiled
            work is about to be submitted.
        FileNotFoundError: If the run published no plan for this tile.
    """
    storage = storage or get_storage()
    submit = submit or submit_shard_stage
    require_shared_storage(storage, submit)
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
    """The stage sequence, shared by a fresh run and a resumed one.

    Two fleets, not five. The offsets side -- resolve, climatology, phase-A
    barrier, per-scene offsets -- is one fused task type
    (:func:`landsat_lst.shard_tasks.run_offsets_stage`), because an offsets-side
    shard computed for about six minutes while its stage held a fleet for about
    thirty: boots and queueing dominated. The composite fleet starts while that
    one is still finishing, and the export is claimed by whichever composite
    worker writes the last band. See ADR-016.
    """
    root = shards.shard_root(run_id, tile)
    summary = TileRunSummary(run_id=run_id, tile=tile, window=job.window_label if job else "")

    # The fused fleet's width has to be settled before the plan exists, because
    # shard 0 of this very fleet is what writes the plan. It travels with the
    # task command so the planner cuts the plan to the fleet that will run it.
    units = shards.offsets_fleet_units()
    plan = _read_plan(run_id, tile, root, storage)

    # Start the fused fleet only when there is offsets work left. A resumed run
    # whose offsets finished has a plan and every partial, and must start
    # nothing -- which is the whole point of resuming.
    if plan is None or _missing(
        storage, f"{root}/offsets/scene/", _expected_keys(plan, "offsets", root)
    ):
        ensure_started(
            stage="offsets",
            run_id=run_id,
            tile=tile,
            root=root,
            storage=storage,
            indexes=list(range(units)),
            submit=submit,
            job=job,
            units=units,
        )

    if plan is None:
        plan = _wait_for_plan(run_id, tile, root, storage)
    summary.window = plan.window
    log.info(
        "shard_plan_read",
        run_id=run_id,
        tile=tile,
        scenes=len(plan.scene_ids),
        units=units,
        ref_shards=plan.ref_shards,
        scene_shards=plan.scene_shards,
        band_shards=len(plan.bands),
        digest=plan.digest,
    )

    # Started from inside the offsets barrier, the moment phase B is
    # demonstrably producing. Guarded by a flag rather than by the submission
    # record alone so the driver does not list on every poll.
    composite_started = False

    def _overlap() -> None:
        nonlocal composite_started
        if composite_started or not _overlap_ready(storage, root, plan):
            return
        composite_started = ensure_started(
            stage="composite",
            run_id=run_id,
            tile=tile,
            root=root,
            storage=storage,
            indexes=list(range(len(plan.bands))),
            submit=submit,
        )
        if composite_started:
            log.info("shard_composite_overlapped", run_id=run_id, tile=tile, bands=len(plan.bands))

    summary.stages.append(
        _await_stage(
            stage="offsets",
            run_id=run_id,
            tile=tile,
            root=root,
            storage=storage,
            prefix=f"{root}/offsets/scene/",
            expected=_expected_keys(plan, "offsets", root),
            submit=submit,
            job=job,
            units=units,
            on_poll=_overlap,
        )
    )

    # In the driver, not on a VM: a kilobyte of JSON in, 600 floats out. The
    # composite shards are already booting and polling for exactly this record.
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
            root=root,
            storage=storage,
            prefix=f"{root}/composite/",
            expected=_expected_keys(plan, "composite", root),
            submit=submit,
        )
    )

    summary.stages.append(
        _await_export(
            run_id=run_id,
            tile=tile,
            root=root,
            storage=storage,
            plan=plan,
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
