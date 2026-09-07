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

from landsat_lst import budgets, quota, shard_tasks, shards
from landsat_lst.batch import stage_cluster_name, submit_fleet_stage, submit_shard_stage
from landsat_lst.config import settings
from landsat_lst.evidence_contract import ContractError
from landsat_lst.storage import PRODUCTS, S3Storage, get_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from landsat_lst.models import ProcessingJob
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: Signature of the stage submitter, so a test can pass one that writes the
#: artifacts synchronously instead of starting a cluster.
Submitter = Callable[..., object]

#: The submitters that actually start Coiled work. Membership in this tuple is
#: what the backend guard and the two preflight gates key off, so a caller that
#: injects its own submitter is exempt from all three -- it starts no clusters,
#: spends no credits, and writes nowhere but a temporary directory. Both the
#: per-tile array (ADR-016) and the consolidated wave (ADR-018) belong here; a
#: new submitter that is missing from it would slip past the gates silently.
COILED_SUBMITTERS = (submit_shard_stage, submit_fleet_stage)

#: What a cluster probe returns: ``(state, reason)`` for one cluster, or
#: ``None`` when nothing is known about it.
ClusterProbe = Callable[[object], "tuple[str, str] | None"]


class Clock:
    """Wall time and sleeping, in one injectable object.

    Every deadline, every poll, and every backoff in this module goes through
    a clock, so a test can run the whole state machine in milliseconds instead
    of waiting out a barrier. That is not only a speed argument: the bug that
    cost the S30W065 night -- a round-2 deadline computed from the *first*
    round's submission -- is a pure time-arithmetic error, and time arithmetic
    that cannot be tested is time arithmetic nobody checks.

    Epoch seconds rather than a monotonic counter, because submission records
    are read by other processes and other machines.
    """

    def now(self) -> float:
        """Seconds since the epoch."""
        return time.time()

    def sleep(self, seconds: float) -> None:
        """Wait, or return immediately for a non-positive interval."""
        if seconds > 0:
            time.sleep(seconds)


#: Substrings that make a control-plane failure permanent. Matching on the
#: message is crude and deliberate: Coiled surfaces the credit quota as a plain
#: ``ServerError`` whose text is the only thing that distinguishes it from an
#: outage, and on 2026-08-22 that text was the difference between a run that
#: reported "you have reached the workspace quota of 400 Coiled credits" and a
#: run that reported nothing at all.
_TERMINAL_MARKERS = (
    "quota",
    "credit",
    "billing",
    "payment",
    "unauthorized",
    "forbidden",
    "invalid api token",
    "authentication",
    "not entitled",
)


def classify_failure(error: BaseException) -> str:
    """``"terminal"`` if retrying cannot help, else ``"transient"``.

    Terminal means the driver stops now and says why. A quota that is already
    exhausted is not going to clear inside a backoff, and burning the remaining
    rounds against it turns a two-line explanation into a 45-minute silence.

    Everything else is transient, *including an error with no message at all*.
    An empty ``ServerError`` killed the driver outright on 2026-08-22; it
    should have been retried and then, if it persisted, reported. Guessing
    "terminal" for the unknown case would reintroduce that failure for every
    ordinary control-plane blip.
    """
    if isinstance(error, ImportError | ContractError):
        # An evidence contract that no longer binds the launch checkout (a
        # tracked edit mid-run, a revision mismatch) is deterministic: retrying
        # the submission against the same tree cannot change the verdict.
        return "terminal"
    text = f"{type(error).__name__}: {error}".lower()
    if any(marker in text for marker in _TERMINAL_MARKERS):
        return "terminal"
    return "transient"


class ShardSubmissionFailed(RuntimeError):
    """A stage could not be started, and retrying will not change that."""

    def __init__(self, stage: str, reason: str, *, attempts: int) -> None:
        self.stage = stage
        self.reason = reason
        self.attempts = attempts
        super().__init__(f"could not start stage {stage!r} after {attempts} attempt(s): {reason}")


def _coiled_credentials_present() -> bool:
    """Whether a coiled token is configured, without touching the network."""
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    if os.environ.get("COILED_TOKEN") or os.environ.get("DASK_COILED__TOKEN"):
        return True
    import dask.config  # noqa: PLC0415

    if dask.config.get("coiled.token", None):
        return True
    return (Path.home() / ".config" / "dask" / "coiled.yaml").is_file()


def coiled_cluster_probe(cluster_id: object) -> tuple[str, str] | None:
    """One cluster's state and the reason it is in it, from Coiled.

    Best-effort and quiet. A probe that cannot answer must never be the thing
    that fails a tile: the artifacts remain the source of truth and the barrier
    still expires on its own. Injectable so the state machine can be tested
    without a control plane.
    """
    if cluster_id is None:
        return None
    if not _coiled_credentials_present():
        # Without a token the coiled client BLOCKS on its interactive auth
        # flow rather than raising -- the except below never fires. On a
        # token-less CI runner that block ate pytest-timeout's full 300 s and
        # took the xdist worker down with it (2026-08-23). Probing is
        # best-effort; no token means no answer, immediately.
        log.warning("shard_cluster_probe_skipped", reason="no coiled credentials")
        return None
    try:
        import coiled  # noqa: PLC0415

        for record in coiled.list_clusters(just_mine=False):
            if str(record.get("id")) != str(cluster_id):
                continue
            state = str(record.get("current_state", {}).get("state", "") or "")
            reason = str(record.get("current_state", {}).get("reason", "") or "")
            return state, reason
    except Exception as e:  # pragma: no cover - probing never fails a tile
        log.warning("shard_cluster_probe_failed", cluster_id=cluster_id, error=str(e))
    return None


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
            f"after {settings.shard_barrier_rounds} submissions: {listed}{more}. "
            "Rounds are counted across drivers, so a resume of this run reaches "
            "the same limit and submits nothing; raise LST_SHARD_BARRIER_ROUNDS "
            "to let one continue."
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
    #: Whether ``early_settle`` ended this stage rather than its artifacts. The
    #: offsets stage is the one that can: its whole output is a record, and a
    #: record another run wrote settles it with scene partials still unwritten.
    settled_early: bool = False

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
            "settled_early": self.settled_early,
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
    coiled_bound = submit is None or submit in COILED_SUBMITTERS
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


def _preflight(submit: Submitter, balance_source: Callable[[], quota.CreditBalance] | None) -> None:
    """State zero: three gates, before a single cluster is created.

    **Identity, then write access, then credits.** An AWS SSO session expires
    within hours, which is less than a tile takes, and this has bitten three
    times: the driver spends its whole startup -- a STAC query, a plan, a
    fleet's boot -- before discovering that nothing it writes can reach S3. It
    also has to come first, because a session that cannot call STS cannot read
    a Coiled balance either, and "log in again" is a better message than "the
    balance could not be read".

    Then write access, because STS passing says nothing about permission. On
    2026-09-02 the default chain resolved a read-only user and all four
    profiles on the machine cleared the identity gate; two of them could not
    write the publication bucket. That run would have booted a fleet, staged
    nothing, and shown up here as shards that never published.

    Then the quota. That one is knowable before a cluster is created too, and
    on 2026-08-22 it cost a night to learn afterwards -- once as an empty
    ``ServerError`` on a create, once as a healthy fleet killed mid-stage. The
    estimate comes from the same budget model the deadlines do, so it moves
    with the geometry.

    Skipped when the caller injected its own submitter: such a run starts no
    clusters, spends no credits, and writes nowhere but a temporary directory.

    Raises:
        IdentityRefused: If AWS credentials are missing or expired.
        WriteAccessRefused: If an identity the run writes as cannot write, read
            back, and delete under the configured bucket and prefix.
        QuotaRefused: If the workspace cannot afford the run, or if the balance
            could not be read and nobody acknowledged a manual check.
    """
    if submit is not submit_shard_stage:
        return
    quota.preflight_identity()
    quota.preflight_write_access()
    estimate = quota.estimate_run_credits()
    balance = quota.preflight_credits(estimate, balance_source=balance_source)
    log.info(
        "quota_preflight",
        estimate_credits=round(estimate, 1),
        remaining=None if balance.remaining is None else round(balance.remaining, 1),
        source=balance.source,
    )


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
    now: float | None = None,
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
        # on a different machine. Taken from the driver's clock so a test can
        # place a record at any age it likes.
        "submitted_at": time.time() if now is None else now,
        "submitted_at_iso": datetime.now(tz=UTC).isoformat(),
    }
    try:
        storage.write_text(
            shards.stage_submission_key(root, stage, submission_round),
            json.dumps(payload, indent=2),
        )
    except Exception as e:  # pragma: no cover - instrumentation never fails a stage
        log.warning("shard_submission_record_failed", stage=stage, error=str(e))


#: Every state one stage barrier can be in. Written down because the driver is
#: a state machine and had been an implicit one: the bug that cost the
#: S30W065 night lived in the seam between "adopt" and "watch", where a round
#: opened against a deadline belonging to the round before it.
STAGE_STATES = ("check", "submit", "adopt", "watch", "settled", "exhausted")


@dataclass
class StageMachine:
    """One stage's barrier, as an explicit state machine.

    ::

        check --(nothing missing)--> settled
        check --(early_settle)-----> settled_early
        check --(fresh record)-----> adopt ---> watch
        check --(no record, rounds left)--> submit ---> watch
        check --(no rounds left)---> exhausted
        watch --(all artifacts)----> settled
        watch --(early_settle)-----> settled_early
        watch --(deadline)---------> check
        watch --(cluster stopped)--> check
        submit --(terminal error)--> ShardSubmissionFailed

    Four rules the states exist to keep honest:

    - **Every round gets its own deadline, computed when that round opens.**
      A resubmission used to inherit the first round's deadline, so on
      2026-08-22 round 2 adopted at T+46min against a deadline that had expired
      at T+45 and failed instantly, having watched for nothing.
    - **Artifacts decide, not clusters.** The cluster probe can only make a
      barrier end *sooner*; it never declares success, and a cluster reported
      dead is re-checked against the bucket first. A stopped fleet expires the
      round rather than the tile, because ``("stopped", "No pending tasks for
      batch run")`` is what Coiled says about a batch run that has ended for
      any reason at all.
    - **The round budget is counted across drivers**, from the submission
      records, so a resume cannot mint itself a fresh allowance.
    - **Artifacts are not always the only evidence.** ``early_settle`` is
      asked before every round and on every poll, because the offsets stage
      exists to produce one record: a record written by a concurrent run over
      the same scene set, or by a merge that raced this driver's pre-barrier
      check, ends the stage with scene partials still unwritten. Every offsets
      shard already makes that check on every boot, so without this the driver
      waited on partials its own shards had decided not to write.
    """

    stage: str
    run_id: str
    tile: str
    root: str
    storage: StorageBackend
    prefix: str
    expected: dict[int, list[str]]
    submit: Submitter
    deadline_s: float
    clock: Clock
    cluster_probe: ClusterProbe | None = None
    job: ProcessingJob | None = None
    units: int | None = None
    on_poll: Callable[[], None] | None = None
    #: A second, cheaper way for this stage to be over. Asked on every poll and
    #: before every round, because a stage's artifacts are not always the only
    #: evidence that its work is done: the offsets stage exists to produce one
    #: record, and a record written by anybody -- a concurrent run over the
    #: same scene set, a merge that raced this driver's own check -- settles it
    #: however few scene partials this run's shards published. Never raises out
    #: of the machine; a predicate that fails is a predicate that said no.
    early_settle: Callable[[], bool] | None = None
    #: Cluster ids the probe has reported stopped or errored. A record naming
    #: one is dead however young it is, so ``check`` opens the next round
    #: instead of re-adopting a fleet that has already been torn down.
    dead_clusters: set = field(default_factory=set)

    def missing(self) -> list[int]:
        return _missing(self.storage, self.prefix, self.expected)

    def _settled_early(self) -> bool:
        """Whether ``early_settle`` says this stage is over. Never raises.

        The asymmetry this closes: the driver asked whether the offsets record
        existed once, before the barrier, while every offsets shard asks on
        every boot. A record that appears between those two moments makes each
        shard exit without a partial and leaves the driver waiting for work
        nobody will do. One flaky read at the driver's single check does the
        same, because its error path is "run the stage" and the shard's error
        path ends, after a Coiled retry, in "exit early".
        """
        if self.early_settle is None:
            return False
        try:
            return bool(self.early_settle())
        except Exception as e:
            log.warning(
                "shard_early_settle_failed",
                run_id=self.run_id,
                tile=self.tile,
                stage=self.stage,
                error=repr(e)[:200],
            )
            return False

    def run(self) -> StageOutcome:
        """Drive this stage to ``settled``, or raise saying why not."""
        started = self.clock.now()
        outcome = StageOutcome(
            stage=self.stage,
            shards=len(self.expected),
            already_done=len(self.expected) - len(self.missing()),
            submissions=0,
            wall_s=0.0,
        )

        state = "check"
        missing: list[int] = []
        cluster: object = None
        deadline = 0.0
        next_round = 0

        # Bounded so a clock that moves backwards cannot spin the loop: every
        # pass either settles, adopts once, or burns one of the stage's rounds.
        for _step in range(4 * settings.shard_barrier_rounds + 4):
            if state == "check":
                state, missing, next_round = self._check()
                if state in ("settled", "settled_early", "exhausted"):
                    break

            elif state == "adopt":
                cluster, deadline = self._adopt(outcome, missing)
                state = "watch"

            elif state == "submit":
                cluster = self._start_round(next_round, missing)
                outcome.submissions += 1
                outcome.cluster_ids.append(cluster if isinstance(cluster, int) else None)
                # The fresh deadline, taken now rather than inherited. This
                # line is the fix for the round-2 collapse.
                deadline = self.clock.now() + self.deadline_s
                state = "watch"

            elif state == "watch":
                watched = self._watch(deadline=deadline, cluster=cluster)
                if watched is None:
                    outcome.settled_early = True
                    break
                missing = watched
                if not missing:
                    break
                state = "check"

        outcome.settled_early = outcome.settled_early or state == "settled_early"
        return self._finish(outcome, started)

    def _check(self) -> tuple[str, list[int], int]:
        """Decide what this stage does next, from the bucket and the records.

        Returns:
            ``(state, missing, next_round)``. ``"settled"`` means every
            artifact is listed, ``"settled_early"`` that ``early_settle``
            answered instead, and ``"exhausted"`` that the stage has no round
            left. ``next_round`` is meaningful only for ``"submit"``.
        """
        missing = self.missing()
        if not missing:
            return "settled", missing, 0
        if self._settled_early():
            log.info(
                "shard_stage_settled_early",
                run_id=self.run_id,
                tile=self.tile,
                stage=self.stage,
                unwritten=len(missing),
            )
            return "settled_early", missing, 0
        records = _submission_records(self.storage, self.root, self.stage)
        latest = records[-1] if records else None
        if latest is not None and self._is_live(latest):
            return "adopt", missing, 0
        next_round = int(latest["round"]) + 1 if latest else 1
        if next_round > settings.shard_barrier_rounds:
            return "exhausted", missing, next_round
        return "submit", missing, next_round

    def _adopt(self, outcome: StageOutcome, missing: Sequence[int]) -> tuple[object, float]:
        """Watch somebody else's live round rather than starting a duplicate."""
        latest = _submission_records(self.storage, self.root, self.stage)[-1]
        outcome.adopted += 1
        cluster = latest.get("cluster_id")
        deadline = float(latest["submitted_at"]) + self.deadline_s
        log.info(
            "shard_stage_adopted",
            run_id=self.run_id,
            tile=self.tile,
            stage=self.stage,
            round=latest.get("round"),
            cluster_name=latest.get("cluster_name"),
            missing=len(missing),
            remaining_s=round(deadline - self.clock.now(), 1),
        )
        return cluster, deadline

    def _finish(self, outcome: StageOutcome, started: float) -> StageOutcome:
        """Close the accounting, and raise if the stage owes artifacts.

        ``settled_early`` is the one way out with keys still unwritten: the
        offsets stage's whole output is a record, and a record another run
        wrote settles it however few scene partials this run published.
        """
        records = _submission_records(self.storage, self.root, self.stage)
        outcome.rounds = int(records[-1]["round"]) if records else 0
        outcome.wall_s = self.clock.now() - started
        missing = self.missing()
        if missing and not outcome.settled_early:
            keys = [key for i in missing for key in self.expected[i]]
            raise ShardStageFailed(self.stage, keys)

        log.info(
            "shard_stage_done",
            run_id=self.run_id,
            tile=self.tile,
            stage=self.stage,
            shards=outcome.shards,
            submissions=outcome.submissions,
            adopted=outcome.adopted,
            settled_early=outcome.settled_early,
            wall_s=round(outcome.wall_s, 1),
        )
        return outcome

    def _is_live(self, record: dict) -> bool:
        """Whether a submission record is young enough to still be running.

        Evidence beats the clock. A cluster the probe has reported stopped is
        not going to publish anything, so its record is dead the moment the
        report arrives -- without this, a stopped round inside its own deadline
        is adopted, watched, found stopped, and adopted again.
        """
        if record.get("cluster_id") in self.dead_clusters:
            return False
        return self.clock.now() < float(record["submitted_at"]) + self.deadline_s

    def _start_round(self, submission_round: int, indexes: Sequence[int]) -> object:
        """Record the submission, then make it, retrying a transient failure.

        The record goes first and the reasoning has not changed: a driver that
        dies between the two leaves a record for a cluster that never ran,
        which costs the next driver one barrier; the other order leaves a live
        cluster nothing mentions, which is a collision.
        """
        cluster_name = stage_cluster_name(self.run_id, self.tile, self.stage, submission_round)
        self._record(submission_round, indexes, cluster_name=cluster_name)

        last = ""
        for attempt in range(1, settings.shard_submit_retries + 1):
            try:
                submission = self.submit(
                    stage=self.stage,
                    run_id=self.run_id,
                    tile=self.tile,
                    indexes=indexes,
                    job=self.job,
                    units=self.units,
                    submission_round=submission_round,
                )
            except Exception as e:
                last = f"{type(e).__name__}: {e}".strip().rstrip(":")
                kind = classify_failure(e)
                log.warning(
                    "shard_submit_failed",
                    run_id=self.run_id,
                    tile=self.tile,
                    stage=self.stage,
                    round=submission_round,
                    attempt=attempt,
                    kind=kind,
                    error=last,
                )
                if kind == "terminal":
                    raise ShardSubmissionFailed(self.stage, last, attempts=attempt) from e
                if attempt >= settings.shard_submit_retries:
                    raise ShardSubmissionFailed(self.stage, last, attempts=attempt) from e
                self.clock.sleep(settings.shard_submit_backoff_s * 2 ** (attempt - 1))
                continue

            cluster_id = getattr(submission, "cluster_id", None)
            self._record(
                submission_round,
                indexes,
                cluster_name=getattr(submission, "name", cluster_name),
                cluster_id=cluster_id,
            )
            log.info(
                "shard_stage_open",
                run_id=self.run_id,
                tile=self.tile,
                stage=self.stage,
                submitted=len(indexes),
                round=submission_round,
                cluster_name=cluster_name,
                deadline_min=round(self.deadline_s / 60, 1),
            )
            return cluster_id

        raise ShardSubmissionFailed(  # pragma: no cover - loop always returns or raises
            self.stage, last, attempts=settings.shard_submit_retries
        )

    def _record(
        self,
        submission_round: int,
        indexes: Sequence[int],
        *,
        cluster_name: str,
        cluster_id: object = None,
    ) -> None:
        _record_submission(
            self.storage,
            self.root,
            self.stage,
            submission_round=submission_round,
            indexes=indexes,
            run_id=self.run_id,
            tile=self.tile,
            cluster_name=cluster_name,
            cluster_id=cluster_id if isinstance(cluster_id, int) else None,
            now=self.clock.now(),
        )

    def _watch(self, *, deadline: float, cluster: object) -> list[int] | None:
        """Poll until the artifacts land, the fleet dies, or the deadline passes.

        ``on_poll`` runs after each check. It is how the *next* stage gets
        started while this one is still going: the offsets barrier starts the
        composite fleet as soon as phase B is demonstrably producing, so those
        VMs boot on time this stage was going to spend anyway. Best-effort -- an
        overlap that fails to start is a slower tile, not a broken one, and it
        must never take down the barrier it is riding on.

        A cluster reported dead ends the wait immediately, returning what is
        still missing. That is the same answer a barrier expiry gives, and it
        reaches the same place: ``run`` opens the next round over those indexes
        alone. Waiting out a deadline against a torn-down fleet buys nothing.

        Returns:
            The indexes still missing, or ``None`` when ``early_settle`` ended
            the stage some other way. The two are not the same answer and the
            caller has to tell them apart: an empty list means every artifact
            landed, while ``None`` means the stage's work is done and some of
            its artifacts will never be written.
        """
        while True:
            missing = self.missing()
            if not missing:
                return missing
            if self._settled_early():
                return None
            if self.on_poll is not None:
                try:
                    self.on_poll()
                except Exception as e:
                    log.warning(
                        "shard_overlap_hook_failed",
                        run_id=self.run_id,
                        stage=self.stage,
                        error=str(e),
                    )
            if self._fleet_stopped(cluster, missing):
                return self.missing()
            if self.clock.now() >= deadline:
                log.warning(
                    "shard_stage_barrier_expired",
                    run_id=self.run_id,
                    tile=self.tile,
                    stage=self.stage,
                    missing=len(missing),
                    deadline_min=round(self.deadline_s / 60, 1),
                )
                return missing
            self.clock.sleep(settings.shard_driver_poll_s)

    def _fleet_stopped(self, cluster: object, missing: Sequence[int]) -> str | None:
        """Whether this round's cluster is gone, and why.

        Only ever ends a barrier *early*; it can never declare success. And a
        cluster reported dead is checked against the bucket once more first,
        because a fleet whose last task uploaded its artifact and then stopped
        is a finished stage, not a killed one.

        The answer is a reason, not an exception. ``("stopped", "No pending
        tasks for batch run")`` is what Coiled says about every batch run that
        reaches its end, healthy or not, so reading it as a kill aborted the
        driver on 2026-09-04 with 24 of 35 composite bands already in the
        bucket and eleven shards left to resubmit. What the caller does with a
        stopped fleet is open the next round over the missing indexes.
        """
        if self.cluster_probe is None or cluster is None:
            return None
        probed = self.cluster_probe(cluster)
        if probed is None:
            return None
        state, reason = probed
        if state.lower() not in ("error", "stopped"):
            return None
        if not self.missing():
            return None
        detail = reason or f"cluster reported {state!r} with no reason"
        self.dead_clusters.add(cluster)
        log.warning(
            "shard_fleet_stopped",
            run_id=self.run_id,
            tile=self.tile,
            stage=self.stage,
            cluster_id=cluster,
            state=state,
            reason=detail,
            missing=len(missing),
        )
        return detail


def ensure_started(
    *,
    stage: str,
    run_id: str,
    tile: str,
    root: str,
    storage: StorageBackend,
    indexes: Sequence[int],
    submit: Submitter,
    deadline_s: float,
    clock: Clock,
    job: ProcessingJob | None = None,
    units: int | None = None,
) -> bool:
    """Start a stage's first round unless somebody already has. No waiting.

    Two callers need to start a fleet without then blocking on it. The fused
    offsets stage is started *before* its expected artifacts are even knowable
    (they come from a plan its own shard 0 writes), and the composite fleet is
    started mid-way through the offsets barrier so its boot overlaps that
    stage's tail. Both then fall through to the ordinary :class:`StageMachine`,
    which sees the fresh submission record and **adopts** it rather than
    submitting again -- the same machinery that keeps two drivers from
    colliding.

    Returns:
        Whether this call started anything.
    """
    if _submission_records(storage, root, stage):
        return False
    StageMachine(
        stage=stage,
        run_id=run_id,
        tile=tile,
        root=root,
        storage=storage,
        prefix="",
        expected={},
        submit=submit,
        deadline_s=deadline_s,
        clock=clock,
        job=job,
        units=units,
    )._start_round(1, list(indexes))
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
    deadline_s: float,
    clock: Clock,
    cluster_probe: ClusterProbe | None = None,
    job: ProcessingJob | None = None,
    units: int | None = None,
    on_poll: Callable[[], None] | None = None,
    early_settle: Callable[[], bool] | None = None,
) -> StageOutcome:
    """Run one stage's barrier to completion. See :class:`StageMachine`."""
    return StageMachine(
        stage=stage,
        run_id=run_id,
        tile=tile,
        root=root,
        storage=storage,
        prefix=prefix,
        expected=expected,
        submit=submit,
        deadline_s=deadline_s,
        clock=clock,
        cluster_probe=cluster_probe,
        job=job,
        units=units,
        on_poll=on_poll,
        early_settle=early_settle,
    ).run()


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
    deadline_s: float,
    clock: Clock,
    cluster_probe: ClusterProbe | None = None,
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
        deadline_s=deadline_s,
        clock=clock,
        cluster_probe=cluster_probe,
        job=job,
    )


def _bootstrap_deadline_s() -> float:
    """How long the driver waits before the plan exists to budget against.

    A chicken-and-egg: every stage budget is computed from the plan, and the
    plan is written by shard 0 of the very stage being started. So this one
    deadline is built from the fixed costs alone -- a boot and a resolve --
    rather than from geometry nobody has yet. It governs only the wait for
    ``plan.json``; the moment that lands, every later deadline is derived.
    """
    if settings.shard_barrier_timeout_s is not None:
        return float(settings.shard_barrier_timeout_s)
    fixed = budgets.VM_BOOT_S + budgets.RESOLVE_S + settings.shard_plan_wait_s
    return fixed * settings.shard_budget_safety


def _read_plan(
    run_id: str, tile: str, root: str, storage: StorageBackend
) -> shards.TilePlan | None:
    """The plan if shard 0 has already published it, else ``None``."""
    if storage.read_text(shards.plan_key(root)) is None:
        return None
    return shard_tasks.load_context(run_id, tile, storage=storage).plan


def _wait_for_plan(
    run_id: str, tile: str, root: str, storage: StorageBackend, *, clock: Clock, deadline_s: float
) -> shards.TilePlan:
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
    deadline = clock.now() + deadline_s
    while True:
        if storage.read_text(shards.plan_key(root)) is not None:
            return shard_tasks.load_context(run_id, tile, storage=storage).plan
        if clock.now() >= deadline:
            raise ShardStageFailed("offsets", [shards.plan_key(root)])
        clock.sleep(settings.shard_driver_poll_s)


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
    clock: Clock,
    cluster_probe: ClusterProbe | None = None,
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
    started = clock.now()
    cog_keys = [storage.cog_key(plan.window, tile, product) for product in PRODUCTS]
    prefix = cog_keys[0].rsplit("/", 1)[0] + "/"
    expected = {0: cog_keys}

    deadline = clock.now() + settings.shard_export_claim_fallback_s
    while _missing(storage, prefix, expected):
        if clock.now() >= deadline:
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
                deadline_s=budgets.stage_budget("export", plan).deadline_s,
                clock=clock,
                cluster_probe=cluster_probe,
            )
            outcome.wall_s += clock.now() - started
            return outcome
        clock.sleep(settings.shard_driver_poll_s)

    log.info("shard_export_claimed_by_worker", run_id=run_id, tile=tile)
    return StageOutcome(
        stage="export",
        shards=1,
        already_done=1,
        submissions=0,
        wall_s=clock.now() - started,
    )


def drive_tile(
    job: ProcessingJob,
    *,
    run_id: str | None = None,
    storage: StorageBackend | None = None,
    submit: Submitter | None = None,
    clock: Clock | None = None,
    cluster_probe: ClusterProbe | None = coiled_cluster_probe,
    balance_source: Callable[[], quota.CreditBalance] | None = None,
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
    _preflight(submit, balance_source)
    run_id = run_id or shard_run_id(job)
    return _drive(
        run_id=run_id,
        tile=job.tile.name,
        job=job,
        storage=storage,
        submit=submit,
        clock=clock or Clock(),
        cluster_probe=cluster_probe,
    )


def resume_tile(
    run_id: str,
    tile: str,
    *,
    storage: StorageBackend | None = None,
    submit: Submitter | None = None,
    clock: Clock | None = None,
    cluster_probe: ClusterProbe | None = coiled_cluster_probe,
    balance_source: Callable[[], quota.CreditBalance] | None = None,
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
    _preflight(submit, balance_source)
    root = shards.shard_root(run_id, tile)

    if storage.read_text(shards.plan_key(root)) is None:
        msg = (
            f"run {run_id!r} published no plan for {tile} at {shards.plan_key(root)}; "
            "there is nothing to resume -- start it with `landsat-lst shard process`"
        )
        raise FileNotFoundError(msg)

    return _drive(
        run_id=run_id,
        tile=tile,
        job=None,
        storage=storage,
        submit=submit,
        clock=clock or Clock(),
        cluster_probe=cluster_probe,
    )


def _offsets_cached(
    run_id: str,
    tile: str,
    storage: StorageBackend,
    plan: shards.TilePlan | None = None,
) -> bool:
    """Whether the canonical offsets record already covers this tile.

    Delegates to the check ``shard_tasks.merge_offsets`` makes, so the driver
    and the merge cannot disagree. Never raises: a record that cannot be read
    is a record the stage will rebuild -- and the offsets barrier asks again on
    every poll, so one failed read no longer decides the stage on its own.

    Pass ``plan`` once there is one. Without it every call re-reads and
    re-parses ``items.json``, which this question never looks at.
    """
    try:
        return shard_tasks.offsets_record_present(run_id, tile, storage=storage, plan=plan)
    except Exception as exc:
        log.warning("shard_offsets_cache_check_failed", tile=tile, error=repr(exc)[:200])
        return False


def _tile_budget_s(plan: shards.TilePlan) -> float:
    """How long this tile may run. See :func:`quota.tile_horizon_s`.

    Delegated rather than repeated, because the fleet preflight asks the same
    question without a plan and two definitions of a tile's horizon would
    drift. ``quota.BUDGET_STAGES`` owns the stage order.
    """
    return quota.tile_horizon_s(plan)


def _drive(
    *,
    run_id: str,
    tile: str,
    job: ProcessingJob | None,
    storage: StorageBackend,
    submit: Submitter,
    clock: Clock,
    cluster_probe: ClusterProbe | None,
) -> TileRunSummary:
    """Drive one tile, and sweep its coarse stage whatever happens.

    The sweep is the whole reason this wrapper exists. Phase A stages its
    coarse observations so phase B stops re-reading the sources (issue #125),
    and on S30W065 that is 8,424 objects and 167 GB per tile. Sweeping only on
    an offsets-stage failure left exactly that behind for each of the two runs
    that died in the composite stage on 2026-09-04.

    A driver that reaches any terminal path is done reading the stage, whether
    it succeeded, failed, or was interrupted. ``sweep_coarse_stage`` swallows
    its own exceptions, so cleanup can never stand in front of the real error.
    """
    try:
        return _drive_stages(
            run_id=run_id,
            tile=tile,
            job=job,
            storage=storage,
            submit=submit,
            clock=clock,
            cluster_probe=cluster_probe,
        )
    finally:
        shard_tasks.sweep_coarse_stage(run_id, tile, storage=storage)


def _drive_stages(
    *,
    run_id: str,
    tile: str,
    job: ProcessingJob | None,
    storage: StorageBackend,
    submit: Submitter,
    clock: Clock,
    cluster_probe: ClusterProbe | None,
) -> TileRunSummary:
    """The stage sequence, shared by a fresh run and a resumed one.

    Two fleets, not five. The offsets side -- resolve, climatology, phase-A
    barrier, per-scene offsets -- is one fused task type
    (:func:`landsat_lst.shard_tasks.run_offsets_stage`), because an offsets-side
    shard computed for about six minutes while its stage held a fleet for about
    thirty: boots and queueing dominated. The composite fleet starts while that
    one is still finishing, and the export is claimed by whichever composite
    worker writes the last band. See ADR-016.

    Every deadline below comes from :mod:`landsat_lst.budgets` -- bytes over
    measured rates, per shard, times one named safety factor -- rather than
    from a hand-entered stage timeout. And every wait goes through ``clock``,
    so the whole sequence is testable in milliseconds.
    """
    root = shards.shard_root(run_id, tile)
    summary = TileRunSummary(run_id=run_id, tile=tile, window=job.window_label if job else "")

    # The fused fleet's width has to be settled before the plan exists, because
    # shard 0 of this very fleet is what writes the plan. It travels with the
    # task command so the planner cuts the plan to the fleet that will run it.
    units = shards.offsets_fleet_units()
    plan = _read_plan(run_id, tile, root, storage)

    # The estimate is keyed by scene set, not by run id, so an earlier run over
    # the same scenes already answered this stage. Asking needs a plan, because
    # the key's digest covers the sorted scene ids -- so a fresh run id asks
    # again below, once shard 0 has published one.
    offsets_cached = plan is not None and _offsets_cached(run_id, tile, storage, plan)

    # Start the fused fleet only when there is offsets work left. A resumed run
    # whose offsets finished has a plan and every partial, and must start
    # nothing -- which is the whole point of resuming.
    if not offsets_cached and (
        plan is None
        or _missing(storage, f"{root}/offsets/scene/", _expected_keys(plan, "offsets", root))
    ):
        ensure_started(
            stage="offsets",
            run_id=run_id,
            tile=tile,
            root=root,
            storage=storage,
            indexes=list(range(units)),
            submit=submit,
            deadline_s=_bootstrap_deadline_s(),
            clock=clock,
            job=job,
            units=units,
        )

    if plan is None:
        plan = _wait_for_plan(
            run_id, tile, root, storage, clock=clock, deadline_s=_bootstrap_deadline_s()
        )
        # First point at which a fresh run can know its own offsets key. The
        # fleet is booting by now; its shards make the same check and exit.
        offsets_cached = _offsets_cached(run_id, tile, storage, plan)
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
    for line in budgets.tile_budget_lines(plan):
        log.info("shard_stage_budget", run_id=run_id, tile=tile, budget=line)

    # A tile outlives a session. On 2026-09-04 the composite overlap submission
    # failed with UnauthorizedSSOTokenError eighteen minutes after the identity
    # preflight passed, because that preflight asks whether the session works
    # now and never how long it has left. This is the first point a budget
    # exists to compare it against.
    #
    # Skipped for an injected submitter, on the same terms as every other gate
    # in this module: a caller with its own fleet has no AWS session in play,
    # and a gate that reads the machine rather than the code is a gate that
    # passes on a laptop and fails on a credential-less runner.
    if submit is submit_shard_stage:
        quota.preflight_session_lifetime(needed_s=_tile_budget_s(plan))

    # Started from inside the offsets barrier, the moment phase B is
    # demonstrably producing. Guarded by a flag rather than by the submission
    # record alone so the driver does not list on every poll.
    composite_started = False

    def _start_composite(*, reason: str) -> None:
        nonlocal composite_started
        if composite_started:
            return
        composite_started = ensure_started(
            stage="composite",
            run_id=run_id,
            tile=tile,
            root=root,
            storage=storage,
            indexes=list(range(len(plan.bands))),
            submit=submit,
            deadline_s=budgets.stage_budget("composite", plan).deadline_s,
            clock=clock,
        )
        if composite_started:
            log.info(
                "shard_composite_started",
                run_id=run_id,
                tile=tile,
                bands=len(plan.bands),
                reason=reason,
            )

    def _overlap() -> None:
        if composite_started or not _overlap_ready(storage, root, plan):
            return
        _start_composite(reason="offsets_overlap")

    if offsets_cached:
        # Nothing to wait for and nothing to merge. The stage's whole output is
        # the record, and the record is already at its canonical key.
        log.info("shard_offsets_cache_hit", run_id=run_id, tile=tile)
        _start_composite(reason="offsets_cached")
    else:
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
                deadline_s=budgets.stage_budget("offsets", plan).deadline_s,
                clock=clock,
                cluster_probe=cluster_probe,
                job=job,
                units=units,
                on_poll=_overlap,
                # The driver's own pre-barrier check is one instant; every
                # offsets shard makes the same check on every boot. A record
                # landing between the two -- a concurrent run over the same
                # scene set, or one failed read here -- makes each shard exit
                # without a partial and leaves this barrier waiting for work
                # nobody will do. Asking again on every poll closes that.
                early_settle=lambda: _offsets_cached(run_id, tile, storage, plan),
            )
        )

        # In the driver, not on a VM: a kilobyte of JSON in, 600 floats out.
        # The composite shards are already booting and polling for this record.
        merged = clock.now()
        key = shard_tasks.merge_offsets(run_id, tile, storage=storage)
        summary.stages.append(
            StageOutcome(
                stage="merge_offsets",
                shards=1,
                already_done=0,
                submissions=0,
                wall_s=clock.now() - merged,
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
            deadline_s=budgets.stage_budget("composite", plan).deadline_s,
            clock=clock,
            cluster_probe=cluster_probe,
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
            clock=clock,
            cluster_probe=cluster_probe,
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
