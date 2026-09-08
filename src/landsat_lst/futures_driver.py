"""One tile as futures on a scheduler that owns the dependencies.

The Batch driver (:mod:`landsat_lst.shard_driver`) sequences stages by polling
S3 barriers, with rounds, adoption, cluster probes, and submission records of
its own. Here the scheduler owns the edges, the retries, and the task state:
a plan future, then an offsets future per unit that depends on it, a merge
future that depends on every offsets future, a composite future per band that
depends on the merge, and an export future that depends on every composite
future. Each future runs :func:`landsat_lst.futures_tasks.run_shard_task`,
which is the same shard the Batch path runs, with its own log, heartbeat,
attempt number, and inner trace (ADR-021).

What this module does not do, on purpose:

- It never imports coiled, frisky, or distributed. It takes an
  :class:`Executor` and a :class:`RunObserver`, both duck-typed, so the whole
  state machine runs against an in-memory fake in tests, credential-less.
- It never polls a cluster to decide task lifecycle. Completion is a future
  resolving, and a shard's *durable* completion is its artifact in the bucket,
  which is checked before any error is classified: a shard that published and
  then died is a success.
- It never trusts a clock it did not create. One :class:`Deadline` is created
  before the cluster exists and checked at every blocking point: waiting for
  workers, waiting for the plan, and every completion of the settle loop. On
  expiry every unfinished future is released and the session is closed; a
  running opaque task cannot be interrupted, so shutdown is what stops it.

Three limits bound spend, in decreasing strength. The worker count is fixed
before launch and never exceeds ``settings.futures_max_workers`` on any stage.
The run deadline is derived from the budget model and covers boot and every
wait. The credit cap is a preflight refusal plus a best-effort stop from a
balance poll, and it is labelled as such: Coiled usage lags billing, and other
jobs on the account move the same balance.

Two things about the fused offsets stage that a reader of the Batch driver
would not expect. Its shards hold an in-process barrier
(``shard_tasks.wait_for_blocks``) and must all run concurrently, so the offsets
width must not exceed the worker cap, and this refuses when it does. And an
offsets index still pending while its peers run past
``settings.futures_offsets_stall_s`` is reported as ``stalled``; the scheduler
cannot see that barrier, so the driver names it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

import structlog

from landsat_lst import budgets, shards
from landsat_lst.config import settings
from landsat_lst.futures_tasks import run_shard_task, session_token, task_key
from landsat_lst.shard_driver import (
    Clock,
    ShardBackendMismatch,
    StageOutcome,
    TileRunSummary,
    _expected_keys,
    _missing,
    _read_plan,
    classify_failure,
    shard_run_id,
)
from landsat_lst.storage import S3Storage, get_storage

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from landsat_lst.models import ProcessingJob
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

RunStatus = Literal["accepted", "completed_unobserved", "incomplete", "failed"]


# --- the seams ------------------------------------------------------------------------


class Future(Protocol):
    """What the driver needs from a future, on either scheduler."""

    key: str

    def done(self) -> bool: ...

    def result(self, timeout: float | None = None) -> Any: ...

    def release(self) -> None: ...


class Executor(Protocol):
    """A cluster session, duck-typed so a test can hand in a fake.

    ``submit`` always submits impure work (a shard is never memoized by key on
    a scheduler), ``as_completed`` yields futures in completion order without
    raising on errors, and ``ensure_workers`` blocks until ``n`` workers are
    present or ``timeout_s`` passes.
    """

    scheduler: str
    dashboard_url: str | None
    cluster_id: object
    n_workers: int

    def submit(
        self, fn: Callable[..., Any], *args: Any, key: str, retries: int, **kw: Any
    ) -> Future: ...

    def as_completed(
        self, futures: Sequence[Future], *, timeout_s: float | None
    ) -> Iterator[Future]: ...

    def ensure_workers(self, n: int, *, timeout_s: float) -> None: ...

    def worker_addresses(self) -> list[str]: ...


@dataclass(frozen=True)
class ShardRef:
    """One submitted future, as the observer and the summary know it."""

    key: str
    stage: str
    tile: str
    index: int
    depends_on: tuple[str, ...]
    submitted_at: float
    state_key: str
    log_key: str
    artifact_keys: tuple[str, ...]


class RunObserver(Protocol):
    """The outer view: what the driver tells whoever watches the run.

    Every method is best-effort on the observer's side. The driver calls them
    and moves on; an observer that raises is a bug in the observer, not a
    reason to stop the tile, so the driver guards each call.
    """

    def register(self, ref: ShardRef) -> None: ...

    def note_completion(
        self, key: str, *, result: Mapping[str, Any] | None, error: str | None
    ) -> None: ...

    def note_stalled(self, keys: Sequence[str]) -> None: ...

    def tick(self) -> None: ...

    def verify(self, phase: str) -> Mapping[str, Any]: ...

    def finalize(self) -> Mapping[str, Any]: ...


class NullObserver:
    """The observer a test or a diagnosis run gets when none is attached."""

    def register(self, ref: ShardRef) -> None:  # noqa: ARG002
        return None

    def note_completion(
        self,
        key: str,  # noqa: ARG002
        *,
        result: Mapping[str, Any] | None,  # noqa: ARG002
        error: str | None,  # noqa: ARG002
    ) -> None:
        return None

    def note_stalled(self, keys: Sequence[str]) -> None:  # noqa: ARG002
        return None

    def tick(self) -> None:
        return None

    def verify(self, phase: str) -> Mapping[str, Any]:  # noqa: ARG002
        return {}

    def finalize(self) -> Mapping[str, Any]:
        return {}


# --- the deadline -----------------------------------------------------------------


class RunTimeout(RuntimeError):
    """The run's wall-clock budget expired at a named blocking point."""

    def __init__(self, what: str, *, elapsed_s: float, total_s: float) -> None:
        self.what = what
        self.elapsed_s = elapsed_s
        self.total_s = total_s
        super().__init__(
            f"run deadline of {total_s / 60:.0f} min expired after {elapsed_s / 60:.0f} min, "
            f"while {what}"
        )


class Deadline:
    """One wall-clock budget for the whole run, created before the cluster.

    Idle timeouts reap a cluster nobody is using; they say nothing about a run
    that is busy for too long. This is the hard stop. It is checked wherever
    the driver blocks, and the check names the blocking point so an expiry
    says *where* the time went, which is the first question anyone asks.
    """

    def __init__(self, *, clock: Clock, total_s: float) -> None:
        self.clock = clock
        self.total_s = float(total_s)
        self.started = clock.now()

    def elapsed_s(self) -> float:
        return self.clock.now() - self.started

    def remaining_s(self) -> float:
        return max(0.0, self.total_s - self.elapsed_s())

    def expired(self) -> bool:
        return self.elapsed_s() >= self.total_s

    def check(self, what: str) -> None:
        if self.expired():
            raise RunTimeout(what, elapsed_s=self.elapsed_s(), total_s=self.total_s)


def derive_run_timeout_s(plan: shards.TilePlan | None, *, bands: int, workers: int) -> float:
    """The run's wall-clock budget from the same model the barriers use.

    Boot, the fused offsets stage, and the composite stage times the number of
    waves the band count needs on ``workers`` workers, times the safety factor.
    Without a plan (a fresh run, before shard 0 resolves one) the plan-wait
    allowance stands in for the two stages; the caller re-derives once the
    plan exists. ``settings.futures_run_timeout_s`` overrides it outright.
    """
    if settings.futures_run_timeout_s is not None:
        return float(settings.futures_run_timeout_s)
    if plan is None:
        return (budgets.VM_BOOT_S + settings.shard_plan_wait_s) * settings.shard_budget_safety
    offsets = budgets.offsets_stage_budget(plan)
    composite = budgets.composite_stage_budget(plan)
    boot, band_work = budgets.split_boot(composite)
    waves = max(1, -(-max(1, bands) // max(1, workers)))
    total = boot + offsets.work_s + band_work * waves + budgets.merge_budget().work_s
    return total * settings.shard_budget_safety


# --- records ----------------------------------------------------------------------


@dataclass
class TileFutures:
    """Every future of one tile, by stage."""

    plan: Future | None = None
    offsets: dict[int, Future] = field(default_factory=dict)
    merge: Future | None = None
    composite: dict[int, Future] = field(default_factory=dict)
    export: Future | None = None

    def all(self) -> list[Future]:
        out: list[Future] = []
        if self.plan is not None:
            out.append(self.plan)
        out.extend(self.offsets.values())
        if self.merge is not None:
            out.append(self.merge)
        out.extend(self.composite.values())
        if self.export is not None:
            out.append(self.export)
        return out

    def stage_of(self, key: str) -> tuple[str, int]:
        if self.plan is not None and self.plan.key == key:
            return "resolve", 0
        for index, fut in self.offsets.items():
            if fut.key == key:
                return "offsets", index
        if self.merge is not None and self.merge.key == key:
            return "merge", 0
        for index, fut in self.composite.items():
            if fut.key == key:
                return "composite", index
        if self.export is not None and self.export.key == key:
            return "export", 0
        msg = f"unknown future key {key!r}"
        raise KeyError(msg)


@dataclass
class FuturesRunSummary(TileRunSummary):
    """A :class:`TileRunSummary` with what only a futures run can say."""

    scheduler: str = ""
    dashboard_url: str | None = None
    cluster_id: object = None
    n_workers: int = 0
    status: RunStatus = "incomplete"
    bands_requested: list[int] = field(default_factory=list)
    finalize: bool = True
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    missing: dict[str, list[int]] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)
    deadline_s: float = 0.0
    stopped_by: str | None = None
    offsets_cached: bool = False

    def as_dict(self) -> dict:
        return {
            **super().as_dict(),
            "scheduler": self.scheduler,
            "dashboard_url": self.dashboard_url,
            "cluster_id": self.cluster_id,
            "n_workers": self.n_workers,
            "status": self.status,
            "bands_requested": self.bands_requested,
            "finalize": self.finalize,
            "results": self.results,
            "errors": self.errors,
            "missing": self.missing,
            "verification": self.verification,
            "observability": self.observability,
            "deadline_s": round(self.deadline_s, 1),
            "stopped_by": self.stopped_by,
            "offsets_cached": self.offsets_cached,
        }


class ShardFuturesFailed(RuntimeError):
    """The tile did not complete, and this says why and what is missing."""

    def __init__(
        self,
        cause: str,
        *,
        terminal: bool,
        missing: Mapping[str, Sequence[int]],
        detail: str = "",
        summary: FuturesRunSummary | None = None,
    ) -> None:
        self.cause = cause
        self.terminal = terminal
        self.missing = {stage: list(indexes) for stage, indexes in missing.items()}
        self.detail = detail
        self.summary = summary
        listed = "; ".join(
            f"{stage} {indexes[:8]}{' ...' if len(indexes) > 8 else ''}"
            for stage, indexes in self.missing.items()
            if indexes
        )
        kind = "terminal" if terminal else "transient"
        super().__init__(
            f"{kind} failure ({cause}){': ' + detail if detail else ''}; "
            f"missing: {listed or 'nothing'}"
        )


# --- classification and reconciliation ------------------------------------------------


_TERMINAL_TYPES = frozenset({"IdentityRefused", "WriteAccessRefused", "QuotaRefused"})
_TERMINAL_TEXT = ("plan digest", "not comparable", "inner scheduler", "band slab", "is missing")


def classify_future_error(exc: BaseException) -> Literal["terminal", "transient"]:
    """Whether a future's error can heal on a retry.

    Terminal: the control-plane markers :func:`shard_driver.classify_failure`
    knows, the preflight refusals, a plan whose digest drifted, the export's
    missing slab (a settled stage cannot heal it), and the wrapper's refusal of
    a wrong inner scheduler. Everything else is transient, *including an error
    with no message*: guessing "terminal" for the unknown case is what killed
    a driver once (ADR-016).

    Matches on type name and text, never on a class import, so the driver
    stays free of distributed and frisky.
    """
    if classify_failure(exc) == "terminal":
        return "terminal"
    name = type(exc).__name__
    if name in _TERMINAL_TYPES:
        return "terminal"
    text = str(exc).lower()
    if name == "ValueError" and any(marker in text for marker in ("digest", "drift", "refused")):
        return "terminal"
    if name == "RuntimeError" and "inner scheduler" in text:
        return "terminal"
    if name == "FileNotFoundError" and ("band slab" in text or "slab" in text):
        return "terminal"
    return "transient"


def reconcile(
    storage: StorageBackend, plan: shards.TilePlan, root: str, *, bands: Sequence[int] | None = None
) -> dict[str, list[int]]:
    """Expected artifacts against the bucket, stage by stage.

    The single definition of "done" is :func:`shard_driver._expected_keys`;
    this only asks the bucket. ``bands`` narrows the composite stage to the
    bands this run asked for, so a bounded run reconciles against its own
    request rather than the whole tile.
    """
    out: dict[str, list[int]] = {}
    for stage, prefix in (
        ("climatology", f"{root}/offsets/ref/"),
        ("offsets", f"{root}/offsets/scene/"),
        ("composite", f"{root}/composite/"),
    ):
        expected = _expected_keys(plan, stage, root)
        if stage == "composite" and bands is not None:
            expected = {i: keys for i, keys in expected.items() if i in set(bands)}
        out[stage] = _missing(storage, prefix, expected)
    out["export"] = [] if storage.cog_exists(plan.window, plan.tile) else [0]
    return out


# --- the driver -----------------------------------------------------------------------


@dataclass
class _Run:
    """Everything one drive holds, in one place, so helpers stay small."""

    run_id: str
    tile: str
    root: str
    storage: StorageBackend
    executor: Executor
    observer: RunObserver
    clock: Clock
    deadline: Deadline
    task: Callable[..., Any]
    token: str
    bands: list[int] | None
    finalize: bool
    inner_scheduler: str
    credit_stop: Callable[[], str | None] | None
    summary: FuturesRunSummary
    futures: TileFutures = field(default_factory=TileFutures)
    refs: dict[str, ShardRef] = field(default_factory=dict)
    started: dict[str, float] = field(default_factory=dict)
    deps: dict[str, list[Future]] = field(default_factory=dict)
    resubmits: dict[tuple[str, int], int] = field(default_factory=dict)
    plan: shards.TilePlan | None = None
    units: int = 0
    offsets_cached: bool = False

    def observe(self, what: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:  # the observer never fails a tile
            log.warning("futures_observer_failed", what=what, error=str(exc))
            return None


def _refuse_storage(storage: StorageBackend, executor: Executor) -> None:
    """The same refusal ``shard_driver.require_shared_storage`` makes.

    Workers write S3 always; a driver on local storage would reconcile against
    a directory nothing writes to, which is the hang shape ADR-016 records. A
    fake executor is exempt: it writes where the driver reads.
    """
    if getattr(executor, "scheduler", "") == "fake":
        return
    if not isinstance(storage, S3Storage):
        msg = (
            "the futures driver polls its shards' outputs, and Coiled workers always "
            f"write S3; storage is {type(storage).__name__}. Set LST_STORAGE_BACKEND=s3."
        )
        raise ShardBackendMismatch(msg)


def _submit(
    run: _Run,
    stage: str,
    index: int,
    deps: Sequence[Future],
    *,
    job: ProcessingJob | None = None,
    units: int | None = None,
    artifact_keys: Sequence[str] = (),
) -> Future:
    """Submit one shard as a future and register it with the observer."""
    times = run.resubmits.get((stage, index), 0)
    token = run.token if times == 0 else f"{run.token}r{times}"
    key = task_key(run.run_id, run.tile, stage, index, token)
    # ``outer_key`` rather than ``key``: the executor consumes ``key`` as the
    # future's own name, and a task keyword by the same name never arrives.
    kwargs: dict[str, Any] = {
        "outer_key": key,
        "inner_scheduler": run.inner_scheduler,
        "use_frisky_sink": run.executor.scheduler == "frisky",
    }
    if job is not None:
        kwargs["job"] = job
    if units is not None:
        kwargs["units"] = units
    future = run.executor.submit(
        run.task,
        stage,
        run.run_id,
        run.tile,
        index,
        *deps,
        key=key,
        retries=settings.futures_retries,
        **kwargs,
    )
    attempt_hint = 1
    ref = ShardRef(
        key=key,
        stage=stage,
        tile=run.tile,
        index=index,
        depends_on=tuple(d.key for d in deps),
        submitted_at=run.clock.now(),
        state_key=shards.shard_state_key(run.root, stage, index, attempt_hint),
        log_key=shards.shard_log_key(run.root, stage, index, attempt_hint),
        artifact_keys=tuple(artifact_keys),
    )
    run.refs[key] = ref
    run.started[key] = run.clock.now()
    run.deps[key] = list(deps)
    run.observe("register", lambda: run.observer.register(ref))
    log.info("futures_submitted", stage=stage, index=index, key=key, depends_on=len(deps))
    return future


def _ensure_workers(run: _Run, n: int, *, what: str) -> None:
    run.deadline.check(f"waiting for {n} workers before {what}")
    run.executor.ensure_workers(n, timeout_s=run.deadline.remaining_s())
    run.deadline.check(f"waiting for {n} workers before {what}")


def _offsets_cached(run: _Run) -> bool:
    """Whether the merged offsets record for this plan is already in the cache.

    ADR-012: only the estimate is cached, and it is keyed by the scene set and
    the settings that decide it. A run whose plan already has its record pays
    no offsets stage and no merge; the composite shards read the record back.
    This is what lets a bounded run compute one band from a retained plan
    without booting fifteen offsets shards it does not need.
    """
    if run.plan is None:
        return False
    from landsat_lst.shard_tasks import _offset_key  # noqa: PLC0415

    key = _offset_key(run.plan).storage_key
    present = run.storage.read_text(key) is not None
    if present:
        log.info("futures_offsets_cached", key=key)
    return present


def submit_prepare(run: _Run, *, job: ProcessingJob | None) -> None:
    """The plan future (when no plan exists) and every offsets future.

    When the plan exists and its merged offsets record is cached, no offsets
    future and no merge future are submitted: the stage is already done in
    the sense that matters, and the fused stage's width need not fit anywhere.
    """
    run.units = shards.offsets_fleet_units()
    run.plan = _read_plan(run.run_id, run.tile, run.root, run.storage)
    if _offsets_cached(run):
        run.offsets_cached = True
        run.summary.offsets_cached = True
        return
    cap = settings.futures_max_workers
    if run.units > cap:
        raise ShardFuturesFailed(
            "offsets_width_exceeds_cap",
            terminal=True,
            missing={},
            detail=(
                f"the fused offsets stage needs {run.units} concurrent shards (its shards "
                f"barrier in-process) but futures_max_workers is {cap}; raise the cap or "
                "lower shard_offset_vms"
            ),
        )
    _ensure_workers(run, run.units, what="the offsets stage")
    deps: list[Future] = []
    if run.plan is None:
        if job is None:
            msg = "a run with no plan needs the job it is resolving; start with `shard process`"
            raise FileNotFoundError(msg)
        run.futures.plan = _submit(
            run,
            "resolve",
            0,
            [],
            job=job,
            units=run.units,
            artifact_keys=(shards.plan_key(run.root),),
        )
        deps = [run.futures.plan]
    # Every offsets index, every time. A shard whose partial exists returns in
    # seconds, and a missing climatology block belongs to an index whose
    # partial may exist, so the cheap rerun of all of them is also the correct
    # one (ADR-016's driver resubmits the whole fused stage for the same reason).
    for index in range(run.units):
        run.futures.offsets[index] = _submit(
            run, "offsets", index, deps, job=job if index == 0 else None, units=run.units
        )


def _await_plan(run: _Run) -> shards.TilePlan:
    if run.plan is not None:
        return run.plan
    assert run.futures.plan is not None
    plan_future = run.futures.plan
    while not plan_future.done():
        run.deadline.check("waiting for shard 0 to publish the plan")
        # Waited through the executor, not a sleep: a bounded ``as_completed``
        # is how every scheduler blocks, and the fake executor runs its
        # tasks inside it.
        wait_s = min(settings.futures_observer_poll_s, max(1.0, run.deadline.remaining_s()))
        _next_landed(run, [plan_future], timeout_s=wait_s)
        run.observe("tick", run.observer.tick)
    plan_future.result()
    plan = _read_plan(run.run_id, run.tile, run.root, run.storage)
    if plan is None:
        raise ShardFuturesFailed(
            "plan_missing",
            terminal=True,
            missing={},
            detail="the resolve future returned but no plan is in the bucket",
        )
    run.plan = plan
    return plan


def submit_rest(run: _Run) -> None:
    """Merge, composite for the requested bands, export when finalizing."""
    assert run.plan is not None
    if run.offsets_cached:
        # The composite shards read the cached record; there is no merge to
        # wait for and no edge to draw. Workers are sized for the bands now.
        upstream: list[Future] = []
        _ensure_workers(
            run,
            min(settings.futures_max_workers, max(1, len(run.bands or run.plan.bands))),
            what="the composite stage",
        )
    else:
        offsets = list(run.futures.offsets.values())
        run.futures.merge = _submit(run, "merge", 0, offsets)
        upstream = [run.futures.merge]
    all_bands = list(range(len(run.plan.bands)))
    requested = all_bands if run.bands is None else [b for b in run.bands if b in all_bands]
    unknown = [] if run.bands is None else [b for b in run.bands if b not in all_bands]
    if unknown:
        raise ShardFuturesFailed(
            "unknown_bands",
            terminal=True,
            missing={},
            detail=f"plan has {len(all_bands)} bands; asked for {unknown}",
        )
    run.summary.bands_requested = requested
    expected = _expected_keys(run.plan, "composite", run.root)
    for index in requested:
        run.futures.composite[index] = _submit(
            run, "composite", index, upstream, artifact_keys=tuple(expected[index])
        )
    if run.finalize:
        run.futures.export = _submit(run, "export", 0, list(run.futures.composite.values()))


def _stalled_offsets(run: _Run) -> list[str]:
    """Offsets futures still pending while peers run past the stall allowance."""
    pending = [f for f in run.futures.offsets.values() if not f.done()]
    if not pending or len(pending) == len(run.futures.offsets):
        return []
    now = run.clock.now()
    return [
        f.key
        for f in pending
        if now - run.started.get(f.key, now) > settings.futures_offsets_stall_s
    ]


def _artifacts_present(run: _Run, key: str) -> bool:
    """Whether the shard behind ``key`` left its artifacts, whatever it raised."""
    ref = run.refs.get(key)
    if ref is None or not ref.artifact_keys:
        return False
    # One listing over the keys' common directory: a composite shard's two
    # products live under sibling prefixes, and listing only the first would
    # read the second as absent.
    directories = [k.rsplit("/", 1)[0] + "/" for k in ref.artifact_keys]
    common = os.path.commonprefix(directories)
    prefix = common[: common.rfind("/") + 1] or common
    present = set(run.storage.list_prefix(prefix))
    return all(k in present for k in ref.artifact_keys)


def _settle(run: _Run) -> None:
    """Drain every future under the deadline, classifying errors as they land."""
    remaining = {f.key: f for f in run.futures.all()}
    state = _SettleState()
    while remaining:
        run.deadline.check(f"waiting for {len(remaining)} futures")
        _check_credit_stop(run)
        stalled = _stalled_offsets(run)
        if stalled:
            run.observe("stalled", lambda keys=stalled: run.observer.note_stalled(keys))
        batch_timeout = min(settings.futures_observer_poll_s, max(1.0, run.deadline.remaining_s()))
        landed = _next_landed(run, list(remaining.values()), timeout_s=batch_timeout)
        run.observe("tick", run.observer.tick)
        for future in landed:
            remaining.pop(future.key, None)
            _handle_landed(run, future, remaining, state)
    if state.terminal is not None:
        state.terminal.missing = _missing_now(run)
        raise state.terminal


@dataclass
class _SettleState:
    terminal: ShardFuturesFailed | None = None
    scaled_after_merge: bool = False


def _check_credit_stop(run: _Run) -> None:
    if run.credit_stop is None:
        return
    reason = run.observe("credit_stop", run.credit_stop)
    if reason:
        raise ShardFuturesFailed(
            "credit_cap", terminal=True, missing=_missing_now(run), detail=str(reason)
        )


def _next_landed(run: _Run, pending: list[Future], *, timeout_s: float) -> list[Future]:
    """The next future to land, or none within ``timeout_s``."""
    try:
        for future in run.executor.as_completed(pending, timeout_s=timeout_s):
            return [future]
    except TimeoutError:
        return []
    return []


def _handle_landed(
    run: _Run, future: Future, remaining: dict[str, Future], state: _SettleState
) -> None:
    stage, index = run.futures.stage_of(future.key)
    try:
        result = future.result()
    except Exception as exc:
        _record_error(run, future.key, stage, index, exc)
        if _artifacts_present(run, future.key):
            log.info("futures_error_after_publish", key=future.key, stage=stage, index=index)
            _resubmit_dependents(run, future.key, remaining)
            return
        if classify_future_error(exc) == "terminal" and state.terminal is None:
            state.terminal = ShardFuturesFailed(
                f"{stage}_terminal",
                terminal=True,
                missing={},
                detail=f"{type(exc).__name__}: {exc}"[:400],
            )
            for other in remaining.values():
                _release(other)
            remaining.clear()
        return
    _record_result(run, future.key, stage, index, result)
    if stage == "merge" and not state.scaled_after_merge:
        state.scaled_after_merge = True
        width = min(settings.futures_max_workers, max(1, len(run.futures.composite)))
        _ensure_workers(run, width, what="the composite stage")


def _resubmit_dependents(run: _Run, dead_key: str, remaining: dict[str, Future]) -> None:
    """Replace every pending dependent of a future that published and then died.

    The scheduler fails a dependent when its dependency errs, whatever the
    bucket holds. The dependency's artifact is there, and the dependent reads
    the bucket rather than the value, so the dependent is submitted again
    with the dead edge removed. The new key carries a resubmission marker so
    no scheduler returns the old, failed future for it.
    """
    for key, deps in list(run.deps.items()):
        if dead_key not in {d.key for d in deps}:
            continue
        old = remaining.get(key)
        if old is None or old.done():
            continue
        stage, index = run.futures.stage_of(key)
        _release(old)
        remaining.pop(key, None)
        run.resubmits[(stage, index)] = run.resubmits.get((stage, index), 0) + 1
        kept = [d for d in deps if d.key != dead_key]
        ref = run.refs[key]
        new = _submit(run, stage, index, kept, artifact_keys=ref.artifact_keys)
        _replace_future(run, stage, index, new)
        remaining[new.key] = new
        log.info("futures_resubmitted_dependent", key=new.key, after=dead_key)


def _replace_future(run: _Run, stage: str, index: int, future: Future) -> None:
    if stage == "resolve":
        run.futures.plan = future
    elif stage == "offsets":
        run.futures.offsets[index] = future
    elif stage == "merge":
        run.futures.merge = future
    elif stage == "composite":
        run.futures.composite[index] = future
    elif stage == "export":
        run.futures.export = future


def _release(future: Future) -> None:
    try:
        future.release()
    except Exception as exc:  # a future that cannot be released is already gone
        log.warning("futures_release_failed", key=getattr(future, "key", "?"), error=str(exc))


def _record_result(run: _Run, key: str, stage: str, index: int, result: Any) -> None:
    record: dict[str, Any] = (
        result.as_dict() if hasattr(result, "as_dict") else {"value": str(result)[:200]}
    )
    record.setdefault("stage", stage)
    record.setdefault("index", index)
    record["key"] = key
    record["wall_s_outer"] = round(run.clock.now() - run.started.get(key, run.clock.now()), 1)
    run.summary.results.append(record)
    run.observe("completion", lambda: run.observer.note_completion(key, result=record, error=None))
    log.info("futures_completed", stage=stage, index=index, key=key, skipped=record.get("skipped"))


def _record_error(run: _Run, key: str, stage: str, index: int, exc: BaseException) -> None:
    error = f"{type(exc).__name__}: {exc}"[:400]
    run.summary.errors.append(
        {
            "key": key,
            "stage": stage,
            "index": index,
            "error": error,
            "class": classify_future_error(exc),
        }
    )
    run.observe("completion", lambda: run.observer.note_completion(key, result=None, error=error))
    log.warning("futures_failed", stage=stage, index=index, key=key, error=error)


def _missing_now(run: _Run) -> dict[str, list[int]]:
    if run.plan is None:
        return {"resolve": [0]}
    return reconcile(run.storage, run.plan, run.root, bands=run.summary.bands_requested or None)


def _stage_outcomes(run: _Run) -> list[StageOutcome]:
    """One outcome per stage, from the results the futures returned."""
    outcomes: list[StageOutcome] = []
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for record in run.summary.results:
        by_stage.setdefault(str(record.get("stage")), []).append(record)
    for stage in ("resolve", "offsets", "merge", "composite", "export"):
        records = by_stage.get(stage)
        submitted = {
            "resolve": 1 if run.futures.plan else 0,
            "offsets": len(run.futures.offsets),
            "merge": 1 if run.futures.merge else 0,
            "composite": len(run.futures.composite),
            "export": 1 if run.futures.export else 0,
        }[stage]
        if not submitted:
            continue
        wall = max((float(r.get("wall_s_outer") or 0.0) for r in records or []), default=0.0)
        outcomes.append(
            StageOutcome(
                stage=stage,
                shards=submitted,
                already_done=sum(1 for r in records or [] if r.get("skipped")),
                submissions=1,
                wall_s=wall,
                rounds=1,
                cluster_ids=[
                    run.executor.cluster_id if isinstance(run.executor.cluster_id, int) else None
                ],
            )
        )
    return outcomes


def _finish(run: _Run, *, stopped_by: str | None) -> FuturesRunSummary:
    summary = run.summary
    summary.stages = _stage_outcomes(run)
    summary.stopped_by = stopped_by
    if run.plan is not None:
        summary.missing = reconcile(
            run.storage, run.plan, run.root, bands=summary.bands_requested or None
        )
        requested_done = not summary.missing.get("composite")
        offsets_ok = run.offsets_cached or not summary.missing.get("offsets")
        summary.completed = (
            run.storage.cog_exists(run.plan.window, run.tile)
            if run.finalize
            else requested_done and offsets_ok
        )
    summary.verification = dict(
        run.observe("verify", lambda: run.observer.verify("complete")) or {}
    )
    summary.observability = dict(run.observe("finalize", run.observer.finalize) or {})
    if stopped_by is not None:
        summary.status = "failed"
    elif not summary.completed:
        summary.status = "incomplete"
    elif summary.observability.get("passed") is True:
        summary.status = "accepted"
    else:
        summary.status = "completed_unobserved"
    summary.deadline_s = run.deadline.total_s
    _publish_summary(run)
    return summary


def _publish_summary(run: _Run) -> None:
    key = f"{shards.fleet_root(run.run_id)}/{run.tile}/state/futures.summary.json"
    try:
        run.storage.write_text(key, json.dumps(run.summary.as_dict(), indent=2, default=str))
    except Exception as exc:  # the summary is evidence, not a step
        log.warning("futures_summary_write_failed", key=key, error=str(exc))


def _drive(
    *,
    run_id: str,
    tile: str,
    job: ProcessingJob | None,
    storage: StorageBackend,
    executor: Executor,
    observer: RunObserver,
    clock: Clock,
    deadline: Deadline,
    task: Callable[..., Any],
    bands: Sequence[int] | None,
    finalize: bool,
    inner_scheduler: str,
    credit_stop: Callable[[], str | None] | None,
) -> FuturesRunSummary:
    _refuse_storage(storage, executor)
    root = shards.shard_root(run_id, tile)
    window = job.window_label if job is not None else _window_from_plan(run_id, tile, root, storage)
    run = _Run(
        run_id=run_id,
        tile=tile,
        root=root,
        storage=storage,
        executor=executor,
        observer=observer,
        clock=clock,
        deadline=deadline,
        task=task,
        token=session_token(),
        bands=list(bands) if bands is not None else None,
        finalize=finalize,
        inner_scheduler=inner_scheduler,
        credit_stop=credit_stop,
        summary=FuturesRunSummary(
            run_id=run_id,
            tile=tile,
            window=window,
            scheduler=executor.scheduler,
            dashboard_url=executor.dashboard_url,
            cluster_id=executor.cluster_id,
            n_workers=executor.n_workers,
            finalize=finalize,
            deadline_s=deadline.total_s,
        ),
    )
    try:
        submit_prepare(run, job=job)
        run.observe("verify", lambda: run.observer.verify("submitted"))
        plan = _await_plan(run)
        for line in budgets.tile_budget_lines(plan):
            log.info("futures_budget", line=line)
        submit_rest(run)
        run.observe("verify", lambda: run.observer.verify("submitted"))
        _settle(run)
    except RunTimeout as exc:
        for future in run.futures.all():
            if not future.done():
                _release(future)
        summary = _finish(run, stopped_by="run_timeout")
        raise ShardFuturesFailed(
            "run_timeout", terminal=False, missing=summary.missing, detail=str(exc), summary=summary
        ) from exc
    except ShardFuturesFailed as exc:
        for future in run.futures.all():
            if not future.done():
                _release(future)
        summary = _finish(run, stopped_by=exc.cause)
        exc.summary = summary
        exc.missing = summary.missing
        raise
    summary = _finish(run, stopped_by=None)
    if not summary.completed:
        raise ShardFuturesFailed(
            "incomplete",
            terminal=False,
            missing=summary.missing,
            detail=(
                f"{len(summary.errors)} future(s) failed; resume with "
                f"`landsat-lst shard resume {run_id} {tile} --executor futures`"
            ),
            summary=summary,
        )
    return summary


def _window_from_plan(run_id: str, tile: str, root: str, storage: StorageBackend) -> str:
    raw = storage.read_text(shards.plan_key(root))
    if raw is None:
        msg = (
            f"run {run_id!r} published no plan for {tile} at {shards.plan_key(root)}; "
            "there is nothing to resume -- start it with `landsat-lst shard process`"
        )
        raise FileNotFoundError(msg)
    return str(json.loads(raw)["window"])


def drive_tile_futures(
    job: ProcessingJob,
    *,
    executor: Executor,
    run_id: str | None = None,
    storage: StorageBackend | None = None,
    observer: RunObserver | None = None,
    task: Callable[..., Any] = run_shard_task,
    clock: Clock | None = None,
    deadline: Deadline | None = None,
    bands: Sequence[int] | None = None,
    finalize: bool = True,
    inner_scheduler: str = "frisky",
    credit_stop: Callable[[], str | None] | None = None,
) -> FuturesRunSummary:
    """Drive one tile through futures on ``executor``.

    Args:
        job: The tile and window. Shard 0 resolves the plan from it.
        executor: The cluster session (:class:`Executor`).
        run_id: Run token; generated when omitted.
        storage: Backend the driver reconciles against. Must be S3 unless the
            executor is a fake.
        observer: The outer view (:class:`RunObserver`); ``None`` attaches none.
        task: The callable each future runs; injectable for tests.
        clock: Time source; injectable for tests.
        deadline: The run's wall-clock budget. Created here from the budget
            model when omitted; a caller that created the cluster under a
            deadline passes the same one so boot counts.
        bands: Composite band indexes to run; ``None`` means every band.
        finalize: Whether to submit the export future. A bounded test passes
            ``False`` so nothing can start the whole tile.
        inner_scheduler: What each shard's own graphs run on; see ADR-021.
        credit_stop: Best-effort spending stop; returns a reason to stop or
            ``None``. Polled once per settle iteration.

    Raises:
        ShardFuturesFailed: The tile did not complete; ``.summary`` and
            ``.missing`` say what is and is not in the bucket.
    """
    storage = storage or get_storage()
    clock = clock or Clock()
    run_id = run_id or shard_run_id(job)
    if deadline is None:
        deadline = Deadline(
            clock=clock,
            total_s=derive_run_timeout_s(
                None, bands=len(bands or []) or 1, workers=executor.n_workers
            ),
        )
    return _drive(
        run_id=run_id,
        tile=job.tile.name,
        job=job,
        storage=storage,
        executor=executor,
        observer=observer or NullObserver(),
        clock=clock,
        deadline=deadline,
        task=task,
        bands=bands,
        finalize=finalize,
        inner_scheduler=inner_scheduler,
        credit_stop=credit_stop,
    )


def resume_tile_futures(
    run_id: str,
    tile: str,
    *,
    executor: Executor,
    storage: StorageBackend | None = None,
    observer: RunObserver | None = None,
    task: Callable[..., Any] = run_shard_task,
    clock: Clock | None = None,
    deadline: Deadline | None = None,
    bands: Sequence[int] | None = None,
    finalize: bool = True,
    inner_scheduler: str = "frisky",
    credit_stop: Callable[[], str | None] | None = None,
) -> FuturesRunSummary:
    """Continue a run from the bucket alone: resubmit everything, recompute nothing.

    Every shard checks its own artifacts first, so a published shard returns in
    seconds with a new attempt number and the summary counts it as skipped.
    A resume before the plan exists cannot work and says so.
    """
    storage = storage or get_storage()
    clock = clock or Clock()
    root = shards.shard_root(run_id, tile)
    if storage.read_text(shards.plan_key(root)) is None:
        _window_from_plan(run_id, tile, root, storage)  # raises the resume message
    if deadline is None:
        plan = _read_plan(run_id, tile, root, storage)
        deadline = Deadline(
            clock=clock,
            total_s=derive_run_timeout_s(
                plan,
                bands=len(bands) if bands else len(plan.bands) if plan else 1,
                workers=executor.n_workers,
            ),
        )
    return _drive(
        run_id=run_id,
        tile=tile,
        job=None,
        storage=storage,
        executor=executor,
        observer=observer or NullObserver(),
        clock=clock,
        deadline=deadline,
        task=task,
        bands=bands,
        finalize=finalize,
        inner_scheduler=inner_scheduler,
        credit_stop=credit_stop,
    )


def summarize_for_operator(summary: FuturesRunSummary) -> Iterable[str]:
    """Lines a person reads after a run, or after a failure with a summary."""
    yield f"status {summary.status}  scheduler {summary.scheduler}  workers {summary.n_workers}"
    if summary.dashboard_url:
        yield f"dashboard {summary.dashboard_url}"
    yield f"bands {summary.bands_requested or 'all'}  finalize {summary.finalize}  deadline {summary.deadline_s / 60:.0f} min"
    for stage, indexes in summary.missing.items():
        if indexes:
            yield f"missing {stage}: {indexes[:12]}{' ...' if len(indexes) > 12 else ''}"
    for error in summary.errors[:8]:
        yield f"error {error['stage']} {error['index']} ({error['class']}): {error['error'][:160]}"
    if summary.stopped_by:
        yield f"stopped by {summary.stopped_by}"
