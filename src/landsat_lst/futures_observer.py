"""The outer view of a futures run, persisted while it runs.

The driver (:mod:`landsat_lst.futures_driver`) knows which futures it
submitted and which came back. The scheduler knows where each one is: pending,
blocked on a dependency, running on a worker, done, failed, or retried after a
worker loss. This module joins the two every ``settings.futures_observer_poll_s``
and writes the join to storage, so an operator reads the run from the bucket
and the evidence outlives the cluster:

- ``_shards/{run_id}/{tile}/state/orchestration.json``, overwritten each poll:
  every shard's state, worker, and error, and every worker's memory and CPU.
- ``orchestration.events.{seq:04d}.jsonl``, one chunk per poll that saw a
  transition or a worker join or loss.
- ``_shards/timings/{run_id}/frisky-spans.{seq:04d}.json``, the Frisky span
  buffer since the previous dump, deduplicated at the cursor.
- ``frisky-verification.json``, cumulative: whether Frisky handled the outer
  futures, as checks with pass/fail and the observed values. This is the
  evidence the ticket asks for in place of a dashboard screenshot.
- ``orchestration.final.json`` and ``frisky-overview.json`` at the end.

Everything here is best-effort. A poll that fails logs and waits for the next
one; a client that disappears ends the observer after ``finalize`` writes what
it has. The driver's summary reads ``finalize()``'s result, and the run's
observability gate is computed from it.

The scheduler is reached through the same duck-typed client the driver holds.
Frisky and plain distributed report state in different shapes;
:func:`normalize_frisky_state` and :func:`normalize_dask_state` turn either
into one :class:`OuterSnapshot`, and the tests feed them canned dicts.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import structlog

from landsat_lst import shards
from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from landsat_lst.futures_driver import ShardRef
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

STATES = ("pending", "blocked", "running", "done", "failed", "retried", "stalled")
ShardState = Literal["pending", "blocked", "running", "done", "failed", "retried", "stalled"]


# --- the registry the driver fills ------------------------------------------------------


class ShardRegistry:
    """Every submitted future, and what the driver has learned about it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._refs: dict[str, ShardRef] = {}
        self._results: dict[str, Mapping[str, Any]] = {}
        self._errors: dict[str, str] = {}
        self._stalled: set[str] = set()
        self._prior_attempts: dict[str, int] = {}

    def register(self, ref: ShardRef) -> None:
        with self._lock:
            self._refs[ref.key] = ref

    def note_completion(
        self, key: str, *, result: Mapping[str, Any] | None, error: str | None
    ) -> None:
        with self._lock:
            if result is not None:
                self._results[key] = dict(result)
            if error is not None:
                self._errors[key] = error

    def note_stalled(self, keys: Sequence[str]) -> None:
        with self._lock:
            self._stalled = set(keys)

    def note_retry(self, key: str) -> None:
        with self._lock:
            self._prior_attempts[key] = self._prior_attempts.get(key, 0) + 1

    def refs(self) -> dict[str, ShardRef]:
        with self._lock:
            return dict(self._refs)

    def view(
        self,
    ) -> tuple[
        dict[str, ShardRef], dict[str, Mapping[str, Any]], dict[str, str], set[str], dict[str, int]
    ]:
        with self._lock:
            return (
                dict(self._refs),
                dict(self._results),
                dict(self._errors),
                set(self._stalled),
                dict(self._prior_attempts),
            )


# --- the snapshot -----------------------------------------------------------------------


@dataclass
class OuterShard:
    key: str
    stage: str
    tile: str
    index: int
    state: str
    worker: str | None = None
    error: str | None = None
    prior_attempts: int = 0
    depends_on: tuple[str, ...] = ()
    submitted_at: float = 0.0
    state_key: str = ""
    log_key: str = ""
    artifact_keys: tuple[str, ...] = ()
    result: dict[str, Any] | None = None


@dataclass
class OuterWorker:
    address: str
    status: str = "unknown"
    cpu_percent: float | None = None
    memory_mb: float | None = None
    memory_limit_mb: float | None = None
    executing: list[str] = field(default_factory=list)


@dataclass
class OuterSnapshot:
    run_id: str
    tile: str
    ts: float
    seq: int
    source: str
    shards: dict[str, OuterShard]
    workers: dict[str, OuterWorker]
    lost_workers: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(STATES, 0)
        for shard in self.shards.values():
            out[shard.state] = out.get(shard.state, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tile": self.tile,
            "ts": self.ts,
            "updated_at": datetime.fromtimestamp(self.ts, tz=UTC).isoformat(),
            "seq": self.seq,
            "source": self.source,
            "counts": self.counts(),
            "shards": {k: asdict(v) for k, v in sorted(self.shards.items())},
            "workers": {k: asdict(v) for k, v in sorted(self.workers.items())},
            "lost_workers": list(self.lost_workers),
        }


def _base_shards(registry: ShardRegistry) -> tuple[dict[str, OuterShard], set[str]]:
    refs, results, errors, stalled, prior = registry.view()
    shards_by_key: dict[str, OuterShard] = {}
    for key, ref in refs.items():
        state: str = "pending"
        if key in results:
            state = "done"
        elif key in errors:
            state = "failed"
        elif key in stalled:
            state = "stalled"
        shards_by_key[key] = OuterShard(
            key=key,
            stage=ref.stage,
            tile=ref.tile,
            index=ref.index,
            state=state,
            error=errors.get(key),
            prior_attempts=prior.get(key, 0),
            depends_on=ref.depends_on,
            submitted_at=ref.submitted_at,
            state_key=ref.state_key,
            log_key=ref.log_key,
            artifact_keys=ref.artifact_keys,
            result=dict(results[key]) if key in results else None,
        )
    return shards_by_key, stalled


def normalize_frisky_state(
    state: Mapping[str, Any],
    registry: ShardRegistry,
    *,
    ts: float,
    seq: int,
    run_id: str,
    tile: str,
) -> OuterSnapshot:
    """One snapshot from ``frisky.Client.get_scheduler_state()``.

    ``Waiting`` with a dependency not in ``Memory`` is ``blocked``, otherwise
    ``pending``; ``Processing`` is ``running`` on ``processing_on``; ``Memory``
    is ``done``; ``Erred`` is ``failed``. A key the registry knows and the
    scheduler does not list is ``pending`` (not yet ingested) unless the driver
    already recorded its outcome. A key with prior attempts still pending or
    blocked is ``retried``.
    """
    shards_by_key, stalled = _base_shards(registry)
    tasks = {str(t.get("key")): t for t in state.get("tasks", []) if isinstance(t, Mapping)}
    memory = {k for k, t in tasks.items() if str(t.get("state")) == "Memory"}
    for key, shard in shards_by_key.items():
        if shard.state in ("done", "failed") or key not in tasks:
            continue
        _apply_frisky_task(shard, tasks[key], memory)
        _apply_retry_and_stall(shard, stalled)
    workers: dict[str, OuterWorker] = {}
    for entry in state.get("workers", []) or []:
        worker = _frisky_worker(entry)
        if worker is not None:
            workers[worker.address] = worker
    return OuterSnapshot(
        run_id=run_id,
        tile=tile,
        ts=ts,
        seq=seq,
        source="frisky",
        shards=shards_by_key,
        workers=workers,
    )


def _apply_frisky_task(shard: OuterShard, task: Mapping[str, Any], memory: set[str]) -> None:
    sched_state = str(task.get("state", ""))
    if sched_state == "Processing":
        shard.state = "running"
        shard.worker = task.get("processing_on")
    elif sched_state == "Memory":
        shard.state = "done"
        who = task.get("who_has") or []
        shard.worker = who[0] if who else shard.worker
    elif sched_state == "Erred":
        shard.state = "failed"
    elif sched_state in ("Waiting", "Queued", "Released"):
        deps = [str(d) for d in (task.get("dependencies") or [])]
        shard.state = "blocked" if any(d not in memory for d in deps) else "pending"


def _apply_retry_and_stall(shard: OuterShard, stalled: set[str]) -> None:
    if shard.state in ("pending", "blocked") and shard.prior_attempts > 0:
        shard.state = "retried"
    if shard.key in stalled and shard.state in ("pending", "blocked", "retried"):
        shard.state = "stalled"


def _frisky_worker(entry: Any) -> OuterWorker | None:
    if isinstance(entry, str):
        return OuterWorker(address=entry)
    if not isinstance(entry, Mapping):
        return None
    address = str(entry.get("address") or entry.get("worker") or "")
    if not address:
        return None
    memory_bytes = entry.get("memory_bytes")
    limit = entry.get("memory_limit_bytes") or entry.get("memory_limit")
    return OuterWorker(
        address=address,
        status=str(entry.get("status") or "alive"),
        cpu_percent=_float(entry.get("cpu_percent")),
        memory_mb=None if memory_bytes is None else round(float(memory_bytes) / 1048576, 1),
        memory_limit_mb=None if limit is None else round(float(limit) / 1048576, 1),
        executing=[str(k) for k in (entry.get("processing_keys") or entry.get("executing") or [])],
    )


def normalize_dask_state(
    scheduler_info: Mapping[str, Any],
    future_status: Mapping[str, str],
    processing: Mapping[str, Sequence[str]],
    who_has: Mapping[str, Sequence[str]],
    registry: ShardRegistry,
    *,
    ts: float,
    seq: int,
    run_id: str,
    tile: str,
) -> OuterSnapshot:
    """One snapshot from plain distributed: ``scheduler_info`` plus per-future status.

    ``future_status`` maps key to ``pending``, ``finished``, ``error``, ``lost``,
    or ``cancelled`` (``distributed.Future.status``); ``processing`` maps a
    worker address to the keys it is running; ``who_has`` maps a key to the
    workers holding its result.
    """
    shards_by_key, stalled = _base_shards(registry)
    running_on = {key: worker for worker, keys in processing.items() for key in keys}
    finished = {k for k, s in future_status.items() if s == "finished"}
    for key, shard in shards_by_key.items():
        if shard.state in ("done", "failed"):
            continue
        _apply_dask_status(shard, future_status.get(key), running_on, who_has, finished)
        _apply_retry_and_stall(shard, stalled)
    workers = {
        str(address): _dask_worker(str(address), info, processing)
        for address, info in (scheduler_info.get("workers") or {}).items()
    }
    return OuterSnapshot(
        run_id=run_id,
        tile=tile,
        ts=ts,
        seq=seq,
        source="dask",
        shards=shards_by_key,
        workers=workers,
    )


def _apply_dask_status(
    shard: OuterShard,
    status: str | None,
    running_on: Mapping[str, str],
    who_has: Mapping[str, Sequence[str]],
    finished: set[str],
) -> None:
    if shard.key in running_on:
        shard.state = "running"
        shard.worker = running_on[shard.key]
    elif status == "finished":
        shard.state = "done"
        holders = list(who_has.get(shard.key, []))
        shard.worker = holders[0] if holders else None
    elif status in ("error", "lost", "cancelled"):
        shard.state = "failed"
        shard.error = shard.error or status
    elif status == "pending":
        shard.state = "blocked" if any(d not in finished for d in shard.depends_on) else "pending"


def _dask_worker(address: str, info: Any, processing: Mapping[str, Sequence[str]]) -> OuterWorker:
    if not isinstance(info, Mapping):
        return OuterWorker(
            address=address, status="alive", executing=list(processing.get(address, []))
        )
    metrics = info.get("metrics") or {}
    memory = metrics.get("memory")
    limit = info.get("memory_limit")
    return OuterWorker(
        address=address,
        status=str(info.get("status", "alive")),
        cpu_percent=_float(metrics.get("cpu")),
        memory_mb=None if memory is None else round(float(memory) / 1048576, 1),
        memory_limit_mb=None if limit is None else round(float(limit) / 1048576, 1),
        executing=list(processing.get(address, [])),
    )


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def diff_snapshots(prev: OuterSnapshot | None, cur: OuterSnapshot) -> list[dict[str, Any]]:
    """Transitions and worker joins and losses between two snapshots."""
    events: list[dict[str, Any]] = []
    before = prev.shards if prev is not None else {}
    for key, shard in cur.shards.items():
        old = before.get(key)
        if old is None or old.state != shard.state or old.worker != shard.worker:
            events.append(
                {
                    "ts": cur.ts,
                    "kind": "transition",
                    "key": key,
                    "stage": shard.stage,
                    "index": shard.index,
                    "from": old.state if old else None,
                    "to": shard.state,
                    "worker": shard.worker,
                    "error": shard.error,
                }
            )
    old_workers = set(prev.workers) if prev is not None else set()
    for address in sorted(set(cur.workers) - old_workers):
        events.append({"ts": cur.ts, "kind": "worker_joined", "address": address})
    lost = sorted(old_workers - set(cur.workers))
    for address in lost:
        events.append({"ts": cur.ts, "kind": "worker_lost", "address": address})
    cur.lost_workers = lost
    return events


# --- the observer ------------------------------------------------------------------------


class FuturesObserver:
    """Poll the scheduler, persist the join, verify Frisky, dump spans.

    ``client`` is the driver's executor; ``state_reader`` is what turns it
    into a normalized snapshot (a closure over the executor's kind), and
    ``span_query`` and ``story`` are Frisky's, injectable so a test needs no
    scheduler. All three are optional: a plain-dask run has no spans and no
    stories, and the verification file says so per check.
    """

    def __init__(
        self,
        *,
        run_id: str,
        tile: str,
        storage: StorageBackend,
        registry: ShardRegistry,
        state_reader: Callable[[ShardRegistry, float, int], OuterSnapshot],
        mode: str,
        dashboard_url: str | None,
        span_query: Callable[..., list[dict[str, Any]]] | None = None,
        story: Callable[[str], Mapping[str, Any] | None] | None = None,
        poll_s: float | None = None,
        clock: Callable[[], float] = time.time,
        tracing_capacity: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.tile = tile
        self.storage = storage
        self.registry = registry
        self.state_reader = state_reader
        self.mode = mode
        self.dashboard_url = dashboard_url
        self.span_query = span_query
        self.story = story
        self.poll_s = settings.futures_observer_poll_s if poll_s is None else poll_s
        self.clock = clock
        self.tracing_capacity = tracing_capacity or settings.frisky_tracing_capacity
        self._seq = 0
        self._span_seq = 0
        self._prev: OuterSnapshot | None = None
        self._last_span_ns: int | None = None
        self._seen_at_cursor: set[Any] = set()
        self._span_chunks: list[str] = []
        self._span_count = 0
        self._checks: list[dict[str, Any]] = []
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.errors: list[str] = []
        self.root = shards.shard_root(run_id, tile)
        self.timings = shards.unit_timing_prefix(run_id)

    # -- RunObserver protocol -------------------------------------------------------------

    def register(self, ref: ShardRef) -> None:
        self.registry.register(ref)

    def note_completion(
        self, key: str, *, result: Mapping[str, Any] | None, error: str | None
    ) -> None:
        self.registry.note_completion(key, result=result, error=error)

    def note_stalled(self, keys: Sequence[str]) -> None:
        self.registry.note_stalled(keys)

    def tick(self) -> None:
        """One poll, from the driver's loop; the thread does the same on a timer."""
        self._guard("poll", self.poll_once)

    def verify(self, phase: str) -> Mapping[str, Any]:
        return self._guard("verify", lambda: self.verify_frisky(phase=phase)) or {}

    def finalize(self) -> Mapping[str, Any]:
        self.stop()
        return self._guard("finalize", self._finalize) or {
            "passed": False,
            "failures": ["finalize failed"],
        }

    # -- lifecycle -----------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="lst-futures-observer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_s * 2)
            self._thread = None

    def _loop(self) -> None:
        while not self._stopping.wait(self.poll_s):
            self._guard("poll", self.poll_once)

    def _guard(self, what: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:  # observing never fails a run
            self.errors.append(f"{what}: {type(exc).__name__}: {exc}"[:300])
            log.warning("futures_observer_failed", what=what, error=str(exc))
            return None

    # -- polling ---------------------------------------------------------------------------

    def poll_once(self) -> OuterSnapshot:
        with self._lock:
            self._seq += 1
            seq = self._seq
            ts = self.clock()
            snapshot = self.state_reader(self.registry, ts, seq)
            events = diff_snapshots(self._prev, snapshot)
            self._prev = snapshot
        self.storage.write_text(
            f"{self.root}/state/orchestration.json",
            json.dumps(snapshot.to_dict(), indent=2, default=str),
        )
        if events:
            self.storage.write_text(
                f"{self.root}/state/orchestration.events.{seq:04d}.jsonl",
                "\n".join(json.dumps(e, default=str) for e in events) + "\n",
                content_type="application/x-ndjson",
            )
        self._guard("span_dump", self.dump_spans)
        self._guard("verify_running", lambda: self.verify_frisky(phase="running"))
        return snapshot

    def dump_spans(self) -> str | None:
        """Frisky spans since the last dump, deduplicated at the cursor."""
        if self.span_query is None or self.dashboard_url is None:
            return None
        spans = self.span_query(
            start_ns=self._last_span_ns,
            limit=settings.frisky_span_dump_limit,
            dashboard_url=self.dashboard_url,
        )
        since = self._last_span_ns
        fresh = [
            s
            for s in spans
            if (since is None or int(s.get("start_ns") or 0) >= since)
            and s.get("span_id") not in self._seen_at_cursor
        ]
        if not fresh:
            return None
        newest = max(int(s.get("start_ns") or 0) for s in fresh)
        self._last_span_ns = newest
        self._seen_at_cursor = {
            s.get("span_id") for s in fresh if int(s.get("start_ns") or 0) == newest
        }
        self._span_seq += 1
        self._span_count += len(fresh)
        key = f"{self.timings}frisky-spans.{self._span_seq:04d}.json"
        payload = {
            "seq": self._span_seq,
            "from_ns": min(int(s.get("start_ns") or 0) for s in fresh),
            "to_ns": newest,
            "count": len(fresh),
            "capacity": self.tracing_capacity,
            "spans": fresh,
        }
        self.storage.write_text(key, json.dumps(payload, default=str))
        self._span_chunks.append(key)
        return key

    # -- verification ------------------------------------------------------------------

    def _check(
        self,
        name: str,
        phase: str,
        passed: bool | None,
        expected: Any,
        observed: Any,
        *,
        detail: str = "",
    ) -> None:
        record = {
            "name": name,
            "phase": phase,
            "passed": passed,
            "expected": expected,
            "observed": observed,
            "detail": detail,
            "checked_at": datetime.now(tz=UTC).isoformat(),
        }
        with self._lock:
            self._checks = [
                c for c in self._checks if not (c["name"] == name and c["phase"] == phase)
            ]
            self._checks.append(record)

    def verify_frisky(self, *, phase: str) -> dict[str, Any]:
        """Whether the outer scheduler handled the futures, as evidence."""
        if self.mode != "frisky":
            self._check(
                "scheduler_is_frisky",
                phase,
                None,
                "frisky",
                self.mode,
                detail="plain-dask run: checks not applicable",
            )
            return self._verification()
        snapshot = self._prev or self._guard("poll", self.poll_once)
        shards_now = snapshot.shards if snapshot else {}
        if phase == "submitted":
            self._check_submitted(phase, shards_now)
        if phase == "running":
            self._check_placements(phase, shards_now)
        if self.span_query is not None and self.dashboard_url is not None:
            self._check_inner_spans(phase, shards_now)
            if phase == "complete":
                self._check_exec_spans(phase, shards_now)
        saturated = self._span_count >= 0.9 * self.tracing_capacity
        self._check(
            "span_buffer_not_saturated",
            phase,
            not saturated,
            f"< {int(0.9 * self.tracing_capacity)}",
            self._span_count,
        )
        if phase == "complete":
            self._check(
                "spans_persisted_after_cluster",
                phase,
                bool(self._span_chunks),
                ">= 1 chunk",
                len(self._span_chunks),
            )
        verification = self._verification()
        self.storage.write_text(
            f"{self.timings}frisky-verification.json",
            json.dumps(verification, indent=2, default=str),
        )
        return verification

    def _check_submitted(self, phase: str, shards_now: Mapping[str, OuterShard]) -> None:
        refs = self.registry.refs()
        self._check(
            "scheduler_state_lists_submitted_keys",
            phase,
            bool(refs) and all(k in shards_now for k in refs),
            len(refs),
            sum(1 for k in refs if k in shards_now),
        )

    def _check_placements(self, phase: str, shards_now: Mapping[str, OuterShard]) -> None:
        running = [k for k, s in shards_now.items() if s.state == "running"][:8]
        if not running or self.story is None:
            return
        placed = 0
        for key in running:
            story = self.story(key) or {}
            if any(e.get("event_type") == "placed" for e in story.get("events") or []):
                placed += 1
        self._check(
            "story_places_running_shard", phase, placed == len(running), len(running), placed
        )

    def _check_inner_spans(self, phase: str, shards_now: Mapping[str, OuterShard]) -> None:
        assert self.span_query is not None
        sample = [k for k, s in shards_now.items() if s.state in ("running", "done")][:8]
        if not sample:
            return
        found = 0
        for key in sample:
            spans = self.span_query(
                name="shard.", task=key, limit=10, dashboard_url=self.dashboard_url
            )
            if spans and any(key in (s.get("keys") or []) for s in spans):
                found += 1
        self._check(
            "inner_spans_keyed_by_outer_key", phase, found == len(sample), len(sample), found
        )

    def _check_exec_spans(self, phase: str, shards_now: Mapping[str, OuterShard]) -> None:
        assert self.span_query is not None
        done = [k for k, s in shards_now.items() if s.state == "done"][:16]
        if not done:
            return
        prior = self.registry.view()[4]
        ok = 0
        for key in done:
            spans = self.span_query(
                name="worker.exec.call", task=key, limit=50, dashboard_url=self.dashboard_url
            )
            if len(spans) >= 1 + prior.get(key, 0):
                ok += 1
        self._check(
            "one_exec_call_span_per_attempt",
            phase,
            ok == len(done),
            len(done),
            ok,
            detail="at least one per attempt; a lost worker's span may never arrive",
        )

    def _verification(self) -> dict[str, Any]:
        with self._lock:
            checks = list(self._checks)
        return {
            "run_id": self.run_id,
            "tile": self.tile,
            "mode": self.mode,
            "dashboard_url_present": self.dashboard_url is not None,
            "checks": checks,
            "passed": all(c["passed"] is not False for c in checks)
            and any(c["passed"] for c in checks),
            "failures": [c["name"] for c in checks if c["passed"] is False],
            "span_chunks": list(self._span_chunks),
            "errors": list(self.errors[-10:]),
        }

    # -- finalize --------------------------------------------------------------------------

    def _finalize(self) -> dict[str, Any]:
        snapshot = self._guard("poll", self.poll_once)
        self._guard("span_dump", self.dump_spans)
        verification = self.verify_frisky(phase="complete")
        overview_key = self._guard("overview", self._write_overview)
        final = {
            "run_id": self.run_id,
            "tile": self.tile,
            "mode": self.mode,
            "finished_at": datetime.now(tz=UTC).isoformat(),
            "snapshot": snapshot.to_dict() if snapshot else None,
            "verification": verification,
            "span_chunks": list(self._span_chunks),
            "overview_key": overview_key,
            "errors": list(self.errors),
        }
        self.storage.write_text(
            f"{self.root}/state/orchestration.final.json", json.dumps(final, indent=2, default=str)
        )
        gate = observability_gate(final, self.registry, self.storage)
        final["observability"] = gate
        return gate

    def _write_overview(self) -> str | None:
        """``frisky observe overview`` over every dumped chunk, when frisky is here."""
        if not self._span_chunks:
            return None
        import shutil  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        if shutil.which("frisky") is None:
            return None
        spans: list[dict[str, Any]] = []
        for key in self._span_chunks:
            raw = self.storage.read_text(key)
            if raw:
                spans.extend(json.loads(raw).get("spans", []))
        with tempfile.TemporaryDirectory(prefix="lst_frisky_overview_") as directory:
            merged = Path(directory) / "frisky-spans.all.json"
            merged.write_text(json.dumps({"spans": spans}))
            result = subprocess.run(
                ["frisky", "observe", "overview", str(merged), "--json"],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                self.errors.append(f"frisky overview: {result.stderr[-200:]}")
                return None
            key = f"{self.timings}frisky-overview.json"
            self.storage.write_text(key, result.stdout)
            return key


# --- the gate ------------------------------------------------------------------------------


def _shard_gate_failures(
    storage: StorageBackend, run_id: str, ref: ShardRef, name: str
) -> list[str]:
    """What one composite shard's persisted trace is missing, if anything."""
    stem = shards.unit_trace_prefix(run_id, "composite", ref.tile, ref.index)
    listing = storage.list_prefix(f"{stem}.inner")
    finals = [k for k in listing if k.endswith(".final.json")]
    if not finals:
        return [f"{name}: no inner final object"]
    out: list[str] = []
    raw = storage.read_text(sorted(finals)[-1])
    payload = json.loads(raw) if raw else {}
    if payload.get("error"):
        out.append(f"{name}: inner trace closed with error {payload['error'][:80]}")
    if payload.get("scheduler") == "frisky" and not payload.get("chunk_keys"):
        out.append(f"{name}: no span chunks")
    graphs = [k for k in listing if ".inner-graph." in k]
    execs = [k for k in listing if ".inner-exec." in k]
    if not graphs or len(graphs) != len(execs):
        out.append(f"{name}: graph files {len(graphs)} vs exec files {len(execs)}")
    state_raw = storage.read_text(ref.state_key)
    inner = (json.loads(state_raw).get("inner") or {}) if state_raw else {}
    if inner.get("sink_error"):
        out.append(f"{name}: frisky sink failed: {inner['sink_error'][:80]}")
    return out


def observability_gate(
    final: Mapping[str, Any], registry: ShardRegistry, storage: StorageBackend
) -> dict[str, Any]:
    """Whether the run left the visibility this ticket requires.

    Computation is never failed by telemetry, and a run without the required
    visibility never passes acceptance. The gate reads the bucket, not the
    scheduler: every composite shard's ``inner.final.json`` present with no
    error and a sink that built; every composite group's graph and exec files
    present and joinable; Frisky verification checks not failed; span chunks
    persisted; the final orchestration object written.
    """
    required: list[str] = []
    failures: list[str] = []
    refs = registry.refs()
    composite = [r for r in refs.values() if r.stage == "composite"]
    run_id = str(final.get("run_id"))
    for ref in composite:
        name = f"composite[{ref.index}]"
        required.extend((f"{name}.inner_final", f"{name}.graph_and_exec_per_group"))
        failures.extend(_shard_gate_failures(storage, run_id, ref, name))
    verification = final.get("verification") or {}
    required.append("frisky_verification")
    if verification.get("failures"):
        failures.append(f"frisky verification failed: {verification['failures']}")
    if final.get("mode") == "frisky":
        required.append("span_chunks_persisted")
        if not final.get("span_chunks"):
            failures.append("no frisky span chunks persisted")
    required.append("orchestration_final_written")
    return {
        "required": required,
        "passed": not failures and bool(composite),
        "failures": failures,
        "composite_shards": len(composite),
    }
