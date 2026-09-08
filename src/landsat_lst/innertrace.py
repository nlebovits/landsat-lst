"""The inner scheduler of one shard, and the trace that makes it visible.

A composite shard runs a dask graph of tens of thousands of tasks inside one
process. Until issue #155 that graph was opaque: the threaded scheduler fires
``dask.callbacks`` hooks, but nothing outside the process could see which task
was running, what it depended on, or how long the main thread spent building the
graph before the first task was dispatched. Weeks went to strace and native
stack captures to answer questions that a scheduler answers for free.

This module gives every shard an inner scheduler that reports, and persists
what it reports while the shard is still running:

- :func:`inner_scheduler` binds dask's ``scheduler`` for the shard. ``"frisky"``
  starts an in-process Frisky cluster (one worker, ``threads`` threads) whose
  every task is a span in this process, readable through
  :func:`frisky.get_spans` with no client, no socket, and no second process.
  ``"threads"`` is dask's threaded scheduler, the production scheduler before
  this change, kept for the pixel-identity comparison. The Batch path keeps
  ``"threads"``; only the futures wrapper asks for ``"frisky"``.
- :class:`InnerTrace` records coarse sections (``composite_graph``, ``encode``,
  one per longitude group, ``upload``), captures **the graph actually handed to
  the scheduler** for every bounded compute, joins the execution history back
  onto that graph with the scheduler's own evidence for every key, and flushes
  all of it to S3 in chunks so an interrupted shard keeps its evidence.

Two things about a Frisky client that were measured on 2026-09-08 decide the
shape here. A Frisky client nested inside a Frisky worker task, submitting to
the shared scheduler, executed its tasks and then closed the scheduler with an
early EOF on both transports, and ``frisky.dask.get`` refuses ``workers=``, so
an inner graph on the shared scheduler could neither be pinned to the VM that
holds the local GeoTIFF target nor survive. An in-process scheduler has neither
problem: the graph never leaves the VM, and its spans are already here.

Graph capture happens at the scheduler's ``get``. ``dask.compute`` hands the
scheduler an expression; the wrapper installed by :func:`inner_scheduler`
materializes it once, records every key and its dependencies from that dict,
and passes **the same dict** on. There is no second optimization pass, and the
capture cost is counted inside the group's construction wall, never subtracted.

"Not executed" is never reported as "ready but not started". A key without an
execution span gets its state from the scheduler (the in-process REST story
under Frisky, the ``start_state`` dict under threads) or is labelled
``unknown``. Readiness is a claim only scheduler evidence can make.

Instrumentation never fails a shard. Every public method logs and swallows.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager, nullcontext, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import psutil
import structlog
from dask.callbacks import Callback
from dask.utils import key_split

from landsat_lst.config import settings
from landsat_lst.exectrace import CLASSES, classify

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence

    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

InnerMode = Literal["threads", "frisky"]

#: Reasons a graph node can carry for not having executed. Only
#: ``ready_not_started`` claims readiness, and only scheduler evidence sets it.
NOT_EXECUTED_REASONS = (
    "blocked_on_deps",
    "released",
    "erred",
    "ready_not_started",
    "completed_without_exec_span",
    "unknown",
)

#: Frisky span names that describe the client-side construction of one graph.
#: Everything before the first of them, measured from ``dask.compute`` entry, is
#: dask's own optimize and materialize work, which Frisky never sees.
CONSTRUCTION_SPANS = (
    "client.convert_legacy",
    "client.dask_order",
    "client.serialize_tasks",
    "client.submit_graph",
)


# --- identity and the binding from the outer task ----------------------------------


@dataclass(frozen=True)
class ShardIdentity:
    """What every record of one shard attempt carries."""

    run_id: str
    stage: str
    tile: str
    index: int
    attempt: int
    trace_id: int
    outer_key: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "tile": self.tile,
            "index": self.index,
            "attempt": self.attempt,
            "trace_id": hex(self.trace_id),
            "outer_key": self.outer_key,
        }


class SpanSink(Protocol):
    """Where coarse shard spans go: the outer scheduler, when there is one."""

    def span(
        self,
        name: str,
        t0_ns: int,
        t1_ns: int,
        *,
        keys: list[str],
        count: int | None,
        trace_id: int,
    ) -> None: ...

    def event(self, kind: str, *, key: str | None, metadata: list[tuple[str, str]]) -> None: ...


class FriskySink:
    """Forward spans to the Frisky scheduler this process is a worker of.

    ``frisky.record_span`` inside a worker task reaches the scheduler keyed by
    whatever ``keys`` it is given (measured 2026-09-08), so an outer task key
    makes the shard's sections queryable beside the outer task itself. Frisky's
    clock is not the epoch clock; the offset is measured once here and applied
    on every emit so the two never drift apart silently.
    """

    def __init__(self) -> None:
        import frisky  # noqa: PLC0415

        self._frisky = frisky
        self.clock_offset_ns = int(frisky.now_ns()) - time.time_ns()

    def span(
        self,
        name: str,
        t0_ns: int,
        t1_ns: int,
        *,
        keys: list[str],
        count: int | None,
        trace_id: int,
    ) -> None:
        self._frisky.record_span(
            name,
            t0_ns + self.clock_offset_ns,
            t1_ns + self.clock_offset_ns,
            trace_id=trace_id,
            count=count,
            keys=keys,
        )

    def event(self, kind: str, *, key: str | None, metadata: list[tuple[str, str]]) -> None:
        self._frisky.record_event(kind, key=key, metadata=metadata)


@dataclass
class OuterBinding:
    """What the futures task wrapper tells the shard about its outer task."""

    outer_key: str | None
    trace_id: int | None = None
    sink: SpanSink | None = None
    inner_scheduler: InnerMode | None = None
    sink_error: str | None = None


_binding: ContextVar[OuterBinding | None] = ContextVar("lst_outer_binding", default=None)
_trace: ContextVar[InnerTrace | None] = ContextVar("lst_inner_trace", default=None)

#: Process-level fallback for the log processor. Dask's worker threads and the
#: heartbeat thread do not inherit context variables, and one shard runs per
#: process (Batch) or per worker thread (futures at ``nthreads=1``), so a module
#: global is correct exactly where the context variable is not visible.
CURRENT: ShardIdentity | None = None


@contextmanager
def outer_binding(
    *,
    outer_key: str | None,
    trace_id: int | None = None,
    sink: SpanSink | None = None,
    inner_scheduler: InnerMode | None = None,
    sink_error: str | None = None,
) -> Iterator[OuterBinding]:
    """Bind the outer task's identity for the shard that runs inside it."""
    binding = OuterBinding(
        outer_key=outer_key,
        trace_id=trace_id,
        sink=sink,
        inner_scheduler=inner_scheduler,
        sink_error=sink_error,
    )
    token = _binding.set(binding)
    try:
        yield binding
    finally:
        _binding.reset(token)


def current_binding() -> OuterBinding | None:
    return _binding.get()


def active_inner_trace() -> InnerTrace | None:
    return _trace.get()


def new_trace_id() -> int:
    """A fresh trace id: 63 random bits.

    Not ``frisky.new_trace_id()``: that is a per-process counter, so two
    shards on two workers both drew ``0x1`` (Demonstration 1, 2026-09-08) and
    their records could not be told apart by trace id.
    """
    return int.from_bytes(os.urandom(8), "big") >> 1


def add_shard_context(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Structlog processor: stamp the running shard's trace id on every line.

    A line already carrying ``trace_id`` (bound through contextvars on the
    shard's own thread) is left alone; lines from dask's worker threads and the
    heartbeat thread, which see no context variables, get the process-level
    identity.
    """
    identity = CURRENT
    if identity is not None and "trace_id" not in event_dict:
        event_dict["trace_id"] = hex(identity.trace_id)
        event_dict["shard"] = f"{identity.stage}.{identity.index:04d}"
        event_dict["attempt"] = identity.attempt
    return event_dict


# --- the inner scheduler --------------------------------------------------------------


@dataclass
class InnerScheduler:
    """The scheduler a shard's graphs run on, and how to reach it."""

    kind: InnerMode
    threads: int
    dashboard_url: str | None = None
    client: Any = None
    cluster: Any = None

    def close(self) -> None:
        for obj in (self.client, self.cluster):
            if obj is not None:
                with suppress(Exception):
                    obj.close()
        self.client = self.cluster = None


def _resolve_threads(threads: int | None) -> int:
    if threads is not None and threads > 0:
        return int(threads)
    if settings.dask_max_threads:
        return int(settings.dask_max_threads)
    return max(1, os.cpu_count() or 1)


def _vm_memory_bytes() -> int:
    return int(psutil.virtual_memory().total)


@contextmanager
def inner_scheduler(
    *,
    mode: InnerMode,
    threads: int | None = None,
    memory_limit: int | None = None,
) -> Iterator[InnerScheduler]:
    """Bind dask's scheduler for the shard that runs inside this block.

    ``"threads"`` is ``dask.config.set(scheduler="threads", num_workers=threads)``,
    exactly what :func:`landsat_lst.job._thread_cap` sets when a cap is
    configured, and what every production shard ran on before issue #155.

    ``"frisky"`` starts an in-process Frisky cluster. ``memory_limit`` defaults
    to the VM's total memory so Frisky's spill thresholds sit where the threaded
    scheduler had no limit at all; a shard that spills is a shard whose memory
    story changed, and the trace counts ``spill.*`` spans so the gate can say
    so. The dashboard binds a random loopback port: the REST story endpoint on
    it is the scheduler evidence behind every ``not_executed_reason``.

    The scheduler installed in dask's config is a callable, not the client, so
    graph capture sits between ``dask.compute`` and Frisky's ``get``.
    """
    import dask  # noqa: PLC0415

    count = _resolve_threads(threads)
    if mode == "threads":
        sched = InnerScheduler(kind="threads", threads=count)
        with dask.config.set(scheduler="threads", num_workers=count):
            yield sched
        return
    if mode != "frisky":
        msg = f"unknown inner scheduler {mode!r}; expected 'threads' or 'frisky'"
        raise ValueError(msg)
    try:
        import frisky  # noqa: PLC0415
        import frisky.dask as frisky_dask  # noqa: PLC0415
    except ImportError as exc:
        msg = "inner scheduler 'frisky' needs the frisky extra installed on the worker"
        raise RuntimeError(msg) from exc

    # One worker in a subprocess, on purpose. An in-process worker shares one
    # span buffer with this process, and ``frisky.get_spans`` drains it: when
    # this process is itself a Frisky worker (the futures path), its worker
    # plugin and this trace would each take the other's spans, and the outer
    # scheduler received nothing (measured 2026-09-08). A subprocess worker
    # keeps its spans in its own process, and the inner scheduler's REST
    # endpoint serves them without draining, incrementally by ``start_ns``.
    # It also puts graph construction (this process) and task execution (the
    # child) under different GILs.
    cluster = frisky.LocalCluster(
        n_workers=1,
        threads_per_worker=count,
        processes=True,
        transport="tcp",
        memory_limit=_vm_memory_bytes() if memory_limit is None else memory_limit,
        dashboard_address="127.0.0.1:0",
        silence_summary=True,
    )
    client = cluster.get_client()
    sched = InnerScheduler(
        kind="frisky",
        threads=count,
        dashboard_url=getattr(client, "dashboard_link", None),
        client=client,
        cluster=cluster,
    )

    def traced_get(dsk: Any, keys: Any, **_kwargs: Any) -> Any:
        return _traced_get(frisky_dask, client, dsk, keys)

    try:
        with dask.config.set(scheduler=traced_get):
            yield sched
    finally:
        sched.close()


def _traced_get(frisky_dask: Any, client: Any, dsk: Any, keys: Any) -> Any:
    """Materialize once, capture, hand the same dict to Frisky, timing each stage.

    The body mirrors :func:`frisky.dask.get` (translate, submit, gather) so the
    three stages can be stamped on this process's clock. Frisky records the
    same stages as ``client.*`` spans in this process's buffer, but that buffer
    is drained by whichever reader gets there first, and on the futures path
    the outer worker plugin is that reader. Local stamps do not depend on it.
    """
    from dask._task_spec import convert_legacy_graph  # noqa: PLC0415

    entry_ns = time.time_ns()
    graph = dsk.__dask_graph__() if hasattr(dsk, "__dask_graph__") else dsk
    graph = convert_legacy_graph(graph)
    materialized_ns = time.time_ns()
    trace = _trace.get()
    if trace is not None:
        trace.capture_graph(
            graph,
            keys,
            entry_ns=entry_ns,
            materialize_ns=materialized_ns - entry_ns,
        )
    keys_list, singleton = (keys, False) if isinstance(keys, list) else ([keys], True)
    trace_id = frisky_dask.new_trace_id()
    translate_t0 = time.time_ns()
    task_specs = frisky_dask.translate_graph(graph, keys_list, trace_id=trace_id)
    output_keys = [_key_str(k) for k in keys_list]
    unique_keys = list(dict.fromkeys(output_keys))
    submit_t0 = time.time_ns()
    unique_futures = client.submit_graph(task_specs, unique_keys, trace_id=trace_id)
    submitted_ns = time.time_ns()
    if trace is not None:
        trace.note_submission(
            translate_ns=submit_t0 - translate_t0,
            submit_ns=submitted_ns - submit_t0,
            submitted_ns=submitted_ns,
            n_specs=len(task_specs),
        )
    by_key = dict(zip(unique_keys, unique_futures, strict=True))
    futures = [by_key[k] for k in output_keys]
    try:
        results = client.gather(futures)
    except BaseException:
        del futures, unique_futures, by_key
        raise
    return results[0] if singleton else results


# --- records -----------------------------------------------------------------------


@dataclass
class Section:
    """One coarse interval of the shard: a phase, or one longitude group."""

    id: int
    name: str
    t0_ns: int
    t1_ns: int | None = None
    group: tuple[int, int] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # Per-group construction boundaries, epoch ns. ``None`` until observed.
    compute_entry_ns: int | None = None
    get_entry_ns: int | None = None
    graph_captured_ns: int | None = None
    first_dispatch_ns: int | None = None
    compute_end_ns: int | None = None
    materialize_s: float = 0.0
    capture_overhead_s: float = 0.0
    tasks_total: int | None = None
    tasks_started_at_open: int = 0
    span_cursor_at_open: int = 0

    def as_dict(self) -> dict[str, Any]:
        s = None if self.t1_ns is None else (self.t1_ns - self.t0_ns) / 1e9
        return {
            "id": self.id,
            "name": self.name,
            "group": list(self.group) if self.group else None,
            "t0_ns": self.t0_ns,
            "t1_ns": self.t1_ns,
            "s": None if s is None else round(s, 3),
            "error": self.error,
            "meta": self.meta,
            "compute_entry_ns": self.compute_entry_ns,
            "get_entry_ns": self.get_entry_ns,
            "graph_captured_ns": self.graph_captured_ns,
            "first_dispatch_ns": self.first_dispatch_ns,
            "compute_end_ns": self.compute_end_ns,
            "materialize_s": round(self.materialize_s, 4),
            "capture_overhead_s": round(self.capture_overhead_s, 4),
            "tasks_total": self.tasks_total,
        }


def _key_str(key: Any) -> str:
    """Frisky's key spelling: ``str(k)`` unless already a string."""
    return key if isinstance(key, str) else str(key)


def _prefix_of(key: Any) -> str:
    raw = key[0] if isinstance(key, tuple) and key else key
    with suppress(Exception):
        return key_split(raw)
    return "other"


def graph_nodes(graph: Mapping[Any, Any]) -> list[dict[str, Any]]:
    """Every node of a materialized graph with its class and dependency edges.

    ``graph`` is a dict of ``dask._task_spec.GraphNode`` (the shape both
    schedulers run); a legacy tuple graph falls back to
    :func:`dask.core.get_dependencies`.
    """
    from dask.core import get_dependencies  # noqa: PLC0415

    dependents: dict[str, int] = {}
    nodes: list[dict[str, Any]] = []
    deps_of: dict[str, list[str]] = {}
    for key, node in graph.items():
        deps = getattr(node, "dependencies", None)
        if deps is None:
            deps = get_dependencies(graph, key)
        names = sorted(_key_str(d) for d in deps)
        name = _key_str(key)
        deps_of[name] = names
        for dep in names:
            dependents[dep] = dependents.get(dep, 0) + 1
    for key in graph:
        name = _key_str(key)
        prefix = _prefix_of(key)
        nodes.append(
            {
                "key": name,
                "prefix": prefix,
                "class": classify(prefix),
                "deps": deps_of[name],
                "n_dependents": dependents.get(name, 0),
            }
        )
    return nodes


def _union_seconds(intervals: Sequence[tuple[int, int]]) -> float:
    """Seconds covered by the union of ``[t0, t1)`` intervals, never their sum."""
    total = 0
    current: tuple[int, int] | None = None
    for t0, t1 in sorted(intervals):
        if current is None or t0 > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = (t0, t1)
        elif t1 > current[1]:
            current = (current[0], t1)
    if current is not None:
        total += current[1] - current[0]
    return total / 1e9


# --- the trace -----------------------------------------------------------------------


class _RestEvidence:
    """Per-key scheduler evidence from the in-process Frisky REST endpoint."""

    def __init__(self, dashboard_url: str | None, *, timeout_s: float = 5.0) -> None:
        self.base = dashboard_url.rstrip("/") if dashboard_url else None
        self.timeout_s = timeout_s

    def story(self, key: str) -> list[dict[str, Any]] | None:
        if self.base is None:
            return None
        url = f"{self.base}/api/story/{urllib.parse.quote(key, safe='')}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_s) as response:
                payload = json.loads(response.read())
        except Exception:
            return None
        events = payload.get("events")
        return list(events) if isinstance(events, list) else None

    def workers(self) -> list[dict[str, Any]]:
        if self.base is None:
            return []
        try:
            with urllib.request.urlopen(f"{self.base}/api/workers", timeout=self.timeout_s) as r:
                payload = json.loads(r.read())
        except Exception:
            return []
        workers = payload.get("workers")
        return list(workers) if isinstance(workers, list) else []


def _story_summary(events: Sequence[Mapping[str, Any]]) -> tuple[list[str], bool, bool]:
    """States visited, whether a placement happened, whether the task completed."""
    states: list[str] = []
    placed = False
    completed = False
    for event in events:
        kind = event.get("event_type") or event.get("kind")
        if kind == "placed":
            placed = True
        elif kind == "task_completed":
            completed = True
        elif kind == "transition":
            states.append(str((event.get("details") or {}).get("to_state", "")))
    return states, placed, completed or "Memory" in states


def reason_from_story(events: Sequence[Mapping[str, Any]] | None, *, executed: bool) -> str:
    """Turn a scheduler story into a ``not_executed_reason``.

    Only a story that shows the key reaching the ready queue or a placement,
    with no execution, yields ``ready_not_started``. A release with no
    placement is ``released`` (cached, cancelled, or unnecessary: the scheduler
    does not say which and neither does this). A story that reached ``Memory``
    or ``task_completed`` is ``completed_without_exec_span``: the scheduler
    says it finished and only the span is missing (an alias, a data node, or a
    span the worker had not flushed yet). No story at all is ``unknown``.
    """
    if executed:
        return ""
    if not events:
        return "unknown"
    states, placed, completed = _story_summary(events)
    rules = (
        ("erred", "Erred" in states),
        ("completed_without_exec_span", completed),
        ("ready_not_started", placed or bool({"Processing", "Ready", "Queued"} & set(states))),
        ("released", bool(states) and states[-1] == "Released"),
        ("blocked_on_deps", "Waiting" in states),
    )
    return next((reason for reason, hit in rules if hit), "unknown")


class InnerTrace:
    """Record, persist, and forward what the inner scheduler does.

    Artifacts, under ``stem = shards.unit_trace_prefix(...)``:

    - ``{stem}.inner-spans.a{attempt:02d}.{seq:04d}.json.gz``: raw spans since
      the previous flush, each stamped with the identity and section.
    - ``{stem}.inner-graph.a{attempt:02d}.g{group:02d}.json.gz``: the graph the
      scheduler received for that group, written before its first dispatch.
    - ``{stem}.inner-exec.a{attempt:02d}.g{group:02d}.json.gz``: per key, when
      it ran and, when it did not, the scheduler's reason.
    - ``{stem}.inner-progress.json``: overwritten every flush; the live view.
    - ``{stem}.inner.a{attempt:02d}.final.json``: written on exit either way.
    """

    def __init__(
        self,
        *,
        identity: ShardIdentity,
        storage: StorageBackend,
        stem: str,
        scheduler: InnerScheduler,
        sink: SpanSink | None = None,
        clock: Callable[[], int] = time.time_ns,
        flush_s: float | None = None,
        flush_spans: int | None = None,
        story_limit: int | None = None,
    ) -> None:
        self.identity = identity
        self.storage = storage
        self.stem = stem
        self.scheduler = scheduler
        self.sink = sink
        self.clock = clock
        self.flush_s = settings.inner_trace_flush_s if flush_s is None else flush_s
        self.flush_spans = settings.inner_trace_flush_spans if flush_spans is None else flush_spans
        self.story_limit = settings.inner_trace_story_limit if story_limit is None else story_limit
        self._lock = threading.Lock()
        self._sections: list[Section] = []
        self._open: dict[int, Section] = {}
        self._group_section: Section | None = None
        self._seq = 0
        self._span_cursor = 0
        self._span_since_ns: int | None = None
        self._span_seen_at_cursor: set[Any] = set()
        self._pending_spans: list[dict[str, Any]] = []
        self._tasks_started = 0
        self._tasks_done = 0
        self._active: dict[str, int] = dict.fromkeys(CLASSES, 0)
        self._graph_keys: dict[str, dict[str, Any]] = {}
        self._graph_files: list[str] = []
        self._exec_files: list[str] = []
        self._chunk_keys: list[str] = []
        self._spill_count = 0
        self._mutex_wait_ns = 0
        self._gil_ns = 0
        self._exec_by_key: dict[str, dict[str, Any]] = {}
        self._callback: Callback | None = None
        self._callback_state: dict[str, Any] | None = None
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._token: Any = None
        self._started_ns = self.clock()
        self._errors: list[str] = []
        self._writers: list[threading.Thread] = []
        self._rest = _RestEvidence(scheduler.dashboard_url)
        self._host = psutil.Process()
        self._net_last: tuple[int, int] | None = None

    # -- helpers ----------------------------------------------------------------

    def _guard(self, what: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:  # instrumentation never fails a shard
            self._errors.append(f"{what}: {type(exc).__name__}: {exc}")
            log.warning("inner_trace_failed", what=what, error=str(exc))
            return None

    def _key_prefix(self) -> str:
        return f"{self.stem}.inner"

    def _write_json_gz(self, key: str, payload: Any, *, background: bool) -> None:
        def write() -> None:
            with tempfile.TemporaryDirectory(prefix="lst_inner_trace_") as directory:
                path = Path(directory) / "payload.json.gz"
                with gzip.open(path, "wt", encoding="utf-8") as stream:
                    json.dump(payload, stream, allow_nan=False)
                self.storage.upload(path, key)

        if background:
            thread = threading.Thread(
                target=lambda: self._guard(f"write {key}", write),
                name="lst-inner-write",
                daemon=True,
            )
            thread.start()
            self._writers.append(thread)
        else:
            self._guard(f"write {key}", write)

    # -- sections -----------------------------------------------------------------

    def open(self, name: str, *, group: tuple[int, int] | None = None, **meta: Any) -> int:
        with self._lock:
            section = Section(
                id=len(self._sections),
                name=name,
                t0_ns=self.clock(),
                group=group,
                meta=dict(meta),
                tasks_started_at_open=self._tasks_started,
                span_cursor_at_open=self._span_cursor,
            )
            self._sections.append(section)
            self._open[section.id] = section
            if group is not None:
                self._group_section = section
            return section.id

    def close(self, section_id: int, *, error: str | None = None, **meta: Any) -> None:
        with self._lock:
            section = self._open.pop(section_id, None)
            if section is None:
                return
            section.t1_ns = self.clock()
            section.error = error
            section.meta.update(meta)
            if self._group_section is section:
                self._group_section = None
            count = self._tasks_started - section.tasks_started_at_open
        self._forward(section, count)
        self._guard("flush", lambda: self.flush(reason=f"section:{section.name}"))

    @contextmanager
    def section(
        self, name: str, *, group: tuple[int, int] | None = None, **meta: Any
    ) -> Iterator[int]:
        section_id = self._guard("open", lambda: self.open(name, group=group, **meta))
        try:
            yield section_id if section_id is not None else -1
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if section_id is not None:
                self._guard("close", lambda: self.close(section_id, error=error))
            raise
        else:
            if section_id is not None:
                self._guard("close", lambda: self.close(section_id))

    def _forward(self, section: Section, count: int) -> None:
        sink, t1_ns = self.sink, section.t1_ns
        if sink is None or t1_ns is None:
            return
        keys = [self.identity.outer_key] if self.identity.outer_key else []

        def emit() -> None:
            sink.span(
                f"shard.{section.name}",
                section.t0_ns,
                t1_ns,
                keys=keys,
                count=count or None,
                trace_id=self.identity.trace_id,
            )

        self._guard("sink.span", emit)

    # -- the observer cog calls -----------------------------------------------------

    def group_observer(self) -> Callable[..., None]:
        """The callback ``cog.write_intermediates_bounded`` accepts."""

        def observe(event: str, index: int, count: int, **fields: Any) -> None:
            self._guard(f"observer:{event}", lambda: self._observe(event, index, count, fields))

        return observe

    def _observe(self, event: str, index: int, count: int, fields: dict[str, Any]) -> None:
        if event == "header_write_start":
            self._header_section = self.open("header_write", product=fields.get("product"))
        elif event == "header_write_end":
            section_id = getattr(self, "_header_section", None)
            if section_id is not None:
                self.close(section_id)
        elif event == "group_start":
            self.open(
                "group",
                group=(index, count),
                lon_start=fields.get("lon_start"),
                lon_stop=fields.get("lon_stop"),
            )
        elif event == "store_built":
            with self._lock:
                if self._group_section is not None:
                    self._group_section.meta["store_built_ns"] = self.clock()
        elif event == "compute_start":
            with self._lock:
                if self._group_section is not None:
                    self._group_section.compute_entry_ns = self.clock()
                    self._group_section.meta["n_stores"] = fields.get("n_stores")
        elif event == "compute_end":
            self._finish_group(fields.get("error"))

    def _finish_group(self, error: str | None) -> None:
        with self._lock:
            section = self._group_section
            if section is None:
                return
            section.compute_end_ns = self.clock()
        self._guard("exec_file", lambda: self._write_exec_file(section))
        self.close(section.id, error=error)

    # -- graph capture (called from the scheduler get, or the Callback) -------------

    def capture_graph(
        self,
        graph: Mapping[Any, Any],
        keys: Any,
        *,
        entry_ns: int,
        materialize_ns: int,
    ) -> None:
        """Record the graph the scheduler is about to run. Never raises."""
        self._guard(
            "capture_graph", lambda: self._capture_graph(graph, keys, entry_ns, materialize_ns)
        )

    def _capture_graph(
        self, graph: Mapping[Any, Any], keys: Any, entry_ns: int, materialize_ns: int
    ) -> None:
        t0 = self.clock()
        nodes = graph_nodes(graph)
        outputs = [_key_str(k) for k in (keys if isinstance(keys, list | tuple) else [keys])]
        roots = [n["key"] for n in nodes if not n["deps"]]
        captured_ns = self.clock()
        with self._lock:
            section = self._group_section
            if section is not None:
                section.get_entry_ns = entry_ns
                section.materialize_s = materialize_ns / 1e9
                section.graph_captured_ns = captured_ns
                section.capture_overhead_s += (captured_ns - t0) / 1e9
                section.tasks_total = len(nodes)
                group = section.group
            else:
                group = None
            self._graph_keys = {n["key"]: n for n in nodes}
        payload = {
            "v": 1,
            **self.identity.as_dict(),
            "group": list(group) if group else None,
            "captured_at_ns": captured_ns,
            "capture_s": round((captured_ns - t0) / 1e9, 4),
            "materialize_s": round(materialize_ns / 1e9, 4),
            "n_tasks": len(nodes),
            "n_edges": sum(len(n["deps"]) for n in nodes),
            "outputs": outputs,
            "roots": roots,
            "nodes": nodes,
        }
        index = group[0] if group else len(self._graph_files)
        key = f"{self._key_prefix()}-graph.a{self.identity.attempt:02d}.g{index:02d}.json.gz"
        self._graph_files.append(key)
        serialize_t0 = self.clock()
        # Serialized here, uploaded on a helper thread: the cost is counted, the
        # dispatch is not delayed by the upload.
        self._write_json_gz(key, payload, background=True)
        with self._lock:
            if section is not None:
                section.capture_overhead_s += (self.clock() - serialize_t0) / 1e9

    # -- spans and counters (Mode I) -----------------------------------------------

    def _new_spans(self) -> list[dict[str, Any]]:
        """Spans since the last read, from the inner scheduler's REST endpoint.

        Non-draining and incremental by ``start_ns`` (measured 2026-09-08), so a
        read can be repeated and a flush cannot lose what a settle read first.
        Spans sharing the cursor's ``start_ns`` are deduplicated by ``span_id``.
        """
        if self.scheduler.kind != "frisky" or self.scheduler.dashboard_url is None:
            return []
        import frisky  # noqa: PLC0415

        with self._lock:
            since = self._span_since_ns
            seen = set(self._span_seen_at_cursor)
        spans = frisky.query_spans(
            start_ns=since,
            limit=settings.inner_trace_query_limit,
            dashboard_url=self.scheduler.dashboard_url,
            request_timeout=10,
        )
        fresh = [s for s in spans if s.get("span_id") not in seen]
        if fresh:
            newest = max(int(s.get("start_ns") or 0) for s in fresh)
            with self._lock:
                self._span_since_ns = newest
                self._span_seen_at_cursor = {
                    s.get("span_id") for s in fresh if int(s.get("start_ns") or 0) == newest
                }
                self._span_cursor += len(fresh)
                # Held until the next flush writes them to a chunk. A settle
                # read before an exec file must not cost the chunk its spans.
                self._pending_spans.extend(fresh)
        self._ingest_spans(fresh)
        return fresh

    def note_submission(
        self, *, translate_ns: int, submit_ns: int, submitted_ns: int, n_specs: int
    ) -> None:
        """Stamp the translate and submit stages on the open group. Never raises."""
        with self._lock:
            section = self._group_section
            if section is None:
                return
            section.meta["translate_s"] = round(translate_ns / 1e9, 6)
            section.meta["submit_s"] = round(submit_ns / 1e9, 6)
            section.meta["submitted_ns"] = submitted_ns
            section.meta["n_specs"] = n_specs

    def _ingest_spans(self, spans: Sequence[Mapping[str, Any]]) -> None:
        with self._lock:
            section = self._group_section
            for span in spans:
                name = span.get("name", "")
                keys = span.get("keys") or []
                key = _key_str(keys[0]) if keys else None
                if name == "worker.exec.call" and key is not None:
                    self._tasks_started += 1
                    self._tasks_done += 1
                    record = self._exec_by_key.setdefault(key, {"key": key})
                    record["start_ns"] = span.get("start_ns")
                    record["end_ns"] = span.get("end_ns")
                    record["worker"] = span.get("worker")
                    if section is not None and section.first_dispatch_ns is None:
                        section.first_dispatch_ns = span.get("start_ns")
                elif name == "worker.exec.gil" and key is not None:
                    self._exec_by_key.setdefault(key, {"key": key})["gil_ns"] = span.get(
                        "duration_ns"
                    )
                    self._gil_ns += int(span.get("duration_ns") or 0)
                elif name == "worker.exec.deserialize" and key is not None:
                    self._exec_by_key.setdefault(key, {"key": key})["deserialize_ns"] = span.get(
                        "duration_ns"
                    )
                elif name == "worker.receive_task" and key is not None:
                    self._exec_by_key.setdefault(key, {"key": key})["received_ns"] = span.get(
                        "start_ns"
                    )
                elif name.startswith(("spill.", "unspill.", "worker.spill", "worker.unspill")):
                    self._spill_count += 1
                elif name.endswith(".mutex_wait"):
                    self._mutex_wait_ns += int(span.get("duration_ns") or 0)

    def _active_by_class(self) -> dict[str, int]:
        """Tasks in flight per class, from the REST worker view under Frisky.

        In-process spans only exist once a task has finished, so "active" comes
        from the scheduler's processing count and the class of the keys that
        have started but not completed, which the Callback path tracks directly.
        """
        if self.scheduler.kind == "threads":
            with self._lock:
                return dict(self._active)
        active = dict.fromkeys(CLASSES, 0)
        for worker in self._rest.workers():
            keys = worker.get("processing_keys") or worker.get("executing") or []
            if isinstance(keys, list):
                for key in keys:
                    active[classify(_prefix_of(key))] += 1
            elif isinstance(worker.get("processing"), int):
                active["other"] += int(worker["processing"])
        return active

    # -- the Callback (Mode T) ---------------------------------------------------------

    def observe_computes(self) -> Any:
        """Under ``threads``, install the Callback that supplies the same records."""
        if self.scheduler.kind != "threads":
            return nullcontext()
        trace = self

        class _Recorder(Callback):
            def _start(self, dsk: Any) -> None:
                trace._guard(
                    "callback.start",
                    lambda: trace._callback_start(dsk),
                )

            def _start_state(self, _dsk: Any, state: Any) -> None:
                trace._callback_state = state
                with trace._lock:
                    section = trace._group_section
                    if section is not None:
                        section.tasks_total = (
                            len(state.get("dependencies", {})) or section.tasks_total
                        )

            def _pretask(self, key: Any, _dsk: Any, _state: Any) -> None:
                name = _key_str(key)
                now = trace.clock()
                with trace._lock:
                    trace._tasks_started += 1
                    trace._active[classify(_prefix_of(key))] += 1
                    record = trace._exec_by_key.setdefault(name, {"key": name})
                    record["start_ns"] = now
                    record["thread"] = threading.get_ident()
                    section = trace._group_section
                    if section is not None and section.first_dispatch_ns is None:
                        section.first_dispatch_ns = now

            def _posttask(self, key: Any, _result: Any, _dsk: Any, _state: Any, _id: Any) -> None:
                name = _key_str(key)
                now = trace.clock()
                with trace._lock:
                    trace._tasks_done += 1
                    trace._active[classify(_prefix_of(key))] = max(
                        0, trace._active[classify(_prefix_of(key))] - 1
                    )
                    trace._exec_by_key.setdefault(name, {"key": name})["end_ns"] = now

            def _finish(self, _dsk: Any, state: Any, _errored: Any) -> None:
                trace._callback_state = state
                with trace._lock:
                    trace._active = dict.fromkeys(CLASSES, 0)

        return _Recorder()

    def _callback_start(self, dsk: Any) -> None:
        entry = self.clock()
        graph = dsk.__dask_graph__() if hasattr(dsk, "__dask_graph__") else dsk
        with self._lock:
            section = self._group_section
        keys: list[Any] = []
        if section is not None and section.compute_entry_ns is not None:
            pass
        self.capture_graph(graph, keys, entry_ns=entry, materialize_ns=0)

    # -- exec file per group ---------------------------------------------------------

    def _settle_spans(self, expected: set[str], *, timeout_s: float = 3.0) -> None:
        """Wait, bounded, for the worker threads' spans to reach the buffer.

        ``gather`` returns when the outputs are in memory; the spans of the
        tasks that produced them are appended by the worker threads and can
        land a few milliseconds later. Reading too early filed 40 executed
        keys of one group as not executed and credited their spans to the
        next group (measured 2026-09-08). Stop as soon as every expected key
        has an execution record, or when the wait expires.
        """
        if self.scheduler.kind != "frisky":
            return
        deadline = time.monotonic() + timeout_s
        while True:
            self._new_spans()
            with self._lock:
                missing = [k for k in expected if "start_ns" not in self._exec_by_key.get(k, {})]
            if not missing or time.monotonic() >= deadline:
                return
            time.sleep(0.02)

    def _fetch_stories(
        self, graph: Mapping[str, dict[str, Any]], not_executed: Sequence[str]
    ) -> dict[str, list[dict[str, Any]] | None]:
        """Scheduler stories for keys without a span, outputs first, bounded."""
        if self.scheduler.kind != "frisky":
            return {}
        priority = [k for k in not_executed if graph[k]["n_dependents"] == 0] + [
            k for k in not_executed if graph[k]["n_dependents"] != 0
        ]
        return {key: self._rest.story(key) for key in priority[: self.story_limit]}

    def _key_state(
        self,
        key: str,
        record: Mapping[str, Any],
        state: Mapping[str, Any] | None,
        stories: Mapping[str, list[dict[str, Any]] | None],
    ) -> tuple[str, str | None]:
        """``(not_executed_reason, scheduler_state)`` for one key, from evidence only."""
        if "start_ns" in record:
            return "", "Memory"
        if self.scheduler.kind == "threads" and state is not None:
            return _reason_from_local_state(state, key)
        if key in stories:
            return reason_from_story(stories[key], executed=False), _last_state(stories[key])
        return "unknown", None

    @staticmethod
    def _exec_row(
        key: str, node: Mapping[str, Any], record: Mapping[str, Any], reason: str, state: str | None
    ) -> dict[str, Any]:
        start, end = record.get("start_ns"), record.get("end_ns")
        return {
            "key": key,
            "class": node["class"],
            "start_ns": start,
            "end_ns": end,
            "duration_ns": None if start is None or end is None else end - start,
            "received_ns": record.get("received_ns"),
            "thread": record.get("thread"),
            "worker": record.get("worker"),
            "gil_ns": record.get("gil_ns"),
            "deserialize_ns": record.get("deserialize_ns"),
            "state": state,
            "not_executed_reason": reason,
        }

    def _write_exec_file(self, section: Section) -> None:
        with self._lock:
            expected = set(self._graph_keys)
        self._settle_spans(expected)
        with self._lock:
            graph = dict(self._graph_keys)
            execs = {k: dict(v) for k, v in self._exec_by_key.items() if k in graph}
            state = self._callback_state
            section.meta["construction"] = self._construction(section)
        not_executed = [k for k in graph if "start_ns" not in execs.get(k, {})]
        stories = self._fetch_stories(graph, not_executed)
        rows = []
        for key, node in graph.items():
            record = execs.get(key, {})
            reason, final_state = self._key_state(key, record, state, stories)
            rows.append(self._exec_row(key, node, record, reason, final_state))
        index = section.group[0] if section.group else 0
        key = f"{self._key_prefix()}-exec.a{self.identity.attempt:02d}.g{index:02d}.json.gz"
        payload = {
            "v": 1,
            **self.identity.as_dict(),
            "group": list(section.group) if section.group else None,
            "scheduler": self.scheduler.kind,
            "construction": section.meta["construction"],
            "n_executed": len(graph) - len(not_executed),
            "n_not_executed": len(not_executed),
            "stories_queried": len(stories),
            "story_limit": self.story_limit,
            "rows": rows,
        }
        self._exec_files.append(key)
        self._write_json_gz(key, payload, background=False)
        with self._lock:
            self._graph_keys = {}
            self._exec_by_key = {k: v for k, v in self._exec_by_key.items() if k not in graph}

    def _construction(self, section: Section) -> dict[str, Any]:
        """The delay from ``dask.compute`` entry to first dispatch, decomposed.

        Computed once, while the scheduler's span buffer is still alive, and
        cached on the section; a later call returns the cached record.

        ``pre_scheduler`` is what Frisky never sees: dask's own compute entry,
        expression optimize, and the whole-band materialization (#154).
        ``capture_overhead`` is this module's own cost and is inside the total,
        not outside it. Under ``threads`` the same total holds and the
        sub-intervals that Frisky reports are absent.
        """
        cached = section.meta.get("construction")
        if cached is not None:
            return cached
        entry = section.compute_entry_ns
        first = section.first_dispatch_ns
        out: dict[str, Any] = {
            "compute_entry_ns": entry,
            "get_entry_ns": section.get_entry_ns,
            "graph_captured_ns": section.graph_captured_ns,
            "first_dispatch_ns": first,
            "construction_s": None
            if entry is None or first is None
            else round((first - entry) / 1e9, 4),
            "pre_scheduler_s": (
                None
                if entry is None or section.get_entry_ns is None
                else round((section.get_entry_ns - entry) / 1e9, 4)
            ),
            "materialize_s": round(section.materialize_s, 4),
            "capture_overhead_s": round(section.capture_overhead_s, 4),
            "store_build_s": (
                None
                if section.meta.get("store_built_ns") is None
                else round((section.meta["store_built_ns"] - section.t0_ns) / 1e9, 4)
            ),
        }
        if self.scheduler.kind == "frisky":
            out["translate_s"] = section.meta.get("translate_s")
            out["submit_s"] = section.meta.get("submit_s")
            out["n_specs"] = section.meta.get("n_specs")
            submitted = section.meta.get("submitted_ns")
            # Frisky starts dispatching while the client is still submitting,
            # so this can be negative. It is reported, never clamped: a negative
            # value is the evidence that submission and execution overlap.
            out["submitted_to_first_dispatch_s"] = (
                None if submitted is None or first is None else round((first - submitted) / 1e9, 6)
            )
        if section.graph_captured_ns is not None and first is not None:
            out["captured_to_first_dispatch_s"] = round(
                (first - section.graph_captured_ns) / 1e9, 6
            )
        return out

    # -- flush, progress, final ---------------------------------------------------------

    def _host_rates(self) -> dict[str, Any]:
        now = time.time_ns()
        try:
            net = psutil.net_io_counters()
            rss = self._host.memory_info().rss
            children = sum(c.memory_info().rss for c in self._host.children(recursive=True))
        except Exception:
            return {}
        # The inner worker is a child process, so the shard's memory is the
        # tree's, not this process's alone.
        rates: dict[str, Any] = {
            "rss_mb": round(rss / 1048576, 1),
            "worker_rss_mb": round(children / 1048576, 1),
            "tree_rss_mb": round((rss + children) / 1048576, 1),
        }
        if self._net_last is not None:
            last_ns, last_recv = self._net_last
            dt = (now - last_ns) / 1e9
            if dt > 0:
                rates["net_recv_mb_s"] = round((net.bytes_recv - last_recv) / 1048576 / dt, 3)
                rates["window_s"] = round(dt, 1)
        self._net_last = (now, net.bytes_recv)
        return rates

    def flush(self, *, reason: str = "periodic") -> None:
        """Persist spans since the last flush and rewrite the progress object."""
        self._new_spans()
        with self._lock:
            fresh = self._pending_spans
            self._pending_spans = []
            self._seq += 1
            seq = self._seq
            section = self._group_section
            section_view = None
            if section is not None:
                section_view = {
                    "name": section.name,
                    "group": list(section.group) if section.group else None,
                    "open_s": round((self.clock() - section.t0_ns) / 1e9, 1),
                    "tasks_started_this_group": self._tasks_started - section.tasks_started_at_open,
                    "compute_entered": section.compute_entry_ns is not None,
                    "graph_captured": section.graph_captured_ns is not None,
                    "first_dispatched": section.first_dispatch_ns is not None,
                }
            counters = {
                "tasks_started": self._tasks_started,
                "tasks_done": self._tasks_done,
                "tasks_total": section.tasks_total if section else None,
                "spill_count": self._spill_count,
                "mutex_wait_s": round(self._mutex_wait_ns / 1e9, 3),
                "gil_s": round(self._gil_ns / 1e9, 3),
            }
        if fresh:
            key = f"{self._key_prefix()}-spans.a{self.identity.attempt:02d}.{seq:04d}.json.gz"
            payload = {
                "v": 1,
                **self.identity.as_dict(),
                "seq": seq,
                "reason": reason,
                "section": section_view,
                "count": len(fresh),
                "spans": list(fresh),
            }
            self._chunk_keys.append(key)
            self._write_json_gz(key, payload, background=False)
        progress = {
            "v": 1,
            **self.identity.as_dict(),
            "scheduler": self.scheduler.kind,
            "seq": seq,
            "reason": reason,
            "elapsed_s": round((self.clock() - self._started_ns) / 1e9, 1),
            "section": section_view,
            "active_by_class": self._active_by_class(),
            **counters,
            "host_rates": self._host_rates(),
            "chunks": len(self._chunk_keys),
            "graph_files": len(self._graph_files),
            "exec_files": len(self._exec_files),
            "errors": list(self._errors[-5:]),
            "final": False,
        }
        self._guard(
            "progress",
            lambda: self.storage.write_text(
                f"{self._key_prefix()}-progress.json", json.dumps(progress, indent=2)
            ),
        )
        sink = self.sink
        if sink is not None:
            self._guard(
                "sink.event",
                lambda: sink.event(
                    "shard.progress",
                    key=self.identity.outer_key,
                    metadata=[
                        ("tasks_done", str(counters["tasks_done"])),
                        ("section", str(section_view["name"] if section_view else "")),
                    ],
                ),
            )

    def heartbeat_fields(self) -> dict[str, Any]:
        """What the heartbeat carries about the inner scheduler, bounded."""
        limit = settings.inner_trace_heartbeat_sections
        with self._lock:
            closed = [s.as_dict() for s in self._sections if s.t1_ns is not None][-limit:]
            open_sections = [
                {
                    "name": s.name,
                    "group": list(s.group) if s.group else None,
                    "open_s": round((self.clock() - s.t0_ns) / 1e9, 1),
                    "tasks_started_this_section": self._tasks_started - s.tasks_started_at_open,
                    "compute_entered": s.compute_entry_ns is not None,
                    "first_dispatched": s.first_dispatch_ns is not None,
                }
                for s in self._open.values()
            ]
            counters = {
                "tasks_started": self._tasks_started,
                "tasks_done": self._tasks_done,
                "spill_count": self._spill_count,
            }
        return {
            "trace_id": hex(self.identity.trace_id),
            "outer_key": self.identity.outer_key,
            "scheduler": self.scheduler.kind,
            "seq": self._seq,
            "open_sections": open_sections,
            "sections": closed,
            "counters": counters,
            "active_by_class": self._active_by_class(),
            "host_rates": self._host_rates(),
            "error": self._errors[-1] if self._errors else None,
        }

    def snapshot(self) -> dict[str, Any]:
        """Frozen aggregates for the exec-trace summary and for tests."""
        with self._lock:
            sections = [s.as_dict() for s in self._sections]
            groups = [
                {**s.as_dict(), "construction": self._construction(s)}
                for s in self._sections
                if s.group is not None
            ]
            return {
                "sections": sections,
                "groups": groups,
                "tasks_started": self._tasks_started,
                "tasks_done": self._tasks_done,
                "spill_count": self._spill_count,
                "mutex_wait_s": round(self._mutex_wait_ns / 1e9, 3),
                "gil_s": round(self._gil_ns / 1e9, 3),
                "chunk_keys": list(self._chunk_keys),
                "graph_files": list(self._graph_files),
                "exec_files": list(self._exec_files),
                "errors": list(self._errors),
            }

    def _write_final(self, *, closed_by: str, error: str | None) -> None:
        snap = self.snapshot()
        exec_intervals: list[tuple[int, int]] = []
        with self._lock:
            for record in self._exec_by_key.values():
                if record.get("start_ns") is not None and record.get("end_ns") is not None:
                    exec_intervals.append((record["start_ns"], record["end_ns"]))
        payload = {
            "v": 1,
            **self.identity.as_dict(),
            "scheduler": self.scheduler.kind,
            "threads": self.scheduler.threads,
            "wall_s": round((self.clock() - self._started_ns) / 1e9, 1),
            "exec_union_s_unflushed": round(_union_seconds(exec_intervals), 3),
            "closed_by": closed_by,
            "error": error,
            **snap,
            "final": True,
        }
        self.storage.write_text(
            f"{self._key_prefix()}.a{self.identity.attempt:02d}.final.json",
            json.dumps(payload, indent=2, allow_nan=False),
        )

    # -- lifecycle ---------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stopping.wait(self.flush_s):
            self._guard("flush", lambda: self.flush(reason="periodic"))

    def __enter__(self) -> InnerTrace:
        global CURRENT  # noqa: PLW0603
        self._token = _trace.set(self)
        CURRENT = self.identity
        self._thread = threading.Thread(target=self._loop, name="lst-inner-trace", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        global CURRENT  # noqa: PLW0603
        exc = exc_info[1] if len(exc_info) > 1 else None
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=self.flush_s)
        error = None if exc is None else f"{type(exc).__name__}: {exc}"
        with self._lock:
            open_ids = list(self._open)
        for section_id in open_ids:
            self._guard("close", lambda sid=section_id: self.close(sid, error=error))
        self._guard("flush", lambda: self.flush(reason="error" if exc else "exit"))
        for writer in self._writers:
            writer.join(timeout=30)
        self._guard(
            "final", lambda: self._write_final(closed_by="error" if exc else "exit", error=error)
        )
        if self._token is not None:
            _trace.reset(self._token)
            self._token = None
        CURRENT = None
        return False


class NullInnerTrace:
    """The trace a shard gets when the real one could not start."""

    def section(self, _name: str, **_meta: Any) -> Any:
        return nullcontext()

    def group_observer(self) -> None:
        return None

    def observe_computes(self) -> Any:
        return nullcontext()

    def heartbeat_fields(self) -> dict[str, Any]:
        return {"error": "inner trace unavailable"}


def _last_state(events: Sequence[Mapping[str, Any]] | None) -> str | None:
    if not events:
        return None
    for event in reversed(events):
        if (event.get("event_type") or event.get("kind")) == "transition":
            return str((event.get("details") or {}).get("to_state"))
    return None


def _reason_from_local_state(state: Mapping[str, Any], key: str) -> tuple[str, str | None]:
    """Reason from dask's local scheduler state dict, by key spelling."""

    def contains(name: str) -> bool:
        bucket = state.get(name)
        if bucket is None:
            return False
        try:
            return any(_key_str(k) == key for k in bucket)
        except TypeError:
            return False

    # dask's local scheduler moves a key through waiting -> ready -> running ->
    # finished, then into ``released`` once every dependent has consumed it. A
    # key in ``released`` or ``finished`` therefore completed; only an alias
    # or a data node gets there without a ``pretask`` call.
    if contains("ready"):
        return "ready_not_started", "Ready"
    if contains("waiting"):
        return "blocked_on_deps", "Waiting"
    if contains("released") or contains("finished") or contains("cache"):
        return "completed_without_exec_span", "Memory"
    return "unknown", None


def build_inner_trace(
    *,
    identity: ShardIdentity,
    storage: StorageBackend,
    stem: str,
    scheduler: InnerScheduler,
    sink: SpanSink | None,
) -> InnerTrace | NullInnerTrace:
    """Construct the trace, or a null one when construction fails."""
    try:
        return InnerTrace(
            identity=identity, storage=storage, stem=stem, scheduler=scheduler, sink=sink
        )
    except Exception as exc:  # instrumentation never fails a shard
        log.warning("inner_trace_start_failed", stem=stem, error=str(exc))
        return NullInnerTrace()


def is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)
