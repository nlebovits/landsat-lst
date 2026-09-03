"""Best-effort execution trace for one composite shard's Dask compute.

The hot path only appends primitive values.  JSON, gzip, CSV, and storage I/O
happen after the wrapped compute has returned, so observing a shard cannot
repeat the logging-lock failure from issue #135.

Host samples come from a child process, not a thread.  The first production
trace (S30W065 band 16, 2026-09-03) ran its sampler as an in-process thread and
held a mean cadence of 1.61 s against a 1.0 s target, with 120 gaps above 2 s,
while the process averaged about one busy core.  The thread was not starved of
CPU; it was starved of the GIL behind sixteen worker threads and the scheduler
thread.  A child process reading ``/proc`` through psutil has no GIL to wait
for, which is also why ``dask.diagnostics.ResourceProfiler`` samples from one.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import multiprocessing
import os
import tempfile
import threading
import time
from array import array
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil
import structlog
from dask.callbacks import Callback
from dask.utils import key_split

from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

CLASSES = ("read", "rechunk", "compute", "store", "other")
_CLASS_ID = {name: index for index, name in enumerate(CLASSES)}
_MIB = 1024**2

TIMELINE_FIELDS = (
    "timestamp",
    "cpu_cores_busy",
    "rss_mb",
    "network_recv_mb_s",
    "network_send_mb_s",
    "disk_read_mb_s",
    "disk_write_mb_s",
    "num_fds",
    "active_read_tasks",
    "active_rechunk_tasks",
    "active_compute_tasks",
    "active_store_tasks",
    "active_other_tasks",
    "sources_started",
    "sources_finished",
)

HOST_FIELDS = TIMELINE_FIELDS[:8]


def classify(prefix: str) -> str:
    """Put a real composite-graph key prefix into one stable coarse class."""
    value = prefix.lower().replace("_", "-")
    if value in {"keixel", "lwir11", "qa-pixel"} or value.startswith(("open-", "cfg-", "grid-")):
        return "read"
    if value in {"open", "cfg", "grid"}:
        return "read"
    if (
        value.startswith(
            (
                "getitem-stack-",
                "where",
                "astype",
                "invert",
                "custom-nanquantile",
                "nanquantile-last",
                "getitem-sum-",
                "stack",
                "sum",
            )
        )
        or value == "sub"
    ):
        return "compute"
    if value.startswith(("rechunk", "shuffle-")) or value in {
        "transpose",
        "getitem",
    }:
        return "rechunk"
    if value.startswith("store"):
        return "store"
    return "other"


def build_timeline(
    host_samples: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    reads: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build timeline rows from plain records without reading global state."""
    starts = np.asarray([float(task["start"]) for task in tasks], dtype=np.float64)
    ends = np.asarray(
        [math.inf if task.get("end") is None else float(task["end"]) for task in tasks],
        dtype=np.float64,
    )
    classes = np.asarray(
        [_CLASS_ID.get(str(task.get("class", "other")), _CLASS_ID["other"]) for task in tasks],
        dtype=np.uint8,
    )
    return _build_timeline(host_samples, starts, ends, classes, reads)


def _build_timeline(
    host_samples: Sequence[Mapping[str, Any]],
    task_starts: np.ndarray,
    task_ends: np.ndarray,
    task_classes: np.ndarray,
    reads: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    boundaries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, class_id in _CLASS_ID.items():
        mask = task_classes == class_id
        boundaries[name] = (np.sort(task_starts[mask]), np.sort(task_ends[mask]))

    read_starts = np.sort(np.asarray([float(item["t_start"]) for item in reads], dtype=np.float64))
    read_ends = np.sort(np.asarray([float(item["t_end"]) for item in reads], dtype=np.float64))
    rows: list[dict[str, Any]] = []
    for sample in sorted(host_samples, key=lambda item: float(item["timestamp"])):
        stamp = float(sample["timestamp"])
        row = {field: sample.get(field) for field in HOST_FIELDS}
        row["timestamp"] = stamp
        for name, (starts, ends) in boundaries.items():
            begun = int(np.searchsorted(starts, stamp, side="right"))
            done = int(np.searchsorted(ends, stamp, side="right"))
            row[f"active_{name}_tasks"] = begun - done
        row["sources_started"] = int(np.searchsorted(read_starts, stamp, side="right"))
        row["sources_finished"] = int(np.searchsorted(read_ends, stamp, side="right"))
        rows.append(row)
    return rows


# --- host sampling -----------------------------------------------------------

_COUNTERS = ("cpu_s", "net_recv", "net_send", "disk_read", "disk_write")


def _raw_host_sample(process: psutil.Process) -> tuple[float, ...]:
    """One raw sample: wall clock plus five monotonically increasing counters, then RSS and fds."""
    stamp = time.time()
    try:
        cpu = process.cpu_times()
        cpu_s = float(cpu.user + cpu.system)
    except Exception:
        cpu_s = math.nan
    try:
        rss_mb = process.memory_info().rss / _MIB
    except Exception:
        rss_mb = math.nan
    try:
        net = psutil.net_io_counters()
        net_recv, net_send = float(net.bytes_recv), float(net.bytes_sent)
    except Exception:
        net_recv = net_send = math.nan
    try:
        disk = psutil.disk_io_counters()
        disk_read = float(disk.read_bytes) if disk is not None else math.nan
        disk_write = float(disk.write_bytes) if disk is not None else math.nan
    except Exception:
        disk_read = disk_write = math.nan
    try:
        num_fds = float(process.num_fds())
    except Exception:
        num_fds = math.nan
    return (stamp, cpu_s, net_recv, net_send, disk_read, disk_write, rss_mb, num_fds)


def _rate(current: float, previous: float, elapsed: float, *, scale: float) -> float:
    if not (math.isfinite(current) and math.isfinite(previous)):
        return math.nan
    return max(0.0, current - previous) / elapsed / scale


def _host_rows(raw: Sequence[tuple[float, ...]]) -> list[dict[str, float | None]]:
    """Turn raw counter samples into per-second rates, in HOST_FIELDS order."""
    rows: list[dict[str, float | None]] = []
    previous: tuple[float, ...] | None = None
    for sample in raw:
        stamp, rss_mb, num_fds = sample[0], sample[6], sample[7]
        rates = [0.0] * 5
        if previous is not None:
            elapsed = stamp - previous[0]
            if elapsed > 0:
                scales = (1.0, _MIB, _MIB, _MIB, _MIB)
                rates = [
                    _rate(sample[i + 1], previous[i + 1], elapsed, scale=scales[i])
                    for i in range(5)
                ]
        previous = sample
        values = (stamp, rates[0], rss_mb, rates[1], rates[2], rates[3], rates[4], num_fds)
        rows.append(
            {name: _finite_or_none(value) for name, value in zip(HOST_FIELDS, values, strict=True)}
        )
    return rows


def _host_sampler_main(conn: Any, pid: int, interval_s: float) -> None:
    """Child-process body: sample ``pid`` on an absolute schedule until told to stop.

    The schedule is absolute (``start + n * interval``) rather than relative, so
    the cost of a sample does not accumulate into drift.  On the stop message
    the child ships every raw sample back in one send and exits.
    """
    process = psutil.Process(pid)
    samples: list[tuple[float, ...]] = []
    samples.append(_raw_host_sample(process))
    conn.send("ready")
    next_at = time.time() + interval_s
    while True:
        wait = next_at - time.time()
        if conn.poll(max(0.0, wait)):
            break
        samples.append(_raw_host_sample(process))
        next_at += interval_s
        # A missed beat is reported, never silently "caught up" with a burst.
        while next_at < time.time():
            next_at += interval_s
    samples.append(_raw_host_sample(process))
    conn.send(samples)
    conn.close()


class _HostRecorder:
    """Host samples from a child process, with an in-thread fallback."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.mode = "not_started"
        self.error: str | None = None
        self.raw: list[tuple[float, ...]] = []
        self._conn: Any = None
        self._process: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        try:
            ctx = multiprocessing.get_context("spawn")
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_host_sampler_main,
                args=(child, os.getpid(), self.interval_s),
                name="exec-trace-host",
                daemon=True,
            )
            proc.start()
            child.close()
            if not parent.poll(30.0) or parent.recv() != "ready":
                raise TimeoutError("host sampler process did not report ready")
            self._conn, self._process = parent, proc
            self.mode = "process"
            return
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        self._thread = threading.Thread(
            target=self._thread_run, name="exec-trace-host", daemon=True
        )
        self._thread.start()
        self.mode = "thread"

    def _thread_run(self) -> None:
        process = psutil.Process()
        self.raw.append(_raw_host_sample(process))
        while not self._stop.wait(self.interval_s):
            self.raw.append(_raw_host_sample(process))
        self.raw.append(_raw_host_sample(process))

    def stop(self) -> None:
        if self.mode == "process":
            try:
                self._conn.send("stop")
                if self._conn.poll(30.0):
                    self.raw = list(self._conn.recv())
                else:
                    self.error = "host sampler process did not return its samples"
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                with suppress(Exception):
                    self._process.join(timeout=5.0)
                with suppress(Exception):
                    if self._process.is_alive():
                        self._process.kill()
        elif self.mode == "thread" and self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(2.0, self.interval_s * 2))

    def records(self) -> list[dict[str, float | None]]:
        return _host_rows(self.raw)


# --- dask task events ------------------------------------------------------------


class _TaskRecorder:
    def __init__(self) -> None:
        self.starts = array("d")
        self.ends = array("d")
        self.prefix_ids = array("I")
        self.threads = array("q")
        self.prefixes: list[str] = []
        self._prefix_ids: dict[str, int] = {}
        self._inflight: dict[Any, int] = {}
        self.callback = Callback(pretask=self._pretask, posttask=self._posttask)

    def _pretask(self, key: Any, _dsk: Any, _state: Any) -> None:
        raw = key[0] if isinstance(key, tuple) and key else key
        prefix = key_split(raw)
        prefix_id = self._prefix_ids.get(prefix)
        if prefix_id is None:
            prefix_id = len(self.prefixes)
            self._prefix_ids[prefix] = prefix_id
            self.prefixes.append(prefix)
        index = len(self.starts)
        self.starts.append(time.time())
        self.ends.append(math.nan)
        self.prefix_ids.append(prefix_id)
        self.threads.append(0)
        self._inflight[key] = index

    def _posttask(self, key: Any, _result: Any, _dsk: Any, _state: Any, worker_id: Any) -> None:
        index = self._inflight.pop(key, None)
        if index is None:
            return
        self.ends[index] = time.time()
        with suppress(TypeError, ValueError, OverflowError):
            self.threads[index] = int(worker_id)

    def arrays(self, compute_finished_at: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        starts = np.frombuffer(self.starts, dtype=np.float64)
        ends = np.frombuffer(self.ends, dtype=np.float64).copy()
        ends[np.isnan(ends)] = compute_finished_at
        prefix_ids = np.frombuffer(self.prefix_ids, dtype=np.uint32)
        prefix_classes = np.asarray(
            [_CLASS_ID[classify(prefix)] for prefix in self.prefixes], dtype=np.uint8
        )
        classes = prefix_classes[prefix_ids] if len(prefix_ids) else np.asarray([], dtype=np.uint8)
        return starts, ends, classes

    def distinct_threads(self) -> int:
        return len({value for value in self.threads if value})


# --- source reads --------------------------------------------------------------------


class _ReadRecorder:
    """Time every Nth ``rio_read``, split at the ``_do_read`` boundary.

    ``odc.loader._rio._rio_read`` opens the dataset with ``rasterio.open``,
    resolves the band and reprojection, then hands the open band to the
    module-global ``_do_read`` for the actual read or warp.  Wrapping both
    module attributes gives three stamps per sampled call: before the open,
    at the ``_do_read`` entry, and after the read returns.  ``open_s`` is the
    first interval and ``read_s`` the second.
    """

    def __init__(self, sample_every: int) -> None:
        self.sample_every = sample_every
        self.total = 0
        self.records: list[dict[str, Any]] = []
        self.status = "not_started"
        self.error: str | None = None
        self._lock = threading.Lock()
        self._local = threading.local()
        self._module: Any = None
        self._originals: dict[str, Any] = {}

    def install(self) -> None:
        try:
            import odc.loader._rio as rio  # noqa: PLC0415

            original_read = rio.rio_read
            original_do_read = rio._do_read
            local = self._local

            def traced_do_read(*args: Any, **kwargs: Any) -> Any:
                if getattr(local, "sampling", False):
                    local.do_read_started = time.time()
                    try:
                        return original_do_read(*args, **kwargs)
                    finally:
                        local.do_read_ended = time.time()
                return original_do_read(*args, **kwargs)

            def traced(src: Any, cfg: Any, dst_geobox: Any, *args: Any, **kwargs: Any) -> Any:
                with self._lock:
                    self.total += 1
                    record_this = self.total % self.sample_every == 0
                if not record_this:
                    return original_read(src, cfg, dst_geobox, *args, **kwargs)
                local.sampling = True
                local.do_read_started = None
                local.do_read_ended = None
                started = time.time()
                try:
                    return original_read(src, cfg, dst_geobox, *args, **kwargs)
                finally:
                    ended = time.time()
                    local.sampling = False
                    try:
                        shape = [int(value) for value in dst_geobox.shape]
                    except Exception:
                        shape = None
                    do_read_started = local.do_read_started
                    do_read_ended = local.do_read_ended
                    record = {
                        "thread": threading.get_ident(),
                        "uri": str(getattr(src, "uri", "")),
                        "t_start": started,
                        "t_open_end": do_read_started,
                        "t_end": ended,
                        "open_s": (do_read_started - started) if do_read_started else None,
                        "read_s": (do_read_ended - do_read_started)
                        if do_read_started and do_read_ended
                        else None,
                        "shape": shape,
                    }
                    with self._lock:
                        self.records.append(record)

            module: Any = rio  # dynamic monkeypatch; the checker sees the real signatures
            self._module = module
            self._originals = {"rio_read": original_read, "_do_read": original_do_read}
            module.rio_read = traced
            module._do_read = traced_do_read
            self.status = "enabled"
        except Exception as exc:
            self.status = "unavailable"
            self.error = f"{type(exc).__name__}: {exc}"

    def restore(self) -> None:
        if self._module is None:
            return
        for name, original in self._originals.items():
            try:
                setattr(self._module, name, original)
            except Exception as exc:
                self.status = "restore_failed"
                self.error = f"{type(exc).__name__}: {exc}"


# --- the trace ---------------------------------------------------------------------


class _ExecutionTrace:
    def __init__(self, storage: StorageBackend, stem: str) -> None:
        self.storage = storage
        self.stem = stem
        self.host = _HostRecorder(settings.exec_trace_interval_s)
        self.tasks = _TaskRecorder()
        self.reads = _ReadRecorder(settings.exec_trace_read_sample)
        self.compute_started_at = 0.0
        self.compute_finished_at = 0.0
        self._callback_entered = False

    def start(self) -> None:
        try:
            self.host.start()
        except Exception as exc:
            log.warning("exec_trace_host_start_failed", error=str(exc))
        if self.host.error:
            log.warning("exec_trace_host_fallback", mode=self.host.mode, error=self.host.error)
        self.reads.install()
        try:
            self.tasks.callback.__enter__()
            self._callback_entered = True
        except Exception as exc:
            log.warning("exec_trace_task_start_failed", error=str(exc))
        self.compute_started_at = time.time()

    def stop(self) -> None:
        self.compute_finished_at = time.time()
        if self._callback_entered:
            with suppress(Exception):
                self.tasks.callback.__exit__(None, None, None)
        self.reads.restore()
        with suppress(Exception):
            self.host.stop()

    def publish(self) -> None:
        try:
            self._publish()
        except Exception as exc:
            log.warning("exec_trace_publish_failed", stem=self.stem, error=str(exc))

    def _publish(self) -> None:
        host = self.host.records()
        starts, ends, classes = self.tasks.arrays(self.compute_finished_at)
        timeline = _build_timeline(host, starts, ends, classes, self.reads.records)
        keys = {
            "events": f"{self.stem}.exectrace.events.jsonl.gz",
            "timeline": f"{self.stem}.exectrace.timeline.csv",
            "summary": f"{self.stem}.exectrace.summary.json",
        }
        with tempfile.TemporaryDirectory(prefix="lst_exec_trace_") as directory:
            root = Path(directory)
            paths = {
                "events": root / "events.jsonl.gz",
                "timeline": root / "timeline.csv",
                "summary": root / "summary.json",
            }
            self._write_events(paths["events"], host)
            _write_timeline(paths["timeline"], timeline)
            upload_started_at = time.time()
            upload_errors: dict[str, str] = {}
            for name in ("events", "timeline"):
                try:
                    self.storage.upload(paths[name], keys[name])
                except Exception as exc:
                    upload_errors[name] = f"{type(exc).__name__}: {exc}"
                    log.warning("exec_trace_upload_failed", artifact=name, error=str(exc))
            upload_finished_at = time.time()
            summary = self._summary(
                host=host,
                starts=starts,
                ends=ends,
                upload_started_at=upload_started_at,
                upload_finished_at=upload_finished_at,
                keys=keys,
                upload_errors=upload_errors,
            )
            paths["summary"].write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            try:
                self.storage.upload(paths["summary"], keys["summary"])
            except Exception as exc:
                log.warning("exec_trace_upload_failed", artifact="summary", error=str(exc))
            log.info(
                "exec_trace_published",
                stem=self.stem,
                tasks=summary["n_tasks"],
                host_samples=summary["host_samples"],
                host_gap_max_s=summary["host_gap_max_s"],
                sampler_mode=summary["host_sampler_mode"],
            )

    def _write_events(self, path: Path, host: Sequence[Mapping[str, Any]]) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for sample in host:
                _write_jsonl(stream, {"kind": "host", **sample})
            for index, started in enumerate(self.tasks.starts):
                ended = self.tasks.ends[index]
                prefix = self.tasks.prefixes[self.tasks.prefix_ids[index]]
                _write_jsonl(
                    stream,
                    {
                        "kind": "task",
                        "start": started,
                        "end": _finite_or_none(ended),
                        "prefix": prefix,
                        "class": classify(prefix),
                        "thread": self.tasks.threads[index] or None,
                    },
                )
            for record in self.reads.records:
                _write_jsonl(stream, {"kind": "read", **record})

    def _summary(
        self,
        *,
        host: Sequence[Mapping[str, Any]],
        starts: np.ndarray,
        ends: np.ndarray,
        upload_started_at: float,
        upload_finished_at: float,
        keys: Mapping[str, str],
        upload_errors: Mapping[str, str],
    ) -> dict[str, Any]:
        prefix_ids = np.frombuffer(self.tasks.prefix_ids, dtype=np.uint32)
        counts = np.bincount(prefix_ids, minlength=len(self.tasks.prefixes))
        durations = np.bincount(
            prefix_ids, weights=np.maximum(0.0, ends - starts), minlength=len(self.tasks.prefixes)
        )
        prefix_table: list[dict[str, Any]] = [
            {
                "id": index,
                "prefix": prefix,
                "class": classify(prefix),
                "tasks": int(counts[index]),
                "task_seconds": float(durations[index]),
            }
            for index, prefix in enumerate(self.tasks.prefixes)
        ]
        per_class = {
            name: {
                "tasks": sum(row["tasks"] for row in prefix_table if row["class"] == name),
                "task_seconds": sum(
                    row["task_seconds"] for row in prefix_table if row["class"] == name
                ),
            }
            for name in CLASSES
        }
        reads = self.reads.records
        totals = np.asarray([r["t_end"] - r["t_start"] for r in reads], dtype=np.float64)
        opens = np.asarray(
            [r["open_s"] for r in reads if r["open_s"] is not None], dtype=np.float64
        )
        datas = np.asarray(
            [r["read_s"] for r in reads if r["read_s"] is not None], dtype=np.float64
        )
        rss = [float(sample["rss_mb"]) for sample in host if sample.get("rss_mb") is not None]
        stamps = [float(sample["timestamp"]) for sample in host]
        gaps = np.diff(np.asarray(stamps, dtype=np.float64)) if len(stamps) > 1 else np.asarray([])
        try:
            import dask  # noqa: PLC0415

            num_workers = dask.config.get("num_workers", None)
        except Exception:
            num_workers = None
        return {
            "compute_started_at": self.compute_started_at,
            "compute_finished_at": self.compute_finished_at,
            "wall_s": self.compute_finished_at - self.compute_started_at,
            "n_tasks": len(self.tasks.starts),
            "per_class": per_class,
            "dask_num_workers": num_workers,
            "distinct_worker_threads": self.tasks.distinct_threads(),
            "host_sampler_mode": self.host.mode,
            "host_sampler_error": self.host.error,
            "host_interval_s": self.host.interval_s,
            "host_samples": len(stamps),
            "host_gap_max_s": float(gaps.max()) if gaps.size else None,
            "host_gap_mean_s": float(gaps.mean()) if gaps.size else None,
            "host_gaps_over_2s": int((gaps >= 2.0).sum()) if gaps.size else 0,
            "n_reads_total": self.reads.total,
            "n_reads_recorded": len(reads),
            "read_sample_rule": f"every {self.reads.sample_every}th rio_read call",
            "read_hook_status": self.reads.status,
            "read_hook_error": self.reads.error,
            "read_split_recorded": int(opens.size),
            "read_duration_s": _quantiles(totals),
            "read_open_s": _quantiles(opens),
            "read_data_s": _quantiles(datas),
            "peak_rss_mb": max(rss, default=None),
            "artifact_keys": dict(keys),
            "upload_started_at": upload_started_at,
            # The summary is uploaded last; this marks completion of its two
            # data artifacts, which is the observable overlap gate.
            "upload_finished_at": upload_finished_at,
            "upload_errors": dict(upload_errors),
            "prefix_table": prefix_table,
        }


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    return {
        "n": int(values.size),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "mean": float(values.mean()) if values.size else None,
        "max": float(values.max()) if values.size else None,
    }


def _percentile(values: np.ndarray, percentile: int) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _write_jsonl(stream: Any, record: Mapping[str, Any]) -> None:
    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
    stream.write("\n")


def _write_timeline(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TIMELINE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def exec_trace(*, storage: StorageBackend, stem: str) -> Iterator[None]:
    """Trace one compute and publish three artifacts; inert unless enabled."""
    if not settings.exec_trace:
        yield
        return
    if settings.profile_dask:
        log.warning("exec_trace_with_dask_profile", stem=stem)
    try:
        trace = _ExecutionTrace(storage, stem)
        trace.start()
    except Exception as exc:
        log.warning("exec_trace_start_failed", stem=stem, error=str(exc))
        yield
        return
    try:
        yield
    finally:
        trace.stop()
        trace.publish()
