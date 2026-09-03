"""Best-effort execution trace for one composite shard's Dask compute.

The hot path only appends primitive values.  JSON, gzip, CSV, and storage I/O
happen after the wrapped compute has returned, so observing a shard cannot
repeat the logging-lock failure from issue #135.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
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


def classify(prefix: str) -> str:
    """Put a real composite-graph key prefix into one stable coarse class."""
    value = prefix.lower().replace("_", "-")
    if value in {"keixel", "lwir11", "qa-pixel"} or value.startswith(("open-", "cfg-", "grid-")):
        return "read"
    if value in {"open", "cfg", "grid"}:
        return "read"
    if value.startswith(
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
    ):
        return "compute"
    if value.startswith(("rechunk", "shuffle-taker")) or value in {
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
        row = {field: sample.get(field) for field in TIMELINE_FIELDS[:8]}
        row["timestamp"] = stamp
        for name, (starts, ends) in boundaries.items():
            begun = int(np.searchsorted(starts, stamp, side="right"))
            done = int(np.searchsorted(ends, stamp, side="right"))
            row[f"active_{name}_tasks"] = begun - done
        row["sources_started"] = int(np.searchsorted(read_starts, stamp, side="right"))
        row["sources_finished"] = int(np.searchsorted(read_ends, stamp, side="right"))
        rows.append(row)
    return rows


class _HostRecorder:
    _FIELDS = (
        "timestamp",
        "cpu_cores_busy",
        "rss_mb",
        "network_recv_mb_s",
        "network_send_mb_s",
        "disk_read_mb_s",
        "disk_write_mb_s",
        "num_fds",
    )

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.values = {name: array("d") for name in self._FIELDS}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="exec-trace-host", daemon=True)
        self._process = psutil.Process()
        self._previous: tuple[float, float, float, float, float, float] | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 2))

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.interval_s):
            self._sample()
        self._sample()

    def _sample(self) -> None:
        stamp = time.time()
        try:
            cpu = self._process.cpu_times()
            cpu_s = float(cpu.user + cpu.system)
        except Exception:
            cpu_s = math.nan
        with suppress(Exception):
            rss_mb = self._process.memory_info().rss / _MIB
        if "rss_mb" not in locals():
            rss_mb = math.nan
        try:
            net = psutil.net_io_counters()
        except Exception:
            net = None
        try:
            disk = psutil.disk_io_counters()
        except Exception:
            disk = None
        net_recv = float(net.bytes_recv) if net is not None else math.nan
        net_send = float(net.bytes_sent) if net is not None else math.nan
        disk_read = float(disk.read_bytes) if disk is not None else math.nan
        disk_write = float(disk.write_bytes) if disk is not None else math.nan
        try:
            num_fds = float(self._process.num_fds())
        except Exception:
            num_fds = math.nan

        cpu_busy = recv_rate = send_rate = read_rate = write_rate = 0.0
        current = (stamp, cpu_s, net_recv, net_send, disk_read, disk_write)
        if self._previous is not None:
            old_t, old_cpu, old_recv, old_send, old_read, old_write = self._previous
            elapsed = stamp - old_t
            if elapsed > 0:
                cpu_busy = _rate(cpu_s, old_cpu, elapsed, scale=1.0)
                recv_rate = _rate(net_recv, old_recv, elapsed, scale=_MIB)
                send_rate = _rate(net_send, old_send, elapsed, scale=_MIB)
                read_rate = _rate(disk_read, old_read, elapsed, scale=_MIB)
                write_rate = _rate(disk_write, old_write, elapsed, scale=_MIB)
        self._previous = current
        row = (
            stamp,
            cpu_busy,
            rss_mb,
            recv_rate,
            send_rate,
            read_rate,
            write_rate,
            num_fds,
        )
        for name, value in zip(self._FIELDS, row, strict=True):
            self.values[name].append(value)

    def records(self) -> list[dict[str, float | None]]:
        return [
            {name: _finite_or_none(self.values[name][index]) for name in self._FIELDS}
            for index in range(len(self.values["timestamp"]))
        ]


def _rate(current: float, previous: float, elapsed: float, *, scale: float) -> float:
    if not (math.isfinite(current) and math.isfinite(previous)):
        return math.nan
    return max(0.0, current - previous) / elapsed / scale


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


class _ReadRecorder:
    def __init__(self, sample_every: int) -> None:
        self.sample_every = sample_every
        self.total = 0
        self.records: list[dict[str, Any]] = []
        self.status = "not_started"
        self.error: str | None = None
        self._lock = threading.Lock()
        self._module: Any = None
        self._original: Any = None
        self._wrapper: Any = None

    def install(self) -> None:
        try:
            import odc.loader._rio as rio  # noqa: PLC0415

            original = rio.rio_read

            def traced(src: Any, cfg: Any, dst_geobox: Any, *args: Any, **kwargs: Any) -> Any:
                with self._lock:
                    self.total += 1
                    record_this = self.total % self.sample_every == 0
                if not record_this:
                    return original(src, cfg, dst_geobox, *args, **kwargs)
                started = time.time()
                try:
                    return original(src, cfg, dst_geobox, *args, **kwargs)
                finally:
                    try:
                        shape = [int(value) for value in dst_geobox.shape]
                    except Exception:
                        shape = None
                    record = {
                        "thread": threading.get_ident(),
                        "uri": str(getattr(src, "uri", "")),
                        "t_start": started,
                        "t_end": time.time(),
                        "shape": shape,
                    }
                    with self._lock:
                        self.records.append(record)

            self._module = rio
            self._original = original
            self._wrapper = traced
            setattr(self._module, "rio_read", traced)  # noqa: B010 - intentional module patch
            self.status = "enabled"
        except Exception as exc:
            self.status = "unavailable"
            self.error = f"{type(exc).__name__}: {exc}"

    def restore(self) -> None:
        if self._module is None or self._original is None:
            return
        try:
            setattr(  # noqa: B010 - restore the intentional module patch
                self._module, "rio_read", self._original
            )
        except Exception as exc:
            self.status = "restore_failed"
            self.error = f"{type(exc).__name__}: {exc}"


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
        read_durations = np.asarray(
            [record["t_end"] - record["t_start"] for record in self.reads.records],
            dtype=np.float64,
        )
        rss = [float(sample["rss_mb"]) for sample in host if sample.get("rss_mb") is not None]
        return {
            "compute_started_at": self.compute_started_at,
            "compute_finished_at": self.compute_finished_at,
            "wall_s": self.compute_finished_at - self.compute_started_at,
            "n_tasks": len(self.tasks.starts),
            "per_class": per_class,
            "n_reads_total": self.reads.total,
            "n_reads_recorded": len(self.reads.records),
            "read_sample_rule": f"every {self.reads.sample_every}th rio_read call",
            "read_hook_status": self.reads.status,
            "read_hook_error": self.reads.error,
            "read_duration_s_p50": _percentile(read_durations, 50),
            "read_duration_s_p95": _percentile(read_durations, 95),
            "peak_rss_mb": max(rss, default=None),
            "artifact_keys": dict(keys),
            "upload_started_at": upload_started_at,
            # The summary is uploaded last; this marks completion of its two
            # data artifacts, which is the observable overlap gate.
            "upload_finished_at": upload_finished_at,
            "upload_errors": dict(upload_errors),
            "prefix_table": prefix_table,
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
