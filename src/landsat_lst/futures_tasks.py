"""What a shard looks like as one future: the wrapper that runs on a worker.

A shard submitted as a Dask future is still the same shard
(:func:`landsat_lst.shard_tasks.run_shard`, with its own log, heartbeat, and
attempt number). What the wrapper adds is the outer task's identity, so the
records the shard writes can be joined to the future the scheduler shows, and
the inner scheduler the shard's graphs run on, which on this path is the
in-process Frisky cluster that makes them observable (issue #155).

Nothing here imports coiled, frisky, or distributed at module level. The Frisky
sink is built lazily and its failure is recorded, never raised: the shard
computes either way, and the run's observability gate reads the failure from
the heartbeat.
"""

from __future__ import annotations

import hashlib
import os
import resource
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import structlog

from landsat_lst.config import settings
from landsat_lst.innertrace import FriskySink, new_trace_id, outer_binding

if TYPE_CHECKING:
    from landsat_lst.innertrace import InnerMode
    from landsat_lst.models import ProcessingJob

log = structlog.get_logger()


def session_token() -> str:
    """Eight hex characters that make one driver session's keys its own."""
    return uuid.uuid4().hex[:8]


def task_key(run_id: str, tile: str, stage: str, index: int, token: str) -> str:
    """The outer future's key: ``lst-{run8}-{stage}-{tile}-{index:04d}-{token}``.

    ``run8`` is the first eight hex digits of the run id's SHA-256, the same
    trick ``batch.stage_cluster_name`` uses so truncation cannot eat the
    marker. The token is per driver session so a resume never collides with a
    key a dead session left on the scheduler. It is the key
    ``frisky.query_spans`` filters on and the key every ``shard.*`` span carries.
    """
    run8 = hashlib.sha256(run_id.encode()).hexdigest()[:8]
    return f"lst-{run8}-{stage}-{tile}-{index:04d}-{token}"


@dataclass(frozen=True)
class ShardResult:
    """What one outer future returns: small, serializable, and joinable."""

    stage: str
    index: int
    key: str | None
    keys: list[str]
    skipped: bool
    wall_s: float
    peak_rss_gb: float | None
    worker: str
    pid: int
    hostname: str
    inner_scheduler: str
    inner_threads: int | None
    trace_id: str
    sink_error: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _peak_rss_gb() -> float | None:
    try:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3)
    except (ValueError, OSError):  # pragma: no cover - non-POSIX
        return None


def run_shard_task(
    stage: str,
    run_id: str,
    tile: str,
    index: int,
    *_deps: Any,
    job: ProcessingJob | None = None,
    units: int | None = None,
    key: str | None = None,
    trace_id: int | None = None,
    inner_scheduler: InnerMode = "frisky",
    use_frisky_sink: bool = True,
    storage: Any = None,
) -> ShardResult:
    """Run one shard inside one future.

    ``*_deps`` are the resolved values of the futures this one depends on.
    They are ignored; they exist so the scheduler enforces the edge.

    The inner scheduler is named here, explicitly, for this path only. A shard
    run by Coiled Batch never passes through this function and keeps the
    threaded scheduler it always had.
    """
    from landsat_lst import shard_tasks  # noqa: PLC0415

    started = time.perf_counter()
    sink = None
    sink_error: str | None = None
    if use_frisky_sink:
        try:
            sink = FriskySink()
        except Exception as exc:  # instrumentation never fails a shard
            sink_error = f"{type(exc).__name__}: {exc}"
            log.warning("frisky_sink_unavailable", error=sink_error)
    resolved_trace_id = trace_id if trace_id is not None else new_trace_id()

    with outer_binding(
        outer_key=key,
        trace_id=resolved_trace_id,
        sink=sink,
        inner_scheduler=inner_scheduler,
        sink_error=sink_error,
    ):
        if stage == "merge":
            shard_tasks.configure_logging()
            shard_tasks.apply_shard_settings()
            merged = shard_tasks.merge_offsets(run_id, tile, storage=storage)
            written = [merged.storage_key] if hasattr(merged, "storage_key") else [str(merged)]
            skipped = False
        else:
            result = shard_tasks.run_shard(
                stage, run_id, tile, index, job=job, units=units, storage=storage
            )
            written = (
                [str(k) for k in result]
                if isinstance(result, list | tuple)
                else ([] if result is None else [str(result)])
            )
            skipped = isinstance(result, list | tuple) and len(result) == 0

    return ShardResult(
        stage=stage,
        index=index,
        key=key,
        keys=written,
        skipped=skipped,
        wall_s=round(time.perf_counter() - started, 3),
        peak_rss_gb=_peak_rss_gb(),
        worker=os.environ.get("FRISKY_WORKER_ADDRESS", ""),
        pid=os.getpid(),
        hostname=socket.gethostname(),
        inner_scheduler=inner_scheduler,
        inner_threads=settings.inner_threads,
        trace_id=hex(resolved_trace_id),
        sink_error=sink_error,
    )
