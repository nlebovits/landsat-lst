"""Live evidence that a running tile is alive: heartbeats and captured logs.

A Coiled Batch task is a plain process on its own VM. It never registers with
the cluster's dask scheduler, so every panel on the cluster dashboard reports
the scheduler's own idleness rather than the tile's work, and stays flat for
hours whether the tile is computing or wedged. Its stdout goes to a file on the
VM that Coiled uploads only at exit, and its exit code is the tee wrapper's, not
the pipeline's. An operator watching that has nothing to read.

This module supplies the two things that replace it, both landing in the same
``_runs/{run_id}/`` prefix reconciliation already reads:

- :class:`TileHeartbeat` rewrites ``{tile}.progress.json`` every
  ``settings.heartbeat_interval_s`` and at every phase change. A tile that
  stops beating is wedged, killed, or preempted, and says so within two
  intervals. ``landsat-lst watch`` renders those objects.
- :func:`capture_task_log` tees the process's stdout and stderr to
  ``{tile}.log`` and uploads it on the way out, whether the tile succeeded or
  died. Capture is at the file-descriptor level, so GDAL and rasterio writing
  straight to fd 2 land in the log alongside Python tracebacks.

Phases are reported from wherever the work happens
(:func:`landsat_lst.pipeline.process_tile` and
:func:`landsat_lst.job.process_tile_job`) through :func:`report_phase`, which
finds the active heartbeat in a context variable. With no heartbeat running --
a local run, a test, a benchmark -- it does nothing, so instrumentation never
becomes a reason a tile fails.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: Phases one tile passes through, in order. Wall clock concentrates in
#: ``destriping`` (the coarse offset pass), ``compositing`` (the native P95),
#: and ``exporting`` (the COG write); the rest are seconds.
PHASES = (
    "starting",
    "stac_query",
    "loading",
    "destriping",
    "compositing",
    "exporting",
    "uploading",
)

#: Phases after which no further heartbeat is expected. A tile sitting on one of
#: these is finished, not stale, however old its heartbeat is.
TERMINAL_PHASES = ("done", "failed")

_PUMP_CHUNK_BYTES = 65536
_PUMP_JOIN_TIMEOUT_S = 5.0

_active: ContextVar[TileHeartbeat | None] = ContextVar("landsat_lst_heartbeat", default=None)


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process in MiB, if measurable.

    Reported on every heartbeat as well as in the final run record: the
    instance size for the global build is chosen from these numbers, and a tile
    that OOMs never gets to write a record.
    """
    try:
        import resource  # noqa: PLC0415

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except (ImportError, ValueError):  # pragma: no cover - non-POSIX
        return None


class TileHeartbeat:
    """Periodic proof that one tile is still working, written to storage.

    Use it as a context manager around the whole tile. Entering starts a daemon
    thread and publishes the first beat; leaving stops the thread and publishes
    a terminal beat (``done``, or ``failed`` with the exception that escaped).

    Every write is best-effort. A heartbeat that cannot be stored is logged and
    dropped: losing observability is a far smaller loss than failing a tile that
    is two hours into its composite.
    """

    def __init__(
        self,
        *,
        run_id: str,
        tile: str,
        window: str,
        storage: StorageBackend,
        interval_s: float | None = None,
    ) -> None:
        self.run_id = run_id
        self.tile = tile
        self.window = window
        self.storage = storage
        self.interval_s = settings.heartbeat_interval_s if interval_s is None else interval_s
        self.key = storage.progress_key(run_id, tile)

        self._host = socket.gethostname()
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase = "starting"
        self._error: str | None = None
        self._counts: dict[str, int] = {}
        self._token: Any = None
        # Seconds spent in each phase so far. Without this a manifest says a
        # tile took three hours and nothing about which phase to optimize.
        self._phase_seconds: dict[str, float] = {}
        self._phase_started = self._started
        # Completed and total dask tasks for the graph currently running, set
        # by :class:`GraphProgress`. Cleared when a graph finishes, so a phase
        # between graphs reports no fraction rather than a stale one.
        self._tasks_done: int | None = None
        self._tasks_total: int | None = None

    def payload(self) -> dict:
        """The heartbeat object as it is stored."""
        now = time.monotonic()
        with self._lock:
            phase, error, counts = self._phase, self._error, dict(self._counts)
            # The current phase's time is added live rather than only on exit,
            # so a tile killed mid-phase still reports where its hours went.
            phase_seconds = dict(self._phase_seconds)
            phase_seconds[phase] = round(
                phase_seconds.get(phase, 0.0) + (now - self._phase_started), 1
            )
            done, total = self._tasks_done, self._tasks_total
        return {
            "run_id": self.run_id,
            "tile": self.tile,
            "window": self.window,
            "phase": phase,
            "elapsed_s": round(now - self._started, 1),
            "phase_seconds": phase_seconds,
            "tasks_done": done,
            "tasks_total": total,
            "peak_rss_mb": peak_rss_mb(),
            "host": self._host,
            "pid": os.getpid(),
            "updated_at": datetime.now(tz=UTC).isoformat(),
            "error": error,
            **counts,
        }

    def set_task_progress(self, done: int | None, total: int | None) -> None:
        """Record progress through the dask graph now running. Never raises."""
        with self._lock:
            self._tasks_done = done
            self._tasks_total = total

    def write(self) -> None:
        """Publish the current state. Never raises."""
        payload = self.payload()
        try:
            self.storage.write_text(self.key, json.dumps(payload, indent=2))
        except Exception as e:
            log.warning(
                "heartbeat_write_failed", tile=self.tile, phase=payload["phase"], error=str(e)
            )

    def set_phase(self, phase: str, **counts: int | None) -> None:
        """Move to ``phase`` and publish immediately.

        Counts (``scenes_found``, ``scenes_kept``) accumulate across calls, so a
        later phase keeps what an earlier one learned. ``None`` values are
        ignored rather than stored, so a caller with nothing to add can pass the
        keyword unconditionally.
        """
        now = time.monotonic()
        with self._lock:
            self._phase_seconds[self._phase] = round(
                self._phase_seconds.get(self._phase, 0.0) + (now - self._phase_started), 1
            )
            self._phase_started = now
            self._phase = phase
            self._counts.update({k: v for k, v in counts.items() if v is not None})
            # A fraction belongs to the graph that reported it, never to the
            # next phase, which may run no graph at all.
            self._tasks_done = self._tasks_total = None
        self.write()

    def set_failed(self, error: str) -> None:
        """Record a terminal failure and its reason."""
        with self._lock:
            self._phase = "failed"
            self._error = error
        self.write()

    @property
    def phase(self) -> str:
        with self._lock:
            return self._phase

    def _loop(self) -> None:
        while not self._stopping.wait(self.interval_s):
            self.write()

    def __enter__(self) -> TileHeartbeat:
        self._token = _active.set(self)
        self._started = time.monotonic()
        # Published from this thread before the loop starts, which also forces
        # the storage backend to build its client here rather than racing to
        # build one in two threads at once.
        self.write()
        self._thread = threading.Thread(
            target=self._loop, name=f"lst-heartbeat-{self.tile}", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        """Stop beating and publish the tile's last word. Never suppresses."""
        exc = exc_info[1] if len(exc_info) > 1 else None
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s)
        try:
            # A failure reported from inside the tile is more specific than
            # anything reconstructable from the exception here, so it wins.
            if self.phase in TERMINAL_PHASES:
                pass
            elif exc is not None:
                self.set_failed(f"{type(exc).__name__}: {exc}")
            else:
                self.set_phase("done")
        finally:
            if self._token is not None:
                _active.reset(self._token)
                self._token = None
        return False


class GraphProgress:
    """Report how far a dask computation has got, into the active heartbeat.

    A phase like ``destriping`` can run for an hour as one ``dask.compute``
    call, and until now the only thing published about it was that it had
    started. Dask's callback hooks carry the same state its own ``ProgressBar``
    renders from, so a tile can report ``4182/18600 tasks`` and let an operator
    tell a slow phase from a wedged one, and estimate a finish time.

    Used as a context manager around a compute. Outside a batch task there is
    no heartbeat and this is inert, so the pipeline can wrap its computes
    unconditionally.

    Counting tasks is not counting work: dask tasks are wildly uneven, so the
    fraction is a progress indication, not a schedule. It is still the
    difference between "something is happening" and nothing at all.
    """

    def __init__(self) -> None:
        self._heartbeat = _active.get()
        self._callback: Any = None

    def __enter__(self) -> GraphProgress:
        if self._heartbeat is None:
            return self
        from dask.callbacks import Callback  # noqa: PLC0415

        heartbeat = self._heartbeat

        class _Reporter(Callback):
            # Same state dask's own ProgressBar reads: the scheduler keeps
            # every task in exactly one of these sets.
            # Dask invokes these positionally, so the arity is the contract and
            # the arguments it does not need are named with a leading
            # underscore rather than dropped.
            def _pretask(self, _key, _dsk, state):
                done = len(state["finished"])
                total = done + sum(len(state[k]) for k in ("ready", "waiting", "running"))
                heartbeat.set_task_progress(done, total)

            def _finish(self, _dsk, _state, _errored):
                heartbeat.set_task_progress(None, None)

        self._callback = _Reporter()
        self._callback.register()
        return self

    def __exit__(self, *exc_info: object) -> bool:
        if self._callback is not None:
            self._callback.unregister()
            self._callback = None
        if self._heartbeat is not None:
            self._heartbeat.set_task_progress(None, None)
        return False


def report_phase(phase: str, **counts: int | None) -> None:
    """Move the active heartbeat to ``phase``; do nothing when none is running."""
    heartbeat = _active.get()
    if heartbeat is not None:
        heartbeat.set_phase(phase, **counts)


def report_failed(error: str) -> None:
    """Mark the active heartbeat failed; do nothing when none is running.

    Needed because a tile that fails deterministically is *returned* as a failed
    result rather than raised, so nothing would otherwise reach the context
    manager's exception path.
    """
    heartbeat = _active.get()
    if heartbeat is not None:
        heartbeat.set_failed(error)


def active_heartbeat() -> TileHeartbeat | None:
    """The heartbeat this tile is reporting to, if any."""
    return _active.get()


def _pump(read_fd: int, mirror_fd: int, sink: Any) -> None:
    """Copy everything written to the pipe into both the log file and stdout.

    Only a read failure or end-of-stream ends the loop. A write failure is
    swallowed on purpose: if this thread ever stopped draining the pipe, the
    process writing to it would block forever once the 64 KiB buffer filled,
    which would hang a tile to protect a log file.
    """
    while True:
        try:
            chunk = os.read(read_fd, _PUMP_CHUNK_BYTES)
        except OSError:
            return
        if not chunk:
            return
        with suppress(OSError, ValueError):
            sink.write(chunk)
            sink.flush()
        with suppress(OSError):
            os.write(mirror_fd, chunk)


def _tail_text(path: Path, max_bytes: int) -> str:
    """The last ``max_bytes`` of the log, with a note when anything was dropped.

    The tail is the useful end: a traceback is the last thing a dying tile
    writes. Decoding replaces undecodable bytes rather than failing, because a
    log that cannot be uploaded explains nothing.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = b""
        if size > max_bytes:
            handle.seek(size - max_bytes)
            prefix = f"[truncated: {size - max_bytes} earlier bytes dropped]\n".encode()
        return (prefix + handle.read()).decode("utf-8", errors="replace")


@contextmanager
def capture_task_log(
    *,
    run_id: str,
    tile: str,
    storage: StorageBackend,
    max_bytes: int | None = None,
) -> Iterator[str]:
    """Tee this process's output to storage, uploading it on the way out.

    Capture is at the file-descriptor level rather than through
    ``sys.stdout``, so output from GDAL, rasterio, and any subprocess is caught
    alongside Python's. Everything is still mirrored to the real stdout, so a
    local run stays as readable as it was.

    An exception on the way out is printed while the capture is still in place
    and then re-raised, which is the only way its traceback reaches the log:
    the interpreter prints an uncaught exception long after this block has
    restored the descriptors. The console therefore shows the traceback twice.

    Args:
        run_id: Run the tile belongs to.
        tile: Tile name, which names the log object.
        storage: Backend the log is uploaded to.
        max_bytes: Ceiling on the uploaded text (default
            ``settings.task_log_max_bytes``). The full log stays on local disk.

    Yields:
        The key the log will be uploaded to.
    """
    from pathlib import Path  # noqa: PLC0415

    key = storage.log_key(run_id, tile)
    limit = settings.task_log_max_bytes if max_bytes is None else max_bytes

    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed in the finally below
        prefix=f"lst_{tile}_", suffix=".log", delete=False
    )
    local = Path(handle.name)

    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    pump = threading.Thread(
        target=_pump, args=(read_fd, saved_out, handle), name=f"lst-log-{tile}", daemon=True
    )
    pump.start()

    try:
        yield key
    except SystemExit:
        # The CLI already printed why, and a SystemExit traceback says nothing.
        raise
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except ValueError:  # pragma: no cover - stream already closed
            pass
        # Restoring both descriptors drops the last references to the pipe's
        # write end, which is what ends the pump's read.
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        # Drain before closing: whatever was still in the pipe is mirrored
        # through saved_out, so closing it first would swallow the last of the
        # output -- usually the traceback -- on its way to the console.
        pump.join(timeout=_PUMP_JOIN_TIMEOUT_S)
        os.close(saved_out)
        os.close(saved_err)
        if not pump.is_alive():
            # Left open when a lingering child still holds the write end;
            # closing an fd another thread is blocked reading is worse.
            os.close(read_fd)
        handle.close()

        try:
            storage.write_text(key, _tail_text(local, limit), content_type="text/plain")
            log.info("task_log_uploaded", tile=tile, key=key, bytes=local.stat().st_size)
        except Exception as e:
            log.warning("task_log_upload_failed", tile=tile, key=key, error=str(e))
        local.unlink(missing_ok=True)
