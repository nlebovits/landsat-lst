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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from landsat_lst.job import JobResult
    from landsat_lst.models import ProcessingJob
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: Phases one tile passes through, in order. Wall clock concentrates in
#: ``destriping`` (the coarse offset pass) and ``exporting``, which is now the
#: tile's only pass over the native stack.
#:
#: ``coverage_check`` used to sit between them and was retired in issue #80: it
#: was a full native pass whose entire output was one log line, and the same
#: numbers now come off the written raster inside ``exporting``. Naming it is
#: what made it measurable, and then deletable.
#:
#: Split finer than the work is, on purpose. A label covering three unrelated
#: things cannot point at any of them: ``compositing`` used to span lazy graph
#: construction, an eager coverage reduction, and the handoff to export, and a
#: nine-minute silence inside it was unattributable. ``composite_graph`` and
#: ``land_mask`` run no dask graph at all and take single-threaded Python time
#: that every concurrency lever we have is blind to, which is exactly why they
#: are named.
PHASES = (
    "starting",
    "stac_query",
    "loading",
    "land_mask",
    "offset_load",
    "destriping",
    "composite_graph",
    "exporting",
    "uploading",
)

#: Phases after which no further heartbeat is expected. A tile sitting on one of
#: these is finished, not stale, however old its heartbeat is.
#:
#: ``skipped`` is terminal too: a tile whose COGs already exist does no work and
#: publishes once. It needs an honest phase of its own rather than borrowing
#: ``done``, which would claim it computed something.
TERMINAL_PHASES = ("done", "failed", "skipped")

#: Version of the published state object. 1 was the split pair of a heartbeat
#: at ``{tile}.progress.json`` and a run record at ``{tile}.json``; 2 merges
#: them and keys them by attempt. A body with no ``schema`` key is 1.
SCHEMA_VERSION = 2

#: A dask graph is running and :class:`GraphProgress` is reporting it.
GRAPH_RUNNING = "running"

#: No dask graph is running. Distinct from "a graph is running but has not
#: reported yet", which is what a bare ``tasks_total=None`` used to mean and
#: could not be told apart from an uninstrumented silence.
GRAPH_IDLE = "idle"

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


def rss_mb() -> float | None:
    """Resident set size right now, in MiB, or ``None`` where unmeasurable.

    Distinct from :func:`peak_rss_mb`, which is a high-water mark and so can
    only rise. A watcher plotting the peak draws a staircase that never falls,
    cannot show a phase releasing memory, and cannot tell "holding 35 GB now"
    from "touched 35 GB an hour ago and is at 4 GB". Headroom against the VM's
    memory needs the current figure, and the alarm needs both.

    Reads ``/proc/self/statm``, so it answers on the Linux VMs that run tiles
    and returns ``None`` on a machine without ``/proc``.
    """
    try:
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, IndexError, ValueError, AttributeError):  # pragma: no cover
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
        job: ProcessingJob,
        storage: StorageBackend,
        attempt: int = 1,
        interval_s: float | None = None,
    ) -> None:
        self.run_id = run_id
        self.job = job
        self.tile = job.tile.name
        self.window = job.window_label
        self.attempt = attempt
        self.storage = storage
        self.interval_s = settings.heartbeat_interval_s if interval_s is None else interval_s
        self.key = storage.run_record_key(run_id, self.tile, attempt)
        self.pointer_key = storage.run_record_key(run_id, self.tile)

        # The tile's outcome, folded in by :meth:`set_result` and published by
        # the terminal beat. Holding it here rather than writing it to a second
        # key is the whole of the merge: the run record and the last heartbeat
        # were one object described twice.
        self._result: dict[str, Any] | None = None
        # Which graph is running, counted from the tile's start. A watcher
        # cannot otherwise tell two graphs apart, and a task rate spliced
        # across that boundary would be an ETA for a graph that already
        # finished.
        self._graph_seq = 0
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
        # Whether a dask graph is running at all, which a task count cannot say:
        # `None` counts mean both "no graph here" and "a graph that has not
        # reported yet", and those need different reactions from a watcher.
        self._graph_state = GRAPH_IDLE

    def _identity(self) -> dict[str, Any]:
        """The fields that never change for the life of this attempt."""
        from landsat_lst.instance import instance_identity  # noqa: PLC0415

        machine = instance_identity()
        return {
            # What this tile actually ran on, so a cost estimate reads the
            # machine rather than the preference list. Resolved through an
            # lru_cache, so the first beat pays for the probe and no other
            # beat does.
            "instance_type": machine.instance_type,
            "instance_lifecycle": machine.lifecycle.value,
            "instance_source": machine.source,
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "tile": self.tile,
            "window": self.window,
            "attempt": self.attempt,
            "year": self.job.year,
            "end_year": self.job.end_year,
            # Carried so a reader can rebuild the job exactly. Without it a
            # sampled run reconciles to window "2021-2025" for tiles whose
            # COGs live under "2021-2025-sample300".
            "max_scenes": self.job.max_scenes,
            "host": self._host,
            "pid": os.getpid(),
        }

    def payload(self) -> dict:
        """The tile's state object as it is stored.

        One object carries both halves. The live half is rewritten every beat;
        the outcome half is ``None`` until :meth:`set_result` folds a result in,
        and ``status`` stays ``None`` for exactly that long. A running tile with
        a status would give a mid-run reconcile a verdict to read and would
        give ``watch`` a second liveness signal to disagree with ``phase``.
        """
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
            graph_state, graph_seq = self._graph_state, self._graph_seq
            result = dict(self._result) if self._result else {}
        return {
            **self._identity(),
            "phase": phase,
            "status": None,
            "elapsed_s": round(now - self._started, 1),
            "duration_s": None,
            "phase_seconds": phase_seconds,
            "tasks_done": done,
            "tasks_total": total,
            "graph_state": graph_state,
            "graph_seq": graph_seq,
            "rss_mb": rss_mb(),
            "peak_rss_mb": peak_rss_mb(),
            "scene_count": None,
            "lst_key": None,
            "qa_key": None,
            "updated_at": datetime.now(tz=UTC).isoformat(),
            "error": error,
            **counts,
            # The outcome wins where the two overlap. Its identity fields are
            # the same job's, so the merge cannot contradict itself.
            **result,
        }

    def set_task_progress(self, done: int | None, total: int | None) -> None:
        """Record progress through the dask graph now running. Never raises."""
        with self._lock:
            self._tasks_done = done
            self._tasks_total = total

    def set_graph_state(self, state: str) -> None:
        """Record whether a dask graph is running. Never raises.

        Owned by :class:`GraphProgress` rather than by :meth:`set_phase`,
        because a phase boundary and a graph boundary are different events:
        ``composite_graph`` and ``exporting`` are two phases inside one stretch
        of work, and only the second one runs a graph.
        """
        with self._lock:
            if state == GRAPH_RUNNING and self._graph_state != GRAPH_RUNNING:
                # Counted on the idle-to-running edge only, so a repeated call
                # cannot inflate it. A watcher divides task counts by time
                # within one sequence number and never across two.
                self._graph_seq += 1
            self._graph_state = state

    def set_result(self, result: JobResult) -> None:
        """Fold this tile's outcome into the object, without publishing.

        The terminal beat publishes it. Keeping the result here rather than
        writing it to a second key is the whole of the merge: the run record
        and the last heartbeat were one object described twice, and the two
        could disagree because a retry overwrote one of them.
        """
        with self._lock:
            self._result = result.to_record()

    def write(self) -> None:
        """Publish the current state. Never raises."""
        payload = self.payload()
        try:
            self.storage.write_text(self.key, json.dumps(payload, indent=2))
        except Exception as e:
            log.warning(
                "heartbeat_write_failed", tile=self.tile, phase=payload["phase"], error=str(e)
            )

    def write_pointer(self) -> None:
        """Copy the settled state to the unsuffixed key. Never raises.

        The body is identical to the attempt's own object, so a reader that
        knows only ``{tile}.json`` finds a superset of what the old run record
        held, with no attempt logic at all. Its presence is also how every
        reader tells a settled tile from a running one.

        Written once, at the terminal boundary, and never from the beat loop. A
        pointer refreshed every minute would double the run's PUT bill to buy a
        key that is already published under its own name.
        """
        payload = self.payload()
        try:
            self.storage.write_text(self.pointer_key, json.dumps(payload, indent=2))
        except Exception as e:
            log.warning("pointer_write_failed", tile=self.tile, error=str(e))

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

    def set_terminal(self, phase: str) -> None:
        """Move straight to a terminal phase and publish, folding its time in.

        Used by a tile that never beat, so its ``phase_seconds`` records the
        phase it settled in rather than an invented history.
        """
        self.set_phase(phase)

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
                # Republish rather than skip. A tile that reported its own
                # failure reached this phase before `set_result` folded the
                # outcome in, so its attempt object still carries a null
                # status, which the schema reads as "still running".
                self.write()
            elif exc is not None:
                self.set_failed(f"{type(exc).__name__}: {exc}")
            else:
                self.set_phase("done")
            if exc is None:
                # An escaping exception means a retry is being scheduled, so
                # this attempt is not the tile's final answer. Its own object
                # is written either way and keeps the evidence; only the
                # settled-state pointer waits for an attempt that finishes.
                self.write_pointer()
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
        # Published before the first task retires, so the window between
        # entering a compute and dask's first callback reads as "running with no
        # count yet" rather than as no graph at all.
        heartbeat.set_graph_state(GRAPH_RUNNING)

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
            self._heartbeat.set_graph_state(GRAPH_IDLE)
        return False


#: Set while a caller is building pipeline graphs for inspection rather than
#: running a tile. See :func:`silence_sections`.
_silenced: ContextVar[bool] = ContextVar("landsat_lst_sections_silenced", default=False)


@contextmanager
def silence_sections() -> Iterator[None]:
    """Stop :func:`timed_section` narrating, for a caller that is not a tile.

    ``landsat-lst plan`` builds the pipeline's real graphs against synthetic
    data to count their tasks. That runs the same instrumented code a tile does,
    but nothing about it is a tile: there is no work to attribute, and one
    ``phase_complete`` line on stdout makes ``plan --json`` unparsable.

    A context variable rather than reconfiguring structlog, which is global,
    order-dependent, and caches bound loggers -- muting it worked in isolation
    and leaked in a full test run.
    """
    token = _silenced.set(True)
    try:
        yield
    finally:
        _silenced.reset(token)


@contextmanager
def timed_section(phase: str, **counts: int | None) -> Iterator[None]:
    """Enter ``phase`` and log what it cost on the way out.

    For the stretches that run no dask graph and so publish no task count:
    graph construction, the land-mask rasterization, the STAC query. Each is
    single-threaded Python, invisible to every concurrency lever we have, and
    none had ever been measured. Anything that can exceed roughly ten seconds
    belongs in one. See issue #77 item 4.

    The duration is logged rather than published, because the heartbeat already
    carries per-phase seconds; this puts the same number in the task log, where
    a post-mortem reads it without a running watcher.
    """
    if _silenced.get():
        yield
        return

    report_phase(phase, **counts)
    started = time.monotonic()
    try:
        yield
    finally:
        log.info("phase_complete", phase=phase, seconds=round(time.monotonic() - started, 1))


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


def write_final_state(
    *,
    run_id: str,
    job: ProcessingJob,
    storage: StorageBackend,
    attempt: int,
    result: JobResult,
    phase: str = "skipped",
) -> None:
    """Publish one settled tile-state object without ever beating.

    A tile whose COGs already exist does no work, so it has nothing to report
    while it runs and no phase history to keep. Constructing the heartbeat and
    publishing twice gives it the same schema every other tile has, for two
    PUTs and no thread. Starting a daemon thread, beating, and joining it would
    cost a resumed 700-tile run several hundred thread churns to publish a
    number that never changes.

    Best-effort, like every write in this module.
    """
    beat = TileHeartbeat(run_id=run_id, job=job, storage=storage, attempt=attempt)
    beat.set_result(result)
    beat.set_terminal(phase)
    beat.write_pointer()


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
    attempt: int = 1,
    max_bytes: int | None = None,
    key: str | None = None,
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
        key: Where to upload, overriding the per-tile default. For a batch task
            that is not a tile -- the synthetic sweep is the one caller --
            writing under ``_runs/`` would make ``runs.classify`` read it as a
            tile attempt and put it in a manifest. Such a caller owns its own
            key grammar and passes the key here.

    Yields:
        The key the log will be uploaded to.
    """
    from pathlib import Path  # noqa: PLC0415

    if key is None:
        key = storage.log_key(run_id, tile, attempt)
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
