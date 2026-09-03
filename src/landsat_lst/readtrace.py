"""Record the shape of a composite shard's remote reads, and nothing else.

A composite shard's ``exporting`` phase scales with the number of distinct
Landsat items its band touches. Over the 32 comparable 512-row bands of run
``shard-S30W065-2021-2025-20260823T102135Z`` the fit is
``exporting = 119 s + 0.691 +/- 0.054 s per distinct item``, R^2 0.844. #129
excluded dependency assembly and falsified the suspected GDAL
cloud-configuration gap, then closed with the physical cause unmeasured. This
module measures the request shape that would explain it: how many round trips a
first touch of a COG costs, how much requests to different files overlap, and
whether any byte range is fetched twice.

It measures. It optimizes nothing, and it must not be extended to.

**How the records are obtained.** GDAL emits one debug message per remote
request. ``rasterio`` pushes ``logging_error_handler`` on every
``GDALEnv.start()``, and that handler routes ``CE_Debug`` to the
``rasterio._env`` logger as ``log.log(level, "%s in %s", code, msg)``. So with
``CPL_DEBUG=ON`` every request arrives as a Python ``LogRecord`` carrying
``record.created``, ``record.thread``, and the raw GDAL text in
``record.args[1]`` -- no string formatting needed in the hot path. Python
logging is process-global, so one handler sees every dask worker thread, and
``record.thread`` gives the per-thread ordering that turns a stream of requests
into a concurrency measurement: on one thread the records are strictly
sequential, so the gap between record *i* and record *i+1* bounds request *i*.

``CPL_LOG`` is deliberately unused. GDAL routes ``CPLDebug`` through the pushed
error handler, ``CPL_TIMESTAMP`` resolves to whole seconds, and a log file
carries no thread id.

**What this cannot answer.** The handler acquires the GIL once per debug
message, so a traced run's wall clock is not comparable to an untraced one. The
summary reports counts, ordering, and concurrency. Never quote a traced
``exporting`` time as a production number.

Off unless :attr:`~landsat_lst.config.Settings.read_trace` is set, and
best-effort throughout: a recorder that cannot start and a dump that cannot be
written are logged and swallowed, exactly as
:func:`~landsat_lst.profiling.profile_compute` does. The one deliberate
exception to "instrumentation never fails a tile" is :class:`ReadTraceComplete`,
which stops the shard once the capture window closes, so a 1,372 s band costs
about $0.05 rather than $0.31. It is unreachable with the flag off. See issue
#135.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from _thread import interrupt_main
from collections import Counter, defaultdict
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Iterator

log = structlog.get_logger(__name__)

#: The logger rasterio routes GDAL's CPLDebug messages to.
GDAL_LOGGER = "rasterio._env"

#: Config read inside a read task, on the thread that issues the requests.
#: ``GDAL_HTTP_VERSION`` and ``GDAL_HTTP_MULTIPLEX`` are the pair that decides
#: whether a ReadMultiRange can overlap at all: multiplexing needs HTTP/2, and
#: the default version is 1.1, under which it is inert.
CONFIG_KEYS = (
    "GDAL_HTTP_VERSION",
    "GDAL_HTTP_MULTIPLEX",
    "GDAL_HTTP_MAX_RETRY",
    "GDAL_HTTP_RETRY_DELAY",
    "GDAL_DISABLE_READDIR_ON_OPEN",
    "GDAL_INGESTED_BYTES_AT_OPEN",
    "GDAL_NUM_THREADS",
    "CPL_VSIL_CURL_CHUNK_SIZE",
    "CPL_VSIL_CURL_CACHE_SIZE",
    "CPL_VSIL_CURL_USE_HEAD",
)

#: Below this offset a range is header or directory structure rather than pixel
#: data. GDAL reads a remote file in 16 KB chunks by default, and a COG's IFDs
#: and tile index sit at the front of the file by construction.
HEADER_BYTES = 16 * 1024

#: The four CPLDebug format strings a remote read emits, recovered with
#: ``strings`` from the libgdal 3.12 shipped inside the rasterio wheel. They are
#: kept as data so a test can assert the regexes below still cover them: GDAL
#: owns this wording, and a wording change would silently empty every trace.
GDAL_FORMAT_STRINGS = (
    "Downloading %s (%s)...",
    "Downloading %s, ..., %s (%llu bytes, %s)...",
    "GetFileSize(%s)=%llu",
    "ReadMultiRange(%s), %s: response_code=%d, msg=%s",
)

_MULTI_DOWNLOAD = re.compile(r"^Downloading (.+?) \((\d+) bytes, (.+)\)\.\.\.$")
_DOWNLOAD = re.compile(r"^Downloading (.+) \((.+)\)\.\.\.$")
_FILE_SIZE = re.compile(r"^GetFileSize\((.+)\)=(\d+)")
_MULTI_RANGE = re.compile(r"^ReadMultiRange\((.+?)\), (.*): response_code=(\d+)")
_RANGE = re.compile(r"^(\d+)-(\d+)$")

#: Request kinds, as recorded.
GET = "get"
MULTI = "multi_get"
SIZE = "file_size"
MULTI_RESPONSE = "multi_response"

#: One recorded request: when, which thread, what kind, which file, and the
#: byte span it asked for.
Record = tuple[float, int, str, str, "int | None", "int | None", "int | None"]


class ReadTraceComplete(Exception):
    """The capture window closed and the trace has been written.

    Raised into the main thread once the window expires, so a traced shard
    stops after about 90 seconds instead of running a full band. It is a
    deliberate exception to the rule that instrumentation never fails a tile,
    and it is unreachable unless ``settings.read_trace`` is on.
    """


#: What one parsed message yields: kind, file, byte span, and size.
Parsed = tuple[str, str, "int | None", "int | None", "int | None"]


def _parse_download(msg: str) -> Parsed | None:
    """Either form of ``Downloading``: one range, or a ReadMultiRange batch."""
    multi = _MULTI_DOWNLOAD.match(msg)
    if multi is not None:
        ranges, nbytes, url = multi.groups()
        first = _RANGE.match(ranges.split(",", 1)[0].strip())
        start = int(first.group(1)) if first is not None else None
        return (MULTI, url, start, None, int(nbytes))
    single = _DOWNLOAD.match(msg)
    if single is None:
        return None
    span, url = single.groups()
    found = _RANGE.match(span.strip())
    if found is None:
        return (GET, url, None, None, None)
    start, end = int(found.group(1)), int(found.group(2))
    return (GET, url, start, end, end - start + 1)


def _parse_file_size(msg: str) -> Parsed | None:
    """``GetFileSize``, with or without its trailing response code."""
    found = _FILE_SIZE.match(msg)
    if found is None:
        return None
    return (SIZE, found.group(1), None, None, int(found.group(2)))


def _parse_multi_response(msg: str) -> Parsed | None:
    """The response line a ReadMultiRange logs after its batch returns."""
    found = _MULTI_RANGE.match(msg)
    if found is None:
        return None
    return (MULTI_RESPONSE, found.group(1), None, None, None)


#: Prefix to parser. The prefix test is what keeps GTiff and PROJ chatter off
#: the regex path, which matters: this runs once per GDAL debug message.
_PARSERS = (
    ("Downloading ", _parse_download),
    ("GetFileSize(", _parse_file_size),
    ("ReadMultiRange(", _parse_multi_response),
)


def parse_message(msg: str) -> Parsed | None:
    """Turn one GDAL debug message into ``(kind, url, start, end, nbytes)``.

    Returns ``None`` for every message that is not a remote request, which is
    most of them: GTiff, PROJ, and driver chatter all reach the same logger.
    """
    for prefix, parser in _PARSERS:
        if msg.startswith(prefix):
            return parser(msg)
    return None


def _thread_config() -> dict[str, Any]:
    """The effective GDAL config on the calling thread."""
    try:
        from rasterio.env import get_gdal_config  # noqa: PLC0415

        return {key: get_gdal_config(key, normalize=False) for key in CONFIG_KEYS}
    except Exception as e:  # pragma: no cover - rasterio is a hard dependency
        return {"error": str(e)}


class TraceHandler(logging.Handler):
    """Append one record per remote request, and drop everything else.

    Reads ``record.args[1]`` rather than calling ``record.getMessage()``: this
    runs once per GDAL debug message on a path that also carries GTiff and PROJ
    chatter, and formatting a string only to discard it is the cost avoided.
    """

    def __init__(self, max_records: int) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[Record] = []
        self.config_by_thread: dict[int, dict[str, Any]] = {}
        self.dropped = 0
        self._max = max_records

    def emit(self, record: logging.LogRecord) -> None:
        args = record.args
        if not isinstance(args, tuple) or len(args) != 2:
            return
        msg = args[1]
        if not isinstance(msg, str):
            return
        parsed = parse_message(msg)
        if parsed is None:
            return
        if len(self.records) >= self._max:
            self.dropped += 1
            return
        thread = record.thread or 0
        if thread not in self.config_by_thread:
            # The first request on this thread, so this is the read task's own
            # effective config rather than the process default.
            self.config_by_thread[thread] = _thread_config()
        kind, url, start, end, nbytes = parsed
        self.records.append((record.created, thread, kind, url, start, end, nbytes))

    def handleError(self, record: logging.LogRecord) -> None:
        """Swallow a recorder defect rather than let it reach the read path."""


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile, so an empty list cannot raise."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[rank]


def _first_touch(records: list[Record]) -> dict[str, Any]:
    """Round trips each file costs before it yields pixel data.

    Answers success criterion 1. A first touch is every request to a URL up to
    and including the first range that starts past the header region, so a
    ``GetFileSize`` plus a 16 KB header read plus the first data range counts
    as three.
    """
    seen: dict[str, int] = defaultdict(int)
    settled: dict[str, int] = {}
    elapsed: dict[str, float] = {}
    started: dict[str, float] = {}
    for created, _tid, kind, url, start, _end, _nbytes in records:
        if url in settled:
            continue
        seen[url] += 1
        started.setdefault(url, created)
        if kind in (GET, MULTI) and start is not None and start >= HEADER_BYTES:
            settled[url] = seen[url]
            elapsed[url] = created - started[url]
    counts = sorted(settled.values())
    spans = sorted(elapsed.values())
    return {
        "files_touched": len(seen),
        "files_reaching_pixel_data": len(settled),
        "requests_before_first_data": {
            "median": _percentile([float(c) for c in counts], 0.5),
            "p90": _percentile([float(c) for c in counts], 0.9),
            "max": max(counts) if counts else 0,
            # String keys, because this summary round-trips through JSON and
            # scripts/summarize_read_trace.py must rebuild it byte for byte.
            "histogram": {str(k): v for k, v in sorted(Counter(counts).items())},
        },
        "seconds_before_first_data": {
            "mean": round(sum(spans) / len(spans), 4) if spans else 0.0,
            "median": round(_percentile(spans, 0.5), 4),
            "p90": round(_percentile(spans, 0.9), 4),
        },
        "requests_per_file_overall": {
            "median": _percentile([float(v) for v in seen.values()], 0.5),
            "max": max(seen.values()) if seen else 0,
        },
    }


def _overlap(records: list[Record]) -> dict[str, Any]:
    """How much requests to different files run at the same time.

    Answers success criterion 2. On one thread the records are strictly
    sequential, so record *i* bounds a request that ended no later than record
    *i+1*. Sampling those intervals gives in-flight concurrency, and counting
    the distinct URLs inside each sample gives the part that matters: four
    requests to one file is not four files in flight.
    """
    by_thread: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for created, thread, _kind, url, _start, _end, _nbytes in records:
        by_thread[thread].append((created, url))
    spans: list[tuple[float, float, str]] = []
    for entries in by_thread.values():
        entries.sort()
        for i, (created, url) in enumerate(entries[:-1]):
            spans.append((created, entries[i + 1][0], url))
    if not spans:
        return {"threads_issuing_requests": len(by_thread), "samples": 0}
    starts = sorted(s for s, _e, _u in spans)
    sample_at = [starts[i] for i in range(0, len(starts), max(1, len(starts) // 400))]
    in_flight: list[float] = []
    distinct: list[float] = []
    for t in sample_at:
        live = [u for s, e, u in spans if s <= t < e]
        in_flight.append(float(len(live)))
        distinct.append(float(len(set(live))))
    return {
        "threads_issuing_requests": len(by_thread),
        "configured_threads": settings.dask_threads_per_worker,
        "samples": len(sample_at),
        "requests_in_flight": {
            "median": _percentile(in_flight, 0.5),
            "p90": _percentile(in_flight, 0.9),
            "max": max(in_flight),
        },
        "distinct_files_in_flight": {
            "median": _percentile(distinct, 0.5),
            "p90": _percentile(distinct, 0.9),
            "max": max(distinct),
        },
    }


def _refetch(records: list[Record]) -> dict[str, Any]:
    """Byte ranges and size probes fetched more than once.

    Answers success criterion 3. A repeated ``GetFileSize`` is a repeated HEAD,
    and a repeated ``(url, start, end)`` is a range the cache did not hold.
    """
    ranges: Counter[tuple[str, int, int]] = Counter()
    sizes: Counter[str] = Counter()
    span_bytes: dict[tuple[str, int, int], int] = {}
    for _created, _tid, kind, url, start, end, nbytes in records:
        if kind == SIZE:
            sizes[url] += 1
        elif kind == GET and start is not None and end is not None:
            key = (url, start, end)
            ranges[key] += 1
            span_bytes[key] = nbytes or 0
    repeats = {k: n for k, n in ranges.items() if n > 1}
    wasted = sum(span_bytes[k] * (n - 1) for k, n in repeats.items())
    top = sorted(repeats.items(), key=lambda kv: -kv[1])[:20]
    return {
        "distinct_ranges": len(ranges),
        "repeated_ranges": len(repeats),
        "refetched_requests": sum(n - 1 for n in repeats.values()),
        "refetched_bytes": wasted,
        "files_probed_more_than_once": sum(1 for n in sizes.values() if n > 1),
        "size_probes": sum(sizes.values()),
        "top_repeats": [
            {"url": url, "start": start, "end": end, "times": n} for (url, start, end), n in top
        ],
    }


def summarize(
    records: list[Record],
    *,
    config_by_thread: dict[int, dict[str, Any]] | None = None,
    window_s: float | None = None,
    dropped: int = 0,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The compact parsed summary that answers #135's four questions.

    The raw log is the evidence and this is derived from it, so
    ``scripts/summarize_read_trace.py`` reproduces this object from a retained
    log without a second cloud run.
    """
    kinds = Counter(kind for _c, _t, kind, _u, _s, _e, _n in records)
    span = 0.0
    if records:
        span = records[-1][0] - records[0][0]
    bytes_requested = sum((r[6] or 0) for r in records if r[2] in (GET, MULTI))
    configs = list((config_by_thread or {}).values())
    return {
        "schema_version": 1,
        "context": dict(context or {}),
        "window_s": window_s,
        "records": len(records),
        "records_dropped": dropped,
        "record_span_s": round(span, 3),
        "requests_per_second": round(len(records) / span, 2) if span > 0 else None,
        "kinds": dict(sorted(kinds.items())),
        "bytes_requested": bytes_requested,
        "first_touch": _first_touch(records),
        "overlap": _overlap(records),
        "refetch": _refetch(records),
        "read_multirange_used": kinds.get(MULTI, 0) > 0 or kinds.get(MULTI_RESPONSE, 0) > 0,
        "gdal_config": configs[0] if configs else {},
        "gdal_config_varies_by_thread": any(c != configs[0] for c in configs[1:]),
        "caveat": (
            "Request shape only. The handler takes the GIL once per GDAL debug "
            "message, so this run's wall clock is not comparable to an untraced "
            "run. Do not quote a traced exporting time as a production number."
        ),
    }


def _destination(stage: str, run_id: str, tile: str, index: int) -> tuple[Any, str] | None:
    """Where the two artifacts go, or ``None`` when nothing can take them.

    Under ``shards.unit_timing_prefix`` rather than beside the shard's profile:
    ``evidence._run_artifacts`` sweeps that prefix wholesale into a bundle,
    while it takes only ``/state/`` keys from the run prefix. A local run has no
    bucket, so it falls back to the manifest directory rather than declining to
    record at all.
    """
    from landsat_lst import shards  # noqa: PLC0415
    from landsat_lst.progress import active_heartbeat  # noqa: PLC0415

    prefix = shards.unit_trace_prefix(run_id, stage, tile, index)
    heartbeat = active_heartbeat()
    if heartbeat is not None:
        return heartbeat.storage, prefix

    from landsat_lst.storage import LocalStorage  # noqa: PLC0415

    return LocalStorage(output_dir=settings.manifest_dir / "readtrace"), prefix


def dump(
    handler: TraceHandler,
    *,
    stage: str,
    run_id: str,
    tile: str,
    index: int,
    window_s: float,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the raw log and the summary, and return the summary.

    Both writes are best-effort. Losing the trace costs a diagnostic; failing
    the block would cost the run.
    """
    records = list(handler.records)
    summary = summarize(
        records,
        config_by_thread=handler.config_by_thread,
        window_s=window_s,
        dropped=handler.dropped,
        context=context,
    )
    destination = _destination(stage, run_id, tile, index)
    if destination is None:  # pragma: no cover - _destination always returns
        return summary
    storage, prefix = destination
    try:
        storage.write_text(f"{prefix}.readtrace.summary.json", json.dumps(summary, indent=2))
    except Exception as e:
        log.warning("read_trace_dump_failed", key=f"{prefix}.readtrace.summary.json", error=str(e))
    # The raw log goes through a temp file because the storage backends expose
    # write_text and upload, never a bytes write, and gzip is what keeps a
    # 40,000-request band inside a megabyte.
    try:
        scratch = Path(tempfile.mkdtemp(prefix="lst_readtrace_"))
        try:
            local = scratch / "readtrace.jsonl.gz"
            with gzip.open(local, "wt", encoding="utf-8") as fh:
                for created, thread, kind, url, start, end, nbytes in records:
                    json.dump(
                        {
                            "t": round(created, 6),
                            "thread": thread,
                            "kind": kind,
                            "url": url,
                            "start": start,
                            "end": end,
                            "bytes": nbytes,
                        },
                        fh,
                    )
                    fh.write("\n")
            storage.upload(local, f"{prefix}.readtrace.jsonl.gz")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    except Exception as e:
        log.warning("read_trace_dump_failed", key=f"{prefix}.readtrace.jsonl.gz", error=str(e))
    log.info(
        "read_trace_written",
        records=len(records),
        dropped=handler.dropped,
        prefix=prefix,
    )
    return summary


def _uninstall(
    gdal_log: logging.Logger,
    handler: TraceHandler | None,
    timer: threading.Timer | None,
    *,
    level: int,
    debug: str | None,
) -> None:
    """Put the logger and the environment back exactly as they were.

    An orphaned handler would keep taking the GIL once per GDAL debug message
    for the life of the process, which is the cost this module exists to
    measure. Every step is suppressed on its own so one failure cannot skip the
    rest.
    """
    if timer is not None:
        with suppress(Exception):
            timer.cancel()
    if handler is not None:
        with suppress(Exception):
            gdal_log.removeHandler(handler)
    with suppress(Exception):
        gdal_log.setLevel(level)
    if debug is None:
        os.environ.pop("CPL_DEBUG", None)
    else:
        os.environ["CPL_DEBUG"] = debug


@contextmanager
def read_trace(
    *,
    stage: str,
    run_id: str,
    tile: str,
    index: int,
    context: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Record the block's remote requests, then stop the shard.

    Off unless ``settings.read_trace`` is set, and inert either way from the
    caller's point of view: a recorder that cannot start is logged and
    swallowed, and the block runs untouched.

    ``CPL_DEBUG`` is set in ``os.environ`` **before** the block, because
    odc-loader snapshots the environment at graph build and hands that snapshot
    to every read task. A ``rasterio.Env`` here would not reach them: it is
    thread-local, and the reads happen on dask worker threads.

    The stop is a ``KeyboardInterrupt`` raised into the main thread by
    ``_thread.interrupt_main``, converted to :class:`ReadTraceComplete` here.
    Python has no way to raise an arbitrary exception into another thread, and
    the alternative -- raising from the logging handler on a worker thread --
    would be swallowed by ``logging`` itself.

    Args:
        stage: Shard stage, for the artifact key.
        run_id: Run token, for the artifact key.
        tile: Tile name, for the artifact key.
        index: Shard index, for the artifact key.
        context: Anything worth recording beside the counts, such as the item
            count the band touches.
    """
    if not settings.read_trace:
        yield
        return

    gdal_log = logging.getLogger(GDAL_LOGGER)
    previous_level = gdal_log.level
    previous_debug = os.environ.get("CPL_DEBUG")
    expired = threading.Event()
    timer: threading.Timer | None = None
    handler: TraceHandler | None = None
    started = time.monotonic()

    def _close_window() -> None:
        expired.set()
        if handler is not None:
            with suppress(Exception):
                dump(
                    handler,
                    stage=stage,
                    run_id=run_id,
                    tile=tile,
                    index=index,
                    window_s=time.monotonic() - started,
                    context=context,
                )
        with suppress(Exception):
            interrupt_main()

    try:
        handler = TraceHandler(settings.read_trace_max_records)
        os.environ["CPL_DEBUG"] = "ON"
        gdal_log.setLevel(logging.DEBUG)
        gdal_log.addHandler(handler)
        timer = threading.Timer(settings.read_trace_seconds, _close_window)
        timer.daemon = True
        timer.start()
        log.info(
            "read_trace_started",
            stage=stage,
            tile=tile,
            index=index,
            seconds=settings.read_trace_seconds,
        )
    except Exception as e:
        log.warning("read_trace_start_failed", error=str(e))
        _uninstall(gdal_log, handler, timer, level=previous_level, debug=previous_debug)
        yield
        return

    try:
        yield
    except KeyboardInterrupt:
        if not expired.is_set():
            raise
        msg = (
            f"read trace captured {len(handler.records)} requests in "
            f"{settings.read_trace_seconds}s; stopping the shard here"
        )
        raise ReadTraceComplete(msg) from None
    finally:
        _uninstall(gdal_log, handler, timer, level=previous_level, debug=previous_debug)
        if not expired.is_set():
            with suppress(Exception):
                dump(
                    handler,
                    stage=stage,
                    run_id=run_id,
                    tile=tile,
                    index=index,
                    window_s=time.monotonic() - started,
                    context=context,
                )
