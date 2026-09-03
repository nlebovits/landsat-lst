"""The read trace records request shape and never costs a shard anything.

Credential-less by construction: nothing here reaches rasterio, a control
plane, or a bucket. The GDAL messages are the four format strings compiled into
libgdal 3.12, verified with ``strings`` against the shipped rasterio wheel.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time

import pytest

from landsat_lst import readtrace, shards
from landsat_lst.config import settings
from landsat_lst.readtrace import (
    GDAL_FORMAT_STRINGS,
    GET,
    MULTI,
    MULTI_RESPONSE,
    SIZE,
    ReadTraceComplete,
    TraceHandler,
    parse_message,
    read_trace,
    summarize,
)

URL_A = "/vsis3/usgs-landsat/collection02/level-2/a/lwir11.TIF"
URL_B = "/vsis3/usgs-landsat/collection02/level-2/b/lwir11.TIF"


def record(t, thread, kind, url, *, start=None, end=None, nbytes=None):
    """One trace record, in the tuple order the handler appends."""
    return (t, thread, kind, url, start, end, nbytes)


# ---------------------------------------------------------------------------
# Parsing the four GDAL message forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("msg", "expected"),
    [
        (
            f"Downloading 0-16383 ({URL_A})...",
            (GET, URL_A, 0, 16383, 16384),
        ),
        (
            f"Downloading 32768-49151 ({URL_A})...",
            (GET, URL_A, 32768, 49151, 16384),
        ),
        (
            f"Downloading 0-1023, ..., 8192-9215 (4096 bytes, {URL_B})...",
            (MULTI, URL_B, 0, None, 4096),
        ),
        (
            f"GetFileSize({URL_A})=104857600",
            (SIZE, URL_A, None, None, 104857600),
        ),
        (
            f"GetFileSize({URL_A})=104857600  response_code=200",
            (SIZE, URL_A, None, None, 104857600),
        ),
        (
            f"ReadMultiRange({URL_B}), 0-1023: response_code=206, msg=OK",
            (MULTI_RESPONSE, URL_B, None, None, None),
        ),
    ],
)
def test_parses_every_gdal_request_form(msg, expected):
    assert parse_message(msg) == expected


@pytest.mark.parametrize(
    "msg",
    [
        "GDAL_DATA found in environment.",
        "GTiff: ScanDirectories()",
        "PROJ: proj_create_from_database",
        "Downloading",
        "",
    ],
)
def test_drops_everything_that_is_not_a_request(msg):
    """Driver and PROJ chatter reaches the same logger and must cost nothing."""
    assert parse_message(msg) is None


def test_the_regexes_still_cover_every_documented_gdal_format():
    """GDAL owns this wording, and a change would silently empty every trace.

    Fills each recovered format string with plausible values and asserts the
    parser recognises the result, so the coupling to libgdal is checked rather
    than assumed.
    """
    filled = {
        "Downloading %s (%s)...": f"Downloading 0-16383 ({URL_A})...",
        "Downloading %s, ..., %s (%llu bytes, %s)...": (
            f"Downloading 0-1023, ..., 4096-5119 (2048 bytes, {URL_A})..."
        ),
        "GetFileSize(%s)=%llu": f"GetFileSize({URL_A})=4096",
        "ReadMultiRange(%s), %s: response_code=%d, msg=%s": (
            f"ReadMultiRange({URL_A}), 0-1023: response_code=206, msg=OK"
        ),
    }
    assert set(filled) == set(GDAL_FORMAT_STRINGS)
    for template in GDAL_FORMAT_STRINGS:
        assert parse_message(filled[template]) is not None, template


# ---------------------------------------------------------------------------
# The handler
# ---------------------------------------------------------------------------


def emit(handler, msg, *, thread=1, created=0.0):
    """Push one message through the handler the way rasterio's does."""
    entry = logging.LogRecord(
        name=readtrace.GDAL_LOGGER,
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg="%s in %s",
        args=("CPLE_None", msg),
        exc_info=None,
    )
    entry.thread = thread
    entry.created = created
    handler.emit(entry)


def test_handler_keeps_requests_and_drops_chatter(monkeypatch):
    monkeypatch.setattr(readtrace, "_thread_config", lambda: {"GDAL_HTTP_VERSION": "1.1"})
    handler = TraceHandler(max_records=100)
    emit(handler, f"Downloading 0-16383 ({URL_A})...", created=1.0)
    emit(handler, "GTiff: something entirely unrelated", created=1.1)
    emit(handler, f"GetFileSize({URL_A})=1024", created=1.2)

    assert len(handler.records) == 2
    assert [r[2] for r in handler.records] == [GET, SIZE]
    assert handler.config_by_thread == {1: {"GDAL_HTTP_VERSION": "1.1"}}


def test_handler_ignores_records_that_are_not_rasterio_shaped():
    """A record whose args are not the (code, msg) pair must not raise."""
    handler = TraceHandler(max_records=10)
    entry = logging.LogRecord(
        name=readtrace.GDAL_LOGGER,
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg="plain message",
        args=None,
        exc_info=None,
    )
    handler.emit(entry)
    assert handler.records == []


def test_handler_caps_records_and_counts_the_drops(monkeypatch):
    monkeypatch.setattr(readtrace, "_thread_config", lambda: {})
    handler = TraceHandler(max_records=2)
    for i in range(5):
        emit(handler, f"Downloading {i}-{i} ({URL_A})...", created=float(i))

    assert len(handler.records) == 2
    assert handler.dropped == 3


def test_handler_snapshots_config_once_per_thread(monkeypatch):
    calls = []

    def fake_config():
        calls.append(1)
        return {"GDAL_HTTP_VERSION": "1.1"}

    monkeypatch.setattr(readtrace, "_thread_config", fake_config)
    handler = TraceHandler(max_records=100)
    emit(handler, f"Downloading 0-1 ({URL_A})...", thread=7, created=1.0)
    emit(handler, f"Downloading 2-3 ({URL_A})...", thread=7, created=1.1)
    emit(handler, f"Downloading 4-5 ({URL_B})...", thread=8, created=1.2)

    assert len(calls) == 2
    assert set(handler.config_by_thread) == {7, 8}


# ---------------------------------------------------------------------------
# The summary, question by question
# ---------------------------------------------------------------------------


def test_first_touch_counts_requests_before_pixel_data():
    """Criterion 1: a size probe, a header read, then the first data range."""
    records = [
        record(0.0, 1, SIZE, URL_A, nbytes=1_000_000),
        record(0.1, 1, GET, URL_A, start=0, end=16383, nbytes=16384),
        record(0.4, 1, GET, URL_A, start=65536, end=81919, nbytes=16384),
        record(0.5, 1, GET, URL_A, start=81920, end=98303, nbytes=16384),
    ]
    touch = summarize(records)["first_touch"]

    assert touch["files_touched"] == 1
    assert touch["files_reaching_pixel_data"] == 1
    assert touch["requests_before_first_data"]["median"] == 3
    assert touch["seconds_before_first_data"]["mean"] == pytest.approx(0.4)


def test_first_touch_ignores_a_file_that_never_reached_pixel_data():
    records = [
        record(0.0, 1, SIZE, URL_A, nbytes=1_000_000),
        record(0.1, 1, GET, URL_A, start=0, end=16383, nbytes=16384),
    ]
    touch = summarize(records)["first_touch"]

    assert touch["files_touched"] == 1
    assert touch["files_reaching_pixel_data"] == 0


def test_overlap_counts_distinct_files_not_just_requests():
    """Criterion 2: four requests to one file is one file in flight."""
    same_file = [
        record(float(i) / 10, i, GET, URL_A, start=65536, end=65551, nbytes=16) for i in range(4)
    ]
    same_file += [
        record(1.0 + i, i, GET, URL_A, start=131072, end=131087, nbytes=16) for i in range(4)
    ]
    overlap = summarize(same_file)["overlap"]

    assert overlap["threads_issuing_requests"] == 4
    assert overlap["distinct_files_in_flight"]["max"] == 1
    assert overlap["requests_in_flight"]["max"] == 4


def test_overlap_sees_two_files_in_flight():
    records = [
        record(0.0, 1, GET, URL_A, start=65536, end=65551, nbytes=16),
        record(0.0, 2, GET, URL_B, start=65536, end=65551, nbytes=16),
        record(5.0, 1, GET, URL_A, start=131072, end=131087, nbytes=16),
        record(5.0, 2, GET, URL_B, start=131072, end=131087, nbytes=16),
    ]
    overlap = summarize(records)["overlap"]

    assert overlap["distinct_files_in_flight"]["max"] == 2


def test_refetch_counts_repeated_ranges_and_repeated_head():
    """Criterion 3: the same range twice, and the same file probed twice."""
    records = [
        record(0.0, 1, SIZE, URL_A, nbytes=1024),
        record(0.1, 1, SIZE, URL_A, nbytes=1024),
        record(0.2, 1, GET, URL_A, start=0, end=16383, nbytes=16384),
        record(0.3, 2, GET, URL_A, start=0, end=16383, nbytes=16384),
        record(0.4, 2, GET, URL_B, start=0, end=16383, nbytes=16384),
    ]
    refetch = summarize(records)["refetch"]

    assert refetch["repeated_ranges"] == 1
    assert refetch["refetched_requests"] == 1
    assert refetch["refetched_bytes"] == 16384
    assert refetch["files_probed_more_than_once"] == 1
    assert refetch["size_probes"] == 2


def test_summary_reports_multirange_use_and_the_http_version():
    """Criterion 4 needs both: multiplexing is inert over HTTP/1.1."""
    records = [record(0.0, 1, MULTI, URL_A, start=0, nbytes=4096)]
    summary = summarize(records, config_by_thread={1: {"GDAL_HTTP_VERSION": "1.1"}})

    assert summary["read_multirange_used"] is True
    assert summary["gdal_config"]["GDAL_HTTP_VERSION"] == "1.1"
    assert summary["gdal_config_varies_by_thread"] is False


def test_summary_carries_the_comparability_caveat():
    """A traced wall clock must never be read as a production number."""
    summary = summarize([])
    assert "not comparable" in summary["caveat"]
    assert summary["records"] == 0


def test_summary_of_an_empty_trace_does_not_raise():
    summary = summarize([])
    assert summary["requests_per_second"] is None
    assert summary["overlap"]["samples"] == 0


# ---------------------------------------------------------------------------
# The context manager
# ---------------------------------------------------------------------------


def test_off_by_default_touches_nothing(monkeypatch):
    """The flag is off, so no handler and no CPL_DEBUG."""
    monkeypatch.setattr(settings, "read_trace", False)
    monkeypatch.delenv("CPL_DEBUG", raising=False)
    gdal_log = logging.getLogger(readtrace.GDAL_LOGGER)
    before = list(gdal_log.handlers)

    with read_trace(stage="composite", run_id="r", tile="S30W065", index=16):
        assert "CPL_DEBUG" not in os.environ
        assert gdal_log.handlers == before

    assert "CPL_DEBUG" not in os.environ
    assert gdal_log.handlers == before


def test_on_sets_cpl_debug_and_restores_it(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "read_trace", True)
    monkeypatch.setattr(settings, "read_trace_seconds", 3600.0)
    monkeypatch.setattr(settings, "manifest_dir", tmp_path)
    monkeypatch.delenv("CPL_DEBUG", raising=False)

    with read_trace(stage="composite", run_id="r", tile="S30W065", index=16):
        assert os.environ["CPL_DEBUG"] == "ON"

    assert "CPL_DEBUG" not in os.environ


def test_a_dump_that_fails_does_not_escape_the_block(monkeypatch, tmp_path):
    """Losing the trace costs a diagnostic; failing the block costs the run."""
    monkeypatch.setattr(settings, "read_trace", True)
    monkeypatch.setattr(settings, "read_trace_seconds", 3600.0)
    monkeypatch.setattr(settings, "manifest_dir", tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("no bucket today")

    monkeypatch.setattr(readtrace, "_destination", boom)

    with read_trace(stage="composite", run_id="r", tile="S30W065", index=16):
        pass


def test_a_recorder_that_cannot_start_still_runs_the_block(monkeypatch):
    monkeypatch.setattr(settings, "read_trace", True)

    def boom(_max):
        raise RuntimeError("handler unavailable")

    monkeypatch.setattr(readtrace, "TraceHandler", boom)
    ran = []
    with read_trace(stage="composite", run_id="r", tile="S30W065", index=16):
        ran.append(True)

    assert ran == [True]


def test_writes_both_artifacts_under_the_timings_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "read_trace", True)
    monkeypatch.setattr(settings, "read_trace_seconds", 3600.0)
    monkeypatch.setattr(settings, "manifest_dir", tmp_path)
    monkeypatch.setattr(readtrace, "_thread_config", lambda: {"GDAL_HTTP_VERSION": "1.1"})

    with read_trace(stage="composite", run_id="r", tile="S30W065", index=16):
        gdal_log = logging.getLogger(readtrace.GDAL_LOGGER)
        for handler in gdal_log.handlers:
            if isinstance(handler, TraceHandler):
                emit(handler, f"Downloading 0-16383 ({URL_A})...", created=1.0)
                emit(handler, f"Downloading 65536-81919 ({URL_A})...", created=1.5)

    root = tmp_path / "readtrace"
    written = sorted(p.name for p in root.rglob("*") if p.is_file())
    assert any(name.endswith(".readtrace.summary.json") for name in written)
    assert any(name.endswith(".readtrace.jsonl.gz") for name in written)


def test_the_window_closing_raises_read_trace_complete(monkeypatch, tmp_path):
    """The deliberate stop, so a 1,372 s band costs a diagnostic."""
    monkeypatch.setattr(settings, "read_trace", True)
    monkeypatch.setattr(settings, "read_trace_seconds", 0.05)
    monkeypatch.setattr(settings, "manifest_dir", tmp_path)

    with (
        pytest.raises(ReadTraceComplete),
        read_trace(stage="composite", run_id="r", tile="S30W065", index=16),
    ):
        # The timer interrupts the main thread once the window closes.
        # KeyboardInterrupt is the only exception Python can raise into another
        # thread, so read_trace converts it while the window flag is set.
        for _ in range(200):
            time.sleep(0.01)


def test_a_real_keyboard_interrupt_is_not_swallowed(monkeypatch, tmp_path):
    """An operator's Ctrl-C must stay a Ctrl-C."""
    monkeypatch.setattr(settings, "read_trace", True)
    monkeypatch.setattr(settings, "read_trace_seconds", 3600.0)
    monkeypatch.setattr(settings, "manifest_dir", tmp_path)

    with (
        pytest.raises(KeyboardInterrupt),
        read_trace(stage="composite", run_id="r", tile="S30W065", index=16),
    ):
        raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# The key grammar and the offline re-parse
# ---------------------------------------------------------------------------


def test_trace_prefix_sits_under_the_unit_timings(monkeypatch):
    """evidence._run_artifacts sweeps this prefix; the run prefix it does not."""
    prefix = shards.unit_trace_prefix("run-1", "composite", "S30W065", 16)

    assert prefix.startswith(shards.unit_timing_prefix("run-1"))
    assert prefix.endswith("composite.S30W065.0016")


def test_the_summary_rebuilds_from_the_raw_log(monkeypatch, tmp_path):
    """The raw log is the evidence, so the summary must derive from it."""
    monkeypatch.setattr(settings, "read_trace", True)
    monkeypatch.setattr(settings, "read_trace_seconds", 3600.0)
    monkeypatch.setattr(settings, "manifest_dir", tmp_path)
    monkeypatch.setattr(readtrace, "_thread_config", lambda: {})

    with read_trace(stage="composite", run_id="r", tile="S30W065", index=16):
        gdal_log = logging.getLogger(readtrace.GDAL_LOGGER)
        for handler in gdal_log.handlers:
            if isinstance(handler, TraceHandler):
                emit(handler, f"GetFileSize({URL_A})=1048576", created=1.0)
                emit(handler, f"Downloading 0-16383 ({URL_A})...", created=1.1)
                emit(handler, f"Downloading 65536-81919 ({URL_A})...", created=1.6)

    raw = next((tmp_path / "readtrace").rglob("*.readtrace.jsonl.gz"))
    written = next((tmp_path / "readtrace").rglob("*.readtrace.summary.json"))
    stored = json.loads(written.read_text())

    records = []
    with gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            records.append(
                (e["t"], e["thread"], e["kind"], e["url"], e["start"], e["end"], e["bytes"])
            )

    rebuilt = summarize(records)
    assert rebuilt["first_touch"] == stored["first_touch"]
    assert rebuilt["refetch"] == stored["refetch"]
    assert rebuilt["kinds"] == stored["kinds"]
