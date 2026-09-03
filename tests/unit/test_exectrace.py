"""The composite execution trace records cheaply and publishes only afterwards."""

from __future__ import annotations

import csv
import gzip
import json
from contextlib import nullcontext
from types import SimpleNamespace

import dask.array as da
import pytest

from landsat_lst.config import settings
from landsat_lst.exectrace import build_timeline, classify, exec_trace
from landsat_lst.storage import LocalStorage

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("keixel", "read"),
        ("lwir11", "read"),
        ("qa_pixel", "read"),
        ("open-rasterio", "read"),
        ("cfg-load", "read"),
        ("grid", "read"),
        ("rechunk-merge", "rechunk"),
        ("transpose", "rechunk"),
        ("getitem", "rechunk"),
        ("shuffle-taker", "rechunk"),
        ("shuffle-split", "rechunk"),
        ("rechunk-split-rechunk-merge", "rechunk"),
        ("where", "compute"),
        ("sub", "compute"),
        ("astype", "compute"),
        ("invert", "compute"),
        ("custom_nanquantile-transpose-getitem", "compute"),
        ("nanquantile_last-transpose", "compute"),
        ("nanquantile_last_0", "compute"),
        ("getitem-sum-sum-aggregate-stack", "compute"),
        ("getitem-stack-sum-sum-aggregate-astype", "compute"),
        ("store", "store"),
        ("store-map", "store"),
        ("finalize", "other"),
    ],
)
def test_classify_real_composite_prefixes(prefix, expected):
    assert classify(prefix) == expected


def test_build_timeline_counts_half_open_task_intervals_and_reads():
    host = [
        {
            "timestamp": stamp,
            "cpu_cores_busy": 1.0,
            "rss_mb": 100.0,
            "network_recv_mb_s": 2.0,
            "network_send_mb_s": 3.0,
            "disk_read_mb_s": 4.0,
            "disk_write_mb_s": 5.0,
            "num_fds": 6.0,
        }
        for stamp in (1.0, 2.0, 3.0)
    ]
    tasks = [
        {"start": 0.5, "end": 1.5, "class": "read"},
        {"start": 1.5, "end": 3.0, "class": "compute"},
        {"start": 2.5, "end": 4.0, "class": "store"},
    ]
    reads = [
        {"t_start": 0.75, "t_end": 1.25},
        {"t_start": 2.5, "t_end": 3.5},
    ]

    rows = build_timeline(host, tasks, reads)

    assert [row["active_read_tasks"] for row in rows] == [1, 0, 0]
    assert [row["active_compute_tasks"] for row in rows] == [0, 1, 0]
    assert [row["active_store_tasks"] for row in rows] == [0, 0, 1]
    assert [row["sources_started"] for row in rows] == [1, 1, 2]
    assert [row["sources_finished"] for row in rows] == [0, 1, 1]


def test_exec_trace_writes_three_artifacts_after_compute_and_restores_hook(monkeypatch, tmp_path):
    import odc.loader._rio as rio

    storage = LocalStorage(tmp_path)
    monkeypatch.setattr(settings, "exec_trace", True)
    monkeypatch.setattr(settings, "exec_trace_interval_s", 0.01)
    monkeypatch.setattr(settings, "exec_trace_read_sample", 1)

    def fake_do_read(*args, **kwargs):
        return nullcontext(), None

    def fake_read(src, cfg, dst_geobox, *args, **kwargs):
        # Mirrors _rio_read: the open happens here, then the module-global
        # _do_read is called by name, which is what the split hook relies on.
        return rio._do_read(None, cfg, dst_geobox, None)

    monkeypatch.setattr(rio, "rio_read", fake_read)
    monkeypatch.setattr(rio, "_do_read", fake_do_read)
    stem = "_shards/timings/run/composite.S30W065.0016"
    with exec_trace(storage=storage, stem=stem):
        rio.rio_read(
            SimpleNamespace(uri="https://example.test/scene_B10.tif"),
            None,
            SimpleNamespace(shape=(32, 64)),
        )
        assert da.ones((64, 64), chunks=32).sum().compute(scheduler="threads") == 4096

    assert rio.rio_read is fake_read
    assert rio._do_read is fake_do_read
    artifacts = sorted((tmp_path / "_shards" / "timings" / "run").iterdir())
    assert [path.name for path in artifacts] == [
        "composite.S30W065.0016.exectrace.events.jsonl.gz",
        "composite.S30W065.0016.exectrace.summary.json",
        "composite.S30W065.0016.exectrace.timeline.csv",
    ]

    summary = json.loads(artifacts[1].read_text())
    assert summary["n_tasks"] > 0
    assert summary["n_reads_total"] == summary["n_reads_recorded"] == 1
    assert summary["read_hook_status"] == "enabled"
    assert summary["read_split_recorded"] == 1
    assert summary["read_open_s"]["n"] == summary["read_data_s"]["n"] == 1
    assert summary["read_open_s"]["p50"] >= 0.0
    assert summary["read_data_s"]["p50"] >= 0.0
    assert summary["host_sampler_mode"] in {"process", "thread"}
    assert summary["host_samples"] >= 2
    assert summary["upload_started_at"] >= summary["compute_finished_at"]

    with gzip.open(artifacts[0], "rt") as stream:
        events = [json.loads(line) for line in stream]
    task_events = [event for event in events if event["kind"] == "task"]
    assert task_events
    assert all(event["end"] <= summary["compute_finished_at"] for event in task_events)
    assert any(event["kind"] == "host" for event in events)
    read_events = [event for event in events if event["kind"] == "read"]
    assert len(read_events) == 1
    assert read_events[0]["t_start"] <= read_events[0]["t_open_end"] <= read_events[0]["t_end"]
    assert read_events[0]["open_s"] is not None and read_events[0]["read_s"] is not None

    with artifacts[2].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert float(rows[0]["timestamp"]) <= summary["compute_finished_at"]


def test_exec_trace_is_inert_when_disabled(monkeypatch, tmp_path):
    storage = LocalStorage(tmp_path)
    monkeypatch.setattr(settings, "exec_trace", False)

    with exec_trace(storage=storage, stem="must-not-exist"):
        assert da.ones(2).sum().compute() == 2

    assert list(tmp_path.iterdir()) == []
