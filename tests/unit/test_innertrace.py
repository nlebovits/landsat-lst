"""The inner scheduler of a shard, and the trace that makes its graph visible.

What is pinned here is the evidence chain, not the numbers. A graph file must
hold the dict the scheduler received, an execution file must join every span
back to a node of that graph, "not executed" must carry a scheduler-evidenced
reason or ``unknown``, and the delay before the first dispatch must be
measured from ``dask.compute`` entry rather than from the first span a
scheduler chose to record. The Batch path must never build a Frisky cluster.
"""

from __future__ import annotations

import gzip
import json
import sys
import threading
from typing import TYPE_CHECKING

import dask
import dask.array as da
import numpy as np
import pytest

from landsat_lst import innertrace
from landsat_lst.config import settings
from landsat_lst.innertrace import (
    NOT_EXECUTED_REASONS,
    InnerTrace,
    NullInnerTrace,
    ShardIdentity,
    active_inner_trace,
    add_shard_context,
    current_binding,
    graph_nodes,
    inner_scheduler,
    outer_binding,
    reason_from_story,
)
from landsat_lst.storage import LocalStorage

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit

STEM = "_shards/timings/run/composite.S30W065.0016"


def _identity(**overrides) -> ShardIdentity:
    fields = {
        "run_id": "run",
        "stage": "composite",
        "tile": "S30W065",
        "index": 16,
        "attempt": 1,
        "trace_id": 0x1234,
        "outer_key": "lst-composite-S30W065-0016-abcd1234",
    }
    fields.update(overrides)
    return ShardIdentity(**fields)


def _read_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _artifacts(root: Path, pattern: str) -> list[Path]:
    return sorted((root / "_shards").rglob(pattern))


class FakeClock:
    """Epoch nanoseconds under test control."""

    def __init__(self, start_ns: int = 1_700_000_000_000_000_000) -> None:
        self.now_ns = start_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1e9)


class RecordingSink:
    def __init__(self) -> None:
        self.spans: list[tuple] = []
        self.events: list[tuple] = []

    def span(self, name, t0_ns, t1_ns, *, keys, count, trace_id):
        self.spans.append((name, t0_ns, t1_ns, tuple(keys), count, trace_id))

    def event(self, kind, *, key, metadata):
        self.events.append((kind, key, tuple(metadata)))


def _bounded_compute(trace: InnerTrace, *, groups: int = 2, marker: str | None = None):
    """Two group computes through the observer protocol, on the bound scheduler."""
    observer = trace.group_observer()
    results = []
    with trace.observe_computes():
        for index in range(groups):
            observer("group_start", index, groups, lon_start=index * 4, lon_stop=index * 4 + 4)
            x = da.ones((64, 64), chunks=16, name=f"src-{index}")
            y = (x + index).sum()
            if marker is not None:
                y = da.map_blocks(lambda b: b, y, name=marker)
            observer("store_built", index, groups)
            observer("compute_start", index, groups, n_stores=1)
            error = None
            try:
                results.append(float(dask.compute(y)[0]))
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                observer("compute_end", index, groups, error=error)
    return results


# ---------------------------------------------------------------------------
# The scheduler binding
# ---------------------------------------------------------------------------


class TestInnerScheduler:
    def test_threads_mode_binds_the_threaded_scheduler(self):
        with inner_scheduler(mode="threads", threads=2) as sched:
            assert sched.kind == "threads"
            assert sched.threads == 2
            assert dask.config.get("scheduler") == "threads"
            assert dask.config.get("num_workers") == 2

    def test_an_unknown_mode_is_refused(self):
        with (
            pytest.raises(ValueError, match="unknown inner scheduler"),
            inner_scheduler(mode="nope", threads=1),
        ):  # type: ignore[arg-type]
            pass

    def test_frisky_mode_refuses_without_the_extra(self, monkeypatch):
        """A worker without frisky must say so, not fall back silently."""
        monkeypatch.setitem(sys.modules, "frisky", None)
        with (
            pytest.raises(RuntimeError, match="frisky extra"),
            inner_scheduler(mode="frisky", threads=1),
        ):
            pass

    def test_frisky_mode_serves_a_rest_endpoint_and_a_callable_get(self):
        pytest.importorskip("frisky")
        with inner_scheduler(mode="frisky", threads=2) as sched:
            assert sched.kind == "frisky"
            assert sched.dashboard_url and sched.dashboard_url.startswith("http://127.0.0.1:")
            assert callable(dask.config.get("scheduler"))
            assert float(da.ones((8, 8), chunks=4).sum().compute()) == 64.0


class TestBinding:
    def test_the_binding_is_visible_inside_and_gone_after(self):
        assert current_binding() is None
        with outer_binding(outer_key="k", trace_id=7, inner_scheduler="threads") as binding:
            assert current_binding() is binding
            assert binding.inner_scheduler == "threads"
        assert current_binding() is None


# ---------------------------------------------------------------------------
# Graph capture: the dict the scheduler received, keys and edges
# ---------------------------------------------------------------------------


class TestGraphCapture:
    def test_nodes_carry_class_and_dependency_edges(self):
        x = da.ones((8, 8), chunks=4, name="ones-x")
        expr = dask.base.collections_to_expr([x.sum()], True).optimize()
        from dask._task_spec import convert_legacy_graph

        graph = convert_legacy_graph(expr.__dask_graph__())
        nodes = graph_nodes(graph)

        keys = {n["key"] for n in nodes}
        assert len(nodes) == len(graph)
        for node in nodes:
            assert node["class"] in ("read", "rechunk", "compute", "store", "other")
            for dep in node["deps"]:
                assert dep in keys, "every edge points at a node of the same graph"
        assert any(n["n_dependents"] == 0 for n in nodes), "the output has no dependents"

    @pytest.mark.parametrize("mode", ["threads", "frisky"])
    def test_the_graph_file_is_the_graph_the_scheduler_ran(self, tmp_path, mode):
        """A marker key injected into the collection must be in the file."""
        if mode == "frisky":
            pytest.importorskip("frisky")
        storage = LocalStorage(tmp_path)
        with inner_scheduler(mode=mode, threads=2) as sched:
            trace = InnerTrace(
                identity=_identity(), storage=storage, stem=STEM, scheduler=sched, flush_s=60
            )
            with trace:
                _bounded_compute(trace, groups=1, marker="marker-node")
        graphs = _artifacts(tmp_path, "*inner-graph*g00*")
        assert len(graphs) == 1
        payload = _read_gz(graphs[0])
        names = {n["key"] for n in payload["nodes"]}
        assert any("marker-node" in k for k in names)
        assert payload["n_tasks"] == len(payload["nodes"])
        assert payload["group"] == [0, 1]
        assert payload["capture_s"] >= 0.0

    @pytest.mark.parametrize("mode", ["threads", "frisky"])
    def test_every_executed_key_is_a_graph_node_with_scheduler_state(self, tmp_path, mode):
        if mode == "frisky":
            pytest.importorskip("frisky")
        storage = LocalStorage(tmp_path)
        with inner_scheduler(mode=mode, threads=2) as sched:
            trace = InnerTrace(
                identity=_identity(), storage=storage, stem=STEM, scheduler=sched, flush_s=60
            )
            with trace:
                _bounded_compute(trace, groups=2)
        for index in range(2):
            graph = _read_gz(_artifacts(tmp_path, f"*inner-graph*g0{index}*")[0])
            execution = _read_gz(_artifacts(tmp_path, f"*inner-exec*g0{index}*")[0])
            nodes = {n["key"] for n in graph["nodes"]}
            rows = execution["rows"]
            assert {r["key"] for r in rows} == nodes
            executed = [r for r in rows if r["start_ns"] is not None]
            assert executed, "the group ran tasks"
            for row in executed:
                assert row["end_ns"] >= row["start_ns"]
                assert row["duration_ns"] == row["end_ns"] - row["start_ns"]
                assert row["state"] == "Memory"
            for row in rows:
                if row["start_ns"] is None:
                    assert row["not_executed_reason"] in NOT_EXECUTED_REASONS
                    if row["not_executed_reason"] == "ready_not_started":
                        pytest.fail("readiness was claimed without a story event")
            assert execution["n_executed"] == len(executed)

    def test_construction_is_measured_from_compute_entry(self, tmp_path):
        """The delay before first dispatch starts when dask.compute is called."""
        pytest.importorskip("frisky")
        storage = LocalStorage(tmp_path)
        with inner_scheduler(mode="frisky", threads=2) as sched:
            trace = InnerTrace(
                identity=_identity(), storage=storage, stem=STEM, scheduler=sched, flush_s=60
            )
            with trace:
                _bounded_compute(trace, groups=1)
        c = _read_gz(_artifacts(tmp_path, "*inner-exec*g00*")[0])["construction"]
        assert c["compute_entry_ns"] < c["get_entry_ns"] <= c["graph_captured_ns"]
        assert c["first_dispatch_ns"] is not None
        assert c["construction_s"] == pytest.approx(
            (c["first_dispatch_ns"] - c["compute_entry_ns"]) / 1e9, abs=1e-3
        )
        assert c["pre_scheduler_s"] > 0.0, "dask's own work before the scheduler is visible"
        assert c["capture_overhead_s"] >= 0.0
        assert c["translate_s"] is not None and c["submit_s"] is not None
        # The sequential main-thread stages sum to the wall between entry and
        # capture; what follows is submission, which overlaps with dispatch.
        sequential = c["pre_scheduler_s"] + c["materialize_s"]
        assert sequential <= c["construction_s"] + 1e-3

    def test_construction_under_threads_uses_the_callback_boundaries(self, tmp_path):
        storage = LocalStorage(tmp_path)
        with inner_scheduler(mode="threads", threads=2) as sched:
            trace = InnerTrace(
                identity=_identity(), storage=storage, stem=STEM, scheduler=sched, flush_s=60
            )
            with trace:
                _bounded_compute(trace, groups=1)
        c = _read_gz(_artifacts(tmp_path, "*inner-exec*g00*")[0])["construction"]
        assert c["construction_s"] is not None and c["construction_s"] >= 0.0
        assert c["pre_scheduler_s"] is not None and c["pre_scheduler_s"] >= 0.0
        assert c.get("translate_s") is None


# ---------------------------------------------------------------------------
# Readiness is a scheduler claim
# ---------------------------------------------------------------------------


class TestNotExecutedReasons:
    def test_no_story_is_unknown(self):
        assert reason_from_story(None, executed=False) == "unknown"
        assert reason_from_story([], executed=False) == "unknown"

    def test_a_placement_without_execution_is_ready_not_started(self):
        events = [
            {
                "event_type": "transition",
                "details": {"from_state": "Released", "to_state": "Waiting"},
            },
            {"event_type": "placed", "details": {"worker": "w", "reason": "no_deps"}},
        ]
        assert reason_from_story(events, executed=False) == "ready_not_started"

    def test_a_key_the_scheduler_finished_is_not_called_unstarted(self):
        events = [
            {
                "event_type": "transition",
                "details": {"from_state": "Waiting", "to_state": "Processing"},
            },
            {
                "event_type": "transition",
                "details": {"from_state": "Processing", "to_state": "Memory"},
            },
            {
                "event_type": "transition",
                "details": {"from_state": "Memory", "to_state": "Released"},
            },
        ]
        assert reason_from_story(events, executed=False) == "completed_without_exec_span"

    def test_waiting_only_is_blocked_on_deps(self):
        events = [
            {
                "event_type": "transition",
                "details": {"from_state": "Released", "to_state": "Waiting"},
            },
        ]
        assert reason_from_story(events, executed=False) == "blocked_on_deps"

    def test_a_release_with_no_placement_is_released(self):
        events = [
            {
                "event_type": "transition",
                "details": {"from_state": "Waiting", "to_state": "Released"},
            },
        ]
        assert reason_from_story(events, executed=False) == "released"

    def test_an_error_wins(self):
        events = [
            {"event_type": "placed", "details": {}},
            {
                "event_type": "transition",
                "details": {"from_state": "Processing", "to_state": "Erred"},
            },
        ]
        assert reason_from_story(events, executed=False) == "erred"


# ---------------------------------------------------------------------------
# Sections, flushes, and the artifacts that survive an interruption
# ---------------------------------------------------------------------------


class TestSectionsAndFlush:
    def test_sections_nest_and_close_with_durations(self, tmp_path):
        storage = LocalStorage(tmp_path)
        clock = FakeClock()
        with inner_scheduler(mode="threads", threads=1) as sched:
            trace = InnerTrace(
                identity=_identity(),
                storage=storage,
                stem=STEM,
                scheduler=sched,
                clock=clock,
                flush_s=60,
            )
            with trace:
                with trace.section("composite_graph"):
                    clock.advance(2.5)
                with trace.section("upload", product="lst_p95"):
                    clock.advance(1.0)
        final = json.loads((tmp_path / f"{STEM}.inner.a01.final.json").read_text())
        by_name = {s["name"]: s for s in final["sections"]}
        assert by_name["composite_graph"]["s"] == pytest.approx(2.5)
        assert by_name["upload"]["s"] == pytest.approx(1.0)
        assert by_name["upload"]["meta"]["product"] == "lst_p95"
        assert final["closed_by"] == "exit"
        assert final["error"] is None

    def test_an_exception_closes_the_open_section_with_the_error(self, tmp_path):
        storage = LocalStorage(tmp_path)
        with inner_scheduler(mode="threads", threads=1) as sched:
            trace = InnerTrace(
                identity=_identity(), storage=storage, stem=STEM, scheduler=sched, flush_s=60
            )
            with pytest.raises(RuntimeError, match="boom"), trace, trace.section("encode"):
                raise RuntimeError("boom")
        final = json.loads((tmp_path / f"{STEM}.inner.a01.final.json").read_text())
        assert final["closed_by"] == "error"
        assert "boom" in final["error"]
        assert final["sections"][0]["error"] == "RuntimeError: boom"

    def test_the_periodic_flush_rewrites_progress(self, tmp_path):
        storage = LocalStorage(tmp_path)
        with inner_scheduler(mode="threads", threads=1) as sched:
            trace = InnerTrace(
                identity=_identity(), storage=storage, stem=STEM, scheduler=sched, flush_s=0.05
            )
            with trace, trace.section("composite_graph"):
                import time

                time.sleep(0.3)
                progress = json.loads((tmp_path / f"{STEM}.inner-progress.json").read_text())
        assert progress["seq"] >= 2
        assert progress["final"] is False
        assert progress["scheduler"] == "threads"
        assert "tree_rss_mb" in progress["host_rates"]

    def test_span_chunks_land_before_the_shard_ends(self, tmp_path):
        pytest.importorskip("frisky")
        storage = LocalStorage(tmp_path)
        with inner_scheduler(mode="frisky", threads=2) as sched:
            trace = InnerTrace(
                identity=_identity(), storage=storage, stem=STEM, scheduler=sched, flush_s=60
            )
            with trace:
                _bounded_compute(trace, groups=1)
                chunks_during = _artifacts(tmp_path, "*inner-spans*")
                assert chunks_during, "a group close flushes its spans before the shard exits"
        chunk = _read_gz(chunks_during[0])
        assert chunk["count"] == len(chunk["spans"]) > 0
        assert chunk["run_id"] == "run"
        assert any(s["name"] == "worker.exec.call" for s in chunk["spans"])

    def test_a_storage_failure_never_raises(self, tmp_path):
        class Broken(LocalStorage):
            def write_text(self, key, text, *, content_type="application/json"):
                raise OSError("bucket gone")

            def upload(self, local, key):
                raise OSError("bucket gone")

        storage = Broken(tmp_path)
        with inner_scheduler(mode="threads", threads=1) as sched:
            trace = InnerTrace(
                identity=_identity(), storage=storage, stem=STEM, scheduler=sched, flush_s=60
            )
            with trace:
                _bounded_compute(trace, groups=1)
        assert any("bucket gone" in e for e in trace.snapshot()["errors"])

    def test_heartbeat_fields_are_bounded_and_name_open_sections(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "inner_trace_heartbeat_sections", 3)
        storage = LocalStorage(tmp_path)
        with inner_scheduler(mode="threads", threads=1) as sched:
            trace = InnerTrace(
                identity=_identity(), storage=storage, stem=STEM, scheduler=sched, flush_s=60
            )
            with trace:
                for i in range(6):
                    with trace.section(f"s{i}"):
                        pass
                with trace.section("open"):
                    fields = trace.heartbeat_fields()
        assert len(fields["sections"]) == 3
        assert [s["name"] for s in fields["open_sections"]] == ["open"]
        assert fields["trace_id"] == hex(0x1234)
        assert fields["outer_key"] == _identity().outer_key


# ---------------------------------------------------------------------------
# Forwarding to the outer scheduler and the log context
# ---------------------------------------------------------------------------


class TestForwarding:
    def test_sections_reach_the_sink_keyed_by_the_outer_key(self, tmp_path):
        storage = LocalStorage(tmp_path)
        sink = RecordingSink()
        clock = FakeClock()
        with inner_scheduler(mode="threads", threads=1) as sched:
            trace = InnerTrace(
                identity=_identity(),
                storage=storage,
                stem=STEM,
                scheduler=sched,
                sink=sink,
                clock=clock,
                flush_s=60,
            )
            with trace, trace.section("encode"):
                clock.advance(0.5)
        names = [s[0] for s in sink.spans]
        assert "shard.encode" in names
        span = next(s for s in sink.spans if s[0] == "shard.encode")
        assert span[3] == (_identity().outer_key,)
        assert span[2] - span[1] == int(0.5e9)
        assert span[5] == 0x1234
        assert any(e[0] == "shard.progress" for e in sink.events)

    def test_a_raising_sink_never_fails_the_shard(self, tmp_path):
        class Exploding:
            def span(self, *a, **k):
                raise RuntimeError("no scheduler")

            def event(self, *a, **k):
                raise RuntimeError("no scheduler")

        storage = LocalStorage(tmp_path)
        with inner_scheduler(mode="threads", threads=1) as sched:
            trace = InnerTrace(
                identity=_identity(),
                storage=storage,
                stem=STEM,
                scheduler=sched,
                sink=Exploding(),
                flush_s=60,
            )
            with trace, trace.section("encode"):
                pass
        assert any("no scheduler" in e for e in trace.snapshot()["errors"])

    def test_the_log_processor_stamps_foreign_threads(self, tmp_path):
        storage = LocalStorage(tmp_path)
        seen: dict = {}

        def worker():
            seen.update(add_shard_context(None, "info", {"event": "x"}))

        with inner_scheduler(mode="threads", threads=1) as sched:
            trace = InnerTrace(
                identity=_identity(), storage=storage, stem=STEM, scheduler=sched, flush_s=60
            )
            with trace:
                assert active_inner_trace() is trace
                thread = threading.Thread(target=worker)
                thread.start()
                thread.join()
        assert seen["trace_id"] == hex(0x1234)
        assert seen["shard"] == "composite.0016"
        assert innertrace.CURRENT is None
        assert add_shard_context(None, "info", {"event": "y"}) == {"event": "y"}


class TestNullTrace:
    def test_the_null_trace_has_the_same_surface(self):
        null = NullInnerTrace()
        with null.section("anything"), null.observe_computes():
            pass
        assert null.group_observer() is None
        assert "error" in null.heartbeat_fields()


def test_pixels_do_not_depend_on_the_inner_scheduler(tmp_path):
    """Frisky and threads run the same tasks; the result must be the same."""
    pytest.importorskip("frisky")
    results = {}
    for mode in ("threads", "frisky"):
        with inner_scheduler(mode=mode, threads=2):
            x = da.random.default_rng(seed=7).random((64, 64), chunks=16)
            results[mode] = (x * 3).sum(axis=0).compute()
    np.testing.assert_array_equal(results["threads"], results["frisky"])
