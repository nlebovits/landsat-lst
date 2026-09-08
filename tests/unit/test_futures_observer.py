"""The outer view: canned scheduler state in, one normalized snapshot out.

No scheduler, no cluster. Frisky's ``get_scheduler_state`` and distributed's
``scheduler_info`` shapes are fed as dicts, the registry is filled by hand,
and what is pinned is the state mapping, the diff, the persisted keys, the
span dump's cursor, the verification file's honesty (checks fail or are
``null``, never silently absent), and the gate's reading of the bucket.
"""

from __future__ import annotations

import json

import pytest

from landsat_lst import shards
from landsat_lst.config import settings
from landsat_lst.futures_driver import ShardRef
from landsat_lst.futures_observer import (
    FuturesObserver,
    ShardRegistry,
    diff_snapshots,
    normalize_dask_state,
    normalize_frisky_state,
    observability_gate,
)
from landsat_lst.storage import LocalStorage

pytestmark = pytest.mark.unit

RUN = "run-x"
TILE = "S30W065"
ROOT = shards.shard_root(RUN, TILE)


def _ref(stage: str, index: int, deps: tuple[str, ...] = ()) -> ShardRef:
    key = f"lst-{stage}-{TILE}-{index:04d}-tok"
    return ShardRef(
        key=key,
        stage=stage,
        tile=TILE,
        index=index,
        depends_on=deps,
        submitted_at=100.0,
        state_key=shards.shard_state_key(ROOT, stage, index, 1),
        log_key=shards.shard_log_key(ROOT, stage, index, 1),
        artifact_keys=tuple(shards.band_key(ROOT, p, index) for p in ("lst_p95", "qa_count"))
        if stage == "composite"
        else (),
    )


@pytest.fixture
def registry():
    reg = ShardRegistry()
    reg.register(_ref("merge", 0))
    reg.register(_ref("composite", 0, (_ref("merge", 0).key,)))
    reg.register(_ref("composite", 1, (_ref("merge", 0).key,)))
    reg.register(_ref("export", 0, (_ref("composite", 0).key, _ref("composite", 1).key)))
    return reg


class TestFriskyNormalizer:
    def test_states_map_from_the_scheduler_and_the_registry(self, registry):
        merge, c0, c1, export = (
            _ref("merge", 0).key,
            _ref("composite", 0).key,
            _ref("composite", 1).key,
            _ref("export", 0).key,
        )
        state = {
            "tasks": [
                {"key": merge, "state": "Memory", "who_has": ["w1"], "dependencies": []},
                {"key": c0, "state": "Processing", "processing_on": "w1", "dependencies": [merge]},
                {"key": c1, "state": "Waiting", "dependencies": [merge]},
                {"key": export, "state": "Waiting", "dependencies": [c0, c1]},
            ],
            "workers": [
                {
                    "address": "w1",
                    "cpu_percent": 50.0,
                    "memory_bytes": 2 * 1048576,
                    "processing_keys": [c0],
                }
            ],
        }
        registry.note_completion(merge, result={"ok": True}, error=None)
        snap = normalize_frisky_state(state, registry, ts=1.0, seq=1, run_id=RUN, tile=TILE)
        assert snap.shards[merge].state == "done"
        assert snap.shards[c0].state == "running" and snap.shards[c0].worker == "w1"
        assert snap.shards[c1].state == "pending", "its one dependency is in memory"
        assert snap.shards[export].state == "blocked", "a dependency is not in memory"
        assert snap.workers["w1"].memory_mb == 2.0 and snap.workers["w1"].executing == [c0]
        assert snap.counts()["running"] == 1

    def test_a_retried_or_stalled_key_is_named(self, registry):
        c1 = _ref("composite", 1).key
        registry.note_retry(c1)
        state = {"tasks": [{"key": c1, "state": "Waiting", "dependencies": []}], "workers": []}
        snap = normalize_frisky_state(state, registry, ts=1.0, seq=1, run_id=RUN, tile=TILE)
        assert snap.shards[c1].state == "retried"
        registry.note_stalled([c1])
        snap = normalize_frisky_state(state, registry, ts=2.0, seq=2, run_id=RUN, tile=TILE)
        assert snap.shards[c1].state == "stalled"

    def test_a_driver_recorded_error_wins_over_scheduler_state(self, registry):
        c0 = _ref("composite", 0).key
        registry.note_completion(c0, result=None, error="boom")
        state = {
            "tasks": [{"key": c0, "state": "Processing", "processing_on": "w1"}],
            "workers": [],
        }
        snap = normalize_frisky_state(state, registry, ts=1.0, seq=1, run_id=RUN, tile=TILE)
        assert snap.shards[c0].state == "failed" and snap.shards[c0].error == "boom"


class TestDaskNormalizer:
    def test_states_map_from_future_status_and_processing(self, registry):
        merge, c0, c1, export = (
            _ref("merge", 0).key,
            _ref("composite", 0).key,
            _ref("composite", 1).key,
            _ref("export", 0).key,
        )
        info = {
            "workers": {
                "tcp://w1": {
                    "metrics": {"cpu": 40.0, "memory": 3 * 1048576},
                    "memory_limit": 64 * 1048576,
                }
            }
        }
        snap = normalize_dask_state(
            info,
            {merge: "finished", c0: "pending", c1: "pending", export: "pending"},
            {"tcp://w1": [c0]},
            {merge: ["tcp://w1"]},
            registry,
            ts=1.0,
            seq=1,
            run_id=RUN,
            tile=TILE,
        )
        assert snap.shards[merge].state == "done" and snap.shards[merge].worker == "tcp://w1"
        assert snap.shards[c0].state == "running"
        assert snap.shards[c1].state == "pending"
        assert snap.shards[export].state == "blocked"
        assert snap.workers["tcp://w1"].memory_limit_mb == 64.0
        assert snap.source == "dask"

    def test_a_lost_future_is_failed(self, registry):
        c0 = _ref("composite", 0).key
        snap = normalize_dask_state(
            {"workers": {}}, {c0: "lost"}, {}, {}, registry, ts=1.0, seq=1, run_id=RUN, tile=TILE
        )
        assert snap.shards[c0].state == "failed" and snap.shards[c0].error == "lost"


class TestDiff:
    def test_transitions_and_worker_loss_are_events(self, registry):
        c0 = _ref("composite", 0).key
        a = normalize_frisky_state(
            {
                "tasks": [{"key": c0, "state": "Processing", "processing_on": "w1"}],
                "workers": ["w1", "w2"],
            },
            registry,
            ts=1.0,
            seq=1,
            run_id=RUN,
            tile=TILE,
        )
        b = normalize_frisky_state(
            {"tasks": [{"key": c0, "state": "Memory", "who_has": ["w1"]}], "workers": ["w1"]},
            registry,
            ts=2.0,
            seq=2,
            run_id=RUN,
            tile=TILE,
        )
        first = diff_snapshots(None, a)
        assert {e["kind"] for e in first} == {"transition", "worker_joined"}
        second = diff_snapshots(a, b)
        kinds = [(e["kind"], e.get("key") or e.get("address")) for e in second]
        assert ("transition", c0) in kinds
        assert ("worker_lost", "w2") in kinds
        assert b.lost_workers == ["w2"]


def _observer(tmp_path, registry, *, mode="frisky", spans=None, story=None):
    storage = LocalStorage(tmp_path)
    calls = {"spans": 0}

    def state_reader(reg, ts, seq):
        keys = list(reg.refs())
        return normalize_frisky_state(
            {
                "tasks": [{"key": k, "state": "Memory", "who_has": ["w1"]} for k in keys],
                "workers": ["w1"],
            },
            reg,
            ts=ts,
            seq=seq,
            run_id=RUN,
            tile=TILE,
        )

    def span_query(**kwargs):
        calls["spans"] += 1
        return list(spans or [])

    observer = FuturesObserver(
        run_id=RUN,
        tile=TILE,
        storage=storage,
        registry=registry,
        state_reader=state_reader,
        mode=mode,
        dashboard_url="http://x" if mode == "frisky" else None,
        span_query=span_query if spans is not None else None,
        story=story,
        poll_s=60,
        clock=lambda: 1234.0,
    )
    return observer, storage, calls


class TestPersistence:
    def test_a_poll_writes_the_snapshot_and_an_event_chunk(self, tmp_path, registry):
        observer, storage, _ = _observer(tmp_path, registry)
        observer.poll_once()
        snapshot = json.loads(storage.read_text(f"{ROOT}/state/orchestration.json"))
        assert snapshot["seq"] == 1 and snapshot["counts"]["done"] == 4
        events = storage.read_text(f"{ROOT}/state/orchestration.events.0001.jsonl")
        assert events and all(
            json.loads(line)["kind"] in ("transition", "worker_joined")
            for line in events.splitlines()
        )
        observer.poll_once()
        assert storage.read_text(f"{ROOT}/state/orchestration.events.0002.jsonl") is None, (
            "no change, no chunk"
        )

    def test_span_dumps_are_incremental_and_deduplicated(self, tmp_path, registry):
        spans = [
            {"span_id": 1, "name": "worker.exec.call", "start_ns": 10, "keys": ["k"]},
            {"span_id": 2, "name": "worker.exec.call", "start_ns": 20, "keys": ["k"]},
        ]
        observer, storage, _ = _observer(tmp_path, registry, spans=spans)
        first = observer.dump_spans()
        assert first is not None and json.loads(storage.read_text(first))["count"] == 2
        assert observer.dump_spans() is None, "the same spans again are not a new chunk"
        spans.append({"span_id": 3, "name": "worker.exec.call", "start_ns": 20, "keys": ["k"]})
        second = observer.dump_spans()
        assert second is not None and json.loads(storage.read_text(second))["count"] == 1

    def test_verification_records_pass_fail_and_never_raises(self, tmp_path, registry):
        c0 = _ref("composite", 0).key
        spans = [{"span_id": 1, "name": "shard.group", "start_ns": 5, "keys": [c0]}]

        def story(key):
            raise RuntimeError("story endpoint down")

        observer, storage, _ = _observer(tmp_path, registry, spans=spans, story=story)
        observer.poll_once()
        verification = observer.verify_frisky(phase="submitted")
        names = {c["name"]: c for c in verification["checks"]}
        assert names["scheduler_state_lists_submitted_keys"]["passed"] is True
        assert names["inner_spans_keyed_by_outer_key"]["passed"] is False, (
            "only one of four keys has a span"
        )
        assert names["span_buffer_not_saturated"]["passed"] is True
        assert verification["passed"] is False
        assert "inner_spans_keyed_by_outer_key" in verification["failures"]
        assert storage.read_text(f"{shards.unit_timing_prefix(RUN)}frisky-verification.json")

    def test_a_plain_dask_run_marks_frisky_checks_not_applicable(self, tmp_path, registry):
        observer, _storage, _ = _observer(tmp_path, registry, mode="dask")
        verification = observer.verify_frisky(phase="complete")
        assert verification["checks"][0]["passed"] is None
        assert verification["passed"] is False


class TestGate:
    def _write_inner(
        self, storage, index, *, error=None, chunks=("c1",), groups=1, sink_error=None
    ):
        stem = shards.unit_trace_prefix(RUN, "composite", TILE, index)
        storage.write_text(
            f"{stem}.inner.a01.final.json",
            json.dumps({"error": error, "scheduler": "frisky", "chunk_keys": list(chunks)}),
        )
        for g in range(groups):
            storage.write_text(f"{stem}.inner-graph.a01.g{g:02d}.json.gz", "x")
            storage.write_text(f"{stem}.inner-exec.a01.g{g:02d}.json.gz", "x")
        storage.write_text(
            shards.shard_state_key(ROOT, "composite", index, 1),
            json.dumps({"inner": {"sink_error": sink_error}}),
        )

    def test_the_gate_passes_only_with_every_artifact(self, tmp_path, registry):
        storage = LocalStorage(tmp_path)
        final = {
            "run_id": RUN,
            "mode": "frisky",
            "verification": {"failures": []},
            "span_chunks": ["s1"],
        }
        gate = observability_gate(final, registry, storage)
        assert gate["passed"] is False and any("no inner final" in f for f in gate["failures"])
        self._write_inner(storage, 0)
        self._write_inner(storage, 1)
        gate = observability_gate(final, registry, storage)
        assert gate["passed"] is True and gate["composite_shards"] == 2

    def test_a_failed_sink_or_a_missing_exec_file_fails_the_gate(self, tmp_path, registry):
        storage = LocalStorage(tmp_path)
        final = {
            "run_id": RUN,
            "mode": "frisky",
            "verification": {"failures": []},
            "span_chunks": ["s1"],
        }
        self._write_inner(storage, 0, sink_error="no scheduler")
        self._write_inner(storage, 1)
        gate = observability_gate(final, registry, storage)
        assert any("sink failed" in f for f in gate["failures"])
        stem = shards.unit_trace_prefix(RUN, "composite", TILE, 1)
        (tmp_path / f"{stem}.inner-exec.a01.g00.json.gz").unlink()
        gate = observability_gate(final, registry, storage)
        assert any("graph files 1 vs exec files 0" in f for f in gate["failures"])

    def test_verification_failures_and_missing_span_chunks_fail_the_gate(self, tmp_path, registry):
        storage = LocalStorage(tmp_path)
        self._write_inner(storage, 0)
        self._write_inner(storage, 1)
        final = {
            "run_id": RUN,
            "mode": "frisky",
            "verification": {"failures": ["x"]},
            "span_chunks": [],
        }
        gate = observability_gate(final, registry, storage)
        assert "frisky verification failed: ['x']" in gate["failures"]
        assert "no frisky span chunks persisted" in gate["failures"]


def test_settings_defaults_bound_every_stage():
    assert settings.futures_max_workers >= 1
    assert settings.shard_executor == "batch", "Batch stays the default until acceptance"
