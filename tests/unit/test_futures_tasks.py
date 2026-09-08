"""The wrapper one future runs: binding, scheduler choice, and a joinable result."""

from __future__ import annotations

import sys

import pytest

from landsat_lst import futures_tasks, shard_tasks
from landsat_lst.futures_tasks import ShardResult, run_shard_task, session_token, task_key
from landsat_lst.innertrace import current_binding

pytestmark = pytest.mark.unit


class TestTaskKey:
    def test_the_key_carries_run_stage_tile_index_and_token(self):
        key = task_key("shard-S30W065-2021-2025-x", "S30W065", "composite", 16, "abcd1234")
        assert key.startswith("lst-")
        assert key.endswith("-composite-S30W065-0016-abcd1234")

    def test_two_runs_never_share_a_key(self):
        assert task_key("run-a", "T", "composite", 0, "t") != task_key(
            "run-b", "T", "composite", 0, "t"
        )

    def test_session_tokens_differ(self):
        assert session_token() != session_token()


class TestRunShardTask:
    def test_the_binding_names_the_scheduler_the_key_and_the_trace_id(self, monkeypatch):
        seen: dict = {}

        def fake_run_shard(stage, run_id, tile, index, *, job=None, units=None, storage=None):
            binding = current_binding()
            seen["binding"] = binding
            seen["args"] = (stage, run_id, tile, index, job, units, storage)
            return ["a/key"]

        monkeypatch.setattr(shard_tasks, "run_shard", fake_run_shard)
        result = run_shard_task(
            "composite",
            "run",
            "S30W065",
            3,
            "dep-value-ignored",
            outer_key="lst-k",
            trace_id=0xABC,
            inner_scheduler="threads",
            use_frisky_sink=False,
            storage="fake-storage",
        )
        assert isinstance(result, ShardResult)
        assert seen["binding"].outer_key == "lst-k"
        assert seen["binding"].trace_id == 0xABC
        assert seen["binding"].inner_scheduler == "threads"
        assert seen["binding"].sink is None
        assert seen["args"] == ("composite", "run", "S30W065", 3, None, None, "fake-storage")
        assert result.keys == ["a/key"] and result.skipped is False
        assert result.trace_id == hex(0xABC)
        assert result.inner_scheduler == "threads"
        assert current_binding() is None, "the binding does not leak out of the task"

    def test_an_empty_result_is_a_skipped_shard(self, monkeypatch):
        monkeypatch.setattr(shard_tasks, "run_shard", lambda *_a, **_k: [])
        result = run_shard_task("composite", "run", "T", 0, use_frisky_sink=False)
        assert result.skipped is True and result.keys == []

    def test_a_missing_frisky_is_recorded_not_raised(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "frisky", None)
        monkeypatch.setattr(shard_tasks, "run_shard", lambda *_a, **_k: ["k"])
        result = run_shard_task("composite", "run", "T", 0, inner_scheduler="threads")
        assert result.sink_error and "frisky" in result.sink_error.lower()
        assert result.keys == ["k"]

    def test_the_merge_stage_calls_merge_offsets(self, monkeypatch):
        class Key:
            storage_key = "_offsets/x.json"

        calls: list = []
        monkeypatch.setattr(shard_tasks, "configure_logging", lambda: calls.append("log"))
        monkeypatch.setattr(shard_tasks, "apply_shard_settings", lambda: calls.append("settings"))
        monkeypatch.setattr(shard_tasks, "merge_offsets", lambda *_a, **_k: Key())
        result = run_shard_task("merge", "run", "T", 0, use_frisky_sink=False)
        assert result.keys == ["_offsets/x.json"]
        assert calls == ["log", "settings"]

    def test_the_result_serializes(self, monkeypatch):
        monkeypatch.setattr(shard_tasks, "run_shard", lambda *_a, **_k: ["k"])
        result = run_shard_task("composite", "run", "T", 0, use_frisky_sink=False)
        payload = result.as_dict()
        assert payload["stage"] == "composite" and payload["pid"] > 0
        assert set(payload) >= {"wall_s", "peak_rss_gb", "hostname", "trace_id", "sink_error"}


def test_the_wrapper_imports_no_backend_at_module_level():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(futures_tasks.__file__).read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert not top & {"coiled", "frisky", "distributed"}
