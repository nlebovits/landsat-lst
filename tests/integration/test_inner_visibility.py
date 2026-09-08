"""Demonstration 1 of issue #155, pinned: one inner task, seen end to end.

Runs ``scripts/inner_visibility_demo.py`` as a subprocess (Frisky's
subprocess workers need a real ``__main__``) and checks the evidence chain it
leaves behind: the graph and execution files per group, the explanation of a
selected task with its dependency tree and its decomposed pre-execution
delay, the ``shard.*`` spans at the outer scheduler, and pixel identity
between the Frisky and threaded inner schedulers.

No network, no credentials. Skips when the ``frisky`` extra is absent.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("frisky")

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "inner_visibility_demo.py"


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("inner-visibility")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--scenes", "12", "--shape", "128", "256"],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "DEMONSTRATION 1: PASS" in result.stdout
    return out


def _read_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def test_the_selected_task_is_explained_with_its_tree_and_its_delay(demo):
    text = (demo / "explain-task.txt").read_text()
    assert "delay before this task, from dask.compute entry:" in text
    assert "construction_total (entry -> first dispatch)" in text
    assert "entry -> this key started" in text
    assert "dependencies (depth 4):" in text
    assert "├─" in text or "└─" in text, "a dependency tree was rendered"
    assert "ran " in text, "execution times are on the tree"
    assert "top prefixes by summed task seconds" in text


def test_every_group_left_a_graph_and_an_execution_file_that_join(demo):
    timings = demo / "bucket-frisky" / "_shards" / "timings" / "inner-visibility-demo"
    graphs = sorted(timings.glob("*inner-graph*.json.gz"))
    execs = sorted(timings.glob("*inner-exec*.json.gz"))
    assert graphs and len(graphs) == len(execs)
    for graph_path, exec_path in zip(graphs, execs, strict=True):
        graph = _read_gz(graph_path)
        execution = _read_gz(exec_path)
        nodes = {n["key"] for n in graph["nodes"]}
        rows = execution["rows"]
        assert {r["key"] for r in rows} == nodes
        assert execution["n_executed"] > 0
        for row in rows:
            if row["start_ns"] is None:
                assert row["not_executed_reason"] != "ready_not_started" or row["state"] not in (
                    "Memory",
                    "Released",
                )
        construction = execution["construction"]
        assert construction["construction_s"] is not None
        assert construction["pre_scheduler_s"] > 0
        assert construction["capture_overhead_s"] >= 0


def test_the_two_schedulers_agree_on_every_pixel(demo):
    comparison = json.loads((demo / "comparison.json").read_text())
    assert comparison["pixels_identical"] == {"lst_p95": True, "qa_count": True}
    assert comparison["frisky"]["spill_count"] == 0
    assert comparison["frisky"]["errors"] == []


def test_the_outer_scheduler_holds_the_shard_spans_keyed_by_the_outer_task(demo):
    comparison = json.loads((demo / "comparison.json").read_text())
    outer = comparison["outer_scheduler"]
    names = {name for name, _keys in outer["shard_spans"]}
    assert "shard.group" in names
    keys = {tuple(k) for _name, k in outer["shard_spans"]}
    assert any("frisky01" in k[0] for k in keys if k)


def test_progress_and_final_objects_exist_with_scheduler_evidence(demo):
    timings = demo / "bucket-frisky" / "_shards" / "timings" / "inner-visibility-demo"
    final = json.loads(next(timings.glob("*.inner.a01.final.json")).read_text())
    assert final["scheduler"] == "frisky"
    assert final["closed_by"] == "exit"
    assert final["tasks_started"] > 0
    assert final["graph_files"] and final["exec_files"] and final["chunk_keys"]
    progress = json.loads(next(timings.glob("*.inner-progress.json")).read_text())
    assert progress["seq"] >= 1
    state = json.loads(
        next((demo / "bucket-frisky" / "_shards").rglob("state/composite.0016.1.json")).read_text()
    )
    assert state["inner"]["scheduler"] == "frisky"
    assert state["inner"]["trace_id"] == final["trace_id"]
