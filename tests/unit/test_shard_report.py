"""Reading a sharded run back out of the bucket.

The fixture is the real thing: ``tests/fixtures/shard_state/`` holds two state
objects verbatim from ``shard-S30W065-2021-2025-20260904T165629Z``, one from a
band that finished and one from a band that was killed at 1,115 seconds of
``exporting``. The 24/11 split, the phase names, and the peak RSS in these
tests are that run's numbers, not invented ones.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from landsat_lst import shards
from landsat_lst.config import settings
from landsat_lst.shard_report import coiled_task_states, reconcile_shard_run
from landsat_lst.storage import LocalStorage
from tests.unit.shard_fixtures import RUN_ID, TILE, FakeFleet, make_plan, publish_plan

#: The eleven composite indexes that never published, from the real run.
KILLED = [3, 4, 5, 6, 8, 14, 15, 17, 24, 27, 28]
BANDS = 35

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "shard_state"


def _state(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(output_dir=tmp_path / "bucket")


@pytest.fixture
def plan():
    rows = BANDS * settings.cog_blocksize
    return dataclasses.replace(
        make_plan(),
        native_shape=(rows, settings.cog_blocksize),
        bands=shards.band_edges(rows, BANDS, settings.cog_blocksize),
        band_shards=BANDS,
    )


@pytest.fixture
def run(storage, plan):
    """The bucket as the killed run left it: 24 bands down, 11 states stuck."""
    root = shards.shard_root(RUN_ID, TILE)
    publish_plan(storage, plan, run_id=RUN_ID)
    # The offsets side finished; the composite stage is where the run died.
    seed = FakeFleet(storage, plan)
    seed(stage="offsets", run_id=RUN_ID, tile=TILE, indexes=seed.all_indexes("offsets"))
    done_body = _state("composite.0000.1.json")
    stuck_body = _state("composite.0003.1.json")

    for index in range(BANDS):
        killed = index in KILLED
        body = dict(stuck_body if killed else done_body)
        body["attempt"] = 1
        storage.write_text(shards.shard_state_key(root, "composite", index, 1), json.dumps(body))
        if killed:
            continue
        for product in ("lst_p95", "qa_count"):
            storage.write_text(shards.band_key(root, product, index), "tif")

    storage.write_text(
        shards.stage_submission_key(root, "composite", 1),
        json.dumps(
            {
                "run_id": RUN_ID,
                "tile": TILE,
                "stage": "composite",
                "round": 1,
                "indexes": list(range(BANDS)),
                "cluster_name": "lst-8c13a85c-S30W065-compo-r1",
                "cluster_id": 2006081,
                "submitted_at": 1_757_000_000.0,
            }
        ),
    )
    return storage


class TestReconcile:
    def test_the_sep_4_run_reports_24_done_and_11_missing(self, run, plan):
        del plan
        report = reconcile_shard_run(RUN_ID, storage=run, task_states={})

        tile = report.tiles[0]
        composite = next(s for s in tile.stages if s.stage == "composite")
        assert len(composite.done) == 24
        assert composite.missing == KILLED
        assert not tile.completed
        assert tile.missing == 11

    def test_a_killed_shard_reports_the_phase_it_died_in(self, run):
        report = reconcile_shard_run(RUN_ID, storage=run, task_states={})

        composite = next(s for s in report.tiles[0].stages if s.stage == "composite")
        attempt = composite.attempts[3][0]
        assert attempt.phase == "exporting"
        assert attempt.status is None, "a shard that never settled publishes no status"
        assert attempt.phase_seconds["exporting"] == pytest.approx(1115.5)
        assert attempt.instance_type == "m6i.4xlarge"
        assert attempt.peak_rss_mb == pytest.approx(19394.4140625)

    def test_the_exit_code_comes_from_coiled_and_nowhere_else(self, run):
        """Exit 137 is the difference between a failure and a kill."""
        states = {("composite", index): {"exit_code": 137, "state": "error"} for index in KILLED}

        report = reconcile_shard_run(RUN_ID, storage=run, task_states=states)

        composite = next(s for s in report.tiles[0].stages if s.stage == "composite")
        assert composite.attempts[3][0].exit_code == 137
        assert composite.attempts[0][0].exit_code is None
        assert report.coiled_read is False

    def test_a_submission_record_is_never_read_as_a_shard(self, run):
        """Three-digit round against four-digit index; see stage_submission_prefix."""
        report = reconcile_shard_run(RUN_ID, storage=run, task_states={})

        composite = next(s for s in report.tiles[0].stages if s.stage == "composite")
        assert composite.rounds == 1
        assert set(composite.attempts) == set(range(BANDS))

    def test_an_earlier_run_s_cogs_do_not_make_this_run_complete(self, run, plan):
        """The postmortem's trap: the key is window and tile, not run id.

        S30W065 was shipped on 2026-08-23, so the killed run of 2026-09-04
        found both COGs at the canonical key and would have read as complete
        with 11 of its own bands never written.
        """
        from landsat_lst.storage import PRODUCTS, collection_prefix

        for product in PRODUCTS:
            path = (
                run.output_dir
                / collection_prefix(plan.window)
                / TILE
                / f"{product}_{plan.window}_{TILE}.tif"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"tif")

        report = reconcile_shard_run(RUN_ID, storage=run, task_states={})

        tile = report.tiles[0]
        assert tile.cogs_present
        assert not tile.completed

    def test_a_staged_prefix_left_behind_is_surfaced(self, run):
        root = shards.shard_root(RUN_ID, TILE)
        run.write_text(f"{shards.tile_stage_prefix(root)}f1-v2-abc/b0000.s00000.npy", "staged")

        report = reconcile_shard_run(RUN_ID, storage=run, task_states={})

        assert report.tiles[0].staged_objects == 1

    def test_a_run_with_no_plan_reports_rather_than_raising(self, storage):
        root = shards.shard_root(RUN_ID, TILE)
        storage.write_text(shards.shard_state_key(root, "composite", 0, 1), json.dumps({}))

        report = reconcile_shard_run(RUN_ID, storage=storage, task_states={})

        assert report.tiles[0].plan_present is False
        assert report.tiles[0].completed is False

    def test_the_report_serializes(self, run):
        report = reconcile_shard_run(RUN_ID, storage=run, task_states={})

        payload = json.loads(json.dumps(report.to_dict(), default=str))
        composite = next(s for s in payload["tiles"][0]["stages"] if s["stage"] == "composite")
        assert payload["run_id"] == RUN_ID
        assert composite["expected"] == BANDS
        assert composite["attempts"]["3"][0]["phase"] == "exporting"


class TestCoiledTaskStates:
    def test_the_index_comes_from_the_submission_not_the_task_id(self, monkeypatch):
        """Coiled numbers by position in ``map_over_values``; a round-2 array
        carries only the missing indexes, so position 0 is not index 0.
        """
        import sys
        import types

        module = types.ModuleType("coiled.batch")

        def _status(_cluster):
            return [{"tasks": [{"array_task_id": 0, "exit_code": 137, "state": "error"}]}]

        module.status = _status
        package = types.ModuleType("coiled")
        package.batch = module
        monkeypatch.setitem(sys.modules, "coiled", package)
        monkeypatch.setitem(sys.modules, "coiled.batch", module)

        states = coiled_task_states([{"stage": "composite", "cluster_id": 7, "indexes": [3, 4, 5]}])

        assert states[("composite", 3)]["exit_code"] == 137
        assert ("composite", 0) not in states

    def test_coiled_being_unreachable_is_not_fatal(self, monkeypatch):
        import sys
        import types

        module = types.ModuleType("coiled.batch")

        def _fail(_cluster):
            raise RuntimeError("no token")

        module.status = _fail
        package = types.ModuleType("coiled")
        package.batch = module
        monkeypatch.setitem(sys.modules, "coiled", package)
        monkeypatch.setitem(sys.modules, "coiled.batch", module)

        assert coiled_task_states([{"stage": "composite", "cluster_id": 7, "indexes": [0]}]) == {}


class TestCli:
    def test_reconcile_prints_the_split_and_the_missing_indexes(self, run, monkeypatch):
        from click.testing import CliRunner

        from landsat_lst.cli import main

        monkeypatch.setattr(settings, "storage_backend", "local")
        monkeypatch.setattr(settings, "output_dir", run.output_dir)

        result = CliRunner().invoke(main, ["shard", "reconcile", RUN_ID, "--no-coiled"])

        assert result.exit_code == 0, result.output
        assert "24/35" in result.output
        assert "3-6, 8" in result.output, "runs of indexes collapse"

    def test_json_is_parseable(self, run, monkeypatch):
        from click.testing import CliRunner

        from landsat_lst.cli import main

        monkeypatch.setattr(settings, "storage_backend", "local")
        monkeypatch.setattr(settings, "output_dir", run.output_dir)

        result = CliRunner().invoke(main, ["shard", "reconcile", RUN_ID, "--no-coiled", "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        composite = next(s for s in payload["tiles"][0]["stages"] if s["stage"] == "composite")
        assert composite["missing"] == KILLED


def test_the_fixture_is_the_real_run(run):
    """Guard against a fixture drifting into something the run never produced."""
    del run
    body = _state("composite.0003.1.json")
    assert body["run_id"] == "shard-S30W065-2021-2025-20260904T165629Z"
    assert body["phase"] == "exporting"
    assert datetime.fromisoformat(body["updated_at"]) > datetime(2026, 9, 4, tzinfo=UTC)
