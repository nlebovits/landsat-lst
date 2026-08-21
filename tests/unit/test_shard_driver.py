"""Sequencing a tile's stages against S3, with no Coiled anywhere.

The driver's whole job is to decide what has finished and what to start next,
and it decides it from one listing. That makes it testable without a cluster:
a fake fleet writes the artifacts a stage's shards would have written, and
everything the driver does follows from which keys exist.

Four properties are worth pinning, and each of them is a way the barrier could
be wrong while still finishing the tile. Stages must run in order, because a
composite band that ran before the offsets merged would apply nothing. A
resubmission must carry *only* the missing indexes, or the barrier is just a
retry of the whole stage. The round cap must fire, or a deterministically
failing shard bills all night. And a resumed run must skip what is already in
the bucket, which is the only reason the driver may be killed at all.
"""

from __future__ import annotations

import pytest

from landsat_lst import shards
from landsat_lst.models import ProcessingJob
from landsat_lst.shard_driver import ShardStageFailed, drive_tile, resume_tile
from landsat_lst.storage import PRODUCTS, LocalStorage
from landsat_lst.tiling import parse_tile_name
from tests.unit.shard_fixtures import (
    RUN_ID,
    TILE,
    FakeFleet,
    make_plan,
)

pytestmark = pytest.mark.unit

STAGE_ORDER = ["resolve", "climatology", "offsets", "composite", "export"]


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(output_dir=tmp_path / "bucket")


@pytest.fixture
def plan():
    return make_plan()


@pytest.fixture
def job():
    return ProcessingJob(tile=parse_tile_name(TILE), year=2021, end_year=2025)


class TestHappyPath:
    def test_stages_advance_in_order_and_the_tile_completes(
        self, storage, plan, job, fast_barriers
    ):
        fleet = FakeFleet(storage, plan)

        summary = drive_tile(job, run_id=RUN_ID, storage=storage, submit=fleet)

        assert fleet.stages == STAGE_ORDER
        assert summary.completed
        assert summary.resubmissions == 0
        assert [s.stage for s in summary.stages] == [
            "resolve",
            "climatology",
            "offsets",
            "merge_offsets",
            "composite",
            "export",
        ]

    def test_the_merge_writes_the_ordinary_offset_cache_record(
        self, storage, plan, job, fast_barriers
    ):
        """Not a shard artifact in a shard-shaped place: the ADR-012 key.

        That is the seam. A composite shard reads it back exactly as a
        single-VM tile would, and a later whole-tile run over the same scenes
        finds it and skips its own offset pass.
        """
        from landsat_lst.shard_tasks import _offset_key

        drive_tile(job, run_id=RUN_ID, storage=storage, submit=FakeFleet(storage, plan))

        record = storage.read_text(_offset_key(plan).storage_key)
        assert record is not None
        assert '"scenes": 4' in record

    def test_every_shard_index_is_submitted_exactly_once(self, storage, plan, job, fast_barriers):
        fleet = FakeFleet(storage, plan)

        drive_tile(job, run_id=RUN_ID, storage=storage, submit=fleet)

        by_stage = dict(fleet.calls)
        assert by_stage["climatology"] == [0, 1]
        assert by_stage["offsets"] == [0, 1]
        assert by_stage["composite"] == [0, 1]
        assert by_stage["export"] == [0]


class TestResubmission:
    def test_only_the_missing_indexes_are_resent(self, storage, plan, job, fast_barriers):
        """A fleet that resent everything would also finish; that is the point.

        Shard 1 fails its first submission and succeeds on the second. The
        second call must carry ``[1]`` alone: index 0's artifact is already in
        the bucket, and restarting it would pay for a block twice.
        """
        fleet = FakeFleet(storage, plan, never={("climatology", 1)}, heal=True)

        summary = drive_tile(job, run_id=RUN_ID, storage=storage, submit=fleet)

        climatology = [indexes for stage, indexes in fleet.calls if stage == "climatology"]
        assert climatology == [[0, 1], [1]]
        assert summary.completed
        assert summary.resubmissions == 1

    def test_a_stage_that_never_finishes_fails_naming_its_keys(
        self, storage, plan, job, fast_barriers
    ):
        """Bounded, and specific. "A shard failed" is not actionable; a key is."""
        fleet = FakeFleet(storage, plan, never={("composite", 1)})

        with pytest.raises(ShardStageFailed) as excinfo:
            drive_tile(job, run_id=RUN_ID, storage=storage, submit=fleet)

        error = excinfo.value
        assert error.stage == "composite"
        assert error.missing == [
            shards.band_key(shards.shard_root(RUN_ID, TILE), product, 1) for product in PRODUCTS
        ]
        # Exactly the configured number of submissions, then it stops.
        assert [c for c in fleet.calls if c[0] == "composite"] == [
            ("composite", [0, 1]),
            ("composite", [1]),
        ]

    def test_the_round_cap_is_the_setting(self, storage, plan, job, fast_barriers, monkeypatch):
        from landsat_lst.config import settings

        monkeypatch.setattr(settings, "shard_barrier_rounds", 3)
        fleet = FakeFleet(storage, plan, never={("climatology", 0)})

        with pytest.raises(ShardStageFailed):
            drive_tile(job, run_id=RUN_ID, storage=storage, submit=fleet)

        assert len([c for c in fleet.calls if c[0] == "climatology"]) == 3


class TestResume:
    """A killed driver picks up from the bucket, not from anything it held."""

    def test_a_run_with_no_plan_cannot_be_resumed(self, storage):
        with pytest.raises(FileNotFoundError, match="nothing to resume"):
            resume_tile(RUN_ID, TILE, storage=storage)

    @pytest.mark.parametrize(
        ("done_through", "expected_stages"),
        [
            ("resolve", ["climatology", "offsets", "composite", "export"]),
            ("climatology", ["offsets", "composite", "export"]),
            ("offsets", ["composite", "export"]),
            ("composite", ["export"]),
            ("export", []),
        ],
    )
    def test_resume_starts_only_what_is_missing(
        self, storage, plan, fast_barriers, done_through, expected_stages
    ):
        """From every stage boundary, and the boundary after the last one.

        A stage whose artifacts are all present must start no cluster at all,
        which is what makes a resume cheap rather than a re-run that happens to
        skip work inside each task.
        """
        seed = FakeFleet(storage, plan)
        for stage in STAGE_ORDER[: STAGE_ORDER.index(done_through) + 1]:
            seed(stage=stage, run_id=RUN_ID, tile=TILE, indexes=_indexes(plan, stage))

        fleet = FakeFleet(storage, plan)
        summary = resume_tile(RUN_ID, TILE, storage=storage, submit=fleet)

        assert fleet.stages == expected_stages
        assert summary.completed
        assert summary.window == plan.window

    def test_a_finished_stage_reports_its_shards_as_already_done(
        self, storage, plan, fast_barriers
    ):
        seed = FakeFleet(storage, plan)
        for stage in ("resolve", "climatology"):
            seed(stage=stage, run_id=RUN_ID, tile=TILE, indexes=_indexes(plan, stage))

        summary = resume_tile(RUN_ID, TILE, storage=storage, submit=FakeFleet(storage, plan))

        climatology = next(s for s in summary.stages if s.stage == "climatology")
        assert climatology.submissions == 0
        assert climatology.already_done == climatology.shards == plan.ref_shards


class TestSummary:
    def test_the_summary_round_trips_to_json_safe_primitives(
        self, storage, plan, job, fast_barriers
    ):
        summary = drive_tile(job, run_id=RUN_ID, storage=storage, submit=FakeFleet(storage, plan))

        payload = summary.as_dict()
        assert payload["tile"] == TILE
        assert payload["completed"] is True
        assert {s["stage"] for s in payload["stages"]} == {
            "resolve",
            "climatology",
            "offsets",
            "merge_offsets",
            "composite",
            "export",
        }


def _indexes(plan, stage: str) -> list[int]:
    counts = {
        "resolve": 1,
        "climatology": plan.ref_shards,
        "offsets": plan.scene_shards,
        "composite": len(plan.bands),
        "export": 1,
    }
    return list(range(counts[stage]))
