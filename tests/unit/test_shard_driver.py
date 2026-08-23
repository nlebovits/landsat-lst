"""Sequencing a tile's stages against S3, with no Coiled anywhere.

The driver's whole job is to decide what has finished and what to start next,
and it decides it from one listing. That makes it testable without a cluster:
a fake fleet writes the artifacts a stage's shards would have written, and
everything the driver does follows from which keys exist.

Five properties are worth pinning, and each of them is a way the barrier could
be wrong while still finishing the tile. Stages must run in order, because a
composite band that ran before the offsets merged would apply nothing. A
resubmission must carry *only* the missing indexes, or the barrier is just a
retry of the whole stage. The round cap must fire, or a deterministically
failing shard bills all night. A resumed run must skip what is already in the
bucket, which is the only reason the driver may be killed at all. And the
consolidation's two savings have to be *observable*: the composite fleet must
start while phase B is still producing, and the export must not cost a
submission when a composite worker already claimed it. Both would finish a
correct tile either way, which is exactly why they need pinning.
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
    LandsOnPoll,
    make_plan,
    publish_plan,
    record_in_flight,
)

pytestmark = pytest.mark.unit

#: What the driver submits now: one fused offsets fleet, one composite fleet.
#: The export is claimed by a composite worker, so it costs no submission.
STAGE_ORDER = ["offsets", "composite"]

#: The sub-phase boundaries a driver can be killed at, which is a longer list
#: than what it submits: "resolve" is shard 0 of the fused stage and "export" is
#: claimed by a composite worker, but both leave artifacts a resume must honour.
SEED_ORDER = ["resolve", "offsets", "composite", "export"]


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
        assert by_stage["offsets"] == [0, 1]
        assert by_stage["composite"] == [0, 1]
        assert "resolve" not in by_stage, "resolve is shard 0 of the fused stage"
        assert "climatology" not in by_stage, "climatology is a sub-phase, not a fleet"
        assert "export" not in by_stage, "a composite worker claimed it"


class TestResubmission:
    def test_only_the_missing_indexes_are_resent(self, storage, plan, job, fast_barriers):
        """A fleet that resent everything would also finish; that is the point.

        Shard 1 fails its first submission and succeeds on the second. The
        second call must carry ``[1]`` alone: index 0's artifact is already in
        the bucket, and restarting it would pay for a block twice.
        """
        fleet = FakeFleet(storage, plan, never={("offsets", 1)}, heal=True)

        summary = drive_tile(job, run_id=RUN_ID, storage=storage, submit=fleet)

        offsets = [indexes for stage, indexes in fleet.calls if stage == "offsets"]
        assert offsets == [[0, 1], [1]]
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
        fleet = FakeFleet(storage, plan, never={("composite", 0)})

        with pytest.raises(ShardStageFailed):
            drive_tile(job, run_id=RUN_ID, storage=storage, submit=fleet)

        assert len([c for c in fleet.calls if c[0] == "composite"]) == 3


class TestResume:
    """A killed driver picks up from the bucket, not from anything it held."""

    def test_a_run_with_no_plan_cannot_be_resumed(self, storage, plan):
        with pytest.raises(FileNotFoundError, match="nothing to resume"):
            resume_tile(RUN_ID, TILE, storage=storage, submit=FakeFleet(storage, plan))

    @pytest.mark.parametrize(
        ("done_through", "expected_stages"),
        [
            ("resolve", ["offsets", "composite"]),
            ("offsets", ["composite"]),
            ("composite", []),
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
        for stage in SEED_ORDER[: SEED_ORDER.index(done_through) + 1]:
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
        for stage in ("resolve", "offsets"):
            seed(stage=stage, run_id=RUN_ID, tile=TILE, indexes=_indexes(plan, stage))

        summary = resume_tile(RUN_ID, TILE, storage=storage, submit=FakeFleet(storage, plan))

        offsets = next(s for s in summary.stages if s.stage == "offsets")
        assert offsets.submissions == 0
        assert offsets.already_done == offsets.shards == plan.scene_shards


class TestSummary:
    def test_the_summary_round_trips_to_json_safe_primitives(
        self, storage, plan, job, fast_barriers
    ):
        summary = drive_tile(job, run_id=RUN_ID, storage=storage, submit=FakeFleet(storage, plan))

        payload = summary.as_dict()
        assert payload["tile"] == TILE
        assert payload["completed"] is True
        assert {s["stage"] for s in payload["stages"]} == {
            "offsets",
            "merge_offsets",
            "composite",
            "export",
        }


class TestStorageBackend:
    """The driver and its shards must read and write one namespace.

    The acceptance run for S30W065 did not: the default local backend had the
    driver listing a directory on the laptop while the VMs -- which inherit
    ``LST_STORAGE_BACKEND=s3`` from ``_worker_environ`` -- published to the
    bucket. ``plan.json`` landed within 3.5 minutes and the resolve barrier
    never closed. A barrier that cannot see its artifacts fails as a hang,
    which is the most expensive shape a failure can take.
    """

    def test_a_local_backend_is_refused_before_anything_is_submitted(self, storage, job):
        from landsat_lst.shard_driver import ShardBackendMismatch

        with pytest.raises(ShardBackendMismatch) as excinfo:
            drive_tile(job, run_id=RUN_ID, storage=storage)

        message = str(excinfo.value)
        assert "storage_backend" in message
        assert "LST_STORAGE_BACKEND=s3" in message
        # Nothing was written, so nothing was started.
        assert storage.read_text(shards.plan_key(shards.shard_root(RUN_ID, TILE))) is None

    def test_resume_refuses_the_same_way(self, storage, plan):
        from landsat_lst.shard_driver import ShardBackendMismatch

        publish_plan(storage, plan)

        with pytest.raises(ShardBackendMismatch):
            resume_tile(RUN_ID, TILE, storage=storage)

    def test_an_injected_submitter_is_driving_something_local_on_purpose(
        self, storage, plan, job, fast_barriers
    ):
        """Which is every test here, and must stay allowed."""
        summary = drive_tile(job, run_id=RUN_ID, storage=storage, submit=FakeFleet(storage, plan))

        assert summary.completed


class TestClusterNames:
    """A round that reuses a name collides with a cluster still in flight."""

    def test_each_round_gets_its_own_name(self, storage, plan, job, fast_barriers):
        """Observed verbatim: ``Unable to add batch jobs to existing cluster
        'lst-shard-S30W065-2021-2025-20260821T194111Z-S30W065-climato'`` -- no
        round marker, and truncated mid-stage so appending one would have been
        eaten.
        """
        fleet = FakeFleet(storage, plan, never={("offsets", 1)}, heal=True)

        drive_tile(job, run_id=RUN_ID, storage=storage, submit=fleet)

        offsets = [
            name
            for (stage, _), name in zip(fleet.calls, fleet.names, strict=True)
            if stage == "offsets"
        ]
        assert offsets == sorted(set(offsets))
        assert len(offsets) == 2
        assert offsets[0].endswith("-r1")
        assert offsets[1].endswith("-r2")

    def test_the_name_stays_well_inside_the_length_limit(self):
        """The run id is hashed rather than spelled: it already holds the tile."""
        from landsat_lst.batch import stage_cluster_name

        name = stage_cluster_name(
            "shard-S30W065-2021-2025-20260821T194111Z", "S30W065", "climatology", 2
        )

        assert name.endswith("-r2")
        assert len(name) < 45


class TestInFlightAdoption:
    """A stage somebody else started is watched, never restarted.

    The artifacts cannot tell "still booting" from "nobody started this" --
    shards publish nothing until they finish -- so a submission record, written
    before the submission, is what distinguishes them.
    """

    def test_a_fresh_submission_record_is_adopted_rather_than_resubmitted(
        self, storage, plan, monkeypatch
    ):
        from landsat_lst.config import settings

        monkeypatch.setattr(settings, "shard_barrier_timeout_s", 300)
        monkeypatch.setattr(settings, "shard_driver_poll_s", 0.001)
        publish_plan(storage, plan)
        record_in_flight(storage, plan, "offsets", [0, 1])

        watching = LandsOnPoll(storage, plan, "offsets", after=4)
        fleet = FakeFleet(storage, plan)
        summary = resume_tile(RUN_ID, TILE, storage=watching, submit=fleet)

        assert "offsets" not in fleet.stages
        assert summary.completed
        offsets = next(s for s in summary.stages if s.stage == "offsets")
        assert offsets.submissions == 0
        assert offsets.adopted == 1

    def test_a_second_driver_submits_nothing_while_round_one_is_fresh(
        self, storage, plan, monkeypatch
    ):
        """The collision, as a regression test.

        Driver one's climatology cluster is still in flight and its shards have
        published nothing. Driver two must not start a second array under the
        same name -- which is what Coiled refused, and what would have paid for
        the same blocks twice if it had not.
        """
        from landsat_lst.config import settings

        monkeypatch.setattr(settings, "shard_barrier_timeout_s", 300)
        monkeypatch.setattr(settings, "shard_driver_poll_s", 0.001)
        publish_plan(storage, plan)
        record_in_flight(storage, plan, "offsets", [0, 1], age_s=5.0)

        watching = LandsOnPoll(storage, plan, "offsets", after=4)
        fleet = FakeFleet(storage, plan)
        resume_tile(RUN_ID, TILE, storage=watching, submit=fleet)

        assert [stage for stage, _ in fleet.calls if stage == "offsets"] == []
        assert watching.polls > 1, "it must actually have watched, not just skipped the stage"

    def test_after_the_deadline_only_the_missing_indexes_go_again(
        self, storage, plan, fast_barriers
    ):
        """A stale record is a cluster that is gone, so the stage restarts --
        under a *new* round, and covering only what never landed.
        """
        publish_plan(storage, plan)
        seed = FakeFleet(storage, plan)
        seed(stage="offsets", run_id=RUN_ID, tile=TILE, indexes=[0])  # round 1 landed one
        record_in_flight(storage, plan, "offsets", [0, 1], age_s=10_000.0)

        fleet = FakeFleet(storage, plan)
        summary = resume_tile(RUN_ID, TILE, storage=storage, submit=fleet)

        assert [indexes for stage, indexes in fleet.calls if stage == "offsets"] == [[1]]
        assert fleet.names[fleet.stages.index("offsets")].endswith("-r2")
        assert summary.completed

    def test_the_round_budget_is_the_stage_not_the_driver(self, storage, plan, fast_barriers):
        """Otherwise every resume would grant the stage a fresh budget."""
        publish_plan(storage, plan)
        for round_no in (1, 2):
            record_in_flight(storage, plan, "offsets", [0, 1], round_no=round_no, age_s=10_000.0)

        fleet = FakeFleet(storage, plan)
        with pytest.raises(ShardStageFailed, match="offsets"):
            resume_tile(RUN_ID, TILE, storage=storage, submit=fleet)

        assert fleet.calls == []

    def test_a_submission_record_is_published_before_the_submission(
        self, storage, plan, job, fast_barriers
    ):
        """A driver that dies mid-submit must leave the record, not the orphan.

        The other order leaves a live cluster nothing mentions, which is
        precisely the collision.
        """
        root = shards.shard_root(RUN_ID, TILE)
        seen: list[bool] = []

        class _Watcher(FakeFleet):
            def __call__(self, **kwargs):
                key = shards.stage_submission_key(root, kwargs["stage"], kwargs["submission_round"])
                seen.append(self.storage.read_text(key) is not None)
                return super().__call__(**kwargs)

        drive_tile(job, run_id=RUN_ID, storage=storage, submit=_Watcher(storage, plan))

        assert seen and all(seen)


class TestCompositeOverlap:
    """The composite fleet boots on the offsets stage's time, or it saves nothing.

    A fleet started after phase B finished would produce the same tile, which
    is why this needs a test that looks at *when* rather than at *whether*.
    """

    def test_composite_starts_before_phase_b_finishes(self, storage, plan, monkeypatch):
        """Constructed so the tile can *only* finish if the overlap fired.

        One partial has landed and the other shard is still working -- the
        moment the overlap exists for. Phase B is then rigged to complete only
        once the composite fleet has been asked for, so a driver that waited
        for the offsets barrier before starting the composite would deadlock
        until the barrier expired and the stage failed.
        """
        from landsat_lst.config import settings

        monkeypatch.setattr(settings, "shard_driver_poll_s", 0.001)
        monkeypatch.setattr(settings, "shard_barrier_timeout_s", 60)
        monkeypatch.setattr(settings, "shard_barrier_rounds", 2)
        monkeypatch.setattr(settings, "shard_offset_vms", 2)
        monkeypatch.setattr(settings, "shard_export_claim_fallback_s", 0)

        publish_plan(storage, plan)
        FakeFleet(storage, plan)(stage="offsets", run_id=RUN_ID, tile=TILE, indexes=[0])
        record_in_flight(storage, plan, "offsets", [0, 1])

        fleet = FakeFleet(storage, plan)
        watching = LandsOnPoll(storage, plan, "offsets", when=lambda: "composite" in fleet.stages)

        summary = resume_tile(RUN_ID, TILE, storage=watching, submit=fleet)

        composite_at = fleet.stages.index("composite")
        assert fleet.partials_at_call[composite_at] < plan.scene_shards
        assert summary.completed

    def test_the_trigger_is_evidence_not_a_timer(self, storage, plan, fast_barriers):
        """Zero partials means the stage may be about to fail; do not gamble a fleet."""
        from landsat_lst.shard_driver import _overlap_ready

        root = publish_plan(storage, plan)

        assert _overlap_ready(storage, root, plan) is False

        FakeFleet(storage, plan)(stage="offsets", run_id=RUN_ID, tile=TILE, indexes=[0])
        assert _overlap_ready(storage, root, plan) is True

    def test_a_full_fraction_disables_the_overlap(self, storage, plan, fast_barriers, monkeypatch):
        from landsat_lst.config import settings
        from landsat_lst.shard_driver import _overlap_ready

        monkeypatch.setattr(settings, "shard_composite_overlap", 1.0)
        root = publish_plan(storage, plan)
        FakeFleet(storage, plan)(stage="offsets", run_id=RUN_ID, tile=TILE, indexes=[0])

        assert _overlap_ready(storage, root, plan) is False


class TestExportClaim:
    """A composite worker runs the export; the driver only covers a lost claim."""

    def test_a_claimed_export_costs_no_submission(self, storage, plan, job, fast_barriers):
        fleet = FakeFleet(storage, plan)

        summary = drive_tile(job, run_id=RUN_ID, storage=storage, submit=fleet)

        assert "export" not in fleet.stages
        assert storage.read_text(shards.export_claim_key(shards.shard_root(RUN_ID, TILE)))
        export = next(s for s in summary.stages if s.stage == "export")
        assert export.submissions == 0
        assert summary.completed

    def test_the_fallback_fires_when_the_claim_goes_unexecuted(
        self, storage, plan, job, fast_barriers
    ):
        """The claiming VM was preempted between writing the claim and running it."""
        fleet = FakeFleet(storage, plan, claims_export=False)

        summary = drive_tile(job, run_id=RUN_ID, storage=storage, submit=fleet)

        assert "export" in fleet.stages
        assert summary.completed

    def test_the_fallback_waits_before_it_submits(self, storage, plan, job, monkeypatch):
        """A worker mid-merge must not be raced by a fleet doing the same merge."""
        from landsat_lst.config import settings
        from landsat_lst.shard_driver import _await_export

        monkeypatch.setattr(settings, "shard_driver_poll_s", 0.001)
        monkeypatch.setattr(settings, "shard_export_claim_fallback_s", 5)
        root = publish_plan(storage, plan)
        watching = LandsOnPoll(storage, plan, "export", after=3)
        fleet = FakeFleet(storage, plan)

        from landsat_lst.shard_driver import Clock

        outcome = _await_export(
            run_id=RUN_ID,
            tile=TILE,
            root=root,
            storage=watching,
            plan=plan,
            submit=fleet,
            clock=Clock(),
        )

        assert fleet.calls == []
        assert outcome.submissions == 0


def _indexes(plan, stage: str) -> list[int]:
    counts = {
        "resolve": 1,
        "climatology": plan.ref_shards,
        "offsets": plan.scene_shards,
        "composite": len(plan.bands),
        "export": 1,
    }
    return list(range(counts[stage]))
