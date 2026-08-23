"""The driver as a state machine, driven through every path it can take.

The driver spends hours waiting and its bugs live in the waiting. Two nights
proved it: a stage barrier whose deadline was measured from the *first*
submission, so a round-2 resubmission adopted at T+46min against a deadline
that had expired at T+45 and failed having watched for nothing; and an empty
``ServerError`` from a cluster create -- the Coiled credit quota, as it turned
out -- that killed the driver outright instead of being retried or reported.
Neither is reachable by reading the code, and neither was reachable by a test
that had to wait out a real barrier.

So time is injected. Every scenario here runs the whole sequence against a
:class:`FakeClock` that advances only when something asks it to, a fleet whose
per-round and per-shard behaviour is scripted, and local storage. The suite
runs in milliseconds and asserts as much, because a state machine nobody can
afford to run exhaustively is a state machine nobody runs.

The states, and what moves between them (``shard_driver.StageMachine``)::

    check --(nothing missing)------------> settled
    check --(fresh record)---------------> adopt --> watch
    check --(no record, rounds left)-----> submit -> watch
    check --(no rounds left)-------------> exhausted -> ShardStageFailed
    watch --(all artifacts present)------> settled
    watch --(deadline passed)------------> check
    submit --(terminal API failure)------> ShardSubmissionFailed
    submit --(transient, retries left)---> submit
    watch  --(cluster reports dead)------> ShardFleetKilled

At tile level the driver walks ``offsets -> merge_offsets -> composite ->
export``, starting the composite fleet from inside the offsets barrier and
leaving the export to whichever composite worker writes the last band.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import ClassVar

import pytest

from landsat_lst import budgets, quota, shards
from landsat_lst.config import settings
from landsat_lst.models import ProcessingJob
from landsat_lst.shard_driver import (
    Clock,
    ShardFleetKilled,
    ShardStageFailed,
    ShardSubmissionFailed,
    classify_failure,
    drive_tile,
    resume_tile,
)
from landsat_lst.storage import PRODUCTS, LocalStorage
from landsat_lst.tiling import parse_tile_name
from tests.unit.shard_fixtures import RUN_ID, TILE, FakeFleet, make_plan, publish_plan

pytestmark = pytest.mark.unit

#: Sub-phase boundaries a driver can be killed at. Longer than what it submits:
#: "resolve" is shard 0 of the fused stage and "export" is claimed by a
#: composite worker, but both leave artifacts a resume must honour.
BOUNDARIES = ["nothing", "resolve", "offsets", "composite", "export"]


class FakeClock(Clock):
    """Time that moves only when something waits, starting at a plausible epoch.

    ``sleep`` advances instead of blocking, so a barrier that would burn
    45 minutes burns none. The epoch is real-ish so a submission record written
    by this clock is indistinguishable from one written by a driver.
    """

    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self._now = start
        #: Every advance, so a test can assert *how long* a barrier waited --
        #: which is the only way to catch a deadline that was never fresh.
        self.slept: list[float] = []

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.slept.append(seconds)
            self._now += seconds

    def advance(self, seconds: float) -> None:
        self._now += seconds

    @property
    def elapsed(self) -> float:
        return sum(self.slept)


class ScriptedFleet(FakeFleet):
    """A fleet whose shards can be slow, dead, or deterministically broken.

    Everything a real fleet can do to a driver, minus the waiting:

    - ``lands_after``: ``(stage, index) -> polls`` before the artifact appears,
      so a barrier has to actually watch rather than find the work already done.
    - ``never``/``heal``: inherited -- a shard that fails once, or forever.
    - ``raise_once`` / ``raise_always``: the submission API failing, which is
      what killed the driver on 2026-08-22.
    - ``cluster_state``: what a probe reports for this stage's cluster, which is
      how a fleet Coiled has already torn down becomes visible before the
      barrier expires.
    """

    def __init__(
        self,
        storage,
        plan,
        *,
        clock: FakeClock,
        lands_after: dict[tuple[str, int], int] | None = None,
        in_flight: dict[tuple[str, int], int] | None = None,
        raise_once: Exception | None = None,
        raise_always: Exception | None = None,
        cluster_state: dict[str, tuple[str, str]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(storage, plan, **kwargs)
        self.clock = clock
        self._lands_after = dict(lands_after or {})
        self._raise_once = raise_once
        self._raise_always = raise_always
        self._cluster_state = dict(cluster_state or {})
        #: Pending writes, as ``(stage, index) -> polls remaining``. Seeded by
        #: ``in_flight`` for shards belonging to *another* driver's fleet --
        #: the adoption case, where nothing was submitted here and the
        #: artifacts still have to arrive from somewhere.
        self._pending: dict[tuple[str, int], int] = dict(in_flight or {})
        self.submit_attempts = 0
        #: ``(stage, clock.now())`` per submission, for the deadline tests.
        self.submitted_at: list[tuple[str, float]] = []

    def __call__(self, **kwargs):
        self.submit_attempts += 1
        if self._raise_always is not None:
            raise self._raise_always
        if self._raise_once is not None:
            error, self._raise_once = self._raise_once, None
            raise error
        self.submitted_at.append((kwargs["stage"], self.clock.now()))
        return super().__call__(**kwargs)

    def _write(self, stage: str, index: int) -> None:
        delay = self._lands_after.get((stage, index), 0)
        if delay:
            self._pending[stage, index] = delay
            return
        super()._write(stage, index)

    def tick(self) -> None:
        """One poll's worth of progress for every pending shard."""
        for key in list(self._pending):
            self._pending[key] -= 1
            if self._pending[key] <= 0:
                del self._pending[key]
                super()._write(*key)

    def probe(self, cluster_id):
        """A cluster probe over the scripted states."""
        for stage, state in self._cluster_state.items():
            if any(s == stage for s, _ in self.calls):
                return state
        del cluster_id
        return None


class TickingStorage:
    """Local storage that advances the fleet one step per artifact listing.

    A poll is what makes time pass in a real run, so a poll is what makes the
    scripted shards progress here. Without this the driver would either find
    everything done immediately or spin against a fleet that never moves.
    """

    def __init__(self, storage, fleet: ScriptedFleet) -> None:
        self._storage = storage
        self._fleet = fleet
        self.listings = 0

    def __getattr__(self, name):
        return getattr(self._storage, name)

    def list_prefix(self, prefix: str):
        self.listings += 1
        self._fleet.tick()
        return self._storage.list_prefix(prefix)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(output_dir=tmp_path / "bucket")


@pytest.fixture
def plan():
    return make_plan()


@pytest.fixture
def job():
    return ProcessingJob(tile=parse_tile_name(TILE), year=2021, end_year=2025)


@pytest.fixture(autouse=True)
def _machine_settings(monkeypatch):
    """Derived deadlines, a fleet the fixture plan fits, and no real sleeping."""
    monkeypatch.setattr(settings, "shard_barrier_timeout_s", None)
    monkeypatch.setattr(settings, "shard_barrier_rounds", 2)
    monkeypatch.setattr(settings, "shard_offset_vms", 2)
    monkeypatch.setattr(settings, "shard_driver_poll_s", 20.0)
    monkeypatch.setattr(settings, "shard_export_claim_fallback_s", 900)
    monkeypatch.setattr(settings, "shard_submit_backoff_s", 5.0)


def _drive(job, storage, fleet, clock, **kwargs):
    return drive_tile(
        job,
        run_id=RUN_ID,
        storage=storage,
        submit=fleet,
        clock=clock,
        cluster_probe=kwargs.pop("cluster_probe", None),
        **kwargs,
    )


def _seed(storage, plan, through: str) -> None:
    """Populate the bucket as if the run had reached ``through``."""
    order = ["resolve", "offsets", "composite", "export"]
    if through == "nothing":
        return
    seed = FakeFleet(storage, plan)
    for stage in order[: order.index(through) + 1]:
        seed(stage=stage, run_id=RUN_ID, tile=TILE, indexes=seed.all_indexes(stage))


# ---------------------------------------------------------------------------
# 1-3: the ordinary paths
# ---------------------------------------------------------------------------


class TestHappyPaths:
    def test_1_fresh_run_reaches_completion_including_the_export_claim(
        self, storage, plan, job, clock
    ):
        fleet = ScriptedFleet(storage, plan, clock=clock)

        summary = _drive(job, storage, fleet, clock)

        assert summary.completed
        assert fleet.stages == ["offsets", "composite"]
        assert storage.read_text(shards.export_claim_key(shards.shard_root(RUN_ID, TILE)))
        assert "export" not in fleet.stages

    def test_2_a_complete_tile_submits_nothing(self, storage, plan, clock):
        """Every artifact present: the driver walks through and starts no fleet."""
        _seed(storage, plan, "export")
        fleet = ScriptedFleet(storage, plan, clock=clock)

        summary = resume_tile(
            RUN_ID, TILE, storage=storage, submit=fleet, clock=clock, cluster_probe=None
        )

        assert fleet.calls == []
        assert summary.completed
        assert clock.elapsed == 0.0

    def test_3_a_live_submission_is_adopted_rather_than_restarted(self, storage, plan, clock):
        """Fresh record, artifacts pending: watch it, do not submit a duplicate."""
        _seed(storage, plan, "resolve")
        fleet = ScriptedFleet(
            storage, plan, clock=clock, in_flight={("offsets", 0): 6, ("offsets", 1): 9}
        )
        ticking = TickingStorage(storage, fleet)
        _record_live(storage, plan, "offsets", clock.now())

        summary = resume_tile(
            RUN_ID, TILE, storage=ticking, submit=fleet, clock=clock, cluster_probe=None
        )

        assert "offsets" not in fleet.stages, "an adopted stage must not be resubmitted"
        offsets = next(s for s in summary.stages if s.stage == "offsets")
        assert offsets.adopted == 1
        assert offsets.submissions == 0


# ---------------------------------------------------------------------------
# 4: death and restart at every boundary
# ---------------------------------------------------------------------------


class TestRestart:
    @pytest.mark.parametrize("boundary", BOUNDARIES)
    def test_4_a_driver_killed_at_any_boundary_resumes_to_completion(
        self, storage, plan, job, clock, boundary
    ):
        """The driver holds no state a crash could lose; the bucket holds it all."""
        _seed(storage, plan, boundary)
        fleet = ScriptedFleet(storage, plan, clock=clock)

        summary = (
            _drive(job, storage, fleet, clock)
            if boundary == "nothing"
            else resume_tile(
                RUN_ID, TILE, storage=storage, submit=fleet, clock=clock, cluster_probe=None
            )
        )

        assert summary.completed
        started = set(fleet.stages)
        if boundary in ("composite", "export"):
            assert started == set(), "everything was already in the bucket"
        elif boundary == "offsets":
            assert started == {"composite"}


# ---------------------------------------------------------------------------
# 5-10: worker failure, rounds, and deadlines
# ---------------------------------------------------------------------------


class TestRounds:
    def test_5_round_two_resubmits_only_the_missing_index(self, storage, plan, job, clock):
        fleet = ScriptedFleet(storage, plan, clock=clock, never={("offsets", 1)}, heal=True)

        summary = _drive(job, storage, fleet, clock)

        offsets = [indexes for stage, indexes in fleet.calls if stage == "offsets"]
        assert offsets == [[0, 1], [1]]
        assert summary.completed

    def test_6_partial_completion_is_honoured_across_rounds(self, storage, plan, job, clock):
        """Shard 0's partial from round 1 must survive into round 2 untouched."""
        fleet = ScriptedFleet(storage, plan, clock=clock, never={("offsets", 1)}, heal=True)
        root = shards.shard_root(RUN_ID, TILE)

        _drive(job, storage, fleet, clock)

        first = shards.scene_partial_key(root, *_range(plan, 0))
        assert storage.read_text(first) is not None
        # Round 2 carried index 1 alone, so index 0 was never recomputed.
        assert [i for stage, idx in fleet.calls if stage == "offsets" for i in idx].count(0) == 1

    def test_7_round_two_gets_a_fresh_deadline(self, storage, plan, job, clock):
        """The regression for the S30W065 collapse.

        A round used to be watched against a deadline belonging to the round
        before it, so a resubmission that opened after the first budget had run
        out expired on its first poll and failed having watched for nothing.

        Measured mechanically: a shard that never lands makes each round watch
        its whole budget, so two rounds must cost *two* budgets of wall clock.
        An inherited deadline would spend one and then quit.
        """
        fleet = ScriptedFleet(storage, plan, clock=clock, never={("offsets", 1)})
        budget = budgets.stage_budget("offsets", plan).deadline_s

        with pytest.raises(ShardStageFailed):
            _drive(job, storage, fleet, clock)

        opened = [at for stage, at in fleet.submitted_at if stage == "offsets"]
        assert len(opened) == 2
        assert opened[1] - opened[0] >= budget, "round 1 watched its whole budget"
        assert clock.elapsed >= 2 * budget, "and so did round 2"

    def test_8_exhausted_rounds_fail_naming_the_keys(self, storage, plan, job, clock):
        fleet = ScriptedFleet(storage, plan, clock=clock, never={("composite", 1)})

        with pytest.raises(ShardStageFailed) as excinfo:
            _drive(job, storage, fleet, clock)

        root = shards.shard_root(RUN_ID, TILE)
        assert excinfo.value.stage == "composite"
        assert excinfo.value.missing == [shards.band_key(root, product, 1) for product in PRODUCTS]
        assert len([c for c in fleet.calls if c[0] == "composite"]) == 2

    def test_9_a_stale_record_starts_a_new_round_rather_than_adopting(self, storage, plan, clock):
        """Past its budget, a record describes a cluster that is gone."""
        _seed(storage, plan, "resolve")
        fleet = ScriptedFleet(storage, plan, clock=clock)
        stale = clock.now() - budgets.stage_budget("offsets", plan).deadline_s - 1.0
        _record_live(storage, plan, "offsets", stale)

        summary = resume_tile(
            RUN_ID, TILE, storage=storage, submit=fleet, clock=clock, cluster_probe=None
        )

        offsets = next(s for s in summary.stages if s.stage == "offsets")
        assert offsets.adopted == 0
        assert offsets.submissions == 1
        assert fleet.names[0].endswith("-r2"), "a new round, not a reuse of round 1"

    def test_10_a_deterministically_broken_shard_fails_boundedly(self, storage, plan, job, clock):
        """Same shard fails every round: two rounds, then a named failure."""
        fleet = ScriptedFleet(storage, plan, clock=clock, never={("offsets", 0)})

        with pytest.raises(ShardStageFailed, match="offsets"):
            _drive(job, storage, fleet, clock)

        assert len([c for c in fleet.calls if c[0] == "offsets"]) == 2


# ---------------------------------------------------------------------------
# 11-12: the control plane
# ---------------------------------------------------------------------------


class TestControlPlaneFailures:
    def test_11_a_transient_submission_failure_is_retried(self, storage, plan, job, clock):
        """The 2026-08-22 regression: an empty ServerError killed the driver.

        It should have been retried. The driver survives a control-plane blip
        because the blip is not the work.
        """
        fleet = ScriptedFleet(storage, plan, clock=clock, raise_once=RuntimeError(""))

        summary = _drive(job, storage, fleet, clock)

        assert summary.completed
        assert fleet.submit_attempts == 3, "one failed attempt, then offsets and composite"
        assert clock.elapsed >= settings.shard_submit_backoff_s

    def test_11b_a_quota_failure_is_terminal_and_surfaces_the_reason(
        self, storage, plan, job, clock
    ):
        """The mask, removed. A quota that is exhausted will not clear in a backoff."""
        quota = RuntimeError(
            "Cluster failed to start: You have reached the workspace quota of 400 Coiled credits"
        )
        fleet = ScriptedFleet(storage, plan, clock=clock, raise_always=quota)

        with pytest.raises(ShardSubmissionFailed) as excinfo:
            _drive(job, storage, fleet, clock)

        assert "400 Coiled credits" in str(excinfo.value)
        assert excinfo.value.attempts == 1, "terminal means now, not after three tries"
        assert fleet.submit_attempts == 1

    def test_a_transient_failure_that_persists_becomes_terminal(self, storage, plan, job, clock):
        """Bounded: the driver reports rather than retrying all night."""
        fleet = ScriptedFleet(storage, plan, clock=clock, raise_always=OSError("connection reset"))

        with pytest.raises(ShardSubmissionFailed) as excinfo:
            _drive(job, storage, fleet, clock)

        assert excinfo.value.attempts == settings.shard_submit_retries
        assert "connection reset" in str(excinfo.value)

    def test_12_a_killed_fleet_is_surfaced_without_waiting_out_the_barrier(
        self, storage, plan, job, clock
    ):
        """A fleet Coiled has torn down produces no artifacts and never will.

        Waiting out the deadline buys nothing and costs the whole barrier,
        which is exactly what the 400-credit quota kill cost on 2026-08-22.
        """
        fleet = ScriptedFleet(
            storage,
            plan,
            clock=clock,
            never={("offsets", 1)},
            cluster_state={
                "offsets": (
                    "error",
                    "Scheduler Stopped -> Instance Stopped: You have reached the "
                    "workspace quota of 400 Coiled credits",
                )
            },
        )

        with pytest.raises(ShardFleetKilled) as excinfo:
            _drive(job, storage, fleet, clock, cluster_probe=fleet.probe)

        assert "400 Coiled credits" in excinfo.value.reason
        assert clock.elapsed < budgets.stage_budget("offsets", plan).deadline_s

    def test_a_stopped_cluster_whose_artifacts_landed_is_not_a_failure(
        self, storage, plan, job, clock
    ):
        """A fleet stops when its last task finishes; that is success, not death."""
        fleet = ScriptedFleet(
            storage, plan, clock=clock, cluster_state={"offsets": ("stopped", "done")}
        )

        summary = _drive(job, storage, fleet, clock, cluster_probe=fleet.probe)

        assert summary.completed


# ---------------------------------------------------------------------------
# 13: the export, both ways
# ---------------------------------------------------------------------------


class TestExport:
    def test_13a_a_composite_worker_claims_it(self, storage, plan, job, clock):
        fleet = ScriptedFleet(storage, plan, clock=clock)

        summary = _drive(job, storage, fleet, clock)

        assert "export" not in fleet.stages
        export = next(s for s in summary.stages if s.stage == "export")
        assert export.submissions == 0
        assert summary.completed

    def test_13b_the_driver_falls_back_when_no_worker_claims(self, storage, plan, job, clock):
        """The claiming VM was preempted between writing the claim and running it."""
        fleet = ScriptedFleet(storage, plan, clock=clock, claims_export=False)

        summary = _drive(job, storage, fleet, clock)

        assert "export" in fleet.stages
        assert summary.completed
        assert clock.elapsed >= settings.shard_export_claim_fallback_s


# ---------------------------------------------------------------------------
# The pieces the states rest on
# ---------------------------------------------------------------------------


class TestPreflightCredits:
    """Scenario zero: nothing submits before the workspace can pay for it.

    A quota is knowable before a cluster is created, and on 2026-08-22 it cost
    a night to learn it afterwards -- once as an empty ``ServerError`` on a
    create, once as a healthy fleet killed mid-stage at 400 credits.
    """

    def test_0a_sufficient_credits_proceed(self):
        balance = quota.preflight_credits(
            10.0, balance_source=lambda: quota.CreditBalance(remaining=500.0, source="fake")
        )

        assert balance.remaining == 500.0

    def test_0b_insufficient_credits_refuse_naming_the_shortfall(self, monkeypatch):
        monkeypatch.setattr(settings, "coiled_credit_safety", 1.5)

        with pytest.raises(quota.QuotaRefused) as excinfo:
            quota.preflight_credits(
                100.0,
                balance_source=lambda: quota.CreditBalance(remaining=120.0, source="fake"),
            )

        message = str(excinfo.value)
        assert "short by 30" in message
        assert quota.TEAM_URL in message

    def test_0c_an_exhausted_quota_flag_refuses_whatever_the_arithmetic_says(self):
        """``has_quota`` is the endpoint's own answer, and it outranks a subtraction."""
        with pytest.raises(quota.QuotaRefused, match="out of credits"):
            quota.preflight_credits(
                1.0,
                balance_source=lambda: quota.CreditBalance(
                    remaining=None, source="usage_endpoint", has_quota=False
                ),
            )

    def test_0d_an_unreadable_balance_refuses_without_an_acknowledgement(self, monkeypatch):
        monkeypatch.setattr(settings, "ack_quota", False)

        with pytest.raises(quota.QuotaRefused) as excinfo:
            quota.preflight_credits(
                42.0,
                balance_source=lambda: quota.CreditBalance(remaining=None, source="unavailable"),
            )

        assert "--ack-quota" in str(excinfo.value)
        assert quota.TEAM_URL in str(excinfo.value)

    def test_0e_an_acknowledged_unreadable_balance_proceeds(self, monkeypatch):
        monkeypatch.setattr(settings, "ack_quota", True)

        balance = quota.preflight_credits(
            42.0, balance_source=lambda: quota.CreditBalance(remaining=None, source="unavailable")
        )

        assert balance.remaining is None

    def test_0f_the_driver_submits_nothing_when_the_preflight_refuses(
        self, storage, plan, job, clock, monkeypatch
    ):
        """The whole point of a preflight: it runs before the first submission."""
        import landsat_lst.shard_driver as driver

        monkeypatch.setattr(settings, "ack_quota", False)
        # The backend guard runs first and has its own tests; standing the
        # fleet where the real submitter stands trips it too, and it is not
        # what this scenario is about.
        monkeypatch.setattr(driver, "require_shared_storage", _allow_any_backend)
        fleet = ScriptedFleet(storage, plan, clock=clock)

        def broke() -> quota.CreditBalance:
            return quota.CreditBalance(remaining=0.0, source="fake")

        with pytest.raises(quota.QuotaRefused):
            drive_tile(
                job,
                run_id=RUN_ID,
                storage=storage,
                submit=_as_real_submitter(fleet, monkeypatch),
                clock=clock,
                cluster_probe=None,
                balance_source=broke,
            )

        assert fleet.calls == []

    def test_the_estimate_is_credits_not_dollars_and_scales_with_the_tile(self, plan):
        """Built on the same budget model the deadlines are, so geometry moves it."""
        from dataclasses import replace

        small = quota.estimate_run_credits(plan)
        big = quota.estimate_run_credits(replace(plan, scene_times=plan.scene_times * 20))

        assert small > 0
        assert big > small

    def test_a_pre_plan_estimate_falls_back_to_the_projection(self):
        """The preflight runs before any plan exists, so it must still answer."""
        assert quota.estimate_run_credits() > 0


class TestCreditCalibration:
    """The model, checked against an invoice rather than against itself.

    The S30W065 acceptance run of 2026-08-23 billed **268.11 credits** where the
    old per-VM-hour model estimated 75 -- wrong by 3.6x, and wrong in the
    direction that lets an unaffordable run start. It could not see that a
    16-vCPU composite VM costs twice an 8-vCPU offsets VM for the same wall
    clock, because Coiled bills per vCPU-hour.
    """

    #: What the run actually was, from its billing events by cluster.
    BILLED = 268.11
    SHAPE: ClassVar[list[tuple[float, int, float]]] = [
        (15.0, 8, 31 / 60),  # offse-r1: ~15 x r6i.2xlarge, ~31 min
        (14.0, 8, 7 / 60),  # offse-r2: 14 x r6i.2xlarge, ~7 min
        (35.0, 16, 26 / 60),  # compo-r1: 35 x m6i.4xlarge, 20-32 min
    ]

    def test_the_billed_run_is_reproduced_within_the_observed_band(self):
        estimate = quota.credits_for_fleets(self.SHAPE)

        assert 0.5 * self.BILLED <= estimate <= 2.0 * self.BILLED, (
            f"{estimate:.0f} credits for the S30W065 shape against {self.BILLED} billed"
        )

    def test_it_errs_high_rather_than_low(self):
        """The old model was 3.6x *low*, which is the dangerous direction.

        This one is about 19% high on the same shape -- inside the observed
        0.6-1.25 band, and conservative: over-estimating refuses a run that
        would have fit, where under-estimating starts one that gets killed
        mid-stage and loses the whole tile.
        """
        estimate = quota.credits_for_fleets(self.SHAPE)

        assert self.BILLED <= estimate <= 1.35 * self.BILLED

    def test_vcpus_are_what_separate_the_two_fleets(self):
        """Same VM count, same wall clock, twice the cores: twice the credits."""
        eight = quota.credits_for_fleets([(35.0, 8, 0.5)])
        sixteen = quota.credits_for_fleets([(35.0, 16, 0.5)])

        assert sixteen == pytest.approx(2 * eight)

    @pytest.mark.parametrize(
        ("vm_type", "expected"),
        [
            ("r6i.2xlarge", 8),
            ("m6i.4xlarge", 16),
            ("m6i.8xlarge", 32),  # not in the table; parsed from the name
            ("c7g.large", 2),
            ("m6i.xlarge", 4),
        ],
    )
    def test_vcpus_are_known_or_parsed(self, vm_type, expected):
        """A type nobody tabulated must not silently price as one core."""
        from landsat_lst.projection import vcpus

        assert vcpus(vm_type) == expected

    def test_the_estimate_prices_both_fleets_at_their_own_vm_type(self, plan):
        offsets, composite, export = quota.run_fleets(plan, units=2)

        assert offsets[1] == 8, "offsets runs on the default preference list"
        assert composite[1] == 16, "composite runs on shard_composite_vm_type"
        assert export[1] == 16

    def test_a_pre_plan_estimate_still_prices_per_vcpu(self):
        """The preflight runs before a plan exists and must still be calibrated."""
        fleets = quota.run_fleets()

        assert fleets
        assert all(cpus in (8, 16) for _, cpus, _ in fleets)
        assert quota.estimate_run_credits() > 0

    def test_the_estimate_stays_comparable_to_an_invoice(self, monkeypatch):
        """The safety factor belongs to the decision, not to the number."""
        monkeypatch.setattr(settings, "coiled_credit_safety", 5.0)

        assert quota.credits_for_fleets([(1.0, 8, 1.0)]) == pytest.approx(8.0)


class TestIdentityPreflight:
    """An expired SSO session, caught before the startup rather than after it.

    Three times now the driver has spent a STAC query, a plan, and a fleet's
    boot before discovering that nothing it wrote could reach S3. The session
    expires within hours; a tile takes longer than that.
    """

    @pytest.mark.parametrize(
        ("name", "message"),
        [
            ("UnauthorizedSSOTokenError", "The SSO session has expired"),
            ("NoCredentialsError", "Unable to locate credentials"),
            ("TokenRetrievalError", "Error when retrieving token"),
            ("ProfileNotFound", "The config profile could not be found"),
        ],
    )
    def test_a_dead_session_refuses_and_names_the_command(self, name, message, monkeypatch):
        monkeypatch.setenv("AWS_PROFILE", "radiant-earth")

        with pytest.raises(quota.IdentityRefused) as excinfo:
            quota.preflight_identity(caller=_raises(_named_error(name, message)))

        assert "aws sso login --profile radiant-earth" in str(excinfo.value)

    def test_an_expired_token_response_is_recognised(self):
        """Credentials that exist but are stale come back as an STS error code."""
        error = _named_error("ClientError", "An error occurred (ExpiredToken)")
        error.response = {"Error": {"Code": "ExpiredToken"}}

        with pytest.raises(quota.IdentityRefused, match="aws sso login"):
            quota.preflight_identity(caller=_raises(error))

    def test_the_hint_falls_back_to_the_configured_profile(self, monkeypatch):
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.setattr(settings, "aws_profile", "some-profile")

        with pytest.raises(quota.IdentityRefused, match="--profile some-profile"):
            quota.preflight_identity(caller=_raises(_named_error("NoCredentialsError", "")))

    def test_a_healthy_session_proceeds_and_returns_the_arn(self):
        arn = "arn:aws:sts::123456789012:assumed-role/dev/nissim"

        assert quota.preflight_identity(caller=lambda: {"Arn": arn}) == arn

    def test_identity_is_checked_before_credits(self, storage, plan, job, clock, monkeypatch):
        """A session that cannot call STS cannot read a Coiled balance either."""
        import landsat_lst.shard_driver as driver

        monkeypatch.setattr(driver, "require_shared_storage", _allow_any_backend)
        fleet = ScriptedFleet(storage, plan, clock=clock)
        asked: list[str] = []

        def identity() -> str:
            asked.append("identity")
            raise quota.IdentityRefused("expired; run: aws sso login")

        def balance() -> quota.CreditBalance:
            asked.append("credits")
            return quota.CreditBalance(remaining=10_000.0, source="fake")

        monkeypatch.setattr(quota, "preflight_identity", identity)

        with pytest.raises(quota.IdentityRefused):
            drive_tile(
                job,
                run_id=RUN_ID,
                storage=storage,
                submit=_as_real_submitter(fleet, monkeypatch),
                clock=clock,
                cluster_probe=None,
                balance_source=balance,
            )

        assert asked == ["identity"], "credits must not be read after identity failed"
        assert fleet.calls == []


class TestErrorTaxonomy:
    @pytest.mark.parametrize(
        "message",
        [
            "You have reached the workspace quota of 400 Coiled credits",
            "402 Payment Required",
            "Unauthorized: invalid api token",
            "your account is not entitled to this instance type",
        ],
    )
    def test_terminal_messages_are_terminal(self, message):
        assert classify_failure(RuntimeError(message)) == "terminal"

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError(""),
            OSError("connection reset by peer"),
            TimeoutError("timed out"),
            RuntimeError("500 Internal Server Error"),
        ],
    )
    def test_everything_else_is_transient(self, error):
        """Including an error with no message.

        An empty ServerError killed the driver once; guessing "terminal" for
        the unknown case would reintroduce that for every ordinary blip.
        """
        assert classify_failure(error) == "transient"

    def test_a_missing_dependency_is_terminal(self):
        assert classify_failure(ImportError("no module named coiled")) == "terminal"


class TestBudgets:
    """Deadlines derived from the plan, not typed into settings."""

    def test_every_stage_has_a_named_phase_breakdown(self, plan):
        for stage in ("offsets", "composite", "export"):
            budget = budgets.stage_budget(stage, plan)
            assert budget.phases
            assert budget.work_s > 0
            assert dict(budget.phases)["boot"] == budgets.VM_BOOT_S

    def test_the_deadline_is_the_work_times_the_named_safety_factor(self, plan, monkeypatch):
        monkeypatch.setattr(settings, "shard_budget_safety", 3.0)
        budget = budgets.stage_budget("offsets", plan)

        assert budget.deadline_s == pytest.approx(budget.work_s * 3.0)

    def test_a_bigger_window_buys_a_bigger_budget(self, plan):
        """The whole point: geometry moves the deadline, nobody edits a constant."""
        from dataclasses import replace

        small = budgets.stage_budget("offsets", plan).work_s
        big = budgets.stage_budget(
            "offsets", replace(plan, scene_times=plan.scene_times * 10)
        ).work_s

        assert big > small

    def test_an_explicit_override_wins_everywhere(self, plan, monkeypatch):
        monkeypatch.setattr(settings, "shard_barrier_timeout_s", 42)

        assert budgets.stage_budget("composite", plan).deadline_s == 42.0

    def test_the_composite_budget_covers_the_tail_it_was_started_during(self, plan):
        """It boots while phase B is still running and polls for the record."""
        phases = dict(budgets.stage_budget("composite", plan).phases)

        assert (
            phases["offsets_tail"] == dict(budgets.stage_budget("offsets", plan).phases)["phase_b"]
        )

    def test_an_unknown_stage_is_refused(self, plan):
        with pytest.raises(ValueError, match="no budget for stage"):
            budgets.stage_budget("polish", plan)


def test_the_whole_state_machine_suite_runs_without_real_waiting():
    """A guard on the guard.

    Every scenario above must run on the fake clock. If a real sleep ever
    creeps back in, the suite stops being something anyone runs.
    """
    started = time.monotonic()
    clock = FakeClock()
    clock.sleep(7200)

    assert clock.elapsed == 7200
    assert time.monotonic() - started < 1.0


def _named_error(name: str, message: str) -> Exception:
    """An exception standing in for a botocore class, by name.

    ``quota`` classifies these by class *name* rather than by identity, so it
    never has to import botocore to decide -- and so a test never has to
    construct one of botocore's several incompatible signatures.
    """
    return type(name, (Exception,), {})(message)


def _raises(error: Exception):
    def caller() -> dict:
        raise error

    return caller


def _allow_any_backend(*_args, **_kwargs) -> None:
    """Stand in for the backend guard, which has its own tests elsewhere."""


def _as_real_submitter(fleet, monkeypatch):
    """Make an injected fleet look like the production submitter.

    ``_preflight`` skips a run that injects its own submitter, because such a
    run starts no clusters and spends no credits. To test the gate itself the
    fleet has to stand where the real one stands.
    """
    import landsat_lst.shard_driver as driver

    monkeypatch.setattr(driver, "submit_shard_stage", fleet)
    return fleet


def _record_live(storage, plan, stage: str, submitted_at: float, *, round_no: int = 1) -> None:
    root = shards.shard_root(RUN_ID, plan.tile)
    storage.write_text(
        shards.stage_submission_key(root, stage, round_no),
        json.dumps(
            {
                "run_id": RUN_ID,
                "tile": plan.tile,
                "stage": stage,
                "round": round_no,
                "indexes": [0, 1],
                "cluster_name": f"lst-fake-{stage}-r{round_no}",
                "cluster_id": 99,
                "submitted_at": submitted_at,
            }
        ),
    )


def _range(plan, index: int) -> tuple[int, int]:
    from landsat_lst.shard_tasks import offsets_group

    group = offsets_group(plan, index)
    return group[0][0], group[-1][1]


def _unused() -> SimpleNamespace:  # pragma: no cover - keeps the import honest
    return SimpleNamespace()


def _publish(storage, plan) -> str:  # pragma: no cover - convenience for debugging
    return publish_plan(storage, plan)
