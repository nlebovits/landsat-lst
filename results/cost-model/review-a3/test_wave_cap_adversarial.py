"""Adversarial probes against the consolidated fleet driver (ADR-017, #108).

REVIEW ARTIFACT. This file lives under ``results/`` deliberately so it is not
collected by this branch's CI, which has no ``fleet_driver`` to import. At
integration it belongs in ``tests/unit/`` beside
``test_fleet_driver.py``, whose fixtures it reuses verbatim.

Run it against Agent 3's commit::

    mkdir -p /tmp/a3rev2
    git archive issue-108-fleet-driver -o /tmp/a3rev2/commit.tar
    tar -xf /tmp/a3rev2/commit.tar -C /tmp/a3rev2
    cp results/cost-model/review-a3/test_wave_cap_adversarial.py \
       /tmp/a3rev2/tests/unit/
    cd /tmp/a3rev2
    PYTHONPATH=/tmp/a3rev2/src <venv>/bin/python -m pytest \
      tests/unit/test_wave_cap_adversarial.py -q

These probe three gaps the 39-scenario suite does not reach. All eight pass
against f6cf6fc, and what that means differs by class -- read the assertions,
not the pass count:

- `TestWaveDeadlineVersusQueueDepth` **asserts the defective behaviour**, so
  passing means the defect is present and reproducible. These are
  characterization tests: when the deadline is scaled by queue depth, they
  must be **inverted**, not deleted.
- `TestPreemptionAfterArtifactUpload` asserts the **correct** behaviour, and
  passing means the driver already gets it right. That gap is missing
  coverage, not a bug. It is worth keeping precisely because nothing else
  pins it at fleet level.

The load-bearing one is `TestWaveDeadlineVersusQueueDepth`. ADR-017 says a
wave "is retired when its units settle or its deadline passes, so the tail
costs at most one wave's deadline of headroom", which reads as a bound. It is
not one. A wave's deadline is ``budgets.stage_budget(stage, plan).deadline_s``
-- one shard's projected work times ``shard_budget_safety`` (2.0) -- while the
wave's actual runtime is that work times its queue depth
``len(units) / max_workers``. The saving ADR-017 claims comes from queue depth
above 1. So every wave deep enough to save anything is a wave that expires
before its work finishes, and expiry returns headroom the VMs still hold.
"""

from __future__ import annotations

import pytest
from tests.unit.test_fleet_driver import (
    RUN_ID,
    TILES,
    FakeWaveFleet,
    TickingStorage,
    _plans,
)

from landsat_lst import shards
from landsat_lst.config import settings
from landsat_lst.fleet_driver import Demand, FleetDriver, TileTrack

pytestmark = pytest.mark.unit


@pytest.fixture
def clock():
    """The same injected clock the driver suite uses."""
    from tests.unit.test_driver_state_machine import FakeClock  # noqa: PLC0415

    return FakeClock()


@pytest.fixture
def storage(tmp_path):
    from landsat_lst.storage import LocalStorage  # noqa: PLC0415

    return LocalStorage(output_dir=tmp_path / "bucket")


def _driver(storage, clock, *, max_vms: int, tiles=TILES[:1], units: int = 2):
    """A hand-built driver, exactly as test_headroom_is_spoken_for_by_live_waves."""
    plans = _plans(tiles)
    fleet = FakeWaveFleet(storage, plans, clock=clock)
    ticking = TickingStorage(storage, fleet)
    for writer in fleet.writers.values():
        writer.storage = ticking
    driver = FleetDriver(
        run_id=RUN_ID,
        tracks=[
            TileTrack(
                tile=tile,
                run_id=RUN_ID,
                root=shards.shard_root(RUN_ID, tile),
                storage=ticking,
                units=units,
                clock=clock,
            )
            for tile in tiles
        ],
        storage=ticking,
        submit=fleet,
        clock=clock,
        units=units,
        max_vms=max_vms,
    )
    return driver, fleet


def _demand(tile: str, stage: str, n_units: int, deadline_s: float) -> Demand:
    return Demand(
        tile=tile,
        stage=stage,
        indexes=tuple(range(n_units)),
        submission_round=1,
        deadline_s=deadline_s,
    )


class TestWaveDeadlineVersusQueueDepth:
    """A wave deep enough to amortize boot is a wave that outlives its deadline."""

    def test_a_queued_wave_expires_before_its_work_can_finish(self, storage, clock):
        """The deadline is one shard's budget; the wave runs queue-depth of them.

        16 units on 4 workers is a queue 4 deep. `shard_budget_safety` is 2.0,
        so the deadline covers 2 units per worker and the wave needs 4. The
        driver retires it at the halfway mark and takes back headroom that 4
        live VMs are still occupying.
        """
        driver, _ = _driver(storage, clock, max_vms=4)
        one_shard_s = 100.0
        deadline_s = one_shard_s * settings.shard_budget_safety

        driver._buffer_demand(_demand(TILES[0], "offsets", 16, deadline_s))
        driver._flush("offsets")
        assert driver.in_flight == 4
        assert driver.headroom == 0

        wave = driver._live[0]
        queue_depth = len(wave.units) / wave.max_workers
        assert queue_depth == 4.0
        # What the wave actually needs, against what it was given.
        assert one_shard_s * queue_depth > wave.deadline_s

        # Advance to the deadline. No unit has settled: nothing was written.
        clock.advance(deadline_s + 1.0)
        driver._retire()

        assert driver._live == [], "wave retired while its units are unsettled"
        assert driver.headroom == 4, (
            "headroom returned in full while 4 VMs are still computing -- "
            "the driver now believes it can start 4 more"
        )

    def test_the_cap_is_exceeded_once_a_retired_wave_is_replaced(self, storage, clock):
        """The over-commit, made concrete: 8 VMs live against a cap of 4."""
        driver, _ = _driver(storage, clock, max_vms=4)
        deadline_s = 200.0

        driver._buffer_demand(_demand(TILES[0], "offsets", 16, deadline_s))
        driver._flush("offsets")
        first = driver._live[0]

        clock.advance(deadline_s + 1.0)
        driver._retire()

        driver._buffer_demand(_demand(TILES[0], "composite", 16, deadline_s))
        driver._flush("composite")
        second = driver._live[-1]

        # The driver's own view.
        assert driver.in_flight == 4
        # Reality: the first wave was never told to stop. Nothing in the
        # design cancels a retired wave's cluster, and its units are unsettled.
        really_live = first.max_workers + second.max_workers
        assert really_live == 8
        assert really_live > 4, "cap over-commit: retiring a wave by deadline does not stop its VMs"

    @pytest.mark.parametrize("depth", [1, 2, 3, 8, 64])
    def test_capture_is_bounded_by_the_safety_factor(self, storage, clock, depth):
        """Boot amortization is 1 - 1/depth, and depth is capped by safety.

        This is the cost model's capture fraction expressed as a property of
        the code. A wave may only run to completion inside its deadline while
        its queue depth is at most `shard_budget_safety`, so the sustainable
        capture is at most 1 - 1/2.0 = 0.50.
        """
        driver, _ = _driver(storage, clock, max_vms=4)
        one_shard_s = 100.0
        deadline_s = one_shard_s * settings.shard_budget_safety

        driver._buffer_demand(_demand(TILES[0], "offsets", 4 * depth, deadline_s))
        driver._flush("offsets")
        wave = driver._live[0]

        runtime_s = one_shard_s * (len(wave.units) / wave.max_workers)
        survives = runtime_s <= wave.deadline_s
        capture = 1.0 - 1.0 / depth

        if survives:
            assert capture <= 0.5, (
                f"depth {depth} survives its deadline but claims capture {capture:.2f}"
            )
        else:
            assert capture > 0.5 or depth == 1


class TestPreemptionAfterArtifactUpload:
    """A dead cluster whose artifacts landed is a finished stage, not a failure.

    CLAUDE.md states the rule directly: "a dead report is re-checked against
    the bucket first (a fleet whose last task uploaded and then stopped is a
    finished stage)". The single-tile suite pins it
    (`test_a_stopped_cluster_whose_artifacts_landed_is_not_a_failure`). The
    fleet suite's only dead-cluster test pairs `dead_clusters` with `never`,
    so the artifacts are absent and the branch is never taken.
    """

    def test_a_dead_cluster_that_already_wrote_is_not_resubmitted(self, storage, clock):
        from tests.unit.test_fleet_driver import _jobs, _run  # noqa: PLC0415

        plans = _plans(TILES)
        fleet = FakeWaveFleet(storage, plans, clock=clock, dead_clusters={1})
        summary, fleet = _run(storage, clock, TILES, fleet=fleet)

        assert summary.failed == [], (
            "a cluster reported dead after its artifacts landed must not fail tiles"
        )
        assert sorted(summary.completed) == sorted(TILES)
        offsets = fleet.calls_for("offsets")
        assert len(offsets) == 1, (
            "the stage was complete in the bucket; a second wave is pure waste"
        )
        del _jobs
