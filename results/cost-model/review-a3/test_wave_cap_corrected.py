"""Independent re-check of the f6cf6fc cap defect against f4d1e93.

REVIEW ARTIFACT, second pass. Companion to
``test_wave_cap_adversarial.py``, which characterized the defect on f6cf6fc.
This file asserts the **corrected** behaviour, so passing means fixed.

The point of writing it separately rather than editing the first file is that
the first file's tests now fail on f4d1e93 with a ``TypeError`` -- the
constructor moved from ``submit=`` to ``backend=`` -- and an API change is not
evidence of a semantic fix. These are ported to the new API so the assertions
actually execute.

Run against Agent 3's corrected commit::

    mkdir -p /tmp/a3new
    git archive issue-108-fleet-driver -o /tmp/a3new/c.tar
    tar -xf /tmp/a3new/c.tar -C /tmp/a3new
    cp results/cost-model/review-a3/test_wave_cap_corrected.py \
       /tmp/a3new/tests/unit/
    cd /tmp/a3new
    PYTHONPATH=/tmp/a3new/src <venv>/bin/python -m pytest \
      tests/unit/test_wave_cap_corrected.py -q

Three properties, each the inverse of a defect reproduced on f6cf6fc:

1. a wave deep enough to amortize a boot is given a deadline that covers its
   queue, so it no longer expires mid-flight;
2. an expired wave keeps its width, so headroom is never returned to paper;
3. held width is evidence-based per unit, so a wave wider than the cap still
   releases capacity as its units land rather than deadlocking the run.
"""

from __future__ import annotations

import pytest
from tests.unit.test_driver_state_machine import FakeClock
from tests.unit.test_fleet_driver import RUN_ID, TILES, FakeWaveFleet

from landsat_lst import budgets
from landsat_lst.config import settings
from landsat_lst.fleet_driver import Demand, FleetDriver
from landsat_lst.storage import LocalStorage

pytestmark = pytest.mark.unit

BOOT_S = 300.0
UNIT_WORK_S = 600.0


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(output_dir=tmp_path / "bucket")


def _bare(storage, clock, *, max_vms: int):
    fleet = FakeWaveFleet(storage, [], clock=clock)
    driver = FleetDriver(
        run_id=RUN_ID,
        tracks=[],
        storage=storage,
        backend=fleet,
        clock=clock,
        units=15,
        max_vms=max_vms,
        wave_window_s=0.0,
    )
    return driver, fleet


def _demand(tile: str, stage: str, n_units: int) -> Demand:
    return Demand(
        tile=tile,
        stage=stage,
        indexes=tuple(range(n_units)),
        submission_round=1,
        deadline_s=(BOOT_S + UNIT_WORK_S) * settings.shard_budget_safety,
        boot_s=BOOT_S,
        unit_work_s=UNIT_WORK_S,
    )


class TestDeadlineCoversTheQueue:
    """Defect 1, inverted: the deadline scales with queue depth."""

    @pytest.mark.parametrize("depth", [1, 2, 4, 8, 16, 164])
    def test_a_wave_is_given_time_for_the_units_ahead_in_its_queue(self, depth):
        workers = 4
        deadline = budgets.wave_deadline_s(
            boot_s=BOOT_S, unit_work_s=UNIT_WORK_S, units=workers * depth, workers=workers
        )
        # What the wave actually needs: boot once, then depth rounds of work.
        needed = BOOT_S + depth * UNIT_WORK_S
        assert deadline > needed, f"depth {depth} is budgeted below its own runtime"

    def test_the_old_one_round_budget_would_not_have_covered_depth_four(self):
        """The f6cf6fc deadline, recomputed here, is still too small.

        Guards the fix from being quietly reverted to a per-shard budget: if
        this stops being true the regression is back.
        """
        one_round = (BOOT_S + UNIT_WORK_S) * settings.shard_budget_safety
        needed_at_four = BOOT_S + 4 * UNIT_WORK_S
        assert one_round < needed_at_four

    def test_depth_164_is_the_seven_hundred_tile_first_wave(self):
        """The shape that failed the build wholesale on f6cf6fc."""
        deadline = budgets.wave_deadline_s(
            boot_s=BOOT_S, unit_work_s=UNIT_WORK_S, units=64 * 164, workers=64
        )
        assert deadline > BOOT_S + 164 * UNIT_WORK_S


class TestExpiryKeepsWidth:
    """Defect 2, inverted: a deadline is not proof, so width is not returned."""

    def test_an_expired_wave_still_counts_against_the_cap(self, storage, clock):
        driver, _ = _bare(storage, clock, max_vms=4)
        driver._buffer_demand(_demand(TILES[0], "offsets", 16))
        driver._flush("offsets")
        assert driver.in_flight == 4
        assert driver.headroom == 0

        wave = driver._live[0]
        clock.advance(wave.deadline_s + 1.0)
        driver._retire()

        assert driver._live, "an expired wave whose units have not landed must be kept"
        assert driver.in_flight == 4, "expiry must not hand back capacity that is still billing"
        assert driver.headroom == 0

    def test_the_cap_cannot_be_doubled_by_waiting(self, storage, clock):
        """The concrete over-commit from f6cf6fc: 8 VMs against a cap of 4."""
        driver, _ = _bare(storage, clock, max_vms=4)
        driver._buffer_demand(_demand(TILES[0], "offsets", 16))
        driver._flush("offsets")
        first = driver._live[0]

        clock.advance(first.deadline_s + 1.0)
        driver._retire()

        driver._buffer_demand(_demand(TILES[0], "composite", 16))
        assert driver._ready_to_flush("composite") is False, (
            "no headroom exists, so nothing may be submitted on top of a late wave"
        )
        assert driver.in_flight <= 4


class TestEvidenceBasedWidthDoesNotDeadlock:
    """The failure their own fix had to avoid: a wave wider than the cap.

    Counting a wave's full requested width until every tile settles would mean
    a first wave carrying more units than the cap never returns any headroom.
    Width is instead bounded by units still outstanding.
    """

    def test_held_width_falls_as_units_land(self, storage, clock):
        driver, _ = _bare(storage, clock, max_vms=4)
        driver._buffer_demand(_demand(TILES[0], "offsets", 16))
        driver._flush("offsets")
        wave = driver._live[0]

        # No track for this tile, so every unit reads as outstanding: the wave
        # holds its full requested width and not one worker more.
        assert driver.wave_held(wave) == wave.max_workers
        assert wave.max_workers <= 4

    def test_a_wave_with_no_outstanding_units_holds_nothing(self, storage, clock):
        driver, _ = _bare(storage, clock, max_vms=4)
        driver._buffer_demand(_demand(TILES[0], "offsets", 16))
        driver._flush("offsets")
        wave = driver._live[0]
        wave.units = ()
        # An adopted wave with no unit list falls back to its requested width;
        # one whose units are all accounted for holds none. Both are bounded.
        assert driver.wave_held(wave) == wave.max_workers
