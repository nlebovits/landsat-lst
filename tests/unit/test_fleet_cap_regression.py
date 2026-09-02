"""The concurrency-cap defect f6cf6fc shipped, as an executable demonstration.

The defect and the saving are the same mechanism. A wave amortizes provisioning
only through queue depth ``R = ceil(units / max_workers)``; capture is
``1 - 1/R``, so a wave worth submitting always has ``R > 1``. f6cf6fc gave every
wave a deadline of **one shard's** budget regardless of ``R``, so any wave deep
enough to save a boot expired long before it could finish -- and an expired wave
was retired, handing its width back as headroom while its workers were still
running. The next submission then went out into capacity that existed only on
paper.

The consequence is not a slow run. At 700 tiles the first wave is of the order
of 10,500 units against a cap of 64, so ``R`` is about 164 and the wave expires
at roughly one percent of its runtime; every tile in it re-demands,
``shard_barrier_rounds`` is 2, and the build fails wholesale.

This file is written to execute against **both** revisions -- it adapts to the
submitter API each one exposes -- so the fix can be demonstrated rather than
asserted. On f6cf6fc it fails; on the corrected branch it passes.

The measurement is deliberately independent of the driver. Peak concurrency is
computed here, from this module's own record of what was submitted and from the
artifacts on disk, because a cap assertion that reads the driver's own
arithmetic would pass on a driver whose arithmetic is the bug.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from landsat_lst import shards
from landsat_lst.fleet_driver import drive_fleet
from landsat_lst.models import ProcessingJob
from landsat_lst.shard_driver import _expected_keys
from landsat_lst.storage import LocalStorage
from landsat_lst.tiling import parse_tile_name
from tests.unit.shard_fixtures import FakeFleet, make_plan
from tests.unit.test_driver_state_machine import FakeClock

pytestmark = pytest.mark.unit

RUN_ID = "cap-regression"
TILES = ["N40W075", "N40W080"]
#: Units per tile. Two tiles at eight units each, against a cap of four, is
#: R = 4 -- the depth the review reproduced at.
UNITS = 8
CAP = 4
#: Seconds of simulated work before a unit's artifact appears. Measured on the
#: injected clock rather than in polls, deliberately: the two revisions issue
#: different numbers of listings per cycle, so a poll-counted delay would land
#: at different simulated times on each and the comparison would be a confound
#: rather than a demonstration.
#:
#: Chosen to sit between the two deadline models. f6cf6fc gives this wave one
#: shard's budget (~2,250 s), so it expires with every unit still running; the
#: corrected model gives it provisioning plus (R + 1) rounds (~9,450 s at
#: R = 4), so it does not.
UNIT_WORK_S = 4000.0


class DualFleet:
    """A submitter that satisfies both revisions' expectations.

    f6cf6fc calls a plain callable; the corrected revision calls
    ``submit(WaveRequest)`` on a backend object and asks it to declare the
    backend contract. Implementing both is what lets one file run on both.
    """

    name = "dual"

    def __init__(self, storage, plans, *, clock, run_id=RUN_ID):
        self.storage = storage
        self.clock = clock
        self.run_id = run_id
        self.plans = {plan.tile: plan for plan in plans}
        self.writers = {plan.tile: FakeFleet(storage, plan, run_id=run_id) for plan in plans}
        self.calls: list[dict] = []
        self._pending: dict[tuple, int] = {}
        self.peak_in_flight = 0

    # -- contract declaration, for the corrected revision -----------------

    @property
    def guarantees(self):
        from landsat_lst.fleet_backend import BACKEND_CONTRACT

        return frozenset(BACKEND_CONTRACT)

    def wave_name(self, run_id, stage, wave):
        return f"dual-{run_id}-{stage}-w{wave}"

    def classify_failure(self, error):
        from landsat_lst.shard_driver import classify_failure

        return classify_failure(error)

    def probe(self, handle_id):
        del handle_id

    def preflight(self, *, tiles):
        del tiles

    def validate_storage(self, storage):
        del storage

    # -- submission, in either shape --------------------------------------

    def _record(self, stage, units, wave, max_workers):
        pairs = [(tile, int(index)) for tile, index in units]
        self.calls.append(
            {
                "stage": stage,
                "wave": wave,
                "units": pairs,
                "max_workers": max_workers or len(pairs),
                "at": self.clock.now(),
            }
        )
        due = self.clock.now() + UNIT_WORK_S
        for tile, index in pairs:
            self._pending.setdefault((tile, stage, index), due)
        self.observe()
        return len(self.calls)

    def submit(self, request):
        """The corrected revision's interface."""
        from landsat_lst.fleet_backend import WaveHandle

        ident = self._record(request.stage, request.units, request.wave, request.max_workers)
        return WaveHandle(
            id=ident,
            name=self.wave_name(request.run_id, request.stage, request.wave),
            max_workers=request.max_workers,
        )

    def __call__(self, **kwargs):
        """f6cf6fc's interface: a plain callable returning a submission-ish object."""
        ident = self._record(
            kwargs["stage"],
            kwargs["units"],
            kwargs.get("wave", 1),
            kwargs.get("max_workers"),
        )
        return SimpleNamespace(cluster_id=ident, job_id=1, wave=kwargs.get("wave", 1))

    # -- scripted progress -------------------------------------------------

    def tick(self):
        """Land every unit whose simulated work is finished, by the clock."""
        now = self.clock.now()
        for key, due in list(self._pending.items()):
            if now >= due:
                del self._pending[key]
                tile, stage, index = key
                self.writers[tile]._write(stage, index)
        self.observe()

    # -- the independent measurement --------------------------------------

    def _unit_done(self, stage, tile, index):
        plan = self.plans[tile]
        if stage == "export":
            return self.storage.cog_exists(plan.window, tile)
        keys = _expected_keys(plan, stage, shards.shard_root(self.run_id, tile)).get(index)
        if keys is None:
            # Nothing for this unit to produce: past the plan's shard count.
            return True
        return all(self.storage.read_text(k) is not None for k in keys)

    def observe(self):
        """Workers that could be running now, across every wave ever started."""
        total = 0
        for call in self.calls:
            outstanding = sum(
                1
                for tile, index in call["units"]
                if not self._unit_done(call["stage"], tile, index)
            )
            total += min(call["max_workers"], outstanding)
        self.peak_in_flight = max(self.peak_in_flight, total)
        return total


class _Ticking:
    """Storage that advances the scripted fleet once per listing."""

    def __init__(self, storage, fleet):
        self._storage = storage
        self._fleet = fleet

    def __getattr__(self, name):
        return getattr(self._storage, name)

    def list_prefix(self, prefix):
        self._fleet.tick()
        return self._storage.list_prefix(prefix)


def _drive(jobs, *, storage, fleet, clock, max_vms):
    """Call ``drive_fleet`` through whichever submitter API this revision has."""
    params = inspect.signature(drive_fleet).parameters
    common = {
        "run_id": RUN_ID,
        "storage": storage,
        "clock": clock,
        "max_vms": max_vms,
        "wave_window_s": 0.0,
    }
    if "backend" in params:
        return drive_fleet(jobs, backend=fleet, **common)
    return drive_fleet(jobs, submit=fleet, **common)


def test_peak_concurrency_never_exceeds_the_cap_at_queue_depth_four(tmp_path):
    """The regression. Fails on f6cf6fc, passes on the corrected branch.

    Two tiles of eight units against a cap of four is ``R = 4``: exactly the
    shape the consolidation exists to produce, and exactly the shape f6cf6fc
    could not hold a cap through. The assertion is on *observed* peak
    concurrency, not on the width requested per submission, because f6cf6fc
    requested a legal width every single time and still ran twice the cap.
    """
    store = LocalStorage(tmp_path)
    clock = FakeClock()
    plans = [make_plan(tile=name) for name in TILES]
    fleet = DualFleet(store, plans, clock=clock)
    ticking = _Ticking(store, fleet)
    for writer in fleet.writers.values():
        writer.storage = ticking

    # Both revisions read the roster from storage on resume; writing it here
    # keeps the two paths identical.
    store.write_text(
        shards.fleet_manifest_key(RUN_ID),
        json.dumps(
            {
                "run_id": RUN_ID,
                "units": UNITS,
                "tiles": [
                    {"tile": t, "year": 2021, "end_year": 2025, "max_scenes": None} for t in TILES
                ],
            }
        ),
    )

    jobs = [ProcessingJob(tile=parse_tile_name(name), year=2021, end_year=2025) for name in TILES]
    _drive(jobs, storage=ticking, fleet=fleet, clock=clock, max_vms=CAP)

    assert fleet.calls, "no wave was ever submitted"
    assert fleet.peak_in_flight <= CAP, (
        f"peak concurrency {fleet.peak_in_flight} exceeded the cap of {CAP}: "
        "a wave was retired while its workers were still running, so a "
        "resubmission went out into headroom that did not exist"
    )
