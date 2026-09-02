"""Adversarial tests for the consolidated fleet driver, measured from outside it.

Every assertion here is made against something the driver does not compute. The
shipped suite's cap tests measure concurrency with ``min(max_workers,
outstanding)``, which is character-for-character
:meth:`landsat_lst.fleet_driver.FleetDriver.wave_held`, so they cannot fail on a
driver whose release rule is the defect. The oracle in this file is
:class:`~tests.unit.adversarial_fleet_harness.WorkerLedger`: worker identities
opened when a submission asks the substrate for them and closed when the
substrate stops billing them, with the peak taken by interval sweep.

The scenarios are the ones that succeeded against this revision when the
reviewer attacked it, plus the pre-registered invariants that had no test:

- ``tail-over-admission`` (INV-04): the artifact-release rule frees a slot when a
  unit's artifact appears, but a Coiled batch array keeps every VM until the
  array finishes, so the next wave boots on top of workers still billing.
- ``preemption-mid-wave`` and ``probe-raises-during-preemption`` (INV-10): a wave
  killed before anything lands holds its width forever, no round is ever
  resubmitted, and ``drive_fleet`` returns normally with every tile in neither
  the completed nor the failed list.
- ``fail-window-intra-poll`` (INV-06): clearing ``outstanding`` in ``_fail``
  hands a live wave's width back inside one poll, between ``step`` and the next
  ``_retire``. All 1,266 shipped tests pass with that mutation applied.
- ``listing-requests-growth`` (INV-26): the per-poll listing is one *call* at
  every tile count and 1 / 4 / 24 S3 *requests* at 10 / 100 / 700 tiles.
- INV-13 and INV-15: the deadline is the queue-depth formula, and there is a
  queue depth at which the one-round budget stops covering the work.

Four mutation tests close the loop: each disables one required fix and asserts
that the external measurement then reports a violation. A suite that cannot
detect the absence of a fix is not evidence that the fix is present.
"""

from __future__ import annotations

import json
import math

import pytest

from landsat_lst import budgets, shards
from landsat_lst.config import settings
from landsat_lst.fleet_driver import Demand, FleetDriver, PollIndex, TileTrack, _tracks
from tests.unit.adversarial_fleet_harness import (
    MemoryStorage,
    SimBackend,
    SimClock,
    TileWriter,
    WorkerLedger,
    assert_every_tile_settled_once,
    build_simulation,
    jobs_for,
    production_plan,
    settlement,
    stage_terms,
    stub_scientific_work,
    tile_names,
)

pytestmark = pytest.mark.unit

CAP = 64


def _tiles(sim) -> list[str]:
    return [plan.tile for plan in sim.plans]


def _reports_stopped(handle_id, backend):
    """A probe that can confirm the death it was asked about."""
    del handle_id, backend
    return ("stopped", "preempted")


def _no_probe(self, *args, **kwargs) -> None:
    """The mutation: nothing is ever confirmed dead."""
    del self


def _cap_message(sim, peak: int) -> str:
    when = sim.ledger.peak_moment()
    live = sim.ledger.live_at(when)
    waves = sorted({identity.name.rsplit("-vm", 1)[0] for identity in live})
    return (
        f"{peak} workers were live at once against a cap of {sim.cap}; "
        f"at t={when - sim.clock.slept[0] if sim.clock.slept else when:.0f} the "
        f"live identities came from {waves}"
    )


# --------------------------------------------------------------------------
# The fixture itself has to be worth measuring against
# --------------------------------------------------------------------------


def test_the_adversarial_plan_prices_work_at_production_scale():
    """Guards the harness: a toy plan makes every capacity property unreachable.

    ``shard_fixtures.make_plan`` is 1,024 pixels square over four scenes in two
    batches, so ``budgets`` prices a composite unit at about 1.4 s while the
    deadline governing it is thousands of seconds. Nothing about queueing,
    expiry or held capacity can be observed in that regime, and thirteen of the
    fifteen units every driver test asks for have no artifact to produce and so
    land instantly.
    """
    plan = production_plan("N40W075")
    terms = stage_terms(plan)

    assert plan.native_shape == (18000, 18000)
    assert terms["composite"][1] > 600.0, (
        f"a composite unit is priced at {terms['composite'][1]:.1f} s; "
        "below production scale no deadline governs any work"
    )
    assert terms["offsets"][1] > 120.0
    # Every unit the driver asks for has an artifact to produce.
    assert plan.scene_shards == 15


# --------------------------------------------------------------------------
# INV-04 / INV-08: the hard cap, against an external ledger of identities
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_tiles", [50, 200, 700])
def test_live_workers_never_exceed_the_cap_at_build_scale(monkeypatch, n_tiles):
    """INV-04, attack ``tail-over-admission``.

    The ledger counts VMs the substrate was asked to start and stops counting
    one when the substrate stops billing it -- which for ``coiled.batch_run`` is
    when the *array* ends, not when a particular unit's artifact appears. That
    single difference is the whole finding: the driver frees a slot on the
    artifact, the bill does not.

    Runs at 50, 200 and 700 tiles because the property being claimed is a
    property of the build, and because the reviewer's brute-force attempt at 200
    was still running after 37 minutes -- it was performing 700 real offsets
    merges. The merge and the plan rebuild are stubbed here; the state machine
    is not.
    """
    sim = build_simulation(monkeypatch, n_tiles=n_tiles, cap=CAP)
    summary = sim.run()

    assert summary.polls < sim.driver._poll_ceiling(), "the run hit its poll ceiling"
    peak = sim.ledger.peak()
    assert peak <= CAP, _cap_message(sim, peak)


def test_no_submission_asks_for_more_workers_than_the_cap(monkeypatch):
    """INV-05. Read off the submissions the backend received, not off the driver."""
    sim = build_simulation(monkeypatch, n_tiles=50, cap=CAP)
    sim.run()

    widest = max(call["max_workers"] for call in sim.backend.calls)
    assert widest <= CAP


def test_the_cap_is_the_setting_and_not_the_queue_depth(monkeypatch):
    """INV-08: the same code path at two caps, measured externally at both."""
    peaks = {}
    for cap in (7, 31):
        sim = build_simulation(monkeypatch, n_tiles=20, cap=cap, run_id=f"cap{cap}")
        sim.run()
        peaks[cap] = sim.ledger.peak()

    assert peaks[7] <= 7, f"peak {peaks[7]} against a cap of 7"
    assert peaks[31] <= 31, f"peak {peaks[31]} against a cap of 31"


# --------------------------------------------------------------------------
# INV-10: every tile ends exactly once
# --------------------------------------------------------------------------


def test_every_tile_ends_exactly_once_on_a_healthy_run(monkeypatch):
    """The negative control, so the termination assertion is not vacuous."""
    sim = build_simulation(monkeypatch, n_tiles=10, cap=CAP)
    summary = sim.run()

    assert_every_tile_settled_once(_tiles(sim), summary)
    assert len(summary.completed) == 10


def test_every_tile_ends_exactly_once_when_a_wave_is_preempted_before_it_lands(monkeypatch):
    """Attack ``preemption-mid-wave``: the limbo the return value cannot express.

    Wave 1 is killed 400 s in, before any unit has finished booting and
    working, so nothing lands and the backend cannot confirm the death. The wave
    holds its full width forever, headroom never returns, nothing is ever
    resubmitted -- and ``run`` returns normally with every tile in neither list.
    A caller that reads ``summary.failed`` sees a clean run.
    """
    sim = build_simulation(monkeypatch, n_tiles=10, cap=CAP, kill_wave=1, kill_after_s=400.0)
    summary = sim.run()

    assert_every_tile_settled_once(_tiles(sim), summary)


def test_every_tile_ends_exactly_once_when_the_probe_is_unavailable(monkeypatch):
    """Attack ``probe-raises-during-preemption``.

    Same preemption, but the control plane is down as well, which is the
    ordinary correlated failure: the thing that killed the fleet is the thing
    that cannot be asked about it. ``_probe`` logs and swallows, correctly --
    instrumentation never fails a tile -- and then the barrier has no evidence
    it will ever get.
    """
    sim = build_simulation(
        monkeypatch,
        n_tiles=10,
        cap=CAP,
        kill_wave=1,
        kill_after_s=400.0,
        probe_error=RuntimeError("control plane unavailable"),
    )
    summary = sim.run()

    assert_every_tile_settled_once(_tiles(sim), summary)


def test_a_confirmable_death_is_the_control_for_the_two_above(monkeypatch):
    """A probe that answers ``stopped`` must let the run recover.

    Without this the two tests above would pass on a driver that failed every
    tile the moment anything went wrong, which is a different defect wearing the
    same green tick.
    """
    sim = build_simulation(
        monkeypatch,
        n_tiles=10,
        cap=CAP,
        kill_wave=1,
        kill_after_s=400.0,
        probe_answer=_reports_stopped,
        probe_waves=True,
    )
    summary = sim.run()

    assert_every_tile_settled_once(_tiles(sim), summary)


def test_a_resumed_driver_over_an_unconfirmable_wave_still_settles_every_tile(monkeypatch):
    """INV-24/INV-25 and attack ``adopted-wave-holds-full-width``.

    A record written before ``unit_tokens`` existed carries no unit list, so the
    adopted wave holds its full requested width until a probe settles it -- and
    the probe is the thing that is down. The resumed driver then offers zero
    headroom for the life of the run.

    Adoption is exercised through the record on storage, and the outcome is read
    off the summary; neither is the driver's capacity arithmetic.
    """
    run_id = "resume-legacy"
    names = tile_names(4)
    plans = [production_plan(name) for name in names]
    storage = MemoryStorage()
    clock = SimClock()
    ledger = WorkerLedger()
    writers = {plan.tile: TileWriter(storage, plan, run_id=run_id) for plan in plans}
    stub_scientific_work(monkeypatch, writers)
    storage.write_text(
        shards.fleet_submission_key(run_id, "offsets", 1),
        json.dumps(
            {
                "run_id": run_id,
                "stage": "offsets",
                "wave": 1,
                "tiles": list(names),
                "max_workers": CAP,
                "submitted_at": clock.now() - 10_000_000.0,
                "deadline_s": 100.0,
                "handle_id": "previous-cluster",
                "handle_name": "lst-previous",
            }
        ),
    )
    backend = SimBackend(
        storage,
        writers,
        clock=clock,
        ledger=ledger,
        terms=stage_terms(plans[0]),
        probe_error=RuntimeError("control plane unavailable"),
    )
    storage.on_list = backend.tick
    monkeypatch.setattr(settings, "fleet_poll_s", 300.0)
    driver = FleetDriver(
        run_id=run_id,
        tracks=_tracks(jobs_for(names), run_id=run_id, storage=storage, units=15, clock=clock),
        storage=storage,
        backend=backend,
        clock=clock,
        units=15,
        max_vms=CAP,
        wave_window_s=0.0,
    )
    summary = driver.run()

    assert_every_tile_settled_once(names, summary)


# --------------------------------------------------------------------------
# INV-13 / INV-15: the deadline formula, and the depth at which one round fails
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("units", "workers"),
    [(1, 1), (4, 4), (16, 4), (64, 4), (150, 64), (10_500, 64), (7, 3)],
)
def test_the_wave_deadline_is_exactly_the_queue_depth_formula(units, workers):
    """INV-15, ADR-018's ``(boot + (R + 1) * work) * safety``.

    Asserted as an identity rather than as a trend. ``deadline = 2 * units``
    satisfies every monotonicity assertion the shipped deadline test makes and
    is catastrophically wrong; the exact formula is the only thing that
    distinguishes them.
    """
    boot, work = 300.0, 433.0
    expected = (boot + (math.ceil(units / workers) + 1) * work) * settings.shard_budget_safety

    got = budgets.wave_deadline_s(boot_s=boot, unit_work_s=work, units=units, workers=workers)

    assert got == pytest.approx(expected)
    assert budgets.queue_depth(units=units, workers=workers) == math.ceil(units / workers)


def test_the_deadline_covers_the_work_and_the_one_round_budget_stops_doing_so():
    """INV-13, and f6cf6fc's failure boundary named rather than described.

    A wave of ``W`` units running on ``C`` workers takes ``boot + ceil(W/C) *
    work``. The per-shard budget the single-tile driver uses describes one shard
    running immediately, so it covers one boot and one unit's work times the
    safety factor -- and there is a queue depth above which that is simply less
    than the makespan. Both directions are asserted: the queue-depth deadline
    covers every depth, and the one-round budget provably does not.
    """
    plan = production_plan("N40W075")
    boot, work = stage_terms(plan)["offsets"]
    one_round = budgets.stage_budget("offsets", plan).deadline_s

    covered, uncovered = [], []
    for depth in range(1, 11):
        makespan = boot + depth * work
        assert (
            budgets.wave_deadline_s(boot_s=boot, unit_work_s=work, units=depth * 64, workers=64)
            >= makespan
        ), f"the queue-depth deadline does not cover a wave {depth} rounds deep"
        (covered if one_round >= makespan else uncovered).append(depth)

    assert uncovered, (
        "no queue depth was found at which the one-round budget stops covering "
        "the work; the boundary this test exists to pin does not exist"
    )
    assert min(uncovered) > min(covered), "the one-round budget fails even at depth 1"
    assert max(covered) + 1 == min(uncovered), "the boundary is not a single crossover"


def test_a_wave_three_rounds_deep_is_dispatched_once_and_never_expires(monkeypatch):
    """INV-13's observable: zero expiry events, one submission per unit.

    Measured at the backend: every unit was handed to the substrate exactly
    once. An expiring deadline would show as a second dispatch of the same
    ``(stage, tile, index)``, which is a fact about the submissions rather than
    about the driver's opinion of them.
    """
    sim = build_simulation(monkeypatch, n_tiles=8, cap=40, run_id="depth3")
    sim.run()

    units = 8 * 15
    assert budgets.queue_depth(units=units, workers=40) == 3
    redispatched = [
        (tile, index)
        for call in sim.backend.calls_for("offsets")
        for tile, index in call["units"]
        if sim.backend.dispatched("offsets", tile, index) > 1
    ]
    assert not redispatched, (
        f"{len(redispatched)} offsets unit(s) were dispatched twice, so a "
        f"deadline expired on work that was still queued: {redispatched[:6]}"
    )
    assert sim.backend.calls_for("offsets") and len(sim.backend.calls_for("offsets")) == 1


# --------------------------------------------------------------------------
# INV-26: listing cost, measured in requests
# --------------------------------------------------------------------------


def _one_poll_listing_cost(monkeypatch, n_tiles: int, *, units: int = 15) -> dict:
    """Requests, calls and keys for exactly one poll of a mid-run build.

    Mid-run rather than empty: an empty bucket lists zero keys and pages once
    whatever the tile count, which is how a listing test passes vacuously.
    """
    run_id = f"listing-{n_tiles}"
    names = tile_names(n_tiles)
    plans = [production_plan(name, scene_shards=units) for name in names]
    storage = MemoryStorage()
    clock = SimClock()
    writers = {plan.tile: TileWriter(storage, plan, run_id=run_id) for plan in plans}
    stub_scientific_work(monkeypatch, writers)
    for writer in writers.values():
        for index in range(units):
            writer.write("offsets", index)

    backend = SimBackend(
        storage, writers, clock=clock, ledger=WorkerLedger(), terms=stage_terms(plans[0])
    )
    driver = FleetDriver(
        run_id=run_id,
        tracks=_tracks(jobs_for(names), run_id=run_id, storage=storage, units=units, clock=clock),
        storage=storage,
        backend=backend,
        clock=clock,
        units=units,
        max_vms=CAP,
        wave_window_s=0.0,
    )
    before = (storage.requests, storage.calls, storage.keys_returned)
    driver.index.refresh()
    for track in driver.tracks:
        track.step()
    return {
        "tiles": n_tiles,
        "requests": storage.requests - before[0],
        "calls": storage.calls - before[1],
        "keys": storage.keys_returned - before[2],
        "driver_counter": driver.index.listings,
    }


def test_listing_cost_per_poll_is_optimal_and_flat_in_calls(monkeypatch):
    """INV-26, attack ``listing-requests-growth``.

    The original assertion demanded that requests grow less than tenfold for a
    hundredfold tile count. They cannot: keys are 17 per tile at every scale and
    ``ListObjectsV2`` returns at most 1,000 of them, so requests are exactly
    ``ceil(keys/1000)`` and therefore linear in tiles by arithmetic. No driver
    can pass it, and a bound no implementation can meet tests nothing.

    What the invariant meant to catch is a driver that lists per tile instead of
    per shared prefix. That shows up in two places: the call count stays flat,
    and every request the driver does pay carries a full page. Real cost at 700
    tiles is about 9,300 LISTs for a whole build, roughly four cents.
    """
    small = _one_poll_listing_cost(monkeypatch, 10)
    large = _one_poll_listing_cost(monkeypatch, 700)

    assert small["driver_counter"] == large["driver_counter"] == 1, (
        "the driver must list once per shared prefix, not once per tile: "
        f"{small['driver_counter']} call(s) at 10 tiles and "
        f"{large['driver_counter']} at 700"
    )
    for label, seen in (("10 tiles", small), ("700 tiles", large)):
        optimal = -(-seen["keys"] // 1000) or 1
        assert seen["requests"] == optimal, (
            f"at {label} the poll paid {seen['requests']} S3 request(s) for "
            f"{seen['keys']} keys, where a full page each would cost {optimal} "
            "-- a wasted page means a prefix is being listed more than once"
        )


def _eager_wave_held(self, wave):
    """``min(max_workers, outstanding)`` -- the release rule under test, restored."""
    if not wave.units:
        return wave.max_workers
    return min(wave.max_workers, self.wave_outstanding(wave))


def test_mutation_restoring_the_artifact_release_rule_is_detected(monkeypatch):
    """M1. Capacity released when an artifact appears rather than when a VM stops.

    The mutation is the rule ``tail-over-admission`` exploits. With it applied,
    the external ledger must report a peak above the cap; a suite that stayed
    green here would be measuring the driver with the driver's own formula,
    which is exactly what five shipped cap tests do.
    """
    monkeypatch.setattr(FleetDriver, "wave_held", _eager_wave_held)
    sim = build_simulation(monkeypatch, n_tiles=50, cap=CAP, run_id="m1")
    sim.run()

    assert sim.ledger.peak() > CAP, (
        "the artifact-release rule was restored and the ledger still saw no "
        "over-admission -- the cap oracle is not measuring what it claims to"
    )


def test_mutation_clearing_outstanding_in_fail_is_detected(monkeypatch):
    """M2, ``fail-window-intra-poll``. Confirmed undetected by all 1,266 tests.

    ``TileTrack._fail`` deliberately does not clear ``outstanding``: a tile
    giving up is not the same fact as its workers stopping. Restoring the clear
    hands a live wave's width back inside a single poll, because the poll order
    is ``_retire`` then ``step`` then ``_flush`` and nothing re-reads the
    evidence in between.

    Driven at the object level in exactly that order, and measured with the
    ledger rather than with ``in_flight``.
    """
    cap = 4
    run_id = "m2"
    names = tile_names(2)
    plans = [production_plan(name) for name in names]
    storage = MemoryStorage()
    clock = SimClock()
    ledger = WorkerLedger()
    writers = {plan.tile: TileWriter(storage, plan, run_id=run_id) for plan in plans}
    stub_scientific_work(monkeypatch, writers)
    backend = SimBackend(
        storage,
        writers,
        clock=clock,
        ledger=ledger,
        terms=stage_terms(plans[0]),
        never={("offsets", names[0], index) for index in range(4)},
    )
    driver = FleetDriver(
        run_id=run_id,
        tracks=_tracks(jobs_for(names), run_id=run_id, storage=storage, units=4, clock=clock),
        storage=storage,
        backend=backend,
        clock=clock,
        units=4,
        max_vms=cap,
        wave_window_s=0.0,
    )
    holder, waiter = driver.tracks
    holder.outstanding["offsets"] = {0, 1, 2, 3}
    driver._buffer_demand(
        Demand(
            tile=holder.tile,
            stage="offsets",
            indexes=(0, 1, 2, 3),
            submission_round=1,
            deadline_s=600.0,
            boot_s=300.0,
            unit_work_s=600.0,
        )
    )
    driver._flush("offsets")
    assert ledger.peak() == cap, "the first wave should hold the whole cap"

    original_fail = TileTrack._fail

    def mutated_fail(self, reason: str) -> None:
        self.outstanding.clear()
        original_fail(self, reason)

    monkeypatch.setattr(TileTrack, "_fail", mutated_fail)
    holder._fail("rounds exhausted")

    # The rest of the same poll: the waiting tile demands, and the buffer flushes.
    driver._buffer_demand(
        Demand(
            tile=waiter.tile,
            stage="offsets",
            indexes=(0, 1, 2, 3),
            submission_round=1,
            deadline_s=600.0,
            boot_s=300.0,
            unit_work_s=600.0,
        )
    )
    driver._flush("offsets")

    assert ledger.peak() > cap, (
        "a tile failed while its wave was still holding four workers, a second "
        "wave was submitted in the same poll, and the ledger saw no over-"
        "admission -- the fail window is not being measured"
    )


def test_mutation_disabling_the_death_probe_is_detected(monkeypatch):
    """M3. Silencing confirmation must change the outcome, and never make limbo.

    The original assertion required that some tile end in *neither* list. That
    is the silent limbo four other tests in this file forbid, and which
    ``run()`` now makes unreachable by settling every straggler on every exit
    path. A mutation test may not demand the defect the suite exists to
    prevent, so this asserts what it meant to assert: that the confirmation
    path carries information. Both runs are performed here, so the comparison
    cannot pass because one of them was configured wrongly.
    """
    live = build_simulation(
        monkeypatch,
        n_tiles=10,
        cap=CAP,
        run_id="m3-live",
        kill_wave=1,
        kill_after_s=400.0,
        probe_answer=_reports_stopped,
        probe_waves=True,
    )
    live_summary = live.run()
    live_where = settlement(_tiles(live), live_summary)

    monkeypatch.setattr(FleetDriver, "_probe", _no_probe)
    silenced = build_simulation(
        monkeypatch,
        n_tiles=10,
        cap=CAP,
        run_id="m3-silenced",
        kill_wave=1,
        kill_after_s=400.0,
        probe_answer=_reports_stopped,
        probe_waves=True,
    )
    silenced_summary = silenced.run()
    silenced_where = settlement(_tiles(silenced), silenced_summary)

    for label, where in (("confirmable", live_where), ("silenced", silenced_where)):
        assert all(len(ends) == 1 for ends in where.values()), (
            f"the {label} run left a tile in neither list or in both: "
            f"{ {k: v for k, v in where.items() if len(v) != 1} }"
        )

    assert live_summary.submissions > silenced_summary.submissions, (
        "confirmation must let the driver reclaim the dead wave's capacity and "
        f"resubmit: {live_summary.submissions} submission(s) with the probe and "
        f"{silenced_summary.submissions} without"
    )
    assert live_summary.wall_s * 2 < silenced_summary.wall_s, (
        "a confirmed death must settle the run promptly, where an unconfirmable "
        f"one grinds to the stall detector: {live_summary.wall_s:.0f}s with the "
        f"probe against {silenced_summary.wall_s:.0f}s without"
    )


def test_mutation_unsharing_the_poll_listing_is_detected(monkeypatch):
    """M4. One listing per tile per poll must show as request growth.

    ``PollIndex`` exists to make the driver's request rate independent of the
    tile count. Bypassing its cache is the regression it guards, and the
    measurement has to be at the storage boundary because the driver's own
    counter is incremented in the same place either way.
    """
    monkeypatch.setattr(
        PollIndex, "list_prefix", lambda self, prefix: self._storage.list_prefix(prefix)
    )
    small = _one_poll_listing_cost(monkeypatch, 10)
    large = _one_poll_listing_cost(monkeypatch, 100)

    # Un-shared, one poll costs about three listings per tile, so a tenfold
    # tile count is a tenfold call count. The bound is loose on purpose: what
    # has to be detected is the change of exponent, not a particular constant.
    assert large["calls"] > 5 * small["calls"], (
        f"the shared per-poll listing was bypassed and one poll still cost "
        f"{small['calls']} calls at 10 tiles and {large['calls']} at 100 -- the "
        f"listing instrument is measuring the wrong boundary"
    )
