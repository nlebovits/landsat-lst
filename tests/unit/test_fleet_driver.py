"""The consolidated fleet driver as a state machine, tile by interleaved tile.

Same premise as ``test_driver_state_machine``: the driver spends hours waiting
and its bugs live in the waiting, so time is injected and every scenario runs
in milliseconds against a scripted fleet and local storage. What is new here is
that the machine has *many* tiles in it at once, and the three properties that
justify the whole module are properties of the interleaving:

- one submission carries units from many tiles, and the submission count does
  not grow with the tile count;
- a slow or dead tile advances nobody else's barrier and stalls nobody else's
  wave;
- the VM cap is a hard cap across the whole run, enforced against live waves
  rather than against submissions.

Each of those is asserted directly rather than inferred from a wall clock. The
per-tile decision table is deliberately the ADR-016 one, so the scenarios that
pin *it* -- artifacts decide, records are per tile, rounds are counted across
drivers, terminal beats transient -- are re-run here against the fleet rather
than restated::

    step --(nothing missing)-----------> next stage
    step --(live record)---------------> watch, demand nothing
    step --(no record, rounds left)----> Demand
    step --(no rounds left)------------> tile failed, run continues
    flush --(headroom full)------------> wave
    flush --(window elapsed)-----------> wave
    flush --(nobody left to join)------> wave
    submit --(terminal)----------------> FleetAborted, run stops
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from landsat_lst import shards
from landsat_lst.config import settings
from landsat_lst.fleet_backend import (
    BACKEND_CONTRACT,
    CoiledFleetBackend,
    WaveHandle,
    WaveRequest,
    check_contract,
)
from landsat_lst.fleet_driver import (
    LIST_PAGE_KEYS,
    Demand,
    FleetAborted,
    FleetDriver,
    PollIndex,
    TileTrack,
    drive_fleet,
    read_manifest,
    resume_fleet,
    write_manifest,
)
from landsat_lst.models import ProcessingJob
from landsat_lst.shard_driver import (
    _bootstrap_deadline_s,
    _expected_keys,
    _submission_records,
    classify_failure,
)
from landsat_lst.storage import PRODUCTS, LocalStorage
from landsat_lst.tiling import parse_tile_name
from tests.unit.shard_fixtures import FakeFleet, make_plan, publish_plan
from tests.unit.test_driver_state_machine import FakeClock

pytestmark = pytest.mark.unit

RUN_ID = "fleet-test"
#: Four real tile names. Distinct roots, distinct plans, one run id.
TILES = ["N40W075", "N40W080", "N35W075", "N35W080"]


def _jobs(names) -> list[ProcessingJob]:
    return [ProcessingJob(tile=parse_tile_name(name), year=2021, end_year=2025) for name in names]


def _demand(tile, stage, indexes, *, round_no=1, deadline_s=100.0, boot=0.0, work=100.0):
    """A hand-built demand, for the cap arithmetic tests."""
    return Demand(
        tile=tile,
        stage=stage,
        indexes=indexes,
        submission_round=round_no,
        deadline_s=deadline_s,
        boot_s=boot,
        unit_work_s=work,
    )


class FakeWaveFleet:
    """A :class:`~landsat_lst.fleet_backend.FleetBackend` that bills nothing.

    Not a mock of the Coiled backend -- a *second implementation* of the same
    contract, which is the point of having one. It declares the whole of
    :data:`~landsat_lst.fleet_backend.BACKEND_CONTRACT` and satisfies each
    guarantee honestly: it runs every unit whatever ``max_workers`` says
    (``queues_surplus``), returns without waiting (``fire_and_forget``), hands
    back a serializable id (``opaque_handle``), and answers ``None`` from
    ``probe`` unless a death was scripted (``probe_is_advisory``).

    One :class:`~tests.unit.shard_fixtures.FakeFleet` per tile does the actual
    writing, so a fleet run and a single-tile run agree, key for key, about what
    a finished shard looks like.
    """

    name = "fake"
    guarantees = frozenset(BACKEND_CONTRACT)

    def __init__(
        self,
        storage,
        plans,
        *,
        clock,
        run_id: str = RUN_ID,
        never: set | None = None,
        heal: bool = False,
        lands_after: dict | None = None,
        raise_once: Exception | None = None,
        raise_always: Exception | None = None,
        dead_handles: set | None = None,
        claims_export: bool = True,
    ) -> None:
        self.storage = storage
        self.clock = clock
        self.run_id = run_id
        self.plans = {plan.tile: plan for plan in plans}
        self.writers = {
            plan.tile: FakeFleet(
                storage, plan, run_id=run_id, claims_export=claims_export, heal=heal
            )
            for plan in plans
        }
        #: ``(tile, stage, index)`` triples this fleet refuses to complete.
        self.never = set(never or ())
        self.heal = heal
        #: ``(tile, stage, index) -> polls`` before the artifact appears, so a
        #: barrier has to watch rather than find the work already done.
        self._lands_after = dict(lands_after or {})
        self._pending: dict[tuple, int] = {}
        self._raise_once = raise_once
        self._raise_always = raise_always
        #: Cluster ids the probe should report dead.
        self.dead_handles = set(dead_handles or ())
        #: One entry per submission: everything a gate needs to assert on.
        self.calls: list[dict] = []
        self.submit_attempts = 0
        #: Highest concurrency this fleet ever had, computed from its own
        #: records and the bucket rather than from anything the driver says.
        #: A cap assertion against the driver's own arithmetic would pass on a
        #: driver whose arithmetic is the bug.
        self.peak_in_flight = 0
        self.live_series: list[int] = []

    def _unit_done(self, stage: str, tile: str, index: int) -> bool:
        plan = self.plans.get(tile)
        if plan is None:
            # A tile this fleet has no plan for is a pure queue-arithmetic
            # fixture: nothing it submits ever lands, which is the honest
            # reading for a capacity test.
            return False
        root = shards.shard_root(self.run_id, tile)
        if stage == "export":
            return self.storage.cog_exists(plan.window, tile)
        keys = _expected_keys(plan, stage, root).get(index)
        if keys is None:
            # A unit past the plan's clamped shard count has nothing to
            # produce: the worker reads the plan, finds no slice of its own,
            # and exits. Counting it as forever-outstanding would make this
            # measurement -- not the driver -- the thing that is wrong.
            return True
        return all(self.storage.read_text(key) is not None for key in keys)

    def observe(self) -> int:
        """Workers that could be running right now, across every wave started.

        A batch array holds every VM it started until the array finishes, so a
        wave holds its full requested width for as long as any of its units is
        still absent, and nothing once they have all landed. Measuring it as
        one worker per outstanding unit measures work rather than machines, and
        an over-admission at the tail then reads as being inside the cap: the
        driver freed slots the substrate had not, the next wave booted on top,
        and the real concurrency was the sum of both.

        Sampled at every submission and every tick, so the recorded peak covers
        the interleavings where a resubmission overlaps a wave that has not
        finished.
        """
        total = 0
        for call in self.calls:
            outstanding = any(
                not self._unit_done(call["stage"], tile, index) for tile, index in call["units"]
            )
            total += call["max_workers"] if outstanding else 0
        self.peak_in_flight = max(self.peak_in_flight, total)
        #: Every sample, so a test can ask whether the fleet was ever left idle
        #: with work still queued rather than only how high it got.
        self.live_series.append(total)
        return total

    # -- the FleetBackend interface ---------------------------------------

    def wave_name(self, run_id, stage, wave):
        return f"fake-{run_id}-{stage}-w{wave}"

    def classify_failure(self, error):
        """Delegated, because the driver's reaction is what is under test here.

        A real alternative backend would map *its own* control-plane errors.
        What the contract fixes is only that an unrecognized one be transient.
        """
        return classify_failure(error)

    def preflight(self, *, tiles):
        """No identity, no balance: this backend starts nothing and spends nothing."""

    def validate_storage(self, storage):
        """Any storage will do: the workers are in this process."""

    def submit(self, request: WaveRequest) -> WaveHandle:
        self.submit_attempts += 1
        if self._raise_always is not None:
            raise self._raise_always
        if self._raise_once is not None:
            error, self._raise_once = self._raise_once, None
            raise error

        pairs = [(tile, int(index)) for tile, index in request.units]
        self.calls.append(
            {
                "stage": request.stage,
                "wave": request.wave,
                "units": pairs,
                "tiles": sorted({tile for tile, _ in pairs}),
                "max_workers": request.max_workers,
                "at": self.clock.now(),
            }
        )
        # ``queues_surplus``: every unit runs, however few workers were asked
        # for. A backend that dropped the surplus would pass every other test
        # here and silently lose slabs in production.
        for tile, index in pairs:
            if (tile, request.stage, index) in self.never:
                if self.heal:
                    self.never.discard((tile, request.stage, index))
                continue
            if tile not in self.writers:
                continue
            delay = self._lands_after.get((tile, request.stage, index), 0)
            if delay:
                self._pending[tile, request.stage, index] = delay
                continue
            self.writers[tile]._write(request.stage, index)
        self.observe()
        return WaveHandle(
            id=len(self.calls),
            name=self.wave_name(request.run_id, request.stage, request.wave),
            max_workers=request.max_workers,
        )

    # -- scripting --------------------------------------------------------

    def tick(self) -> None:
        """One poll's worth of progress for every pending unit."""
        for key in list(self._pending):
            self._pending[key] -= 1
            if self._pending[key] <= 0:
                del self._pending[key]
                tile, stage, index = key
                self.writers[tile]._write(stage, index)
        self.observe()

    def probe(self, handle_id):
        if handle_id in self.dead_handles:
            return ("stopped", "scripted death")
        return None

    # -- assertions -------------------------------------------------------

    @property
    def stages(self) -> list[str]:
        return [call["stage"] for call in self.calls]

    def calls_for(self, stage: str) -> list[dict]:
        return [call for call in self.calls if call["stage"] == stage]


class TickingStorage:
    """Local storage that advances the fleet one step per artifact listing.

    A poll is what makes time pass in a real run, so a poll is what makes the
    scripted units progress here.
    """

    def __init__(self, storage, fleet: FakeWaveFleet) -> None:
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
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


def _plans(names=TILES):
    return [make_plan(tile=name) for name in names]


def driver_deadline(summary, stage: str, wave: int) -> float:
    """The horizon one wave was actually given, for the timing assertions.

    Read back off the summary rather than recomputed, so a test cannot quietly
    disagree with the driver about how long a wave was allowed to run -- which
    is the disagreement that produced the defect in the first place.
    """
    for record in summary.waves:
        if record.stage == stage and record.wave == wave:
            return record.deadline_s
    msg = f"no {stage} wave {wave} in this run"
    raise AssertionError(msg)


def _driver_for(names, *, storage, backend, clock, run_id=RUN_ID, max_vms=None):
    """A driver over real tracks, for the tests that need the object itself."""
    from landsat_lst.fleet_driver import _tracks

    return FleetDriver(
        run_id=run_id,
        tracks=_tracks(_jobs(names), run_id=run_id, storage=storage, units=15, clock=clock),
        storage=storage,
        backend=backend,
        clock=clock,
        units=15,
        max_vms=max_vms,
        wave_window_s=0.0,
    )


def _bare_fleet(storage, clock, *, max_vms):
    """An empty driver plus its backend, for cap and queue arithmetic."""
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


def _run(storage, clock, names=TILES, *, fleet=None, run_id=RUN_ID, **kwargs):
    """Drive a fleet over ``names`` with a scripted submitter. Returns both."""
    plans = _plans(names)
    fleet = fleet or FakeWaveFleet(storage, plans, clock=clock, run_id=run_id)
    fleet.storage = storage
    ticking = TickingStorage(storage, fleet)
    for writer in fleet.writers.values():
        writer.storage = ticking
    summary = drive_fleet(
        _jobs(names),
        run_id=run_id,
        storage=ticking,
        backend=fleet,
        clock=clock,
        wave_window_s=kwargs.pop("wave_window_s", 0.0),
        **kwargs,
    )
    return summary, fleet


# ---------------------------------------------------------------------------


class TestConsolidation:
    """One array per stage per wave, carrying many tiles. The headline gate."""

    def test_one_offsets_wave_carries_every_tile(self, storage, clock):
        summary, fleet = _run(storage, clock)

        offsets = fleet.calls_for("offsets")
        assert len(offsets) == 1
        assert offsets[0]["tiles"] == sorted(TILES)
        assert summary.submissions_for("offsets") == 1

    def test_submission_count_does_not_scale_with_tile_count(self, tmp_path, clock):
        """The gate the issue names. Three tiles and twelve, same submissions."""
        counts = {}
        for n, names in ((3, TILES[:3]), (12, [f"N{30 + i}W075" for i in range(12)])):
            store = LocalStorage(tmp_path / f"run{n}")
            summary, _ = _run(store, FakeClock(), names, run_id=f"fleet-{n}")
            assert len(summary.completed) == len(names)
            counts[n] = summary.submissions

        assert counts[3] == counts[12]

    def test_a_wave_queues_more_units_than_it_starts_vms(self, storage, clock):
        """Where the boot saving comes from: surplus units queue onto booted VMs.

        The cap bounds ``max_workers``; it must never bound the unit list, or a
        tile's demand would be split across waves and the remainder would cost
        it a barrier round it never used.
        """
        summary, fleet = _run(storage, clock, max_vms=2)

        offsets = fleet.calls_for("offsets")[0]
        assert offsets["max_workers"] == 2
        assert len(offsets["units"]) > 2
        assert len(offsets["tiles"]) == len(TILES)
        assert sorted(summary.completed) == sorted(TILES)

    def test_every_tile_completes(self, storage, clock):
        summary, _ = _run(storage, clock)

        assert sorted(summary.completed) == sorted(TILES)
        assert summary.failed == []

    def test_units_are_tile_qualified_tokens(self, storage, clock):
        """A wave's values name the tile, because one command serves them all."""
        _, fleet = _run(storage, clock)

        pairs = fleet.calls_for("offsets")[0]["units"]
        tokens = [shards.fleet_unit_token(tile, index) for tile, index in pairs]
        assert shards.parse_fleet_unit(tokens[0]) == pairs[0]
        assert len({tile for tile, _ in pairs}) == len(TILES)


class TestBackendAbstraction:
    """The state machine depends on an interface, not on Coiled.

    The point is not tidiness. Boot amortization is a property of the
    submission substrate, so "would AWS Batch do this cheaper" has to be
    answerable without rewriting the driver -- and answerable against a written
    contract rather than by reading the driver and guessing.
    """

    def test_the_driver_refuses_a_backend_that_has_not_declared_the_contract(self, storage, clock):
        """A backend missing ``queues_surplus`` is not a slow fleet; it is a
        fleet whose whole reason to exist is absent. Finding that out from a
        bill is the expensive way.
        """

        class Partial(FakeWaveFleet):
            guarantees = frozenset({"fire_and_forget"})

        with pytest.raises(ValueError, match="queues_surplus"):
            FleetDriver(
                run_id=RUN_ID,
                tracks=[],
                storage=storage,
                backend=Partial(storage, [], clock=clock),
                clock=clock,
                units=2,
            )

    def test_check_contract_names_every_missing_guarantee(self):
        class Nothing:
            name = "nothing"
            guarantees = frozenset()

        with pytest.raises(ValueError) as excinfo:
            check_contract(Nothing())

        for guarantee in BACKEND_CONTRACT:
            assert guarantee in str(excinfo.value)

    def test_the_coiled_backend_declares_the_whole_contract(self):
        check_contract(CoiledFleetBackend())

    def test_a_non_coiled_backend_is_a_first_class_implementation(self, storage, clock):
        """The suite's own backend drives a complete run, Coiled uninstalled."""
        summary, fleet = _run(storage, clock)

        assert fleet.name == "fake"
        assert sorted(summary.completed) == sorted(TILES)

    def test_nothing_coiled_shaped_crosses_the_boundary(self, storage, clock):
        """What the driver keeps of a submission is an id, a name, and a width.

        If a cluster id, a ``ServerError`` or a ``spot_policy`` were reachable
        from a :class:`Wave`, the next backend would have to impersonate Coiled
        rather than merely satisfy the contract.
        """
        _, fleet = _run(storage, clock)
        wave = fleet.calls[0]

        assert set(WaveHandle(id=1, name="n", max_workers=2).__dict__) == {
            "id",
            "name",
            "max_workers",
        }
        assert set(wave) == {"stage", "wave", "units", "tiles", "max_workers", "at"}

    def test_the_request_carries_only_neutral_terms(self, storage, clock):
        request = WaveRequest(
            stage="offsets",
            run_id=RUN_ID,
            units=(("N40W075", 0), ("N35W080", 1)),
            wave=1,
            max_workers=2,
        )

        assert request.tiles == ["N40W075", "N35W080"]
        assert set(request.__dict__) == {
            "stage",
            "run_id",
            "units",
            "wave",
            "max_workers",
            "fleet_units",
        }

    def test_the_coiled_backend_hands_back_an_opaque_handle(self, monkeypatch):
        """``opaque_handle``: an id the driver can persist and probe with.

        Patched at ``submit_fleet_stage`` rather than at ``coiled.batch_run``,
        deliberately: the latter reaches ``_worker_environ`` and a real STS
        call, which would make this test read the machine rather than the code.
        What the Coiled *request* looks like is pinned in ``test_batch``, where
        that plumbing is already stubbed.
        """
        from landsat_lst import batch

        captured: dict = {}

        def fake_submit(**kwargs):
            captured.update(kwargs)
            return batch.FleetStageSubmission(
                stage=kwargs["stage"],
                units=list(kwargs["units"]),
                cluster_id=4242,
                job_id=77,
                command="",
                name="lst-abc-fleet-offse-w2",
                wave=kwargs["wave"],
                max_workers=kwargs["max_workers"],
            )

        monkeypatch.setattr(batch, "submit_fleet_stage", fake_submit)

        handle = CoiledFleetBackend().submit(
            WaveRequest(
                stage="offsets",
                run_id="r1",
                units=(("N40W075", 0), ("N40W075", 1), ("N35W080", 0)),
                wave=2,
                max_workers=2,
                fleet_units=8,
            )
        )

        assert handle.id == 4242
        assert handle.name == "lst-abc-fleet-offse-w2"
        assert handle.max_workers == 2
        # opaque_handle: it has to survive a round trip through the wave record.
        assert json.loads(json.dumps(handle.id)) == 4242
        # queues_surplus is asked for, not assumed: three units, two workers.
        assert len(captured["units"]) == 3
        assert captured["max_workers"] == 2
        assert captured["fleet_units"] == 8

    def test_the_driver_module_imports_nothing_backend_specific(self):
        """The boundary, checked structurally rather than by reading.

        Prose in this module names Coiled freely -- the history is the reason
        the contract says what it says. What must not appear is a *dependency*:
        no coiled import, and nothing from ``landsat_lst.batch``, which is where
        ``batch_run``, the cluster naming and the spot policy live.
        """
        import ast
        import inspect

        from landsat_lst import fleet_driver

        tree = ast.parse(inspect.getsource(fleet_driver))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        assert not [name for name in imported if "coiled" in name.lower()]
        assert "landsat_lst.batch" not in imported

    def test_the_spot_policy_is_never_named_outside_the_backends(self):
        """``no_silent_cost_substitution`` is expressed once, behind the boundary."""
        import inspect

        from landsat_lst import fleet_driver

        assert "spot_policy" not in inspect.getsource(fleet_driver)


class TestConcurrencyCap:
    """``fleet_max_vms`` is a hard cap across the run, not per stage."""

    def test_no_wave_ever_starts_more_vms_than_the_cap(self, storage, clock):
        _, fleet = _run(storage, clock, max_vms=3)

        assert fleet.calls
        assert all(call["max_workers"] <= 3 for call in fleet.calls)

    def test_headroom_is_spoken_for_by_live_waves(self, storage, clock):
        """A second stage waits for headroom rather than racing the first."""
        driver, _track, _fleet = self._bare_driver(storage, clock, max_vms=4)
        assert driver.headroom == 4

        driver._buffer_demand(_demand(TILES[0], "offsets", (0, 1, 2, 3)))
        driver._flush("offsets")

        assert driver.in_flight == 4
        assert driver.headroom == 0
        # With no headroom nothing else can be submitted, whatever is buffered.
        driver._buffer_demand(_demand(TILES[0], "composite", (0,)))
        assert driver._ready_to_flush("composite") is False

    def test_held_capacity_is_the_arrays_width_until_the_array_ends(self, storage, clock):
        """A landed artifact is a finished unit, not a released VM.

        A batch array keeps every VM it started until the array itself
        finishes: the worker that published this artifact takes the next unit
        off the queue, or waits in a cluster that is still billing. Counting
        the width down per landed artifact measures work rather than machines,
        and the difference is a whole wave at the tail -- the driver freed
        slots the substrate had not, the next wave booted on top, and a run
        capped at 64 peaked at 82.

        The evidence is read from the bucket rather than from the track,
        because the bucket is what the driver is required to believe.
        """
        driver, _track, fleet = self._bare_driver(
            storage, clock, max_vms=8, never={(TILES[0], "offsets", 0), (TILES[0], "offsets", 1)}
        )
        driver._buffer_demand(_demand(TILES[0], "offsets", (0, 1)))
        driver._flush("offsets")
        driver.index.refresh()
        driver._retire()
        assert driver.in_flight == 2
        assert driver.headroom == 6

        fleet.writers[TILES[0]]._write("offsets", 0)
        driver.index.refresh()
        driver._retire()
        assert driver.in_flight == 2, "half a wave's artifacts is not half a wave's VMs"
        assert driver.capacity_ledger()["waves"][0]["outstanding"] == [
            shards.fleet_unit_token(TILES[0], 1)
        ]

        fleet.writers[TILES[0]]._write("offsets", 1)
        driver.index.refresh()
        driver._retire()
        assert driver.in_flight == 0
        assert driver._live == []

    def _bare_driver(self, storage, clock, *, max_vms, never=()):
        """A driver with one planned track and no scripted work, for cap arithmetic.

        ``never`` stops the fake from landing a unit the moment it is submitted,
        which is what lets a test choose when the evidence appears.
        """
        plans = _plans(TILES[:1])
        fleet = FakeWaveFleet(storage, plans, clock=clock, never=set(never))
        track = TileTrack(
            tile=TILES[0],
            run_id=RUN_ID,
            root=shards.shard_root(RUN_ID, TILES[0]),
            storage=storage,
            units=2,
            clock=clock,
            plan=plans[0],
        )
        driver = FleetDriver(
            run_id=RUN_ID,
            tracks=[track],
            storage=storage,
            backend=fleet,
            clock=clock,
            units=2,
            max_vms=max_vms,
        )
        fleet.writers[TILES[0]].storage = storage
        return driver, track, fleet


class TestIndependentBarriers:
    """One tile's trouble is one tile's trouble."""

    def test_a_slow_tile_does_not_hold_the_others(self, storage, clock):
        """The fast tiles reach composite while the slow one is still in offsets."""
        slow = TILES[0]
        plans = _plans()
        fleet = FakeWaveFleet(
            storage,
            plans,
            clock=clock,
            lands_after={(slow, "offsets", 0): 6, (slow, "offsets", 1): 6},
        )
        summary, fleet = _run(storage, clock, fleet=fleet)

        composite = fleet.calls_for("composite")
        assert composite, "no composite wave was ever submitted"
        first = composite[0]
        assert slow not in first["tiles"]
        assert len(first["tiles"]) == len(TILES) - 1
        assert sorted(summary.completed) == sorted(TILES)

    def test_a_failing_tile_does_not_fail_the_run(self, storage, clock):
        broken = TILES[1]
        plans = _plans()
        fleet = FakeWaveFleet(
            storage,
            plans,
            clock=clock,
            never={(broken, "offsets", 0), (broken, "offsets", 1)},
        )
        summary, _ = _run(storage, clock, fleet=fleet)

        assert summary.failed == [broken]
        assert sorted(summary.completed) == sorted(t for t in TILES if t != broken)

    def test_a_failed_tile_names_the_keys_that_never_appeared(self, storage, clock):
        broken = TILES[2]
        plans = _plans()
        fleet = FakeWaveFleet(
            storage, plans, clock=clock, never={(broken, "offsets", 0), (broken, "offsets", 1)}
        )
        summary, _ = _run(storage, clock, fleet=fleet)

        reason = next(t.reason for t in summary.tiles if t.tile == broken)
        assert "offsets" in reason
        assert "unwritten" in reason

    def test_retries_are_bounded_per_tile(self, storage, clock):
        """``shard_barrier_rounds`` submissions, then the tile fails. Not the run."""
        broken = TILES[3]
        plans = _plans()
        fleet = FakeWaveFleet(
            storage, plans, clock=clock, never={(broken, "offsets", 0), (broken, "offsets", 1)}
        )
        summary, fleet = _run(storage, clock, fleet=fleet)

        carrying = [call for call in fleet.calls_for("offsets") if broken in call["tiles"]]
        assert len(carrying) == settings.shard_barrier_rounds
        assert summary.failed == [broken]
        # Each round costs its own deadline, so two rounds cost two of them --
        # the arithmetic that broke once already, measured on the fake clock.
        one = driver_deadline(summary, "offsets", 1)
        assert clock.elapsed >= one * (settings.shard_barrier_rounds - 1)

    def test_a_resubmission_carries_only_the_missing_units(self, storage, clock):
        """The point of the barrier: resend what is missing, not the stage."""
        flaky = TILES[0]
        plans = _plans()
        fleet = FakeWaveFleet(storage, plans, clock=clock, never={(flaky, "offsets", 1)}, heal=True)
        summary, fleet = _run(storage, clock, fleet=fleet)

        offsets = fleet.calls_for("offsets")
        second = [c for c in offsets if c["wave"] == 2]
        assert second, "the barrier never resubmitted"
        assert second[0]["units"] == [(flaky, 1)]
        assert sorted(summary.completed) == sorted(TILES)
        # Time is the point: round 2 opened only after round 1's horizon, and
        # the clock had to move for it to happen.
        first_wave = driver_deadline(summary, "offsets", 1)
        assert second[0]["at"] - offsets[0]["at"] >= first_wave
        assert clock.elapsed >= first_wave


class TestRoundBudget:
    """Counted across drivers, out of the bucket, per tile."""

    def test_an_expired_record_from_another_driver_costs_a_round(self, storage, clock):
        """A resume cannot mint itself a fresh allowance."""
        tile = TILES[0]
        root = shards.shard_root(RUN_ID, tile)
        storage.write_text(
            shards.stage_submission_key(root, "offsets", 1),
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "tile": tile,
                    "stage": "offsets",
                    "round": 1,
                    "indexes": [0, 1],
                    "cluster_name": "someone-else",
                    "cluster_id": 99,
                    "submitted_at": clock.now() - 10_000_000,
                }
            ),
        )
        plans = _plans()
        fleet = FakeWaveFleet(
            storage, plans, clock=clock, never={(tile, "offsets", 0), (tile, "offsets", 1)}
        )
        summary, fleet = _run(storage, clock, fleet=fleet)

        carrying = [c for c in fleet.calls_for("offsets") if tile in c["tiles"]]
        # Round 1 was spent by the other driver, so this one gets exactly one.
        assert len(carrying) == settings.shard_barrier_rounds - 1
        assert summary.failed == [tile]

    def test_a_live_record_is_adopted_rather_than_resubmitted(self, storage, clock):
        """Shards publish nothing while booting, so the record is the evidence."""
        tile = TILES[0]
        root = shards.shard_root(RUN_ID, tile)
        storage.write_text(
            shards.stage_submission_key(root, "offsets", 1),
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "tile": tile,
                    "stage": "offsets",
                    "round": 1,
                    "indexes": [0, 1],
                    "cluster_name": "someone-else",
                    "cluster_id": 99,
                    "submitted_at": clock.now(),
                }
            ),
        )
        plans = _plans()
        fleet = FakeWaveFleet(storage, plans, clock=clock)
        ticking = TickingStorage(storage, fleet)
        for writer in fleet.writers.values():
            writer.storage = ticking
        # The adopted tile's work arrives from "elsewhere" while we watch.
        publish_plan(ticking, fleet.plans[tile], run_id=RUN_ID)

        driver = FleetDriver(
            run_id=RUN_ID,
            tracks=[
                TileTrack(
                    tile=tile,
                    run_id=RUN_ID,
                    root=root,
                    storage=ticking,
                    units=2,
                    clock=clock,
                )
            ],
            storage=ticking,
            backend=fleet,
            clock=clock,
            units=2,
            wave_window_s=0.0,
        )
        for track in driver.tracks:
            demands = track.step()
        assert demands == []

        # ...and the same record stops being live once the clock passes its
        # horizon, which is the only thing that distinguishes watching from
        # waiting forever.
        clock.advance(_bootstrap_deadline_s() * 10)
        for track in driver.tracks:
            later = track.step()
        assert later and later[0].submission_round == 2


class TestControlPlaneFailures:
    """Terminal stops the run; transient does not."""

    def test_a_quota_refusal_aborts_the_whole_run(self, storage, clock):
        plans = _plans()
        fleet = FakeWaveFleet(
            storage,
            plans,
            clock=clock,
            raise_always=RuntimeError("you have reached your workspace quota of 400 credits"),
        )
        with pytest.raises(FleetAborted, match="quota"):
            _run(storage, clock, fleet=fleet)

        assert fleet.submit_attempts == 1

    def test_an_empty_server_error_is_transient(self, storage, clock):
        """The failure that killed the single-tile driver outright once."""
        plans = _plans()
        fleet = FakeWaveFleet(storage, plans, clock=clock, raise_once=RuntimeError(""))
        summary, fleet = _run(storage, clock, fleet=fleet)

        assert fleet.submit_attempts > 1
        assert sorted(summary.completed) == sorted(TILES)
        assert settings.shard_submit_backoff_s in clock.slept

    def test_an_auth_failure_aborts_rather_than_retrying(self, storage, clock):
        plans = _plans()
        fleet = FakeWaveFleet(
            storage, plans, clock=clock, raise_always=RuntimeError("invalid api token")
        )
        with pytest.raises(FleetAborted):
            _run(storage, clock, fleet=fleet)


class TestClusterLiveness:
    """A probe can end a barrier sooner. It can never declare success."""

    def test_a_dead_cluster_frees_headroom_and_reopens_the_barrier(self, storage, clock):
        plans = _plans(TILES[:2])
        fleet = FakeWaveFleet(
            storage,
            plans,
            clock=clock,
            never={
                (TILES[0], "offsets", 0),
                (TILES[0], "offsets", 1),
                (TILES[1], "offsets", 0),
                (TILES[1], "offsets", 1),
            },
            dead_handles={1},
        )
        ticking = TickingStorage(storage, fleet)
        for writer in fleet.writers.values():
            writer.storage = ticking

        summary = drive_fleet(
            _jobs(TILES[:2]),
            run_id=RUN_ID,
            storage=ticking,
            backend=fleet,
            clock=clock,
            wave_window_s=0.0,
            probe_waves=True,
        )

        # The first wave's submission is reported dead, so round 2 opens without
        # waiting out the deadline -- and the run still ends bounded.
        assert len(fleet.calls_for("offsets")) == settings.shard_barrier_rounds
        assert sorted(summary.failed) == sorted(TILES[:2])
        assert clock.elapsed < 10_000

    def test_probing_is_off_unless_asked_for(self, storage, clock):
        """Which is what the driver did before the backend abstraction existed.

        A probe is one control-plane call per live wave per poll. Turning that
        on by default while refactoring would be a semantic change smuggled
        into a rename -- and because a probe can only end a barrier *sooner*,
        leaving it off costs a deadline, never correctness.
        """
        plans = _plans(TILES[:1])
        fleet = FakeWaveFleet(storage, plans, clock=clock, dead_handles={1})
        probed: list = []
        fleet.probe = probed.append  # type: ignore[method-assign]

        _run(storage, clock, TILES[:1], fleet=fleet)

        assert probed == []


class TestManifest:
    """The roster: the one thing a listing cannot recover."""

    def test_roundtrip(self, storage):
        jobs = _jobs(TILES[:2])
        write_manifest(storage, RUN_ID, jobs, units=7)
        back, units = read_manifest(storage, RUN_ID)

        assert [job.tile.name for job in back] == TILES[:2]
        assert units == 7
        assert back[0].end_year == 2025

    def test_tiles_may_carry_different_windows(self, storage):
        jobs = [
            ProcessingJob(tile=parse_tile_name(TILES[0]), year=2021, end_year=2025),
            ProcessingJob(tile=parse_tile_name(TILES[1]), year=2024, max_scenes=300),
        ]
        write_manifest(storage, RUN_ID, jobs, units=4)
        back, _ = read_manifest(storage, RUN_ID)

        assert back[1].year == 2024
        assert back[1].end_year is None
        assert back[1].max_scenes == 300

    def test_a_vm_reads_its_own_job_from_the_roster(self, storage):
        from landsat_lst.fleet_driver import job_for_token

        write_manifest(storage, RUN_ID, _jobs(TILES[:2]), units=4)

        assert job_for_token(storage, RUN_ID, TILES[1]).tile.name == TILES[1]
        with pytest.raises(KeyError):
            job_for_token(storage, RUN_ID, "N00E000")

    def test_a_missing_roster_is_nothing_to_resume(self, storage):
        with pytest.raises(FileNotFoundError, match="nothing to resume"):
            read_manifest(storage, "never-ran")


class TestResume:
    """Reconstructed from storage, submitting only what is missing."""

    def test_a_finished_run_resumes_into_no_submissions_at_all(self, storage, clock):
        summary, _ = _run(storage, clock)
        assert sorted(summary.completed) == sorted(TILES)

        plans = _plans()
        second = FakeWaveFleet(storage, plans, clock=clock)
        ticking = TickingStorage(storage, second)
        for writer in second.writers.values():
            writer.storage = ticking
        resumed = resume_fleet(
            RUN_ID, storage=ticking, backend=second, clock=clock, wave_window_s=0.0
        )

        assert second.calls == []
        assert sorted(resumed.completed) == sorted(TILES)

    def test_a_resume_submits_only_the_tiles_that_are_unfinished(self, storage, clock):
        """A driver killed mid-run leaves a bucket, and the bucket is the state."""
        names = TILES[:3]
        done = names[0]
        plans = _plans(names)
        # Play out one tile completely, by hand, exactly as its fleet would.
        writer = FakeFleet(storage, plans[0], run_id=RUN_ID)
        for index in range(plans[0].scene_shards):
            writer._write("offsets", index)
        for index in range(len(plans[0].bands)):
            writer._write("composite", index)
        writer._write("export", 0)
        assert storage.cog_exists(plans[0].window, done)

        write_manifest(storage, RUN_ID, _jobs(names), units=2)
        fleet = FakeWaveFleet(storage, plans, clock=clock)
        ticking = TickingStorage(storage, fleet)
        for w in fleet.writers.values():
            w.storage = ticking

        summary = resume_fleet(
            RUN_ID, storage=ticking, backend=fleet, clock=clock, wave_window_s=0.0
        )

        assert sorted(summary.completed) == sorted(names)
        started = {tile for call in fleet.calls for tile in call["tiles"]}
        assert done not in started

    def test_a_resume_does_not_restart_a_partially_written_tile_from_scratch(self, storage, clock):
        names = TILES[:2]
        plans = _plans(names)
        # One tile has published its plan and one of its two scene partials.
        writer = FakeFleet(storage, plans[0], run_id=RUN_ID)
        writer._write("offsets", 0)

        write_manifest(storage, RUN_ID, _jobs(names), units=2)
        fleet = FakeWaveFleet(storage, plans, clock=clock)
        ticking = TickingStorage(storage, fleet)
        for w in fleet.writers.values():
            w.storage = ticking

        resume_fleet(RUN_ID, storage=ticking, backend=fleet, clock=clock, wave_window_s=0.0)

        first = fleet.calls_for("offsets")[0]
        carried = [index for tile, index in first["units"] if tile == names[0]]
        assert carried == [1]


class TestWaveRecords:
    """What a resumed driver has to know before it submits anything."""

    def test_a_wave_is_published_so_the_next_driver_can_see_it(self, storage, clock):
        _run(storage, clock)

        keys = storage.list_prefix(shards.fleet_submission_prefix(RUN_ID, "offsets"))
        assert keys
        body = json.loads(storage.read_text(sorted(keys)[0]))
        assert body["stage"] == "offsets"
        assert sorted(body["tiles"]) == sorted(TILES)
        assert body["max_workers"] >= 1

    def test_a_resume_counts_another_driver_s_vms_against_its_own_cap(self, storage, clock):
        """The cap is a property of the run, not of whoever happens to drive it."""
        storage.write_text(
            shards.fleet_submission_key(RUN_ID, "offsets", 1),
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "stage": "offsets",
                    "wave": 1,
                    "tiles": list(TILES),
                    "max_workers": 6,
                    "submitted_at": clock.now(),
                    "deadline_s": 10_000.0,
                    "handle_id": 5,
                    "handle_name": "lst-x-fleet-offse-w1",
                }
            ),
        )
        plans = _plans()
        fleet = FakeWaveFleet(storage, plans, clock=clock)
        driver = FleetDriver(
            run_id=RUN_ID,
            tracks=[],
            storage=storage,
            backend=fleet,
            clock=clock,
            units=2,
            max_vms=6,
        )
        driver.adopt_live_waves()

        assert driver.in_flight == 6
        assert driver.headroom == 0

    def test_a_resume_does_not_rebuild_a_live_wave_name(self, storage, clock):
        """The ADR-016 collision, one level up: numbering must not restart at 1.

        Backend-neutral by construction: the name comes from
        ``backend.wave_name``, and the contract requires only that it be unique
        per wave. What must not happen is a resumed driver reusing a number.
        """
        storage.write_text(
            shards.fleet_submission_key(RUN_ID, "offsets", 1),
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "stage": "offsets",
                    "wave": 1,
                    "tiles": list(TILES),
                    "max_workers": 1,
                    "submitted_at": clock.now(),
                    "deadline_s": 10_000.0,
                    "handle_id": 5,
                    "handle_name": "fake-fleet-test-offsets-w1",
                }
            ),
        )
        write_manifest(storage, RUN_ID, _jobs(TILES), units=2)
        plans = _plans()
        fleet = FakeWaveFleet(storage, plans, clock=clock)
        ticking = TickingStorage(storage, fleet)
        for writer in fleet.writers.values():
            writer.storage = ticking

        resume_fleet(RUN_ID, storage=ticking, backend=fleet, clock=clock, wave_window_s=0.0)

        offsets = fleet.calls_for("offsets")
        assert offsets
        assert all(call["wave"] > 1 for call in offsets)

    def test_an_expired_adopted_wave_keeps_its_capacity_until_it_is_confirmed(self, storage, clock):
        """Long expired is not the same fact as gone. See ADR-018.

        The first draft of adoption kept only waves whose deadline had not
        passed, which is the f6cf6fc defect one process later: a resumed driver
        would ignore a merely late wave, submit into headroom that existed only
        on paper, and run at twice its cap. Here the record is ancient and its
        units have not landed, so the width stays spoken for until the backend
        says the submission is gone.
        """
        tokens = [f"{TILES[0]}:{i}" for i in range(6)]
        self._write_wave_record(storage, clock, workers=6, tokens=tokens)
        driver = self._adopting_driver(storage, clock)
        driver.adopt_live_waves()

        assert driver.in_flight == 6
        assert driver.headroom == 2

        driver._dead_handles.add(5)
        driver._retire()

        assert driver.in_flight == 0
        assert driver.headroom == 8

    def test_an_adopted_wave_is_probed_once_whatever_its_record_claims(self, storage, clock):
        """The horizon in the record is the previous driver's arithmetic.

        A wave this process never submitted is one it has confirmed nothing
        about, and a long deadline would otherwise postpone the only question
        that can release it -- for as long as that deadline says. So adoption
        asks once, whatever ``probe_waves`` and the record's horizon say.
        """
        tokens = [f"{TILES[0]}:{i}" for i in range(6)]
        self._write_wave_record(storage, clock, workers=6, tokens=tokens, deadline_s=1_000_000.0)
        driver = self._adopting_driver(storage, clock, dead={5})
        driver.adopt_live_waves()

        assert driver.in_flight == 0
        assert driver.headroom == 8

    def test_an_adopted_wave_whose_units_landed_gives_its_capacity_back(self, storage, clock):
        """The other half of the same rule: artifacts settle a wave, at any age."""
        plan = _plans(TILES[:1])[0]
        fleet = FakeWaveFleet(storage, [plan], clock=clock)
        fleet.writers[TILES[0]]._write("offsets", 0)
        self._write_wave_record(storage, clock, workers=6, tokens=[f"{TILES[0]}:0"])
        driver = FleetDriver(
            run_id=RUN_ID,
            tracks=[
                TileTrack(
                    tile=TILES[0],
                    run_id=RUN_ID,
                    root=shards.shard_root(RUN_ID, TILES[0]),
                    storage=storage,
                    units=2,
                    clock=clock,
                    plan=plan,
                )
            ],
            storage=storage,
            backend=fleet,
            clock=clock,
            units=2,
            max_vms=8,
        )
        driver.index.refresh()
        driver.adopt_live_waves()

        assert driver.in_flight == 0
        assert driver.headroom == 8

    def test_a_record_without_a_unit_list_holds_its_requested_width(self, storage, clock):
        """A wave written before ``unit_tokens`` existed, whose tiles left nothing.

        Neither a unit list nor a per-tile record nor a track with a plan, so
        nothing in the bucket refers to it and no artifact can ever retire it.
        Holding its width is the conservative answer and the only safe one:
        releasing it is exactly the over-admission the resume path exists to
        avoid. What it must not be is silent, so the driver says so and
        :meth:`FleetDriver._stalled` ends the run rather than polling out.
        """
        self._write_wave_record(storage, clock, workers=6, tokens=None)
        driver = self._adopting_driver(storage, clock)
        driver.adopt_live_waves()

        assert driver.in_flight == 6
        assert driver._unobservable(driver._live[0])

    def test_a_legacy_record_recovers_its_units_from_the_per_tile_records(self, storage, clock):
        """The unit list a pre-``unit_tokens`` wave lacks is already in the bucket.

        Every wave writes one submission record per tile before it submits, and
        that record carries the wave number and the tile's indexes. Rebuilding
        from those is what lets an adopted wave be *observed* to have settled,
        rather than holding its requested width for the life of the resumed
        driver because only a probe could ever release it.
        """
        plan = _plans(TILES[:1])[0]
        fleet = FakeWaveFleet(storage, [plan], clock=clock)
        fleet.writers[TILES[0]]._write("offsets", 0)
        storage.write_text(
            shards.stage_submission_key(shards.shard_root(RUN_ID, TILES[0]), "offsets", 1),
            json.dumps({"stage": "offsets", "round": 1, "wave": 1, "indexes": [0]}),
        )
        self._write_wave_record(storage, clock, workers=6, tokens=None)
        driver = FleetDriver(
            run_id=RUN_ID,
            tracks=[
                TileTrack(
                    tile=TILES[0],
                    run_id=RUN_ID,
                    root=shards.shard_root(RUN_ID, TILES[0]),
                    storage=storage,
                    units=2,
                    clock=clock,
                    plan=plan,
                )
            ],
            storage=storage,
            backend=fleet,
            clock=clock,
            units=2,
            max_vms=8,
        )
        driver.index.refresh()
        driver.adopt_live_waves()

        assert driver.in_flight == 0
        assert driver.headroom == 8

    def _write_wave_record(self, storage, clock, *, workers, tokens, deadline_s=100.0):
        body = {
            "run_id": RUN_ID,
            "stage": "offsets",
            "wave": 1,
            "tiles": [TILES[0]],
            "max_workers": workers,
            "submitted_at": clock.now() - 10_000_000,
            "deadline_s": deadline_s,
            "handle_id": 5,
            "handle_name": "old",
        }
        if tokens is not None:
            body["unit_tokens"] = tokens
        storage.write_text(shards.fleet_submission_key(RUN_ID, "offsets", 1), json.dumps(body))

    def _adopting_driver(self, storage, clock, *, dead=()):
        """A driver with no tracks, for adoption arithmetic.

        The backend's probe answers ``None`` -- unknown -- unless a handle is
        named dead, because adoption forces one probe of every inherited wave
        and a backend that confirms death on sight would settle every one of
        them before the arithmetic under test could be read.
        """
        return FleetDriver(
            run_id=RUN_ID,
            tracks=[],
            storage=storage,
            backend=FakeWaveFleet(storage, [], clock=clock, dead_handles=set(dead)),
            clock=clock,
            units=2,
            max_vms=8,
        )


class TestWaveBatching:
    """The three flush conditions, and none of them counts tiles."""

    def test_quiescence_flushes_without_waiting_out_the_window(self, storage, clock):
        """Every tile has demanded, so there is provably nobody left to wait for."""
        summary, fleet = _run(storage, clock, wave_window_s=10_000.0)

        assert fleet.calls_for("offsets")
        assert sorted(summary.completed) == sorted(TILES)
        # A 10,000 s window was configured and the first wave did not wait it
        # out: every tile had already demanded, so there was provably nobody
        # left to wait for. The submission timestamp is the evidence.
        started = fleet.calls_for("offsets")[0]["at"]
        assert started - FakeClock().now() < 10_000.0

    def test_the_window_bounds_how_long_a_wave_waits_for_stragglers(self, storage, clock):
        """A tile that is still in offsets keeps the composite wave open --
        until the window expires, which is what bounds the submission count.
        """
        slow = TILES[0]
        plans = _plans()
        fleet = FakeWaveFleet(
            storage,
            plans,
            clock=clock,
            lands_after={(slow, "offsets", 0): 20, (slow, "offsets", 1): 20},
        )
        summary, fleet = _run(storage, clock, fleet=fleet, wave_window_s=60.0)

        composite = fleet.calls_for("composite")
        # Bounded: the stragglers batch instead of one wave per tile.
        assert len(composite) <= len(TILES)
        assert sorted(summary.completed) == sorted(TILES)


class TestPreservedSeams:
    """What must not have changed underneath the consolidation."""

    def test_the_offset_record_lands_at_the_canonical_key(self, storage, clock):
        """The ADR-012 boundary: the merge runs in the driver, every band reads it."""
        summary, _ = _run(storage, clock)

        offsets = [key for key in storage.list_prefix("_offsets/") if key.endswith(".json")]
        assert len({key.split("/")[1] for key in offsets}) == len(TILES)
        assert sorted(summary.completed) == sorted(TILES)

    def test_shard_objects_stay_under_the_shard_prefix(self, storage, clock):
        _run(storage, clock)

        assert not storage.list_prefix("_runs/")
        assert storage.list_prefix(f"{shards.SHARD_PREFIX}/{RUN_ID}/")

    def test_every_tile_keeps_its_own_root_under_one_run_id(self, storage, clock):
        _run(storage, clock)

        for tile in TILES:
            root = shards.shard_root(RUN_ID, tile)
            assert storage.read_text(shards.plan_key(root)) is not None

    def test_completion_is_the_canonical_cogs(self, storage, clock):
        summary, _ = _run(storage, clock)

        for tile in TILES:
            for product in PRODUCTS:
                assert storage.read_text(storage.cog_key("2021-2025", tile, product)) is not None
        assert sorted(summary.completed) == sorted(TILES)

    def test_the_export_is_claimed_by_a_composite_worker_not_submitted(self, storage, clock):
        """The claim saves a whole VM boot, so the driver's first move is to wait."""
        summary, fleet = _run(storage, clock)

        assert fleet.calls_for("export") == []
        assert sorted(summary.completed) == sorted(TILES)

    def test_an_unfulfilled_claim_falls_back_to_an_export_wave(self, storage, clock):
        """A claim written and never executed is a VM preempted between the two.

        The fallback batches like every other stage, so a fleet of preempted
        claims costs one wave rather than one per tile.
        """
        plans = _plans()
        fleet = FakeWaveFleet(storage, plans, clock=clock, claims_export=False)
        summary, fleet = _run(storage, clock, fleet=fleet)

        exports = fleet.calls_for("export")
        assert len(exports) == 1
        assert exports[0]["tiles"] == sorted(TILES)
        assert sorted(summary.completed) == sorted(TILES)

    def test_per_tile_submission_records_survive_a_shared_wave(self, storage, clock):
        """One array, many records: adoption and the round budget are per tile."""
        _run(storage, clock)

        for tile in TILES:
            root = shards.shard_root(RUN_ID, tile)
            records = storage.list_prefix(shards.stage_submission_prefix(root, "offsets"))
            assert records
            body = json.loads(storage.read_text(sorted(records)[0]))
            assert body["tile"] == tile
            assert body["wave"] == 1


class TestQueueDepth:
    """The gates the review named. Queue depth is the design, so it is tested.

    Capture -- the share of provisioning a wave avoids -- is ``1 - 1/R`` for
    ``R = ceil(units / workers)``. Every property below is about what has to
    stay true as ``R`` grows, because the first version of this driver was
    correct only at ``R = 1`` and every wave worth submitting has ``R > 1``.
    """

    def test_eight_units_at_cap_four_settle_completely(self, storage, clock):
        """The review's reproduction, driven all the way to settlement."""
        names = TILES[:4]
        summary, fleet = _run(storage, clock, names, max_vms=4)

        assert sorted(summary.completed) == sorted(names)
        assert summary.failed == []
        assert fleet.peak_in_flight <= 4
        assert all(call["max_workers"] <= 4 for call in fleet.calls)

    @pytest.mark.parametrize("cap", [1, 2, 4, 8, 16])
    def test_settles_at_every_queue_depth(self, tmp_path, cap):
        """One cap per depth, each run to completion with the peak asserted."""
        store = LocalStorage(tmp_path / f"cap{cap}")
        summary, fleet = _run(store, FakeClock(), TILES, run_id=f"depth-{cap}", max_vms=cap)

        assert sorted(summary.completed) == sorted(TILES)
        assert fleet.peak_in_flight <= cap

    def test_the_deadline_grows_with_queue_depth(self):
        """The defect, at the level of the arithmetic that caused it.

        A wave four deep gets roughly four rounds of work, not one. Before the
        fix every wave got one round's budget however deep it was, so any wave
        that saved a boot expired before it could finish.
        """
        from landsat_lst import budgets

        shallow = budgets.wave_deadline_s(boot_s=300, unit_work_s=600, units=4, workers=4)
        deep = budgets.wave_deadline_s(boot_s=300, unit_work_s=600, units=16, workers=4)
        deeper = budgets.wave_deadline_s(boot_s=300, unit_work_s=600, units=64, workers=4)

        assert budgets.queue_depth(units=16, workers=4) == 4
        assert deep > shallow
        assert deeper > deep
        # Work scales with depth; provisioning does not.
        assert (deeper - deep) > (deep - shallow) * 3.5

    def test_a_seven_hundred_tile_queue_is_one_wave_within_the_cap(self, storage, clock):
        """Production scale, as a queue rather than as 700 real plans.

        What matters at 700 tiles is the batching and the arithmetic: one wave,
        a width inside the cap, and a deadline that covers the depth. Building
        700 real tile plans would test the fixture, not the driver.
        """
        from landsat_lst import budgets

        # 35 latitudes x 20 longitudes = 700 genuinely distinct tiles. Demands
        # are keyed by tile, so a colliding name would silently shrink the queue
        # and the test would measure a smaller build than it claims to.
        tiles = [f"N{20 + i // 20:02d}W{100 + i % 20:03d}" for i in range(700)]
        assert len(set(tiles)) == 700
        driver, _fleet = _bare_fleet(storage, clock, max_vms=64)
        for i, tile in enumerate(tiles):
            driver._by_tile[tile] = TileTrack(
                tile=tile,
                run_id=RUN_ID,
                root=shards.shard_root(RUN_ID, tile),
                storage=storage,
                units=15,
                clock=clock,
            )
            driver._by_tile[tile].outstanding["offsets"] = set(range(15))
            driver._buffer_demand(
                _demand(tile, "offsets", tuple(range(15)), work=600.0, boot=300.0)
            )
            del i
        driver.tracks = list(driver._by_tile.values())
        driver._flush("offsets")

        wave = driver.summary.waves[0]
        assert len(driver.summary.waves) == 1
        assert len(wave.units) == 700 * 15
        assert wave.max_workers == 64
        depth = budgets.queue_depth(units=len(wave.units), workers=64)
        assert depth > 100
        # The deadline has to cover the depth, or the whole build fails on the
        # first wave -- which is exactly what the review reproduced.
        assert wave.deadline_s > depth * 600.0

    def test_peak_in_flight_never_exceeds_the_cap_across_resubmission(self, storage, clock):
        """Expiry interleaved with resubmission, measured independently.

        The failure this guards is precise: a wave that expires while its
        workers are still running used to hand its width back, a resubmission
        went out into headroom that existed only on paper, and the run held two
        waves' worth of VMs against a one-wave cap.
        """
        slow = TILES[0]
        plans = _plans()
        fleet = FakeWaveFleet(
            storage,
            plans,
            clock=clock,
            lands_after={(slow, "offsets", i): 8 for i in range(15)},
        )
        summary, fleet = _run(storage, clock, fleet=fleet, max_vms=4)

        assert fleet.peak_in_flight <= 4
        assert sorted(summary.completed) == sorted(TILES)
        assert clock.elapsed > 0

    def test_live_work_at_its_deadline_keeps_its_capacity(self, storage, clock):
        """A deadline is not evidence that the workers stopped."""
        driver, _track = self._live_wave(storage, clock)

        clock.advance(driver.summary.waves[0].deadline_s + 1.0)
        driver._retire()

        assert driver.summary.waves[0].expired(clock.now())
        assert driver.in_flight == 4
        assert driver.headroom == 0

    def test_capacity_returns_only_on_confirmation(self, storage, clock):
        """Either the units land, or the backend says the submission is gone."""
        driver, _track = self._live_wave(storage, clock)
        clock.advance(driver.summary.waves[0].deadline_s + 1.0)

        driver._dead_handles.add(driver.summary.waves[0].handle_id)
        driver._retire()

        assert driver.in_flight == 0

    def _live_wave(self, storage, clock):
        driver, _fleet = _bare_fleet(storage, clock, max_vms=4)
        track = TileTrack(
            tile=TILES[0],
            run_id=RUN_ID,
            root=shards.shard_root(RUN_ID, TILES[0]),
            storage=storage,
            units=4,
            clock=clock,
        )
        driver.tracks = [track]
        driver._by_tile = {track.tile: track}
        track.outstanding["offsets"] = {0, 1, 2, 3}
        driver._buffer_demand(_demand(TILES[0], "offsets", (0, 1, 2, 3), work=600.0))
        driver._flush("offsets")
        return driver, track


class TestDuplicateDriversAndRestart:
    """Two drivers, and one driver twice. Neither may double-count or duplicate."""

    def test_a_second_driver_adopts_rather_than_duplicating_live_work(self, storage, clock):
        """Gate 4. The first driver's live submission is visible to the second."""
        names = TILES[:2]
        plans = _plans(names)
        first = FakeWaveFleet(
            storage,
            plans,
            clock=clock,
            lands_after={(name, "offsets", i): 200 for name in names for i in range(15)},
        )
        ticking = TickingStorage(storage, first)
        for writer in first.writers.values():
            writer.storage = ticking
        write_manifest(storage, RUN_ID, _jobs(names), units=15)

        driver_a = _driver_for(names, storage=ticking, backend=first, clock=clock)
        driver_a.index.refresh()
        for track in driver_a.tracks:
            for demand in track.step():
                driver_a._buffer_demand(demand)
        driver_a._flush("offsets")
        assert len(first.calls) == 1

        # A second driver over the same run, while that wave is still live.
        second = FakeWaveFleet(storage, plans, clock=clock)
        driver_b = _driver_for(names, storage=ticking, backend=second, clock=clock)
        driver_b.adopt_live_waves()
        driver_b.index.refresh()
        for track in driver_b.tracks:
            for demand in track.step():
                driver_b._buffer_demand(demand)

        assert second.calls == []
        # And the first driver's workers are counted, not forgotten.
        assert driver_b.in_flight > 0

    @pytest.mark.parametrize("kill_after", [1, 2, 3])
    def test_a_restart_resumes_from_whatever_the_bucket_holds(self, tmp_path, kill_after):
        """Restart at several barrier positions, from a *real* prior run.

        The prior state is written by an actual driver rather than by a
        hand-built JSON blob, so the resume is tested against the shapes the
        driver really produces -- including the heterogeneous case where tiles
        are at different stages.
        """
        store = LocalStorage(tmp_path / f"restart{kill_after}")
        clock = FakeClock()
        names = TILES[:3]
        plans = _plans(names)
        first = FakeWaveFleet(store, plans, clock=clock, run_id=RUN_ID)
        ticking = TickingStorage(store, first)
        for writer in first.writers.values():
            writer.storage = ticking
        write_manifest(store, RUN_ID, _jobs(names), units=15)

        driver = _driver_for(names, storage=ticking, backend=first, clock=clock)
        for _ in range(kill_after):
            driver.index.refresh()
            for track in driver.tracks:
                for demand in track.step():
                    driver._buffer_demand(demand)
            for stage in ("offsets", "composite", "export"):
                if driver._ready_to_flush(stage):
                    driver._flush(stage)
            clock.sleep(settings.fleet_poll_s)

        # The driver is gone. Everything a resume needs is in the bucket.
        second = FakeWaveFleet(store, plans, clock=clock, run_id=RUN_ID)
        ticking2 = TickingStorage(store, second)
        for writer in second.writers.values():
            writer.storage = ticking2
        summary = resume_fleet(
            RUN_ID, storage=ticking2, backend=second, clock=clock, wave_window_s=0.0
        )

        assert sorted(summary.completed) == sorted(names)

    def test_retry_exhaustion_does_not_leave_duplicate_live_work(self, storage, clock):
        """Rounds are counted across restarts, and the tile fails once."""
        broken = TILES[0]
        names = TILES[:2]
        plans = _plans(names)
        fleet = FakeWaveFleet(
            storage,
            plans,
            clock=clock,
            never={(broken, "offsets", i) for i in range(15)},
        )
        summary, fleet = _run(storage, clock, names, fleet=fleet)

        carrying = [call for call in fleet.calls if broken in call["tiles"]]
        assert len(carrying) == settings.shard_barrier_rounds
        assert summary.failed == [broken]
        # No two live submissions for the same units at the same moment.
        assert fleet.peak_in_flight <= settings.fleet_max_vms


class TestBootBound:
    """A consolidation regression must fail a test, not just cost money."""

    def test_launches_do_not_scale_with_tile_count(self, tmp_path):
        """Boots are the thing being bought. Count them.

        A driver that regressed to one submission per tile would still finish
        every tile and still pass a completion test; the only thing that would
        move is the number of VM launches, so that is what is asserted.
        """
        launches = {}
        for n in (2, 8):
            store = LocalStorage(tmp_path / f"boots{n}")
            names = [f"N{30 + i}W075" for i in range(n)]
            summary, fleet = _run(store, FakeClock(), names, run_id=f"boots-{n}", max_vms=8)
            assert len(summary.completed) == n
            launches[n] = sum(call["max_workers"] for call in fleet.calls)

        assert launches[8] <= launches[2] * 2
        assert launches[8] < 8 * 3


class TestWorkUnitEquivalence:
    """The fleet path must hand ``run_shard`` what the single-tile path does."""

    def test_the_two_paths_agree_argument_for_argument(self, monkeypatch, tmp_path):
        """Scientific equivalence, at the only place the paths differ.

        Both CLI entry points end in ``run_shard(stage, run_id, tile, index,
        job=..., units=...)``. The consolidated path resolves the tile from a
        token and the window from the roster instead of from flags; if that
        resolution drifted, a fleet tile would silently composite a different
        scene set from a per-tile one.
        """
        from click.testing import CliRunner

        from landsat_lst import cli, fleet_driver
        from landsat_lst.storage import LocalStorage as _Local

        store = _Local(tmp_path)
        monkeypatch.setattr("landsat_lst.storage.get_storage", lambda: store)
        monkeypatch.setattr("landsat_lst.fleet_driver.get_storage", lambda: store)
        fleet_driver.write_manifest(store, "r1", _jobs([TILES[0]]), units=8)

        seen: list[dict] = []

        def record(stage, run_id, tile, index, *, job=None, units=None, storage=None):
            seen.append(
                {
                    "stage": stage,
                    "run_id": run_id,
                    "tile": tile,
                    "index": index,
                    "year": None if job is None else job.year,
                    "end_year": None if job is None else job.end_year,
                    "max_scenes": None if job is None else job.max_scenes,
                    "units": units,
                }
            )
            return "ok"

        monkeypatch.setattr("landsat_lst.shard_tasks.run_shard", record)

        runner = CliRunner()
        single = runner.invoke(
            cli.main,
            [
                "shard",
                "offsets",
                "--run-id",
                "r1",
                "--tile",
                TILES[0],
                "--year",
                "2021",
                "--end-year",
                "2025",
                "--units",
                "8",
                "--index",
                "3",
            ],
        )
        fleet = runner.invoke(
            cli.main,
            [
                "shard",
                "unit",
                "--run-id",
                "r1",
                "--stage",
                "offsets",
                "--token",
                f"{TILES[0]}:3",
                "--units",
                "8",
            ],
        )

        assert single.exit_code == 0, single.output
        assert fleet.exit_code == 0, fleet.output
        assert len(seen) == 2
        assert seen[0] == seen[1]

    @pytest.mark.parametrize("stage", ["composite", "export"])
    def test_the_later_stages_agree_too(self, monkeypatch, tmp_path, stage):
        """Every stage a wave can carry, not only the one that reads a job.

        ``offsets`` is the interesting case because its shard 0 resolves and so
        needs the manifest; the others take their window from the plan. That is
        precisely why they are worth pinning separately -- a token parsed one
        way for one stage and another way for the next would put a band of one
        tile under another tile's key.
        """
        from click.testing import CliRunner

        from landsat_lst import cli

        store = LocalStorage(tmp_path)
        monkeypatch.setattr("landsat_lst.storage.get_storage", lambda: store)
        monkeypatch.setattr("landsat_lst.fleet_driver.get_storage", lambda: store)
        write_manifest(store, "r1", _jobs([TILES[0]]), units=8)

        seen: list[dict] = []

        def record(stage, run_id, tile, index, *, job=None, units=None, storage=None):
            seen.append({"stage": stage, "run_id": run_id, "tile": tile, "index": index})
            return "ok"

        monkeypatch.setattr("landsat_lst.shard_tasks.run_shard", record)
        runner = CliRunner()
        single = runner.invoke(
            cli.main,
            ["shard", stage, "--run-id", "r1", "--tile", TILES[0], "--index", "3"],
        )
        fleet = runner.invoke(
            cli.main,
            [
                "shard",
                "unit",
                "--run-id",
                "r1",
                "--stage",
                stage,
                "--token",
                f"{TILES[0]}:3",
            ],
        )

        assert single.exit_code == 0, single.output
        assert fleet.exit_code == 0, fleet.output
        assert len(seen) == 2
        assert seen[0] == seen[1]


class TestListingCost:
    """The poll loop's request rate must not grow with the tile count."""

    def test_listings_per_cycle_are_flat_in_tile_count(self, tmp_path):
        """One shared listing per prefix, not one per tile.

        Measured before this: about two listings per tile per cycle, which at
        700 tiles is ~1,400 serial round trips against a 30 s poll -- the loop
        stops keeping up with itself somewhere around 300 tiles, silently.
        """
        per_cycle = {}
        for n in (2, 10):
            store = LocalStorage(tmp_path / f"lists{n}")
            clock = FakeClock()
            names = [f"N{30 + i}W075" for i in range(n)]
            plans = _plans(names)
            fleet = FakeWaveFleet(store, plans, clock=clock, run_id=f"list-{n}")
            ticking = TickingStorage(store, fleet)
            for writer in fleet.writers.values():
                writer.storage = ticking
            write_manifest(store, f"list-{n}", _jobs(names), units=15)
            driver = _driver_for(
                names, storage=ticking, backend=fleet, clock=clock, run_id=f"list-{n}"
            )
            driver.index.refresh()
            for track in driver.tracks:
                track.step()
            per_cycle[n] = driver.index.listings

        assert per_cycle[10] == per_cycle[2]

    def test_requests_are_pages_and_listings_are_calls(self, tmp_path):
        """The counter has to measure what S3 charges for.

        A call over a prefix holding more keys than one page is several
        requests, so counting calls reads flat while the bill climbs with the
        keys a run has published. Reporting the flat number as the cost was an
        accounting defect rather than a wrong claim: what sharing the listing
        buys is a change in the exponent on the tile count, and only a request
        count can say so honestly.
        """
        store = LocalStorage(tmp_path / "pages")
        index = PollIndex(store, "pages")
        for i in range(LIST_PAGE_KEYS + 1):
            store.write_text(f"{shards.SHARD_PREFIX}/pages/k{i:06d}", "x")

        index.refresh()

        assert index.listings == 1
        assert index.requests == 2


class TestSpeed:
    """A suite nobody can afford to run is a suite nobody runs."""

    def test_the_whole_interleaving_runs_without_real_waiting(self, storage, clock):
        import time as wall

        started = wall.perf_counter()
        summary, _ = _run(storage, clock, [f"N{30 + i}W075" for i in range(8)], run_id="fast")

        assert len(summary.completed) == 8
        assert wall.perf_counter() - started < 10.0


class TestCapacityIsReleasedOnlyOnEvidence:
    """Three ways a driver can talk itself into headroom it does not have.

    All three are the same mistake as the deadline defect ADR-018 records: a
    fact about the *driver's* state -- a round budget spent, a tile given up on,
    a stage the track has stopped looking at -- read as a fact about whether a
    VM is still running. Only two things retire a wave: the artifacts land, or
    the backend says the submission is gone.
    """

    def test_a_failed_tile_keeps_holding_the_workers_it_never_heard_from(self, storage, clock):
        """A tile out of barrier rounds is a tile that gave up, not a VM that stopped.

        The rounds ran out because a record aged out, and the workers that
        record describes may be running still. An earlier draft cleared the
        tile's outstanding set and retired every wave whose tiles had settled,
        which handed a live wave's width back at exactly the moment the run was
        most likely to resubmit into it.
        """
        driver, track, _fleet = self._one_tile(storage, clock, max_vms=8)
        driver._buffer_demand(_demand(TILES[0], "offsets", (0, 1)))
        driver._flush("offsets")
        assert driver.in_flight == 2

        track._fail("out of rounds")
        driver.index.refresh()
        driver._retire()

        assert track.terminal
        assert driver.in_flight == 2, "a failed tile released capacity that was still in use"
        assert driver.headroom == 6

        # And the width does come back, on the one thing that settles it.
        driver._dead_handles.add(driver.summary.waves[0].handle_id)
        driver._retire()
        assert driver.in_flight == 0

    def test_a_submission_that_may_have_started_holds_its_capacity(self, storage, clock):
        """An unanswered submission is not an unstarted one -- once per attempt.

        The control plane can accept the request, boot the workers, and lose
        the answer on the way back. A driver that counts only what it was told
        about then reports its whole cap free while the workers run, and the
        next wave boots on top of them. So the width is held, and only a census
        that stops reporting the identity releases it.

        Held for **every attempt**, not once for the wave. There is no
        idempotency key on this path and the submission is a non-atomic
        two-step, so each of the three attempts may have created a cluster and
        booted its workers: the charge is ``attempts * width``. It over-counts
        whenever the substrate's name guard did refuse the retries, and that is
        the direction to be wrong in -- under-counting is the measured
        90-up / 30-counted window, and a census corrects the over-count on the
        first poll that can be taken.
        """
        fleet = FakeWaveFleet(
            storage, _plans(TILES[:1]), clock=clock, raise_always=RuntimeError("control plane")
        )
        driver = self._driver_with(storage, clock, fleet, max_vms=4)
        driver._buffer_demand(_demand(TILES[0], "offsets", (0, 1)))
        driver._flush("offsets")

        assert driver.summary.submissions == 1
        assert fleet.submit_attempts == settings.shard_submit_retries
        assert driver.in_flight == 2 * settings.shard_submit_retries
        assert driver.headroom == 0
        wave = driver.summary.waves[0]
        assert wave.acknowledged is False
        assert wave.handle_id is None
        # The per-tile record is still written, so the tile watches and spends
        # the round rather than resubmitting into the same failure at once.
        assert _submission_records(storage, shards.shard_root(RUN_ID, TILES[0]), "offsets")

    def test_an_unacknowledged_wave_is_recorded_before_the_call_that_starts_it(
        self, storage, clock
    ):
        """Recovery is idempotent because the record precedes the submission.

        A driver that dies inside the control-plane call leaves workers nothing
        mentions. The wave record is written first, at a key that is a pure
        function of ``(run_id, stage, wave)``, so the next driver -- or a
        duplicate of this one -- adopts that wave rather than minting a second
        one beside it.
        """
        fleet = FakeWaveFleet(
            storage, _plans(TILES[:1]), clock=clock, raise_always=RuntimeError("control plane")
        )
        driver = self._driver_with(storage, clock, fleet, max_vms=4)
        driver._buffer_demand(_demand(TILES[0], "offsets", (0, 1)))
        driver._flush("offsets")

        body = json.loads(storage.read_text(shards.fleet_submission_key(RUN_ID, "offsets", 1)))
        assert body["acknowledged"] is False
        assert body["max_workers"] == 2
        # The attempt count is part of the record, because it is part of the
        # charge. A resumed driver that read only the width would give back two
        # thirds of what this one is holding.
        assert body["attempts"] == settings.shard_submit_retries

        charged = 2 * settings.shard_submit_retries
        resumed = self._driver_with(storage, clock, fleet, max_vms=4)
        resumed.adopt_live_waves()
        assert resumed.in_flight == charged
        assert len(resumed._live) == 1
        # And again: adoption is a read, so a third driver counts the same.
        again = self._driver_with(storage, clock, fleet, max_vms=4)
        again.adopt_live_waves()
        assert again.in_flight == charged

    def test_an_overlapped_composite_wave_gives_width_back_when_its_bands_land(
        self, storage, clock
    ):
        """A stage the track has stopped looking at still has to be measured.

        The composite fleet is demanded from inside the offsets barrier, and
        the track will not revisit the composite stage until it has merged its
        offsets. Read from the track's memory that wave would hold its width
        for the whole run, because nothing would ever look at it again.
        """
        driver, _track, fleet = self._one_tile(storage, clock, max_vms=8)
        driver._buffer_demand(_demand(TILES[0], "composite", (0, 1)))
        driver._flush("composite")
        driver.index.refresh()
        driver._retire()
        assert driver.in_flight == 2

        fleet.writers[TILES[0]]._write("composite", 0)
        driver.index.refresh()
        driver._retire()
        assert driver.in_flight == 2, "one band of two is not the end of the array"

        fleet.writers[TILES[0]]._write("composite", 1)
        driver.index.refresh()
        driver._retire()
        assert driver.in_flight == 0
        assert driver.headroom == 8

    def _one_tile(self, storage, clock, *, max_vms):
        """One planned track whose units never land unless a test writes them."""
        plans = _plans(TILES[:1])
        never = {
            (TILES[0], stage, index) for stage in ("offsets", "composite") for index in range(4)
        }
        fleet = FakeWaveFleet(storage, plans, clock=clock, never=never)
        fleet.writers[TILES[0]].storage = storage
        driver = self._driver_with(storage, clock, fleet, max_vms=max_vms, plan=plans[0])
        return driver, driver.tracks[0], fleet

    def _driver_with(self, storage, clock, fleet, *, max_vms, plan=None):
        track = TileTrack(
            tile=TILES[0],
            run_id=RUN_ID,
            root=shards.shard_root(RUN_ID, TILES[0]),
            storage=storage,
            units=2,
            clock=clock,
            plan=plan if plan is not None else _plans(TILES[:1])[0],
        )
        return FleetDriver(
            run_id=RUN_ID,
            tracks=[track],
            storage=storage,
            backend=fleet,
            clock=clock,
            units=2,
            max_vms=max_vms,
        )


class TestPreregisteredScenarios:
    """Scenarios the independent audit named, observed as counts and as timing.

    Each asserts on something the driver cannot fake for itself: peak
    concurrency reconstructed by the scripted fleet, wall clock read off the
    injected clock, or the number of submissions a second driver made.
    """

    def test_the_cap_is_reached_and_never_exceeded(self, storage, clock):
        """A cap that is never reached is money left on the table, not safety.

        Both halves matter. Peak above the cap is the over-run this design was
        corrected for; a peak far below it with work still queued means the
        wave batching is not filling the fleet it paid to start.
        """
        summary, fleet = _run(storage, clock, TILES, max_vms=4)

        assert sorted(summary.completed) == sorted(TILES)
        assert fleet.peak_in_flight == 4

    def test_no_poll_is_spent_idle_before_the_first_wave_goes_out(self, storage, clock):
        """Work that could start now starts now, however the window is set.

        The quiescence condition exists for exactly this: with every tile
        demanding in the first poll there is nobody left to wait for, so waiting
        out ``fleet_wave_window_s`` would be an hour of paid-for nothing at the
        head of every run. Asserted against the clock, not against a count.
        """
        started = clock.now()
        summary, fleet = _run(storage, clock, TILES, wave_window_s=100_000.0, max_vms=4)

        assert sorted(summary.completed) == sorted(TILES)
        assert fleet.calls_for("offsets")[0]["at"] == started

    def test_a_unit_that_lands_past_its_deadline_still_completes_its_tile(self, storage, clock):
        """Late is not failed. The f6cf6fc reproduction, driven to settlement.

        The first wave expires with its units still running. The old model
        retired it, released width, and let the tile burn both barrier rounds on
        healthy workers. Here the tile finishes.
        """
        settings_rounds = settings.shard_barrier_rounds
        plans = _plans(TILES[:1])
        fleet = FakeWaveFleet(
            storage,
            plans,
            clock=clock,
            lands_after={(TILES[0], "offsets", index): 6 for index in range(2)},
        )
        summary, fleet = _run(storage, clock, TILES[:1], fleet=fleet, max_vms=4)

        assert summary.completed == TILES[:1]
        assert summary.failed == []
        assert len(fleet.calls_for("offsets")) <= settings_rounds

    def test_two_rounds_cost_two_budgets_of_wall_clock(self, storage, clock):
        """A second round opens on its own deadline, not on the first one's.

        The defect this pins is the one ADR-016 already paid for once: a round
        measured from the *first* submission expires having watched for nothing.
        Read off the injected clock, which is the only thing that can see it.
        """
        broken = TILES[0]
        fleet = FakeWaveFleet(
            storage,
            _plans(TILES[:2]),
            clock=clock,
            never={(broken, "offsets", index) for index in range(15)},
        )
        summary, fleet = _run(storage, clock, TILES[:2], fleet=fleet)

        carrying = [call for call in fleet.calls_for("offsets") if broken in call["tiles"]]
        assert len(carrying) == settings.shard_barrier_rounds
        first = driver_deadline(summary, "offsets", carrying[0]["wave"])
        assert carrying[1]["at"] - carrying[0]["at"] >= first

    def test_a_second_driver_over_a_round_exhausted_run_submits_nothing(self, storage, clock):
        """Rounds are counted across drivers, so a resume does not mint an allowance."""
        broken = TILES[0]
        fleet = FakeWaveFleet(
            storage,
            _plans(TILES[:2]),
            clock=clock,
            never={(broken, "offsets", index) for index in range(15)},
        )
        summary, fleet = _run(storage, clock, TILES[:2], fleet=fleet)
        assert summary.failed == [broken]

        second = FakeWaveFleet(storage, _plans(TILES[:2]), clock=clock)
        ticking = TickingStorage(storage, second)
        resume_fleet(RUN_ID, storage=ticking, backend=second, clock=clock, wave_window_s=0.0)

        assert second.calls == []

    def test_a_duplicate_submission_is_counted_against_the_cap(self, storage, clock):
        """At-least-once is permitted, so the accounting has to survive it.

        Two live waves carrying the same unit is waste, never corruption -- the
        unit is idempotent at its artifact key. What it must not do is read as
        one worker.
        """
        driver, _track, _fleet = self._never_landing(storage, clock, max_vms=8)
        driver._buffer_demand(_demand(TILES[0], "offsets", (0, 1)))
        driver._flush("offsets")
        driver._buffer_demand(_demand(TILES[0], "offsets", (0, 1), round_no=2))
        driver._flush("offsets")

        assert len(driver._live) == 2
        assert driver.in_flight == 4
        assert driver.in_flight <= driver.max_vms

    def test_an_instrumentation_write_failure_never_fails_the_run(self, storage, clock):
        """Losing a bookkeeping object costs observability, never a composite."""

        class RefusingState:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def write_text(self, key, text):
                if "/state/" in key:
                    msg = "no writes for you"
                    raise OSError(msg)
                return self._inner.write_text(key, text)

        plans = _plans(TILES)
        fleet = FakeWaveFleet(storage, plans, clock=clock)
        refusing = RefusingState(TickingStorage(storage, fleet))
        for writer in fleet.writers.values():
            writer.storage = refusing
        summary = drive_fleet(
            _jobs(TILES),
            run_id=RUN_ID,
            storage=refusing,
            backend=fleet,
            clock=clock,
            wave_window_s=0.0,
        )

        assert sorted(summary.completed) == sorted(TILES)

    def test_a_backend_that_cannot_see_the_storage_refuses_the_run(self, storage, clock):
        """The failure a barrier turns into a hang, caught before anything boots."""
        from landsat_lst.shard_driver import ShardBackendMismatch

        with pytest.raises(ShardBackendMismatch):
            drive_fleet(
                _jobs(TILES[:1]),
                run_id=RUN_ID,
                storage=storage,
                backend=CoiledFleetBackend(),
                clock=clock,
            )

    def test_the_credit_estimate_scales_with_the_build_size(self, monkeypatch):
        """One tile's estimate times the roster. Over-estimates, which is the safe way."""
        from landsat_lst import quota

        seen: list[float] = []
        monkeypatch.setattr(quota, "preflight_identity", lambda: None)
        monkeypatch.setattr(quota, "estimate_run_credits", lambda: 100.0)
        monkeypatch.setattr(
            quota,
            "preflight_credits",
            lambda estimate, **_kwargs: (
                seen.append(estimate) or SimpleNamespace(remaining=None, source="stub")
            ),
        )

        CoiledFleetBackend().preflight(tiles=7)

        assert seen == [700.0]

    def test_each_wave_records_when_its_first_and_last_unit_landed(self, storage, clock):
        """Provisioning idle has to be attributable per wave, not per run.

        Poll-resolution observations of the bucket, so they can only be late.
        That is stated rather than hidden: a driver cannot see a worker's clock,
        and a total that cannot be split into waiting and working calibrates
        nothing.
        """
        summary, _fleet = _run(storage, clock, TILES, max_vms=8)

        offsets = [wave for wave in summary.waves if wave.stage == "offsets"]
        assert offsets
        for wave in offsets:
            assert wave.first_completion_at is not None
            assert wave.last_completion_at is not None
            assert wave.last_completion_at >= wave.first_completion_at
            assert wave.provisioning_idle_s >= 0.0
        body = json.loads(
            storage.read_text(shards.fleet_submission_key(RUN_ID, "offsets", offsets[0].wave))
        )
        assert body["first_completion_at"] is not None

    def test_each_unit_publishes_how_long_it_ran(self, monkeypatch, tmp_path):
        """Per-wave stamps bound billed time; only durations measure idle.

        A worker between units and a worker running one are indistinguishable
        from the bucket, so billed minus boot minus work needs the work term,
        and the work term can only come from the unit.
        """
        from landsat_lst import fleet_driver

        store = LocalStorage(tmp_path)
        write_manifest(store, "r1", _jobs([TILES[0]]), units=8)
        monkeypatch.setattr("landsat_lst.shard_tasks.run_shard", lambda *_a, **_k: "ok")

        fleet_driver.run_unit("r1", "composite", f"{TILES[0]}:3", storage=store)

        body = json.loads(store.read_text(shards.unit_timing_key("r1", "composite", TILES[0], 3)))
        assert body["duration_s"] >= 0.0
        assert body["ended_at"] >= body["started_at"]
        assert body["failed"] is False
        # Out of the prefix the driver lists every poll, on purpose.
        assert not shards.unit_timing_key("r1", "composite", TILES[0], 3).startswith(
            shards.fleet_root("r1") + "/"
        )

    def test_a_failed_unit_still_publishes_its_interval(self, monkeypatch, tmp_path):
        """A unit that died after twenty minutes cost twenty minutes."""
        from landsat_lst import fleet_driver

        store = LocalStorage(tmp_path)
        write_manifest(store, "r1", _jobs([TILES[0]]), units=8)

        def boom(*_args, **_kwargs):
            raise RuntimeError("unit died")

        monkeypatch.setattr("landsat_lst.shard_tasks.run_shard", boom)

        with pytest.raises(RuntimeError):
            fleet_driver.run_unit("r1", "composite", f"{TILES[0]}:1", storage=store)

        body = json.loads(store.read_text(shards.unit_timing_key("r1", "composite", TILES[0], 1)))
        assert body["failed"] is True

    def test_a_timing_write_failure_never_fails_the_unit(self, monkeypatch, tmp_path):
        """INV-27, applied to the newest instrumentation in the codebase."""
        from landsat_lst import fleet_driver

        class RefusingTimings:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def write_text(self, key, text):
                if "/timings/" in key:
                    msg = "no"
                    raise OSError(msg)
                return self._inner.write_text(key, text)

        store = LocalStorage(tmp_path)
        write_manifest(store, "r1", _jobs([TILES[0]]), units=8)
        monkeypatch.setattr("landsat_lst.shard_tasks.run_shard", lambda *_a, **_k: "ok")

        assert (
            fleet_driver.run_unit(
                "r1", "composite", f"{TILES[0]}:0", storage=RefusingTimings(store)
            )
            == "ok"
        )

    def _never_landing(self, storage, clock, *, max_vms):
        plans = _plans(TILES[:1])
        fleet = FakeWaveFleet(
            storage,
            plans,
            clock=clock,
            never={(TILES[0], "offsets", index) for index in range(4)},
        )
        track = TileTrack(
            tile=TILES[0],
            run_id=RUN_ID,
            root=shards.shard_root(RUN_ID, TILES[0]),
            storage=storage,
            units=2,
            clock=clock,
            plan=plans[0],
        )
        driver = FleetDriver(
            run_id=RUN_ID,
            tracks=[track],
            storage=storage,
            backend=fleet,
            clock=clock,
            units=2,
            max_vms=max_vms,
        )
        return driver, track, fleet
