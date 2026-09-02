"""The census, and the capacity rule built on it.

Every assertion here is about the one invariant that replaced the ad-hoc
capacity rules::

    H(t) = max( intent_charge(t), census(t).total )

with ``intent_charge`` counting every submission *attempt* at its full
requested width from the instant the call is issued, and a charge discharged
only by a census taken afterwards that does not contain its identity.

The three divergence windows that motivated it are each a scenario below, and
each is stated as the pair of executions the old observation model could not
tell apart:

- **Window A**, hung task: ``W`` workers with one unit stuck, against zero
  workers because the array was preempted. Identical buckets, identical clocks;
  different censuses.
- **Window B**, lost acknowledgement: the submission started VMs and returned
  nothing. The driver holds no id at all, so only an identity fixed *before* the
  call can find them.
- **Window C**, submission failure: a recorded wave that never started. The old
  driver held its width for the life of the run and failed every tile;
  a census discharges it on the first poll.

Nothing here reaches a control plane. :class:`InMemoryFleetBackend` is a second
implementation of the whole contract, which is what makes the contract a
contract rather than a description of ``CoiledFleetBackend``.
"""

from __future__ import annotations

import sys
import types

import pytest

from landsat_lst import batch, shards
from landsat_lst.config import settings
from landsat_lst.fleet_backend import (
    BACKEND_CONTRACT,
    InMemoryFleetBackend,
    WaveRequest,
    WorkerCensus,
    check_contract,
)
from landsat_lst.fleet_driver import FleetDriver, GhostLedger, TileTrack, Wave
from landsat_lst.storage import LocalStorage
from tests.unit.shard_fixtures import make_plan
from tests.unit.test_driver_state_machine import FakeClock

pytestmark = pytest.mark.unit

RUN_ID = "census-test"
TILE = "N40W075"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path)


def _driver(storage, clock, backend, *, max_vms=4) -> FleetDriver:
    track = TileTrack(
        tile=TILE,
        run_id=RUN_ID,
        root=shards.shard_root(RUN_ID, TILE),
        storage=storage,
        units=2,
        clock=clock,
        plan=make_plan(tile=TILE),
    )
    return FleetDriver(
        run_id=RUN_ID,
        tracks=[track],
        storage=storage,
        backend=backend,
        clock=clock,
        units=2,
        max_vms=max_vms,
    )


def _wave(**kw) -> Wave:
    base = {
        "stage": "offsets",
        "wave": 1,
        "units": ((TILE, 0), (TILE, 1)),
        "tiles": (TILE,),
        "max_workers": 2,
        "requested_workers": 2,
        "submitted_at": 0.0,
        "deadline_s": 100.0,
        "identity": "id-1",
        "attempts": 1,
        "last_attempt_at": 0.0,
    }
    return Wave(**{**base, **kw})


# --------------------------------------------------------------------------
# the contract, satisfied twice
# --------------------------------------------------------------------------


class TestTheContractHasTwoImplementations:
    def test_the_in_memory_backend_declares_and_answers_the_whole_contract(self):
        """A protocol with one implementation is a description of that implementation.

        The census additions are only worth their weight if a second substrate
        can satisfy them, so the fake is held to the same check the driver
        applies to the real one at construction.
        """
        backend = InMemoryFleetBackend()
        check_contract(backend)
        assert {"enumerable_by_run", "census_is_authoritative"} <= BACKEND_CONTRACT
        assert backend.guarantees >= BACKEND_CONTRACT

    def test_the_identity_is_a_pure_function_of_the_request(self):
        """Computable before the call, and the same string every time.

        This is the whole of ``enumerable_by_run``: an id that arrives in a
        reply cannot recover a resource whose reply was lost.
        """
        backend = InMemoryFleetBackend()
        first = backend.submission_identity(RUN_ID, "offsets", 3)
        assert first == backend.submission_identity(RUN_ID, "offsets", 3)
        assert first != backend.submission_identity(RUN_ID, "offsets", 4)
        assert first != backend.submission_identity("other", "offsets", 3)

    def test_the_coiled_identity_is_the_cluster_name_and_shares_the_run_prefix(self):
        """Coiled's own name is already the identity; nothing new is invented.

        Asserted because the census finds a run's clusters by this prefix, so a
        name that stopped starting with it would be a silent loss of every
        orphan -- the one resource class nothing else can see.
        """
        name = batch.fleet_cluster_name(RUN_ID, "offsets", 2)
        assert name.startswith(batch.fleet_cluster_prefix(RUN_ID))
        assert not name.startswith(batch.fleet_cluster_prefix("another-run"))

    def test_reap_is_a_request_and_the_census_is_the_confirmation(self):
        """The delete returns nothing, so only a later census can settle it."""
        backend = InMemoryFleetBackend()
        backend.submit(WaveRequest(stage="offsets", run_id=RUN_ID, units=(), wave=1, max_workers=3))
        identity = backend.submission_identity(RUN_ID, "offsets", 1)
        assert backend.census(RUN_ID).total == 3

        assert backend.reap(RUN_ID, identity) is None
        assert backend.census(RUN_ID).identities == frozenset()
        # Idempotent: repeating it until a census agrees must be safe.
        backend.reap(RUN_ID, identity)
        assert backend.reaped == [identity, identity]


# --------------------------------------------------------------------------
# unavailable is not empty
# --------------------------------------------------------------------------


class TestUnknownIsNeverZero:
    def test_an_unanswerable_substrate_yields_none_rather_than_an_empty_census(self):
        backend = InMemoryFleetBackend(answerable=False)
        backend.submit(WaveRequest(stage="offsets", run_id=RUN_ID, units=(), wave=1, max_workers=3))
        assert backend.census(RUN_ID) is None

    def test_a_census_that_raises_degrades_rather_than_freeing_the_cap(self, storage, clock):
        """The failure mode this exists to stop: a blip reading as an empty fleet.

        A driver that treated an unreachable control plane as "nothing is
        running" would offer its whole cap as headroom at exactly the moment it
        had lost sight of the bill.
        """
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend)
        driver._live.append(_wave())

        def boom(_run_id):
            msg = "control plane unavailable"
            raise RuntimeError(msg)

        backend.census = boom
        driver.take_census()

        assert driver._census is None
        assert driver.in_flight == 2, "an unreadable census must not release a charge"

    def test_a_backend_with_no_census_at_all_runs_degraded_and_says_so(self, storage, clock):
        """Logged, not inferred. The code this replaces was permanently in a
        worse version of this mode and silent about it.
        """

        class NoCensus(InMemoryFleetBackend):
            """A backend written before ``enumerable_by_run`` existed."""

            census = None

        backend = NoCensus(clock=clock)
        driver = _driver(storage, clock, backend)
        driver.take_census()

        assert driver._census is None
        assert driver._degraded is True
        assert driver.capacity_ledger()["degraded"] is True
        assert driver.capacity_ledger()["census_total"] is None

    def test_the_degraded_transition_is_logged_once_and_not_every_poll(self, storage, clock):
        """A multi-hour run polls tens of thousands of times."""
        backend = InMemoryFleetBackend(clock=clock, answerable=False)
        driver = _driver(storage, clock, backend)

        seen: list[str] = []
        driver._note_census_mode = lambda census: seen.append(
            "degraded" if census is None else "ok"
        )
        for _ in range(3):
            driver.take_census()
        assert seen == ["degraded"] * 3, "the hook is per poll; the *log* is per transition"

        driver = _driver(storage, clock, backend)
        driver.take_census()
        assert driver._degraded is True
        driver.take_census()
        assert driver._degraded is True, "still degraded, and no second crossing to log"
        backend.answerable = True
        driver.take_census()
        assert driver._degraded is False


# --------------------------------------------------------------------------
# the invariant
# --------------------------------------------------------------------------


class TestTheCapacityInvariant:
    def test_h_is_the_maximum_of_the_intent_charge_and_the_census(self, storage, clock):
        """Neither term dominates, which is why neither is dropped.

        The census lags a submission, so the intent charge covers the boot. The
        intent charge cannot see a worker belonging to no attempt this driver
        holds, so the census covers the orphan.
        """
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend, max_vms=64)
        driver._live.append(_wave(requested_workers=5, max_workers=5))

        driver._census = WorkerCensus(as_of=1.0, total=0, by_identity={}, identities=frozenset())
        assert driver.in_flight == 5, "a census that has not caught up must not undercut intent"

        driver._census = WorkerCensus(
            as_of=1.0, total=9, by_identity={"id-1": 5, "orphan": 4}, identities=frozenset({"id-1"})
        )
        assert driver.in_flight == 9, "workers this driver holds no attempt for still bill"

    def test_an_orphan_the_driver_holds_no_handle_for_is_still_charged(self, storage, clock):
        """Window B. The measured 90-up / 30-counted, from the other side.

        Nothing the driver recorded refers to these workers. Only the run-wide
        census does, which is the whole reason the census is run-wide.
        """
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend, max_vms=64)
        driver._census = WorkerCensus(
            as_of=1.0, total=30, by_identity={"lost": 30}, identities=frozenset({"lost"})
        )
        assert driver._live == []
        assert driver.intent_charge == 0
        assert driver.in_flight == 30
        assert driver.headroom == 34
        assert driver.capacity_ledger()["census_unattributed"] == 0

    def test_a_lost_acknowledgement_leaves_workers_only_the_run_id_can_find(self, clock):
        """The recovery path, end to end, on a substrate that models the window.

        The submission starts its workers and then raises, so the caller holds
        no handle and no id. The identity was fixed before the call, so the
        census finds them anyway.
        """
        backend = InMemoryFleetBackend(clock=clock)
        backend.lose_next_answer()
        request = WaveRequest(stage="offsets", run_id=RUN_ID, units=(), wave=1, max_workers=7)
        with pytest.raises(ConnectionError):
            backend.submit(request)

        census = backend.census(RUN_ID)
        assert census.total == 7
        assert census.identities == frozenset({backend.submission_identity(RUN_ID, "offsets", 1)})

    def test_every_attempt_is_charged_at_its_full_requested_width(self):
        """Window B's mechanism. There is no idempotency key on this path.

        An attempt that raises on the way back may have created a cluster and
        booted its workers, so charging only the attempt that answered charges a
        subset of what may be running.
        """
        wave = _wave(attempts=0, requested_workers=6, max_workers=6)
        assert wave.intent_charge() == 6, "a wave is charged before its first attempt returns"

        wave.attempts = settings.shard_submit_retries
        assert wave.intent_charge() == 6 * settings.shard_submit_retries

    def test_the_charge_takes_the_requested_width_over_a_clamped_handle(self):
        """A handle can only ever report a width the substrate clamped downward."""
        wave = _wave(requested_workers=8, max_workers=3)
        assert wave.intent_charge() == 8

    def test_a_discharged_wave_is_charged_nothing(self):
        wave = _wave()
        wave.discharged = True
        assert wave.intent_charge() == 0


# --------------------------------------------------------------------------
# discharge: what may give a charge back, and what may not
# --------------------------------------------------------------------------


class TestDischargeIsAboutMachinesNotArtifacts:
    def test_a_census_that_omits_the_identity_discharges_the_charge(self, storage, clock):
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend, max_vms=64)
        driver._census = WorkerCensus(as_of=9.0, total=0, by_identity={}, identities=frozenset())

        assert driver._discharge_reason(_wave(last_attempt_at=1.0), 10.0) == "census_absent"

    def test_a_census_taken_before_the_attempt_may_not_discharge_it(self, storage, clock):
        """It could not have seen workers the attempt had not yet asked for."""
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend, max_vms=64)
        driver._census = WorkerCensus(as_of=1.0, total=0, by_identity={}, identities=frozenset())

        assert driver._discharge_reason(_wave(last_attempt_at=5.0), 10.0) is None

    def test_landed_artifacts_do_not_discharge_a_wave_the_census_still_reports(
        self, storage, clock
    ):
        """The category error, pinned. Window A's hung task, from the safe side.

        Every unit of this wave has published. The substrate says the VMs are
        still there. Bytes in a bucket are evidence that work completed and
        carry no function of whether a machine exists, so the census wins.
        """
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend, max_vms=64)
        driver._census = WorkerCensus(
            as_of=9.0, total=2, by_identity={"id-1": 2}, identities=frozenset({"id-1"})
        )
        wave = _wave(last_attempt_at=1.0)
        wave.outstanding_units = set()

        assert driver._discharge_reason(wave, 10.0) is None
        driver._live.append(wave)
        assert driver.in_flight == 2

    def test_a_probe_confirming_death_discharges_whatever_the_census_says(self, storage, clock):
        """The one thing a probe is allowed to assert, and it still is."""
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend, max_vms=64)
        driver._census = WorkerCensus(
            as_of=9.0, total=2, by_identity={"id-1": 2}, identities=frozenset({"id-1"})
        )
        wave = _wave(handle_id=77, last_attempt_at=1.0)
        driver._dead_handles.add(77)

        assert driver._discharge_reason(wave, 10.0) == "probe_dead"

    def test_a_stranded_wave_is_never_released_while_a_census_reports_it(self, storage, clock):
        """``stranded_at`` is demoted to a barrier input, and this is the demotion.

        A forecast may be exceeded; a bound may not. Substituting the first for
        the second was the whole of Window A, so a wave past its budget that the
        substrate still reports keeps its charge.
        """
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend, max_vms=64)
        driver._census = WorkerCensus(
            as_of=9_000.0, total=2, by_identity={"id-1": 2}, identities=frozenset({"id-1"})
        )
        wave = _wave(last_attempt_at=1.0)
        wave.outstanding_units = {(TILE, 1)}

        assert wave.stranded(9_000.0) is True
        assert driver._discharge_reason(wave, 9_000.0) is None

    def test_a_census_settles_a_window_c_record_that_started_nothing(self, storage, clock):
        """Window C, which cost four tiles of four.

        A recorded wave whose submission never happened has no handle to probe
        and no artifact that will ever land, so the old driver held its width
        for the life of the run. The census does not report it, so it is gone on
        the first poll and the tile gets its cap back.
        """
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend, max_vms=4)
        wave = _wave(requested_workers=4, max_workers=4, last_attempt_at=0.0, units=())
        driver._live.append(wave)
        driver._census = WorkerCensus(as_of=1.0, total=0, by_identity={}, identities=frozenset())
        assert driver.headroom == 0

        driver._retire()

        assert driver._live == []
        assert driver.headroom == 4
        assert wave.discharged_by == "census_absent"

    def test_a_settled_identity_lets_a_tile_stop_waiting_out_its_deadline(self, storage, clock):
        """Liveness inference routed off the bucket and onto the census.

        The handle set cannot reach a record whose submission answer was lost --
        its ``cluster_id`` is ``None``. The identity always can.
        """
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend, max_vms=4)
        wave = _wave(last_attempt_at=0.0)
        driver._live.append(wave)
        driver._census = WorkerCensus(as_of=1.0, total=0, by_identity={}, identities=frozenset())
        driver._retire()

        assert "id-1" in driver._dead_identities
        track = driver.tracks[0]
        track.dead_identities = driver._dead_identities
        record = {"cluster_id": None, "cluster_name": "id-1", "submitted_at": 0.0}
        assert track._is_live(record, deadline_s=10_000.0) is False


# --------------------------------------------------------------------------
# the degraded policy
# --------------------------------------------------------------------------


class TestTheGhostLedger:
    def test_an_entry_is_charged_until_its_ttl_and_not_after(self):
        ghosts = GhostLedger()
        ghosts.add("a", 5, now=0.0, ttl=100.0)
        assert ghosts.width(50.0) == 5
        assert ghosts.horizon(50.0) == 50.0
        assert ghosts.width(100.0) == 0
        assert ghosts.entries == {}

    def test_evidence_beats_the_timer(self):
        """The array ending settles an entry early rather than paying out the TTL."""
        ghosts = GhostLedger()
        ghosts.add("a", 5, now=0.0, ttl=100.0)
        ghosts.settle("a")
        assert ghosts.width(1.0) == 0

    def test_the_widest_release_of_one_identity_is_the_one_charged(self):
        ghosts = GhostLedger()
        ghosts.add("a", 2, now=0.0, ttl=10.0)
        ghosts.add("a", 6, now=0.0, ttl=10.0)
        assert ghosts.width(1.0) == 6

    def test_a_run_that_loses_its_census_releases_a_stranded_wave_but_keeps_charging_it(
        self, storage, clock
    ):
        """Escape E1, and the only path that reaches it.

        Holding a stranded wave forever is safe and deadlocks the run; releasing
        it at once is live and doubles the bill. Releasing it into a ledger that
        keeps charging the width is both, bounded at twice the cap.
        """
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend, max_vms=4)
        driver.take_census()
        assert driver._census_seen is True

        backend.answerable = False
        driver.take_census()
        wave = _wave(requested_workers=4, max_workers=4)
        wave.outstanding_units = {(TILE, 0), (TILE, 1)}
        driver._live.append(wave)
        clock.advance(100_000.0)

        driver._retire()

        assert driver._live == []
        assert wave.discharged_by == "stranded_unconfirmed"
        assert driver.in_flight == 4, "released, and still charged"
        assert driver.headroom == 0
        clock.advance(settings.fleet_ghost_ttl_s + 1.0)
        assert driver.in_flight == 0
        assert driver.headroom == 4

    def test_a_substrate_that_never_answered_holds_instead_of_guessing(self, storage, clock):
        """The gate on rule 5, and the reason it is there.

        A control plane that answered this run and stopped is a transient, and a
        TTL is a real bound on the release. A substrate that has never answered
        offers no evidence that anything it started is enumerable at all, so a
        timed release would be a guess with nothing behind it. There the driver
        holds, and the run ends loudly rather than quietly at twice its cap.
        """
        backend = InMemoryFleetBackend(clock=clock, answerable=False)
        driver = _driver(storage, clock, backend, max_vms=4)
        driver.take_census()
        assert driver._census_seen is False

        wave = _wave(requested_workers=4, max_workers=4)
        wave.outstanding_units = {(TILE, 0)}
        driver._live.append(wave)
        clock.advance(100_000.0)

        assert driver._discharge_reason(wave, clock.now()) is None
        assert driver.in_flight == 4

    def test_the_array_ending_discharges_without_a_ghost(self, storage, clock):
        """``queues_surplus`` ties the array's end to its last unit.

        That is the substrate's own contract read through the bucket, observed
        late and never early, so it is not a release that needs a ghost behind
        it. It is also the only bucket-shaped release left, and it exists only
        while no census can be taken.
        """
        backend = InMemoryFleetBackend(clock=clock, answerable=False)
        driver = _driver(storage, clock, backend, max_vms=4)
        driver.take_census()
        wave = _wave(requested_workers=4, max_workers=4)
        wave.outstanding_units = set()

        assert driver._discharge_reason(wave, clock.now()) == "array_ended"
        driver._discharge(wave, "array_ended", clock.now())

        assert wave.intent_charge() == 0
        assert driver._ghosts.width(clock.now()) == 0
        assert driver.in_flight == 0


# --------------------------------------------------------------------------
# the two caps
# --------------------------------------------------------------------------


class TestTheCapIsConcurrencyAndNotSpend:
    def test_the_ledger_names_the_concurrency_cap_and_reports_no_budget(self, storage, clock):
        """Named distinctly on purpose.

        ``fleet_max_vms`` bounds concurrency. Spend is the integral of
        concurrency over time and nothing here bounds the time, so the ledger
        publishes a concurrency cap and a measured concurrency, and says
        nothing about a budget it cannot enforce.
        """
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend, max_vms=17)
        driver.take_census()
        ledger = driver.capacity_ledger()

        assert ledger["concurrency_cap"] == 17
        assert "max_vms" not in ledger
        assert not any("budget" in key or "spend" in key for key in ledger)

    def test_flat_concurrency_does_not_bound_the_bill(self, storage, clock):
        """The property this work does *not* deliver, asserted so nobody claims it.

        Two censuses at the cap, an hour apart, with entirely different
        identities: concurrency never moved and the launches accumulated. The
        driver reports the same ``in_flight`` for both, because that is all a
        concurrency measurement can say.
        """
        backend = InMemoryFleetBackend(clock=clock)
        driver = _driver(storage, clock, backend, max_vms=4)

        driver._census = WorkerCensus(
            as_of=0.0, total=4, by_identity={"first": 4}, identities=frozenset({"first"})
        )
        first = driver.in_flight
        driver._census = WorkerCensus(
            as_of=3600.0, total=4, by_identity={"second": 4}, identities=frozenset({"second"})
        )

        assert first == driver.in_flight == 4
        assert driver.headroom == 0


# --------------------------------------------------------------------------
# the Coiled census, without a control plane
# --------------------------------------------------------------------------


class TestTheCoiledCensusCountsConservatively:
    def test_an_unrecognized_worker_state_counts_as_live(self):
        """The vocabulary is not pinned by the client source.

        It names ``pending``, ``assigned`` and ``error`` and substring-matches
        ``done``, and never enumerates the set. So an unknown state counts live,
        which is the rule ``classify_failure`` already follows for an unknown
        error: guess the answer that cannot silently free capacity.
        """
        assert batch._live_workers({"workers": [{"current_state": {"state": "provisioning"}}]}) == 1
        assert batch._live_workers({"workers": [{"current_state": {}}]}) == 1
        assert batch._live_workers({"workers": [{}]}) == 1

    def test_a_stopping_worker_is_still_billing(self):
        assert batch._live_workers({"workers": [{"current_state": {"state": "stopping"}}]}) == 1

    def test_a_worker_is_terminal_only_when_it_and_its_instance_are(self):
        # No instance projected: Coiled returns ``None`` there for a worker
        # that has none, and reading that as an unknown state would count every
        # torn-down worker as live forever.
        assert batch._live_workers({"workers": [{"current_state": {"state": "stopped"}}]}) == 0
        assert (
            batch._live_workers(
                {
                    "workers": [
                        {
                            "current_state": {"state": "stopped"},
                            "instance": {"current_state": {"state": "running"}},
                        }
                    ]
                }
            )
            == 1
        ), "a stopped worker process on a running instance is a running instance"

    def test_a_census_without_credentials_is_unknown_rather_than_empty(self, monkeypatch):
        """Credential-less is the CI runner, and it must not read as an idle fleet."""
        monkeypatch.setattr("landsat_lst.shard_driver._coiled_credentials_present", lambda: False)
        assert batch.fleet_worker_census(RUN_ID) is None

    def test_a_reap_without_credentials_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr("landsat_lst.shard_driver._coiled_credentials_present", lambda: False)
        assert batch.fleet_reap_identity(RUN_ID, "whatever") is None

    def test_an_unrecognized_cluster_state_counts_as_live(self):
        """The cluster-level half of the same rule, and it needs its own pin.

        A listed cluster retires only on a state the allowlist names. Inverting
        this into a running-allowlist makes the census report a *smaller* fleet
        than exists, which is the single direction it may never be wrong in,
        and it used to be possible to do that without failing a test.
        """
        assert batch._cluster_is_live({"current_state": {"state": "scaling"}})
        assert batch._cluster_is_live({"current_state": {"state": "starting"}})
        assert batch._cluster_is_live({"current_state": {}})
        assert batch._cluster_is_live({}), "an absent state is unknown, and unknown is live"
        assert batch._cluster_is_live({"current_state": {"state": "stopping"}}), (
            "a cluster being torn down is a cluster being billed"
        )

    def test_a_cluster_retires_only_on_a_state_the_allowlist_names(self):
        assert not batch._cluster_is_live({"current_state": {"state": "stopped"}})
        assert not batch._cluster_is_live({"current_state": {"state": "error"}})
        assert not batch._cluster_is_live({"current_state": "terminated"})

    def test_the_census_counts_a_cluster_whose_state_it_cannot_read(self, monkeypatch):
        """The same rule at the call site, since that is where it was invertible.

        Two clusters, one in a state this client has never heard of and one
        genuinely stopped. The census has to report the first and drop the
        second, and it has to do it without a control plane.
        """
        listed = [
            {
                "name": batch.fleet_cluster_prefix(RUN_ID) + "offse-w1",
                "id": 1,
                "current_state": {"state": "reticulating"},
            },
            {
                "name": batch.fleet_cluster_prefix(RUN_ID) + "offse-w2",
                "id": 2,
                "current_state": {"state": "stopped"},
            },
            {"name": "someone-elses-cluster", "id": 3, "current_state": {"state": "running"}},
        ]
        details = {1: {"workers": [{"current_state": {"state": "running"}}] * 5}}

        class _Cloud:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            @staticmethod
            def cluster_details(cluster_id):
                return details[cluster_id]

        monkeypatch.setattr("landsat_lst.shard_driver._coiled_credentials_present", lambda: True)
        monkeypatch.setitem(
            sys.modules, "coiled", types.SimpleNamespace(list_clusters=lambda **_kw: listed)
        )
        monkeypatch.setitem(sys.modules, "coiled.v2", types.ModuleType("coiled.v2"))
        monkeypatch.setitem(
            sys.modules,
            "coiled.v2.core",
            types.SimpleNamespace(Cloud=_Cloud),
        )

        census = batch.fleet_worker_census(RUN_ID)
        assert census is not None
        assert census.total == 5
        assert census.identities == frozenset({batch.fleet_cluster_prefix(RUN_ID) + "offse-w1"})


class TestACollidingSubmissionNameIsRefused:
    """``unique_wave_names``, and why stability alone was not enough.

    There is no idempotency key on this path, so a retry after a lost answer
    reuses the identity of an attempt that may already be billing. A substrate
    that widens the existing array instead of refusing puts
    ``shard_submit_retries`` times the cap up: 180 workers against a cap of 64,
    with 120 units dispatched twice, and no census is consulted between two
    attempts inside one retry loop. Coiled refuses, which is why the run is
    clean today; the contract now asks for it.
    """

    def test_the_contract_names_the_requirement(self):
        assert "unique_wave_names" in BACKEND_CONTRACT

    def test_the_in_memory_backend_refuses_a_second_live_submission(self, clock):
        backend = InMemoryFleetBackend(clock=clock)
        request = WaveRequest(stage="offsets", run_id=RUN_ID, units=(), wave=1, max_workers=7)
        backend.submit(request)

        with pytest.raises(RuntimeError, match="already running"):
            backend.submit(request)

        census = backend.census(RUN_ID)
        assert census.total == 7, "the refusal leaves the first array alone"

    def test_a_refused_retry_cannot_stack_a_second_width(self, clock):
        """The measured consequence, from the substrate's side rather than the driver's."""
        backend = InMemoryFleetBackend(clock=clock)
        backend.lose_next_answer()
        request = WaveRequest(stage="offsets", run_id=RUN_ID, units=(), wave=1, max_workers=60)
        with pytest.raises(ConnectionError):
            backend.submit(request)

        for _ in range(settings.shard_submit_retries):
            with pytest.raises(RuntimeError, match="already running"):
                backend.submit(request)

        assert backend.census(RUN_ID).total == 60, (
            "one identity, one width, however many times the retry loop asks"
        )

    def test_a_reaped_identity_may_be_submitted_again(self, clock):
        """The refusal is about what is *running*, not about a name being used once."""
        backend = InMemoryFleetBackend(clock=clock)
        request = WaveRequest(stage="offsets", run_id=RUN_ID, units=(), wave=1, max_workers=3)
        backend.submit(request)
        backend.reap(RUN_ID, backend.submission_identity(RUN_ID, "offsets", 1))

        backend.submit(request)
        assert backend.census(RUN_ID).total == 3
