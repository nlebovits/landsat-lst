"""An adversarial harness for the consolidated fleet driver: physics, not arithmetic.

Everything here exists because the shipped fleet tests measure the driver with
the driver's own formula. ``FakeWaveFleet.observe`` computes concurrency as
``min(call["max_workers"], outstanding)``, which is character-for-character
:meth:`landsat_lst.fleet_driver.FleetDriver.wave_held`. Five tests assert a cap
against it, and all five pass on a driver whose release rule is the bug: an
independent cluster-lifetime model of the same 50-tile run peaks at 82 VMs
against a cap of 64.

So nothing in this module reads driver state. Three deliberate separations:

- **Identities, not counts.** :class:`WorkerLedger` opens one record per worker
  a submission asks the substrate to start and closes it when the substrate
  stops billing it. The peak is a sweep over intervals. There is no subtraction
  anywhere, so no release rule to agree with.
- **The substrate's release rule, not the driver's.** :class:`SimBackend` models
  ``coiled.batch_run`` as it is documented to behave and as ADR-018 relies on it
  behaving: ``W`` VMs boot once and drain a FIFO of ``U`` units, and the *array*
  is what ends. A VM whose own share of the queue is empty is still up and still
  billing until the last unit of the wave lands.
- **Production-shaped work.** :func:`production_plan` prices a composite unit in
  the tens of minutes. ``shard_fixtures.make_plan`` cuts four scenes into two
  batches while every driver test asks for fifteen units, so thirteen of every
  fifteen units land instantly and a composite unit is budgeted at about 1.4 s.
  No capacity or deadline property survives that.

Two stubs keep the scale simulations bounded. ``merge_offsets`` assembles a real
offset record from real partials; at 700 tiles that is 700 scientific merges and
the reviewer's brute-force run was still going after 37 minutes. ``_read_plan``
rebuilds a plan through ``load_context``, which wants a STAC item list. Neither
is under test here -- the state machine is -- and both are replaced by
:func:`stub_scientific_work` with objects the driver reads identically.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

from landsat_lst import budgets, shard_tasks, shards
from landsat_lst.config import settings
from landsat_lst.fleet_backend import BACKEND_CONTRACT, WaveHandle, WorkerCensus
from landsat_lst.models import ProcessingJob
from landsat_lst.shard_driver import Clock, _expected_keys, classify_failure
from landsat_lst.storage import PRODUCTS, StorageBackend, collection_prefix
from landsat_lst.tiling import parse_tile_name

if TYPE_CHECKING:
    from pathlib import Path

WINDOW = "2021-2025"

#: S3 ``ListObjectsV2`` returns at most this many keys per request. A driver
#: that issues one *call* per poll still pays one request per page of keys, and
#: the difference is the whole of INV-26.
S3_PAGE = 1000


# --------------------------------------------------------------------------
# storage, counted in requests
# --------------------------------------------------------------------------


class MemoryStorage(StorageBackend):
    """Keys in a dict, with the listing cost recorded in requests as well as calls.

    In memory rather than on disk because the scale simulations write tens of
    thousands of keys and list them hundreds of times; the sorted key list is
    maintained lazily so a prefix slice costs a bisect plus the keys it matches.

    ``requests`` is the number that matters. ``calls`` is the number the
    driver's own ``PollIndex.listings`` counter reports, kept beside it so a
    test can pin the discrepancy rather than argue about it.
    """

    def __init__(self) -> None:
        self.objects: dict[str, str] = {}
        self.stamps: dict[str, datetime] = {}
        self._sorted: list[str] = []
        self._dirty = False
        self.calls = 0
        self.keys_returned = 0
        self.requests = 0
        #: A hook the simulated substrate uses to publish whatever became due
        #: since the last observation. A poll is what makes time pass.
        self.on_list = None

    # -- StorageBackend ---------------------------------------------------

    def write_text(self, key: str, text: str, *, content_type: str = "application/json") -> None:
        del content_type
        if key not in self.objects:
            self._dirty = True
            self._sorted.append(key)
        self.objects[key] = text
        self.stamps[key] = datetime.now(tz=UTC)

    def read_text(self, key: str) -> str | None:
        return self.objects.get(key)

    def list_prefix(self, prefix: str) -> dict[str, datetime]:
        if self.on_list is not None:
            self.on_list()
        if self._dirty:
            self._sorted.sort()
            self._dirty = False
        out: dict[str, datetime] = {}
        for key in self._sorted[bisect.bisect_left(self._sorted, prefix) :]:
            if not key.startswith(prefix):
                break
            out[key] = self.stamps[key]
        self.calls += 1
        self.keys_returned += len(out)
        self.requests += max(1, math.ceil(len(out) / S3_PAGE))
        return out

    def delete_prefix(self, prefix: str) -> int:
        doomed = [k for k in self.objects if k.startswith(prefix)]
        for key in doomed:
            del self.objects[key]
            self.stamps.pop(key, None)
            self._sorted.remove(key)
        return len(doomed)

    def cog_exists(self, window: str, tile: str) -> bool:
        return all(self.cog_key(window, tile, product) in self.objects for product in PRODUCTS)

    def upload(self, local: Path, key: str) -> None:
        self.write_text(key, "cog")

    def list_completed(self, window: str) -> set[str]:
        prefix = f"{collection_prefix(window)}/"
        return {key[len(prefix) :].split("/")[0] for key in self.objects if key.startswith(prefix)}

    def download(self, key: str, local: Path) -> bool:
        return key in self.objects


# --------------------------------------------------------------------------
# the clock
# --------------------------------------------------------------------------


class SimClock(Clock):
    """Injected time. ``sleep`` advances instead of blocking."""

    def __init__(self, start: float = 1_800_000_000.0) -> None:
        self._now = start
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


# --------------------------------------------------------------------------
# the external ledger of worker identities
# --------------------------------------------------------------------------


@dataclass
class WorkerIdentity:
    """One VM the substrate was asked to start, and when it stopped billing."""

    wave: int
    stage: str
    slot: int
    started_at: float
    stopped_at: float | None = None

    @property
    def name(self) -> str:
        return f"{self.stage}-w{self.wave}-vm{self.slot}"


class WorkerLedger:
    """Potentially-live worker identities, observed at the backend boundary.

    The cap oracle. It knows two events -- a submission asked for ``W`` workers,
    and the substrate stopped billing one of them -- and derives concurrency by
    sweeping the resulting intervals. It never reads ``FleetDriver.in_flight``,
    ``wave_held``, ``headroom`` or ``TileTrack.outstanding``, and it contains no
    expression of the form ``min(max_workers, outstanding)``.

    A worker with no stop time is still live: the run ended with it up. That is
    the honest reading of a wave nobody could confirm dead, and it is precisely
    the case the driver's own accounting turns into free headroom.
    """

    def __init__(self) -> None:
        self.identities: list[WorkerIdentity] = []

    def opened(self, *, stage: str, wave: int, width: int, at: float) -> list[WorkerIdentity]:
        made = [
            WorkerIdentity(wave=wave, stage=stage, slot=slot, started_at=at)
            for slot in range(int(width))
        ]
        self.identities.extend(made)
        return made

    @staticmethod
    def closed(identities: list[WorkerIdentity], at: float) -> None:
        for identity in identities:
            if identity.stopped_at is None:
                identity.stopped_at = at

    # -- measurement ------------------------------------------------------

    def live_at(self, when: float) -> list[WorkerIdentity]:
        return [
            identity
            for identity in self.identities
            if identity.started_at <= when
            and (identity.stopped_at is None or when < identity.stopped_at)
        ]

    def peak(self) -> int:
        """Maximum simultaneously-live identities, by interval sweep."""
        deltas: list[tuple[float, int]] = []
        for identity in self.identities:
            deltas.append((identity.started_at, 1))
            if identity.stopped_at is not None:
                deltas.append((identity.stopped_at, -1))
        deltas.sort(key=lambda pair: (pair[0], pair[1]))
        live = peak = 0
        for _, delta in deltas:
            live += delta
            peak = max(peak, live)
        return peak

    def peak_moment(self) -> float:
        """When the peak happened, so a failure message can name it."""
        best_t, best_n = 0.0, -1
        for when in sorted({identity.started_at for identity in self.identities}):
            n = len(self.live_at(when))
            if n > best_n:
                best_t, best_n = when, n
        return best_t

    @property
    def never_stopped(self) -> list[WorkerIdentity]:
        return [identity for identity in self.identities if identity.stopped_at is None]


# --------------------------------------------------------------------------
# the substrate
# --------------------------------------------------------------------------


class SimWave:
    """``W`` VMs booting once and draining a FIFO of ``U`` units.

    ``coiled.batch_run`` starts ``max_workers`` VMs against one job array and
    hands them the values in order. Unit ``i`` therefore runs on VM ``i % W``
    after ``i // W`` units have gone before it on that VM, and the array -- the
    thing that is billed -- ends when its last unit does. That last fact is the
    one the driver's release rule contradicts.
    """

    def __init__(
        self,
        *,
        stage: str,
        wave: int,
        units: tuple[tuple[str, int], ...],
        width: int,
        submitted_at: float,
        boot_s: float,
        unit_work_s: float,
        never: frozenset[tuple[str, int]] = frozenset(),
        killed_at: float | None = None,
    ) -> None:
        self.stage = stage
        self.wave = wave
        self.units = units
        self.width = max(1, int(width))
        self.submitted_at = submitted_at
        self.boot_s = boot_s
        self.unit_work_s = unit_work_s
        self.never = never
        self.killed_at = killed_at
        self.due: list[tuple[float, tuple[str, int]]] = []
        for position, unit in enumerate(units):
            finish = submitted_at + boot_s + (position // self.width + 1) * unit_work_s
            if unit not in never:
                self.due.append((finish, unit))
        self.due.sort(key=lambda pair: pair[0])
        self._cursor = 0
        #: The array's end: the last unit's finish. ``None`` when some unit
        #: never lands, because then the array never ends either.
        if never:
            self.array_ends_at: float | None = None
        elif self.due:
            self.array_ends_at = self.due[-1][0]
        else:
            self.array_ends_at = submitted_at
        if killed_at is not None:
            self.array_ends_at = killed_at

    def newly_landed(self, now: float) -> list[tuple[str, int]]:
        if self.killed_at is not None and now >= self.killed_at:
            # Everything not already published dies with the VMs.
            self._cursor = len(self.due)
            return []
        out = []
        while self._cursor < len(self.due) and self.due[self._cursor][0] <= now:
            if self.killed_at is not None and self.due[self._cursor][0] > self.killed_at:
                break
            out.append(self.due[self._cursor][1])
            self._cursor += 1
        return out


class SimBackend:
    """A second implementation of the whole backend contract, that bills nothing.

    It declares every guarantee in :data:`BACKEND_CONTRACT` and honours each:
    every unit runs whatever ``max_workers`` says (``queues_surplus``), submit
    returns at once (``fire_and_forget``), the handle is an int
    (``opaque_handle``), and ``probe`` answers ``None`` unless a death was
    scripted (``probe_is_advisory``).
    """

    name = "adversarial-sim"
    guarantees = frozenset(BACKEND_CONTRACT)

    def __init__(
        self,
        storage: MemoryStorage,
        writers: dict,
        *,
        clock: SimClock,
        ledger: WorkerLedger,
        terms: dict[str, tuple[float, float]],
        never: set[tuple[str, str, int]] | None = None,
        kill_wave: int | None = None,
        kill_after_s: float = 0.0,
        probe_answer=None,
        probe_error: Exception | None = None,
        submit_error=None,
    ) -> None:
        self.storage = storage
        self.writers = writers
        self.clock = clock
        self.ledger = ledger
        self.terms = terms
        self.never = set(never or ())
        self.kill_wave = kill_wave
        self.kill_after_s = kill_after_s
        self.probe_answer = probe_answer
        self.probe_error = probe_error
        self.submit_error = submit_error
        self.waves: list[SimWave] = []
        self._slots: list[list[WorkerIdentity]] = []
        #: One record per submission: stage, units, width, moment. The only
        #: thing a test asserts submission behaviour against.
        self.calls: list[dict] = []
        self.submit_attempts = 0
        self.probe_calls = 0
        self._last_tick: float | None = None

    # -- FleetBackend -----------------------------------------------------

    def wave_name(self, run_id: str, stage: str, wave: int) -> str:
        return f"sim-{run_id}-{stage}-w{wave}"

    def classify_failure(self, error: BaseException) -> str:
        return classify_failure(error)

    def preflight(self, *, tiles: int) -> None:
        del tiles

    def validate_storage(self, storage) -> None:
        del storage

    def probe(self, handle_id):
        self.probe_calls += 1
        if self.probe_error is not None:
            raise self.probe_error
        if self.probe_answer is None:
            return None
        return self.probe_answer(handle_id, self)

    def submit(self, request) -> WaveHandle:
        self.submit_attempts += 1
        if self.submit_error is not None:
            error = self.submit_error(request, self)
            if error is not None:
                raise error
        units = tuple((tile, int(index)) for tile, index in request.units)
        width = int(request.max_workers or len(units))
        now = self.clock.now()
        boot_s, unit_work_s = self.terms.get(request.stage, (300.0, 600.0))
        never = frozenset(
            (tile, index) for tile, index in units if (request.stage, tile, index) in self.never
        )
        wave = SimWave(
            stage=request.stage,
            wave=request.wave,
            units=units,
            width=width,
            submitted_at=now,
            boot_s=boot_s,
            unit_work_s=unit_work_s,
            never=never,
            killed_at=(now + self.kill_after_s if self.kill_wave == len(self.waves) + 1 else None),
        )
        self.waves.append(wave)
        self._slots.append(
            self.ledger.opened(stage=request.stage, wave=request.wave, width=width, at=now)
        )
        self.calls.append(
            {
                "stage": request.stage,
                "wave": request.wave,
                "units": units,
                "tiles": sorted({tile for tile, _ in units}),
                "max_workers": width,
                "at": now,
            }
        )
        return WaveHandle(
            id=len(self.waves),
            name=self.wave_name(request.run_id, request.stage, request.wave),
            max_workers=width,
        )

    # -- the substrate running --------------------------------------------

    def tick(self) -> None:
        """Publish whatever became due, and close whatever stopped billing."""
        now = self.clock.now()
        if self._last_tick is not None and now == self._last_tick:
            return
        self._last_tick = now
        for wave, slots in zip(self.waves, self._slots, strict=True):
            for tile, index in wave.newly_landed(now):
                writer = self.writers.get(tile)
                if writer is not None:
                    writer.write(wave.stage, index)
            if wave.array_ends_at is not None and now >= wave.array_ends_at:
                self.ledger.closed(slots, wave.array_ends_at)

    def calls_for(self, stage: str) -> list[dict]:
        return [call for call in self.calls if call["stage"] == stage]

    def dispatched(self, stage: str, tile: str, index: int) -> int:
        """How many times one unit was handed to the substrate."""
        return sum(
            1 for call in self.calls if call["stage"] == stage and (tile, index) in call["units"]
        )


class CensusSimBackend(SimBackend):
    """:class:`SimBackend` that can also be *asked what it is billing*.

    The plain one predates the census contract, so every simulation built on it
    drives the driver's degraded, never-seen-a-census path. That is a real mode
    and worth keeping, but it is not the one production runs in, and the
    liveness defect the reap wiring closes only appears when a census answers:
    with one, a stranded wave is charged until the substrate stops reporting it,
    and something has to ask the substrate to stop.

    ``reap`` is honoured rather than recorded. A stub that only counted the ask
    would let a driver that never gets its width back still look correct here,
    which is the whole failure being pinned.
    """

    name = "adversarial-sim-census"

    def __init__(self, *args, answerable: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.answerable = answerable
        self.census_calls = 0
        self.reap_calls: list[str] = []

    def submission_identity(self, run_id: str, stage: str, wave: int) -> str:
        return self.wave_name(run_id, stage, wave)

    def census(self, run_id: str):
        self.census_calls += 1
        if not self.answerable:
            return None
        now = self.clock.now()
        by_identity: dict[str, int] = {}
        for call, wave in zip(self.calls, self.waves, strict=False):
            ends = wave.array_ends_at
            if call["at"] > now or (ends is not None and now >= ends):
                continue
            identity = self.submission_identity(run_id, call["stage"], call["wave"])
            by_identity[identity] = by_identity.get(identity, 0) + call["max_workers"]
        return WorkerCensus(
            as_of=now,
            total=sum(by_identity.values()),
            by_identity=dict(by_identity),
            identities=frozenset(by_identity),
        )

    def reap(self, run_id: str, identity: str) -> None:
        """Stop one array, the way ``delete_cluster`` does. Idempotent, and silent."""
        self.reap_calls.append(identity)
        now = self.clock.now()
        for call, wave in zip(self.calls, self.waves, strict=False):
            if self.submission_identity(run_id, call["stage"], call["wave"]) != identity:
                continue
            if wave.killed_at is None or wave.killed_at > now:
                wave.killed_at = now
                wave.array_ends_at = now


# --------------------------------------------------------------------------
# tile artifacts
# --------------------------------------------------------------------------


class TileWriter:
    """Writes exactly the keys ``_expected_keys`` names, and nothing else.

    One per tile, so a fleet run and a single-tile run agree key for key about
    what a finished shard looks like. The plan is published by offsets index 0,
    which is what the fused offsets stage does: shard 0 resolves.
    """

    def __init__(self, storage: MemoryStorage, plan, *, run_id: str, claims_export: bool = True):
        self.storage = storage
        self.plan = plan
        self.run_id = run_id
        self.root = shards.shard_root(run_id, plan.tile)
        self.claims_export = claims_export
        self.plan_published = False

    def publish_plan(self) -> None:
        if self.plan_published:
            return
        self.storage.write_text(shards.plan_key(self.root), json.dumps(self.plan.to_dict()))
        self.storage.write_text(shards.items_key(self.root), "[]")
        self.plan_published = True

    def write(self, stage: str, index: int) -> None:
        if stage == "offsets":
            if index == 0:
                self.publish_plan()
            if index >= self.plan.scene_shards:
                return
            for key in _expected_keys(self.plan, "offsets", self.root)[index]:
                self.storage.write_text(key, "{}")
            return
        if stage == "composite":
            if index >= len(self.plan.bands):
                return
            for key in _expected_keys(self.plan, "composite", self.root)[index]:
                self.storage.write_text(key, "band")
            if self.claims_export:
                self._maybe_claim_export()
            return
        if stage == "export":
            self.write_cogs()

    def _maybe_claim_export(self) -> None:
        expected = _expected_keys(self.plan, "composite", self.root)
        if any(key not in self.storage.objects for keys in expected.values() for key in keys):
            return
        claim = shards.export_claim_key(self.root)
        if self.storage.read_text(claim) is not None:
            return
        self.storage.write_text(claim, json.dumps({"tile": self.plan.tile}))
        self.write_cogs()

    def write_cogs(self) -> None:
        for product in PRODUCTS:
            self.storage.write_text(
                self.storage.cog_key(self.plan.window, self.plan.tile, product), "cog"
            )


# --------------------------------------------------------------------------
# a production-shaped plan
# --------------------------------------------------------------------------


def production_plan(
    tile: str,
    *,
    scenes: int = 300,
    scene_shards: int = 15,
    ref_shards: int = 15,
    band_shards: int = 8,
):
    """A plan with production *shapes*, so ``budgets`` prices it realistically.

    No array is ever built: the driver reads shapes, scene counts and the key
    grammar, nothing else. What this buys is that ``stage_budget`` returns the
    seconds a real tile would be budgeted -- an 18,000-square native grid and a
    9,000-square offset grid -- instead of the ~1.4 s a 1,024-square toy plan
    prices a composite unit at. Under the toy plan every deadline in a run is
    three orders of magnitude larger than the work it governs, so no deadline
    ever governs anything and the whole capacity category is unreachable.
    """
    total = max(scene_shards * 2, scenes)
    native = (18000, 18000)
    coarse = (9000, 9000)
    block_edge = 2250
    blocks = shards.block_spans(coarse, block_edge)
    return shards.TilePlan(
        tile=tile,
        window=WINDOW,
        scene_ids=[f"scene-{i}" for i in range(total)],
        scene_times=_scene_times(total),
        offset_factor=settings.destripe_offset_resolution_factor,
        coarse_shape=coarse,
        native_shape=native,
        block_edge=block_edge,
        blocks=blocks,
        block_has_land=[True] * len(blocks),
        scene_batches=[(i, i + 2) for i in range(0, total, 2)],
        bands=shards.band_edges(native[0], band_shards, settings.cog_blocksize),
        ref_shards=min(ref_shards, len(blocks)),
        scene_shards=scene_shards,
        band_shards=band_shards,
    )


def _scene_times(n: int) -> list[str]:
    """Stamps with real sub-second components, as the planner writes them."""
    base = 1_625_406_312
    return [
        datetime.fromtimestamp(base + 61 * 86400 * i, tz=UTC)
        .replace(microsecond=482_915 + 137 * i)
        .isoformat()
        .replace("+00:00", "")
        for i in range(n)
    ]


def stage_terms(plan) -> dict[str, tuple[float, float]]:
    """``{stage: (boot_s, unit_work_s)}`` from the driver's own budget model.

    A unit takes exactly as long as ``budgets`` says it will, with the safety
    factor left out -- that factor is slack for the driver's patience, not work
    the substrate performs. Deriving the durations rather than typing them is
    what makes a deadline test about the deadline instead of about a constant.
    """
    out: dict[str, tuple[float, float]] = {}
    for stage in ("offsets", "composite", "export"):
        budget = budgets.stage_budget(stage, plan)
        boot = float(dict(budget.phases).get("boot", 0.0))
        out[stage] = (boot, max(1.0, budget.work_s - boot))
    return out


# --------------------------------------------------------------------------
# stubs for the work that is not under test
# --------------------------------------------------------------------------


def stub_scientific_work(monkeypatch, writers: dict) -> None:
    """Replace the offset merge and the plan reader with cheap equivalents.

    Neither is under test in a state-machine simulation, and both are what made
    the reviewer's 200-tile brute-force run fail to terminate: ``merge_offsets``
    assembles a real record from real partials, and ``_read_plan`` rebuilds a
    plan through ``load_context``, which wants a serialized STAC item list per
    tile. The driver's reads are identical either way -- a merge that returns an
    object with a ``storage_key``, and a plan once ``plan.json`` exists.
    """

    def fake_merge(run_id, tile, *, storage=None):
        root = shards.shard_root(run_id, tile)
        key = f"{root}/merged-offsets.json"
        if storage is not None:
            storage.write_text(key, "{}")
        return SimpleNamespace(storage_key=key)

    def fake_read_plan(run_id, tile, root, storage):
        if storage.read_text(shards.plan_key(root)) is None:
            return None
        writer = writers.get(tile)
        return None if writer is None else writer.plan

    monkeypatch.setattr(shard_tasks, "merge_offsets", fake_merge)
    monkeypatch.setattr("landsat_lst.fleet_driver._read_plan", fake_read_plan)
    cache_expected_keys(monkeypatch)


def cache_expected_keys(monkeypatch) -> None:
    """Memoize ``_expected_keys``, which is a pure function of plan and root.

    Every tile recomputes its expected-key map on every poll, and each offsets
    map re-partitions the plan's scene batches once per shard index. At 700
    tiles that is hundreds of millions of list operations spent deriving an
    answer that cannot change, and it is the second reason a brute-force scale
    run does not terminate. Caching it changes no answer: the map is a function
    of ``(plan, stage, root)`` and a plan is immutable once published.
    """
    from landsat_lst import shard_driver as _sd

    original = _sd._expected_keys
    cache: dict[tuple[int, str, str], dict] = {}

    def cached(plan, stage: str, root: str):
        key = (id(plan), stage, root)
        if key not in cache:
            cache[key] = original(plan, stage, root)
        return cache[key]

    monkeypatch.setattr(_sd, "_expected_keys", cached)
    monkeypatch.setattr("landsat_lst.fleet_driver._expected_keys", cached)


# --------------------------------------------------------------------------
# building and running one simulation
# --------------------------------------------------------------------------


def tile_names(n: int) -> list[str]:
    """``n`` distinct tile names on the real 5-degree grid."""
    out: list[str] = []
    for lat in range(-55, 65, 5):
        for lon in range(-180, 180, 5):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            out.append(f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}")
            if len(out) == n:
                return out
    msg = f"only {len(out)} tile names available, wanted {n}"
    raise ValueError(msg)


def jobs_for(names) -> list[ProcessingJob]:
    return [ProcessingJob(tile=parse_tile_name(name), year=2021, end_year=2025) for name in names]


@dataclass
class Simulation:
    """One built-but-not-yet-run simulation, and everything to assert against."""

    run_id: str
    storage: MemoryStorage
    backend: SimBackend
    ledger: WorkerLedger
    clock: SimClock
    driver: object
    plans: list = field(default_factory=list)
    writers: dict = field(default_factory=dict)
    cap: int = 64

    def run(self):
        summary = self.driver.run()
        self.backend.tick()
        return summary


def build_simulation(
    monkeypatch,
    *,
    n_tiles: int,
    cap: int = 64,
    units: int = 15,
    scenes: int = 300,
    run_id: str = "adv",
    poll_s: float | None = None,
    polls_target: int | None = None,
    wave_window_s: float = 0.0,
    probe_waves: bool = False,
    backend_cls: type[SimBackend] | None = None,
    **backend_kw,
) -> Simulation:
    """A whole run, wired up: production-shaped plans, a physical substrate, a ledger.

    ``poll_s`` defaults to a value that keeps the poll count near
    ``polls_target``. The poll interval is documented as a request-rate knob
    rather than a latency one -- the barriers are minutes wide -- so coarsening
    it for a 700-tile simulation coarsens the *observation* of a run without
    changing which waves overlap. Typing 30 s at 700 tiles would make one
    simulation cost sixteen thousand full listings, which is how the reviewer's
    brute-force attempt failed to terminate.
    """
    from landsat_lst.fleet_driver import FleetDriver, _tracks

    names = tile_names(n_tiles)
    plans = [production_plan(name, scenes=scenes, scene_shards=units) for name in names]
    storage = MemoryStorage()
    clock = SimClock()
    ledger = WorkerLedger()
    writers = {plan.tile: TileWriter(storage, plan, run_id=run_id) for plan in plans}
    stub_scientific_work(monkeypatch, writers)

    terms = stage_terms(plans[0])
    # Resolved here rather than bound as a default, which is the rule this
    # project already keeps for the billing source: a default evaluated at
    # definition time cannot be substituted, and a caller that swaps the class
    # on the module gets the original one without being told.
    cls = backend_cls if backend_cls is not None else SimBackend
    backend = cls(storage, writers, clock=clock, ledger=ledger, terms=terms, **backend_kw)
    storage.on_list = backend.tick

    if poll_s is None:
        target = polls_target if polls_target is not None else max(80, 20_000 // max(1, n_tiles))
        makespan = 0.0
        for stage, per_tile in (("offsets", units), ("composite", len(plans[0].bands))):
            boot, work = terms[stage]
            depth = math.ceil((n_tiles * per_tile) / max(1, cap))
            makespan += boot + (depth + 1) * work
        poll_s = max(30.0, makespan / target)
    monkeypatch.setattr(settings, "fleet_poll_s", float(poll_s))

    jobs = jobs_for(names)
    driver = FleetDriver(
        run_id=run_id,
        tracks=_tracks(jobs, run_id=run_id, storage=storage, units=units, clock=clock),
        storage=storage,
        backend=backend,
        clock=clock,
        units=units,
        max_vms=cap,
        wave_window_s=wave_window_s,
        probe_waves=probe_waves,
    )
    return Simulation(
        run_id=run_id,
        storage=storage,
        backend=backend,
        ledger=ledger,
        clock=clock,
        driver=driver,
        plans=plans,
        writers=writers,
        cap=cap,
    )


# --------------------------------------------------------------------------
# the termination assertion
# --------------------------------------------------------------------------


def settlement(names, summary) -> dict[str, list[str]]:
    """How each input tile ended, as a list so ``twice`` is as visible as ``never``."""
    out: dict[str, list[str]] = {name: [] for name in names}
    for tile in summary.completed:
        out.setdefault(tile, []).append("completed")
    for tile in summary.failed:
        out.setdefault(tile, []).append("failed")
    return out


def assert_every_tile_settled_once(names, summary) -> None:
    """Every input tile ends exactly once, in completed or in failed.

    Not zero times: ``drive_fleet`` returning normally with ten tiles in neither
    list is the observed limbo of a wave preempted mid-flight, and it is
    indistinguishable from success to a caller that only reads
    ``summary.failed``. Not twice either, which no shipped test would catch.
    """
    where = settlement(names, summary)
    unreported = sorted(tile for tile, ends in where.items() if not ends)
    doubled = sorted(tile for tile, ends in where.items() if len(ends) > 1)
    assert not unreported, (
        f"{len(unreported)} tile(s) ended in neither completed nor failed "
        f"while the run returned normally: {unreported[:8]}"
    )
    assert not doubled, f"{len(doubled)} tile(s) reported twice: {doubled[:8]}"
