"""Many tiles through one work array per stage, sequenced by S3 barriers.

:mod:`landsat_lst.shard_driver` cuts one tile across many VMs and pays a
fleet's boot to do it: ``budgets.VM_BOOT_S`` is 300 s, paid concurrently by
every VM in the array, once per stage, once per tile. On S30W065 an
offsets-side shard computed for about six minutes while its stage held a fleet
for about thirty. Repeat that 700 times on each of two stages and the
provisioning idle, not the work, is what the build is buying.

So this module drives **many tiles at once**. The unit of submission is a
*wave*: one work array for one stage, carrying units from every tile that was
ready when the wave flushed. A wave with more units than workers is where the
saving comes from -- the substrate queues the surplus onto workers that have
already booted, so a boot is paid once per VM per wave rather than once per VM
per stage per tile.

**Nothing here knows what a Coiled cluster is.** Waves are started through
:class:`~landsat_lst.fleet_backend.FleetBackend`, and what comes back is an
opaque :class:`~landsat_lst.fleet_backend.WaveHandle`. That is not decoration:
the boot amortization is a property of the submission substrate, so whether
AWS Batch or ECS does it cheaper is a question worth being able to ask without
touching this state machine. What the machine *requires* of any substrate is
written down and machine-checked --
:data:`~landsat_lst.fleet_backend.BACKEND_CONTRACT`, enforced by
:func:`~landsat_lst.fleet_backend.check_contract` at construction.

Nothing underneath changes. The work-unit bodies are the ones
:mod:`landsat_lst.shard_tasks` already runs, the artifact keys are the ones
:mod:`landsat_lst.shards` already owns, the offset merge still happens in this
process and writes the ordinary ADR-012 record, and completion is still bytes
in the bucket. What changes is that the driver is a single non-blocking loop
over per-tile tracks instead of a blocking sequence for one tile, so a slow
tile costs its own wall clock and nobody else's. See ADR-018.

Three properties are worth stating because they are what the tests assert:

- **Submissions do not scale with tiles.** A wave flushes when it fills the VM
  headroom, when ``fleet_wave_window_s`` has passed, or when no tile that has
  not yet demanded could still join. None of those mentions the tile count.
- **Barriers advance per tile.** A track that is watching, exhausted, or failed
  is stepped and returns nothing. A tile that fails is recorded and the loop
  continues; only a terminal control-plane failure (quota, credits, auth) stops
  the run, because that one is not about any tile.
- **The cap is on VMs, never on queued units.** A wave is submitted at
  ``max_workers = min(units, headroom)``, so one tile demanding more units than
  the whole cap still runs -- queued, not split. Splitting a tile's demand
  across two waves would write a submission record for a partial index set, and
  the remainder would cost the tile a barrier round it never used.

**The cap is measured, and it bounds concurrency rather than spend.** Held
width used to be inferred from the bucket, and a listing of work products is not
a function of whether a VM exists -- which is how the same expression
over-charged a wave that was gone and under-charged one that was merely slow.
So the driver asks the substrate: ``backend.census(run_id)`` reports every
worker being billed for this run, found from the run id alone, and

    H(t) = max( intent_charge(t), census(t).total )

where ``intent_charge`` counts every submission *attempt* at its full requested
width from the instant the call is issued. Bytes in the bucket are still what
decides whether *work* is done; they no longer decide whether a machine is up.

``settings.fleet_max_vms`` is a **concurrency** ceiling and not a budget. Spend
is the integral of concurrency over time, the substrate reports when a worker
started and not when it stopped, and nothing server-side remembers the cap at
all. The census enforces the concurrency bound and yields only a *lower* bound
on the bill. See ADR-018.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import structlog

from landsat_lst import budgets, shard_tasks, shards
from landsat_lst.config import settings
from landsat_lst.fleet_backend import (
    CoiledFleetBackend,
    WaveRequest,
    check_contract,
)
from landsat_lst.shard_driver import (
    Clock,
    ShardStageFailed,
    _bootstrap_deadline_s,
    _expected_keys,
    _missing,
    _overlap_ready,
    _read_plan,
    _submission_records,
)
from landsat_lst.storage import PRODUCTS, get_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from landsat_lst.fleet_backend import FleetBackend, WaveHandle, WorkerCensus
    from landsat_lst.models import ProcessingJob
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: The order a tile's stages settle in. ``merge`` runs in this process and
#: submits nothing, which is why it is here rather than in ``shards.STAGES``.
TILE_STAGES = ("offsets", "merge", "composite", "export")

#: Keys one S3 list page returns. ``list_objects_v2`` caps a page at 1,000
#: keys, so a listing's cost in *requests* is the number of pages it takes.
LIST_PAGE_KEYS = 1000

#: Stages a wave can carry. ``merge`` is local; everything else is a fleet.
WAVE_STAGES = ("offsets", "composite", "export")


def _stage_terms(stage: str, plan: shards.TilePlan) -> tuple[float, float, float]:
    """``(one_round_deadline, boot_s, unit_work_s)`` for one stage of one tile.

    The three numbers a demand carries. The first is the single-round horizon
    the per-tile driver has always used; the other two let the wave this demand
    lands in scale its own deadline by queue depth.
    """
    budget = budgets.stage_budget(stage, plan)
    boot_s, unit_work_s = budgets.split_boot(budget)
    return budget.deadline_s, boot_s, unit_work_s


def _stage_prefix(root: str, stage: str) -> str:
    """Where one stage's shard artifacts live, in one place.

    Both the barrier and the capacity accounting list this, and they have to
    agree: a second spelling would let a wave read as finished against one
    prefix while the tile still waits on another.
    """
    return f"{root}/offsets/scene/" if stage == "offsets" else f"{root}/{stage}/"


def _bootstrap_terms() -> tuple[float, float, float]:
    """The same three, before any plan exists to budget against.

    Shard 0 of the fused offsets stage writes the plan, so the first wave is
    budgeted from named fixed costs: a boot, a resolve, and the wait its peers
    spend on the plan.
    """
    unit_work_s = budgets.RESOLVE_S + settings.shard_plan_wait_s
    return _bootstrap_deadline_s(), budgets.VM_BOOT_S, float(unit_work_s)


class _Listing:
    """One prefix's keys, sorted once so that a per-tile slice is a bisect.

    Sorting matters as much as sharing the request. A cached listing that every
    tile filters end to end is still ``O(tiles x keys)`` of CPU per poll, and at
    700 tiles that is tens of millions of prefix comparisons inside a 30 s
    budget. Keys come back sorted from S3 and are sorted here for any backend
    that does not promise it, so one tile's slice costs a binary search plus the
    keys it actually matched.
    """

    __slots__ = ("keys", "stamps")

    def __init__(self, stamps: dict) -> None:
        self.stamps = stamps
        self.keys = sorted(stamps)

    def slice(self, prefix: str) -> dict:
        out: dict = {}
        for key in self.keys[bisect_left(self.keys, prefix) :]:
            if not key.startswith(prefix):
                break
            out[key] = self.stamps[key]
        return out


class PollIndex:
    """One listing per shared prefix per poll, served to every tile.

    Every tile's barrier asks the same question -- "which of my keys are
    there" -- against prefixes that all sit under one run prefix. Asking per
    tile makes the driver's request rate linear in the tile count: at 700 tiles
    that is roughly 1,400 serial listings a cycle, which measured at about 70 s
    against a 30 s poll. The loop stops keeping up with itself somewhere around
    300 tiles, and the failure is invisible -- the run just polls slower and
    slower.

    So the listing is done once per cycle over ``_shards/{run_id}/`` and every
    tile is served from it. What that buys, stated exactly rather than
    generously: the number of LIST *operations* the driver issues per poll stops
    depending on how many tiles it is driving and starts depending on how many
    keys the run has published, because a paginated listing returns 1,000 keys a
    request. At 700 tiles and roughly 50 keys a tile that is about 35 requests a
    poll against the 1,400 the per-tile form issued, and the exponent on the
    tile count is what changed, not the presence of one. Claiming a constant
    here would be wrong, and it is the kind of wrong that only shows up at the
    scale nobody tests at.

    Both numbers are published, because they are different numbers:
    :attr:`listings` is calls and :attr:`requests` is pages, and only the
    second is what S3 bills. A call over a prefix holding 24,000 keys is 24
    requests, so a counter of calls reads flat while the charge climbs with the
    keys a run has published.

    Bodies are cached against the modification time the listing already
    returns, which is exact rather than a guess: a key whose timestamp has not
    moved has not been rewritten, so its body can be reused. That takes the
    per-cycle GETs for submission records down to only the records that
    actually changed.

    Unknown prefixes fall through to the backend, so this is a cache and never
    a different answer.
    """

    def __init__(self, storage: StorageBackend, run_id: str) -> None:
        self._storage = storage
        self._prefixes: list[str] = [f"{shards.SHARD_PREFIX}/{run_id}/"]
        self._listing: dict[str, _Listing] = {}
        self._bodies: dict[str, tuple[object, str | None]] = {}
        #: Listing *calls* issued against the backend, for the test that pins
        #: this staying flat in the tile count. It is not the bill.
        self.listings = 0
        #: Listing *requests*, which is the bill. One call over a prefix
        #: holding more keys than :data:`LIST_PAGE_KEYS` costs several
        #: requests, so a counter of calls reads flat while the charge grows
        #: with the keys the run has published. Reporting calls as cost was an
        #: accounting defect rather than a wrong claim: what sharing buys is a
        #: change in the exponent on the tile count, and only a request count
        #: can show that honestly.
        self.requests = 0

    def watch(self, prefix: str) -> None:
        """Add a prefix to the set refreshed each cycle."""
        if prefix not in self._prefixes:
            self._prefixes.append(prefix)

    def _charge(self, keys: int) -> None:
        """Bill one listing call at the number of pages it took."""
        self.listings += 1
        self.requests += max(1, -(-keys // LIST_PAGE_KEYS))

    def refresh(self) -> None:
        for prefix in list(self._prefixes):
            try:
                listed = dict(self._storage.list_prefix(prefix))
            except Exception as e:
                log.warning("fleet_index_listing_failed", prefix=prefix, error=str(e))
                self._charge(0)
                self._listing[prefix] = _Listing({})
            else:
                self._charge(len(listed))
                self._listing[prefix] = _Listing(listed)

    def _covering(self, prefix: str) -> _Listing | None:
        for cached, listing in self._listing.items():
            if prefix.startswith(cached):
                return listing
        return None

    def list_prefix(self, prefix: str) -> dict:
        listing = self._covering(prefix)
        if listing is None:
            listed = dict(self._storage.list_prefix(prefix))
            self._charge(len(listed))
            return listed
        return listing.slice(prefix)

    def read_text(self, key: str) -> str | None:
        listing = self._covering(key)
        stamp = listing.stamps.get(key) if listing is not None else None
        if stamp is not None:
            cached = self._bodies.get(key)
            if cached is not None and cached[0] == stamp:
                return cached[1]
        body = self._storage.read_text(key)
        if stamp is not None:
            self._bodies[key] = (stamp, body)
        return body


class _IndexedStorage:
    """A storage view that answers listings and reads from a :class:`PollIndex`.

    Everything else -- writes, ``cog_key``, downloads -- passes straight
    through, so a tile cannot tell it is reading a cache except by counting
    requests, which is the point.
    """

    def __init__(self, storage: StorageBackend, index: PollIndex) -> None:
        self._storage = storage
        self._index = index

    def __getattr__(self, name: str):
        return getattr(self._storage, name)

    def list_prefix(self, prefix: str):
        return self._index.list_prefix(prefix)

    def read_text(self, key: str) -> str | None:
        return self._index.read_text(key)

    def cog_exists(self, window: str, tile: str) -> bool:
        """Both assets, answered from the shared listing of the COG prefix.

        Registered on first use rather than up front, because the window is not
        known until a plan has been read.
        """
        keys = [self._storage.cog_key(window, tile, product) for product in PRODUCTS]
        # ``lst-p95-{window}/{tile}/{product}...`` -- two segments up is the
        # prefix every tile in this window shares.
        collection = keys[0].rsplit("/", 2)[0] + "/"
        self._index.watch(collection)
        listed = self._index.list_prefix(collection)
        return all(key in listed for key in keys)


def _opt_float(raw: object) -> float | None:
    """A float, or ``None`` for a record written before the field existed."""
    if not isinstance(raw, (int, float, str)):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _unit_tokens(raw: object) -> tuple[tuple[str, int], ...]:
    """``["N40W075:3", ...]`` back into unit pairs, dropping what will not parse.

    Best-effort, like everything else read back for bookkeeping: a token that
    cannot be parsed yields one fewer piece of evidence about what a wave is
    holding, and the wave then holds its requested width. Refusing the whole
    record would be the worse trade.
    """
    if not isinstance(raw, list):
        return ()
    out: list[tuple[str, int]] = []
    for token in raw:
        try:
            out.append(shards.parse_fleet_unit(str(token)))
        except ValueError:
            log.warning("fleet_wave_unit_malformed", token=token)
    return tuple(out)


class FleetAborted(RuntimeError):
    """A terminal control-plane failure stopped the whole run.

    Distinct from a tile failing. A quota, credit, or auth refusal is not about
    the tile whose wave happened to hit it -- retrying it under another tile's
    name costs the same silence -- so it ends the run and names the reason,
    exactly as :class:`~landsat_lst.shard_driver.ShardSubmissionFailed` ends a
    single-tile one.
    """


def fleet_run_id(prefix: str = "fleet") -> str:
    """A run token for a multi-tile run.

    Deliberately carries no tile, unlike
    :func:`~landsat_lst.shard_driver.shard_run_id`. That is safe because
    :func:`landsat_lst.shards.shard_root` appends the tile itself, so every
    tile in one fleet still gets its own prefix, its own plan, and its own
    submission records; only the roster and the wave records are shared.
    """
    return f"{prefix}-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"


# --------------------------------------------------------------------------
# The roster
# --------------------------------------------------------------------------


def write_manifest(
    storage: StorageBackend, run_id: str, jobs: Sequence[ProcessingJob], *, units: int
) -> str:
    """Publish which tiles this run drives, before anything is submitted.

    The one part of a fleet run that a listing cannot recover. A listing shows
    the tiles that got far enough to write something, which is precisely not
    the set a resume has to drive: a tile whose first wave was preempted before
    it booted has published nothing and would silently drop out of the run.
    """
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "units": int(units),
        "tiles": [
            {
                "tile": job.tile.name,
                "year": job.year,
                "end_year": job.end_year,
                "max_scenes": job.max_scenes,
            }
            for job in jobs
        ],
    }
    key = shards.fleet_manifest_key(run_id)
    storage.write_text(key, json.dumps(payload, indent=2))
    log.info("fleet_manifest_written", run_id=run_id, tiles=len(payload["tiles"]), key=key)
    return key


def read_manifest(storage: StorageBackend, run_id: str) -> tuple[list[ProcessingJob], int]:
    """The roster back, as jobs.

    Raises:
        FileNotFoundError: If this run never published one, which means there
            is nothing to resume rather than an empty run to drive.
    """
    from landsat_lst.models import ProcessingJob  # noqa: PLC0415
    from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415

    raw = storage.read_text(shards.fleet_manifest_key(run_id))
    if raw is None:
        msg = (
            f"run {run_id!r} published no fleet manifest at "
            f"{shards.fleet_manifest_key(run_id)}; there is nothing to resume"
        )
        raise FileNotFoundError(msg)
    payload = json.loads(raw)
    jobs = [
        ProcessingJob(
            tile=parse_tile_name(entry["tile"]),
            year=entry["year"],
            end_year=entry.get("end_year"),
            max_scenes=entry.get("max_scenes"),
        )
        for entry in payload["tiles"]
    ]
    return jobs, int(payload.get("units") or shards.offsets_fleet_units())


def job_for_token(storage: StorageBackend, run_id: str, tile: str) -> ProcessingJob:
    """One tile's job, read on the VM from the roster.

    A consolidated array carries one command for every tile, so the window and
    the scene cap cannot travel on it. They are read here instead. A *missing*
    one would silently revert to a default and resolve a different scene set,
    which is why this raises rather than falling back.

    Raises:
        FileNotFoundError: If the run has no manifest.
        KeyError: If the manifest does not list this tile.
    """
    jobs, _ = read_manifest(storage, run_id)
    for job in jobs:
        if job.tile.name == tile:
            return job
    msg = f"tile {tile!r} is not in the fleet manifest for run {run_id!r}"
    raise KeyError(msg)


def run_unit(
    run_id: str,
    stage: str,
    token: str,
    *,
    units: int | None = None,
    storage: StorageBackend | None = None,
) -> object:
    """One unit of a consolidated wave, on the VM, timed.

    The body is :func:`landsat_lst.shard_tasks.run_shard` and is called with the
    same arguments the per-tile command passes; nothing about the work changes.
    What is added around it is a start and an end, written to
    :func:`landsat_lst.shards.unit_timing_key`.

    That record is what turns the cost model's idle term from an assumption into
    a measurement. Per-wave stamps say how long a wave's VMs were billed and
    cannot say how much of that was work: a worker waiting for its next unit and
    a worker running one look identical from the bucket. Subtracting real
    durations is the only way round it.

    Written for a failed unit too, and best-effort throughout. Losing a duration
    costs a term in a cost model; failing a tile over a bookkeeping write would
    cost a composite.

    Raises:
        ValueError: If the token is not exactly one tile and one index.
    """
    from landsat_lst.shard_tasks import run_shard  # noqa: PLC0415

    tile, index = shards.parse_fleet_unit(token)
    storage = storage or get_storage()
    # Only the stages whose shard 0 resolves need a job, and reading one for the
    # others would make every task depend on the manifest for nothing.
    job = job_for_token(storage, run_id, tile) if stage in ("resolve", "offsets") else None

    started = datetime.now(tz=UTC).timestamp()
    try:
        result = run_shard(stage, run_id, tile, index, job=job, units=units)
    except BaseException:
        _record_unit_timing(storage, run_id, stage, tile, index=index, started=started, failed=True)
        raise
    _record_unit_timing(storage, run_id, stage, tile, index=index, started=started, failed=False)
    return result


def _record_unit_timing(
    storage: StorageBackend,
    run_id: str,
    stage: str,
    tile: str,
    *,
    index: int,
    started: float,
    failed: bool,
) -> None:
    """Publish one unit's interval. Never raises."""
    ended = datetime.now(tz=UTC).timestamp()
    try:
        storage.write_text(
            shards.unit_timing_key(run_id, stage, tile, index),
            json.dumps(
                {
                    "run_id": run_id,
                    "stage": stage,
                    "tile": tile,
                    "index": index,
                    "started_at": started,
                    "ended_at": ended,
                    "duration_s": round(ended - started, 3),
                    "failed": failed,
                }
            ),
        )
    except Exception as e:
        log.warning("fleet_unit_timing_failed", stage=stage, tile=tile, index=index, error=str(e))


# --------------------------------------------------------------------------
# What one tile wants next
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Demand:
    """One tile asking for one stage's missing units to be started.

    Carries the barrier round it belongs to and the deadline that round should
    get, because both are per tile even when the submission is not: the round
    budget is counted per tile across drivers, and two tiles in one wave can
    legitimately be on different rounds.
    """

    tile: str
    stage: str
    indexes: tuple[int, ...]
    submission_round: int
    #: One *round's* horizon, used only as the fallback liveness window for a
    #: record that predates wave deadlines. The wave this demand lands in gets
    #: a deadline scaled by its queue depth -- see :func:`landsat_lst.budgets.wave_deadline_s`.
    deadline_s: float
    #: Cold start, paid once per worker rather than once per unit.
    boot_s: float = 0.0
    #: One unit's work with the boot taken out. This is the term that repeats
    #: when a wave is deeper than it is wide, and the term the old deadline
    #: failed to multiply.
    unit_work_s: float = 0.0


@dataclass
class TileOutcome:
    """How one tile ended, and what it cost in submissions."""

    tile: str
    completed: bool = False
    failed: bool = False
    reason: str = ""
    stage: str = "offsets"
    rounds: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "tile": self.tile,
            "completed": self.completed,
            "failed": self.failed,
            "reason": self.reason,
            "stage": self.stage,
            "rounds": dict(self.rounds),
        }


@dataclass
class TileTrack:
    """One tile's barrier sequence, stepped rather than blocked.

    The single-tile :class:`~landsat_lst.shard_driver.StageMachine` owns its
    thread: it polls, sleeps, and returns when the stage settles. That is
    exactly what a fleet cannot have, because one tile's wait would be every
    tile's wait. So this is the same decision table with the waiting removed --
    :meth:`step` is called once per poll and returns what the tile wants
    started, or nothing.

    The decision rules are unchanged from ADR-016, and deliberately so:

    - Artifacts decide. A stage is done when its keys are listed.
    - A live submission record means watch, do not submit. The record is still
      per tile (:func:`landsat_lst.shards.stage_submission_key`), so adoption
      and the across-drivers round budget work exactly as they did.
    - Every round gets a deadline computed when that round opens.
    - A cluster reported dead can only end a barrier *sooner*, and only after
      the bucket has been re-checked -- which :meth:`step` does first, always.
    """

    tile: str
    run_id: str
    root: str
    storage: StorageBackend
    units: int
    clock: Clock
    job: ProcessingJob | None = None
    dead_handles: set = field(default_factory=set)
    #: Submission identities an authoritative census stopped reporting, shared
    #: by the driver for the same reason ``dead_handles`` is. It is the wider of
    #: the two and the only one that covers a submission this driver holds no
    #: handle for -- which is precisely the record a resumed driver inherits.
    #: A **barrier** input, never a capacity one: it lets a tile stop waiting
    #: out a deadline on machines that are gone.
    dead_identities: set = field(default_factory=set)
    #: Per stage, the unit indexes a wave the driver still counts as live is
    #: carrying for this tile. Refreshed by the driver each poll, exactly as
    #: ``dead_handles`` is, because a track cannot see a wave.
    held_units: dict[str, set[int]] = field(default_factory=dict)

    stage: str = "offsets"
    plan: shards.TilePlan | None = None
    #: Stages this tile can no longer contribute units to, so the wave batcher
    #: can tell a tile that might still join a stage from one that never will.
    done_stages: set[str] = field(default_factory=set)
    #: Per stage, the unit indexes whose artifacts are *still absent* as of the
    #: last poll. This is how a wave's held capacity is counted: a unit whose
    #: artifact is listed is a worker that has moved on, and holding its width
    #: against the cap for the life of the wave would deadlock a run whose
    #: first wave is wider than the cap. Empty once the stage settles.
    outstanding: dict[str, set[int]] = field(default_factory=dict)
    #: Set by each ``_step_*`` to say whether it moved the tile on. A step that
    #: returns nothing without advancing is a step that is waiting.
    _advanced: bool = False
    #: When the tile entered the export wait, for the claim fallback.
    _export_since: float | None = None
    #: Whether the overlapped composite start has already been demanded.
    _overlap_demanded: bool = False
    #: When this tile was first seen without a plan, for the bootstrap deadline.
    _bootstrap_since: float | None = None

    def __post_init__(self) -> None:
        self.outcome = TileOutcome(tile=self.tile)

    # -- state ------------------------------------------------------------

    @property
    def terminal(self) -> bool:
        return self.outcome.completed or self.outcome.failed

    def has_settled(self, stage: str) -> bool:
        """Whether this tile can no longer contribute units to ``stage``."""
        return self.terminal or stage in self.done_stages

    def _fail(self, reason: str) -> None:
        """Record that this tile is over, and say why.

        Capacity is deliberately untouched. A tile that has given up is a tile
        the driver has stopped waiting for, which is not the same fact as a
        tile whose VMs have stopped: a barrier round expires because a record
        aged out, and the workers that record describes may still be billing.
        Releasing a wave's width on the strength of a tile's own despair is the
        clock defect wearing a different name.

        Nothing here *can* touch capacity any more, which is the point. The
        driver counts what a wave holds from :attr:`Wave.outstanding_units`,
        refreshed from the bucket once a poll, so a tile failing between that
        refresh and the flush in the same poll cannot move the number. It used
        to: ``outstanding`` lived on the track, this method could clear it, and
        the window between the two was one poll wide and invisible to every
        test. The width comes back when the artifacts land or the backend
        confirms the submission is gone, exactly as it does for a healthy tile.
        """
        self.outcome.failed = True
        self.outcome.reason = reason
        self.outcome.stage = self.stage
        log.error(
            "fleet_tile_failed",
            run_id=self.run_id,
            tile=self.tile,
            stage=self.stage,
            reason=reason,
        )

    def _advance(self, stage: str) -> None:
        # A settled stage holds no workers: every artifact it was waiting on is
        # listed, so every unit of it has finished.
        self.outstanding.pop(self.stage, None)
        self.done_stages.add(self.stage)
        self.stage = stage
        self.outcome.stage = stage
        self._advanced = True

    # -- evidence, for whatever a live wave is still holding ---------------

    def refresh_outstanding(self, stage: str, indexes: Sequence[int]) -> set[int]:
        """Recompute, from the bucket, which of these units have not published.

        Called by the driver for every ``(tile, stage)`` a live wave references,
        and called for a **terminal** tile as well as a healthy one. That is the
        whole point: a completed tile's stage is settled and its units are on
        the ground, but a *failed* tile's units may still be running, and the
        only honest reading of "is a worker still holding a slot" is whether the
        thing it was asked to write is there.

        Conservative where it cannot see: with no plan yet there is no way to
        name the artifacts, so every index in the wave counts as outstanding.
        Over-counting delays a submission; under-counting doubles a bill.

        Returns:
            The indexes still absent. Returned as well as stored because the
            caller keeps the driver's capacity ledger on the *wave*, where a
            track cannot reach it.
        """
        if self.plan is None:
            self.outstanding[stage] = set(indexes)
        elif stage == "export":
            done = self.storage.cog_exists(self.plan.window, self.tile)
            self.outstanding[stage] = set() if done else set(indexes)
        else:
            expected = _expected_keys(self.plan, stage, self.root)
            present = set(self.storage.list_prefix(_stage_prefix(self.root, stage)))
            # An index past the plan's clamped shard count has nothing to
            # produce: the worker reads the plan, finds no slice of its own,
            # and exits.
            self.outstanding[stage] = {
                index
                for index in indexes
                if any(key not in present for key in expected.get(index, ()))
            }
        return set(self.outstanding[stage])

    # -- the barrier decision, without the wait ---------------------------

    def step(self) -> list[Demand]:
        """One poll's worth of progress. Returns what this tile wants started.

        Never blocks and never sleeps: the fleet's loop owns the clock. A tile
        that is watching a live submission returns an empty list, which is what
        makes one slow tile cost only its own wall clock.
        """
        demands: list[Demand] = []
        # Bounded so a tile that settles several stages in one poll still
        # cannot spin: at most one transition per stage.
        for _ in range(len(TILE_STAGES) + 1):
            if self.terminal:
                return demands
            self._advanced = False
            if self.stage == "offsets":
                demand = self._step_offsets(demands)
            elif self.stage == "merge":
                demand = self._step_merge()
            elif self.stage == "composite":
                demand = self._step_composite()
            else:
                demand = self._step_export()
            if demand is not None:
                demands.append(demand)
                return demands
            if not self._advanced:
                return demands
        return demands

    def _step_offsets(self, pending: list[Demand]) -> Demand | None:
        self.plan = self.plan or _read_plan(self.run_id, self.tile, self.root, self.storage)

        if self.plan is None:
            # No plan yet: the fused fleet's shard 0 writes it, so the demand is
            # the whole fleet width and the deadline is the bootstrap one. The
            # single-tile driver blocks in ``_wait_for_plan`` here; a fleet
            # cannot, so the same allowance is spent watching the record.
            if self._bootstrap_since is None:
                self._bootstrap_since = self.clock.now()
            self.outstanding["offsets"] = set(range(self.units))
            return self._demand("offsets", tuple(range(self.units)), *_bootstrap_terms())

        missing = _missing(
            self.storage,
            _stage_prefix(self.root, "offsets"),
            _expected_keys(self.plan, "offsets", self.root),
        )
        self.outstanding["offsets"] = set(missing)
        if not missing:
            self._advance("merge")
            return None

        # The overlap: the composite fleet starts from inside the offsets
        # barrier as soon as phase B is demonstrably producing, so its VMs boot
        # on time this stage is going to spend anyway. Best-effort -- an overlap
        # that never fires makes a tile slower, never wrong.
        if not self._overlap_demanded and _overlap_ready(self.storage, self.root, self.plan):
            overlap = self._demand_composite()
            if overlap is not None:
                self._overlap_demanded = True
                pending.append(overlap)

        return self._demand("offsets", tuple(missing), *_stage_terms("offsets", self.plan))

    def _step_merge(self) -> Demand | None:
        """Assemble the partials into the canonical ADR-012 record, here.

        Unchanged boundary: kilobytes of JSON in, ~600 floats out, written at the
        ordinary offset key that every band reads back. A VM would spend longer
        booting than working.
        """
        try:
            key = shard_tasks.merge_offsets(self.run_id, self.tile, storage=self.storage)
        except Exception as e:
            self._fail(f"merge_offsets failed: {type(e).__name__}: {e}")
            return None
        log.info(
            "fleet_offsets_ready",
            run_id=self.run_id,
            tile=self.tile,
            key=getattr(key, "storage_key", None),
        )
        self._advance("composite")
        return None

    def _step_composite(self) -> Demand | None:
        missing = self._composite_missing()
        if not missing:
            self._advance("export")
            self._export_since = self.clock.now()
            return None
        assert self.plan is not None
        return self._demand("composite", tuple(missing), *_stage_terms("composite", self.plan))

    def _composite_missing(self) -> list[int]:
        assert self.plan is not None
        missing = _missing(
            self.storage,
            _stage_prefix(self.root, "composite"),
            _expected_keys(self.plan, "composite", self.root),
        )
        self.outstanding["composite"] = set(missing)
        return missing

    def _demand_composite(self) -> Demand | None:
        """The composite demand, for the overlapped start inside the offsets barrier."""
        missing = self._composite_missing()
        if not missing:
            return None
        assert self.plan is not None
        return self._demand("composite", tuple(missing), *_stage_terms("composite", self.plan))

    def _step_export(self) -> Demand | None:
        """Wait for a composite worker's claim; submit only if none runs it.

        The last composite worker to write a band claims the export and runs it
        there, which saves a whole VM boot. So the first move is to wait. The
        fallback is for a claim that was written and never executed, which is a
        VM preempted between the two.
        """
        assert self.plan is not None
        if self.storage.cog_exists(self.plan.window, self.tile):
            self.outcome.completed = True
            self.outstanding.pop("export", None)
            self.done_stages.add("export")
            log.info("fleet_tile_done", run_id=self.run_id, tile=self.tile)
            return None
        if self._export_since is None:
            self._export_since = self.clock.now()
        if self.clock.now() - self._export_since < settings.shard_export_claim_fallback_s:
            return None
        self.outstanding["export"] = {0}
        return self._demand("export", (0,), *_stage_terms("export", self.plan))

    # -- the round budget, counted across drivers -------------------------

    def _demand(
        self,
        stage: str,
        indexes: tuple[int, ...],
        deadline_s: float,
        boot_s: float,
        unit_work_s: float,
    ) -> Demand | None:
        """A demand, unless somebody's submission for this stage is still live.

        This is where the round budget is spent, and it is read from the bucket
        rather than from memory so a resumed driver cannot mint itself a fresh
        allowance.

        A unit a live wave is still carrying is not re-demanded, and this is a
        separate question from the round budget rather than a duplicate of it.
        The record ages out on a horizon derived when the wave was submitted;
        the wave is retired on evidence. When the first runs out before the
        second, the tile asks for units that are running right now, and the
        driver dispatches them a second time beside the first. The composite
        wave demanded from inside the offsets barrier is where that bites --
        its record is written long before the tile reaches the stage. Units are
        idempotent at their artifact keys, so this was waste and a spent
        barrier round rather than corruption, but a run that pays for the same
        band twice while a cap is holding other tiles out is paying twice for
        nothing. A *stranded* wave is excluded by the driver, because a wave
        past its own budget is exactly the case a barrier round exists for.
        """
        indexes = tuple(index for index in indexes if index not in self.held_units.get(stage, ()))
        if not indexes:
            return None
        records = _submission_records(self.storage, self.root, stage)
        latest = records[-1] if records else None
        if latest is not None and self._is_live(latest, deadline_s):
            return None
        next_round = int(latest["round"]) + 1 if latest else 1
        if next_round > settings.shard_barrier_rounds:
            self._fail(str(ShardStageFailed(stage, self._missing_keys(stage, indexes))))
            return None
        self.outcome.rounds[stage] = next_round
        return Demand(
            tile=self.tile,
            stage=stage,
            indexes=indexes,
            submission_round=next_round,
            deadline_s=deadline_s,
            boot_s=boot_s,
            unit_work_s=unit_work_s,
        )

    def _is_live(self, record: dict, deadline_s: float) -> bool:
        """Whether that round is still allowed to be booting.

        A cluster the probe reported dead is not: the fleet whose last task
        uploaded and then stopped has already been re-checked against the bucket
        by the caller, so what is left here is a fleet that will never produce
        anything, and waiting out its deadline buys nothing.

        A census that no longer reports the submission's identity says the same
        thing and says it about records this driver holds no handle for. That
        is the case the handle set cannot reach: an adopted record, or one whose
        submission answer was lost, has ``cluster_id`` ``None`` and would
        otherwise be waited out in full against a substrate that is running
        nothing.
        """
        if record.get("cluster_id") is not None and record["cluster_id"] in self.dead_handles:
            return False
        if record.get("cluster_name") and record["cluster_name"] in self.dead_identities:
            return False
        # The wave that wrote this record knows its own queue depth; the caller
        # only knows one round's worth. Preferring the record is what stops a
        # deep wave's tiles from re-demanding at a fraction of its runtime and
        # burning both barrier rounds. Records without the field are the
        # single-tile driver's, where one round is the truth.
        horizon = float(record.get("deadline_s") or deadline_s)
        return self.clock.now() < float(record["submitted_at"]) + horizon

    def _missing_keys(self, stage: str, indexes: Sequence[int]) -> list[str]:
        """The artifact keys that never appeared, for the failure message."""
        if self.plan is None:
            return [shards.plan_key(self.root)]
        if stage == "export":
            return [
                self.storage.cog_key(self.plan.window, self.tile, product) for product in PRODUCTS
            ]
        expected = _expected_keys(self.plan, stage, self.root)
        return [key for index in indexes for key in expected.get(index, [])]


# --------------------------------------------------------------------------
# Waves
# --------------------------------------------------------------------------


@dataclass
class Wave:
    """One submitted array, and the workers it is holding.

    A wave exists to answer two questions: how much of the concurrency cap is
    spoken for, and which submission to blame when the probe says something
    died. It is never an answer to "is this shard done" -- that is the bucket.

    ``handle_id`` and ``handle_name`` are whatever the backend returned and are
    opaque here on purpose (:class:`~landsat_lst.fleet_backend.WaveHandle`). On
    Coiled they are a cluster; on another substrate they are a job or a task
    group, and nothing in this module would change.
    """

    stage: str
    wave: int
    units: tuple[tuple[str, int], ...]
    tiles: tuple[str, ...]
    max_workers: int
    submitted_at: float
    deadline_s: float
    handle_id: object = None
    handle_name: str = ""
    #: When this wave's *first* unit was observed to have published, and its
    #: last. Both are poll-resolution observations made by the driver against
    #: the bucket, not worker clocks: they can only be late, never early, and
    #: they are bounded by ``fleet_poll_s``. Recorded because provisioning idle
    #: is the whole cost this design attacks, and a total that cannot be split
    #: into "waiting for VMs" and "doing work" cannot calibrate anything.
    first_completion_at: float | None = None
    last_completion_at: float | None = None
    #: Units observed as landed at the last refresh, so a rising edge is
    #: detectable without keeping a second copy of the evidence.
    landed: int = 0
    #: Whether the backend acknowledged the submission. An unacknowledged wave
    #: may still be running: the control-plane call can be made, start work,
    #: and have its answer lost, so the width is held until the artifacts
    #: settle it rather than released on the strength of a missing reply.
    acknowledged: bool = False
    #: Units whose artifacts were absent at the last evidence refresh, or
    #: ``None`` when the driver has read no evidence at all. Owned by the wave,
    #: never by the tracks it carries: capacity must not be a function of a
    #: tile's own despair, and while this lived on the tracks a tile failing
    #: between the refresh and the flush in the same poll moved the number.
    #:
    #: **Never an input to capacity.** It answers "has this wave's work landed",
    #: which is a question about the bucket, and it feeds barriers, retirement
    #: in degraded mode, and the completion timestamps. What it must not do
    #: again is set a width.
    outstanding_units: set[tuple[str, int]] | None = None
    #: The name this wave is discoverable under, from
    #: :meth:`~landsat_lst.fleet_backend.FleetBackend.submission_identity`. A
    #: pure function of ``(run_id, stage, wave)``, fixed *before* the first
    #: submission attempt, and therefore the one key that survives an attempt
    #: whose answer was lost. It is what a census is matched against.
    identity: str = ""
    #: Submission attempts issued for this wave, counting the ones that raised.
    #: Each is charged at full width until a census says otherwise.
    attempts: int = 0
    #: When the last attempt was issued. A charge may only be discharged by a
    #: census taken *after* this: an earlier one could not have seen the
    #: workers it started.
    last_attempt_at: float = 0.0
    #: Width asked for, kept separately from :attr:`max_workers` because a
    #: handle overwrites that with what the substrate granted.
    requested_workers: int = 0
    #: Whether the charge has been given back, and on what evidence. Set once;
    #: a discharged wave leaves ``_live`` in the same pass.
    discharged: bool = False
    discharged_by: str = ""

    def stranded(self, now: float) -> bool:
        """Past a whole budget and a boot since this wave last showed a sign of life.

        The deadline says the wave is late. This says it has stopped producing:
        a wave still landing units is alive by the only evidence there is, and
        a late wave that is *working* must not be treated as a dead one. So the
        clock runs from the last completion the driver observed, and from the
        submission when there has never been one. Measured from submission
        instead, a deep composite wave that overran its estimate while landing
        units steadily was declared stranded and took ten tiles down with it.

        The grace on top is one :data:`~landsat_lst.budgets.VM_BOOT_S`, the one
        interval a wave can pass without producing anything and still be
        healthy: a worker replaced late has to boot before it can work.
        """
        return now >= self.stranded_at()

    def stranded_at(self) -> float:
        """When this wave stops counting as alive, absent further evidence."""
        since = self.submitted_at if self.last_completion_at is None else self.last_completion_at
        return since + self.deadline_s + budgets.VM_BOOT_S

    def intent_charge(self) -> int:
        """Width charged for this wave: **every attempt, at the width it asked for.**

        The whole of the capacity rule, and it reads nothing about work. A
        worker cannot exist before its attempt was issued, so charging from the
        instant of the call is sound by construction; and a charge is given
        back only by a census that does not contain this wave's identity, which
        is the one observation that is a function of machines.

        Per *attempt*, not per wave, because there is no idempotency key on the
        submission path this drives: ``coiled.batch_run`` is a non-atomic
        two-step, and an attempt whose answer was lost may have created a
        cluster and booted its workers. Charging only the attempt that answered
        is charging for a subset of what may be running.

        At the width **requested**, not the width the handle reported. A handle
        can only ever report a width the substrate clamped downward, and an
        attempt that lost its answer reports nothing at all while billing at the
        width it was given. So the charge takes the larger of the two.

        There is deliberately no second regime. The deleted one degraded the
        charge to one worker per outstanding unit once a wave overran its
        budget, which is the category error this class exists to stop making:
        it answered "is a VM billing" with a count of artifacts, over-charged a
        wave that was gone (Window C) and under-charged one that was merely
        slow (Window A) from the same expression.
        """
        if self.discharged:
            return 0
        return max(1, self.attempts) * max(self.requested_workers, self.max_workers)

    def expired(self, now: float) -> bool:
        return now >= self.submitted_at + self.deadline_s

    def note_landed(self, landed: int, now: float) -> bool:
        """Record the first and last completion this wave was seen to make.

        Called once per poll with the count of units whose artifacts are on the
        ground. Only a rising edge moves anything, so a re-listed key does not
        reset a timestamp.

        Returns:
            Whether anything moved, so the caller can republish the record
            rather than rewriting an unchanged object every poll.
        """
        if landed <= self.landed:
            return False
        if self.first_completion_at is None:
            self.first_completion_at = now
        self.last_completion_at = now
        self.landed = landed
        return True

    @property
    def provisioning_idle_s(self) -> float | None:
        """Submission to first completion: boot, queueing, and the first unit's work.

        Not a pure provisioning figure, and deliberately not presented as one.
        It is the interval a cost model has to attribute, and the driver's job
        is to record it rather than to interpret it: separating the boot from
        the first unit's execution needs a number only the VM can report.
        """
        if self.first_completion_at is None:
            return None
        return self.first_completion_at - self.submitted_at

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "wave": self.wave,
            "units": len(self.units),
            "tiles": list(self.tiles),
            "max_workers": self.max_workers,
            "requested_workers": self.requested_workers,
            "handle_name": self.handle_name,
            "identity": self.identity,
            "attempts": self.attempts,
            "last_attempt_at": self.last_attempt_at,
            "acknowledged": self.acknowledged,
            "first_completion_at": self.first_completion_at,
            "last_completion_at": self.last_completion_at,
            "provisioning_idle_s": self.provisioning_idle_s,
        }


@dataclass
class GhostLedger:
    """Width released while the substrate could not be asked about it.

    The degraded half of the capacity rule, and the only place the driver ever
    gives back a charge on something other than a census. When no census can be
    taken, a wave that has overrun its budget and cannot be confirmed dead is
    one of two executions the driver cannot tell apart -- ``W`` workers with a
    hung unit, or zero workers because the array was preempted. Holding the
    width is safe and kills liveness (Window C: the run failed four tiles of
    four while nothing was running). Releasing it is live and unsafe (Window A:
    127 VMs against a cap of 64).

    So the width is released *and* kept charged, here, for a bounded time. That
    is escape E1 in ``evidence/design/properties.md``, and the ledger is what
    makes it ``2C``-safe rather than unbounded: admission requires
    ``intent + ghosts + x <= C``, so at most one cap's worth of released width
    can be outstanding at a time. A naive "release after D" with no ledger grows
    the over-admission as ``ceil(L/D) * C`` over a long run, which is the
    version that is not worth having.

    Two things end an entry. The TTL, which is the hard bound and is what makes
    the run live again. Or the wave's units all landing, which is the substrate
    contract saying the array is over -- evidence beats a timer, so
    :meth:`settle` drops the entry early rather than paying out the rest of it.
    """

    #: ``identity -> (width, expires_at)``.
    entries: dict[str, tuple[int, float]] = field(default_factory=dict)

    def add(self, identity: str, width: int, now: float, ttl: float) -> None:
        if width <= 0:
            return
        held, expires = self.entries.get(identity, (0, now))
        self.entries[identity] = (max(held, int(width)), max(expires, now + float(ttl)))

    def settle(self, identity: str) -> None:
        """Drop an entry the substrate's own contract has now accounted for."""
        self.entries.pop(identity, None)

    def width(self, now: float) -> int:
        """Released-but-unconfirmed width still charged against the cap."""
        expired = [key for key, (_, expires) in self.entries.items() if now >= expires]
        for key in expired:
            del self.entries[key]
        return sum(width for width, _ in self.entries.values())

    def horizon(self, now: float) -> float:
        """Seconds until the last entry expires, so a poll budget can cover it."""
        if not self.entries:
            return 0.0
        return max(0.0, max(expires for _, expires in self.entries.values()) - now)


@dataclass
class FleetSummary:
    """One consolidated run, as it ended."""

    run_id: str
    tiles: list[TileOutcome] = field(default_factory=list)
    waves: list[Wave] = field(default_factory=list)
    wall_s: float = 0.0
    polls: int = 0

    @property
    def submissions(self) -> int:
        return len(self.waves)

    @property
    def completed(self) -> list[str]:
        return [tile.tile for tile in self.tiles if tile.completed]

    @property
    def failed(self) -> list[str]:
        return [tile.tile for tile in self.tiles if tile.failed]

    def submissions_for(self, stage: str) -> int:
        return sum(1 for wave in self.waves if wave.stage == stage)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "wall_s": round(self.wall_s, 1),
            "polls": self.polls,
            "submissions": self.submissions,
            "completed": self.completed,
            "failed": self.failed,
            "tiles": [tile.as_dict() for tile in self.tiles],
            "waves": [wave.as_dict() for wave in self.waves],
        }


class FleetDriver:
    """The poll loop: step every tile, batch what they want, submit waves."""

    def __init__(
        self,
        *,
        run_id: str,
        tracks: Sequence[TileTrack],
        storage: StorageBackend,
        backend: FleetBackend,
        clock: Clock,
        units: int,
        max_vms: int | None = None,
        wave_window_s: float | None = None,
        probe_waves: bool = False,
    ) -> None:
        check_contract(backend)
        self.run_id = run_id
        self.tracks = list(tracks)
        self._by_tile = {track.tile: track for track in self.tracks}
        self.storage = storage
        #: One listing per prefix per cycle, shared by every tile, so the
        #: driver's request rate does not grow with the tile count.
        self.index = PollIndex(storage, run_id)
        # Cast rather than subclass: the view answers every call a track makes
        # and delegates the rest, but inheriting from StorageBackend would drag
        # in abstract methods a read-through cache has no business defining.
        view = cast("StorageBackend", _IndexedStorage(storage, self.index))
        for track in self.tracks:
            track.storage = view
        self.backend = backend
        self.clock = clock
        self.units = units
        self.max_vms = int(max_vms or settings.fleet_max_vms)
        self.wave_window_s = (
            settings.fleet_wave_window_s if wave_window_s is None else float(wave_window_s)
        )
        #: Whether to ask the backend about live waves each poll. Off by
        #: default, which is what the driver did before the backend abstraction existed
        #: (the probe was an injected callable nobody passed). It is one
        #: control-plane call per live wave per poll, which over a multi-hour
        #: run is real API pressure to take on silently -- and because a probe
        #: can only ever end a barrier *sooner*, leaving it off costs a
        #: deadline's wait on a dead fleet and never correctness.
        self.probe_waves = probe_waves
        self.summary = FleetSummary(run_id=run_id)
        #: Buffered demands per stage, keyed by tile so a re-demand merges
        #: rather than duplicating. A tile keeps re-demanding while it sits in
        #: the buffer, because nothing durable says it was asked for yet.
        self._buffer: dict[str, dict[str, Demand]] = {stage: {} for stage in WAVE_STAGES}
        self._buffered_since: dict[str, float] = {}
        self._live: list[Wave] = []
        self._wave_no: dict[str, int] = dict.fromkeys(WAVE_STAGES, 0)
        self._dead_handles: set = set()
        #: The last census taken, or ``None`` when the substrate could not be
        #: asked. ``None`` is *unknown*, never *empty* -- see :meth:`in_flight`.
        self._census: WorkerCensus | None = None
        #: Width released without a census to confirm it. Empty whenever a
        #: census is available, because then nothing is released without one.
        self._ghosts = GhostLedger()
        #: Whether the run is currently accounting without a census. Held so
        #: the transition is logged once rather than every poll -- a
        #: multi-hour run polls tens of thousands of times.
        self._degraded: bool | None = None
        #: Whether an authoritative census has ever been obtained on this run.
        #: Separates "the control plane stopped answering" from "this substrate
        #: cannot be asked at all" -- see :meth:`_discharge_reason` rule 5.
        self._census_seen: bool = False
        #: Identities a census or a probe has settled, published to the tracks.
        self._dead_identities: set[str] = set()

    # -- capacity ---------------------------------------------------------

    def take_census(self) -> None:
        """Ask the substrate what it is billing this run for. Once per poll.

        One control-plane query for the whole run, replacing one probe per live
        wave, and it answers a question the probe could not put at all: what is
        running that this driver holds no handle for.

        Best-effort in the same sense as every other instrumentation call in
        this project -- a failure is logged and swallowed. What it must *not* do
        is degrade to zero, so a failure yields ``None`` and
        :attr:`in_flight` switches to the ghost-ledger rule.

        Resolved by lookup rather than by attribute access on the protocol,
        because a backend written before ``enumerable_by_run`` existed has no
        ``census`` at all, and the honest reading of that is the same as a
        control plane that will not answer: unknown, run degraded, say so.
        """
        take = getattr(self.backend, "census", None)
        census: WorkerCensus | None = None
        if take is not None:
            try:
                census = take(self.run_id)
            except Exception as e:
                log.warning("fleet_census_failed", run_id=self.run_id, error=str(e))
        self._census = census
        self._census_seen = self._census_seen or census is not None
        self._note_census_mode(census)

    def _note_census_mode(self, census: WorkerCensus | None) -> None:
        """Log the degraded transition, once per crossing rather than per poll."""
        degraded = census is None
        if degraded == self._degraded:
            return
        self._degraded = degraded
        if degraded:
            log.warning(
                "fleet_census_degraded",
                run_id=self.run_id,
                backend=getattr(self.backend, "name", "?"),
                concurrency_cap=self.max_vms,
                ghost_ttl_s=settings.fleet_ghost_ttl_s,
                note=(
                    "no authoritative worker census; capacity is the intent charge plus a "
                    "ghost ledger, which bounds concurrency at 2x the cap rather than at it"
                ),
            )
            return
        log.info(
            "fleet_census_available",
            run_id=self.run_id,
            total=census.total if census else 0,
            identities=len(census.identities) if census else 0,
        )

    def _discharge_reason(self, wave: Wave, now: float) -> str | None:
        """The evidence, if any, that this wave's charge may be given back.

        Ordered by strength, and the ordering is the whole design:

        1. **The backend confirmed the submission dead.** A statement about
           machines, and the only thing a probe is allowed to assert.
        2. **A census taken after the last attempt omits the identity.** The
           authoritative case. ``after`` matters: an earlier census could not
           have seen workers an attempt had not yet asked for.
        3. Nothing else, *while a census is available*. In particular, every
           unit of the wave having landed does **not** discharge it -- the
           artifacts say the work finished, the census says the VMs are still
           there, and the census is the one that is billing.

        With no census, two degraded rules take over and neither is as good:

        4. **Every unit landed.** ``queues_surplus`` ties the array's end to its
           last unit, so this is the substrate's own contract read through the
           bucket rather than an inference about idle workers. Observed at poll
           resolution, so late and never early.
        5. **Stranded past its budget with nothing able to confirm it**, on a
           run whose substrate *has* answered a census at some point. The
           genuinely undecidable case, released into the :class:`GhostLedger`
           so the width keeps counting while the tiles get to move again.

        Rule 5 is gated on having seen a census, and the gate is the difference
        between two situations that look alike and are not. A control plane that
        answered this run and has now stopped answering is a transient, the
        identity is known to be enumerable, and a TTL is a real bound on the
        release -- escape E1, ``2C``-safe. A substrate that has *never* answered
        offers no evidence that anything it started is enumerable at all, so a
        timed release would be a guess with nothing behind it. There the driver
        holds, and the run ends loudly through :meth:`_stalled` rather than
        quietly at twice its cap. That is the liveness price of soundness when
        there is no measurement, and it is the price the design accepts: it is
        exactly Window C, and it is bounded, stated, and one tile-failure per
        tile rather than a silent doubling of the bill.
        """
        if wave.handle_id is not None and wave.handle_id in self._dead_handles:
            return "probe_dead"
        census = self._census
        if census is not None:
            if census.as_of >= wave.last_attempt_at and wave.identity not in census.identities:
                return "census_absent"
            return None
        if wave.units and wave.outstanding_units is not None and not wave.outstanding_units:
            return "array_ended"
        if self._census_seen and wave.units and wave.stranded(now):
            return "stranded_unconfirmed"
        return None

    def _discharge(self, wave: Wave, reason: str, now: float) -> None:
        """Give a wave's charge back, ghosting it when nothing confirmed the release."""
        width = self.wave_held(wave)
        wave.discharged = True
        wave.discharged_by = reason
        if reason == "stranded_unconfirmed":
            self._ghosts.add(wave.identity, width, now, settings.fleet_ghost_ttl_s)
            log.warning(
                "fleet_wave_ghosted",
                run_id=self.run_id,
                stage=wave.stage,
                wave=wave.wave,
                identity=wave.identity,
                width=width,
                ttl_s=settings.fleet_ghost_ttl_s,
                note="width released without confirmation; still charged until the TTL expires",
            )
            return
        # Confirmed, or as good as: drop any ghost this identity was carrying
        # from an earlier release rather than paying out the rest of its TTL.
        self._ghosts.settle(wave.identity)
        if reason in ("census_absent", "probe_dead"):
            # Told to the tracks, so a tile stops waiting out a deadline on
            # workers the substrate has stopped reporting.
            self._dead_identities.add(wave.identity)
        log.info(
            "fleet_wave_discharged",
            run_id=self.run_id,
            stage=wave.stage,
            wave=wave.wave,
            identity=wave.identity,
            reason=reason,
            width=width,
        )

    def _retire(self) -> None:
        """Give back the charge of waves the substrate is no longer billing for.

        **A deadline is not proof and neither is an artifact.** An expired wave
        whose units have not landed is a wave that is late, and a late wave's
        workers are still billing. A wave whose units *have* landed may still
        have an array draining. Releasing on either is how a run comes to hold
        two waves' worth of VMs against a one-wave cap: the tiles re-demand, a
        fresh wave is submitted into headroom that only exists on paper, and the
        real concurrency is the sum of both.

        So the release rule is :meth:`_discharge_reason`, whose first two
        answers are statements about machines. The bucket is consulted only when
        no census can be taken, and then the run is already logged as degraded.

        The barrier is a separate question and still expires on time: a tile
        re-demands when *its record* ages out, whether or not the wave that
        wrote it has given back its capacity. Expiry therefore delays a
        resubmission rather than permitting an over-run, which is the safe way
        round.

        There is deliberately no shortcut through a track's own state. An
        earlier draft retired a wave once every tile it carried had *settled*
        that stage, which reads as evidence and is not: a tile settles when it
        finishes a stage, and it settles just as surely when it gives up after
        its last barrier round. In the second case the artifacts are absent
        precisely because the workers are still running, so that shortcut handed
        a live wave's width back on the strength of the tile's own despair.
        """
        self._refresh_wave_evidence()
        now = self.clock.now()
        kept: list[Wave] = []
        for wave in self._live:
            reason = self._discharge_reason(wave, now)
            if reason is not None:
                self._discharge(wave, reason, now)
                continue
            if wave.expired(now):
                log.warning(
                    "fleet_wave_overdue",
                    run_id=self.run_id,
                    stage=wave.stage,
                    wave=wave.wave,
                    max_workers=wave.max_workers,
                    note="still charged against the cap until a census stops reporting it",
                )
            kept.append(wave)
        self._live = kept

    def _publish_held_units(self) -> None:
        """Tell each track which of its units a live wave is still carrying.

        Read once per poll, after retirement, so a track only ever sees waves
        the driver still believes hold workers. Stranded waves are left out:
        past its own budget a wave is what the barrier round exists for.
        """
        held: dict[str, dict[str, set[int]]] = {}
        now = self.clock.now()
        for wave in self._live:
            if wave.stranded(now):
                continue
            for tile, index in wave.units:
                held.setdefault(tile, {}).setdefault(wave.stage, set()).add(index)
        for track in self.tracks:
            track.held_units = held.get(track.tile, {})

    def _unobservable(self, wave: Wave) -> bool:
        """A wave the bucket can never settle, because it names no units.

        Only a record written before ``unit_tokens`` existed, and whose tiles
        left no per-tile records and are not on this driver's roster, gets
        here. Nothing in the bucket refers to it, so evidence cannot retire it
        and only the backend could -- which has been asked, at adoption and on
        every poll since.
        """
        return not wave.units

    def _stalled(self) -> bool:
        """Whether waiting can still change anything.

        True when there is no headroom and every wave holding it is either
        stranded or unobservable: no artifact is coming, so no wave retires, so
        no headroom returns, so nothing can be submitted. Without this the
        driver polls to its ceiling and returns *normally* with every tile in
        neither the completed nor the failed list, which is the one answer a
        build must never give. A wave preempted 1200 s into its first stage
        produced exactly that: 19,440 polls, no completions, no failures, ten
        tiles unaccounted for.

        Deliberately narrow. Waves in either state that leave headroom behind
        them are not a stall: the tiles they carried re-demand when their own
        records age out and either run or exhaust their barrier rounds, which
        is the ordinary bounded-failure path and already correct.
        """
        now = self.clock.now()
        return (
            bool(self._live)
            and self.headroom <= 0
            and all(wave.stranded(now) or self._unobservable(wave) for wave in self._live)
        )

    def _settle_stragglers(self, reason: str) -> None:
        """Fail every tile the run is about to abandon, so none is left unstated.

        The invariant this exists for: :meth:`run` never returns with a tile in
        neither list. A tile that ends a run neither built nor failed is worse
        than a failed one, because a caller reconciling the manifest cannot
        tell it from a tile that was never asked for.
        """
        for track in self.tracks:
            if not track.terminal:
                track._fail(reason)

    def _abandon_stranded(self) -> None:
        """End the run on the waves that will never answer, naming them."""
        waves = ", ".join(
            f"{wave.stage} wave {wave.wave} "
            f"({wave.handle_name or 'unnamed'}, {wave.max_workers} workers, "
            f"{'acknowledged' if wave.acknowledged else 'unacknowledged'})"
            for wave in self._live
        )
        reason = (
            "the run holds no headroom and no live wave can still be settled -- each has "
            f"either overrun its budget or names no units the bucket can answer for: {waves}"
        )
        log.error(
            "fleet_run_stranded",
            run_id=self.run_id,
            waves=len(self._live),
            in_flight=self.in_flight,
            degraded=self._census is None,
            reason=reason,
        )
        self._settle_stragglers(reason)

    def _refresh_wave_evidence(self, *, stamp: bool = True) -> None:
        """Re-read the bucket for every ``(tile, stage)`` a live wave references.

        A healthy track already refreshes its own stage each poll on its way
        through :meth:`TileTrack.step`. Two cases do not, and both are cases
        where getting it wrong costs VMs: a **terminal** track is never stepped
        again, and a track that demanded a composite wave from inside its
        offsets barrier will not look at the composite stage until it gets
        there. Left to memory, the first hands back width that is still in use
        and the second holds width that has already been given up.
        """
        wanted: dict[tuple[str, str], set[int]] = {}
        for wave in self._live:
            for tile, index in wave.units:
                wanted.setdefault((tile, wave.stage), set()).add(index)
        absent: dict[tuple[str, str], set[int]] = {}
        for (tile, stage), indexes in wanted.items():
            track = self._by_tile.get(tile)
            # A wave may name a tile this driver is not driving, which is what
            # a roster that shrank between runs looks like. Nothing can be read
            # for it, so every unit of it stays counted.
            absent[tile, stage] = (
                track.refresh_outstanding(stage, sorted(indexes))
                if track is not None
                else set(indexes)
            )
        for wave in self._live:
            if wave.units:
                wave.outstanding_units = {
                    (tile, index)
                    for tile, index in wave.units
                    if index in absent.get((tile, wave.stage), set())
                }
        now = self.clock.now()
        for wave in self._live:
            if not wave.units:
                continue
            landed = len(wave.units) - self.wave_outstanding(wave)
            if not stamp:
                # Adoption: take the count without stamping a time. A wave that
                # completed under the previous driver must not report this
                # driver's start as the moment its first unit landed.
                wave.landed = max(wave.landed, landed)
                continue
            if wave.note_landed(landed, now):
                # Republished so the interval survives this process. A cost
                # model reading the bucket afterwards is the only consumer that
                # matters, and it cannot ask a driver that has exited.
                self._record_wave(wave)

    def wave_held(self, wave: Wave) -> int:
        """Workers this wave is charged for. See :meth:`Wave.intent_charge`.

        A wave's own charge, never a track's: nothing a tile does to itself may
        move the cap.
        """
        return wave.intent_charge()

    def wave_outstanding(self, wave: Wave) -> int:
        """Units of this wave whose artifacts were absent at the last refresh.

        A wave with no evidence read yet counts every unit it carries, and one
        adopted with no unit list at all counts nothing -- there is no unit to
        count. Its width is held through :attr:`Wave.held`, which does not go
        through this number.
        """
        if wave.outstanding_units is None:
            return len(wave.units)
        return len(wave.outstanding_units)

    @property
    def intent_charge(self) -> int:
        """``intent_charge(t)``: every undischarged submission attempt, at full width."""
        return sum(self.wave_held(wave) for wave in self._live)

    @property
    def in_flight(self) -> int:
        """``H(t) = max(intent_charge(t), census(t).total)``. The one capacity rule.

        Sound because a worker cannot exist outside the interval between the
        attempt that asked for it and the census that stops reporting it, and
        both ends of that interval are represented. Convergent because every
        charge is discharged by the first census that omits its identity, and a
        census is taken every poll.

        The maximum rather than either term alone. The census can lag a
        submission -- a cluster does not appear the instant ``batch_run``
        returns -- so the intent charge covers the boot. The intent charge can
        miss workers that belong to this run and to no attempt this driver
        holds -- an orphan from a lost acknowledgement, a spot replacement above
        the cap -- so the census covers those. Neither dominates, which is why
        neither is dropped.

        **When there is no census, this is a different guarantee and says so.**
        ``None`` from the backend means "cannot answer", so the driver falls
        back to the intent charge plus :class:`GhostLedger`: still bounded, but
        by ``2C`` rather than ``C``, and only because the ledger keeps released
        width charged for a TTL. That mode is logged (``fleet_census_degraded``)
        rather than inferred, which the code it replaces never did -- it was
        permanently in a worse version of this mode and silent about it.
        """
        intent = self.intent_charge
        census = self._census
        if census is None:
            return intent + self._ghosts.width(self.clock.now())
        return max(intent, census.total)

    def capacity_ledger(self) -> dict:
        """Every worker the driver is counting, by the identity that holds it.

        The cap is only as good as the arithmetic behind it, and arithmetic
        that can only be checked against itself is not checked. So the driver
        publishes the *identities* it counts -- which wave, which units, and
        whether the backend ever acknowledged the submission -- and an
        independent ledger of potentially-live workers can be reconciled
        against it without sharing a line of this module's counting.
        """
        census = self._census
        now = self.clock.now()
        return {
            "run_id": self.run_id,
            #: A ceiling on *concurrent* VMs. Not a budget: see
            #: :attr:`~landsat_lst.config.Settings.fleet_max_vms`.
            "concurrency_cap": self.max_vms,
            "in_flight": self.in_flight,
            "intent_charge": self.intent_charge,
            "headroom": self.headroom,
            #: ``null`` means the substrate could not be asked, which is not
            #: the same as a census reporting nothing.
            "census_total": None if census is None else census.total,
            "census_as_of": None if census is None else census.as_of,
            "census_unattributed": None if census is None else census.unattributed,
            "degraded": census is None,
            "ghost_width": self._ghosts.width(now),
            "waves": [
                {
                    "stage": wave.stage,
                    "wave": wave.wave,
                    "handle_name": wave.handle_name,
                    "identity": wave.identity,
                    "attempts": wave.attempts,
                    "acknowledged": wave.acknowledged,
                    "max_workers": wave.max_workers,
                    "requested_workers": wave.requested_workers,
                    "census_workers": None
                    if census is None
                    else census.by_identity.get(wave.identity, 0),
                    "held": self.wave_held(wave),
                    "outstanding": sorted(
                        shards.fleet_unit_token(tile, index)
                        for tile, index in (wave.outstanding_units or ())
                    )
                    if wave.outstanding_units is not None
                    else None,
                }
                for wave in self._live
            ],
        }

    @property
    def headroom(self) -> int:
        """VMs the cap still allows. The one hard limit on a consolidated run."""
        return max(0, self.max_vms - self.in_flight)

    # -- batching ---------------------------------------------------------

    def _buffer_demand(self, demand: Demand) -> None:
        buf = self._buffer[demand.stage]
        if not buf:
            self._buffered_since[demand.stage] = self.clock.now()
        buf[demand.tile] = demand

    def _could_still_join(self, stage: str) -> bool:
        """Whether any tile not already buffered could add units to ``stage``.

        The third flush condition. Without it a wave would always wait out the
        window even when there is provably nobody left to wait for, which is
        both the single-tile case and the tail of every run.
        """
        buffered = self._buffer[stage]
        return any(
            not track.has_settled(stage) and track.tile not in buffered for track in self.tracks
        )

    def _ready_to_flush(self, stage: str) -> bool:
        buf = self._buffer[stage]
        if not buf or self.headroom <= 0:
            return False
        units = sum(len(demand.indexes) for demand in buf.values())
        if units >= self.headroom:
            return True
        if self.clock.now() - self._buffered_since.get(stage, 0.0) >= self.wave_window_s:
            return True
        return not self._could_still_join(stage)

    # -- submission -------------------------------------------------------

    def _flush(self, stage: str) -> None:
        """Submit one wave, recording every tile in it first.

        Tile granularity, always: ``max_workers`` is capped but the unit list is
        not, so a tile demanding more units than the whole cap runs queued
        rather than split. A split demand would write a submission record for a
        partial index set, and the remainder would cost the tile a barrier round
        it never used.
        """
        demands = list(self._buffer[stage].values())
        units = tuple((demand.tile, index) for demand in demands for index in demand.indexes)
        self._wave_no[stage] += 1
        wave_no = self._wave_no[stage]
        workers = max(1, min(len(units), self.headroom))
        # Scaled by queue depth, not by one shard's budget. A wave deep enough
        # to amortize a boot runs ceil(units/workers) serial rounds, so the
        # single-round deadline the per-tile driver uses would guarantee that
        # every wave worth submitting expires before it finishes -- and an
        # expired wave lets every tile in it spend a barrier round it does not
        # have. See budgets.wave_deadline_s.
        deadline_s = budgets.wave_deadline_s(
            boot_s=max(demand.boot_s for demand in demands),
            unit_work_s=max(demand.unit_work_s for demand in demands),
            units=len(units),
            workers=workers,
        )
        # The identity, before anything is called. It is a pure function of
        # ``(run_id, stage, wave)``, so it is the same string a census will be
        # matched against and the same string a resumed driver rebuilds -- and,
        # crucially, the same string whether or not the submission answers.
        name = self._identity(stage, wave_no)
        submitted_at = self.clock.now()

        # Records first, and one per tile: that is what keeps adoption, the
        # across-drivers round budget, and resume working per tile even though
        # the submission is shared. A driver that dies between the record and
        # the submission costs the next one a barrier; the other order leaves a
        # live submission nothing mentions, which is the collision this whole
        # mechanism exists to remove.
        for demand in demands:
            self._record(demand, wave_no, name, submitted_at, handle_id=None, deadline_s=deadline_s)

        wave = Wave(
            stage=stage,
            wave=wave_no,
            units=units,
            tiles=tuple(dict.fromkeys(demand.tile for demand in demands)),
            max_workers=workers,
            requested_workers=workers,
            submitted_at=submitted_at,
            deadline_s=deadline_s,
            handle_name=name,
            identity=name,
            # Seed the evidence for what is about to be submitted. A demand
            # only ever carries indexes whose artifacts are absent, so at the
            # moment of submission the wave holds its full width, and saying so
            # here removes an ordering dependency between stepping tracks and
            # flushing stages.
            outstanding_units=set(units),
        )
        for demand in demands:
            track = self._by_tile.get(demand.tile)
            if track is not None:
                track.outstanding.setdefault(stage, set()).update(demand.indexes)

        # Live, recorded, and counted against the cap *before* the call that
        # starts it. A submission is not an event the driver witnesses: the
        # control plane can accept the request, boot the workers, and lose the
        # answer on the way back, and a driver that only counts what it was
        # told about then reports full headroom while ninety VMs run. The same
        # ordering is what makes recovery idempotent -- the record is at a key
        # that is a pure function of ``(run_id, stage, wave)``, so a restart or
        # a duplicate driver adopts the wave rather than minting a second one.
        self._live.append(wave)
        self.summary.waves.append(wave)
        self._record_wave(wave)

        handle = self._submit_with_retries(wave)
        if handle is not None:
            wave.handle_id = handle.id
            wave.handle_name = handle.name or name
            wave.max_workers = handle.max_workers
            wave.acknowledged = True
            for demand in demands:
                self._record(
                    demand,
                    wave_no,
                    wave.handle_name,
                    submitted_at,
                    handle_id=handle.id,
                    deadline_s=deadline_s,
                )
            self._record_wave(wave)
        else:
            # Retries exhausted on transient errors, which is exactly the case
            # that cannot be read as "nothing started": every attempt may have
            # created a cluster whose answer was lost. All of them stay charged,
            # and the census finds them by ``identity`` even though no handle
            # came back -- which is the whole point of fixing the name before
            # the call rather than reading it out of the reply.
            log.error(
                "fleet_wave_unacknowledged",
                run_id=self.run_id,
                stage=stage,
                wave=wave_no,
                identity=name,
                attempts=wave.attempts,
                charged=self.wave_held(wave),
                note="submission unconfirmed; every attempt stays charged until a census clears it",
            )
            # Republished with the attempt count, so a resumed driver charges
            # what this one charged. Without it the record says one attempt, the
            # next driver charges one width, and the under-count survives
            # exactly the crash the record exists to survive.
            self._record_wave(wave)
        self._buffer[stage] = {}
        self._buffered_since.pop(stage, None)
        log.info(
            "fleet_wave_submitted",
            run_id=self.run_id,
            stage=stage,
            wave=wave_no,
            tiles=len(wave.tiles),
            units=len(units),
            max_workers=wave.max_workers,
            acknowledged=wave.acknowledged,
            in_flight=self.in_flight,
        )

    def _identity(self, stage: str, wave_no: int) -> str:
        """The name this wave will be discoverable under, before it is submitted.

        ``submission_identity`` where the backend has it, ``wave_name`` where it
        does not. The fallback is not politeness: the two have always been the
        same string on every backend here, and a backend predating the census
        contract still produces a deterministic, pre-call name -- which is all
        an identity has to be.
        """
        identify = getattr(self.backend, "submission_identity", None)
        if identify is None:
            return self.backend.wave_name(self.run_id, stage, wave_no)
        return identify(self.run_id, stage, wave_no)

    def _submit_with_retries(self, wave: Wave) -> WaveHandle | None:
        """Submit, retrying transient control-plane failures, **charging per attempt.**

        Every attempt is charged at full width from the instant the call is
        issued, and stays charged until a census omits the wave's identity.
        That is not defensive book-keeping: there is no idempotency key on this
        path, the submission is a non-atomic two-step, and an attempt that
        raises on the way back may have created a cluster and booted its
        workers. Counting only the attempt that answered counts a subset of
        what may be billing, which is the measured 90-up / 30-counted window.

        Terminal means the run stops now and says why: a quota that is already
        exhausted is not going to clear inside a backoff, and burning the
        remaining rounds turns a two-line explanation into a long silence.

        *Which* errors are terminal is the backend's judgement, not this
        module's -- see ``classified_failures`` in
        :data:`~landsat_lst.fleet_backend.BACKEND_CONTRACT`. All the state
        machine requires is that an unrecognized error be transient.
        """
        stage, wave_no, units = wave.stage, wave.wave, wave.units
        last = ""
        for attempt in range(1, settings.shard_submit_retries + 1):
            try:
                # Charged *before* the call, for the same reason the wave is
                # recorded before it: the substrate can accept the request,
                # start the VMs, and lose the answer.
                wave.attempts += 1
                wave.last_attempt_at = self.clock.now()
                return self.backend.submit(
                    WaveRequest(
                        stage=stage,
                        run_id=self.run_id,
                        units=units,
                        wave=wave_no,
                        max_workers=wave.requested_workers,
                        fleet_units=self.units,
                    )
                )
            except Exception as e:
                last = f"{type(e).__name__}: {e}".strip().rstrip(":")
                kind = self.backend.classify_failure(e)
                log.warning(
                    "fleet_wave_submit_failed",
                    run_id=self.run_id,
                    stage=stage,
                    wave=wave_no,
                    attempt=attempt,
                    kind=kind,
                    error=last,
                )
                if kind == "terminal":
                    msg = f"stage {stage!r}: {last}"
                    raise FleetAborted(msg) from e
                if attempt >= settings.shard_submit_retries:
                    # Transient and out of attempts. The records are already
                    # written, so every tile in this wave watches, expires, and
                    # spends a round: one wasted wait, not a dead run.
                    log.error(
                        "fleet_wave_submit_exhausted",
                        run_id=self.run_id,
                        stage=stage,
                        wave=wave_no,
                        error=last,
                    )
                    return None
                self.clock.sleep(settings.shard_submit_backoff_s * 2 ** (attempt - 1))
        return None

    def _record_wave(self, wave: Wave) -> None:
        """Publish the wave itself, for the next driver's cap and numbering.

        The per-tile records answer "has anyone started this tile's stage". They
        cannot answer the two questions a *resumed* driver has to ask before it
        submits anything: how many workers a previous driver already has up, and
        which wave number is free. Restarting the numbering at 1 would rebuild a
        name that is still in flight -- harmless on a substrate that tolerates
        duplicate names, fatal on one that does not, which is why
        ``unique_wave_names`` is in the contract rather than assumed.
        """
        try:
            self.storage.write_text(
                shards.fleet_submission_key(self.run_id, wave.stage, wave.wave),
                json.dumps(
                    {
                        "run_id": self.run_id,
                        **wave.as_dict(),
                        "submitted_at": wave.submitted_at,
                        "deadline_s": wave.deadline_s,
                        "handle_id": wave.handle_id,
                        # The unit list, so a resumed driver can ask the bucket
                        # what this wave is still holding instead of guessing
                        # from a timestamp. Tokens rather than pairs, reusing
                        # the grammar the task input already speaks.
                        "unit_tokens": [
                            shards.fleet_unit_token(tile, index) for tile, index in wave.units
                        ],
                    }
                ),
            )
        except Exception as e:
            log.warning("fleet_wave_record_failed", stage=wave.stage, error=str(e))

    def adopt_live_waves(self) -> None:
        """Reconstruct another driver's in-flight waves, before submitting any.

        Called once at startup, for two reasons. A wave that is still holding
        workers is capacity this driver must count against its own cap, and a
        wave number that has been used is a name this driver must not rebuild.

        Adoption follows the same rule as retirement: **evidence, never the
        clock**. An earlier draft adopted only waves whose deadline had not
        passed, which quietly reintroduced the defect the retirement path was
        written to remove -- a resumed driver would ignore a wave that was
        merely late, submit into headroom that existed only on paper, and run at
        twice its cap. So every recorded wave is adopted and then retired by
        :meth:`_retire` on the ordinary terms: its units' artifacts are on the
        ground, or the backend confirms the submission is gone. That
        terminates in practice because :meth:`_probe` always probes an overdue
        wave, whatever ``probe_waves`` says, so an ancient record is retired on
        the first poll by a backend that answers and holds capacity loudly by
        one that cannot.

        A record written before ``unit_tokens`` existed carries no unit list of
        its own, but the per-tile submission records do: each names its wave and
        the indexes that tile contributed. Rebuilding from those is what lets an
        adopted wave be *observed* to have settled. Without it the wave held its
        requested width for the whole life of the resumed driver, since the only
        thing that could ever release it was a probe that may not answer.
        """
        for stage in WAVE_STAGES:
            try:
                keys = sorted(
                    self.storage.list_prefix(shards.fleet_submission_prefix(self.run_id, stage))
                )
            except Exception as e:
                log.warning("fleet_wave_listing_failed", stage=stage, error=str(e))
                continue
            for key in keys:
                raw = self.storage.read_text(key)
                if raw is None:
                    continue
                try:
                    body = json.loads(raw)
                except ValueError:
                    log.warning("fleet_wave_record_malformed", key=key)
                    continue
                number = int(body.get("wave", 0))
                self._wave_no[stage] = max(self._wave_no[stage], number)
                tiles = tuple(body.get("tiles") or ())
                units = _unit_tokens(body.get("unit_tokens")) or self._units_from_tile_records(
                    stage, number, tiles
                )
                wave = Wave(
                    stage=stage,
                    wave=number,
                    units=units,
                    tiles=tiles,
                    max_workers=int(body.get("max_workers") or 0),
                    requested_workers=int(
                        body.get("requested_workers") or body.get("max_workers") or 0
                    ),
                    submitted_at=float(body.get("submitted_at") or 0.0),
                    deadline_s=float(body.get("deadline_s") or 0.0),
                    handle_id=body.get("handle_id"),
                    handle_name=str(body.get("handle_name") or ""),
                    # A record written before identities existed still has one,
                    # because the identity was always a pure function of the
                    # key's own fields. Rebuilding it rather than defaulting to
                    # empty is what lets a census discharge an adopted wave.
                    identity=str(body.get("identity") or self._identity(stage, number)),
                    attempts=int(body.get("attempts") or 1),
                    last_attempt_at=float(
                        body.get("last_attempt_at") or body.get("submitted_at") or 0.0
                    ),
                    # A record written before this field existed describes a
                    # wave that was submitted, so absence reads as acknowledged.
                    acknowledged=bool(body.get("acknowledged", True)),
                    first_completion_at=_opt_float(body.get("first_completion_at")),
                    last_completion_at=_opt_float(body.get("last_completion_at")),
                )
                self._live.append(wave)
                log.info(
                    "fleet_wave_adopted",
                    run_id=self.run_id,
                    stage=stage,
                    wave=number,
                    max_workers=wave.max_workers,
                    units=len(wave.units),
                )
        # Take the evidence without stamping a completion time, then retire
        # whatever the bucket already settles, so a resume into a finished run
        # starts with its whole cap rather than with a roster of ghosts.
        # The census first: it answers for every adopted wave at once, and for
        # anything the previous driver started and never recorded. A wave this
        # process never submitted is a wave it has confirmed nothing about, and
        # its record's horizon is the previous driver's arithmetic rather than
        # this one's -- so the probe is forced too, for the degraded case where
        # no census can be taken.
        self.take_census()
        self._probe(force=True)
        self._refresh_wave_evidence(stamp=False)
        self._retire()
        for wave in self._live:
            if self._unobservable(wave):
                log.warning(
                    "fleet_wave_unobservable",
                    run_id=self.run_id,
                    stage=wave.stage,
                    wave=wave.wave,
                    max_workers=wave.max_workers,
                    note="record names no units; its width is held until the backend settles it",
                )

    def _units_from_tile_records(
        self, stage: str, wave_no: int, tiles: Sequence[str]
    ) -> tuple[tuple[str, int], ...]:
        """One wave's units, rebuilt from the per-tile records it wrote first.

        Every wave writes one submission record per tile before it submits, and
        that record carries the wave number and the indexes the tile
        contributed. So the evidence a pre-``unit_tokens`` wave record lacks is
        already in the bucket, one object per tile. Best-effort like everything
        read back for bookkeeping: what cannot be read yields fewer units, and
        a wave with none holds its full width.
        """
        units: list[tuple[str, int]] = []
        for tile in tiles:
            root = shards.shard_root(self.run_id, tile)
            matched = [
                (tile, int(index))
                for record in _submission_records(self.storage, root, stage)
                if int(record.get("wave", 0)) == wave_no
                for index in record.get("indexes") or ()
            ]
            if matched:
                units.extend(matched)
                continue
            # Nothing named the tile's indexes, but the plan does: a tile this
            # driver is already driving has one, and its stage's index set is
            # the widest a wave for it could have carried. Over-counting the
            # units delays a retirement rather than permitting an over-run.
            track = self._by_tile.get(tile)
            if track is not None and track.plan is not None:
                units.extend((tile, index) for index in _expected_keys(track.plan, stage, root))
        return tuple(dict.fromkeys(units))

    def _record(
        self,
        demand: Demand,
        wave_no: int,
        handle_name: str,
        submitted_at: float,
        *,
        handle_id: object,
        deadline_s: float,
    ) -> None:
        """One tile's submission record, in the shared ADR-016 schema.

        The ``cluster_*`` field names are deliberately kept even though the
        driver now speaks in handles: this object is read by the single-tile
        driver's ``_is_live`` and by anything else that inspects a stage
        submission, and a fleet run must not write a shape they cannot read.
        The *value* is whatever the backend's handle carries.
        """
        root = shards.shard_root(self.run_id, demand.tile)
        payload = {
            "run_id": self.run_id,
            "tile": demand.tile,
            "stage": demand.stage,
            "round": demand.submission_round,
            "indexes": list(demand.indexes),
            "cluster_name": handle_name,
            "cluster_id": handle_id,
            "wave": wave_no,
            "submitted_at": submitted_at,
            # The *wave's* horizon, not this tile's single-round one. A tile's
            # unit may be scheduled last in the queue, so the honest liveness
            # window for its record is the whole wave's. Readers that predate
            # this field fall back to their own estimate.
            "deadline_s": deadline_s,
        }
        try:
            self.storage.write_text(
                shards.stage_submission_key(root, demand.stage, demand.submission_round),
                json.dumps(payload),
            )
        except Exception as e:
            log.warning("fleet_submission_record_failed", tile=demand.tile, error=str(e))

    # -- liveness ---------------------------------------------------------

    def _probe(self, *, force: bool = False) -> None:
        """Ask the backend whether any live wave's submission is gone.

        Advisory by contract (``probe_is_advisory``): it may only end a barrier
        *sooner*, never declare success, and the tracks re-check the bucket
        before they act on it -- so a fleet whose last task uploaded and then
        stopped reads as a finished stage rather than a killed one.

        Args:
            force: Ask about every live wave whatever its deadline says. Used
                once at adoption: a wave inherited from another driver is the
                one case where this process has confirmed nothing itself, and
                a long horizon in the record would otherwise postpone the only
                question that can release it.
        """
        now = self.clock.now()
        for wave in self._live:
            if wave.handle_id is None or wave.handle_id in self._dead_handles:
                continue
            # An overdue wave is always probed, whatever ``probe_waves`` says:
            # since expiry no longer releases its width, confirmation is the
            # only way that capacity comes back, and a run that cannot ask
            # would stall holding VMs that are already gone.
            if not force and not self.probe_waves and not wave.expired(now):
                continue
            try:
                probed = self.backend.probe(wave.handle_id)
            except Exception as e:
                log.warning("fleet_probe_failed", handle_id=wave.handle_id, error=str(e))
                continue
            if probed is None:
                continue
            state, reason = probed
            if state.lower() in ("error", "stopped"):
                log.warning(
                    "fleet_wave_handle_dead",
                    run_id=self.run_id,
                    stage=wave.stage,
                    wave=wave.wave,
                    state=state,
                    reason=reason,
                )
                self._dead_handles.add(wave.handle_id)

    # -- the loop ---------------------------------------------------------

    def run(self) -> FleetSummary:
        started = self.clock.now()
        # Before anything is submitted: whatever another driver still has up
        # counts against this driver's cap, and its wave numbers are taken.
        self.adopt_live_waves()
        # Bounded so a clock that stops advancing is loud rather than infinite.
        ceiling = self._poll_ceiling()
        finished = False
        while self.summary.polls < ceiling:
            self.summary.polls += 1
            # One listing per shared prefix, before any tile looks at anything.
            self.index.refresh()
            # One control-plane query for the whole run, before any capacity
            # decision is taken against it.
            self.take_census()
            self._probe()
            for track in self.tracks:
                track.dead_handles = self._dead_handles
                track.dead_identities = self._dead_identities
            self._retire()
            self._publish_held_units()
            if self._stalled():
                # Loud already, through ``fleet_run_stranded`` and one failure
                # per tile. The ceiling did not end this run.
                self._abandon_stranded()
                finished = True
                break

            for track in self.tracks:
                for demand in track.step():
                    self._buffer_demand(demand)

            for stage in WAVE_STAGES:
                if self._ready_to_flush(stage):
                    self._flush(stage)

            if all(track.terminal for track in self.tracks):
                finished = True
                break
            self.clock.sleep(settings.fleet_poll_s)
            ceiling = max(ceiling, self._live_wave_ceiling())
        if not finished:
            log.error("fleet_poll_ceiling_reached", run_id=self.run_id, polls=self.summary.polls)

        # Whatever ended the loop, no tile leaves it unstated. A run that
        # breaks on every track being terminal passes through here untouched.
        self._settle_stragglers(
            f"the driver stopped after {self.summary.polls} polls with this tile still in "
            "neither state; nothing it was waiting on ever settled"
        )
        self.summary.tiles = [track.outcome for track in self.tracks]
        self.summary.wall_s = self.clock.now() - started
        log.info(
            "fleet_run_done",
            run_id=self.run_id,
            tiles=len(self.tracks),
            completed=len(self.summary.completed),
            failed=len(self.summary.failed),
            submissions=self.summary.submissions,
            wall_s=round(self.summary.wall_s, 1),
        )
        return self.summary

    def _poll_ceiling(self) -> int:
        """A generous bound on polls, so a stuck clock is loud rather than endless."""
        rounds = settings.shard_barrier_rounds * len(TILE_STAGES) + 4
        per_round = max(1, int(_bootstrap_deadline_s() / max(settings.fleet_poll_s, 1e-6)) + 2)
        return rounds * per_round * max(1, len(self.tracks))

    def _live_wave_ceiling(self) -> int:
        """Polls the deepest live wave may still legitimately take.

        The base ceiling counts barrier rounds, not queue depth. A cap narrow
        against the work admits few workers at a time, which turns one wave
        into as many serial rounds as it has units: at a cap of one, four
        tiles' offsets is sixty rounds in a single wave, and the run was cut
        off with its one worker still working. So the bound follows the
        deadlines the waves were actually given, and it extends only while a
        wave is live and producing. With nothing live it does not move, which
        is the case a ceiling exists for.
        """
        now = self.clock.now()
        # A ghost holds capacity with no wave behind it, so a run waiting one
        # out has nothing in ``_live`` to extend the budget and would be cut off
        # for being patient. The horizon is finite by construction.
        remaining = self._ghosts.horizon(now)
        if self._live:
            remaining = max(remaining, max(wave.stranded_at() for wave in self._live) - now)
        if remaining <= 0:
            return 0
        return self.summary.polls + int(remaining / max(settings.fleet_poll_s, 1e-6)) + 2


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def _gates(backend: FleetBackend, storage: StorageBackend, n_tiles: int) -> None:
    """The two gates a run passes before it may submit anything.

    Both are the backend's: *what* an identity is, *what* a run costs, and
    *where* the workers can write are all facts about the substrate, not about
    the state machine. A backend that starts nothing implements them as no-ops,
    which is precisely why a run with an injected backend is exempt from all
    three -- it starts no clusters, spends nothing, and writes nowhere but a
    temporary directory. That exemption is what lets the state-machine suite
    run on a credential-less machine.
    """
    backend.validate_storage(storage)
    backend.preflight(tiles=n_tiles)


def _tracks(
    jobs: Sequence[ProcessingJob],
    *,
    run_id: str,
    storage: StorageBackend,
    units: int,
    clock: Clock,
) -> list[TileTrack]:
    return [
        TileTrack(
            tile=job.tile.name,
            run_id=run_id,
            root=shards.shard_root(run_id, job.tile.name),
            storage=storage,
            units=units,
            clock=clock,
            job=job,
        )
        for job in jobs
    ]


def drive_fleet(
    jobs: Sequence[ProcessingJob],
    *,
    run_id: str | None = None,
    storage: StorageBackend | None = None,
    backend: FleetBackend | None = None,
    clock: Clock | None = None,
    max_vms: int | None = None,
    wave_window_s: float | None = None,
    probe_waves: bool = False,
) -> FleetSummary:
    """Drive many tiles through consolidated waves. See ADR-018.

    Args:
        jobs: The tiles to build. Their windows may differ; each tile's is
            written to the manifest and read back on the VM.
        run_id: Run token, shared by every tile. Generated when omitted. Print
            it: it is all :func:`resume_fleet` needs.
        storage: Backend the shards publish to. Defaults to the configured one.
        backend: What starts the work. Defaults to
            :class:`~landsat_lst.fleet_backend.CoiledFleetBackend`. Any
            implementation of :class:`~landsat_lst.fleet_backend.FleetBackend`
            that declares the whole contract is a first-class alternative --
            AWS Batch and ECS are the ones worth evaluating.
        max_vms: Concurrency cap. Defaults to ``settings.fleet_max_vms``.
        wave_window_s: Batching window. Defaults to
            ``settings.fleet_wave_window_s``.

    Returns:
        The :class:`FleetSummary`.

    Raises:
        ShardBackendMismatch: If the configured storage is not S3 while Coiled
            work is about to be submitted.
        ValueError: If the backend does not declare the whole contract.
        FleetAborted: On a terminal control-plane failure.
    """
    storage = storage or get_storage()
    backend = backend or CoiledFleetBackend()
    _gates(backend, storage, len(jobs))
    run_id = run_id or fleet_run_id()
    units = shards.offsets_fleet_units()
    write_manifest(storage, run_id, jobs, units=units)
    clock = clock or Clock()
    return FleetDriver(
        run_id=run_id,
        tracks=_tracks(jobs, run_id=run_id, storage=storage, units=units, clock=clock),
        storage=storage,
        backend=backend,
        clock=clock,
        units=units,
        max_vms=max_vms,
        wave_window_s=wave_window_s,
        probe_waves=probe_waves,
    ).run()


def resume_fleet(
    run_id: str,
    *,
    storage: StorageBackend | None = None,
    backend: FleetBackend | None = None,
    clock: Clock | None = None,
    max_vms: int | None = None,
    wave_window_s: float | None = None,
    probe_waves: bool = False,
) -> FleetSummary:
    """Continue a killed fleet driver, entirely from the bucket.

    The roster comes from the manifest and every tile's position comes from a
    listing, so a resume submits only the units that are still missing --
    including none at all, for a tile that finished while the driver was gone.

    Raises:
        FileNotFoundError: If the run published no manifest.
    """
    storage = storage or get_storage()
    backend = backend or CoiledFleetBackend()
    jobs, units = read_manifest(storage, run_id)
    _gates(backend, storage, len(jobs))
    clock = clock or Clock()
    log.info("fleet_resume", run_id=run_id, tiles=len(jobs), units=units)
    return FleetDriver(
        run_id=run_id,
        tracks=_tracks(jobs, run_id=run_id, storage=storage, units=units, clock=clock),
        storage=storage,
        backend=backend,
        clock=clock,
        units=units,
        max_vms=max_vms,
        wave_window_s=wave_window_s,
        probe_waves=probe_waves,
    ).run()
