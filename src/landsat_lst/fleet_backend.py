"""The backend abstraction between the state machine and what starts work.

:mod:`landsat_lst.fleet_driver` decides *what* should run and *when*: which
tiles want which units, how many VMs the run is allowed, when a barrier has
waited long enough. None of that is about Coiled. This module is the boundary
where it becomes about Coiled -- or, later, about AWS Batch, ECS, or a plain
EC2 Spot fleet.

The rule is one-directional: the driver may depend on :class:`FleetBackend`,
and a backend may depend on anything it likes. Nothing Coiled-shaped travels
back across this boundary -- not a cluster id, not a cluster name, not a
``ServerError``. A wave gets a :class:`WaveHandle`, which is an opaque id and a
display name, and the only questions the driver asks about it are "is this
thing dead" and "was that failure worth retrying", both of which the backend
answers.

Why bother, when there is exactly one backend today: the boot amortization
this whole design buys (ADR-018) is a property of the *submission substrate*,
not of the pipeline. Whether AWS Batch or an ECS service does it cheaper is a
question worth being able to ask without rewriting the state machine, and the
honest way to keep that option open is to write down what the state machine
actually needs. That is :data:`BACKEND_CONTRACT`.

The contract is declared, not inferred. Every backend publishes a
``guarantees`` set, and :class:`~landsat_lst.fleet_driver.FleetDriver` refuses
one that does not cover the contract. An evaluation of AWS Batch therefore has
a checklist rather than a reading exercise: name each guarantee, say how the
substrate provides it, and if one cannot be provided, that is the finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: What the fleet state machine requires of any submission substrate. Each id
#: is one property the driver's correctness or its economics rests on; the
#: prose is the whole of what an alternative backend has to satisfy.
#:
#: ``queues_surplus``
#:     Given ``len(units) > max_workers``, the substrate runs *every* unit on
#:     at most ``max_workers`` concurrent workers, reusing a worker for the
#:     next unit when one finishes. This is the entire boot saving: without it
#:     a wave is just a differently-spelled per-tile submission, because each
#:     unit would pay its own start-up. A substrate that instead refuses, or
#:     silently drops the surplus, fails the contract rather than merely
#:     performing worse.
#: ``fire_and_forget``
#:     ``submit`` returns promptly with a handle and does not block until the
#:     work finishes. The driver sequences stages by polling storage and must
#:     stay free to step every other tile.
#: ``at_least_once``
#:     A unit may run more than once -- retried, preempted and restarted,
#:     duplicated by a resubmission. Units are idempotent at their artifact
#:     keys, so this is permitted rather than merely tolerated. What the
#:     substrate must *not* do is promise at-most-once and silently skip a
#:     unit without the driver being able to notice, which it does by listing
#:     the artifact rather than by trusting an exit code.
#: ``no_dependencies_needed``
#:     The substrate is not asked to order anything. Stage ordering is the
#:     driver's poll loop (ADR-010, ADR-016), so a substrate with a DAG
#:     feature is fine and its DAG feature is unused.
#: ``unique_wave_names``
#:     ``wave_name`` is unique per ``(run_id, stage, wave)`` and stable, so two
#:     drivers computing the same wave agree, and a resumed driver does not
#:     rebuild the name of something still running.
#: ``opaque_handle``
#:     The handle id is JSON-serializable and stable for the life of the wave.
#:     It is persisted in the wave record and handed back to ``probe`` by a
#:     later process, possibly on another machine.
#: ``probe_is_advisory``
#:     ``probe`` may say "dead" or "unknown". It may never assert success --
#:     completion is bytes in the bucket -- so it can only ever end a barrier
#:     sooner, never conclude one. Subsumed by ``census`` where a census can be
#:     taken: an identity absent from an authoritative census is dead, which is
#:     the same answer for the whole run in one query. It stays in the contract
#:     because the census may be *unavailable*, and then a per-wave probe is
#:     the only thing left that speaks about machines.
#: ``enumerable_by_run``
#:     Every billing resource a run creates is discoverable from ``run_id``
#:     alone -- **including one created by a call whose response was lost.**
#:     This is the guarantee whose absence causes all three divergence windows.
#:     It is what makes ``submission_identity`` load-bearing rather than
#:     cosmetic: the identity is a pure function of the request, fixed before
#:     the call, so it survives the answer going missing. On a substrate whose
#:     submission is a non-atomic two-step with no idempotency key -- which is
#:     what ``coiled.batch_run`` is -- listing by that identity is the *only*
#:     recovery path for VMs that are billing under an id nobody holds.
#: ``census_is_authoritative``
#:     ``census`` reflects the substrate's own record of what it is billing,
#:     not an inference from work products. It may be **stale** and it may be
#:     **unavailable** (``None``); it may not be **wrong by construction**. The
#:     distinction is the whole redesign: bytes in a bucket are evidence that
#:     work *completed*, and carry no function of whether a VM *exists*.
#:     Unavailable is never zero -- a backend that cannot answer says ``None``
#:     and the driver runs a degraded policy that says so in the log.
#: ``classified_failures``
#:     Control-plane errors map to ``"terminal"`` or ``"transient"``, and an
#:     *unrecognized* error maps to transient. Guessing terminal for the
#:     unknown case turns every ordinary blip into a dead run.
#: ``no_silent_cost_substitution``
#:     However the substrate expresses a cheap-capacity preference, it must not
#:     silently substitute the expensive class. This is the one failure mode
#:     that converts a spot-priced build into an on-demand bill with nobody
#:     deciding it.
BACKEND_CONTRACT: frozenset[str] = frozenset(
    {
        "queues_surplus",
        "fire_and_forget",
        "at_least_once",
        "no_dependencies_needed",
        "unique_wave_names",
        "opaque_handle",
        "probe_is_advisory",
        "enumerable_by_run",
        "census_is_authoritative",
        "classified_failures",
        "no_silent_cost_substitution",
    }
)


@dataclass(frozen=True)
class WaveRequest:
    """One wave, in backend-neutral terms.

    ``units`` are ``(tile, index)`` pairs and nothing else: the backend turns
    them into whatever its substrate maps over. ``max_workers`` is a
    concurrency *ceiling*, never a unit count -- see ``queues_surplus``.
    """

    stage: str
    run_id: str
    units: tuple[tuple[str, int], ...]
    wave: int
    max_workers: int
    #: The fused offsets fleet's width, which the planner needs and which the
    #: driver fixed before any plan existed. Opaque to the backend; it only
    #: has to reach the task.
    fleet_units: int | None = None

    @property
    def tiles(self) -> list[str]:
        seen: dict[str, None] = {}
        for tile, _ in self.units:
            seen.setdefault(tile, None)
        return list(seen)


@dataclass(frozen=True)
class WaveHandle:
    """What the driver keeps after a wave is started.

    Deliberately thin. An id it can persist and hand back to ``probe``, a name
    for a human reading a log, and the worker count it actually got.

    The width is **not** what the driver charges. A handle can only ever report
    a width the substrate clamped *downward* (``coiled.batch_run`` clamps to the
    unit count), and a submission attempt whose answer was lost reports no width
    at all while its VMs bill at the width that was asked for. So the charge is
    the requested width and this is a record of what came back.

    Deliberately still three fields. A substrate may hand out more ids than
    this -- Coiled creates the batch job with one request and the cluster with
    another -- and none of them belongs here, because none of them is a
    *recovery* key: an id that arrives in a reply is exactly the thing a lost
    reply does not carry. What recovers a resource is
    :meth:`FleetBackend.submission_identity`, which is fixed before the call.
    """

    id: object
    name: str
    max_workers: int


@dataclass(frozen=True)
class WorkerCensus:
    """What the substrate says it is billing for one run, right now.

    Not a count the driver derived; a count the substrate reported. The
    difference is the point: ``total`` is a function of worker *existence*,
    where every listing of work products the driver reads is a function of work
    *completion*, and the two coincide only by luck.

    ``identities`` carries every resource found for the run, including any whose
    creating call lost its answer -- that is what ``enumerable_by_run`` buys, and
    it is why an identity must be computable before the call rather than read
    out of its reply.

    Being *stale* is permitted and expected: it is a snapshot, at
    poll resolution, of something that moves. Being absent is expressed as
    ``None`` in place of a census, never as a census reporting zero.
    """

    #: Wall clock at which the substrate was asked. A charge is discharged only
    #: by a census taken *after* the attempt that raised it, so a reader has to
    #: be able to order the two.
    as_of: float
    #: Workers this run is billing for, run-wide, across every identity.
    total: int
    #: ``submission_identity -> workers``, so a driver can tell which of its own
    #: submissions a worker belongs to, and see the ones belonging to none.
    by_identity: dict[str, int]
    #: Every resource found for this run, whether or not the driver knows it.
    identities: frozenset[str]

    @property
    def unattributed(self) -> int:
        """Workers found for the run that no listed identity accounts for.

        Never negative, and normally zero. A positive value is the substrate
        reporting capacity the ``by_identity`` breakdown cannot place, which is
        exactly the shape of an orphan.
        """
        return max(0, self.total - sum(self.by_identity.values()))


@runtime_checkable
class FleetBackend(Protocol):
    """Everything the fleet state machine is allowed to know about a substrate.

    See :data:`BACKEND_CONTRACT` for the guarantees an implementation is
    promising by declaring them in :attr:`guarantees`.
    """

    #: Short name, for logs.
    name: str
    #: Which contract ids this backend claims. The driver refuses a backend
    #: that does not cover :data:`BACKEND_CONTRACT`, so an incomplete
    #: implementation fails at construction rather than halfway through a run.
    guarantees: frozenset[str]

    def wave_name(self, run_id: str, stage: str, wave: int) -> str:
        """A stable, unique display name for one wave.

        The older spelling of :meth:`submission_identity`, kept because it is
        what a backend written before the census contract implements. A driver
        prefers ``submission_identity`` and falls back to this.
        """
        ...

    def submission_identity(self, run_id: str, stage: str, wave: int) -> str:
        """The name this submission will be discoverable under. See ``enumerable_by_run``.

        Three requirements, and each of them is load-bearing:

        - a **pure function of the request**, so two drivers agree and a resumed
          one rebuilds it rather than remembering it;
        - computed **before** the call, so it survives the call's answer being
          lost -- which is the whole recovery path for a submission that started
          VMs and returned nothing;
        - **discoverable by listing**, so :meth:`census` can find the resource
          from ``run_id`` alone.

        An id that comes back *in* the reply satisfies none of these, which is
        why the driver never uses one as an identity.
        """
        ...

    def census(self, run_id: str) -> WorkerCensus | None:
        """Every worker the substrate bills for this run, from ``run_id`` alone.

        Includes resources whose creating call lost its response, which is the
        one thing no other channel can see.

        Returns:
            The :class:`WorkerCensus`, or ``None`` when the substrate **cannot
            be asked** -- no credentials, control plane down, listing refused.
            ``None`` means "cannot answer" and never "nothing is running": a
            driver that read it as zero would offer the whole cap as headroom at
            exactly the moment it had lost sight of the bill.
        """
        ...

    def reap(self, run_id: str, identity: str) -> None:
        """Ask the substrate to terminate one identity. Advisory, and idempotent.

        A request, not a confirmation. The substrate may acknowledge a delete
        and go on billing for the teardown, so the only evidence that a reap
        worked is a later :meth:`census` that omits the identity. Being
        idempotent, it is safe to repeat every poll until one does.
        """
        ...

    def submit(self, request: WaveRequest) -> WaveHandle:
        """Start one wave and return promptly."""
        ...

    def classify_failure(self, error: BaseException) -> str:
        """``"terminal"`` if retrying cannot help, else ``"transient"``."""
        ...

    def probe(self, handle_id: object) -> tuple[str, str] | None:
        """``(state, reason)`` for one wave, or ``None`` when nothing is known."""
        ...

    def preflight(self, *, tiles: int) -> None:
        """Refuse a run the substrate cannot start or cannot pay for."""
        ...

    def validate_storage(self, storage: StorageBackend) -> None:
        """Refuse a storage backend the substrate's workers cannot write to."""
        ...


def check_contract(backend: FleetBackend) -> None:
    """Refuse a backend that has not declared the whole contract.

    Cheap, and it fails at the right moment. A backend missing
    ``queues_surplus`` is not a slower fleet; it is a fleet whose entire reason
    to exist is absent, and discovering that from a bill is the expensive way.

    Raises:
        ValueError: If any contract id is undeclared.
    """
    missing = BACKEND_CONTRACT - set(getattr(backend, "guarantees", frozenset()))
    if missing:
        msg = (
            f"fleet backend {getattr(backend, 'name', backend)!r} does not declare "
            f"{len(missing)} required guarantee(s): {', '.join(sorted(missing))}. "
            "See landsat_lst.fleet_backend.BACKEND_CONTRACT."
        )
        raise ValueError(msg)


@dataclass
class CoiledFleetBackend:
    """Coiled Batch behind the backend abstraction. Everything here is Coiled-specific.

    Four quirks live in this class and nowhere else, and each is a lesson the
    project already paid for:

    - **A name that collides with a running cluster is refused outright.**
      Observed as ``Unable to add batch jobs to existing cluster``, which is
      why the wave number is in the name and the run id is hashed rather than
      spelled (it would be truncated away). Other substrates may not care;
      the contract asks only for uniqueness.
    - **An error with no message at all is ordinary.** An empty ``ServerError``
      from a cluster create -- the credit quota, as it turned out -- killed the
      driver outright on 2026-08-22. So unknown maps to transient, and only the
      named markers map to terminal.
    - **Cost is credits, and identity is AWS SSO.** The preflight is two gates
      in that order, because a session that cannot call STS cannot read a
      Coiled balance either.
    - **Workers always write S3.** A driver polling a local directory would
      wait forever on a barrier whose artifacts are in a bucket.

    The surplus-queuing guarantee is ``coiled.batch_run``'s own: it starts
    ``max_workers`` VMs and hands them ``map_over_values`` in order.
    """

    name: str = "coiled"
    guarantees: frozenset[str] = field(default_factory=lambda: frozenset(BACKEND_CONTRACT))
    #: Injectable for the credit gate, so the decision is testable without a
    #: control plane -- resolved at call time rather than bound at definition,
    #: which is how a unit test once reached the real billing API.
    balance_source: Any = None

    def wave_name(self, run_id: str, stage: str, wave: int) -> str:
        return self.submission_identity(run_id, stage, wave)

    def submission_identity(self, run_id: str, stage: str, wave: int) -> str:
        """The cluster name, which Coiled already makes a pure function of the request.

        Nothing new is invented here. ``fleet_cluster_name`` hashes the run id
        and spells the stage and wave, ``batch_run`` is handed it as ``name=``,
        and the tags carry ``run_id`` besides -- all of it fixed before the call.
        That is precisely why a cluster whose creation answer was lost is still
        findable: the key was never in the reply.
        """
        from landsat_lst.batch import fleet_cluster_name  # noqa: PLC0415

        return fleet_cluster_name(run_id, stage, wave)

    def census(self, run_id: str) -> WorkerCensus | None:
        from landsat_lst.batch import fleet_worker_census  # noqa: PLC0415

        return fleet_worker_census(run_id)

    def reap(self, run_id: str, identity: str) -> None:
        from landsat_lst.batch import fleet_reap_identity  # noqa: PLC0415

        fleet_reap_identity(run_id, identity)

    def submit(self, request: WaveRequest) -> WaveHandle:
        from landsat_lst.batch import submit_fleet_stage  # noqa: PLC0415

        submission = submit_fleet_stage(
            stage=request.stage,
            run_id=request.run_id,
            units=list(request.units),
            wave=request.wave,
            max_workers=request.max_workers,
            fleet_units=request.fleet_units,
        )
        return WaveHandle(
            id=submission.cluster_id,
            name=submission.name,
            max_workers=submission.max_workers,
        )

    def classify_failure(self, error: BaseException) -> str:
        # The single-tile driver owns this mapping and has been corrected by
        # two incidents; a second copy here would drift away from both.
        from landsat_lst.shard_driver import classify_failure  # noqa: PLC0415

        return classify_failure(error)

    def probe(self, handle_id: object) -> tuple[str, str] | None:
        from landsat_lst.shard_driver import coiled_cluster_probe  # noqa: PLC0415

        return coiled_cluster_probe(handle_id)

    def preflight(self, *, tiles: int) -> None:
        """Identity, then credits, priced for many tiles.

        The estimate is the per-tile one multiplied by the tile count, which
        ignores the boot amortization the consolidation buys and therefore
        over-estimates -- the safe direction for a gate whose job is to refuse
        a run the workspace cannot pay for.
        """
        from landsat_lst import quota  # noqa: PLC0415

        quota.preflight_identity()
        estimate = quota.estimate_run_credits() * max(1, tiles)
        balance = quota.preflight_credits(estimate, balance_source=self.balance_source)
        log.info(
            "fleet_quota_preflight",
            tiles=tiles,
            estimate=round(estimate, 1),
            remaining=balance.remaining,
        )

    def validate_storage(self, storage: StorageBackend) -> None:
        from landsat_lst.batch import submit_fleet_stage  # noqa: PLC0415
        from landsat_lst.shard_driver import require_shared_storage  # noqa: PLC0415

        require_shared_storage(storage, submit_fleet_stage)


def units_for(request: WaveRequest) -> Sequence[str]:
    """The task-input values one wave maps over.

    Here rather than in a backend because the *token* grammar is the driver's
    (``shards.fleet_unit_token``) and is read back by ``landsat-lst shard
    unit`` on the worker. What is backend-specific is how the value reaches
    the process -- an environment variable named by the substrate -- not what
    the value says.
    """
    from landsat_lst import shards  # noqa: PLC0415

    return [shards.fleet_unit_token(tile, index) for tile, index in request.units]


@dataclass
class InMemoryFleetBackend:
    """The whole contract, satisfied by a dict. Starts nothing, bills nothing.

    Here rather than in a test module because the census contract is only worth
    anything if a second implementation of it exists: a protocol with one
    implementation is a description of that implementation. This one is also
    the answer to a rule the project already keeps -- a state machine must be
    testable on a credential-less machine -- and nothing on the import path of
    this class reaches ``coiled`` or ``boto3``.

    The census is modelled the way a substrate's is, not the way a driver would
    like it to be:

    - a submission opens workers under the identity that was computed *before*
      it, so :meth:`lose_next_answer` can simulate the window that has no
      recovery except listing: the workers exist, the caller holds no handle,
      and only ``run_id`` finds them;
    - :meth:`reap` is a request. It stops the workers, but a caller learns that
      from the next census rather than from the return value;
    - :attr:`answerable` off makes :meth:`census` return ``None``, which is the
      "cannot answer" case a driver has to treat as unknown rather than empty.
    """

    name: str = "in-memory"
    guarantees: frozenset[str] = field(default_factory=lambda: frozenset(BACKEND_CONTRACT))
    #: Wall clock, injectable so a test does not sleep.
    clock: Any = None
    #: Whether the substrate can be asked at all. Off models a control plane
    #: that is down or credentials that are absent -- the degraded case.
    answerable: bool = True
    #: ``run_id -> identity -> live workers``.
    workers: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Every request received, in order, for a test to assert against.
    submissions: list[WaveRequest] = field(default_factory=list)
    #: Identities asked to terminate, in order. Repeats are expected.
    reaped: list[str] = field(default_factory=list)
    #: When set, the next submit starts its workers and then raises, which is
    #: the lost-acknowledgement window in one line.
    _lose_answer: bool = False
    _next_id: int = 0

    def lose_next_answer(self) -> None:
        """Make the next submission start workers and return no handle."""
        self._lose_answer = True

    def _now(self) -> float:
        if self.clock is not None:
            return float(self.clock.now())
        import time  # noqa: PLC0415

        return time.time()

    # -- FleetBackend -----------------------------------------------------

    def wave_name(self, run_id: str, stage: str, wave: int) -> str:
        return self.submission_identity(run_id, stage, wave)

    def submission_identity(self, run_id: str, stage: str, wave: int) -> str:
        return f"mem-{run_id}-{stage}-w{wave}"

    def census(self, run_id: str) -> WorkerCensus | None:
        if not self.answerable:
            return None
        found = {
            identity: count for identity, count in self.workers.get(run_id, {}).items() if count > 0
        }
        return WorkerCensus(
            as_of=self._now(),
            total=sum(found.values()),
            by_identity=dict(found),
            identities=frozenset(found),
        )

    def reap(self, run_id: str, identity: str) -> None:
        self.reaped.append(identity)
        self.workers.get(run_id, {}).pop(identity, None)

    def submit(self, request: WaveRequest) -> WaveHandle:
        self.submissions.append(request)
        identity = self.submission_identity(request.run_id, request.stage, request.wave)
        width = max(1, int(request.max_workers))
        # Workers first, handle second: that ordering is the substrate's, and
        # reversing it here would hide the only window worth simulating.
        run = self.workers.setdefault(request.run_id, {})
        run[identity] = run.get(identity, 0) + width
        if self._lose_answer:
            self._lose_answer = False
            msg = f"lost the answer for {identity!r}"
            raise ConnectionError(msg)
        self._next_id += 1
        return WaveHandle(id=self._next_id, name=identity, max_workers=width)

    def classify_failure(self, error: BaseException) -> str:
        del error
        return "transient"

    def probe(self, handle_id: object) -> tuple[str, str] | None:
        del handle_id
        return None

    def preflight(self, *, tiles: int) -> None:
        del tiles

    def validate_storage(self, storage: StorageBackend) -> None:
        del storage
