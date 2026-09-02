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
#:     sooner, never conclude one.
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
    for a human reading a log, and the worker count it actually got -- which
    the driver counts against the run's cap.
    """

    id: object
    name: str
    max_workers: int


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
        """A stable, unique display name for one wave."""
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
        from landsat_lst.batch import fleet_cluster_name  # noqa: PLC0415

        return fleet_cluster_name(run_id, stage, wave)

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
