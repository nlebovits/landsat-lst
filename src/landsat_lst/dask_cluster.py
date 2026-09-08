"""The only module that imports coiled, distributed, and frisky.

It builds the Coiled cluster the futures driver runs on, hands the driver a
:class:`ClusterSession` that satisfies :class:`landsat_lst.futures_driver.Executor`,
and takes the cluster down again with the shutdown *confirmed* rather than
assumed. Everything that is Coiled-shaped, Frisky-shaped, or
distributed-shaped is here, so the driver and its tests never see any of it,
the same discipline :mod:`landsat_lst.fleet_backend` keeps for Batch.

Decisions that live here, with their reasons:

- **One VM type, ``settings.shard_composite_vm_type``, for every stage.** It
  is already the primary type of the Batch offsets fleet; the m6i.4xlarge
  fallback bills twice per vCPU-hour (#156). One type means a spot shortage
  shows as a wait, never as a silent 16-vCPU upgrade.
- **``worker_options={"nthreads": 1}``.** One shard per VM at a time. The
  Frisky worker plugin honours the Dask worker's thread count, so this holds
  under both outer schedulers. Inner parallelism is the shard's own scheduler.
- **No software environment argument.** Coiled package-syncs the local venv,
  which is how the ``frisky`` extra reaches the workers; the preflight refuses
  a local environment without it when the outer scheduler is Frisky.
- **Shutdown is confirmed through the control plane**, polled until the
  cluster reports stopped or ``futures_cleanup_timeout_s`` passes, and the
  result is written to ``state/cleanup.json``. A driver that dies leaves a
  cluster the idle timeout reaps; ``landsat-lst shard stop`` shuts it by name.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from landsat_lst import quota, shards
from landsat_lst.config import settings
from landsat_lst.shard_driver import coiled_cluster_probe

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from landsat_lst.futures_driver import Deadline, Future
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

_CLUSTER_NAME_MAX = 60


def futures_cluster_name(run_id: str, tile: str) -> str:
    """``lst-{run8}-{tile}-fut``, hashed so truncation cannot eat the marker."""
    run8 = hashlib.sha256(run_id.encode()).hexdigest()[:8]
    return f"lst-{run8}-{tile}-fut"[:_CLUSTER_NAME_MAX]


def cluster_kwargs(
    *,
    run_id: str,
    tile: str,
    n_workers: int,
    vm_type: str | None = None,
    environ: Mapping[str, str] | None = None,
    scheduler: str = "frisky",
) -> dict[str, Any]:
    """The exact ``coiled.Cluster`` arguments, as a pure function.

    Everything a test wants to pin is here and nothing here touches a control
    plane. ``environ`` defaults to :func:`landsat_lst.job._worker_environ`,
    which freezes AWS credentials and forwards ``LST_*`` overrides; the
    scheduler settings ride along as ``DASK_*`` variables the workers read.
    """
    if environ is None:
        from landsat_lst.job import _worker_environ  # noqa: PLC0415

        environ = _worker_environ()
    env = {
        **environ,
        "DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES": str(settings.futures_worker_losses),
        "DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL": settings.futures_worker_ttl,
        "FRISKY_TRACING_CAPACITY": str(settings.frisky_tracing_capacity),
        "LST_INNER_SCHEDULER": settings.inner_scheduler,
        "LST_LOAD_CHUNK_SIZE": str(settings.shard_composite_chunk),
    }
    return {
        "name": futures_cluster_name(run_id, tile),
        "n_workers": int(n_workers),
        "worker_vm_types": [vm_type or settings.shard_composite_vm_type],
        "worker_options": {"nthreads": 1},
        "region": settings.coiled_region,
        "spot_policy": settings.shard_spot_policy,
        "environ": env,
        "tags": {
            "project": "landsat-lst",
            "run_id": run_id,
            "tile": tile,
            "executor": "futures",
            "scheduler": scheduler,
        },
        "idle_timeout": settings.futures_idle_timeout,
        "scheduler_options": {"idle_timeout": settings.futures_idle_timeout},
        "shutdown_on_close": True,
        "wait_for_workers": False,
    }


def preflight(
    *,
    balance_source: Callable[[], quota.CreditBalance] | None = None,
    plan: Any = None,
    units: int | None = None,
    credit_cap: float | None = None,
    scheduler: str = "frisky",
) -> tuple[float, quota.CreditBalance]:
    """Identity, then write access, then credits, then the operator's cap.

    Same order as the Batch driver and for the same reasons (ADR-016). The
    cap is the operator's number for *this run*; the estimate must fit under
    it before anything boots. Frisky must be importable locally when it is the
    outer scheduler, because package sync ships what this environment has.
    """
    if scheduler == "frisky":
        try:
            import frisky  # noqa: F401, PLC0415
        except ImportError as exc:
            msg = "the frisky extra is not installed locally; workers are package-synced from here"
            raise RuntimeError(msg) from exc
    quota.preflight_identity()
    quota.preflight_write_access()
    estimate = quota.estimate_run_credits(plan, units=units)
    if credit_cap is not None and estimate * settings.coiled_credit_safety > credit_cap:
        raise quota.QuotaRefused(
            f"estimated {estimate:.1f} credits x safety {settings.coiled_credit_safety} exceeds "
            f"the run's cap of {credit_cap:.1f}",
            estimate=estimate,
            remaining=credit_cap,
        )
    balance = quota.preflight_credits(estimate, balance_source=balance_source)
    return estimate, balance


@dataclass
class ClusterSession:
    """A live cluster and its client, in the shape the driver expects."""

    scheduler: str
    dashboard_url: str | None
    cluster_id: object
    cluster_name: str
    vm_type: str
    n_workers: int
    client: Any
    dask_client: Any
    cluster: Any
    cap: int
    closed_at: str | None = None
    cleanup: dict[str, Any] = field(default_factory=dict)

    # -- Executor ------------------------------------------------------------------------

    def submit(
        self, fn: Callable[..., Any], *args: Any, key: str, retries: int, **kw: Any
    ) -> Future:
        if self.scheduler == "frisky":
            return self.client.submit(fn, *args, key=key, retries=retries, **kw)
        return self.client.submit(fn, *args, key=key, retries=retries, pure=False, **kw)

    def as_completed(
        self, futures: Sequence[Future], *, timeout_s: float | None
    ) -> Iterator[Future]:
        if not futures:
            return iter(())
        if self.scheduler == "frisky":
            import frisky  # noqa: PLC0415

            return frisky.as_completed(list(futures), raise_errors=False, timeout=timeout_s)
        import distributed  # noqa: PLC0415

        return distributed.as_completed(list(futures), raise_errors=False, timeout=timeout_s)

    def ensure_workers(self, n: int, *, timeout_s: float) -> None:
        """Scale to ``n`` (never above the cap) and wait for them, bounded."""
        target = max(1, min(int(n), self.cap))
        if target != self.n_workers:
            log.info("futures_cluster_scale", from_workers=self.n_workers, to_workers=target)
            self.cluster.scale(target)
            self.n_workers = target
        self.dask_client.wait_for_workers(target, timeout=max(1.0, timeout_s))

    def worker_addresses(self) -> list[str]:
        try:
            info = self.dask_client.scheduler_info()
            return sorted(str(a) for a in (info.get("workers") or {}))
        except Exception:
            return []

    # -- state for the observer -------------------------------------------------------------

    def scheduler_state(self) -> dict[str, Any] | None:
        try:
            if self.scheduler == "frisky":
                return dict(self.client.get_scheduler_state())
            return {"scheduler_info": self.dask_client.scheduler_info()}
        except Exception as exc:
            log.warning("futures_scheduler_state_failed", error=str(exc))
            return None

    def future_status(self, futures: Sequence[Future]) -> dict[str, str]:
        return {f.key: str(getattr(f, "status", "")) for f in futures}

    def processing(self) -> dict[str, list[str]]:
        try:
            return {
                str(w): [str(k) for k in keys] for w, keys in self.dask_client.processing().items()
            }
        except Exception:
            return {}

    def dask_task_states(self) -> dict[str, str]:
        """Every task's scheduler state by key, on plain distributed."""
        try:
            return dict(
                self.dask_client.run_on_scheduler(
                    lambda dask_scheduler: {
                        str(k): str(ts.state) for k, ts in dask_scheduler.tasks.items()
                    }
                )
            )
        except Exception:
            return {}

    def who_has_all(self) -> dict[str, list[str]]:
        try:
            return {str(k): [str(w) for w in ws] for k, ws in self.dask_client.who_has().items()}
        except Exception:
            return {}

    def who_has(self, futures: Sequence[Future]) -> dict[str, list[str]]:
        try:
            return {
                str(k): [str(w) for w in ws]
                for k, ws in self.dask_client.who_has(list(futures)).items()
            }
        except Exception:
            return {}

    # -- teardown ------------------------------------------------------------------------------

    def close(
        self,
        *,
        storage: StorageBackend | None = None,
        run_id: str | None = None,
        tile: str | None = None,
    ) -> dict[str, Any]:
        """Close the client, shut the cluster down, and confirm it stopped."""
        closed = datetime.now(tz=UTC).isoformat()
        errors: list[str] = []
        workers_at_close = self.worker_addresses()
        for name, obj in (("client", self.client), ("dask_client", self.dask_client)):
            try:
                obj.close()
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}"[:200])
        try:
            self.cluster.shutdown()
        except Exception as exc:
            errors.append(f"shutdown: {type(exc).__name__}: {exc}"[:200])
        confirmed, final_state = _confirm_stopped(
            self.cluster_id, timeout_s=settings.futures_cleanup_timeout_s
        )
        self.closed_at = closed
        self.cleanup = {
            "closed_at": closed,
            "confirmed": confirmed,
            "final_state": final_state,
            "cluster_id": self.cluster_id,
            "cluster_name": self.cluster_name,
            "workers_seen_at_close": workers_at_close,
            "errors": errors,
        }
        if storage is not None and run_id and tile:
            try:
                storage.write_text(
                    f"{shards.shard_root(run_id, tile)}/state/cleanup.json",
                    json.dumps(self.cleanup, indent=2, default=str),
                )
            except Exception as exc:
                log.warning("futures_cleanup_record_failed", error=str(exc))
        log.info(
            "futures_cluster_closed",
            **{k: v for k, v in self.cleanup.items() if k != "workers_seen_at_close"},
        )
        return self.cleanup


def _confirm_stopped(
    cluster_id: object,
    *,
    timeout_s: float,
    probe: Callable[[object], tuple[str, str] | None] = coiled_cluster_probe,
    poll_s: float | None = None,
) -> tuple[bool, str | None]:
    """Poll the control plane until the cluster reports stopped, bounded.

    ``stopping`` is not ``stopped``: the poll continues until the final state
    or the timeout. ``error`` ends it, unconfirmed, with the state recorded.
    """
    if cluster_id is None:
        return False, None
    deadline = time.monotonic() + timeout_s
    interval = min(10.0, max(1.0, timeout_s / 20)) if poll_s is None else poll_s
    last: str | None = None
    while True:
        answer = probe(cluster_id)
        if answer is not None:
            last = answer[0]
            if last.lower() == "stopped":
                return True, last
            if last.lower() == "error":
                return False, last
        if time.monotonic() >= deadline:
            return False, last
        time.sleep(interval)


def stop_cluster(name: str) -> dict[str, Any]:
    """Shut a futures cluster down by name, for a driver that died."""
    import coiled  # noqa: PLC0415

    found = None
    for record in coiled.list_clusters(just_mine=False):
        if str(record.get("name")) == name:
            found = record
            break
    if found is None:
        return {"name": name, "found": False}
    cluster_id = found.get("id")
    coiled.Cluster(name=name, shutdown_on_close=True).shutdown()  # ty: ignore[invalid-argument-type, no-matching-overload]
    confirmed, state = _confirm_stopped(cluster_id, timeout_s=settings.futures_cleanup_timeout_s)
    return {
        "name": name,
        "found": True,
        "cluster_id": cluster_id,
        "confirmed": confirmed,
        "final_state": state,
    }


@contextmanager
def cluster_session(
    *,
    run_id: str,
    tile: str,
    n_workers: int,
    deadline: Deadline,
    scheduler: str = "frisky",
    vm_type: str | None = None,
    storage: StorageBackend | None = None,
) -> Iterator[ClusterSession]:
    """Boot the cluster, wait for its first workers under the deadline, yield.

    The caller runs the preflight gates before this, so nothing here spends
    credits before identity, write access, and the cap were checked. On exit,
    success or not, the cluster is shut down and its stop confirmed.
    """
    import coiled  # noqa: PLC0415

    cap = settings.futures_max_workers
    workers = max(1, min(int(n_workers), cap))
    kwargs = cluster_kwargs(
        run_id=run_id, tile=tile, n_workers=workers, vm_type=vm_type, scheduler=scheduler
    )
    log.info(
        "futures_cluster_starting",
        name=kwargs["name"],
        n_workers=workers,
        vm_type=kwargs["worker_vm_types"][0],
        deadline_min=round(deadline.total_s / 60, 1),
    )
    deadline.check("creating the cluster")
    # kwargs is a plain dict pinned by tests; coiled's signature is typed per argument.
    cluster = coiled.Cluster(**kwargs)  # ty: ignore[invalid-argument-type]
    session: ClusterSession | None = None
    try:
        dask_client = cluster.get_client()
        deadline.check("waiting for the first workers")
        dask_client.wait_for_workers(
            workers, timeout=min(settings.futures_boot_timeout_s, deadline.remaining_s())
        )
        deadline.check("waiting for the first workers")
        client = dask_client
        dashboard = getattr(dask_client, "dashboard_link", None)
        if scheduler == "frisky":
            import frisky  # noqa: PLC0415

            client = frisky.hijack(dask_client)
            dashboard = getattr(client, "dashboard_link", dashboard)
            client.get_scheduler_state()  # liveness: the hijack must answer before any submit
            log.info("frisky_active", dashboard=dashboard)
        session = ClusterSession(
            scheduler=scheduler,
            dashboard_url=dashboard,
            cluster_id=getattr(cluster, "cluster_id", None),
            cluster_name=kwargs["name"],
            vm_type=kwargs["worker_vm_types"][0],
            n_workers=workers,
            client=client,
            dask_client=dask_client,
            cluster=cluster,
            cap=cap,
        )
        yield session
    finally:
        if session is not None:
            session.close(storage=storage, run_id=run_id, tile=tile)
        else:
            try:
                cluster.shutdown()
            except Exception as exc:
                log.warning("futures_cluster_shutdown_failed", error=str(exc))


def credit_stop(
    *, cap: float, balance_source: Callable[[], quota.CreditBalance] | None = None
) -> Callable[[], str | None]:
    """A best-effort spending stop from the workspace balance.

    Reads ``spent`` at construction and again on every call; when the
    drawdown since the start exceeds the cap, returns a reason. Coiled usage
    lags billing and other jobs move the same balance, so this can only
    detect an overrun after the fact. The worker count and the deadline are
    the guaranteed limits.
    """
    source = balance_source or quota.read_balance
    try:
        start_spent = source().spent
    except Exception:
        start_spent = None

    def check() -> str | None:
        if start_spent is None:
            return None
        try:
            spent = source().spent
        except Exception:
            return None
        if spent is None:
            return None
        drawdown = spent - start_spent
        if drawdown > cap:
            return f"balance drawdown {drawdown:.1f} credits exceeds the cap of {cap:.1f} (best-effort stop)"
        return None

    return check


def frisky_span_query() -> Callable[..., list[dict[str, Any]]] | None:
    try:
        import frisky  # noqa: PLC0415
    except ImportError:
        return None
    return frisky.query_spans


def frisky_story(dashboard_url: str | None) -> Callable[[str], Mapping[str, Any] | None] | None:
    if dashboard_url is None:
        return None
    try:
        import frisky  # noqa: PLC0415
    except ImportError:
        return None

    def story(key: str) -> Mapping[str, Any] | None:
        try:
            return frisky.story(key, dashboard_url=dashboard_url)
        except Exception:
            return None

    return story


def environ_summary() -> dict[str, str]:
    """What the launch prints so the operator sees the limits before spend."""
    return {
        "vm_type": settings.shard_composite_vm_type,
        "max_workers": str(settings.futures_max_workers),
        "spot_policy": settings.shard_spot_policy,
        "region": settings.coiled_region,
        "idle_timeout": settings.futures_idle_timeout,
        "aws_profile": os.environ.get("AWS_PROFILE", settings.aws_profile or ""),
    }
