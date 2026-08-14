"""Per-run JSON manifests for distributed batch runs.

Each :func:`landsat_lst.batch.reconcile_run` call writes one manifest to
``settings.manifest_dir / f"{run_id}.json"``. The manifest is the durable
record of a paid run: which tiles completed, skipped, or failed (and why),
plus the per-tile duration, scene count, and peak memory that a costed
validation run needs to project the price of the global build. Coiled's own
dashboard forgets; this file does not.

Three things the dashboard never held at all. The **attempt series** says what
each retry of a tile did, so a run that succeeded on the third try can still
explain the two VMs it paid for. The **cost block** turns billed seconds into
dollars, as ranges carrying their provenance rather than as scalars that would
read as measurements. The **plan block** puts the memory floor the run was
submitted expecting next to the peak it reached, which is how a configuration
gets argued with after the fact.

A manifest is written after the run rather than during it. Nothing has to
survive on the submitting machine while tiles are computing, so a closed laptop
costs a manifest that has not been generated yet, never a manifest that was
lost.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from landsat_lst import pricing
from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from landsat_lst.job import JobResult

#: Decimal places for a dollar figure. Four, because a tile costs cents and a
#: fleet projection multiplies one tile by 700: rounding the per-tile figure to
#: the cent would move the fleet total by dollars.
_USD_PLACES = 4


def _range(value: pricing.CostRange) -> dict[str, Any]:
    """One cost interval, with the label that says how much to trust it."""
    return {
        "low": round(value.low, _USD_PLACES),
        "high": round(value.high, _USD_PLACES),
        "provenance": str(value.provenance),
    }


def _tile_cost_block(estimate: pricing.CostEstimate | None) -> dict[str, Any] | None:
    """One tile's price, or ``None`` when nothing priceable was published."""
    if estimate is None:
        return None
    return {
        **_range(estimate.usd),
        "usd_hour": _range(estimate.usd_hour),
        "billed_s": estimate.billed_s,
        "instance_type": estimate.instance_type,
        "lifecycle": str(estimate.lifecycle),
    }


def _fleet_block(fleet: pricing.FleetCost | None) -> dict[str, Any] | None:
    """The projection to a full build, or ``None`` when no tile was priced.

    An absent projection rather than a zero. Zero dollars for 700 tiles is a
    claim, and the claim would be false.
    """
    if fleet is None:
        return None
    return {
        **_range(fleet.usd),
        "tiles": fleet.tiles,
        "observed_tiles": fleet.observed_tiles,
        "mean_usd_per_tile": _range(fleet.mean_usd_per_tile),
    }


def _cost_block(costs: Mapping[str, pricing.CostEstimate]) -> dict[str, Any]:
    """What the run cost, and what a full build would cost at the same rate.

    The run total is :func:`~landsat_lst.pricing.fleet_cost` over exactly the
    tiles priced, so the total and the projection cannot disagree about how
    provenance combines across a run.
    """
    estimates = list(costs.values())
    total = pricing.fleet_cost(estimates, tiles=len(estimates))
    return {
        "currency": "USD",
        "region": settings.coiled_region,
        "priced_tiles": len(estimates),
        "total": _range(total.usd) if total else None,
        "fleet": _fleet_block(pricing.fleet_cost(estimates)),
        "disclaimer": pricing.DISCLAIMER,
    }


def _attempts_block(attempts: Mapping[str, list[dict[str, Any]]]) -> dict[str, int]:
    """How many attempts the run paid for, over how many tiles.

    ``tiles_retried`` counts tiles that were tried more than once, which is the
    number that says whether a run was expensive because tiles are slow or
    because VMs kept dying.
    """
    rows = list(attempts.values())
    return {
        "tiles_retried": sum(1 for r in rows if len(r) > 1),
        "total": sum(len(r) for r in rows),
        "max": max((len(r) for r in rows), default=0),
    }


def _tile_block(
    result: JobResult,
    rows: list[dict[str, Any]],
    estimate: pricing.CostEstimate | None,
) -> dict[str, Any]:
    """One tile's row in the manifest.

    ``attempt`` is the newest attempt that published, which is the one the
    verdict came from. It is zero for a skipped tile and for a run written
    before attempts were numbered, both of which published nothing under a
    number.
    """
    return {
        "tile": result.job.tile.name,
        "window": result.job.window_label,
        "status": result.status,
        "duration_s": result.duration_s,
        "scene_count": result.scene_count,
        "peak_rss_mb": result.peak_rss_mb,
        "error": result.error,
        "lst_key": result.lst_key,
        "qa_key": result.qa_key,
        "attempt": max((row["attempt"] for row in rows), default=0),
        "attempts": rows,
        "cost": _tile_cost_block(estimate),
    }


def write_run_manifest(
    results: list[JobResult],
    *,
    run_id: str,
    window: str,
    started_at: datetime,
    retries: int,
    cluster_id: int | None = None,
    job_id: int | None = None,
    attempts: Mapping[str, list[dict[str, Any]]] | None = None,
    costs: Mapping[str, pricing.CostEstimate] | None = None,
    plan: dict[str, Any] | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Write the manifest for one distributed run and return its path.

    Args:
        results: Every job outcome in the run, including skipped and failed.
        run_id: Unique run token; becomes the manifest filename.
        window: Window label the run covered (``"2021-2025"`` or ``"multi"``).
        started_at: UTC timestamp taken before task submission.
        retries: Per-task retry budget the run was configured with.
        cluster_id: Coiled cluster the batch job ran on, for log retrieval
            after the fact. ``None`` when every tile was already complete.
        job_id: Coiled batch job id, for the same reason.
        attempts: Per-tile attempt series, keyed by tile name. A tile missing
            from the mapping reports an empty series, which is what a skipped
            tile and a pre-attempt run both have.
        costs: Per-tile cost estimate, keyed by tile name. Tiles that published
            no duration are absent rather than priced at zero.
        plan: Planned floors against observed peaks, from
            :func:`landsat_lst.batch._plan_comparison`. Omitted from the
            manifest entirely when the submission stored no plan.
        out_dir: Manifest directory (default ``settings.manifest_dir``).
    """
    out_dir = out_dir or settings.manifest_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts = attempts or {}
    costs = costs or {}

    payload = {
        "run_id": run_id,
        "window": window,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "cluster_id": cluster_id,
        "job_id": job_id,
        "config": {
            "region": settings.coiled_region,
            "vm_types": settings.coiled_vm_types,
            "spot_policy": settings.coiled_spot_policy,
            "max_workers": settings.coiled_max_workers,
            "job_timeout": settings.coiled_job_timeout,
            "retries": retries,
            "s3_bucket": settings.s3_bucket,
            "s3_prefix": settings.s3_prefix,
        },
        "counts": {
            "total": len(results),
            "completed": sum(1 for r in results if r.status == "completed"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
            "failed": sum(1 for r in results if r.status == "failed"),
        },
        "attempts": _attempts_block(attempts),
        "cost": _cost_block(costs),
        "tiles": [
            _tile_block(r, attempts.get(r.job.tile.name, []), costs.get(r.job.tile.name))
            for r in results
        ],
    }
    if plan is not None:
        payload["plan"] = plan

    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
