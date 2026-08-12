"""Per-run JSON manifests for distributed batch runs.

Each :func:`landsat_lst.job.run_distributed` call writes one manifest to
``settings.manifest_dir / f"{run_id}.json"``. The manifest is the durable
record of a paid run: which tiles completed, skipped, or failed (and why),
plus the per-tile duration, scene count, and peak memory that a costed
validation run needs to project the price of the global build. Coiled's own
dashboard forgets; this file does not.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from landsat_lst.config import settings

if TYPE_CHECKING:
    from pathlib import Path

    from landsat_lst.job import JobResult


def write_run_manifest(
    results: list[JobResult],
    *,
    run_id: str,
    window: str,
    started_at: datetime,
    retries: int,
    out_dir: Path | None = None,
) -> Path:
    """Write the manifest for one distributed run and return its path.

    Args:
        results: Every job outcome in the run, including skipped and failed.
        run_id: Unique run token; becomes the manifest filename.
        window: Window label the run covered (``"2021-2025"`` or ``"multi"``).
        started_at: UTC timestamp taken before task submission.
        retries: Per-task retry budget the run was configured with.
        out_dir: Manifest directory (default ``settings.manifest_dir``).
    """
    out_dir = out_dir or settings.manifest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "window": window,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "config": {
            "region": settings.coiled_region,
            "vm_types": settings.coiled_vm_types,
            "spot_policy": settings.coiled_spot_policy,
            "n_workers": settings.coiled_n_workers,
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
        "tiles": [
            {
                "tile": r.job.tile.name,
                "window": r.job.window_label,
                "status": r.status,
                "duration_s": r.duration_s,
                "scene_count": r.scene_count,
                "peak_rss_mb": r.peak_rss_mb,
                "error": r.error,
                "lst_key": r.lst_key,
                "qa_key": r.qa_key,
            }
            for r in results
        ],
    }

    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
