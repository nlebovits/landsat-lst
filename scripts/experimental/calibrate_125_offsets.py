"""One staged offsets-stage calibration for issue #125.

Submits the production offsets stage for S30W065 2021-2025 over the *retained*
plan and item set from ``shard-S30W065-2021-2025-20260823T102135Z``, with the
coarse stage on, and collects what the shards publish. One attempt. No
composite, no fleet, no second arm: the direct-path values already exist as the
v2 offsets record, and the direct-path timings already exist as that run's
``phase_seconds``.

    python scripts/experimental/calibrate_125_offsets.py --dry-run
    python scripts/experimental/calibrate_125_offsets.py
    python scripts/experimental/calibrate_125_offsets.py --collect <run-id>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Where the frozen plan and item set come from. It must be a run whose plan
# digest matches what this code computes, and the digest covers the exact
# transform (algorithm_version 2), so the August anchor run's plan is refused.
# Same tile, window and 1,031-scene set as the anchor, so the offsets record
# key -- and therefore the value comparison -- is unchanged.
PLAN_SOURCE = "shard-S30W065-2021-2025-20260903T231831Z-group6-r1"
#: Where the phase timings are compared against.
BASELINE = "shard-S30W065-2021-2025-20260823T102135Z"
TILE = "S30W065"
YEAR, END_YEAR = 2021, 2025
UNITS = 15
EST_CREDITS = 60.0
POLL_S = 45
MAX_WAIT_S = 3900


def _root(run_id: str) -> str:
    from landsat_lst import shards  # noqa: PLC0415

    return shards.shard_root(run_id, TILE)


def seed_plan(storage, run_id: str) -> dict:
    """Copy the retained plan and items into this run's prefix, verbatim."""
    src, dst = _root(PLAN_SOURCE), _root(run_id)
    out = {}
    for name in ("plan.json", "items.json"):
        body = storage.read_text(f"{src}/{name}")
        if body is None:
            msg = f"retained {name} missing at {src}/{name}"
            raise RuntimeError(msg)
        storage.write_text(f"{dst}/{name}", body)
        out[name] = len(body)
    return out


def phase_table(storage, run_id: str) -> dict:
    rows = []
    for key in sorted(storage.list_prefix(f"{_root(run_id)}/state/")):
        if not re.search(r"offsets\.\d+\.\d+\.json$", key):
            continue
        rows.append(json.loads(storage.read_text(key)))
    table = {}
    phases = sorted({p for r in rows for p in (r.get("phase_seconds") or {})})
    for phase in phases:
        vals = [r["phase_seconds"][phase] for r in rows if phase in (r.get("phase_seconds") or {})]
        table[phase] = {
            "n": len(vals),
            "min": round(min(vals), 1),
            "median": round(statistics.median(vals), 1),
            "max": round(max(vals), 1),
        }
    peaks = [r.get("peak_rss_mb") for r in rows if r.get("peak_rss_mb")]
    return {
        "shards": len(rows),
        "phases": table,
        "peak_rss_mb": {"max": max(peaks), "median": statistics.median(peaks)} if peaks else None,
    }


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect", metavar="RUN_ID")
    ap.add_argument("--suffix", default="125cal", help="run id suffix")
    ap.add_argument(
        "--merge",
        action="store_true",
        help="after all partials land, write the ADR-012 offsets record",
    )
    args = ap.parse_args()

    from landsat_lst.config import settings  # noqa: PLC0415
    from landsat_lst.models import ProcessingJob  # noqa: PLC0415
    from landsat_lst.storage import get_storage  # noqa: PLC0415
    from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415

    if settings.storage_backend != "s3":
        print("refusing: LST_STORAGE_BACKEND must be s3", file=sys.stderr)
        return 2
    storage = get_storage()
    job = ProcessingJob(tile=parse_tile_name(TILE), year=YEAR, end_year=END_YEAR)

    if args.dry_run:
        plan = json.loads(storage.read_text(f"{_root(PLAN_SOURCE)}/plan.json"))
        scenes, h, w = len(plan["scene_times"]), *plan["coarse_shape"]
        coarse = h * w * scenes * 2 * 2
        print(
            json.dumps(
                {
                    "tile": TILE,
                    "window": f"{YEAR}-{END_YEAR}",
                    "units": UNITS,
                    "plan_source": PLAN_SOURCE,
                    "timing_baseline": BASELINE,
                    "scenes": scenes,
                    "coarse_shape": [h, w],
                    "blocks": len(plan["blocks"]),
                    "land_blocks": sum(plan["block_has_land"]),
                    "source_pass_gb": round(coarse / 1e9, 1),
                    "staged_gb": round(coarse / 2 / 1e9, 1),
                    "staged_per_shard_gb": round(coarse / 2 / UNITS / 1e9, 2),
                    "staged_objects": len(plan["blocks"]) * len(plan["scene_batches"]),
                    "vm_types": settings.coiled_vm_types,
                    "spot_policy": settings.shard_spot_policy,
                    "stage_coarse": settings.destripe_stage_coarse,
                    "est_credits": EST_CREDITS,
                },
                indent=2,
            )
        )
        return 0

    if args.collect:
        run_id = args.collect
    else:
        from landsat_lst import quota  # noqa: PLC0415
        from landsat_lst.batch import submit_shard_stage  # noqa: PLC0415

        print(f"identity: {quota.preflight_identity()}", flush=True)
        print(f"write:    {quota.preflight_write_access()}", flush=True)
        quota.preflight_credits(EST_CREDITS)

        run_id = (
            f"shard-{TILE}-{YEAR}-{END_YEAR}-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}-{args.suffix}"
        )
        print(f"run id:   {run_id}", flush=True)
        print(f"seeded:   {seed_plan(storage, run_id)}", flush=True)
        sub = submit_shard_stage(
            stage="offsets",
            run_id=run_id,
            tile=TILE,
            indexes=list(range(UNITS)),
            job=job,
            units=UNITS,
        )
        print(f"cluster {sub.cluster_id}, job {sub.job_id}, {UNITS} units", flush=True)
        (ROOT / "results" / "probe").mkdir(parents=True, exist_ok=True)
        (ROOT / "results" / "probe" / f"125-{args.suffix}.launch.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "cluster_id": sub.cluster_id,
                    "job_id": sub.job_id,
                    "units": UNITS,
                },
                indent=2,
            )
        )

        prefix = f"{_root(run_id)}/offsets/scene/"
        deadline = time.time() + MAX_WAIT_S
        while time.time() < deadline:
            landed = len(storage.list_prefix(prefix))
            staged = len(storage.list_prefix(f"{_root(run_id)}/stage/"))
            print(f"  partials {landed}/{UNITS}  staged objects {staged}", flush=True)
            if landed >= UNITS:
                break
            time.sleep(POLL_S)

    if args.merge:
        from landsat_lst.shard_tasks import merge_offsets  # noqa: PLC0415

        landed = len(storage.list_prefix(f"{_root(run_id)}/offsets/scene/"))
        if landed < UNITS:
            print(f"NOT merging: only {landed}/{UNITS} partials landed", flush=True)
        else:
            key = merge_offsets(run_id, TILE, storage=storage)
            print(f"merged offsets record: {key.storage_key}", flush=True)

    out = {"run_id": run_id, "timings": phase_table(storage, run_id)}
    out["staged_objects_now"] = len(storage.list_prefix(f"{_root(run_id)}/stage/"))
    out["partials"] = len(storage.list_prefix(f"{_root(run_id)}/offsets/scene/"))
    path = ROOT / "results" / "probe" / f"125-{args.suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
