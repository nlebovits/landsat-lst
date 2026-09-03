"""Submit exactly one S30W065 composite shard on a 32 GiB worker (issue #136).

The cloud discriminator for ADR-019: does the uint16 DN stack put the worst
measured 512-row band (index 27, 43.9 GB on a 64 GiB m6i.4xlarge in run
shard-S30W065-2021-2025-20260823T102135Z) under 28 GiB on a 32 GiB VM of the
same core count. c6i.4xlarge is 16 vCPU and 32 GiB, so the dask thread count,
and with it the number of fused P95 tasks in flight, is unchanged.

The run prefix is fresh and holds server-side copies of the retained run's
plan.json and items.json, so the shard reads the production scene set, the
production plan, and the production merged offsets at their canonical key.
It writes only under its own prefix. With one band of 35 present the export
claim cannot fire, so no COG is written and nothing shipped is touched.

Three gates run first, in the driver's order: identity, write access, credits.
The credit gate takes an operator limit of 20 credits for this run: 16 vCPU
for an hour is 16 credits at CREDITS_PER_VCPU_HOUR, and the band took 23 min.

Usage::

    AWS_PROFILE=radiant-earth LST_STORAGE_BACKEND=s3 \\
    LST_SHARD_COMPOSITE_VM_TYPE=c6i.4xlarge \\
    uv run python scripts/experimental/submit_u16_gate_shard.py --run-id <id> --index 27
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

INDEX_DEFAULT = 27
OPERATOR_CREDIT_LIMIT = 20.0
ESTIMATED_CREDITS = 8.0  # 16 vCPU x 0.5 h, the band's 23 min with boot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tile", default="S30W065")
    ap.add_argument("--index", type=int, default=INDEX_DEFAULT)
    ap.add_argument("--out", type=Path, default=Path("results/issue-136/cloud/submission.json"))
    args = ap.parse_args()

    if os.environ.get("LST_STORAGE_BACKEND") != "s3":
        print("refusing: LST_STORAGE_BACKEND must be s3", file=sys.stderr)
        return 2
    if "LST_STAC_URL" in os.environ:
        print("refusing: LST_STAC_URL is set; workers must use the config default", file=sys.stderr)
        return 2

    from landsat_lst import batch, quota, shards  # noqa: PLC0415
    from landsat_lst.config import settings  # noqa: PLC0415
    from landsat_lst.shard_tasks import load_context  # noqa: PLC0415

    print(f"vm_type={settings.shard_composite_vm_type} spot={settings.shard_spot_policy}")
    if settings.shard_composite_vm_type != "c6i.4xlarge":
        print("refusing: this gate runs on c6i.4xlarge only", file=sys.stderr)
        return 2

    arn = quota.preflight_identity()
    print(f"identity ok: {arn}")
    probed = quota.preflight_write_access()
    print(f"write access ok: {probed}")
    # Coiled publishes the workspace's credit limit through no API, and its
    # 30-day spend straddles the 2026-09-01 renewal, so no remainder can be
    # derived. The operator greenlit this one run on 2026-09-03 under a dollar
    # and a 20-credit cap; that is the acknowledgement the gate documents as
    # its escape. The usage endpoint's has_quota=false would still refuse.
    balance = quota.preflight_credits(ESTIMATED_CREDITS, acknowledged=True)
    print(f"credits ok: {balance}")

    ctx = load_context(args.run_id, args.tile)
    start, stop = ctx.plan.bands[args.index]
    print(f"plan loaded: {len(ctx.plan.bands)} bands, band {args.index} rows {start}:{stop}")
    if stop - start != 512:
        print("refusing: band is not 512 rows", file=sys.stderr)
        return 2
    existing = ctx.keys("composite/")
    if existing:
        print(f"refusing: run prefix already holds {len(existing)} composite keys", file=sys.stderr)
        return 2

    submission = batch.submit_shard_stage(
        stage="composite",
        run_id=args.run_id,
        tile=args.tile,
        indexes=[args.index],
        submission_round=1,
    )
    record = {
        "run_id": args.run_id,
        "tile": args.tile,
        "index": args.index,
        "rows": [start, stop],
        "vm_type": settings.shard_composite_vm_type,
        "spot_policy": settings.shard_spot_policy,
        "cluster_id": submission.cluster_id,
        "job_id": submission.job_id,
        "name": submission.name,
        "submitted_at": datetime.now(UTC).isoformat(),
        "state_key": shards.shard_state_key(ctx.root, "composite", args.index, 1),
        "band_keys": {p: shards.band_key(ctx.root, p, args.index) for p in ("lst_p95", "qa_count")},
        "baseline_run": "shard-S30W065-2021-2025-20260823T102135Z",
        "baseline_peak_rss_mb": 43948.75,
        "baseline_elapsed_s": 1365.8,
        "identity": arn,
        "operator_credit_limit": OPERATOR_CREDIT_LIMIT,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
