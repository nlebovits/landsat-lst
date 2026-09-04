"""Driver for the #125 staging throughput probe: one task, one VM, one attempt.

    python scripts/experimental/s3_staging_probe.py --dry-run
    python scripts/experimental/s3_staging_probe.py
    python scripts/experimental/s3_staging_probe.py --collect <cluster_id>

Follows scripts/experimental/s3_microprobe.py: the task body ships base64 in
the command and its stdout is read back from the batch log. The VM class is
the offsets class, r6i.2xlarge, because the number under test is the rate a
phase-A writer and a phase-B reader would see, and those run on offsets VMs.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

JOB_TIMEOUT = "30m"
POLL_S = 20
MAX_WAIT_S = 2100
VM_TYPE = "r6i.2xlarge"
VCPUS = 8
EST_CREDITS = 3.0


def command() -> str:
    blob = base64.b64encode((HERE / "s3_staging_probe_task.py").read_bytes()).decode()
    return f"#!/bin/bash\necho {blob} | base64 -d > /tmp/probe_task.py\npython /tmp/probe_task.py\n"


def wait_for(cluster_id: int) -> str:
    import coiled  # noqa: PLC0415

    deadline = time.time() + MAX_WAIT_S
    last = ""
    while time.time() < deadline:
        try:
            rows = coiled.batch.status(cluster_id)
        except Exception:
            time.sleep(POLL_S)
            continue
        states = [r.get("state") for r in rows]
        last = ",".join(sorted({s for s in states if s}))
        if states and all(s in ("done", "error", "failed") for s in states):
            return last
        print(f"  ... {last}", flush=True)
        time.sleep(POLL_S)
    return f"timeout({last})"


def collect(cluster_id: int) -> list[dict]:
    raw = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "coiled"), "logs", str(cluster_id), "--no-color"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    rows = []
    for m in re.finditer(r'\{"phase": .*', raw):
        try:
            rows.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect", type=int, metavar="CLUSTER_ID")
    ap.add_argument("--main-gb", type=float, default=34.0)
    ap.add_argument("--anchor-gb", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="s3_staging_probe.json")
    args = ap.parse_args()

    from landsat_lst.config import settings  # noqa: PLC0415

    out_path = ROOT / "results" / "probe" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = f"s3stage-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"
    probe_prefix = f"{settings.s3_prefix}/_microprobe/{run_id}"
    env_extra = {
        "PROBE_BUCKET": settings.s3_bucket,
        "PROBE_PREFIX": probe_prefix,
        "PROBE_REGION": settings.s3_region,
        "PROBE_MAIN_GB": str(args.main_gb),
        "PROBE_ANCHOR_GB": str(args.anchor_gb),
        "PROBE_THREADS": str(args.threads),
    }

    if args.dry_run:
        obj_mb = 1024 * 1024 * 10 * 2 / 1e6
        print(
            json.dumps(
                {
                    "vm_type": VM_TYPE,
                    "vcpus": VCPUS,
                    "region": settings.coiled_region,
                    "bucket": settings.s3_bucket,
                    "probe_prefix": probe_prefix,
                    "object_mb": round(obj_mb, 1),
                    "main_objects": int(args.main_gb * 1e9 // (obj_mb * 1e6)),
                    "anchor_objects": int(args.anchor_gb * 1e9 // (obj_mb * 1e6)),
                    "bytes_moved_gb": round(2 * (args.main_gb + args.anchor_gb), 1),
                    "expected_minutes": "8-14 including boot",
                    "expected_credits": EST_CREDITS,
                    "command_chars": len(command()),
                },
                indent=2,
            )
        )
        return 0

    if args.collect:
        cid = args.collect
        rows = collect(cid)
    else:
        import coiled  # noqa: PLC0415

        from landsat_lst import quota  # noqa: PLC0415
        from landsat_lst.job import _worker_environ  # noqa: PLC0415

        print(f"identity: {quota.preflight_identity()}", flush=True)
        print(f"write access: {quota.preflight_write_access()}", flush=True)
        quota.preflight_credits(EST_CREDITS)

        res = coiled.batch_run(
            command=command(),
            name=f"lst-{run_id}",
            region=settings.coiled_region,
            vm_type=[VM_TYPE],
            spot_policy=settings.coiled_spot_policy,
            max_workers=1,
            ntasks=1,
            max_retries=0,
            job_timeout=JOB_TIMEOUT,
            env={**_worker_environ(), **env_extra},
            tag={"project": "landsat-lst", "run_id": run_id, "kind": "probe", "issue": "125"},
            forward_aws_credentials=False,
        )
        cid = res.get("cluster_id")
        print(f"cluster {cid}, job {res.get('job_id')}, waiting ...", flush=True)
        print(f"final state: {wait_for(cid)}", flush=True)
        rows = collect(cid)

    out_path.write_text(json.dumps({"cluster_id": cid, "run_id": run_id, "rows": rows}, indent=2))
    print(f"\n{'arm':8} {'op':6} {'objects':>8} {'GB':>7} {'elapsed s':>10} {'MB/s':>8}")
    for r in rows:
        if r.get("phase") != "arm":
            continue
        print(
            f"{r.get('arm', ''):8} {r.get('op', ''):6} {r.get('objects', ''):>8} "
            f"{r.get('gb', ''):>7} {r.get('elapsed_s', ''):>10} {r.get('mb_s', ''):>8}"
        )
    for r in rows:
        if r.get("phase") in ("cycle", "cleanup", "cleanup_failed", "error", "done"):
            print(json.dumps(r))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
