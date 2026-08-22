"""Stage-2 probe: per-VM read throughput vs request concurrency, one VM.

Submits ``probe_io_ladder_task.py`` to Coiled Batch as a single task on one
VM. The task runs the ladder's arms sequentially in fresh subprocesses; this
driver waits, collects the per-arm JSON lines from the batch log, and turns
the measured rates into the numbers the 60-minute-tile plan needs:

- the per-VM rate R at each concurrency, and where it plateaus;
- projected offset-pass hours (2 x 949.3 GB) and native-pass hours
  (3,797 GB) at each R;
- the VM counts N_offsets and N_composite that fit the phase budgets
  (15 min and 38 min) at the best measured R.

Cost: one VM for well under an hour -- tens of cents on r6i.2xlarge.

    python scripts/probe_io_ladder.py --dry-run
    python scripts/probe_io_ladder.py
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
sys.path.insert(0, str(HERE.parent / "src"))

JOB_TIMEOUT = "90m"
POLL_S = 30
MAX_WAIT_S = 5400
VM_HOURLY = 0.504  # r6i.2xlarge on-demand

# 9000^2 is the factor-2 offset grid (18000^2 native). An earlier model used
# 4500^2 and was wrong by 4x in every derived number.
OFFSET_PASS_BYTES = 9000**2 * 2930 * 2 * 2  # 949.3 GB, read twice per tile
NATIVE_PASS_BYTES = 18000**2 * 2930 * 2 * 2  # 3,797 GB, read once

OFFSET_BUDGET_S = 15 * 60
COMPOSITE_BUDGET_S = 38 * 60


def command() -> str:
    blob = base64.b64encode((HERE / "probe_io_ladder_task.py").read_bytes()).decode()
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
        time.sleep(POLL_S)
    return f"timeout({last})"


def collect(cluster_id: int) -> list[dict]:
    venv = HERE.parent / ".venv" / "bin" / "coiled"
    raw = subprocess.run(
        [str(venv), "logs", str(cluster_id), "--no-color"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    rows = []
    for m in re.finditer(r'\{"phase": "arm".*', raw):
        try:
            rows.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    seen: dict[int, dict] = {}
    for r in rows:
        seen[r.get("arm", -1)] = r
    return [seen[k] for k in sorted(seen)]


def project(rows: list[dict]) -> dict:
    # Arm 0 is the warmup by convention (ladder v2): it absorbs first-run
    # warming (v1's control ran 1.81x its first arm) and is excluded from
    # both the best-rate pick and the drift check.
    ok = [r for r in rows if "decoded_mb_s" in r and r.get("arm") != 0]
    if not ok:
        return {"verdict": "no data"}
    best = max(ok, key=lambda r: r["decoded_mb_s"])
    rate = best["decoded_mb_s"] * 1e6
    control = ok[-1]
    first = next(
        (
            r
            for r in ok
            if r is not control
            and (r["io_threads"], r["chunk"]) == (control["io_threads"], control["chunk"])
        ),
        None,
    )
    drift = (
        round(control["decoded_mb_s"] / first["decoded_mb_s"], 3)
        if first and control and first["decoded_mb_s"]
        else None
    )
    n_off = 2 * OFFSET_PASS_BYTES / (rate * OFFSET_BUDGET_S)
    n_comp = NATIVE_PASS_BYTES / (rate * COMPOSITE_BUDGET_S)
    return {
        "best_arm": {
            k: best[k]
            for k in (
                "io_threads",
                "chunk",
                "decoded_mb_s",
                "wire_mb_s",
                "cpu_cores_busy",
                "peak_rss_gb",
            )
        },
        "control_over_first": drift,
        "per_vm_R_mb_s": best["decoded_mb_s"],
        "offset_two_pass_hours_1vm": round(2 * OFFSET_PASS_BYTES / rate / 3600, 2),
        "native_pass_hours_1vm": round(NATIVE_PASS_BYTES / rate / 3600, 2),
        "n_vms_offsets_for_15min": round(n_off, 1),
        "n_vms_composite_for_38min": round(n_comp, 1),
        "single_vm_60min_feasible": (2 * OFFSET_PASS_BYTES + NATIVE_PASS_BYTES) / rate <= 3600,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--collect",
        type=int,
        metavar="CLUSTER_ID",
        help="skip submission; summarize an existing cluster's log",
    )
    ap.add_argument(
        "--vm-type",
        default=None,
        help="Coiled VM type override (e.g. m6i.4xlarge); default: settings",
    )
    ap.add_argument(
        "--factor",
        type=int,
        default=2,
        help="resolution factor: 2 = coarse/offset shape, 1 = native/composite",
    )
    ap.add_argument("--scenes-per-arm", type=int, default=24)
    ap.add_argument(
        "--arms",
        default=None,
        help='override arm list as "io:chunk,io:chunk,..." (first = warmup)',
    )
    ap.add_argument(
        "--out",
        default="io_ladder.json",
        help="result filename under results/probe/",
    )
    args = ap.parse_args()

    out_path = HERE.parent / "results" / "probe" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "arms": 9,
                    "scenes_per_arm": 24,
                    "decoded_gb_per_arm": round(24 * 9000**2 * 2 * 2 / 1e9, 1),
                    "est_cost_usd": round(VM_HOURLY * 1.0, 2),
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

        from landsat_lst.config import settings  # noqa: PLC0415
        from landsat_lst.job import _worker_environ  # noqa: PLC0415

        run_id = f"ioladder-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"
        env = {
            **_worker_environ(),
            "PROBE_FACTOR": str(args.factor),
            "PROBE_SCENES_PER_ARM": str(args.scenes_per_arm),
        }
        if args.arms:
            env["PROBE_ARMS"] = args.arms
        res = coiled.batch_run(
            command=command(),
            name=f"lst-{run_id}",
            region=settings.coiled_region,
            vm_type=[args.vm_type] if args.vm_type else settings.coiled_vm_types,
            spot_policy=settings.coiled_spot_policy,
            max_workers=1,
            ntasks=1,
            max_retries=0,
            job_timeout=JOB_TIMEOUT,
            env=env,
            tag={"project": "landsat-lst", "run_id": run_id, "kind": "probe"},
            forward_aws_credentials=False,
        )
        cid = res.get("cluster_id")
        print(f"cluster {cid}, waiting ...", flush=True)
        state = wait_for(cid)
        print(f"final state: {state}", flush=True)
        rows = collect(cid)

    summary = {"cluster_id": cid, "arms": rows, "projection": project(rows)}
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["projection"], indent=2))
    print(
        f"\n{'io':>4} {'chunk':>6} {'MB/s':>8} {'wire':>8} {'cores':>6} "
        f"{'RSS GB':>7} {'throttle':>9}"
    )
    for r in rows:
        if "decoded_mb_s" not in r:
            print(
                f"{r.get('io_threads', '?'):>4} {r.get('chunk', '?'):>6}  "
                f"ERROR: {str(r.get('error', ''))[:80]}"
            )
            continue
        print(
            f"{r['io_threads']:>4} {r['chunk']:>6} {r['decoded_mb_s']:>8.2f} "
            f"{r['wire_mb_s']:>8.2f} {r['cpu_cores_busy']:>6.2f} "
            f"{r['peak_rss_gb']:>7.2f} {r.get('throttle_mentions', 0):>9}"
        )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
