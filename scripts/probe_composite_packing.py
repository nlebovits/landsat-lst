"""Intra-VM composite packing: how many shards should share one VM?

Stage-3 sizing question. A composite row band is I/O-bound for part of its life
and CPU-bound for the rest; if the two phases interleave badly, one shard per VM
buys a VM that is idle half the time. This ladder measures the *aggregate*
useful throughput of K concurrent shards on one VM, and the price of it, at two
VM shapes:

    m6i.4xlarge (64 GiB, 16 vCPU)  at K in {1, 2, 3}
    r6i.4xlarge (128 GiB, 16 vCPU) at K in {2, 4}

Each arm is its own single-VM batch submission, run sequentially, and every arm
reads a **disjoint** set of row bands so nothing is served from a previous arm's
page cache. Arm 0 is a discardable warmup and the last arm repeats arm 1's
configuration as a warm control -- the ladder discipline of
``scripts/probe_io_ladder.py``. Note the one thing the control cannot mean here:
arms run on *different VMs*, so it detects drift across the ladder's wall-clock
window (S3 conditions, spot placement), not page-cache warming.

Acceptance is four-part, and an arm that wins on one part and loses on another
loses:

1. aggregate decoded MB/s;
2. USD per GB decoded, from ``VM_HOURLY``;
3. memory margin -- the *minimum* ``MemAvailable`` sampled at 2 s through the
   arm, as a fraction of ``MemTotal``. Below ``HEADROOM_FLOOR`` the arm is
   flagged: "fits at 95% RAM" is not a fit, whatever the throughput;
4. stability -- the spread of per-child wall times and the count of children
   that exited non-zero or were reaped. A dead child leaves an error row, the
   arm reports partial results, and the ladder continues.

What the probe writes
---------------------
``_shards/probe-pack-{timestamp}/{tile}/`` and nothing else. The frozen
production run named by ``--source-run`` is read twice per arm (``plan.json``,
``items.json``) and never written to.

Usage::

    python scripts/probe_composite_packing.py                  # dry run
    python scripts/probe_composite_packing.py --launch         # submits
    python scripts/probe_composite_packing.py --collect 1 2 3  # cluster ids
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

TILE = "S30W065"
SOURCE_RUN = "shard-S30W065-2021-2025-20260821T194111Z"

#: On-demand list price, us-west-2. Spot is cheaper and variable; a probe this
#: short is priced at list so the arms are comparable to each other rather than
#: to a market.
VM_HOURLY = {"m6i.4xlarge": 0.768, "r6i.4xlarge": 1.008}
VM_RAM_GIB = {"m6i.4xlarge": 64, "r6i.4xlarge": 128}
VCPUS = 16

#: An arm whose minimum MemAvailable falls below this fraction of MemTotal is
#: flagged regardless of throughput.
HEADROOM_FLOOR = 0.20

JOB_TIMEOUT = "75m"
CHILD_TIMEOUT_S = 3600
POLL_S = 30
MAX_WAIT_S = 5400

#: Probe bands: 512 rows each, cut from the top of the tile where the per-band
#: scene depth is flat. Measured on the frozen S30W065 item list, bands 0..13
#: hold 999-1013 solar-day groups -- a 1.4% spread, so no arm is advantaged by
#: its rows. (Bands 22+ fall to 795, which is why the ladder does not run there.)
BAND_ROWS = 512
N_BANDS = 14

#: (vm_type, K, label). Arm 0 warms, the last arm controls.
ARMS: list[tuple[str, int, str]] = [
    ("m6i.4xlarge", 1, "warmup -- discard"),
    ("m6i.4xlarge", 1, "reference"),
    ("m6i.4xlarge", 2, ""),
    ("m6i.4xlarge", 3, ""),
    ("r6i.4xlarge", 2, ""),
    ("r6i.4xlarge", 4, ""),
    ("m6i.4xlarge", 1, "control -- warm repeat of arm 1"),
]

#: Aggregate read rate assumed for the cost estimate. Deliberately below the
#: 155 MB/s the Stage-2 ladder measured on one VM: an estimate that flatters
#: the probe is an estimate that under-books the budget.
EST_MB_S = 150.0

#: The composite is not a read; it is a read plus a P95 over the time axis plus
#: two windowed writes. The estimate inflates pure-I/O time by this much.
EST_COMPUTE_FACTOR = 2.5

#: Fixed per-arm overhead: VM boot, image pull, interpreter import.
EST_OVERHEAD_S = 200.0

#: Measured on the frozen item list, 2026-08-22: items intersecting a 512-row
#: band, times their overlap with it. See the module docstring of the task
#: script for why this is not the same as the decoded volume.
EST_BAND_READ_GB = 16.5


def band_edges() -> list[tuple[int, int]]:
    return [(i * BAND_ROWS, (i + 1) * BAND_ROWS) for i in range(N_BANDS)]


def assignments() -> list[list[int]]:
    """Disjoint band indices per arm, in ladder order."""
    out: list[list[int]] = []
    nxt = 0
    for _, k, _ in ARMS:
        out.append(list(range(nxt, nxt + k)))
        nxt += k
    return out


def per_task_gb(chunk: int, times: int) -> float:
    """The rechunked block one composite task holds, in GB.

    ADR-013 rechunks time to a single chunk before either product is built, so
    a task holds ``chunk**2 * times * 4`` bytes of float32. ``times`` is the
    band's time axis -- and the measurement in this probe's report is that a
    band's axis is ~90% of the whole tile's, not a small fraction of it.
    """
    return chunk * chunk * times * 4 / 1e9


def estimate() -> list[dict]:
    rows = []
    for (vm, k, label), idx in zip(ARMS, assignments(), strict=True):
        io_s = k * EST_BAND_READ_GB * 1e3 / EST_MB_S
        wall = EST_OVERHEAD_S + io_s * EST_COMPUTE_FACTOR
        rows.append(
            {
                "vm_type": vm,
                "k": k,
                "label": label,
                "bands": idx,
                "est_wall_s": round(wall),
                "est_cost_usd": round(VM_HOURLY[vm] * wall / 3600, 3),
            }
        )
    return rows


def command() -> str:
    blob = base64.b64encode((HERE / "probe_composite_packing_task.py").read_bytes()).decode()
    return f"#!/bin/bash\necho {blob} | base64 -d > /tmp/probe_pack.py\npython /tmp/probe_pack.py\n"


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
    return rows


def submit(arm: int, vm: str, indices: list[int], run_id: str, chunk: int) -> int:
    import coiled  # noqa: PLC0415

    from landsat_lst.config import settings  # noqa: PLC0415
    from landsat_lst.job import _worker_environ  # noqa: PLC0415

    env = {
        **_worker_environ(),
        "PROBE_TILE": TILE,
        "PROBE_SOURCE_ROOT": f"_shards/{SOURCE_RUN}/{TILE}",
        "PROBE_RUN_ID": run_id,
        "PROBE_ARM": str(arm),
        "PROBE_VM_TYPE": vm,
        "PROBE_VM_HOURLY": str(VM_HOURLY[vm]),
        "PROBE_CHUNK": str(chunk),
        "PROBE_THREADS": str(max(1, VCPUS // len(indices))),
        "PROBE_CHILD_TIMEOUT_S": str(CHILD_TIMEOUT_S),
        "PROBE_BANDS": ",".join(f"{a}:{b}" for a, b in band_edges()),
        "PROBE_INDICES": ",".join(str(i) for i in indices),
    }
    res = coiled.batch_run(
        command=command(),
        name=f"lst-{run_id}-a{arm}",
        region=settings.coiled_region,
        vm_type=[vm],
        spot_policy=settings.coiled_spot_policy,
        max_workers=1,
        ntasks=1,
        max_retries=0,
        job_timeout=JOB_TIMEOUT,
        env=env,
        tag={"project": "landsat-lst", "run_id": run_id, "kind": "probe-pack"},
        forward_aws_credentials=False,
    )
    return res.get("cluster_id")


def _flag(row: dict) -> str:
    frac = row.get("min_headroom_frac")
    if frac is None:
        return "?"
    return "LOW" if frac < HEADROOM_FLOOR else "ok"


def render(rows: list[dict]) -> None:
    print(
        f"\n{'arm':>3} {'vm_type':<13} {'K':>2} {'aggMB/s':>8} {'$/GB':>9} "
        f"{'minFree%':>9} {'hdrm':>5} {'wall s':>8} {'spread':>7} {'peakGB':>7} {'fail':>5}"
    )
    for r in rows:
        if "decoded_mb_s" not in r or r.get("error"):
            print(
                f"{r.get('arm', '?'):>3} {r.get('vm_type', '?'):<13} "
                f"{r.get('k', '?'):>2}  ERROR: {str(r.get('error', ''))[:70]}"
            )
            continue
        frac = r.get("min_headroom_frac")
        print(
            f"{r['arm']:>3} {r['vm_type']:<13} {r['k']:>2} "
            f"{r['decoded_mb_s']:>8.1f} {r.get('usd_per_gb_decoded')!s:>9} "
            f"{(frac * 100 if frac is not None else float('nan')):>9.1f} {_flag(r):>5} "
            f"{r['wall_s']:>8.0f} {r.get('child_wall_spread')!s:>7} "
            f"{r.get('peak_child_vmhwm_gb')!s:>7} {r.get('failures', 0):>5}"
        )
    low = [r["arm"] for r in rows if r.get("min_headroom_frac", 1) < HEADROOM_FLOOR]
    if low:
        print(
            f"\nFLAG: arms {low} fell below {HEADROOM_FLOOR:.0%} headroom -- "
            "they do not pass acceptance whatever their throughput."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--launch",
        action="store_true",
        help="actually submit the arms; without it the script only prints the plan",
    )
    ap.add_argument("--dry-run", action="store_true", help="explicit no-op (the default anyway)")
    ap.add_argument(
        "--collect",
        type=int,
        nargs="+",
        metavar="CLUSTER_ID",
        help="skip submission; summarize these clusters' logs, in arm order",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=512,
        help="spatial chunk edge each shard loads at (settings.shard_composite_chunk)",
    )
    ap.add_argument("--out", default="composite_packing.json")
    args = ap.parse_args()

    out_path = HERE.parent / "results" / "probe" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    est = estimate()
    total = sum(r["est_cost_usd"] for r in est)
    if not args.collect:
        print(f"tile {TILE}   source run {SOURCE_RUN} (read-only)")
        print(f"probe writes to _shards/probe-pack-<ts>/{TILE}/ only")
        print(f"{N_BANDS} disjoint bands of {BAND_ROWS} rows; chunk {args.chunk}")
        print(
            f"\n{'arm':>3} {'vm_type':<13} {'K':>2} {'thr/ch':>6} {'bands':<16} "
            f"{'est s':>6} {'est $':>7}  note"
        )
        for n, r in enumerate(est):
            print(
                f"{n:>3} {r['vm_type']:<13} {r['k']:>2} {max(1, VCPUS // r['k']):>6} "
                f"{r['bands']!s:<16} {r['est_wall_s']:>6} {r['est_cost_usd']:>7.3f}  "
                f"{r['label']}"
            )
        print(
            f"\nestimated total: ${total:.2f}   "
            f"({EST_BAND_READ_GB} GB/band at {EST_MB_S:.0f} MB/s x{EST_COMPUTE_FACTOR} compute, "
            f"+{EST_OVERHEAD_S:.0f}s/arm overhead)"
        )
        if total > 6:
            print("OVER BUDGET: shrink BAND_ROWS or N_BANDS before launching.")
        print("\nper-task working set (ADR-013 rechunk, chunk^2 x times x 4 B):")
        for chunk in (512, 1024):
            gb = per_task_gb(chunk, 1031)
            for vm in VM_HOURLY:
                thr = VCPUS
                print(
                    f"  chunk {chunk:>4}  {gb:5.2f} GB/task   "
                    f"{vm:<13} {thr} threads -> {gb * thr:6.1f} GB vs {VM_RAM_GIB[vm]} GiB"
                )

    if not args.launch or args.collect:
        if not args.collect:
            print("\ndry run: nothing submitted. Pass --launch to run the ladder.")
            return 0
        rows = [r for cid in args.collect for r in collect(cid)]
        cluster_ids = args.collect
    else:
        run_id = f"probe-pack-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"
        print(f"\nLAUNCHING {len(ARMS)} arms, estimated ${total:.2f} total, run {run_id}")
        rows = []
        cluster_ids = []
        for n, ((vm, _, _), idx) in enumerate(zip(ARMS, assignments(), strict=True)):
            cid = submit(n, vm, idx, run_id, args.chunk)
            cluster_ids.append(cid)
            print(f"arm {n}: cluster {cid} ({vm}, K={len(idx)}, bands {idx}) ...", flush=True)
            print(f"arm {n}: {wait_for(cid)}", flush=True)
            rows.extend(collect(cid))

    rows.sort(key=lambda r: r.get("arm", 0))
    out_path.write_text(json.dumps({"clusters": cluster_ids, "arms": rows}, indent=2))
    render(rows)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
