"""Run the composite shard's graph locally on synthetic leaves and time each part.

Issue #129 asks what the 14 idle threads of a production composite shard are
blocked on. The shard reads real pixels from S3, and that tier has no local
instance. This probe removes the network and keeps everything else: the same
``compute_annual_composite`` chain a row band runs (QA mask, Celsius, land
mask, offset join, single-time-chunk rechunk, P95, monthly counts), the same
``_encode_native`` and ``write_intermediates`` call, on
:func:`landsat_lst.profiling.synthetic_dataset` leaves at production chunking.
It answers one bounded question: how many CPU-seconds the compute side costs
per spatial chunk, and how wide it runs on 16 threads.

Every point runs in a fresh subprocess under ``RLIMIT_AS`` and an RSS watchdog,
because an unbounded local graph build has taken a 64 GB desktop down before
(CLAUDE.md, "Never run unbounded graph builds locally"). The band is narrow on
purpose: a full 36-chunk production band at T=1031 is 39 GB of stack.

Usage::

    python scripts/probe_composite_local.py --threads 16 --col-chunks 8
    python scripts/probe_composite_local.py --threads 4 --col-chunks 8
    python scripts/probe_composite_local.py --reject-frac 0.115 --keep-dir results/probe/arm-a
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404 -- fixed argv, no shell; a fresh interpreter is the point
import sys
import time
from pathlib import Path

CHILD = r"""
import json, os, resource, sys, threading, time

rlimit_bytes = int(os.environ["LSTP_RLIMIT_GB"]) * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (rlimit_bytes, rlimit_bytes))
rss_cap_mb = int(os.environ["LSTP_RSS_CAP_GB"]) * 1024


def _vm(field):
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024
    return 0.0


def _watchdog():
    while True:
        if _vm("VmRSS") > rss_cap_mb:
            sys.stderr.write(json.dumps({"watchdog": "rss_cap_exceeded", "rss_mb": _vm("VmRSS")}) + "\n")
            sys.stderr.flush()
            os._exit(97)
        time.sleep(1.0)


threading.Thread(target=_watchdog, daemon=True).start()

rows = int(os.environ["LSTP_ROWS"])
cols = int(os.environ["LSTP_COLS"])
scenes = int(os.environ["LSTP_SCENES"])
chunk = int(os.environ["LSTP_CHUNK"])
threads = int(os.environ["LSTP_THREADS"])
bin_s = float(os.environ.get("LSTP_BIN_S", "5"))

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import Profiler, ResourceProfiler

from landsat_lst.config import settings

settings.load_chunk_size = chunk
settings.ged_gap_mask = False
dask.config.set(scheduler="threads", num_workers=threads)

from landsat_lst.cog import lst_product, qa_product, write_intermediates
from landsat_lst.job import _encode_native
from landsat_lst.pipeline import TIME_CHUNK, compute_annual_composite
from landsat_lst.profiling import synthetic_dataset

assert TIME_CHUNK == 10, f"TIME_CHUNK moved to {TIME_CHUNK}; re-pin the probe"

t_build = time.monotonic()
data = synthetic_dataset(shape=(rows, cols), scenes=scenes, chunk_size=chunk)
land = xr.DataArray(
    np.ones((rows, cols), dtype=bool),
    dims=("latitude", "longitude"),
    coords={"latitude": data.latitude, "longitude": data.longitude},
)
# Zero offsets keep every scene, so the graph carries the isel + broadcast
# subtract a real band carries, with nothing rejected.
reject_frac = float(os.environ.get("LSTP_REJECT_FRAC", "0"))
# A rejected scene carries an offset past the cap, so debias_with_offsets drops it
# through isel(time=kept_idx) exactly as production does (912 of 1,031 kept on S30W065).
rejected = np.random.default_rng(1).random(data.sizes["time"]) < reject_frac
# float32, as OffsetCache.read hands a shard (offsets.py, dtype=np.float32).
# The first version of this probe built float64 here, and since the
# subtraction takes the wider dtype it measured a float64 stack production
# never runs: every local point in findings-composite-shard-bottleneck.md is
# 2x the real working set (docs/findings-composite-precision-audit.md).
offset = xr.DataArray(
    np.where(rejected, 99.0, 0.0).astype(np.float32), dims=("time",), coords={"time": data.time}
)
n_valid = xr.DataArray(
    np.full(data.sizes["time"], 10**9, dtype=np.int64), dims=("time",), coords={"time": data.time}
)
composite = compute_annual_composite(data, land_mask=land, offsets=(offset, n_valid))
composite["lst_p95"] = composite["lst_p95"].where(land)
composite["qa_count"] = composite["qa_count"].where(land, 0).astype(np.uint8)
composite.attrs.update({"tile": "S30W065", "year": 2021, "window": "2021-2025", "scene_count": scenes})
native = _encode_native(composite)
graph_build_s = time.monotonic() - t_build

import tempfile
from pathlib import Path

keep_dir = os.environ.get("LSTP_KEEP_DIR", "")
scratch = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix="lst_probe_local_"))
scratch.mkdir(parents=True, exist_ok=True)
paths = {"lst_p95": scratch / "lst_p95.tif", "qa_count": scratch / "qa_count.tif"}
products = [lst_product(native, paths["lst_p95"]), qa_product(native, paths["qa_count"])]

ru0 = resource.getrusage(resource.RUSAGE_SELF)
t0 = time.monotonic()
with Profiler() as prof, ResourceProfiler(dt=1.0) as rprof:
    write_intermediates([(p.da, path) for p, path in zip(products, paths.values())])
wall = time.monotonic() - t0
ru1 = resource.getrusage(resource.RUSAGE_SELF)
cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)

import shutil
sizes = {k: p.stat().st_size for k, p in paths.items()}
if not keep_dir:
    shutil.rmtree(scratch, ignore_errors=True)

# Per-task-prefix accounting. A key is a tuple whose first element is
# "<name>-<hash>"; fusion joins names with "-" freely, so the prefix is the
# name with its trailing 32-hex hash removed.
import re
hash_re = re.compile(r"-[0-9a-f]{32}$")


def prefix(key):
    name = key[0] if isinstance(key, tuple) else key
    return hash_re.sub("", str(name))


t_start = min(r.start_time for r in prof.results) if prof.results else t0
by_prefix = {}
n_bins = int(wall // bin_s) + 2
timeline = {}
for r in prof.results:
    p = prefix(r.key)
    d = r.end_time - r.start_time
    acc = by_prefix.setdefault(p, {"tasks": 0, "seconds": 0.0})
    acc["tasks"] += 1
    acc["seconds"] += d
    # Thread-seconds per bin, so concurrency per category is seconds / bin_s.
    b0 = int((r.start_time - t_start) // bin_s)
    b1 = int((r.end_time - t_start) // bin_s)
    row = timeline.setdefault(p, [0.0] * n_bins)
    for b in range(b0, min(b1, n_bins - 1) + 1):
        lo = max(r.start_time, t_start + b * bin_s)
        hi = min(r.end_time, t_start + (b + 1) * bin_s)
        if hi > lo:
            row[b] += hi - lo

total_task_s = sum(v["seconds"] for v in by_prefix.values())
for v in by_prefix.values():
    v["share"] = v["seconds"] / total_task_s if total_task_s else 0.0
ranked = dict(sorted(by_prefix.items(), key=lambda kv: -kv[1]["seconds"]))

mem = [r.mem for r in rprof.results]
cpu = [r.cpu for r in rprof.results]
print(json.dumps({
    "rows": rows, "cols": cols, "scenes": scenes, "chunk": chunk, "threads": threads,
    "reject_frac": reject_frac, "scenes_kept": int((~rejected).sum()),
    "spatial_chunks": (rows // chunk) * (cols // chunk),
    "graph_build_s": round(graph_build_s, 2),
    "wall_s": round(wall, 2),
    "cpu_s": round(cpu_s, 1),
    "cores_busy": round(cpu_s / wall, 2) if wall else None,
    "task_thread_s": round(total_task_s, 1),
    "mean_task_concurrency": round(total_task_s / wall, 2) if wall else None,
    "peak_vmhwm_mb": round(_vm("VmHWM"), 1),
    "rprof_peak_mem_mb": round(max(mem), 1) if mem else None,
    "rprof_mean_cpu_pct": round(sum(cpu) / len(cpu), 1) if cpu else None,
    "n_tasks": len(prof.results),
    "by_prefix": ranked,
    "timeline_bin_s": bin_s,
    "timeline": {k: [round(x, 1) for x in v] for k, v in timeline.items()},
    "output_bytes": sizes,
}))
"""


def run_point(args: argparse.Namespace) -> dict:
    env = {
        **os.environ,
        "LSTP_ROWS": str(args.rows),
        "LSTP_COLS": str(args.col_chunks * args.chunk),
        "LSTP_SCENES": str(args.scenes),
        "LSTP_CHUNK": str(args.chunk),
        "LSTP_THREADS": str(args.threads),
        "LSTP_RLIMIT_GB": str(args.rlimit_gb),
        "LSTP_RSS_CAP_GB": str(args.rss_cap_gb),
        "LSTP_BIN_S": str(args.bin_s),
        "LSTP_REJECT_FRAC": str(args.reject_frac),
        "LSTP_KEEP_DIR": str(args.keep_dir) if args.keep_dir else "",
    }
    t0 = time.monotonic()
    try:
        proc = subprocess.run(  # nosec B603 -- fixed argv, no shell
            [sys.executable, "-c", CHILD],
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "outcome": "timeout",
            "timeout_s": args.timeout_s,
            "elapsed_s": time.monotonic() - t0,
        }
    stderr_tail = proc.stderr.strip().splitlines()[-8:]
    if proc.returncode != 0:
        return {
            "outcome": "failed",
            "returncode": proc.returncode,
            "elapsed_s": round(time.monotonic() - t0, 1),
            "stderr_tail": stderr_tail,
        }
    last = proc.stdout.strip().splitlines()[-1]
    result = json.loads(last)
    result["outcome"] = "ok"
    result["stderr_tail"] = stderr_tail
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=512, help="band height in pixels")
    parser.add_argument("--col-chunks", type=int, default=8, help="spatial chunks across")
    parser.add_argument("--scenes", type=int, default=1031, help="time steps (S30W065 band: 1031)")
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--rlimit-gb", type=int, default=30, help="RLIMIT_AS for the child")
    parser.add_argument(
        "--rss-cap-gb", type=int, default=24, help="watchdog kills the child above this RSS"
    )
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--bin-s", type=float, default=5.0)
    parser.add_argument(
        "--reject-frac", type=float, default=0.0, help="fraction of scenes de-striping rejects"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="append the result JSON to this file"
    )
    parser.add_argument(
        "--keep-dir",
        type=Path,
        default=None,
        help="write lst_p95.tif and qa_count.tif here and keep them, so two arms can be diffed",
    )
    args = parser.parse_args()

    stack_gb = args.rows * args.col_chunks * args.chunk * args.scenes * 4 / 1024**3
    print(
        json.dumps({"stack_gb_float32": round(stack_gb, 2), "rss_cap_gb": args.rss_cap_gb}),
        file=sys.stderr,
    )
    result = run_point(args)
    line = json.dumps(result)
    print(line)
    if args.out:
        with args.out.open("a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
