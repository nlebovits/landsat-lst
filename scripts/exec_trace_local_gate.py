#!/usr/bin/env python
"""Run the bounded, real-data local gate for the composite execution trace."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from landsat_lst import shard_tasks, shards
from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings
from landsat_lst.job import DEFAULT_WINDOW
from landsat_lst.models import ProcessingJob
from landsat_lst.offsets import OffsetCache
from landsat_lst.shard_tasks import _offset_key, _time_coord, run_shard
from landsat_lst.storage import LocalStorage
from landsat_lst.tiling import parse_tile_name

# The first production trace sampled from an in-process thread and held a mean
# cadence of 1.61 s with 120 gaps above 2 s.  The sampler now runs in a child
# process; this bound is what the contract holds it to on a 25-minute run.
GAP_LIMIT_S = 5.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", default="S30W065")
    parser.add_argument("--max-scenes", type=int, default=40)
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument("--index", type=int, default=16)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _seed_zero_offsets(storage: LocalStorage, plan: shards.TilePlan) -> None:
    coord = _time_coord(plan)
    OffsetCache(storage=storage, key=_offset_key(plan)).write(
        xr.DataArray(
            np.zeros(coord.size, dtype=np.float32),
            dims=["time"],
            coords={"time": coord},
        ),
        xr.DataArray(
            np.full(coord.size, 1000, dtype=np.int64),
            dims=["time"],
            coords={"time": coord},
        ),
    )


def _read_artifacts(storage: LocalStorage, stem: str) -> tuple[dict, list[dict[str, str]]]:
    summary = json.loads((storage.output_dir / f"{stem}.exectrace.summary.json").read_text())
    with (storage.output_dir / f"{stem}.exectrace.timeline.csv").open(newline="") as stream:
        timeline = list(csv.DictReader(stream))
    return summary, timeline


def _gate_results(summary: dict, timeline: list[dict[str, str]]) -> list[tuple[str, bool, str]]:
    per_class = summary["per_class"]
    required = ("read", "rechunk", "compute", "store")
    class_counts = {name: int(per_class[name]["tasks"]) for name in required}
    timestamps = [float(row["timestamp"]) for row in timeline]
    gaps = [right - left for left, right in pairwise(timestamps)]
    spans_compute = bool(timestamps) and (
        timestamps[0] <= summary["compute_started_at"]
        and timestamps[-1] >= summary["compute_finished_at"]
    )
    max_gap = max(gaps, default=0.0)
    thread_count = settings.dask_max_threads or os.cpu_count() or 1
    max_active = max(
        (
            sum(
                int(row[f"active_{name}_tasks"])
                for name in ("read", "rechunk", "compute", "store", "other")
            )
            for row in timeline
        ),
        default=0,
    )
    return [
        (
            "all four main task classes present",
            all(class_counts.values()),
            repr(class_counts),
        ),
        (
            f"host samples span compute with max gap < {GAP_LIMIT_S:.0f} s",
            spans_compute and max_gap < GAP_LIMIT_S,
            (
                f"samples={len(timestamps)}, max_gap_s={max_gap:.3f}, "
                f"sampler={summary.get('host_sampler_mode')}"
            ),
        ),
        (
            "active task count does not exceed thread count",
            max_active <= thread_count,
            f"max_active={max_active}, thread_count={thread_count}",
        ),
        (
            "artifact upload starts after compute",
            summary["upload_started_at"] >= summary["compute_finished_at"],
            (
                f"compute_finished_at={summary['compute_finished_at']:.6f}, "
                f"upload_started_at={summary['upload_started_at']:.6f}"
            ),
        ),
    ]


def main() -> None:
    args = _arguments()
    if not settings.exec_trace:
        raise SystemExit("refusing: set LST_EXEC_TRACE=1")
    if settings.stac_url != STAC_PLANETARY_COMPUTER:
        raise SystemExit("refusing local gate: set LST_STAC_URL to the Planetary Computer endpoint")
    if args.max_scenes < 1 or args.columns < 1:
        raise SystemExit("--max-scenes and --columns must be positive")

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="lst_exec_trace_gate_"))
    storage = LocalStorage(output_dir)
    run_id = args.run_id or f"exectrace-gate-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    job = ProcessingJob(
        tile=parse_tile_name(args.tile),
        year=DEFAULT_WINDOW[0],
        end_year=DEFAULT_WINDOW[1],
        max_scenes=args.max_scenes,
    )

    print(f"output_dir={output_dir}")
    print(f"run_id={run_id}")
    plan = run_shard("resolve", run_id, args.tile, 0, job=job, storage=storage)
    if not 0 <= args.index < len(plan.bands):
        raise SystemExit(
            f"--index {args.index} outside resolved composite bands 0..{len(plan.bands) - 1}"
        )
    _seed_zero_offsets(storage, plan)

    production_geobox_for_bbox = shard_tasks.geobox_for_bbox

    def gate_geobox_for_bbox(bbox, resolution_factor: int = 1):
        geobox = production_geobox_for_bbox(bbox, resolution_factor)
        if resolution_factor == 1:
            return geobox[:, : args.columns]
        return geobox

    shard_module: Any = shard_tasks
    shard_module.geobox_for_bbox = gate_geobox_for_bbox
    band_start, band_stop = plan.bands[args.index]
    print(
        "gate_only_deviation="
        f"production pixel size; rows {band_start}:{band_stop}; columns 0:{args.columns}"
    )
    try:
        run_shard("composite", run_id, args.tile, args.index, storage=storage)
    finally:
        shard_module.geobox_for_bbox = production_geobox_for_bbox

    stem = shards.unit_trace_prefix(run_id, "composite", args.tile, args.index)
    summary, timeline = _read_artifacts(storage, stem)
    print("\nprefix                                      class       tasks     task_s")
    print("-" * 78)
    for row in sorted(summary["prefix_table"], key=lambda item: item["task_seconds"], reverse=True):
        print(
            f"{row['prefix'][:43]:43} {row['class']:9} "
            f"{row['tasks']:8d} {row['task_seconds']:10.2f}"
        )

    print("\nchecks")
    failed = False
    for label, passed, detail in _gate_results(summary, timeline):
        failed |= not passed
        print(f"{'PASS' if passed else 'FAIL'}  {label}: {detail}")
    print(
        "INFO  sampled source reads: "
        f"{summary['n_reads_recorded']}/{summary['n_reads_total']} "
        f"({summary['read_hook_status']}, split recorded for "
        f"{summary['read_split_recorded']})"
    )
    for label in ("read_duration_s", "read_open_s", "read_data_s"):
        q = summary[label]
        print(
            f"INFO  {label}: n={q['n']} p50={q['p50']} p95={q['p95']} p99={q['p99']} max={q['max']}"
        )
    print(
        f"INFO  dask threads: config={summary['dask_num_workers']} "
        f"distinct_worker_threads={summary['distinct_worker_threads']}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
