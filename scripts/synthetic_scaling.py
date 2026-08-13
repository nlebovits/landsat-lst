"""Measure how tasks and peak memory grow with scene count, on synthetic data.

``scripts/measure_memory_scaling.py`` answered this question against a 0.25
degree AOI over Philadelphia, on the assumption that "the **ratios** are what
transfer". They do not. Below roughly one degree the whole time stack fits in
RAM and dask never streams; a five degree tile streams from the first block. The
old script measures a regime production never runs in, so its ratios describe
somebody else's pipeline. It is deprecated in favour of this one. See ADR-011.

What transfers is geometry. Peak memory during de-striping is set by the chunk
edge, the thread count, and the depth of the time axis, none of which need real
pixels behind them. So this builds the stack out of ``dask.array.random`` at
production chunking and runs the real
:func:`~landsat_lst.pipeline.compute_annual_composite` against it: same graph,
same memory curve, no STAC query and no egress.

The spatial extent is a knob rather than the full 18,000 squared tile, and that
is not the flaw the old script had. Peak memory is ``threads * chunk**2 *
scenes * itemsize`` and does not grow with tile width once there are more blocks
than threads -- wall clock does. So the default 8 x 8 blocks keeps every
regime-defining property (production chunk edge, sixteen times more blocks than
in-flight threads, real time depth) while finishing in minutes. Pass
``--blocks 36`` for the true tile width and a much longer wait.

Each configuration runs in a **fresh subprocess**. ``getrusage`` reports a
high-water mark for the whole process, so a second configuration inside the
first one's interpreter would inherit its peak and draw a flat curve whatever
the truth was.

    uv run python scripts/synthetic_scaling.py
    uv run python scripts/synthetic_scaling.py --scenes 50 100 200
    uv run python scripts/synthetic_scaling.py --blocks 16 --threads 8

The report fits peak RSS against scene count, extrapolates to the 2,930 scenes a
five-year land tile pulls, and prints that beside the static floor from
:func:`landsat_lst.profiling.predict_peak`. The ratio between them is the number
this script exists to produce: the floor is arithmetic anybody can do, and how
far above it a real run lands is what nobody knew.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

#: Scene counts to sweep. Spread wide enough that a linear fit means something,
#: and low enough at the bottom that a first failure is cheap.
DEFAULT_SCENES = [50, 100, 200, 400, 800]

#: Blocks per side, in production-sized chunks. Eight gives 64 blocks against a
#: default four threads: comfortably more blocks than threads, so dask streams.
DEFAULT_BLOCKS = 8

#: Threads for the local scheduler. Matches the cap a batch tile now runs under.
DEFAULT_THREADS = 4

#: Scenes a five-year land tile pulls, the target every fit extrapolates to.
PRODUCTION_SCENES = 2930

#: How far peak RSS must move across a sweep before a memory fit means anything.
#: Below this the stack fits in RAM, peak is the process baseline, and the line
#: describes the interpreter rather than the pipeline.
MIN_PEAK_SPREAD = 1.2

OUT_JSON = Path("results/decision/synthetic_scaling.json")


@dataclass
class Measurement:
    """One composite built under one configuration, in its own interpreter."""

    scenes: int
    blocks: int
    chunk_size: int
    threads: int
    height: int = 0
    width: int = 0
    offset_tasks: int = 0
    composite_tasks: int = 0
    peak_rss_mb: float = 0.0
    floor_mb: float = 0.0
    wall_s: float = 0.0
    error: str | None = None


def _child_source() -> str:
    """The program each subprocess runs: one synthetic composite, then its peak.

    Written as a string rather than imported so every child starts from a clean
    interpreter with its own high-water mark. The child prints one JSON object
    on stdout; anything else it writes is diagnostic noise the parent ignores.
    """
    return """
import json, os, resource, time

import dask

scenes = int(os.environ["SS_SCENES"])
height = int(os.environ["SS_HEIGHT"])
width = int(os.environ["SS_WIDTH"])
chunk = int(os.environ["SS_CHUNK"])
threads = int(os.environ["SS_THREADS"])

from landsat_lst.config import settings

settings.load_chunk_size = chunk

# The lever itself: the threaded scheduler holds one block per thread.
dask.config.set(scheduler="threads", num_workers=threads)

import xarray as xr

from landsat_lst.normalization import offset_graph
from landsat_lst.pipeline import compute_annual_composite
from landsat_lst.profiling import graph_stats, predict_peak, synthetic_dataset
from landsat_lst.qa import apply_qa_mask, convert_to_celsius

t0 = time.monotonic()
data = synthetic_dataset(shape=(height, width), scenes=scenes, chunk_size=chunk)

# The coarse second stack de-striping estimates its offsets from, exactly as
# process_tile builds it. Leaving it out would measure a pipeline that is not
# the one that ran out of memory.
factor = settings.destripe_offset_resolution_factor
offset_source = None
if settings.destripe and factor > 1:
    offset_source = synthetic_dataset(
        shape=(height // factor, width // factor), scenes=scenes, chunk_size=chunk
    )

source = data if offset_source is None else offset_source
offsets = offset_graph(convert_to_celsius(apply_qa_mask(source)["lwir11"]))
offset_stats = graph_stats(xr.Dataset({"offset": offsets[0], "n_valid": offsets[1]}))

# No land mask: this is a memory measurement, and a synthetic stack has no
# coastline for one to mean anything against.
composite = compute_annual_composite(data, offset_source=offset_source)
composite_stats = graph_stats(composite)
composite["lst_p95"].compute()

floor = predict_peak(
    scenes=scenes,
    chunk_size=chunk,
    threads=threads,
    height=source.sizes["latitude"],
    width=source.sizes["longitude"],
)

print(json.dumps({
    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "wall_s": time.monotonic() - t0,
    "offset_tasks": offset_stats.tasks,
    "composite_tasks": composite_stats.tasks,
    "floor_mb": floor.total_bytes / (1024 * 1024),
}))
"""


def _run_one(*, scenes: int, blocks: int, chunk: int, threads: int) -> Measurement:
    """Run one configuration in a fresh interpreter and collect its peak RSS."""
    side = blocks * chunk
    m = Measurement(
        scenes=scenes,
        blocks=blocks,
        chunk_size=chunk,
        threads=threads,
        height=side,
        width=side,
    )
    label = f"{scenes:>4} scenes  {side}x{side}"
    env = {
        "SS_SCENES": str(scenes),
        "SS_HEIGHT": str(side),
        "SS_WIDTH": str(side),
        "SS_CHUNK": str(chunk),
        "SS_THREADS": str(threads),
    }
    print(f"  {label}: running...", flush=True)
    started = time.monotonic()
    # Fixed argv and no shell: the only variable part is the environment.
    proc = subprocess.run(
        [sys.executable, "-c", _child_source()],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        m.error = (proc.stderr or "").strip()[-400:] or f"exit {proc.returncode}"
        m.wall_s = time.monotonic() - started
        print(f"  {label}: FAILED ({m.error.splitlines()[-1] if m.error else '?'})", flush=True)
        return m

    payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
    m.peak_rss_mb = payload["peak_rss_mb"]
    m.wall_s = payload["wall_s"]
    m.offset_tasks = payload["offset_tasks"]
    m.composite_tasks = payload["composite_tasks"]
    m.floor_mb = payload["floor_mb"]
    print(
        f"  {label}: {m.peak_rss_mb / 1024:.1f} GB peak "
        f"({m.peak_rss_mb / m.floor_mb:.1f}x floor), "
        f"{m.offset_tasks:,} offset tasks, {m.wall_s / 60:.1f} min",
        flush=True,
    )
    return m


def _fit(xs: list[int], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and intercept, or zeros when there is nothing to fit."""
    if len(xs) < 2:
        return 0.0, 0.0
    slope, intercept = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys), 1)
    return float(slope), float(intercept)


def _report(results: list[Measurement], target: int) -> dict[str, float]:
    """Print the table, fit both curves, and extrapolate to a production tile."""
    print("\n" + "=" * 78)
    print(f"{'scenes':>8}{'peak GB':>10}{'floor GB':>11}{'ratio':>8}{'offset tasks':>16}{'min':>8}")
    print("-" * 78)
    for m in results:
        if m.error:
            print(f"{m.scenes:>8}{'FAILED':>10}")
            continue
        print(
            f"{m.scenes:>8}{m.peak_rss_mb / 1024:>10.1f}{m.floor_mb / 1024:>11.1f}"
            f"{m.peak_rss_mb / m.floor_mb:>8.1f}{m.offset_tasks:>16,}{m.wall_s / 60:>8.1f}"
        )
    print("=" * 78)

    ok = [m for m in results if not m.error]
    if len(ok) < 2:
        print("\nToo few configurations survived to fit a curve.")
        return {}

    scenes = [m.scenes for m in ok]
    peaks = [m.peak_rss_mb for m in ok]
    mem_slope, mem_intercept = _fit(scenes, peaks)
    task_slope, task_intercept = _fit(scenes, [float(m.offset_tasks) for m in ok])
    projected_tasks = task_slope * target + task_intercept
    mean_ratio = float(np.mean([m.peak_rss_mb / m.floor_mb for m in ok]))

    # Task count is arithmetic over the graph, so it fits in any regime.
    print(f"\nDe-striping tasks grow at {task_slope:,.0f} per scene.")
    print(f"At {target:,} scenes that projects to {projected_tasks:,.0f} offset tasks.")

    # Memory does not. Below the streaming regime the whole stack fits in RAM,
    # peak RSS is just the process baseline, and a line fitted through it says
    # nothing -- it can even slope downward and project a negative tile. Refusing
    # to extrapolate is the whole lesson of ADR-011: a number from the wrong
    # regime is worse than no number.
    spread = max(peaks) / min(peaks) if min(peaks) else 0.0
    if mem_slope <= 0 or spread < MIN_PEAK_SPREAD:
        print(
            f"\nPeak RSS barely moved across this sweep "
            f"({min(peaks) / 1024:.1f} -> {max(peaks) / 1024:.1f} GB, "
            f"{spread:.2f}x). The stack still fits in RAM at this geometry, so "
            "dask never streams and there is no memory scaling to fit. This is "
            "the regime ADR-011 warns about, and no projection is printed for "
            "it. Raise --blocks or --scenes until peak RSS climbs."
        )
        return {
            "tasks_per_scene": task_slope,
            "projected_offset_tasks": projected_tasks,
            "mean_peak_over_floor": mean_ratio,
            "peak_spread": spread,
            "streaming_regime": False,
        }

    projected_mb = mem_slope * target + mem_intercept
    print(
        f"\nPeak RSS grows at {mem_slope:.1f} MB per scene "
        f"(intercept {mem_intercept / 1024:.1f} GB)."
    )
    print(f"At {target:,} scenes this geometry projects to {projected_mb / 1024:.1f} GB.")
    print(f"Measured peak runs {mean_ratio:.1f}x the static floor across the sweep.")
    # 64 GiB is what both entries in settings.coiled_vm_types carry.
    if projected_mb / 1024 > 64:
        print(
            "That does not fit a 64 GiB VM. Cut threads or chunk size and "
            "re-run, or plan on a larger instance."
        )
    return {
        "mem_mb_per_scene": mem_slope,
        "mem_intercept_mb": mem_intercept,
        "tasks_per_scene": task_slope,
        "projected_peak_mb": projected_mb,
        "projected_offset_tasks": projected_tasks,
        "mean_peak_over_floor": mean_ratio,
        "peak_spread": spread,
        "streaming_regime": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, nargs="+", default=DEFAULT_SCENES)
    parser.add_argument(
        "--blocks",
        type=int,
        default=DEFAULT_BLOCKS,
        help="Blocks per side. 36 is a real 5-degree tile at chunk 512.",
    )
    parser.add_argument("--chunk", type=int, default=512, help="Spatial chunk edge in px")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--target-scenes", type=int, default=PRODUCTION_SCENES)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    side = args.blocks * args.chunk
    print(
        f"Synthetic sweep at {side}x{side} px "
        f"({args.blocks**2} blocks of {args.chunk}), {args.threads} threads:"
    )
    results = [
        _run_one(scenes=n, blocks=args.blocks, chunk=args.chunk, threads=args.threads)
        for n in args.scenes
    ]
    fit = _report(results, args.target_scenes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "blocks": args.blocks,
                "chunk_size": args.chunk,
                "threads": args.threads,
                "target_scenes": args.target_scenes,
                "fit": fit,
                "measurements": [asdict(m) for m in results],
            },
            indent=2,
        )
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
