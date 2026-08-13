"""Measure how peak memory scales with window length and the levers against it.

A single-year N40W075 tile reached 44 GB of RSS while de-striping on a 64 GiB
VM (run ``2024-20260813T084842Z``). The production window is five years. Whether
that fits depends on a number nobody has measured: how peak memory grows with
the number of scenes pooled.

Two answers are possible and they lead to opposite decisions.

* If peak grows with the **stack** (scene count), five years needs roughly five
  times the memory and no 64 GiB VM will hold a heavy tile.
* If peak is dominated by the **per-pixel monthly climatology** de-striping
  builds -- ``(12, lat, lon)``, fixed whatever the window -- then five years
  costs little more than one, and the current VM is fine.

So this script sweeps the window length and reports the growth, then sweeps the
two levers that would buy headroom if growth turns out to be steep:

* ``load_chunk_size``: peak scales with chunk area, so 512 -> 256 is ~4x. Phase
  0 measured the cost at ~1.7x slower loads (docs/findings-phase0.md).
* dask's thread count: the threaded scheduler materializes one chunk per thread,
  so capping threads cuts concurrent chunks proportionally. Nothing in the
  pipeline sets this today, which makes it the cheapest untried lever.

The AOI is small on purpose. Absolute numbers here are not tile numbers; the
**ratios** are what transfer, and a small AOI makes each run minutes rather than
half an hour. Measure the shape of the curve cheaply, then apply it to the tile
peak already measured on the VM.

Peak RSS is read from ``resource.getrusage``, the same source as the heartbeat's
``peak_rss_mb``, so a number here is comparable to a number from a real run. It
is a high-water mark for the whole process, so each configuration runs in a
**fresh subprocess**: one run inside another's interpreter would inherit its
high-water mark and report a flat curve no matter what the truth was.

    uv run python scripts/measure_memory_scaling.py
    uv run python scripts/measure_memory_scaling.py --windows 1 3 5

Planetary Computer per CLAUDE.md: Earth Search costs egress from a laptop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# A 0.25 degree AOI over Philadelphia: dense Landsat coverage, and the urban
# scene this product exists to describe. Small enough that a five-year window
# is minutes, large enough that the climatology is not a rounding error.
DEFAULT_BBOX = [-75.25, 39.87, -75.0, 40.12]
DEFAULT_START = 2021

#: Window lengths in years. The first is the baseline every ratio is taken
#: against.
DEFAULT_WINDOWS = [1, 2, 3, 5]

#: Levers swept at the longest window, where headroom actually matters.
DEFAULT_CHUNK_SIZES = [512, 256]
DEFAULT_THREADS = [0, 2]  # 0 means "leave dask's default alone"

OUT_JSON = Path("results/decision/memory_scaling.json")


@dataclass
class Measurement:
    """One pipeline run under one configuration."""

    label: str
    years: int
    chunk_size: int
    threads: int
    peak_rss_mb: float = 0.0
    wall_s: float = 0.0
    scenes: int = 0
    error: str | None = None


def _child_source() -> str:
    """The program each subprocess runs: one composite, then its peak RSS.

    Written as a string rather than imported so the parent can hand every child
    a clean interpreter. The child prints one JSON object on stdout; anything
    else it writes is diagnostic noise the parent ignores.
    """
    return """
import json, os, resource, sys, time

bbox = json.loads(os.environ["MS_BBOX"])
year = int(os.environ["MS_YEAR"])
end_year = int(os.environ["MS_END_YEAR"])
threads = int(os.environ["MS_THREADS"])

import dask
from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings

settings.stac_url = STAC_PLANETARY_COMPUTER
settings.load_chunk_size = int(os.environ["MS_CHUNK"])

# The lever itself: the threaded scheduler materializes one chunk per thread.
if threads:
    dask.config.set(scheduler="threads", num_workers=threads)

import pystac_client
from landsat_lst.azure_auth import enable_pc_azure_refresh
from landsat_lst.pipeline import compute_annual_composite, load_scenes

t0 = time.monotonic()
catalog = pystac_client.Client.open(settings.stac_url)
items = list(
    catalog.search(
        collections=[settings.collection],
        bbox=bbox,
        datetime=f"{year}-01-01/{end_year}-12-31",
        query={
            "eo:cloud_cover": {"lt": settings.max_cloud_cover},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
    ).items()
)
if not items:
    raise SystemExit("no scenes")

patch_url = enable_pc_azure_refresh(items)
data = load_scenes(items, bbox, patch_url=patch_url, fail_on_error=False)

# The coarse second load de-striping estimates its offsets from, exactly as
# process_tile builds it. Leaving it out would measure a pipeline that is not
# the one that ran out of memory.
offset_source = None
factor = settings.destripe_offset_resolution_factor
if settings.destripe and factor > 1:
    offset_source = load_scenes(
        items, bbox, patch_url=patch_url, fail_on_error=False, resolution_factor=factor
    )

# No land mask: this is a memory measurement, and an inland AOI would make the
# masked and unmasked paths identical anyway.
composite = compute_annual_composite(data, offset_source=offset_source)
composite["lst_p95"].compute()

print(json.dumps({
    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "wall_s": time.monotonic() - t0,
    "scenes": len(items),
}))
"""


def _run_one(
    *, label: str, bbox: list[float], year: int, end_year: int, chunk: int, threads: int
) -> Measurement:
    """Run one configuration in a fresh interpreter and collect its peak RSS."""
    m = Measurement(label=label, years=end_year - year + 1, chunk_size=chunk, threads=threads)
    env = {
        "MS_BBOX": json.dumps(bbox),
        "MS_YEAR": str(year),
        "MS_END_YEAR": str(end_year),
        "MS_CHUNK": str(chunk),
        "MS_THREADS": str(threads),
    }
    print(f"  {label}: running...", flush=True)
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", _child_source()],
        env={**dict(__import__("os").environ), **env},
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        m.error = (proc.stderr or "").strip()[-400:] or f"exit {proc.returncode}"
        m.wall_s = time.monotonic() - started
        print(f"  {label}: FAILED ({m.error.splitlines()[-1] if m.error else '?'})")
        return m

    # The child's JSON is its last stdout line; warnings may precede it.
    payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
    m.peak_rss_mb = payload["peak_rss_mb"]
    m.wall_s = payload["wall_s"]
    m.scenes = payload["scenes"]
    print(
        f"  {label}: {m.peak_rss_mb / 1024:.1f} GB peak, {m.wall_s / 60:.1f} min, {m.scenes} scenes"
    )
    return m


def _report(results: list[Measurement], baseline: Measurement | None) -> None:
    """Print the ratios, which are the only transferable part of the run."""
    print("\n" + "=" * 68)
    print(f"{'configuration':<28}{'peak GB':>10}{'vs base':>10}{'min':>8}{'scenes':>9}")
    print("-" * 68)
    for m in results:
        if m.error:
            print(f"{m.label:<28}{'FAILED':>10}")
            continue
        ratio = (
            f"{m.peak_rss_mb / baseline.peak_rss_mb:.2f}x"
            if baseline and baseline.peak_rss_mb
            else "-"
        )
        print(
            f"{m.label:<28}{m.peak_rss_mb / 1024:>10.1f}{ratio:>10}"
            f"{m.wall_s / 60:>8.1f}{m.scenes:>9}"
        )
    print("=" * 68)

    windows = [m for m in results if m.chunk_size == DEFAULT_CHUNK_SIZES[0] and not m.threads]
    if len(windows) >= 2 and windows[0].peak_rss_mb:
        first, last = windows[0], windows[-1]
        mem_growth = last.peak_rss_mb / first.peak_rss_mb
        scene_growth = (last.scenes / first.scenes) if first.scenes else 0
        print(
            f"\nOver {first.years} -> {last.years} years: "
            f"{scene_growth:.1f}x scenes, {mem_growth:.1f}x memory."
        )
        # A stack-bound peak tracks scene count; a climatology-bound one does not.
        if scene_growth and mem_growth / scene_growth > 0.6:
            print(
                "Peak tracks the scene stack. Extrapolate the tile's measured "
                "peak by the same factor and expect a 64 GiB VM to be too small."
            )
        else:
            print(
                "Peak grows far slower than the stack, so it is dominated by "
                "fixed per-pixel state rather than by pooled scenes."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=DEFAULT_CHUNK_SIZES)
    parser.add_argument("--threads", type=int, nargs="+", default=DEFAULT_THREADS)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    bbox = list(args.bbox)
    results: list[Measurement] = []

    print(f"Window sweep at chunk={args.chunk_sizes[0]}, dask defaults:")
    for years in args.windows:
        end = args.start_year + years - 1
        results.append(
            _run_one(
                label=f"{years}y chunk={args.chunk_sizes[0]}",
                bbox=bbox,
                year=args.start_year,
                end_year=end,
                chunk=args.chunk_sizes[0],
                threads=0,
            )
        )

    baseline = next((m for m in results if not m.error), None)
    longest = max(args.windows)
    end = args.start_year + longest - 1

    print("\nLever sweep at the longest window:")
    for chunk in args.chunk_sizes[1:]:
        results.append(
            _run_one(
                label=f"{longest}y chunk={chunk}",
                bbox=bbox,
                year=args.start_year,
                end_year=end,
                chunk=chunk,
                threads=0,
            )
        )
    for threads in args.threads:
        if not threads:
            continue
        results.append(
            _run_one(
                label=f"{longest}y threads={threads}",
                bbox=bbox,
                year=args.start_year,
                end_year=end,
                chunk=args.chunk_sizes[0],
                threads=threads,
            )
        )
    # Both levers together, which is the configuration a heavy tile would use.
    if len(args.chunk_sizes) > 1 and any(args.threads):
        smallest_chunk = args.chunk_sizes[-1]
        fewest = max(t for t in args.threads if t)
        results.append(
            _run_one(
                label=f"{longest}y chunk={smallest_chunk} threads={fewest}",
                bbox=bbox,
                year=args.start_year,
                end_year=end,
                chunk=smallest_chunk,
                threads=fewest,
            )
        )

    _report(results, baseline)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "bbox": bbox,
                "start_year": args.start_year,
                "measurements": [asdict(m) for m in results],
            },
            indent=2,
        )
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
