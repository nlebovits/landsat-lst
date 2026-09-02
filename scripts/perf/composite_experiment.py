"""Local harness for the pre-registered composite performance experiment.

Implements the measurement protocol of ``docs/perf/composite-experiment-contract.md``
(sections 3, 4 and 5) and nothing else. It never touches the network, never
starts a cluster, and refuses any configuration whose projected peak exceeds the
contract's 20 GB cap.

Three entry points:

``pin-offsets``
    Estimate the offset vector for one fixture **once** and write it to
    ``results/perf/offsets-<tile>.json``. The estimator is out of scope for the
    experiment (contract section 3), so every subsequent run reads this file
    rather than re-estimating.

``child``
    One measured configuration, in its own interpreter. ``getrusage`` reports a
    high-water mark for the life of a process, so a second configuration
    measured inside the first one's interpreter inherits its peak; this is why
    the parent never measures anything itself.

``sweep``
    The parent. Spawns three fresh children per configuration and appends one
    JSON object per run to ``results/perf/composite-experiment.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

PERF_DIR = Path("results/perf")

#: Contract section 3: refuse to launch any configuration whose projected peak
#: exceeds this.
PEAK_CAP_GB = 20.0

#: Multiplier over the per-block time stack (``threads x chunk^2 x scenes x 4``).
#: The contract's section 3 carries 3.9 from E5, which was synthetic. The first
#: baseline run on the real S30W065 fixture (chunk 512, 4 threads, 300 scenes)
#: peaked at 12,096 MB against a 1,258 MB per-block stack -- 9.6x, not 3.9x. The
#: guard uses the measured number, because a launch bound that under-predicts is
#: not a bound. At 9.6x, 8 threads projects to 24.2 GB and is refused.
PEAK_MULTIPLIER = 9.6


def projected_peak_gb(*, chunk: int, threads: int, scenes: int) -> float:
    """The contract's launch bound: threads x chunk^2 x scenes x 4 B x 3.9."""
    return threads * chunk * chunk * scenes * 4 * PEAK_MULTIPLIER / 1e9


def _spec(name: str):
    from landsat_lst.fixture import FixtureSpec  # noqa: PLC0415

    tile, window, scenes, factor = name.split("_", 3)
    year, end_year = window.split("-")
    return FixtureSpec(
        tile=tile,
        year=int(year),
        end_year=int(end_year),
        max_scenes=int(scenes.removeprefix("n")),
        factor=int(factor.removeprefix("f")),
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_mb() -> float:
    """``VmHWM`` from procfs, ``ru_maxrss`` as the no-procfs fallback.

    Verbatim in behaviour from ``landsat_lst.benchmarks._child_source._peak_mb``:
    Linux does not reset ``ru_maxrss`` across ``execve``, so a child forked from
    a fat parent reports the parent's high-water mark. ``VmHWM`` comes from the
    ``mm`` that ``execve`` created and is the child's own truth.
    """
    try:
        with Path("/proc/self/status").open() as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _loadavg() -> list[float]:
    """The three ``/proc/loadavg`` figures, or an empty list off Linux.

    Captured on both sides of the timed section. This laptop is shared with
    other agents' jobs, and without a contention record a reviewer cannot tell a
    real speedup from a quiet window.
    """
    try:
        return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]
    except (OSError, ValueError):
        return []


def _machine() -> dict:
    model = "unknown"
    mem_total_kb = 0
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
                break
    except OSError:
        pass
    return {"cpu_model": model, "cpu_count": os.cpu_count(), "mem_total_kb": mem_total_kb}


def _provenance() -> dict:
    import dask  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "git_sha": sha,
        # The arm under test is a working-tree edit to one file, so the git sha
        # alone cannot tell two arms apart. This can.
        "pipeline_sha256": _sha256_file(Path("src/landsat_lst/pipeline.py")),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "dask": dask.__version__,
        "xarray": xr.__version__,
        **_machine(),
    }


# --------------------------------------------------------------------------
# Offsets: estimated once, pinned forever after.
# --------------------------------------------------------------------------


def pin_offsets(fixture: str) -> Path:
    """Estimate and persist one fixture's offset vector.

    Written with times as nanosecond ISO strings for provenance only. The child
    rebuilds the offset axis from the *fixture's own* time coordinate and
    asserts the stored stamps match it exactly, so the JSON round trip cannot
    silently shift the join (see the offsets rule in CLAUDE.md).
    """
    import numpy as np  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    from landsat_lst.fixture import load_fixture  # noqa: PLC0415
    from landsat_lst.normalization import scene_offsets  # noqa: PLC0415
    from landsat_lst.qa import apply_qa_mask, convert_to_celsius  # noqa: PLC0415

    spec = _spec(fixture)
    data = load_fixture(spec)
    lst = convert_to_celsius(apply_qa_mask(data)["lwir11"])

    started = time.monotonic()
    offset, n_valid = scene_offsets(lst)
    duration = time.monotonic() - started

    times = xr.DataArray(lst.time).values.astype("datetime64[ns]").astype("int64")
    payload = {
        "fixture": fixture,
        "times_ns": [int(t) for t in times],
        "offset": [None if not np.isfinite(v) else float(v) for v in np.asarray(offset.values)],
        "n_valid": [int(v) for v in np.asarray(n_valid.values)],
        "duration_s": duration,
        "provenance": _provenance(),
    }
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    out = PERF_DIR / f"offsets-{spec.tile}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def _load_pinned_offsets(fixture: str, time_coord):
    """Rebuild ``(offset, n_valid)`` on the fixture's own time axis."""
    import numpy as np  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    tile = fixture.split("_", 1)[0]
    path = PERF_DIR / f"offsets-{tile}.json"
    payload = json.loads(path.read_text())

    stored = np.asarray(payload["times_ns"], dtype="int64")
    actual = np.asarray(time_coord.values).astype("datetime64[ns]").astype("int64")
    if not np.array_equal(stored, actual):
        msg = f"pinned offsets for {tile} carry a different time axis than the fixture"
        raise ValueError(msg)

    offset = xr.DataArray(
        np.array([np.nan if v is None else v for v in payload["offset"]], dtype="float64"),
        dims=["time"],
        coords={"time": time_coord},
    )
    n_valid = xr.DataArray(
        np.asarray(payload["n_valid"], dtype="int64"),
        dims=["time"],
        coords={"time": time_coord},
    )
    return offset, n_valid, _sha256_file(path)


# --------------------------------------------------------------------------
# The child: one configuration, one interpreter.
# --------------------------------------------------------------------------


def child(  # noqa: PLR0915 -- one straight-line measurement; splitting it would
    # move allocations into another frame and change what the peak measures.
    fixture: str,
    chunk: int,
    threads: int,
    truth: str | None,
    arm: str,
) -> dict:
    """Measure one configuration and emit one JSON object on stdout."""
    import dask  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from landsat_lst.config import settings  # noqa: PLC0415

    # Pinned before anything builds a graph: a stack chunked differently from a
    # real load builds a different graph.
    settings.load_chunk_size = chunk
    dask.config.set(scheduler="threads", num_workers=threads)

    from landsat_lst.encoding import encode_lst_uint16  # noqa: PLC0415
    from landsat_lst.fixture import load_fixture  # noqa: PLC0415
    from landsat_lst.normalization import debias_with_offsets  # noqa: PLC0415
    from landsat_lst.pipeline import TIME_CHUNK, _composite_graph  # noqa: PLC0415
    from landsat_lst.profiling import graph_stats, predict_peak  # noqa: PLC0415
    from landsat_lst.qa import apply_qa_mask, convert_to_celsius  # noqa: PLC0415

    assert TIME_CHUNK == 10, f"TIME_CHUNK moved to {TIME_CHUNK}; re-pin the experiment"

    reads: list[int] = []

    def _tally(block):
        reads.append(1)  # list.append is atomic; the scheduler is threaded
        return block

    spec = _spec(fixture)
    data = load_fixture(spec)
    scenes, height, width = (int(n) for n in data["lwir11"].shape)

    projected = projected_peak_gb(chunk=chunk, threads=threads, scenes=scenes)
    if projected > PEAK_CAP_GB:
        msg = f"projected peak {projected:.1f} GB exceeds the contract cap of {PEAK_CAP_GB} GB"
        raise SystemExit(msg)

    # Count every materialization of a source block. An explicit meta keeps dask
    # from calling _tally once on a sample block to infer the output type, which
    # would show up as a phantom extra read.
    counted = data["lwir11"].data.map_blocks(
        _tally, dtype=data["lwir11"].dtype, meta=np.array((), dtype=data["lwir11"].dtype)
    )
    source_blocks = int(counted.npartitions)
    data["lwir11"] = (data["lwir11"].dims, counted)

    lst = convert_to_celsius(apply_qa_mask(data)["lwir11"])

    offset, n_valid, offsets_sha = _load_pinned_offsets(fixture, lst.time)
    lst, _, keep = debias_with_offsets(
        lst,
        offset,
        n_valid,
        max_offset_c=settings.destripe_max_offset_c,
        min_scene_pixels=settings.destripe_min_scene_pixels,
        min_offset_samples=settings.destripe_min_offset_samples,
        offset_source_given=True,
    )

    t_build = time.monotonic()
    composite = _composite_graph(lst)
    graph_build_s = time.monotonic() - t_build

    composite_tasks = graph_stats(composite, optimize=True).tasks

    load_before = _loadavg()
    t0 = time.monotonic()
    (result,) = dask.compute(composite)
    wall_s = time.monotonic() - t0
    load_after = _loadavg()

    ru = resource.getrusage(resource.RUSAGE_SELF)
    cpu_s = ru.ru_utime + ru.ru_stime

    lst_p95 = result["lst_p95"]
    qa_count = result["qa_count"]

    # Shipped dimension order and dtype, checked explicitly: the transpose under
    # test is exactly the kind of change that would silently return qa_count as
    # (latitude, longitude, month).
    shape_checks = {
        "lst_p95_dims": list(lst_p95.dims),
        "lst_p95_dtype": str(lst_p95.dtype),
        "qa_count_dims": list(qa_count.dims),
        "qa_count_dtype": str(qa_count.dtype),
    }
    assert shape_checks["lst_p95_dims"] == ["latitude", "longitude"], shape_checks
    assert shape_checks["qa_count_dims"] == ["month", "latitude", "longitude"], shape_checks
    assert lst_p95.dtype == np.float32, shape_checks
    assert qa_count.dtype == np.uint8, shape_checks

    lst_arr = np.ascontiguousarray(lst_p95.values)
    qa_arr = np.ascontiguousarray(qa_count.values)
    enc_arr = np.ascontiguousarray(encode_lst_uint16(lst_p95).values)

    row = {
        "arm": arm,
        "fixture": fixture,
        "chunk": chunk,
        "threads": threads,
        "scenes": scenes,
        "height": height,
        "width": width,
        "scenes_kept": int(np.asarray(keep.values).sum()),
        "wall_s": wall_s,
        "graph_build_s": graph_build_s,
        "peak_rss_mb": _peak_mb(),
        "cpu_s": cpu_s,
        "cores_busy": cpu_s / wall_s if wall_s else 0.0,
        "source_blocks": source_blocks,
        "source_reads": len(reads),
        "native_passes": len(reads) / source_blocks if source_blocks else 0.0,
        "composite_tasks": composite_tasks,
        "floor_mb": predict_peak(
            scenes=scenes, chunk_size=chunk, threads=threads, height=height, width=width
        ).total_bytes
        / (1024 * 1024),
        "projected_peak_gb": projected,
        "lst_sha256": hashlib.sha256(lst_arr.tobytes()).hexdigest(),
        "qa_sha256": hashlib.sha256(qa_arr.tobytes()).hexdigest(),
        "encoded_sha256": hashlib.sha256(enc_arr.tobytes()).hexdigest(),
        "offsets_sha256": offsets_sha,
        "loadavg_before": load_before,
        "loadavg_after": load_after,
        "provenance": _provenance(),
    }

    if truth:
        truth_path = Path(truth)
        if truth_path.exists():
            ref = np.load(truth_path)
            row["equal_lst"] = bool(np.array_equal(lst_arr, ref["lst_p95"], equal_nan=True))
            row["equal_qa"] = bool(np.array_equal(qa_arr, ref["qa_count"]))
            row["equal_encoded"] = bool(np.array_equal(enc_arr, ref["encoded"]))
            row["bit_identical"] = row["equal_lst"] and row["equal_qa"] and row["equal_encoded"]
        else:
            truth_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(truth_path, lst_p95=lst_arr, qa_count=qa_arr, encoded=enc_arr)
            row["truth_written"] = str(truth_path)

    return row


# --------------------------------------------------------------------------
# The parent: three fresh children per configuration.
# --------------------------------------------------------------------------


def sweep(
    fixtures: list[str],
    *,
    chunk: int,
    thread_list: list[int],
    arm: str,
    reps: int,
    rep_offset: int = 0,
) -> int:
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = PERF_DIR / "composite-experiment.jsonl"
    failures = 0

    for fixture in fixtures:
        tile = fixture.split("_", 1)[0]
        truth = PERF_DIR / f"truth-{tile}.npz"
        for threads in thread_list:
            projected = projected_peak_gb(chunk=chunk, threads=threads, scenes=300)
            if projected > PEAK_CAP_GB:
                print(f"SKIP {fixture} c{chunk} t{threads}: projected {projected:.1f} GB > cap")
                continue
            walls, peaks = [], []
            for rep in range(reps):
                proc = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "child",
                        "--fixture",
                        fixture,
                        "--chunk",
                        str(chunk),
                        "--threads",
                        str(threads),
                        "--truth",
                        str(truth),
                        "--arm",
                        arm,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    failures += 1
                    print(proc.stdout[-4000:], file=sys.stderr)
                    print(proc.stderr[-4000:], file=sys.stderr)
                    print(f"FAIL {fixture} c{chunk} t{threads} rep{rep}")
                    break
                row = json.loads(proc.stdout.strip().splitlines()[-1])
                row["rep"] = rep + rep_offset
                row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                with jsonl.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                walls.append(row["wall_s"])
                peaks.append(row["peak_rss_mb"])
                flag = "" if row.get("bit_identical", True) else "  BIT-DIFFERENCE"
                print(
                    f"{arm} {fixture} c{chunk} t{threads} rep{rep}: "
                    f"wall {row['wall_s']:.2f}s peak {row['peak_rss_mb']:.0f}MB "
                    f"passes {row['native_passes']:.2f} tasks {row['composite_tasks']}{flag}"
                )
                if not row.get("bit_identical", True):
                    failures += 1
            if walls:
                spread = max(walls) / min(walls)
                print(
                    f"  -> median wall {median(walls):.2f}s "
                    f"[{min(walls):.2f}-{max(walls):.2f}, spread {spread:.2f}x], "
                    f"median peak {median(peaks):.0f}MB"
                )
                if spread > 1.3:
                    print("  -> SPREAD ABOVE 1.3x: row invalid on a contended machine")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pin = sub.add_parser("pin-offsets")
    p_pin.add_argument("--fixture", required=True)

    p_child = sub.add_parser("child")
    p_child.add_argument("--fixture", required=True)
    p_child.add_argument("--chunk", type=int, default=512)
    p_child.add_argument("--threads", type=int, default=4)
    p_child.add_argument("--truth", default=None)
    p_child.add_argument("--arm", default="baseline")

    p_sweep = sub.add_parser("sweep")
    p_sweep.add_argument("--fixture", action="append", required=True)
    p_sweep.add_argument("--chunk", type=int, default=512)
    p_sweep.add_argument("--threads", default="4")
    p_sweep.add_argument("--arm", required=True)
    p_sweep.add_argument("--reps", type=int, default=3)
    p_sweep.add_argument("--rep-offset", type=int, default=0)

    args = parser.parse_args()

    if args.cmd == "pin-offsets":
        print(pin_offsets(args.fixture))
        return 0
    if args.cmd == "child":
        print(json.dumps(child(args.fixture, args.chunk, args.threads, args.truth, args.arm)))
        return 0
    threads = [int(t) for t in args.threads.split(",")]
    return (
        1
        if sweep(
            args.fixture,
            chunk=args.chunk,
            thread_list=threads,
            arm=args.arm,
            reps=args.reps,
            rep_offset=args.rep_offset,
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
