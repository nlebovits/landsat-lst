"""One VM's concurrency ladder: per-arm read throughput vs in-flight requests.

Runs on the Coiled VM. Each arm executes in a fresh subprocess (``getrusage``
is a process-lifetime high-water mark) and reads a **disjoint** slice of the
scene list keyed to the arm index, so no arm is served from a previous arm's
cache and no object is requested twice. The control arm re-runs the first
configuration last: if first and last disagree, the arms between them measured
drift, not concurrency (the GDAL A/B's ``baseline_repeat`` lesson).

Arms vary the two request-side levers Stage 2 exists to size:

- ``io``: width of the thread pool the read runs under, which is the number
  of concurrent S3 range requests in flight.
- ``chunk``: ``settings.load_chunk_size``, which sets bytes per request
  (~86 KB at 256 on the factor-2 grid; 4x that at 512, at a quarter the
  request count).

Output: one JSON line per arm tagged ``"phase": "arm"``; the driver collects
them from the batch log and does the projection arithmetic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time

TILE = os.environ.get("PROBE_TILE", "N40W075")
SCENES_PER_ARM = int(os.environ.get("PROBE_SCENES_PER_ARM", "24"))
FACTOR = int(os.environ.get("PROBE_FACTOR", "2"))

#: (io_threads, chunk) per arm; arm 0 is a discardable warmup, the last arm
#: is a warm control. History: v1 (cluster 1944816) showed request size
#: dominates -- 55.8 MB/s at (32, 512) against a flat 12-24 MB/s across the
#: whole thread column at chunk 256 -- with a 1.81x first-run warming drift.
#: v2 (cluster 1944879) walked the request axis up: (16, 1024) = 155.8 MB/s
#: at 3.9 cores, and throughput fell monotonically with MORE threads at
#: every chunk (16 > 32 > 64 > 128). v3 asks the two questions that size
#: Stage 3: does 8 beat 16, and what are the warm rates for the two shapes
#: production can actually run -- chunk 1024 for the offset pass (a 1024
#: block edge costs ~12.3 GB per unit, so ~4 workers on 64 GiB) and chunk
#: 512 for the composite (the single-time-chunk rechunk makes a 1024^2 task
#: hold 12.3 GB, which caps the composite at 512).
#: Overridable via PROBE_ARMS ("io:chunk,io:chunk,..."), so a variant probe
#: (native factor, other VM type) reuses this harness without editing it.
_DEFAULT_ARMS: list[tuple[int, int]] = [
    (16, 1024),  # warmup -- discard; absorbs first-run warming
    (8, 1024),
    (16, 1024),
    (8, 512),
    (16, 512),
    (16, 1024),  # control: warm repeat of the reference arm
]


def _parse_arms(spec: str) -> list[tuple[int, int]]:
    return [(int(io), int(chunk)) for io, chunk in (arm.split(":") for arm in spec.split(","))]


ARMS = _parse_arms(os.environ["PROBE_ARMS"]) if os.environ.get("PROBE_ARMS") else _DEFAULT_ARMS

_CHILD = """
    import json, resource, sys, time
    from pathlib import Path

    def net_rx():
        total = 0
        for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
            name, _, rest = line.partition(":")
            if name.strip() != "lo":
                total += int(rest.split()[0])
        return total

    io_threads = {io}
    chunk = {chunk}
    arm_index = {arm_index}

    from concurrent.futures import ThreadPoolExecutor
    import dask
    import numpy as np
    from landsat_lst.config import settings
    # Both fields, because load_scenes reads load_chunk_size_offsets for a
    # coarse (factor > 1) load and load_chunk_size for a native one; the
    # arm's chunk must apply whichever path the factor takes.
    settings.load_chunk_size = chunk
    settings.load_chunk_size_offsets = chunk
    from landsat_lst.models import ProcessingJob
    from landsat_lst.pipeline import load_scenes, query_stac
    from landsat_lst.progress import silence_sections
    from landsat_lst.tiling import parse_tile_name

    tile = parse_tile_name({tile!r})
    with silence_sections():
        items = query_stac(ProcessingJob(tile=tile, year=2021, end_year=2025))
    lo = arm_index * {per_arm}
    mine = items[lo : lo + {per_arm}]
    ds = load_scenes(mine, tile.bbox, fail_on_error=False,
                     resolution_factor={factor})
    band = ds["lwir11"].data
    decoded = int(np.prod(band.shape)) * band.dtype.itemsize

    pool = ThreadPoolExecutor(max_workers=io_threads)
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    rx0 = net_rx()
    w0 = time.monotonic()
    checksum = float(
        dask.compute(band.sum(dtype="float64"),
                     scheduler="threads", pool=pool)[0]
    )
    wall = time.monotonic() - w0
    rx = net_rx() - rx0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    print(json.dumps({{
        "io_threads": io_threads,
        "chunk": chunk,
        "arm_index": arm_index,
        "scenes": len(mine),
        "wall_s": round(wall, 1),
        "decoded_bytes": decoded,
        "decoded_mb_s": round(decoded / wall / 1e6, 2),
        "wire_bytes": rx,
        "wire_mb_s": round(rx / wall / 1e6, 2),
        "cpu_s": round(cpu, 1),
        "cpu_cores_busy": round(cpu / wall, 2),
        "peak_rss_gb": round(r1.ru_maxrss / 1024**2, 2),
        "checksum": checksum,
    }}))
"""


def main() -> int:
    for n, (io_threads, chunk) in enumerate(ARMS):
        src = textwrap.dedent(
            _CHILD.format(
                io=io_threads,
                chunk=chunk,
                arm_index=n,
                tile=TILE,
                per_arm=SCENES_PER_ARM,
                factor=FACTOR,
            )
        )
        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-c", src],
            capture_output=True,
            text=True,
            timeout=2400,
            check=False,
        )
        row: dict = {
            "phase": "arm",
            "arm": n,
            "io_threads": io_threads,
            "chunk": chunk,
            "elapsed_s": round(time.monotonic() - t0, 1),
        }
        if proc.returncode == 0:
            try:
                row.update(json.loads(proc.stdout.strip().splitlines()[-1]))
            except (json.JSONDecodeError, IndexError):
                row["error"] = f"unparseable stdout: {proc.stdout[-500:]}"
        else:
            row["error"] = proc.stderr[-1500:]
        # SlowDown / 503 evidence, if GDAL logged any.
        throttles = proc.stderr.count("SlowDown") + proc.stderr.count("503")
        row["throttle_mentions"] = throttles
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
