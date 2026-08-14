"""Build one pipeline graph on synthetic data and report what it cost.

Three tiers of instrument run in this project, and confusing them is its own
failure mode. Graph inspection (:mod:`landsat_lst.profiling`) costs seconds on a
laptop and moves no data. This module is the next tier up: it *executes* a graph
at production geometry against ``dask.array.random`` leaves, so it reports a
real peak RSS without a STAC query or a byte of egress. The tier above it is a
full window on real pixels, and nothing here substitutes for that.

Two callers share the code, which is why it lives in the package rather than in
either of them:

``scripts/synthetic_scaling.py``
    Sweeps scene count on a Coiled VM of the production instance type and fits
    the curve. Answers how far a real peak lands above the static floor.
``tests/benchmark/``
    Pins one small configuration and asserts on bands. Answers whether a change
    moved the number, on a CI runner that cannot reach a production peak and
    should not try.

Every measurement runs in a **fresh subprocess**. ``getrusage`` reports a
high-water mark for the life of the process, so a second configuration measured
inside the first one's interpreter would inherit its peak and draw a flat curve
whatever the truth was.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

__all__ = [
    "CI_GEOMETRY",
    "DEFAULT_SWEEP_SCENES",
    "PRODUCTION_SCENES",
    "Geometry",
    "Measurement",
    "measure",
    "sweep",
    "sweep_report",
]

#: Which graphs a measurement builds and computes.
GRAPH_OFFSETS = "offsets"
GRAPH_COMPOSITE = "composite"
GRAPH_BOTH = "both"

#: The two export paths ADR-013 distinguishes. ``export`` writes both COGs from
#: one ``dask.compute``; ``export_separate`` writes them one at a time, which is
#: the shape that cost a second pass over the native stack. Measured as a pair
#: so the benchmark asserts a relationship rather than a constant.
GRAPH_EXPORT = "export"
GRAPH_EXPORT_SEPARATE = "export_separate"

#: Seconds a child may run before the parent gives up on it. A CI-scale point
#: takes single-digit seconds; a production-scale one on a VM takes minutes.
DEFAULT_TIMEOUT_S = 1800


@dataclass(frozen=True)
class Geometry:
    """One configuration to measure, in the terms that set the memory curve.

    Peak memory during de-striping is ``threads * chunk**2 * scenes * itemsize``
    and does not grow with tile width once there are more blocks than threads.
    So ``blocks`` buys streaming behaviour rather than realism: what has to be
    real is the chunk edge, the thread count, and the depth of the time axis.
    """

    scenes: int
    blocks: int = 8
    chunk: int = 512
    threads: int = 4
    graph: str = GRAPH_BOTH

    @property
    def side(self) -> int:
        """Pixels per side of the square stack."""
        return self.blocks * self.chunk

    @property
    def label(self) -> str:
        return f"{self.graph} {self.scenes}sc {self.side}x{self.side} c{self.chunk} t{self.threads}"


#: The configuration ``tests/benchmark/`` pins. Small enough that a CI runner
#: finishes it in seconds and never approaches its own memory ceiling, and still
#: 16 blocks against 2 threads, so dask streams rather than holding the stack.
#: The chunk edge is halved from production's 512 purely for runtime; every
#: other property of the graph is the one a tile builds.
CI_GEOMETRY = Geometry(scenes=24, blocks=4, chunk=256, threads=2)


@dataclass
class Measurement:
    """What one configuration cost, or why it produced nothing."""

    geometry: Geometry
    peak_rss_mb: float = 0.0
    floor_mb: float = 0.0
    wall_s: float = 0.0
    offset_tasks: int = 0
    composite_tasks: int = 0
    source_blocks: int = 0
    source_reads: int = 0
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def peak_over_floor(self) -> float:
        """How far the measured peak lands above the arithmetic floor.

        The number this whole module exists to produce. The floor is arithmetic
        anybody can do from :func:`landsat_lst.profiling.predict_peak`; how far
        above it a real run lands is what nobody knew.
        """
        return self.peak_rss_mb / self.floor_mb if self.floor_mb else 0.0

    @property
    def native_passes(self) -> float:
        """Passes over the source stack, 1.0 when every block is read once.

        ADR-013's property, measured rather than asserted. A tile used to read
        the full native stack three times and now reads it once; either half of
        that fix reverting gives a pass back.
        """
        return self.source_reads / self.source_blocks if self.source_blocks else 0.0

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["geometry"] = asdict(self.geometry)
        payload["peak_over_floor"] = round(self.peak_over_floor, 3)
        payload["native_passes"] = round(self.native_passes, 3)
        return payload


def _child_source() -> str:
    """The program each subprocess runs: one composite, then its high-water mark.

    Written as a string rather than imported so every child starts from a clean
    interpreter. It prints one JSON object on stdout as its last line; anything
    else it writes is diagnostic noise the parent ignores.

    The counting of source reads is the same trick ``tests/integration/test_cog``
    uses on the export: a ``map_blocks`` tally on the way into the graph counts
    block *executions*, which survives fusion. Fusion renames keys freely, so
    counting task keys would answer a different question.
    """
    return """
import json, os, resource, time

import dask
import numpy as np

scenes = int(os.environ["LSTB_SCENES"])
side = int(os.environ["LSTB_SIDE"])
chunk = int(os.environ["LSTB_CHUNK"])
threads = int(os.environ["LSTB_THREADS"])
graph = os.environ["LSTB_GRAPH"]

from landsat_lst.config import settings

# Pinned before anything builds a graph. A stack chunked differently from a
# real load builds a different graph, and the value of the exercise is that
# it does not.
settings.load_chunk_size = chunk

# The lever itself: the threaded scheduler holds one block per thread.
dask.config.set(scheduler="threads", num_workers=threads)

import xarray as xr

from landsat_lst.normalization import offset_graph
from landsat_lst.pipeline import TIME_CHUNK, _composite_graph
from landsat_lst.profiling import graph_stats, predict_peak, synthetic_dataset
from landsat_lst.qa import apply_qa_mask, convert_to_celsius

assert TIME_CHUNK == 10, f"TIME_CHUNK moved to {TIME_CHUNK}; re-pin the benchmarks"

reads = []


def _tally(block):
    reads.append(1)  # list.append is atomic; the scheduler is threaded
    return block


t0 = time.monotonic()

if graph in ("export", "export_separate"):
    # ADR-013's property, measured rather than asserted. Both COG products
    # descend from one stack, exactly as a real composite's do, so the tally
    # counts passes over the scenes. Counting block executions rather than task
    # keys is what survives fusion, which renames keys freely.
    import tempfile
    from pathlib import Path

    import dask.array as da

    from landsat_lst.cog import cog_export, export_lst_cog, export_qa_cog

    stack = da.random.default_rng(0).random((scenes, side, side), chunks=(TIME_CHUNK, chunk, chunk))
    stack = stack.map_blocks(_tally, dtype=stack.dtype, meta=np.array((), dtype=stack.dtype))
    source_blocks = stack.npartitions
    total = stack.sum(axis=0)
    coords = {
        "latitude": np.linspace(-30.0, -35.0, side),
        "longitude": np.linspace(-65.0, -60.0, side),
    }
    native = xr.Dataset(
        {
            "lst_p95": xr.DataArray(
                (total * 1000).astype(np.uint16), dims=["latitude", "longitude"], coords=coords
            ),
            "qa_count": xr.DataArray(
                da.broadcast_to(total.astype(np.uint8), (12, side, side)).rechunk(
                    (12, chunk, chunk)
                ),
                dims=["month", "latitude", "longitude"],
                coords={"month": np.arange(1, 13), **coords},
            ),
        },
        attrs={"tile": "S30W065", "year": 2021, "window": "2021-2025", "scene_count": scenes},
    )

    with tempfile.TemporaryDirectory() as tmp:
        lst_path, qa_path = Path(tmp) / "lst.tif", Path(tmp) / "qa.tif"
        if graph == "export":
            cog_export(native, lst_path, qa_path)
        else:
            export_lst_cog(native, lst_path)
            export_qa_cog(native, qa_path)

    floor = predict_peak(
        scenes=scenes, chunk_size=chunk, threads=threads, height=side, width=side
    )
    print(json.dumps({
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "wall_s": time.monotonic() - t0,
        "offset_tasks": 0,
        "composite_tasks": 0,
        "source_blocks": source_blocks,
        "source_reads": len(reads),
        "floor_mb": floor.total_bytes / (1024 * 1024),
    }))
    raise SystemExit(0)

data = synthetic_dataset(shape=(side, side), scenes=scenes, chunk_size=chunk)

# Count every materialization of a source block. An explicit meta keeps dask
# from calling _tally once on a sample block to infer the output type, which
# would show up as a phantom extra read.
counted = data["lwir11"].data.map_blocks(
    _tally, dtype=data["lwir11"].dtype, meta=np.array((), dtype=data["lwir11"].dtype)
)
source_blocks = counted.npartitions
data["lwir11"] = (data["lwir11"].dims, counted)

lst = convert_to_celsius(apply_qa_mask(data)["lwir11"])

offset_tasks = composite_tasks = 0
pending = []

if graph in ("offsets", "both"):
    offset, n_valid = offset_graph(lst)
    paired = xr.Dataset({"offset": offset, "n_valid": n_valid})
    offset_tasks = graph_stats(paired).tasks
    pending.append(paired)

if graph in ("composite", "both"):
    composite = _composite_graph(lst)
    composite_tasks = graph_stats(composite).tasks
    pending.append(composite)

# One compute for everything asked for, so the reported peak covers execution
# and the read tally counts passes over the source. Both composite products in
# a single compute is what cog_export does; asking for them one at a time is
# what costs the second pass ADR-013 removed.
dask.compute(*pending)

floor = predict_peak(
    scenes=scenes, chunk_size=chunk, threads=threads, height=side, width=side
)

print(json.dumps({
    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "wall_s": time.monotonic() - t0,
    "offset_tasks": offset_tasks,
    "composite_tasks": composite_tasks,
    "source_blocks": source_blocks,
    "source_reads": len(reads),
    "floor_mb": floor.total_bytes / (1024 * 1024),
}))
"""


def measure(geometry: Geometry, *, timeout_s: int = DEFAULT_TIMEOUT_S) -> Measurement:
    """Run one configuration in a fresh interpreter and collect what it cost.

    Never raises for a child that fails: the error lands on the
    :class:`Measurement` so a sweep reports the points that survived rather than
    dying on the first configuration too large for the machine.

    Args:
        geometry: The configuration to build and compute.
        timeout_s: Seconds before the child is killed.

    Returns:
        The measurement, with ``error`` set if the child produced no payload.
    """
    m = Measurement(geometry=geometry)
    env = {
        "LSTB_SCENES": str(geometry.scenes),
        "LSTB_SIDE": str(geometry.side),
        "LSTB_CHUNK": str(geometry.chunk),
        "LSTB_THREADS": str(geometry.threads),
        "LSTB_GRAPH": geometry.graph,
    }
    started = time.monotonic()
    try:
        # Fixed argv and no shell: the only variable part is the environment.
        proc = subprocess.run(
            [sys.executable, "-c", _child_source()],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        m.error = f"timed out after {timeout_s}s"
        m.wall_s = time.monotonic() - started
        return m

    if proc.returncode != 0:
        m.error = (proc.stderr or "").strip()[-600:] or f"exit {proc.returncode}"
        m.wall_s = time.monotonic() - started
        return m

    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        m.error = "child wrote no payload"
        m.wall_s = time.monotonic() - started
        return m

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        m.error = f"unparsable payload: {e}"
        m.wall_s = time.monotonic() - started
        return m

    m.peak_rss_mb = payload["peak_rss_mb"]
    m.wall_s = payload["wall_s"]
    m.offset_tasks = payload["offset_tasks"]
    m.composite_tasks = payload["composite_tasks"]
    m.source_blocks = payload["source_blocks"]
    m.source_reads = payload["source_reads"]
    m.floor_mb = payload["floor_mb"]
    return m


# ---------------------------------------------------------------------------
# The sweep: scene count against peak RSS, at fixed geometry.
# ---------------------------------------------------------------------------

#: Scene counts to sweep. Spread wide enough that a linear fit means something,
#: low enough at the bottom that a first failure is cheap.
DEFAULT_SWEEP_SCENES = (50, 100, 200, 400, 800)

#: Scenes a five-year land tile pulls, the target every fit extrapolates to.
PRODUCTION_SCENES = 2930

#: How far peak RSS must move across a sweep before a memory fit means anything.
#: Below this the stack fits in RAM, peak is the process baseline, and the line
#: describes the interpreter rather than the pipeline. See ADR-011.
MIN_PEAK_SPREAD = 1.2

#: Memory carried by both entries in ``settings.coiled_vm_types``.
VM_GIB = 64.0


def sweep(
    scene_counts: list[int],
    *,
    blocks: int = 8,
    chunk: int = 512,
    threads: int = 4,
    graph: str = GRAPH_BOTH,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    on_result=None,
) -> list[Measurement]:
    """Measure one configuration per scene count, each in its own interpreter.

    Args:
        scene_counts: Time depths to measure.
        blocks: Blocks per side, in chunks. Eight gives 64 blocks against four
            threads, so dask streams rather than holding the stack.
        chunk: Spatial chunk edge in pixels.
        threads: Concurrent dask threads.
        graph: Which graphs to build. See the ``GRAPH_*`` constants.
        timeout_s: Per-configuration ceiling.
        on_result: Optional callback taking each :class:`Measurement` as it
            lands, so a long sweep can report progress rather than going quiet.

    Returns:
        One measurement per scene count, in the order given. A configuration
        that failed carries its error rather than being dropped.
    """
    results = []
    for n in scene_counts:
        m = measure(
            Geometry(scenes=n, blocks=blocks, chunk=chunk, threads=threads, graph=graph),
            timeout_s=timeout_s,
        )
        results.append(m)
        if on_result is not None:
            on_result(m)
    return results


def _fit(xs: list[int], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and intercept, or zeros when there is nothing to fit."""
    if len(xs) < 2:
        return 0.0, 0.0

    import numpy as np  # noqa: PLC0415

    slope, intercept = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys), 1)
    return float(slope), float(intercept)


def sweep_report(results: list[Measurement], *, target: int = PRODUCTION_SCENES) -> dict:
    """Fit both curves and say what the sweep is entitled to conclude.

    Task count is arithmetic over the graph, so it fits in any regime. Memory
    does not. Below the streaming regime the whole stack fits in RAM, peak RSS
    is the process baseline, and a line fitted through it says nothing -- it can
    slope downward and project a negative tile. Refusing to extrapolate there is
    the whole lesson of ADR-011: a number from the wrong regime is worse than no
    number at all.

    Args:
        results: What :func:`sweep` returned.
        target: Scene count to extrapolate to.

    Returns:
        The fit, plus ``verdict``: one of ``constant_ratio``, ``growing_ratio``,
        or ``not_streaming``. The verdict is the deliverable. A constant ratio
        makes ``predict_peak`` correctable and ``plan`` predictive; a growing one
        localizes a leak to something the model treats as fixed; ``not_streaming``
        means the configuration was too small to answer the question.
    """
    ok = [m for m in results if m.ok]
    if len(ok) < 2:
        return {"verdict": "insufficient", "measured": len(ok)}

    scenes = [m.geometry.scenes for m in ok]
    peaks = [m.peak_rss_mb for m in ok]
    ratios = [m.peak_over_floor for m in ok]

    mem_slope, mem_intercept = _fit(scenes, peaks)
    task_slope, task_intercept = _fit(scenes, [float(m.offset_tasks) for m in ok])
    spread = max(peaks) / min(peaks) if min(peaks) else 0.0

    payload: dict = {
        "scenes": scenes,
        "peak_rss_mb": peaks,
        "peak_over_floor": [round(r, 3) for r in ratios],
        "tasks_per_scene": task_slope,
        "projected_offset_tasks": task_slope * target + task_intercept,
        "peak_spread": spread,
        "target_scenes": target,
    }

    if mem_slope <= 0 or spread < MIN_PEAK_SPREAD:
        payload["verdict"] = "not_streaming"
        payload["streaming_regime"] = False
        return payload

    payload["streaming_regime"] = True
    payload["mem_mb_per_scene"] = mem_slope
    payload["mem_intercept_mb"] = mem_intercept
    payload["projected_peak_mb"] = mem_slope * target + mem_intercept
    payload["projected_fits_vm"] = payload["projected_peak_mb"] / 1024 <= VM_GIB

    # The interpretation the issue asks for, decided here rather than by eye.
    # A ratio that holds across an 8x span of scene counts is a correction
    # factor; one that climbs means something scales that the model treats as
    # fixed, which localizes the leak to the shuffle or the anomaly broadcast.
    ratio_growth = max(ratios) / min(ratios) if min(ratios) else 0.0
    payload["ratio_growth"] = ratio_growth
    payload["mean_peak_over_floor"] = sum(ratios) / len(ratios)
    payload["verdict"] = "growing_ratio" if ratio_growth > 1.5 else "constant_ratio"
    return payload


# ---------------------------------------------------------------------------
# The cloud tier: one VM of the production instance type, about 20 minutes.
# ---------------------------------------------------------------------------


def benchmark_key(run_id: str) -> str:
    """Where a sweep publishes its result.

    Its own prefix rather than a run prefix. ``runs.py`` owns the key grammar
    for tile artifacts and classifies everything under ``_runs/`` as a tile
    attempt; a sweep is not a tile and must not appear in a manifest.
    """
    return f"_benchmarks/{run_id}/synthetic_scaling.json"


def submit_sweep(
    scene_counts: list[int],
    *,
    blocks: int = 8,
    chunk: int = 512,
    threads: int = 4,
    run_id: str | None = None,
) -> dict:
    """Run the sweep on one Coiled VM of the production instance type.

    The dev box is the wrong machine for this question three times over. It
    carries less memory than the VM, so the ceiling under test is unreachable.
    The answer is about production hardware, which is the only hardware whose
    peak RSS matters. And synthetic data means the VM does no I/O, so the run is
    minutes rather than the hours a real load takes -- about 20 minutes and well
    under a dollar, against three hours of local compute that answers a
    different question.

    Retries are pinned to zero whatever ``settings.coiled_retries`` says. A
    retry would restart the VM and destroy the evidence of the failure, which on
    a diagnostic run is the entire product.

    Args:
        scene_counts: Time depths to measure.
        blocks: Blocks per side, in chunks.
        chunk: Spatial chunk edge in pixels.
        threads: Concurrent dask threads on the VM.
        run_id: Run token; generated from a UTC timestamp when omitted.

    Returns:
        ``{"run_id", "cluster_id", "job_id", "key", "command"}``. The result
        lands at ``key`` when the task finishes; read it with
        :func:`fetch_sweep`.

    Raises:
        ImportError: If Coiled is not installed.
    """
    try:
        import coiled  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - coiled is a hard dependency
        msg = "Coiled is required for the cloud tier. Install with: pip install coiled"
        raise ImportError(msg) from e

    from datetime import UTC, datetime  # noqa: PLC0415

    from landsat_lst.config import settings  # noqa: PLC0415
    from landsat_lst.job import _worker_environ  # noqa: PLC0415

    run_id = run_id or f"scaling-{datetime.now(tz=UTC):%Y%m%dT%H%M%SZ}"

    parts = ["python", "-m", "landsat_lst.cli", "benchmark", "--run-id", run_id]
    for n in scene_counts:
        parts += ["--scenes", str(n)]
    parts += ["--blocks", str(blocks), "--chunk", str(chunk), "--threads", str(threads)]
    # A "#!" script is shipped to the VM verbatim. Coiled splits a plain string
    # on whitespace and rejoins a list, and that round trip has mangled quoting
    # before. See issue #66.
    command = "#!/bin/bash\n" + " ".join(parts) + "\n"

    result = coiled.batch_run(
        command=command,
        name=f"lst-bench-{run_id}",
        region=settings.coiled_region,
        vm_type=settings.coiled_vm_types,
        spot_policy=settings.coiled_spot_policy,
        max_workers=1,
        ntasks=1,
        # Not settings.coiled_retries: a failure here is the evidence.
        max_retries=0,
        job_timeout=settings.coiled_job_timeout,
        env=_worker_environ(),
        tag={"project": "landsat-lst", "run_id": run_id, "kind": "benchmark"},
        forward_aws_credentials=False,
    )
    return {
        "run_id": run_id,
        "cluster_id": result.get("cluster_id"),
        "job_id": result.get("job_id"),
        "key": benchmark_key(run_id),
        "command": command,
    }


def fetch_sweep(run_id: str) -> dict | None:
    """Read a published sweep back, or ``None`` if it has not landed yet."""
    from landsat_lst.storage import get_storage  # noqa: PLC0415

    text = get_storage().read_text(benchmark_key(run_id))
    return json.loads(text) if text else None
