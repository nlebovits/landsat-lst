"""Answer cost questions about a tile without paying for a cloud run.

Every question we asked about pipeline cost used to cost a Coiled submission and
twenty minutes. Run ``2021-2025-sample300-20260813T123249Z`` is the pattern: the
de-striping graph turned out to hold 598,604 tasks, and we learned that sixty
seconds into a cloud run. Task count depends on array shape and chunking, never
on pixel values, so it was knowable on a laptop in ten seconds.

This module supplies the two cheap layers underneath a real run.

:func:`graph_stats` reads a dask collection's graph and reports how many tasks
it holds, broken down by key prefix. That breakdown is the part
:class:`~landsat_lst.progress.GraphProgress` cannot give: a heartbeat reading
``4182/18600`` looks identical whether the hour is going into
``median-aggregate``, into ``open_rasterio``, or into a rechunk shuffle.

:func:`synthetic_dataset` builds a stack with production geometry, production
chunking, and a real time axis out of ``dask.array.random``. Feed it to
:func:`~landsat_lst.pipeline.compute_annual_composite` or to
:func:`~landsat_lst.normalization.offset_graph` and you get the same graph a
real tile builds, with no STAC query and no egress. Its source layers are one
task per block, matching a per-block read; what it cannot reproduce is the
handful of extra layers odc-stac wraps each band in. Everything downstream --
the groupby, the medians, the rechunk before the quantile, which is where the
tasks actually live -- is identical.

:func:`predict_peak` states the memory a configuration needs as a **floor**, not
a forecast. The floor is worth having on its own: a configuration whose floor
already exceeds the VM is disqualified for free. It is emphatically not a
measurement. On the 300-scene N40W075 sample the floor came to a few GB against
78.6 GB of observed RSS, and closing that gap is what
``scripts/synthetic_scaling.py`` exists to do. :func:`plan_memory` applies it to
a named tile without building anything, which is cheap enough that a 700-tile
submission can record the plan it was submitted against.

:func:`profile_compute` is the third layer, and the only one that costs a real
run anything. It wraps a compute in ``dask.diagnostics`` profilers and dumps the
summary beside the tile's heartbeat. It is gated on ``settings.profile_dask``
and off by default, and every write is best-effort: instrumentation never fails
a tile. See issue #76.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.progress import active_heartbeat, silence_sections

if TYPE_CHECKING:
    from collections.abc import Iterator

log = structlog.get_logger()

#: Bytes in a gibibyte, so a report can talk in the same units as a VM spec.
GIB = 1024**3

#: Months in the climatology de-striping builds. Fixed whatever the window is:
#: a five-year window pools into the same twelve buckets a one-year window does.
MONTHS = 12

#: Bytes per element once ``convert_to_celsius`` has turned DN into float32.
#: The stack that dominates memory is the Celsius one, not the uint16 source.
ITEMSIZE = 4

#: Peak RSS a tile reaches before it has loaded anything: interpreter, GDAL,
#: rasterio, the land polygons. Measured at roughly 2 GiB on a batch VM, and
#: only a starting constant -- ``predict_peak`` takes it as an argument.
DEFAULT_BASELINE_GIB = 2.0

#: Scenes a five-year land tile pulls, from run ``2021-2025-20260812T142408Z``.
PRODUCTION_SCENES = 2930

#: Scenes ``landsat-lst plan`` uses unless told otherwise, matching the sample
#: in run ``2021-2025-sample300-20260813T123249Z``. Not :data:`PRODUCTION_SCENES`,
#: because planning the full window is not a laptop operation: the composite's
#: quantile rechunks 293 time chunks into one across 1,296 spatial blocks, and
#: one attempt at it ran past fifteen minutes before being killed. A 300-scene
#: plan answers the same shape of question in about half a minute.
DEFAULT_PLAN_SCENES = 300

#: Labels under which :func:`profile_compute` dumps its summaries.
PROFILE_DESTRIPE_OFFSETS = "destripe_offsets"

#: Ceilings on a dumped profile, so instrumentation cannot outgrow what it
#: instruments. The resource curve is strided rather than truncated, because
#: the shape of the whole run is the point of it.
_MAX_PREFIXES = 25
_MAX_CURVE_SAMPLES = 5000


@dataclass(frozen=True)
class PrefixStats:
    """One key prefix's share of a graph.

    A dask key is ``prefix-token``, and the prefix names the operation:
    ``median-aggregate``, ``rechunk-merge``, ``open_rasterio``. Grouping by it
    turns a task count into an answer about which operation the count belongs
    to.

    Counts only. A static graph has no timings, and the wall clock a prefix
    actually consumed comes from :func:`profile_compute` on a real run.
    """

    prefix: str
    tasks: int


@dataclass(frozen=True)
class GraphStats:
    """What a dask graph holds, read without computing any of it.

    ``tasks`` counts the graph **after** ``dask.optimize``, because that is the
    graph the scheduler runs and therefore the one
    :class:`~landsat_lst.progress.GraphProgress` counts against on a live tile.
    A plan that disagreed with the heartbeat it is meant to predict would be
    worse than no plan.

    ``raw_tasks`` is the count before fusion. The two are kept apart rather than
    reconciled by a constant because the ratio is not one: measured at 1.48x on
    the offset graph at 300 scenes, 1.59x at 1,000, and 2.71x on the composite.
    Reading the raw count as a task count reverses a real conclusion -- raw
    makes the composite graph look twice the offset graph, when after fusion
    the two are within 10% of each other.
    """

    tasks: int
    raw_tasks: int
    layers: int
    blocks: int
    by_prefix: tuple[PrefixStats, ...]
    optimized: bool

    @property
    def fusion(self) -> float:
        """How much fusion removed, as ``raw_tasks / tasks``."""
        return self.raw_tasks / self.tasks if self.tasks else 1.0

    def top(self, n: int = 5) -> tuple[PrefixStats, ...]:
        """The ``n`` prefixes holding the most tasks, largest first."""
        return self.by_prefix[:n]

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe view, for a manifest or a sweep table."""
        return {
            "tasks": self.tasks,
            "raw_tasks": self.raw_tasks,
            "optimized": self.optimized,
            "fusion": round(self.fusion, 2),
            "layers": self.layers,
            "blocks": self.blocks,
            "by_prefix": [{"prefix": p.prefix, "tasks": p.tasks} for p in self.by_prefix],
        }


def _block_count(obj: Any) -> int:
    """Blocks in a dask-backed collection, or 0 when it does not report chunks.

    A Dataset is reduced to its widest variable rather than asked for its own
    ``chunks``, which raises whenever two variables disagree along a dim. They
    routinely do here: a reduction that rechunks time sits in the same Dataset
    as one that does not.
    """
    data_vars = getattr(obj, "data_vars", None)
    if data_vars is not None:
        return max((_block_count(v) for v in data_vars.values()), default=0)

    try:
        chunks = getattr(obj, "chunks", None)
    except ValueError:  # pragma: no cover - only reachable on odd collections
        return 0
    if not chunks:
        return 0
    per_dim = chunks.values() if isinstance(chunks, dict) else chunks
    counts = [len(c) for c in per_dim]
    return int(np.prod(counts)) if counts else 0


def _graph_of(obj: Any) -> Any:
    """The dask graph behind a collection, or a TypeError explaining its absence."""
    graph = getattr(obj, "__dask_graph__", None)
    if graph is None or (dsk := graph()) is None:
        msg = (
            f"{type(obj).__name__} has no dask graph. Load it lazily "
            "(chunks=...) before asking what it would cost."
        )
        raise TypeError(msg)
    return dsk


def _count_by_prefix(dsk: Any) -> tuple[Counter[str], int]:
    """Task counts per key prefix, and the number of layers they came from."""
    from dask.utils import key_split  # noqa: PLC0415

    layers = getattr(dsk, "layers", None)
    counts: Counter[str] = Counter()
    if layers:
        for name, layer in layers.items():
            counts[key_split(name)] += len(layer)
        return counts, len(layers)

    # An optimized graph is a flat mapping, so prefixes come from the keys.
    for key in dsk:
        counts[key_split(key)] += 1
    return counts, 1


def graph_stats(obj: Any, *, optimize: bool = True) -> GraphStats:
    """Count the tasks in a lazy collection's graph, grouped by key prefix.

    Costs no data and no network. The whole point is that this is knowable
    before a run rather than sixty seconds into one.

    By default the count is taken after ``dask.optimize``, which is the graph
    the scheduler actually runs. That makes it directly comparable to the live
    fraction :class:`~landsat_lst.progress.GraphProgress` publishes: the
    unoptimized graph for a 300-scene N40W075 offset pass holds 905,923 tasks
    where the run reported 598,604, and optimizing brings the plan to 613,240.
    Fusion is not a constant factor, so the raw count cannot be corrected after
    the fact -- it has to be optimized.

    Optimizing costs real time: about 11 seconds for that 300-scene graph and
    31 for a 1,000-scene one. ``optimize=False`` skips it for a quick sketch,
    and marks the result so nothing downstream mistakes it for a task count.

    Args:
        obj: A dask collection, or an xarray object backed by one.
        optimize: Whether to fuse the graph before counting.

    Returns:
        Task, layer, and block counts, with a per-prefix breakdown sorted by
        task count.

    Raises:
        TypeError: If ``obj`` has no dask graph, which usually means it was
            loaded eagerly and there is nothing to plan.
    """
    raw_dsk = _graph_of(obj)
    raw_counts, layers = _count_by_prefix(raw_dsk)
    raw_tasks = sum(raw_counts.values())

    counts = raw_counts
    if optimize:
        import dask  # noqa: PLC0415

        counts, _ = _count_by_prefix(_graph_of(dask.optimize(obj)[0]))

    return GraphStats(
        tasks=sum(counts.values()),
        raw_tasks=raw_tasks,
        layers=layers,
        blocks=_block_count(obj),
        optimized=optimize,
        by_prefix=tuple(
            PrefixStats(prefix=prefix, tasks=tasks) for prefix, tasks in counts.most_common()
        ),
    )


@dataclass(frozen=True)
class PeakEstimate:
    """A floor on the memory one de-striping configuration needs.

    Three terms, each one something the pipeline demonstrably holds:

    ``stack_bytes``
        The per-block time stacks in flight. Dask's threaded scheduler holds
        one block per thread, and the monthly median needs that block's whole
        time axis at once, so this is ``threads * chunk**2 * scenes *
        itemsize``.
    ``climatology_bytes``
        The per-pixel monthly reference, ``months * height * width *
        itemsize``. Every scene's anomaly reads it, so it stays resident for
        the length of the phase. Unlike the stack term it does not grow with
        the window: twelve buckets hold five years as readily as one.
    ``baseline_bytes``
        Interpreter, GDAL, rasterio, land polygons.

    Read the sum as a floor and nothing more. On the 300-scene N40W075 sample
    it lands far under the 78.6 GB actually observed, and the size of that gap
    is the open question ``scripts/synthetic_scaling.py`` measures. A floor
    still earns its place: a configuration that cannot fit even this is ruled
    out without spending twenty minutes finding out.
    """

    scenes: int
    chunk_size: int
    threads: int
    height: int
    width: int
    months: int
    stack_bytes: int
    climatology_bytes: int
    baseline_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.stack_bytes + self.climatology_bytes + self.baseline_bytes

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    def fits_in(self, gib: float) -> bool:
        """Whether the floor alone leaves room on a VM of ``gib`` memory."""
        return self.total_gib < gib

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe view, in GiB, for a sweep table or a manifest."""
        return {
            "scenes": self.scenes,
            "chunk_size": self.chunk_size,
            "threads": self.threads,
            "stack_gib": round(self.stack_bytes / GIB, 2),
            "climatology_gib": round(self.climatology_bytes / GIB, 2),
            "baseline_gib": round(self.baseline_bytes / GIB, 2),
            "floor_gib": round(self.total_gib, 2),
        }


def predict_peak(
    *,
    scenes: int,
    chunk_size: int,
    threads: int,
    height: int,
    width: int,
    months: int = MONTHS,
    itemsize: int = ITEMSIZE,
    baseline_gib: float = DEFAULT_BASELINE_GIB,
) -> PeakEstimate:
    """Compute the memory floor for one de-striping configuration.

    Args:
        scenes: Scenes pooled into the window.
        chunk_size: Spatial chunk edge in pixels (``settings.load_chunk_size``).
        threads: Concurrent dask threads. Each holds one block.
        height: Rows in the stack offsets are estimated on. At a
            ``destripe_offset_resolution_factor`` above 1 this is the coarse
            grid, not the native tile.
        width: Columns in that same stack.
        months: Buckets in the climatology. Twelve, whatever the window.
        itemsize: Bytes per element after conversion to Celsius.
        baseline_gib: Process memory before any data is loaded.

    Returns:
        The three terms and their sum. See :class:`PeakEstimate` on why the sum
        is a floor rather than a forecast.
    """
    block_pixels = chunk_size * chunk_size
    return PeakEstimate(
        scenes=scenes,
        chunk_size=chunk_size,
        threads=threads,
        height=height,
        width=width,
        months=months,
        stack_bytes=threads * block_pixels * scenes * itemsize,
        climatology_bytes=months * height * width * itemsize,
        baseline_bytes=int(baseline_gib * GIB),
    )


def synthetic_dataset(
    *,
    shape: tuple[int, int],
    scenes: int,
    chunk_size: int | None = None,
    start_year: int = 2021,
    end_year: int = 2025,
    cloud_percent: int = 30,
    seed: int = 0,
) -> xr.Dataset:
    """Build a stack with production geometry and no data behind it.

    The bands, dtypes, dim names, and chunking match what
    :func:`~landsat_lst.pipeline.load_scenes` returns, so the graph built on top
    of this is the graph a real tile builds. What differs is the source: one
    ``dask.array.random`` task per block, where a real load has odc-stac's read
    layers. That changes the leaves, never the reductions above them.

    Scene timestamps are spread evenly across the window, which gives
    ``groupby("time.month")`` all twelve buckets to fill for any window of a
    year or more. Even spacing is also what ``ProcessingJob.max_scenes``
    sampling produces, so a synthetic run and a sampled real run agree on the
    shape of the time axis.

    Args:
        shape: ``(height, width)`` in SOURCE pixels. Pass a real tile's shape
            from :func:`~landsat_lst.tiling.tile_geobox` to plan a real tile.
            Rounded up to a whole number of delivered cells, which a real tile
            shape already is.
        scenes: Time steps to build.
        chunk_size: Spatial chunk edge. Defaults to
            ``settings.load_chunk_size``.
        start_year: First year of the window.
        end_year: Last year of the window, inclusive.
        cloud_percent: Roughly what fraction of pixels carry the cloud bit, so
            the QA mask drops a realistic share rather than nothing.
        seed: Seed for the random source, so a sweep is reproducible.

    Returns:
        A Dataset with ``lwir11`` and ``qa_pixel``, both ``uint16`` on
        ``(time, latitude, longitude)``.

    Raises:
        ValueError: If ``scenes`` or either dimension is not positive.
    """
    import dask.array as da  # noqa: PLC0415

    from landsat_lst.aggregate import aligned_source_chunk  # noqa: PLC0415
    from landsat_lst.pipeline import TIME_CHUNK  # noqa: PLC0415

    height, width = shape
    if scenes < 1 or height < 1 or width < 1:
        msg = f"synthetic_dataset needs a positive shape and scene count, got {shape} x {scenes}"
        raise ValueError(msg)

    # A real source stack always covers a whole number of delivered cells: the
    # source and delivered global grids share an origin and 3600 is 3 x 1200.
    # A synthetic stack that did not would build a graph no tile builds -- it
    # would carry ``coarsen``'s implicit rechunk, which is the one thing the
    # planner must not measure and production never pays. Rounding up rather
    # than refusing keeps every caller's chosen geometry usable. See ADR-017.
    height = aligned_source_chunk(height)
    width = aligned_source_chunk(width)

    csize = aligned_source_chunk(settings.load_chunk_size if chunk_size is None else chunk_size)
    chunks = (min(TIME_CHUNK, scenes), min(csize, height), min(csize, width))
    size = (scenes, height, width)
    rng = da.random.default_rng(seed)

    # DN bounds bracketing roughly 0 C to 50 C under the Collection 2 scaling
    # convert_to_celsius applies, so the plausibility clamp keeps the values
    # rather than dropping the whole stack and shrinking the graph.
    lwir11 = rng.integers(36322, 50953, size=size, chunks=chunks, dtype="uint16")

    # Bit 3 is Cloud. Setting only that bit is enough to exercise create_qa_mask,
    # which reads five bits and combines them into one boolean.
    cloudy = rng.integers(0, 100, size=size, chunks=chunks, dtype="uint16")
    qa_pixel = ((cloudy < cloud_percent) * np.uint16(8)).astype("uint16")

    # Descending latitude, matching the north-down global grid of ADR-008.
    return xr.Dataset(
        {
            "lwir11": (("time", "latitude", "longitude"), lwir11),
            "qa_pixel": (("time", "latitude", "longitude"), qa_pixel),
        },
        coords={
            "time": pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", periods=scenes),
            "latitude": np.linspace(40.0, 35.0, height),
            "longitude": np.linspace(-75.0, -70.0, width),
        },
    )


@contextmanager
def destripe_disabled() -> Iterator[None]:
    """Turn de-striping off for the block, then restore it.

    :func:`~landsat_lst.pipeline.compute_annual_composite` computes the scene
    offsets eagerly, which is exactly what a planner must not do. Disabling
    de-striping leaves the composite half of the pipeline, which stays lazy and
    can be inspected; the offset half is inspected separately through
    :func:`~landsat_lst.normalization.offset_graph`.
    """
    original = settings.destripe
    settings.destripe = False
    try:
        # The composite half runs the same instrumented code a tile does, and
        # nothing here is a tile: `plan --json` promises nothing but JSON on
        # stdout, and one phase_complete line makes it unparsable.
        with silence_sections():
            yield
    finally:
        settings.destripe = original


#: The two graphs a tile builds, in the order it builds them.
PHASE_OFFSETS = "destripe_offsets"
PHASE_COMPOSITE = "composite"

#: Raw tasks one graph may hold before :func:`plan_tile` refuses to build it.
#: Building a graph is Python object allocation and has nothing to do with
#: computing it, so a plan can exhaust a machine that would have run the tile
#: fine. Calibrated against measurements on this repo: an 18,000 squared
#: composite at chunk 512 and 300 scenes holds 1.8M raw tasks in about 2.8 GB,
#: and the same tile at chunk 128 and 2,930 scenes comes to roughly 274M, which
#: took a 64 GB desktop down. 50M sits above every configuration worth planning
#: and below the ones that cannot finish.
MAX_PLAN_TASKS = 50_000_000

#: Raw tasks per block-step (one spatial block at one time chunk), measured on
#: real geometry: ~93 for the offset graph, ~47 for the composite. The larger is
#: used for both, so the estimate errs toward refusing.
_TASKS_PER_BLOCK_STEP = 95


class PlanTooLarge(RuntimeError):
    """Raised when building a graph would cost more memory than it is worth.

    Carries the numbers, so the message can tell the caller which lever to pull
    rather than only that it declined.
    """

    def __init__(self, *, phase: str, estimated: int, limit: int, chunk_size: int, scenes: int):
        self.phase = phase
        self.estimated = estimated
        self.limit = limit
        super().__init__(
            f"Building the {phase} graph would allocate roughly {estimated:,} tasks, "
            f"over the {limit:,} ceiling. Graph construction is Python objects, so "
            f"this can exhaust the machine even though the tile itself would run. "
            f"Raise --chunk above {chunk_size}, drop --scenes below {scenes}, or "
            f"pass --max-tasks to override deliberately."
        )


def estimate_raw_tasks(*, height: int, width: int, chunk_size: int, scenes: int) -> int:
    """Roughly how many raw tasks a graph over this geometry will hold.

    Cheap enough to run before allocating anything, which is the whole point:
    the expensive thing to discover late is that a graph does not fit.

    Args:
        height: Rows in the stack.
        width: Columns in the stack.
        chunk_size: Spatial chunk edge in pixels.
        scenes: Time steps.

    Returns:
        An estimate that errs high, so the guard refuses before it should.
    """
    from landsat_lst.pipeline import TIME_CHUNK  # noqa: PLC0415

    blocks_per_side_h = -(-height // chunk_size)
    blocks_per_side_w = -(-width // chunk_size)
    time_chunks = -(-scenes // TIME_CHUNK)
    return blocks_per_side_h * blocks_per_side_w * time_chunks * _TASKS_PER_BLOCK_STEP


def _guard(
    *, phase: str, height: int, width: int, chunk_size: int, scenes: int, limit: int
) -> None:
    """Refuse to build a graph whose construction would not fit. Never allocates."""
    estimated = estimate_raw_tasks(height=height, width=width, chunk_size=chunk_size, scenes=scenes)
    if estimated > limit:
        raise PlanTooLarge(
            phase=phase,
            estimated=estimated,
            limit=limit,
            chunk_size=chunk_size,
            scenes=scenes,
        )


@dataclass(frozen=True)
class PlanPhase:
    """What one phase of a tile would cost, read off its graph."""

    name: str
    height: int
    width: int
    scenes: int
    graph: GraphStats
    peak: PeakEstimate

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe view, for `landsat-lst plan --json`."""
        return {
            "phase": self.name,
            "shape": [self.height, self.width],
            "scenes": self.scenes,
            "graph": self.graph.as_dict(),
            "memory": self.peak.as_dict(),
        }


def _phase(name: str, collection: Any, *, peak: PeakEstimate, optimize: bool) -> PlanPhase:
    """Pair a built graph with the memory floor for the configuration behind it."""
    return PlanPhase(
        name=name,
        height=peak.height,
        width=peak.width,
        scenes=peak.scenes,
        graph=graph_stats(collection, optimize=optimize),
        peak=peak,
    )


def _levers(
    chunk_size: int | None, threads: int | None, offset_factor: int | None
) -> tuple[int, int, int]:
    """Fill in whatever a caller left to settings, as ``(chunk, threads, factor)``.

    One resolution site, so :func:`plan_memory` and :func:`plan_tile` cannot
    disagree about what "the default configuration" means.
    """
    csize = settings.load_chunk_size if chunk_size is None else chunk_size
    nthreads = threads or settings.dask_max_threads or os.cpu_count() or 1
    factor = settings.destripe_offset_resolution_factor if offset_factor is None else offset_factor
    return csize, nthreads, factor


def plan_memory(
    *,
    tile: Any,
    scenes: int = PRODUCTION_SCENES,
    chunk_size: int | None = None,
    threads: int | None = None,
    offset_factor: int | None = None,
    baseline_gib: float = DEFAULT_BASELINE_GIB,
) -> tuple[PeakEstimate, PeakEstimate]:
    """Price both phases of a tile from arithmetic alone, offsets first.

    Pure arithmetic over the tile's grid. No graph is built and no synthetic
    stack is allocated, so this returns in microseconds even at production
    scene counts. That is what lets a 700-tile submission record the plan it
    submitted against, which is the only way a later reconcile can tell whether
    the planner was right. Recomputing a plan at reconcile time would price
    whatever the settings say then, not what the run actually ran.

    :func:`plan_tile` reports these same two floors alongside task counts, and
    calls this function to get them.

    Args:
        tile: A :class:`~landsat_lst.models.TileId`.
        scenes: Scenes in the window. Defaults to :data:`PRODUCTION_SCENES`.
        chunk_size: Spatial chunk edge. Defaults to ``settings.load_chunk_size``.
        threads: Concurrent dask threads. Defaults to
            ``settings.dask_max_threads``, or the machine's CPU count when that
            is unset, which is what dask itself would use.
        offset_factor: Resolution factor for the offset pass. Defaults to
            ``settings.destripe_offset_resolution_factor``.
        baseline_gib: Process memory before any data is loaded.

    Returns:
        The offsets-phase floor and the composite-phase floor, in that order.
        Each is a floor rather than a forecast. See :class:`PeakEstimate`.
    """
    from landsat_lst.tiling import tile_geobox  # noqa: PLC0415

    csize, nthreads, factor = _levers(chunk_size, threads, offset_factor)
    coarse_h, coarse_w = tile_geobox(tile, factor).shape
    native_h, native_w = tile_geobox(tile).shape

    def floor(*, height: int, width: int, months: int) -> PeakEstimate:
        return predict_peak(
            scenes=scenes,
            chunk_size=csize,
            threads=nthreads,
            height=height,
            width=width,
            months=months,
            baseline_gib=baseline_gib,
        )

    return (
        # Only de-striping builds the float32 monthly climatology. Its blocks
        # are read by every scene's anomaly, so they stay resident and the whole
        # (12, h, w) array is charged. It runs on the coarse offsets grid.
        floor(height=coarse_h, width=coarse_w, months=MONTHS),
        # The composite builds no such array. Its twelve-month band is
        # ``qa_count``, a uint8 result streamed to the COG writer block by
        # block, never a resident float32 cube. Charging it one put 14.5 GiB
        # into every row of a --sweep, swamping the levers the sweep exists to
        # rank: chunk 128 at 8 threads scored better than chunk 512 at 1.
        floor(height=native_h, width=native_w, months=0),
    )


def plan_memory_record(
    *,
    tile: Any,
    scenes: int = PRODUCTION_SCENES,
    chunk_size: int | None = None,
    threads: int | None = None,
    offset_factor: int | None = None,
    baseline_gib: float = DEFAULT_BASELINE_GIB,
) -> dict[str, Any]:
    """A JSON-safe :func:`plan_memory`, for storing on a submission record.

    Carries the levers as well as the floors, because a floor read back without
    the configuration that produced it cannot be argued with.

    Args:
        tile: A :class:`~landsat_lst.models.TileId`.
        scenes: Scenes in the window. Defaults to :data:`PRODUCTION_SCENES`.
        chunk_size: Spatial chunk edge. Defaults to ``settings.load_chunk_size``.
        threads: Concurrent dask threads. Defaults as in :func:`plan_memory`.
        offset_factor: Resolution factor for the offset pass. Defaults to
            ``settings.destripe_offset_resolution_factor``.
        baseline_gib: Process memory before any data is loaded.

    Returns:
        Plain lists, dicts, and numbers, keyed by :data:`PHASE_OFFSETS` and
        :data:`PHASE_COMPOSITE` under ``phases``.
    """
    csize, nthreads, factor = _levers(chunk_size, threads, offset_factor)
    offsets, composite = plan_memory(
        tile=tile,
        scenes=scenes,
        chunk_size=csize,
        threads=nthreads,
        offset_factor=factor,
        baseline_gib=baseline_gib,
    )
    return {
        "scenes": scenes,
        "chunk_size": csize,
        "threads": nthreads,
        "offset_factor": factor,
        "phases": {
            PHASE_OFFSETS: offsets.as_dict(),
            PHASE_COMPOSITE: composite.as_dict(),
        },
    }


def plan_tile(
    *,
    tile: Any,
    scenes: int = PRODUCTION_SCENES,
    chunk_size: int | None = None,
    threads: int | None = None,
    offset_factor: int | None = None,
    baseline_gib: float = DEFAULT_BASELINE_GIB,
    optimize: bool = True,
    max_tasks: int = MAX_PLAN_TASKS,
) -> tuple[PlanPhase, ...]:
    """Build both of a tile's graphs against synthetic data and read their size.

    Costs no network and no pixels. Grids come from
    :func:`~landsat_lst.tiling.tile_geobox`, so the shapes are the tile's real
    ones: 18,000 squared at source resolution, and that divided by
    ``destripe_offset_resolution_factor`` for the offset pass. The composite
    phase loads that same source stack and aggregates it to the delivered
    6,000 squared grid, so its graph is the production one (ADR-017).

    The composite phase is built with de-striping disabled, because
    :func:`~landsat_lst.pipeline.compute_annual_composite` computes the offsets
    eagerly and a planner must not run anything. That also makes its scene count
    an upper bound: in a real run de-striping has already discarded roughly 22%
    of scenes by the time the composite is built.

    Memory floors come from :func:`plan_memory`, which builds nothing. Call that
    directly when the task counts are not worth the graph.

    Args:
        tile: A :class:`~landsat_lst.models.TileId`.
        scenes: Scenes in the window. Defaults to :data:`PRODUCTION_SCENES`.
        chunk_size: Spatial chunk edge. Defaults to ``settings.load_chunk_size``.
        threads: Concurrent dask threads. Defaults to
            ``settings.dask_max_threads``, or the machine's CPU count when that
            is unset, which is what dask itself would use.
        offset_factor: Resolution factor for the offset pass. Defaults to
            ``settings.destripe_offset_resolution_factor``.
        baseline_gib: Process memory before any data is loaded.
        optimize: Fuse each graph before counting, so the totals match what a
            live heartbeat reports. Costs tens of seconds per phase at
            production scene counts.
        max_tasks: Refuse to build a graph estimated above this many raw tasks.

    Returns:
        One :class:`PlanPhase` per phase, offsets first.

    Raises:
        PlanTooLarge: If either graph would allocate past ``max_tasks``. Checked
            before anything is built, so the refusal costs nothing.
    """
    from landsat_lst.normalization import offset_graph  # noqa: PLC0415
    from landsat_lst.pipeline import compute_annual_composite  # noqa: PLC0415
    from landsat_lst.qa import apply_qa_mask, convert_to_celsius  # noqa: PLC0415

    # The floors, the two grids, and the months charged against each all come
    # from plan_memory, so a plan and a stored submission record can never price
    # the same tile differently.
    offsets_peak, composite_peak = plan_memory(
        tile=tile,
        scenes=scenes,
        chunk_size=chunk_size,
        threads=threads,
        offset_factor=offset_factor,
        baseline_gib=baseline_gib,
    )
    csize = offsets_peak.chunk_size

    # Both shapes are checked before either graph is built. A refusal has to
    # arrive before the allocation it is refusing, or it is worth nothing.
    for phase_name, peak in ((PHASE_OFFSETS, offsets_peak), (PHASE_COMPOSITE, composite_peak)):
        _guard(
            phase=phase_name,
            height=peak.height,
            width=peak.width,
            chunk_size=csize,
            scenes=scenes,
            limit=max_tasks,
        )

    coarse_shape = (offsets_peak.height, offsets_peak.width)
    coarse = synthetic_dataset(shape=coarse_shape, scenes=scenes, chunk_size=csize)
    offsets = offset_graph(convert_to_celsius(apply_qa_mask(coarse)["lwir11"]))

    native_shape = (composite_peak.height, composite_peak.width)
    native = synthetic_dataset(shape=native_shape, scenes=scenes, chunk_size=csize)
    with destripe_disabled():
        composite = compute_annual_composite(native)

    return (
        _phase(
            PHASE_OFFSETS,
            xr.Dataset({"offset": offsets[0], "n_valid": offsets[1]}),
            peak=offsets_peak,
            optimize=optimize,
        ),
        _phase(PHASE_COMPOSITE, composite, peak=composite_peak, optimize=optimize),
    )


#: Levers a `--sweep` crosses by default. Chunk size divides the memory floor
#: by its square; thread count divides it linearly and, unlike a smaller chunk,
#: adds no graph nodes. Both are what run 2021-2025-sample300-20260813T123249Z
#: had to test one twenty-minute cloud submission at a time.
SWEEP_CHUNK_SIZES = (512, 256, 128)
SWEEP_THREAD_COUNTS = (1, 2, 4, 8)

#: Memory a batch VM carries. Both entries in ``settings.coiled_vm_types`` are
#: 64 GiB, chosen after a 32 GiB r6i.xlarge OOMed a heavy tile at 28.77 GiB.
DEFAULT_VM_GIB = 64.0


@dataclass(frozen=True)
class SweepRow:
    """One (chunk size, thread count) configuration, priced statically."""

    chunk_size: int
    threads: int
    offsets_tasks: int
    composite_tasks: int
    floor_gib: float
    vm_gib: float
    optimized: bool

    @property
    def fits(self) -> bool:
        """Whether the floor alone leaves room on the VM."""
        return self.floor_gib < self.vm_gib

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "threads": self.threads,
            "offsets_tasks": self.offsets_tasks,
            "composite_tasks": self.composite_tasks,
            "optimized": self.optimized,
            "floor_gib": round(self.floor_gib, 2),
            "fits": self.fits,
        }


def sweep_plan(
    *,
    tile: Any,
    scenes: int = PRODUCTION_SCENES,
    chunk_sizes: tuple[int, ...] = SWEEP_CHUNK_SIZES,
    thread_counts: tuple[int, ...] = SWEEP_THREAD_COUNTS,
    vm_gib: float = DEFAULT_VM_GIB,
    baseline_gib: float = DEFAULT_BASELINE_GIB,
    optimize: bool = True,
    max_tasks: int = MAX_PLAN_TASKS,
) -> tuple[SweepRow, ...]:
    """Price every combination of chunk size and thread count, statically.

    Graphs are built once per chunk size rather than once per row: task count
    follows from array shape and chunking, and the thread count changes only
    how many blocks are in flight. A four-by-three sweep therefore costs three
    graph builds, not twelve.

    Fusion dominates the cost of a sweep, since it runs once per chunk size.
    ``optimize=False`` trades comparable totals for speed, which is a fair trade
    here: a sweep's decision variable is the memory floor, which is exact and
    free either way, and its task counts only have to rank configurations
    against one another. Every row carries ``optimized`` so a raw count is never
    mistaken for one a heartbeat would report.

    Each row's floor is the larger of the two phases, since they run one after
    the other and the binding constraint is whichever peaks higher.

    Args:
        tile: A :class:`~landsat_lst.models.TileId`.
        scenes: Scenes in the window.
        chunk_sizes: Spatial chunk edges to try.
        thread_counts: Concurrent dask threads to try.
        vm_gib: Memory of the VM each row is judged against.
        baseline_gib: Process memory before any data is loaded.
        optimize: Fuse each graph before counting its tasks.

    Returns:
        One row per combination, cheapest floor first.
    """
    rows: list[SweepRow] = []
    for chunk_size in chunk_sizes:
        # A sweep exists to find viable configurations, so one that cannot even
        # be planned is dropped rather than allowed to fail the whole command.
        # It is logged, never silently: a table that quietly lost its smallest
        # chunk would read as though that chunk had been considered and lost.
        try:
            phases = plan_tile(
                tile=tile,
                scenes=scenes,
                chunk_size=chunk_size,
                threads=1,
                baseline_gib=baseline_gib,
                optimize=optimize,
                max_tasks=max_tasks,
            )
        except PlanTooLarge as e:
            log.warning(
                "sweep_chunk_skipped",
                chunk_size=chunk_size,
                scenes=scenes,
                estimated_tasks=e.estimated,
                limit=e.limit,
            )
            continue
        tasks = {phase.name: phase.graph.tasks for phase in phases}
        for threads in thread_counts:
            # months comes from the phase, not the default: only de-striping
            # holds a monthly climatology, and charging the composite one too
            # added a constant 14.5 GiB to every row and flattened the table.
            floor = max(
                predict_peak(
                    scenes=scenes,
                    chunk_size=chunk_size,
                    threads=threads,
                    height=phase.height,
                    width=phase.width,
                    months=phase.peak.months,
                    baseline_gib=baseline_gib,
                ).total_gib
                for phase in phases
            )
            rows.append(
                SweepRow(
                    chunk_size=chunk_size,
                    threads=threads,
                    offsets_tasks=tasks.get(PHASE_OFFSETS, 0),
                    composite_tasks=tasks.get(PHASE_COMPOSITE, 0),
                    floor_gib=floor,
                    vm_gib=vm_gib,
                    optimized=optimize,
                )
            )
    return tuple(sorted(rows, key=lambda r: r.floor_gib))


def _prefix_seconds(results: list) -> tuple[Counter[str], defaultdict[str, float]]:
    """Task counts and total wall seconds per key prefix, from Profiler results."""
    from dask.utils import key_split  # noqa: PLC0415

    counts: Counter[str] = Counter()
    seconds: defaultdict[str, float] = defaultdict(float)
    for entry in results:
        prefix = key_split(entry.key)
        counts[prefix] += 1
        seconds[prefix] += entry.end_time - entry.start_time
    return counts, seconds


def _task_summary(results: list) -> dict[str, Any]:
    """Where the wall clock went, by task prefix.

    Sorted by seconds rather than by count, because the question this answers
    is which operation owns the hour, and dask tasks are wildly uneven.
    """
    counts, seconds = _prefix_seconds(results)
    total = sum(seconds.values())
    ranked = sorted(seconds.items(), key=lambda kv: kv[1], reverse=True)[:_MAX_PREFIXES]
    return {
        "total": len(results),
        "total_task_seconds": round(total, 2),
        "by_prefix": [
            {
                "prefix": prefix,
                "tasks": counts[prefix],
                "seconds": round(secs, 2),
                "mean_s": round(secs / counts[prefix], 4),
                "share": round(secs / total, 4) if total else 0.0,
            }
            for prefix, secs in ranked
        ],
    }


def _resource_summary(results: list, interval_s: float) -> dict[str, Any]:
    """The RSS and CPU curve we currently reconstruct by hand from heartbeats.

    The curve is strided rather than cut short: a peak that arrives in the last
    minute of a two-hour phase is exactly the peak worth keeping.
    """
    if not results:
        return {"samples": 0}
    stride = max(1, len(results) // _MAX_CURVE_SAMPLES)
    start = results[0].time
    return {
        "samples": len(results),
        "interval_s": interval_s,
        "stride": stride,
        "peak_mem_mb": round(max(r.mem for r in results), 1),
        "mean_cpu_pct": round(sum(r.cpu for r in results) / len(results), 1),
        "curve": [
            [round(r.time - start, 1), round(r.mem, 1), round(r.cpu, 1)] for r in results[::stride]
        ],
    }


def _cache_summary(results: list) -> dict[str, Any]:
    """Bytes dask held in memory, and which prefixes held them.

    ``peak_bytes`` is a sweep over cache and free events rather than a sum: the
    direct answer to "why is RSS climbing" is how much was resident at once,
    not how much passed through.
    """
    if not results:
        return {"entries": 0}
    events: list[tuple[float, int]] = []
    held: defaultdict[str, int] = defaultdict(int)

    from dask.utils import key_split  # noqa: PLC0415

    for entry in results:
        metric = int(entry.metric)
        events.append((entry.cache_time, metric))
        events.append((entry.free_time, -metric))
        held[key_split(entry.key)] += metric

    events.sort()
    running = peak = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)

    ranked = sorted(held.items(), key=lambda kv: kv[1], reverse=True)[:_MAX_PREFIXES]
    return {
        "entries": len(results),
        "peak_bytes": peak,
        "peak_gib": round(peak / GIB, 3),
        "by_prefix": [{"prefix": p, "bytes": b} for p, b in ranked],
    }


def _profile_destination(label: str) -> tuple[Any, str] | None:
    """Where a profile dump goes: beside the tile's heartbeat, or on local disk.

    A batch tile publishes into its run prefix, so the profile lands with the
    heartbeat and the log an operator is already reading. A local run has no
    run prefix, so it falls back to the manifest directory rather than
    declining to profile at all.
    """
    heartbeat = active_heartbeat()
    if heartbeat is not None:
        key = heartbeat.storage.profile_key(
            heartbeat.run_id, heartbeat.tile, label, heartbeat.attempt
        )
        return heartbeat.storage, key

    from landsat_lst.storage import LocalStorage  # noqa: PLC0415

    return LocalStorage(output_dir=settings.manifest_dir / "profiles"), f"{label}.profile.json"


@contextmanager
def profile_compute(label: str) -> Iterator[None]:
    """Profile the dask compute in this block, per task key.

    Off unless ``settings.profile_dask`` is set, and inert either way from the
    caller's point of view: the block runs, and a profiler that cannot start or
    a dump that cannot be written is logged and swallowed. Instrumentation
    never fails a tile.

    ``CacheProfiler`` is behind its own ``settings.profile_dask_cache`` because
    it retains one record per task. At the 598,604 tasks the de-striping graph
    reached on N40W075, on a run already close to its memory ceiling, that is
    not a cost to take by default.

    Args:
        label: Names the dumped object, e.g. ``destripe_offsets``.
    """
    if not settings.profile_dask:
        yield
        return

    try:
        from dask.diagnostics import CacheProfiler, Profiler, ResourceProfiler  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - dask.diagnostics ships with dask
        log.warning("profile_unavailable", label=label, error=str(e))
        yield
        return

    interval = settings.profile_dask_interval_s
    task_prof = Profiler()
    resource_prof = ResourceProfiler(dt=interval)
    cache_prof = CacheProfiler() if settings.profile_dask_cache else None

    # Each profiler is started on its own and its failure is swallowed, rather
    # than one `with` covering all three. ResourceProfiler spawns a tracker
    # process and raises from __enter__ when psutil is missing, and a profiler
    # that cannot start must cost the tile nothing. One that fails simply
    # contributes no results to the dump.
    stack = ExitStack()
    for prof in (task_prof, resource_prof, cache_prof):
        if prof is None:
            continue
        try:
            stack.enter_context(prof)
        except Exception as e:
            log.warning(
                "profile_start_failed", label=label, profiler=type(prof).__name__, error=str(e)
            )

    started = time.monotonic()
    try:
        yield
    finally:
        with suppress(Exception):
            stack.close()
        _dump_profile(
            label=label,
            wall_s=time.monotonic() - started,
            task_prof=task_prof,
            resource_prof=resource_prof,
            cache_prof=cache_prof,
            interval_s=interval,
        )
        # The resource profiler runs a tracker process; clearing it releases
        # that as well as the retained samples.
        for prof in (task_prof, resource_prof, cache_prof):
            if prof is not None:
                with suppress(Exception):
                    prof.clear()


def _dump_profile(
    *,
    label: str,
    wall_s: float,
    task_prof: Any,
    resource_prof: Any,
    cache_prof: Any,
    interval_s: float,
) -> None:
    """Summarize and store one profile. Never raises."""
    try:
        heartbeat = active_heartbeat()
        payload: dict[str, Any] = {
            "label": label,
            "wall_s": round(wall_s, 2),
            "run_id": heartbeat.run_id if heartbeat else None,
            "tile": heartbeat.tile if heartbeat else None,
            "tasks": _task_summary(task_prof.results),
            "resource": _resource_summary(resource_prof.results, interval_s),
        }
        if cache_prof is not None:
            payload["cache"] = _cache_summary(cache_prof.results)

        destination = _profile_destination(label)
        if destination is None:  # pragma: no cover - both branches return one
            return
        storage, key = destination
        storage.write_text(key, json.dumps(payload, indent=2))
        log.info(
            "dask_profile_written",
            label=label,
            key=key,
            tasks=payload["tasks"]["total"],
            peak_mem_mb=payload["resource"].get("peak_mem_mb"),
        )
    except Exception as e:
        log.warning("dask_profile_failed", label=label, error=str(e))
