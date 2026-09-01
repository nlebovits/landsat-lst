"""Season-aware per-scene normalization (de-striping).

Landsat Collection 2 Level-2 surface temperature is produced one scene at a
time, and its atmospheric correction carries a per-scene error of roughly 1-5 K
that applies uniformly across the whole scene. WRS-2 footprints are tilted
about 10 degrees from vertical, so the set of scenes contributing to a pixel
changes abruptly at footprint edges. Those per-scene biases therefore surface
as diagonal rectangular seams in the composite. More years do not average them
away; the seams are structural.

The correction estimates one scalar offset per scene from its deviation from a
per-pixel *monthly* climatology and subtracts it. Because the same constant
applies to every pixel in the scene, it shifts only the scene's baseline: it
does not alter within-scene contrasts, create or erase hot spots, or change
spatial structure. The monthly reference is what makes this safe -- an annual
reference would absorb the seasonal cycle itself and cool the composite badly
(measured: 40.6 C to 29.8 C, spatial r=0.44 against baseline).

Scenes whose correction cannot be trusted are **discarded, never clamped**.
Clamping a -73 C offset to a cap of -15 C would leave roughly 58 C of
uncorrected bias in the stack, which is worse than dropping the scene. See
issue #46 and ADR-007.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import dask
import numpy as np
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.kernels import sort_median_axis0
from landsat_lst.profiling import PROFILE_DESTRIPE_OFFSETS, profile_compute
from landsat_lst.progress import GraphProgress, report_phase, timed_section
from landsat_lst.shards import block_spans

if TYPE_CHECKING:
    from landsat_lst.offsets import OffsetCache

_TIME_DIM = "time"

# One shared executor for every unit read in the process. dask's threaded
# scheduler accepts it via ``pool=`` and just submits into it, so concurrent
# ``get`` calls from several unit workers share one bounded set of I/O
# threads. Without it, each non-main thread that computes a graph gets its
# own per-thread pool (dask.threaded keys pools by thread), and the in-flight
# request count multiplies uncontrolled. The pool is I/O-sized, not CPU-sized:
# these threads spend their time in GIL-released S3 reads, and the number of
# them is the number of concurrent range requests -- the lever the 2026-08
# investigation found capped at 4 while the VM used ~1% of its NIC.
_io_pool_lock = threading.Lock()
_io_pool: ThreadPoolExecutor | None = None


def _read_pool() -> ThreadPoolExecutor:
    global _io_pool  # noqa: PLW0603
    with _io_pool_lock:
        if _io_pool is None or _io_pool._max_workers != settings.destripe_io_threads:
            _io_pool = ThreadPoolExecutor(
                max_workers=settings.destripe_io_threads,
                thread_name_prefix="lst-unit-io",
            )
        return _io_pool


def _read_values(da: xr.DataArray, dtype: np.dtype) -> np.ndarray:
    """Materialize one unit's slice, routing dask reads through the I/O pool."""
    data = da.data
    if dask.is_dask_collection(data):
        (arr,) = dask.compute(data, scheduler="threads", pool=_read_pool())
        return np.asarray(arr, dtype=dtype)
    return np.asarray(da.values, dtype=dtype)


def _unit_workers(unit_bytes: int) -> int:
    """How many work units may run concurrently.

    Each worker holds at most one unit resident, so in-flight memory is
    ``workers x unit_bytes``, and the count is clamped so that product stays
    inside ``destripe_total_memory_gb`` -- a fixed aggregate bound, not a
    per-unit one, because unit sizes span 3.2 GB (a phase-B batch) to
    12.3 GB (a chunk-1024 phase-A block) and multiplying a per-unit budget
    by the worker count would let the large units overrun the VM.
    """
    configured = settings.destripe_unit_workers or min(8, os.cpu_count() or 8)
    total = int(settings.destripe_total_memory_gb * 1024**3)
    return max(1, min(configured, total // max(unit_bytes, 1)))


def _spatial_dims(lst: xr.DataArray) -> list[str]:
    """Every dim but time. Avoids hard-coding latitude/longitude naming."""
    return [str(d) for d in lst.dims if d != _TIME_DIM]


def _pixel_count(lst: xr.DataArray) -> int:
    """Pixels in one time slice."""
    return int(np.prod([lst.sizes[d] for d in _spatial_dims(lst)]))


def _offset_graph_groupby(lst: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """The original groupby formulation, kept as the equivalence oracle.

    Production does not call this: at 2,930 scenes the groupby shuffle costs
    more than 48.5 GB of graph to materialize, which is what killed every
    full-window run (measured 2026-08-14, results/batch1-investigation/).
    ``tests/unit/test_destripe_normalization.py`` pins :func:`offset_graph`
    against this bit for bit, which is the claim that lets the reformulation
    ship without an :data:`~landsat_lst.offsets.ALGORITHM_VERSION` bump.
    """
    spatial = _spatial_dims(lst)
    ref_month = lst.groupby("time.month").median(skipna=True)
    anomaly = lst.groupby("time.month") - ref_month
    return anomaly.median(dim=spatial, skipna=True), lst.notnull().sum(dim=spatial)


def offset_graph(lst: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Build the lazy ``(offset, n_valid)`` pair without computing either.

    Separate from :func:`scene_offsets` so the graph can be inspected before it
    is paid for. ``landsat-lst plan`` reads it through
    :func:`landsat_lst.profiling.graph_stats` in seconds, on a laptop, against
    a stack with no data behind it. See issue #76.

    The estimator is a per-pixel monthly-median climatology, an anomaly
    against it, and a spatial median per scene -- unchanged since ADR-007.
    The *graph* is not the textbook ``groupby("time.month")`` spelling:
    xarray's groupby routes both the median and the broadcast through a
    shuffle whose materialized size grows superlinearly in scene count.
    5.4 GB at 1,500 scenes became >48.5 GB at 2,930 and OOMed every
    full-window run (measured 2026-08-14). Twelve explicit per-month
    reductions plus a month-selection broadcast hold the same values in a
    13.8 GB graph at 2,930 scenes, and the fused graph carries no shuffle
    layers at all. Values are bit-identical: pinned against
    :func:`_offset_graph_groupby` in the unit tests and measured at
    max |delta| = 0.0 over 300 real scenes. See
    results/batch1-investigation/report.md.

    Args:
        lst: Celsius LST on ``(time, latitude, longitude)``, chunked.

    Returns:
        ``(offset, n_valid)`` as unevaluated DataArrays indexed on time.
    """
    spatial = _spatial_dims(lst)
    months = lst.time.dt.month.values

    # Per-pixel, per-calendar-month median, pooling every year in the window.
    # Median rather than mean so residual cloud cannot drag the reference.
    # One explicit reduction per month; the median needs each month's whole
    # time axis in one chunk, and stating that rechunk directly keeps the
    # groupby shuffle out of the graph. ~244 scenes x 512^2 x 4 B is a 256 MB
    # block, which is the price the median was always going to charge.
    parts = []
    for month in np.unique(months):
        sub = lst.isel(time=np.flatnonzero(months == month))
        if sub.chunks is not None:
            sub = sub.chunk({_TIME_DIM: -1})
        parts.append(sub.median(_TIME_DIM, skipna=True).expand_dims(month=[int(month)]))
    ref_month = xr.concat(parts, dim="month")

    # Broadcast by selection over the 12-deep month dim: one indexing layer,
    # then a plain blockwise subtract.
    ref_full = ref_month.sel(month=lst.time.dt.month)
    if "month" in ref_full.coords:
        ref_full = ref_full.drop_vars("month")
    anomaly = lst - ref_full

    # One scalar per scene. Spatial median again for robustness: a handful of
    # contaminated pixels cannot move it.
    return anomaly.median(dim=spatial, skipna=True), lst.notnull().sum(dim=spatial)


def _chunk_sizes(lst: xr.DataArray, dim: str) -> tuple[int, ...] | None:
    """Source chunk lengths along ``dim``, or None for an eager array."""
    if lst.chunks is None:
        return None
    sizes = dict(zip([str(d) for d in lst.dims], lst.chunks, strict=True))
    return tuple(int(c) for c in sizes[dim])


def _io_block_edge(lst: xr.DataArray, budget_gb: float) -> int:
    """Largest power-of-two block edge whose stack fits the resident budget.

    Phase A holds one block's whole time series, so the block edge is what
    bounds memory. Solving ``edge^2 * scenes * 4 <= budget`` and rounding down
    to a power of two keeps unit memory flat as the window grows: a five-year
    tile gets a smaller block than a one-year tile and costs the same RAM.

    Never smaller than the source's spatial chunk, whatever the budget says. A
    block below the chunk edge would make neighbouring blocks re-read the same
    chunk, turning a memory saving into extra reads over the network.
    """
    scenes = int(lst.sizes[_TIME_DIM])
    spatial = _spatial_dims(lst)
    largest = min(int(lst.sizes[d]) for d in spatial)
    edge = 64
    while edge * 2 <= largest and (edge * 2) ** 2 * scenes * 4 <= budget_gb * 1024**3:
        edge *= 2

    floor = 1
    for dim in spatial:
        chunks = _chunk_sizes(lst, dim)
        if chunks:
            floor = max(floor, *chunks)
    return max(edge, min(floor, largest))


def _scene_batches(lst: xr.DataArray, batch: int) -> list[tuple[int, int]]:
    """Half-open scene ranges to read, aligned to the source's time chunks.

    A batch that straddles a chunk boundary makes that chunk materialize twice,
    once for each batch touching it. With the shipped ``TIME_CHUNK = 10`` and a
    batch of 8 the boundaries never line up, and the offset pass pays roughly a
    quarter of an extra read of the whole stack for nothing. Grouping whole
    chunks instead keeps the promise phase B is supposed to make: one traversal.
    """
    n_scenes = int(lst.sizes[_TIME_DIM])
    chunks = _chunk_sizes(lst, _TIME_DIM)
    if not chunks:
        return [(s, min(s + batch, n_scenes)) for s in range(0, n_scenes, batch)]

    out: list[tuple[int, int]] = []
    start = pos = 0
    for length in chunks:
        pos += length
        if pos - start >= batch:
            out.append((start, pos))
            start = pos
    if start < n_scenes:
        out.append((start, n_scenes))
    return out


def _panel_median(panel: np.ndarray) -> np.ndarray:
    """Per-pixel median over time for one panel, NaNs skipped.

    ``kernels.sort_median_axis0`` rather than ``np.nanmedian``: below 600
    scenes per month numpy takes its masked-array path, measured at 2.43 s
    per 256-square panel, and it holds the GIL. The sort kernel is pure C,
    releases the GIL, and ``tests/unit/test_kernels.py`` pins bit-exactness
    against ``np.nanmedian`` on both numpy paths -- the same standard E1
    established for the unit form, re-earned by the replacement kernel.
    """
    return sort_median_axis0(panel)


def climatology_by_blocks(
    lst: xr.DataArray,
    *,
    block: int | None = None,
    panel: int | None = None,
    land_mask: xr.DataArray | None = None,
    spans: list[tuple[int, int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase A: the 12-month per-pixel climatology, blocks run concurrently.

    Sharded over space, which is the axis the per-pixel median is parallel in.
    Each block is read with one small dask graph and reduced in memory, so the
    scheduler never sees the whole stack and there is no rechunk gathering
    space. The I/O block and the compute panel are separate on purpose: reads
    want to be few and large, the kernel wants a working set that stays in
    cache, and E3 measured 256 running 1.8x faster than any larger panel while
    producing an identical checksum.

    Blocks write disjoint slices of ``ref`` and the loop body carries no
    state, so they run in a thread pool. Reads release the GIL in GDAL and
    the median kernel releases it in ``np.sort``, so threads overlap both
    I/O and compute; in-flight memory is bounded at one block per worker
    (see :func:`_unit_workers`). The serial form ran the same 324 blocks one
    at a time on ~1 core of 8 -- that loop, not the estimator, was the
    offset pass's wall clock.

    Args:
        lst: Land-masked, QA-masked stack in Celsius.
        block: I/O block edge. Defaults to the largest power of two fitting
            ``settings.destripe_unit_memory_gb``.
        panel: Compute panel edge. Defaults to ``settings.destripe_compute_panel``.
        land_mask: Optional boolean mask on ``lst``'s spatial grid. A block
            with no land pixel is filled with NaN and never read: its every
            panel median would be a median of an all-NaN stack, which is NaN,
            so the skip is value-identical and the tests hold it to that.
        spans: Compute only these blocks, which must come from
            :func:`landsat_lst.shards.block_spans` at this ``block`` so their
            indices mean the same thing here as in the plan. One shard of the
            phase-A stage passes its own slice of that list. ``ref`` is still
            allocated at full height and width: ``np.empty`` touches no page it
            does not write, so an unrequested block costs address space rather
            than resident memory, and the returned array can be sliced by span
            index without an offset table. **Blocks outside ``spans`` are left
            uninitialized** -- read only the spans you asked for.

    Returns:
        ``(ref, months)`` where ``ref`` is ``(n_months, y, x)`` float32 and
        ``months`` holds the calendar month of each ``ref`` plane.
    """
    block = block or _io_block_edge(lst, settings.destripe_unit_memory_gb)
    panel = panel or settings.destripe_compute_panel
    y_dim, x_dim = _spatial_dims(lst)
    height, width = int(lst.sizes[y_dim]), int(lst.sizes[x_dim])

    # The input dtype is carried through rather than pinned. Production
    # loads float32 and the graph form reduces in float32; forcing a cast here
    # would make the two forms differ in the last bits on any other input,
    # which is exactly the equivalence the unit tests exist to hold.
    dtype = np.dtype(lst.dtype)
    scene_months = lst[_TIME_DIM].dt.month.values.astype(np.int16)
    uniq = np.unique(scene_months)
    month_sel = [scene_months == month for month in uniq]
    ref = np.empty((uniq.size, height, width), dtype=dtype)

    mask = None
    if land_mask is not None:
        mask = np.asarray(land_mask.values, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(
                f"land_mask shape {mask.shape} does not match the stack's "
                f"spatial grid {(height, width)}; the mask must be on the "
                "same grid offsets are estimated on"
            )

    spans = list(spans) if spans is not None else block_spans((height, width), block)
    total = len(spans)
    n_scenes = int(lst.sizes[_TIME_DIM])

    def _one_block(span: tuple[int, int, int, int]) -> None:
        y0, y1, x0, x1 = span
        if mask is not None and not mask[y0:y1, x0:x1].any():
            ref[:, y0:y1, x0:x1] = np.nan
            return
        # One bounded graph per block. Everything after this is numpy.
        data = _read_values(lst.isel({y_dim: slice(y0, y1), x_dim: slice(x0, x1)}), dtype)
        for py in range(0, y1 - y0, panel):
            py1 = min(py + panel, y1 - y0)
            for px in range(0, x1 - x0, panel):
                px1 = min(px + panel, x1 - x0)
                tile = data[:, py:py1, px:px1]
                for i, sel in enumerate(month_sel):
                    ref[i, y0 + py : y0 + py1, x0 + px : x0 + px1] = _panel_median(tile[sel])
        del data

    workers = _unit_workers(block * block * n_scenes * dtype.itemsize)
    done = 0
    progress = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one_block, span) for span in spans]
        for fut in as_completed(futures):
            fut.result()
            with progress:
                done += 1
                report_phase("destripe_climatology", blocks_done=done, blocks_total=total)
    return ref, uniq


def offsets_by_scene(
    lst: xr.DataArray,
    ref: np.ndarray,
    months: np.ndarray,
    *,
    batch: int | None = None,
    batches: list[tuple[int, int]] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Phase B: one scalar offset and one valid count per scene, independently.

    Sharded over scene, which is the axis the spatial median is parallel in.
    A scene's offset depends on nothing but that scene and the climatology, so
    the loop carries no state and a failure costs one scene rather than a tile.
    Scenes are read in batches only to give the reader something to overlap;
    E2 measured the compute at 12.4 ms per scene, constant from 50 to 300, so
    the batch size is an I/O lever and not a compute one.

    Args:
        lst: The same stack phase A reduced.
        ref: ``(n_months, y, x)`` climatology from :func:`climatology_by_blocks`.
        months: Calendar month of each ``ref`` plane.
        batch: Scenes per read. Defaults to ``settings.destripe_scene_batch``.
        batches: Compute only these half-open scene ranges, which must come
            from :func:`_scene_batches` so they still group whole source time
            chunks -- a range cut anywhere else makes its boundary chunk
            materialize twice. One shard of the phase-B stage passes its own
            slice of that list, and the returned arrays then span only the
            scenes it computed, so the partial it publishes carries the time
            coordinate the merge joins on. Defaults to every batch, which
            reproduces the whole-tile result exactly.

    Returns:
        ``(offset, n_valid)`` on the time axis, matching :func:`offset_graph`.
    """
    batch = batch or settings.destripe_scene_batch
    scene_months = lst[_TIME_DIM].dt.month.values.astype(np.int16)
    plane = {int(m): i for i, m in enumerate(months)}
    n_scenes = int(lst.sizes[_TIME_DIM])
    y_dim, x_dim = _spatial_dims(lst)
    footprint = int(lst.sizes[y_dim]) * int(lst.sizes[x_dim])

    dtype = np.dtype(ref.dtype)
    offset = np.empty(n_scenes, dtype=dtype)
    n_valid = np.empty(n_scenes, dtype=np.int64)

    batches = list(batches) if batches is not None else _scene_batches(lst, batch)

    def _one_batch(span: tuple[int, int]) -> int:
        start, stop = span
        chunk = _read_values(lst.isel({_TIME_DIM: slice(start, stop)}), dtype)
        for j in range(stop - start):
            scene = chunk[j]
            anomaly = scene - ref[plane[int(scene_months[start + j])]]
            offset[start + j] = np.nanmedian(anomaly)
            n_valid[start + j] = int(np.isfinite(scene).sum())
        del chunk
        return stop - start

    # Batches write disjoint slices of ``offset``/``n_valid`` and share only
    # the read-only climatology, so they run in the same bounded pool shape
    # as phase A. The unit here spans the full footprint, so on a
    # native-resolution stack _unit_workers shrinks the count to keep
    # in-flight memory inside the phase-A envelope.
    max_batch = max((stop - start) for start, stop in batches)
    requested = sum(stop - start for start, stop in batches)
    workers = _unit_workers(footprint * max_batch * dtype.itemsize)
    scenes_done = 0
    progress = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one_batch, span) for span in batches]
        for fut in as_completed(futures):
            n_done = fut.result()
            with progress:
                scenes_done += n_done
                report_phase(
                    "destripe_offsets",
                    scenes_done=scenes_done,
                    scenes_total=requested,
                )

    # Return only what was computed. With the default batches this is every
    # scene in order, so ``taken`` is ``arange(n_scenes)`` and the arrays are
    # the whole-tile ones untouched; a shard gets a stack it can publish
    # without a separate index telling a reader which entries are real.
    taken = np.concatenate([np.arange(start, stop) for start, stop in batches])
    time_coord = lst[_TIME_DIM].isel({_TIME_DIM: taken})
    coord = {_TIME_DIM: time_coord}
    return (
        xr.DataArray(offset[taken], dims=[_TIME_DIM], coords=coord),
        xr.DataArray(n_valid[taken], dims=[_TIME_DIM], coords=coord),
    )


def offsets_as_units(
    lst: xr.DataArray, *, land_mask: xr.DataArray | None = None
) -> tuple[xr.DataArray, xr.DataArray]:
    """The two phases, run back to back. Bit-exact against :func:`offset_graph`.

    Equivalence is not assumed. E1 ran both forms over a real 300-scene fixture
    and compared: ``max |delta| = 0``, identical NaN patterns, identical valid
    counts, and the same result at two block sizes and from two median kernels.
    ``tests/unit/test_destripe_normalization.py`` pins it on synthetic stacks so
    a regression fails in CI rather than in a five-hour tile.
    """
    with timed_section("destripe_climatology"):
        ref, months = climatology_by_blocks(lst, land_mask=land_mask)
    try:
        with timed_section("destripe_offsets"):
            return offsets_by_scene(lst, ref, months)
    finally:
        del ref


def scene_offsets(
    lst: xr.DataArray,
    *,
    cache: OffsetCache | None = None,
    land_mask: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Estimate one scalar offset per scene, plus each scene's valid-pixel count.

    Split out from ``seasonal_debias`` so the cap can be calibrated: computing
    the offsets is the expensive part, and a sweep over candidate caps should
    pay it once. With ``cache`` supplied it pays it once across *processes* too,
    which is what makes that sentence true outside a single sweep script.

    Args:
        lst: The stack to estimate against, already masked to land.
        cache: Optional :class:`~landsat_lst.offsets.OffsetCache`. A hit returns
            immediately; a miss computes and then writes. The cache never
            raises, so passing one cannot fail a tile that would have succeeded.
        land_mask: Optional boolean mask on ``lst``'s grid, forwarded to the
            unit form so phase A can skip blocks with no land pixel. Purely a
            work-skip: it never changes a value, so it does not enter the
            cache digest. The graph form ignores it.

    Returns:
        ``(offset, n_valid)``, both indexed on time and eagerly evaluated.
    """
    if cache is not None and (hit := cache.read(lst.time)) is not None:
        return hit

    started = time.monotonic()
    # Both reductions read the same stack, so they are computed together: two
    # `.compute()` calls would walk it twice, and on a 5-year tile that second
    # walk is a full re-read of every scene for a reduction that costs almost
    # nothing on its own. dask.compute shares the loaded chunks between the two
    # graphs, which is the same trick scripts/validate_offset_subsampling.py
    # already uses to sweep factors in one pass.
    #
    # GraphProgress publishes the task fraction to the heartbeat; profile_compute
    # records which task prefixes the hours went to, and is off by default.
    if settings.destripe_bounded_units:
        # No GraphProgress: there is no single graph to report a task fraction
        # against. Each phase publishes its own unit counts instead, which is
        # the more useful number anyway -- "block 44 of 81" localizes a stall
        # where a task fraction over a fused graph never could.
        offset, n_valid = offsets_as_units(lst, land_mask=land_mask)
    else:
        # Both reductions read the same stack, so they are computed together:
        # two `.compute()` calls would walk it twice, and on a 5-year tile that
        # second walk is a full re-read of every scene for a reduction that
        # costs almost nothing on its own.
        with GraphProgress(), profile_compute(PROFILE_DESTRIPE_OFFSETS):
            offset, n_valid = dask.compute(*offset_graph(lst))

    if cache is not None:
        cache.write(offset, n_valid, duration_s=time.monotonic() - started)
    return offset, n_valid


def scene_keep_mask(
    offset: xr.DataArray,
    n_valid: xr.DataArray,
    *,
    max_offset_c: float,
    floor: int,
) -> xr.DataArray:
    """Which scenes survive rejection, given their offsets and sample counts.

    Split out so a caller that only wants the *decision* -- ``landsat-lst
    offsets`` reporting a rejection fraction, a sweep over candidate caps --
    applies the identical rule rather than a re-implementation of it that drifts.

    Args:
        offset: One scalar per scene.
        n_valid: Valid-pixel count per scene, on the grid the offset was
            estimated on.
        max_offset_c: Discard a scene whose absolute offset exceeds this.
        floor: Minimum valid pixels, in that same grid's pixels.

    Returns:
        Boolean mask on the scene axis.
    """
    return offset.notnull() & (n_valid >= floor) & (np.abs(offset) <= max_offset_c)


def rejection_floor(*, offset_source_given: bool) -> int:
    """The sparse floor that applies, in the grid the offset actually rests on.

    A coarse count cannot be scaled up to stand in for a native one: GDAL's
    average ignores nodata, so a block holding one valid fine pixel still yields
    a valid coarse pixel. Measured at Pergamino, a scene with exactly 1 valid
    native pixel reported 13 at factor 8; scaling that by 64 would claim 816 and
    wave through a scene the native path rejects outright. So the two floors
    replace each other rather than converting into each other. See
    docs/findings-offset-subsampling.md.
    """
    if offset_source_given:
        return settings.destripe_min_offset_samples
    return settings.destripe_min_scene_pixels


def debias_with_offsets(
    lst: xr.DataArray,
    offset: xr.DataArray,
    n_valid: xr.DataArray,
    *,
    max_offset_c: float,
    min_scene_pixels: int,
    min_offset_samples: int = 0,
    offset_source_given: bool,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Apply rejection and subtraction to an estimate that already exists.

    The tail of :func:`seasonal_debias`, split out because the estimate and its
    application no longer have to happen in the same process. A sharded tile
    estimates the offsets once, over the whole tile, and then hands the *same*
    600 floats to every row band -- which is what makes the bands' composites
    concatenate into the tile's composite rather than into a striped
    approximation of it.

    **Offsets are joined to ``lst`` by time coordinate value, never by
    position.** A shard's stack is a spatial subset, and a spatial subset can
    lose time steps: ``odc-stac`` drops a step whose scenes miss the band's
    rows entirely, and ``fail_on_error=False`` can leave one band a scene the
    others have. Aligning by index would then apply scene *k*'s offset to
    scene *k+1* from the first missing step onward -- a silent, plausible,
    entirely wrong correction. Where the axes do match, which is every
    whole-tile call, the join is a no-op and the arithmetic is bit-for-bit what
    the fused function did.

    Args:
        lst: Celsius LST to correct. Its time coordinate must be a subset of
            ``offset``'s.
        offset: One scalar per scene, on the estimation stack's time axis.
        n_valid: Valid-pixel count per scene, on that same axis and grid.
        max_offset_c: Discard a scene whose absolute offset exceeds this.
        min_scene_pixels: Sparse floor for a native-resolution estimate.
        min_offset_samples: Sparse floor for a coarse estimate.
        offset_source_given: Whether the estimate came off a coarser grid,
            which selects between the two floors. They replace rather than
            convert into each other; see :func:`rejection_floor`.

    Returns:
        ``(debiased, offset, keep)``, as :func:`seasonal_debias` returns them.
        ``offset`` and ``keep`` stay on the estimate's own time axis so a
        caller can report what was rejected across the whole tile.

    Raises:
        ValueError: If ``lst`` carries a time step the estimate does not, or if
            no scene survives rejection.
    """
    floor = min_offset_samples if offset_source_given else min_scene_pixels
    keep = scene_keep_mask(offset, n_valid, max_offset_c=max_offset_c, floor=floor)

    try:
        keep_here = keep.sel({_TIME_DIM: lst[_TIME_DIM]})
        offset_here = offset.sel({_TIME_DIM: lst[_TIME_DIM]})
    except KeyError as e:
        msg = (
            "lst carries a time step the offsets do not: the estimate must "
            "cover every scene the stack holds. Both must come from the same "
            f"items and groupby ({e})."
        )
        raise ValueError(msg) from e

    kept_idx = np.flatnonzero(np.asarray(keep_here.values))
    if kept_idx.size == 0:
        msg = (
            f"All {keep.sizes['time']} scenes rejected by de-striping "
            f"(max_offset_c={max_offset_c}, min_scene_pixels={min_scene_pixels}, "
            f"min_offset_samples={min_offset_samples}). "
            "Refusing to emit an empty composite."
        )
        raise ValueError(msg)

    debiased = lst.isel(time=kept_idx) - offset_here.isel(time=kept_idx)

    return debiased, offset, keep


def seasonal_debias(
    lst: xr.DataArray,
    *,
    max_offset_c: float,
    min_scene_pixels: int,
    min_offset_samples: int = 0,
    offset_source: xr.DataArray | None = None,
    cache: OffsetCache | None = None,
    land_mask: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """De-bias each scene against a per-pixel monthly climatology.

    Rejected scenes are dropped from the returned stack rather than passed
    through with a zero offset, so nothing downstream sees a scene whose
    correction was not applied.

    **The production pipeline no longer calls this.** Under ADR-017 the
    estimate is made on the source or offset grid and the correction is applied
    to the aggregated delivered stack, which are two different grids and so
    cannot be one call; ``compute_annual_composite`` therefore pairs
    :func:`scene_offsets` with :func:`debias_with_offsets` directly. This
    function remains the single-grid form -- the equivalence oracle the tests
    check the split against, and what a caller with one grid still wants.

    The offset is one scalar per scene, so it does not need a full-resolution
    climatology to estimate. Passing a coarser ``offset_source`` computes it
    from far fewer pixels; the correction still applies at ``lst``'s own
    resolution, since a constant has no resolution.

    Args:
        lst: Celsius LST with dims ``(time, latitude, longitude)`` and a
            datetime ``time`` coordinate. Apply the land mask before calling,
            so offsets are estimated over land only.
        max_offset_c: Discard a scene whose absolute offset exceeds this.
        min_scene_pixels: Sparse floor when estimating at native resolution,
            in ``lst`` pixels. Ignored when ``offset_source`` is given.
        min_offset_samples: Sparse floor when estimating from a coarse
            ``offset_source``, in that grid's pixels. Replaces rather than
            supplements ``min_scene_pixels``, because a coarse valid-pixel
            count cannot be scaled back to a native one (see below).
        offset_source: Optional coarser stack to estimate offsets from. Must
            share ``lst``'s time coordinate. Defaults to ``lst`` itself.
        cache: Optional offset cache. Only the estimate is cached, never the
            rejection: ``max_offset_c`` and both floors are applied to whatever
            the cache returns, so sweeping a cap costs one lookup per candidate
            instead of one 27-minute pass. This is the whole point of caching
            here rather than caching ``debiased``.
        land_mask: Optional boolean mask on the grid offsets are estimated on
            (``offset_source``'s grid when given, else ``lst``'s). Lets the
            unit form skip phase-A blocks with no land pixel. A work-skip
            only; values are identical with or without it.

    Returns:
        ``(debiased, offset, keep)``. ``debiased`` covers only the surviving
        scenes. ``offset`` and ``keep`` are indexed on the *original* time axis
        so callers can report what was rejected and why.

    Raises:
        ValueError: If no scene survives rejection, or if ``offset_source``
            does not align with ``lst`` in time.
    """
    source = lst if offset_source is None else offset_source

    if offset_source is not None and not lst.time.equals(offset_source.time):
        msg = (
            "offset_source does not share lst's time coordinate "
            f"({offset_source.sizes.get('time')} vs {lst.sizes.get('time')} steps). "
            "Both stacks must come from the same items and groupby."
        )
        raise ValueError(msg)

    offset, n_valid = scene_offsets(source, cache=cache, land_mask=land_mask)

    # The sparse guard is stated on whichever grid the offset was estimated on,
    # because that is the grid the median actually rests on. See
    # :func:`rejection_floor` for why the two floors replace rather than convert
    # into each other.
    return debias_with_offsets(
        lst,
        offset,
        n_valid,
        max_offset_c=max_offset_c,
        min_scene_pixels=min_scene_pixels,
        min_offset_samples=min_offset_samples,
        offset_source_given=offset_source is not None,
    )


def offset_diagnostics(offset: xr.DataArray, keep: xr.DataArray) -> dict[str, float]:
    """Summarize the offset distribution and what rejection removed.

    The rejection fraction is the number that matters. Log it per tile: the cap
    was calibrated on mid-latitude cropland, and a tile whose rejection fraction
    departs sharply from the ~22% seen there is telling you something.
    """
    values = np.asarray(offset.values, dtype="float64")
    finite = values[np.isfinite(values)]
    kept = np.asarray(keep.values, dtype=bool)

    if finite.size == 0:
        return {"n_scenes": float(values.size), "n_kept": 0.0, "rejected_frac": 1.0}

    p1, p50, p99 = (float(v) for v in np.percentile(finite, [1, 50, 99]))
    return {
        "n_scenes": float(values.size),
        "n_kept": float(kept.sum()),
        "rejected_frac": round(float(1.0 - kept.mean()), 4),
        "std": round(float(finite.std()), 2),
        "min": round(float(finite.min()), 2),
        "max": round(float(finite.max()), 2),
        "p1": round(p1, 2),
        "p50": round(p50, 2),
        "p99": round(p99, 2),
    }
