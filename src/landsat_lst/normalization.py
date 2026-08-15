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

import time
from typing import TYPE_CHECKING

import dask
import numpy as np
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.profiling import PROFILE_DESTRIPE_OFFSETS, profile_compute
from landsat_lst.progress import GraphProgress

if TYPE_CHECKING:
    from landsat_lst.offsets import OffsetCache

_TIME_DIM = "time"


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


def scene_offsets(
    lst: xr.DataArray, *, cache: OffsetCache | None = None
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


def seasonal_debias(
    lst: xr.DataArray,
    *,
    max_offset_c: float,
    min_scene_pixels: int,
    min_offset_samples: int = 0,
    offset_source: xr.DataArray | None = None,
    cache: OffsetCache | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """De-bias each scene against a per-pixel monthly climatology.

    Rejected scenes are dropped from the returned stack rather than passed
    through with a zero offset, so nothing downstream sees a scene whose
    correction was not applied.

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

    offset, n_valid = scene_offsets(source, cache=cache)

    # The sparse guard is stated on whichever grid the offset was estimated on,
    # because that is the grid the median actually rests on. See
    # :func:`rejection_floor` for why the two floors replace rather than convert
    # into each other.
    floor = min_scene_pixels if offset_source is None else min_offset_samples

    keep = scene_keep_mask(offset, n_valid, max_offset_c=max_offset_c, floor=floor)

    kept_idx = np.flatnonzero(np.asarray(keep.values))
    if kept_idx.size == 0:
        msg = (
            f"All {keep.sizes['time']} scenes rejected by de-striping "
            f"(max_offset_c={max_offset_c}, min_scene_pixels={min_scene_pixels}, "
            f"min_offset_samples={min_offset_samples}). "
            "Refusing to emit an empty composite."
        )
        raise ValueError(msg)

    debiased = lst.isel(time=kept_idx) - offset.isel(time=kept_idx)

    return debiased, offset, keep


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
