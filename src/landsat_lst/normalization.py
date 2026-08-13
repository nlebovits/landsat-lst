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

import dask
import numpy as np
import xarray as xr

from landsat_lst.profiling import PROFILE_DESTRIPE_OFFSETS, profile_compute
from landsat_lst.progress import GraphProgress

_TIME_DIM = "time"


def _spatial_dims(lst: xr.DataArray) -> list[str]:
    """Every dim but time. Avoids hard-coding latitude/longitude naming."""
    return [str(d) for d in lst.dims if d != _TIME_DIM]


def _pixel_count(lst: xr.DataArray) -> int:
    """Pixels in one time slice."""
    return int(np.prod([lst.sizes[d] for d in _spatial_dims(lst)]))


def offset_graph(lst: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Build the lazy ``(offset, n_valid)`` pair without computing either.

    Separate from :func:`scene_offsets` so the graph can be inspected before it
    is paid for. This is the graph that turned out to hold 598,604 tasks on a
    300-scene N40W075 sample, and its size follows from array shape and
    chunking alone. ``landsat-lst plan`` reads it through
    :func:`landsat_lst.profiling.graph_stats` in seconds, on a laptop, against
    a stack with no data behind it. See issue #76.

    Args:
        lst: Celsius LST on ``(time, latitude, longitude)``, chunked.

    Returns:
        ``(offset, n_valid)`` as unevaluated DataArrays indexed on time.
    """
    spatial = _spatial_dims(lst)

    # Per-pixel, per-calendar-month median, pooling every year in the window.
    # Median rather than mean so residual cloud cannot drag the reference.
    ref_month = lst.groupby("time.month").median(skipna=True)

    anomaly = lst.groupby("time.month") - ref_month

    # One scalar per scene. Spatial median again for robustness: a handful of
    # contaminated pixels cannot move it.
    return anomaly.median(dim=spatial, skipna=True), lst.notnull().sum(dim=spatial)


def scene_offsets(lst: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Estimate one scalar offset per scene, plus each scene's valid-pixel count.

    Split out from ``seasonal_debias`` so the cap can be calibrated: computing
    the offsets is the expensive part, and a sweep over candidate caps should
    pay it once.

    Returns:
        ``(offset, n_valid)``, both indexed on time and eagerly evaluated.
    """
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
    return offset, n_valid


def seasonal_debias(
    lst: xr.DataArray,
    *,
    max_offset_c: float,
    min_scene_pixels: int,
    min_offset_samples: int = 0,
    offset_source: xr.DataArray | None = None,
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

    offset, n_valid = scene_offsets(source)

    # The sparse guard is stated on whichever grid the offset was estimated on,
    # because that is the grid the median actually rests on.
    #
    # A coarse count cannot be scaled back up to stand in for the native one.
    # Coarse loading inflates apparent validity: GDAL's average ignores nodata,
    # so a block holding one valid fine pixel still yields a valid coarse pixel,
    # and qa_pixel is nearest-sampled independently of it. Measured at Pergamino,
    # a scene with exactly 1 valid native pixel reported 13 valid pixels at
    # factor 8 -- scaling that by 64 would claim 816 and wave through a scene the
    # native path rejects outright. See docs/findings-offset-subsampling.md.
    floor = min_scene_pixels if offset_source is None else min_offset_samples

    keep = offset.notnull() & (n_valid >= floor) & (np.abs(offset) <= max_offset_c)

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
