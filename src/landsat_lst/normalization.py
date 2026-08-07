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

import numpy as np
import xarray as xr

_TIME_DIM = "time"


def _spatial_dims(lst: xr.DataArray) -> list[str]:
    """Every dim but time. Avoids hard-coding latitude/longitude naming."""
    return [str(d) for d in lst.dims if d != _TIME_DIM]


def scene_offsets(lst: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Estimate one scalar offset per scene, plus each scene's valid-pixel count.

    Split out from ``seasonal_debias`` so the cap can be calibrated: computing
    the offsets is the expensive part, and a sweep over candidate caps should
    pay it once.

    Returns:
        ``(offset, n_valid)``, both indexed on time and eagerly evaluated.
    """
    spatial = _spatial_dims(lst)

    # Per-pixel, per-calendar-month median, pooling every year in the window.
    # Median rather than mean so residual cloud cannot drag the reference.
    ref_month = lst.groupby("time.month").median(skipna=True)

    anomaly = lst.groupby("time.month") - ref_month

    # One scalar per scene. Spatial median again for robustness: a handful of
    # contaminated pixels cannot move it.
    offset = anomaly.median(dim=spatial, skipna=True).compute()
    n_valid = lst.notnull().sum(dim=spatial).compute()
    return offset, n_valid


def seasonal_debias(
    lst: xr.DataArray,
    *,
    max_offset_c: float,
    min_scene_pixels: int,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """De-bias each scene against a per-pixel monthly climatology.

    Rejected scenes are dropped from the returned stack rather than passed
    through with a zero offset, so nothing downstream sees a scene whose
    correction was not applied.

    Note this forces an eager reduction: the offsets must be materialized
    before the time axis can be subset, which costs one full traversal of
    ``lst`` on top of the later percentile pass.

    Args:
        lst: Celsius LST with dims ``(time, latitude, longitude)`` and a
            datetime ``time`` coordinate. Apply the land mask before calling,
            so offsets are estimated over land only.
        max_offset_c: Discard a scene whose absolute offset exceeds this.
        min_scene_pixels: Discard a scene with fewer valid pixels than this;
            its offset is too sparsely estimated to trust.

    Returns:
        ``(debiased, offset, keep)``. ``debiased`` covers only the surviving
        scenes. ``offset`` and ``keep`` are indexed on the *original* time axis
        so callers can report what was rejected and why.

    Raises:
        ValueError: If no scene survives rejection.
    """
    offset, n_valid = scene_offsets(lst)

    keep = offset.notnull() & (n_valid >= min_scene_pixels) & (np.abs(offset) <= max_offset_c)

    kept_idx = np.flatnonzero(np.asarray(keep.values))
    if kept_idx.size == 0:
        msg = (
            f"All {keep.sizes['time']} scenes rejected by de-striping "
            f"(max_offset_c={max_offset_c}, min_scene_pixels={min_scene_pixels}). "
            "Refusing to emit an empty composite."
        )
        raise ValueError(msg)

    debiased = lst.isel(time=kept_idx) - offset.isel(time=kept_idx)

    return debiased, offset, keep


def offset_diagnostics(offset: xr.DataArray, keep: xr.DataArray) -> dict[str, float]:
    """Summarize the offset distribution and what rejection removed.

    The rejection fraction is the number that matters: the cap is provisional
    until a real tile shows how much of the stack it actually discards.
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
