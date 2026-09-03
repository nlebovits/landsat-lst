"""Vectorized nan-reduction kernels that replace numpy's slowest paths.

Two numpy reductions dominated tile wall clock, and both take numpy's
worst-scaling implementation on production shapes:

- ``np.nanquantile`` with an axis falls back to ``np.apply_along_axis`` --
  one Python-level call per pixel. Measured on production depth (T=2300):
  four threads deliver 1.06x the throughput of one, because every call
  holds the GIL. Dask ships a vectorized escape hatch
  (``_custom_nanquantile``) but bails back to numpy above 1,000 elements
  on the reduced axis, and the production window is ~2,930 scenes.
- ``np.nanmedian`` below 600 elements takes the masked-array small path
  (``_nanmedian_small``), measured at 2.43 s per 256-square panel of a
  244-scene month.

Both kernels here are a single ``np.sort`` plus ``take_along_axis``: pure
C, GIL-releasing, and bit-compatible with what production already writes.

Bit-compatibility is exact, not approximate, and the tests in
``tests/unit/test_kernels.py`` pin it:

- ``sort_median_axis0`` reproduces ``np.nanmedian(x, axis=0)`` bit for
  bit on float32 and float64, on both the small-T and large-T numpy
  paths.
- ``nanquantile_last`` reproduces ``np.nanquantile(x, np.array([q]),
  axis=-1)[0]`` bit for bit -- the array-q ("non-weak") form, because
  that is the form xarray's ``quantile`` passes down and therefore the
  form whose values the shipped composite contains. The scalar-q form
  rounds differently in float32 (NEP 50 weak promotion) and is NOT the
  production value.

NaN counting uses ``~np.isnan`` rather than ``np.isfinite`` so +/-inf is
treated as data, exactly as numpy's reductions treat it. (The
``convert_to_celsius`` clamp makes inf unreachable in production; the
choice matters only for keeping equivalence unconditional.)
"""

from __future__ import annotations

import numpy as np


def sort_median_axis0(x: np.ndarray) -> np.ndarray:
    """Per-pixel median over axis 0, NaNs skipped. Bit-exact vs np.nanmedian.

    NaNs sort to the end, so the per-pixel valid count locates the median
    without a mask pass. An even count averages the two middle values in
    the input dtype, which is exactly what ``np.nanmedian`` does.

    Args:
        x: ``(time, ...)`` array, float32 or float64.

    Returns:
        Median over axis 0, in ``x``'s dtype, NaN where no valid values.
    """
    if x.shape[0] == 0:
        return np.full(x.shape[1:], np.nan, dtype=x.dtype)
    n = (~np.isnan(x)).sum(axis=0)
    s = np.sort(x, axis=0)
    last = x.shape[0] - 1
    lo = np.clip((n - 1) // 2, 0, last)
    hi = np.clip(n // 2, 0, last)
    a = np.take_along_axis(s, lo[None], 0)[0]
    b = np.take_along_axis(s, hi[None], 0)[0]
    out = 0.5 * (a + b)
    out[n == 0] = np.nan
    return out.astype(x.dtype, copy=False)


def nanquantile_last(x: np.ndarray, q: float) -> np.ndarray:
    """Linear-method nanquantile over the last axis, float64 output.

    Bit-exact against ``np.nanquantile(x, np.array([q]), axis=-1)[0]``,
    which is the computation the shipped composite runs per dask block
    (xarray passes q as a 1-element float64 array). The lerp is carried
    in float64 with numpy's own branch structure: forward interpolation
    below gamma 0.5, backward at and above it. The caller casts to
    float32 exactly where the old path did, so the written product is
    unchanged bit for bit.

    Args:
        x: ``(..., time)`` array, float32 or float64.
        q: Quantile in [0, 1].

    Returns:
        float64 array of quantiles; NaN where a pixel has no valid values.
    """
    if x.shape[-1] == 0:
        return np.full(x.shape[:-1], np.nan, dtype=np.float64)
    n = (~np.isnan(x)).sum(axis=-1)
    s = np.sort(x, axis=-1)
    last = x.shape[-1] - 1
    h = (n - 1) * np.float64(q)  # virtual index, float64 throughout
    lo = np.clip(np.floor(h).astype(np.int64), 0, last)
    hi = np.clip(lo + 1, 0, np.clip(n - 1, 0, last))
    a = np.take_along_axis(s, lo[..., None], -1)[..., 0]
    b = np.take_along_axis(s, hi[..., None], -1)[..., 0]
    t = h - lo
    diff = b - a
    out = np.asarray(a + diff * t, dtype=np.float64)
    # numpy's _lerp switches to the backward form at t >= 0.5 to bound
    # rounding error; reproducing the branch is what makes this bit-exact.
    np.subtract(b, diff * (1 - t), out=out, where=t >= 0.5)
    out[n == 0] = np.nan
    return out


def quantile_last_sentinel(x: np.ndarray, q: float, sentinel: int = 0) -> np.ndarray:
    """Linear-method quantile over the last axis of an integer stack, float64 out.

    The composite's stack is ``uint16`` DN with ``sentinel`` marking no
    observation (issue #136). An integer array has no NaN, so the sentinel
    sorts *first* rather than last, and the gather starts at ``T - n`` where
    the valid run begins. The lerp is the same two-branch float64 arithmetic
    :func:`nanquantile_last` runs, so on the same values the two agree bit
    for bit; ``tests/unit/test_kernels.py`` pins that against
    ``np.nanquantile`` on the float64 image of the stack.

    Args:
        x: ``(..., time)`` integer array; ``sentinel`` must sort below every
            valid value (0 for DN).
        q: Quantile in [0, 1].
        sentinel: The no-observation value.

    Returns:
        float64 array of quantiles in the stack's units; NaN where a pixel has
        no valid values.
    """
    if x.shape[-1] == 0:
        return np.full(x.shape[:-1], np.nan, dtype=np.float64)
    n = (x != sentinel).sum(axis=-1)
    s = np.sort(x, axis=-1)
    last = x.shape[-1] - 1
    first = x.shape[-1] - n
    h = (n - 1) * np.float64(q)
    lo = np.clip(np.floor(h).astype(np.int64), 0, last)
    hi = np.clip(lo + 1, 0, np.clip(n - 1, 0, last))
    # A pixel with n == 0 has first == T; clip keeps the gather in bounds and
    # its result is overwritten with NaN below.
    ia = np.clip(first + lo, 0, last)
    ib = np.clip(first + hi, 0, last)
    a = np.take_along_axis(s, ia[..., None], -1)[..., 0].astype(np.float64)
    b = np.take_along_axis(s, ib[..., None], -1)[..., 0].astype(np.float64)
    t = h - lo
    diff = b - a
    out = np.asarray(a + diff * t, dtype=np.float64)
    np.subtract(b, diff * (1 - t), out=out, where=t >= 0.5)
    out[n == 0] = np.nan
    return out
