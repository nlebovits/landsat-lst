"""Bit-exactness pins for the sort-based nan-reduction kernels.

These kernels replace numpy's slowest paths inside the offset pass and the
composite P95. The replacement is admissible only because it changes no bit
of output; every test here asserts exact equality, never closeness. If a
numpy upgrade breaks one of these, the kernel (or its target) moved -- treat
that as a product change, not a flaky test.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from landsat_lst.kernels import nanquantile_last, sort_median_axis0


def _stack(rng, t, shape=(40, 30), nan_frac=0.35, dtype=np.float32):
    x = rng.normal(20.0, 8.0, (t, *shape)).astype(dtype)
    x[rng.random(x.shape) < nan_frac] = np.nan
    return x


class TestSortMedianAxis0:
    # 601 crosses numpy's small/large nanmedian threshold at 600; the two
    # paths round identically today, but the pin covers both regardless.
    @pytest.mark.parametrize("t", [1, 2, 3, 25, 244, 601])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_bit_exact_against_nanmedian(self, t, dtype):
        rng = np.random.default_rng(42)
        x = _stack(rng, t, dtype=dtype)
        x[:, 0, 0] = np.nan  # an all-NaN pixel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanmedian(x, axis=0)
        assert np.array_equal(sort_median_axis0(x), expected, equal_nan=True)
        assert sort_median_axis0(x).dtype == x.dtype

    def test_empty_time_axis_is_all_nan(self):
        x = np.empty((0, 4, 5), dtype=np.float32)
        out = sort_median_axis0(x)
        assert out.shape == (4, 5)
        assert np.isnan(out).all()

    def test_inf_counts_as_data_like_nanmedian(self):
        # The graph form's notnull() and the old unit form's isfinite()
        # diverge on +/-inf (latent, unreachable past the Celsius clamp).
        # The kernel sides with np.nanmedian: inf is data.
        x = np.array([[1.0], [np.inf], [3.0], [np.nan]], dtype=np.float32)
        x = x.reshape(4, 1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanmedian(x, axis=0)
        assert np.array_equal(sort_median_axis0(x), expected, equal_nan=True)


class TestNanquantileLast:
    # The target is the array-q ("non-weak") form because that is what
    # xarray's quantile passes per dask block; the scalar-q form rounds
    # float32 differently (NEP 50 weak promotion) and is NOT what the
    # shipped composite contains.
    @pytest.mark.parametrize("t", [1, 2, 3, 37, 244, 1200, 2930])
    def test_bit_exact_against_blockwise_nanquantile_f32(self, t):
        rng = np.random.default_rng(7)
        x = np.moveaxis(_stack(rng, t, shape=(25, 20)), 0, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanquantile(x, np.array([0.95]), axis=-1)[0]
        got = nanquantile_last(x, 0.95)
        assert got.dtype == np.float64
        # apply_along_axis quantizes its output to float32 when the block's
        # first pixel is empty and keeps float64 otherwise; both collapse to
        # the same float32, which is the dtype the product is written in.
        assert np.array_equal(
            got.astype(np.float32),
            expected.astype(np.float32),
            equal_nan=True,
        )

    def test_bit_exact_in_float64_when_numpy_stays_float64(self):
        rng = np.random.default_rng(11)
        x = np.moveaxis(_stack(rng, 57, shape=(25, 20), dtype=np.float64), 0, -1)
        expected = np.nanquantile(x, np.array([0.95]), axis=-1)[0]
        assert np.array_equal(nanquantile_last(x, 0.95), expected, equal_nan=True)

    @pytest.mark.parametrize("q", [0.0, 0.05, 0.5, 0.95, 0.99, 1.0])
    def test_other_quantiles(self, q):
        rng = np.random.default_rng(13)
        x = np.moveaxis(_stack(rng, 61, shape=(30, 10)), 0, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanquantile(x, np.array([q]), axis=-1)[0]
        assert np.array_equal(
            nanquantile_last(x, q).astype(np.float32),
            expected.astype(np.float32),
            equal_nan=True,
        )

    def test_all_nan_pixel_and_empty_axis(self):
        x = np.full((3, 4, 5), np.nan, dtype=np.float32)
        x = np.moveaxis(x, 0, -1)
        assert np.isnan(nanquantile_last(x, 0.95)).all()
        empty = np.empty((4, 5, 0), dtype=np.float32)
        out = nanquantile_last(empty, 0.95)
        assert out.shape == (4, 5)
        assert np.isnan(out).all()

    def test_matches_shipped_xarray_dask_path_above_dask_cutoff(self):
        # dask's _custom_nanquantile bails to numpy above 1,000 elements on
        # the reduced axis, which is the path every production window takes
        # (~2,930 scenes). Pin float32 equality against the exact shipped
        # call: xarray quantile over a dask array with a single time chunk.
        xr = pytest.importorskip("xarray")
        rng = np.random.default_rng(17)
        a = _stack(rng, 1050, shape=(12, 10))
        da = xr.DataArray(a, dims=["time", "y", "x"]).chunk({"y": 6, "time": -1})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            shipped = da.quantile(0.95, dim="time", skipna=True).values
        got = nanquantile_last(np.moveaxis(a, 0, -1), 0.95)
        assert np.array_equal(
            got.astype(np.float32),
            shipped.astype(np.float32),
            equal_nan=True,
        )
