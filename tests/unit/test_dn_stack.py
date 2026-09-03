"""The composite's uint16 DN representation (issue #136).

Three things hold the representation: the DN stack keeps exactly the samples
the Celsius path kept, the integer P95 kernel reproduces ``np.nanquantile`` on
the float64 image of the stack bit for bit, and the whole composite departs
from the float32 Celsius path by at most one encoded DN with ``qa_count``
exactly equal. Every difference is then attributable to one thing: the offset
rounded to a whole DN.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.encoding import encode_lst_uint16
from landsat_lst.kernels import nanquantile_last, quantile_last_sentinel
from landsat_lst.normalization import debias_with_offsets
from landsat_lst.pipeline import _composite_graph, compute_annual_composite
from landsat_lst.qa import (
    DN_SCALE_K,
    DN_SENTINEL,
    apply_qa_mask,
    celsius_stack,
    convert_to_celsius,
    debias_dn,
    dn_clamp_bounds,
    dn_stack,
    dn_to_celsius,
)

CLOUD = 1 << 3


def _dataset(rng, t=40, shape=(12, 10), *, integer=True):
    times = pd.date_range("2021-01-03T13:56:04.018086", periods=t, freq="37D")
    dn = rng.integers(20_000, 62_000, size=(t, *shape), dtype=np.uint16)
    dn[rng.random(dn.shape) < 0.05] = 0  # source fill
    qa = np.zeros((t, *shape), dtype=np.uint16)
    qa[rng.random(qa.shape) < 0.3] = CLOUD
    lwir = dn if integer else dn.astype(np.float32)
    return xr.Dataset(
        {
            "lwir11": (("time", "latitude", "longitude"), lwir),
            "qa_pixel": (("time", "latitude", "longitude"), qa),
        },
        coords={
            "time": times,
            "latitude": np.arange(shape[0], dtype=float),
            "longitude": np.arange(shape[1], dtype=float),
        },
    )


class TestDnStack:
    def test_is_uint16_with_zero_sentinel(self):
        ds = _dataset(np.random.default_rng(1))
        stack = dn_stack(ds)
        assert stack.dtype == np.uint16
        assert (stack.values[ds["qa_pixel"].values == CLOUD] == DN_SENTINEL).all()

    def test_keeps_exactly_the_samples_the_celsius_path_keeps(self):
        ds = _dataset(np.random.default_rng(2))
        celsius = convert_to_celsius(apply_qa_mask(ds)["lwir11"])
        stack = dn_stack(ds)
        np.testing.assert_array_equal(stack.values != DN_SENTINEL, np.isfinite(celsius.values))
        kept = stack.values != DN_SENTINEL
        np.testing.assert_array_equal(stack.values[kept], ds["lwir11"].values[kept])

    def test_celsius_view_is_bit_identical_to_the_old_path(self):
        ds = _dataset(np.random.default_rng(3))
        old = convert_to_celsius(apply_qa_mask(ds)["lwir11"])
        new = celsius_stack(dn_stack(ds))
        assert new.dtype == np.float32
        np.testing.assert_array_equal(new.values, old.values)

    def test_clamp_bounds_are_the_float_clamp_read_back_as_dn(self):
        lo, hi = dn_clamp_bounds()
        below = convert_to_celsius(xr.DataArray(np.array([lo - 1, lo], dtype=np.float32)))
        above = convert_to_celsius(xr.DataArray(np.array([hi, hi + 1], dtype=np.float32)))
        assert np.isnan(below.values[0]) and np.isfinite(below.values[1])
        assert np.isfinite(above.values[0]) and np.isnan(above.values[1])

    def test_clamp_bounds_follow_the_setting(self, monkeypatch):
        lo0, hi0 = dn_clamp_bounds()
        monkeypatch.setattr(settings, "lst_valid_max", 40.0)
        lo1, hi1 = dn_clamp_bounds()
        assert lo1 == lo0
        assert hi1 < hi0
        assert dn_to_celsius(np.array([hi1], dtype=np.float64))[0] <= 40.0

    def test_a_float_fixture_is_rounded_to_dn(self):
        ds = _dataset(np.random.default_rng(4), integer=False)
        ds["lwir11"] = ds["lwir11"] + np.float32(0.4)
        stack = dn_stack(ds)
        assert stack.dtype == np.uint16
        kept = stack.values != DN_SENTINEL
        np.testing.assert_array_equal(
            stack.values[kept], np.rint(ds["lwir11"].values[kept]).astype(np.uint16)
        )

    def test_dask_input_stays_lazy_and_uint16(self):
        ds = _dataset(np.random.default_rng(5)).chunk({"time": 10, "latitude": 6})
        stack = dn_stack(ds)
        assert stack.chunks is not None
        assert stack.dtype == np.uint16


class TestDebiasDn:
    def test_shifts_by_the_rounded_dn_and_keeps_the_sentinel(self):
        stack = xr.DataArray(
            np.array([[[30_000, 0], [40_000, 50_000]]], dtype=np.uint16).repeat(3, axis=0),
            dims=("time", "y", "x"),
        )
        offset = xr.DataArray(np.array([1.0, -0.5, 0.0017], dtype=np.float32), dims=("time",))
        out = debias_dn(stack, offset)
        assert out.dtype == np.uint16
        shift = np.rint(offset.values.astype(np.float64) / DN_SCALE_K).astype(int)
        assert shift.tolist() == [293, -146, 0]
        for i, s in enumerate(shift):
            np.testing.assert_array_equal(out.values[i, 0, 1], 0)
            np.testing.assert_array_equal(out.values[i, 1], np.array([40_000, 50_000]) - s)

    def test_a_shift_out_of_the_uint16_range_becomes_the_sentinel(self):
        stack = xr.DataArray(np.array([[[100, 65_500]]], dtype=np.uint16), dims=("time", "y", "x"))
        big = xr.DataArray(np.array([1.0], dtype=np.float32), dims=("time",))  # +293 DN
        small = xr.DataArray(np.array([-1.0], dtype=np.float32), dims=("time",))
        assert debias_dn(stack, big).values.tolist() == [[[0, 65_207]]]
        assert debias_dn(stack, small).values.tolist() == [[[393, 0]]]

    def test_debias_with_offsets_takes_the_integer_branch(self):
        times = pd.date_range("2021-01-01", periods=4, freq="16D")
        stack = xr.DataArray(
            np.full((4, 2, 2), 40_000, dtype=np.uint16),
            dims=("time", "y", "x"),
            coords={"time": times},
        )
        offset = xr.DataArray(
            np.array([0.5, 99.0, -0.5, 0.0], dtype=np.float32),
            dims=("time",),
            coords={"time": times},
        )
        n_valid = xr.DataArray(np.full(4, 10_000), dims=("time",), coords={"time": times})
        debiased, _, keep = debias_with_offsets(
            stack,
            offset,
            n_valid,
            max_offset_c=15.0,
            min_scene_pixels=1,
            offset_source_given=True,
        )
        assert debiased.dtype == np.uint16
        assert keep.values.tolist() == [True, False, True, True]
        assert debiased.values[:, 0, 0].tolist() == [40_000 - 146, 40_000 + 146, 40_000]


class TestQuantileLastSentinel:
    @pytest.mark.parametrize("t", [1, 2, 3, 37, 244, 1200])
    @pytest.mark.parametrize("q", [0.0, 0.5, 0.95, 1.0])
    def test_bit_exact_against_nanquantile_on_the_float64_image(self, t, q):
        rng = np.random.default_rng(t)
        x = rng.integers(1, 65_536, size=(25, 20, t), dtype=np.uint16)
        x[rng.random(x.shape) < 0.35] = 0
        x[0, 0, :] = 0  # an empty pixel
        image = np.where(x == 0, np.nan, x.astype(np.float64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanquantile(image, np.array([q]), axis=-1)[0]
        got = quantile_last_sentinel(x, q)
        assert got.dtype == np.float64
        assert np.array_equal(got, expected, equal_nan=True)

    def test_agrees_with_nanquantile_last_on_the_same_values(self):
        rng = np.random.default_rng(9)
        x = rng.integers(1, 65_536, size=(10, 10, 300), dtype=np.uint16)
        x[rng.random(x.shape) < 0.5] = 0
        image = np.where(x == 0, np.nan, x.astype(np.float64))
        assert np.array_equal(
            quantile_last_sentinel(x, 0.95), nanquantile_last(image, 0.95), equal_nan=True
        )

    def test_empty_axis(self):
        out = quantile_last_sentinel(np.empty((4, 5, 0), dtype=np.uint16), 0.95)
        assert out.shape == (4, 5)
        assert np.isnan(out).all()


class TestCompositeOnDn:
    """The whole composite against the float32 Celsius path it replaced."""

    def _both(self, rng, *, chunk=False):
        ds = _dataset(rng, t=120, shape=(24, 20))
        if chunk:
            ds = ds.chunk({"time": 10, "latitude": 12, "longitude": 10})
        times = ds.time
        offset = xr.DataArray(
            rng.normal(0.0, 5.7, ds.sizes["time"]).astype(np.float32),
            dims=("time",),
            coords={"time": times},
        )
        n_valid = xr.DataArray(
            np.full(ds.sizes["time"], 10_000), dims=("time",), coords={"time": times}
        )
        keep = np.abs(offset.values) <= settings.destripe_max_offset_c

        new = compute_annual_composite(ds, offsets=(offset, n_valid)).compute()

        # The float32 Celsius path, spelled out as it was shipped.
        celsius = convert_to_celsius(apply_qa_mask(ds)["lwir11"])
        celsius = celsius.isel(time=np.flatnonzero(keep)) - offset.isel(time=np.flatnonzero(keep))
        old = _composite_graph(celsius).compute()
        return old, new

    @pytest.mark.parametrize("chunk", [False, True])
    def test_at_most_one_encoded_dn_apart_and_counts_equal(self, chunk):
        old, new = self._both(np.random.default_rng(21), chunk=chunk)
        assert new["lst_p95"].dtype == np.float32
        assert new["qa_count"].dtype == np.uint8
        np.testing.assert_array_equal(new["qa_count"].values, old["qa_count"].values)
        a = encode_lst_uint16(old["lst_p95"]).values.astype(np.int64)
        b = encode_lst_uint16(new["lst_p95"]).values.astype(np.int64)
        np.testing.assert_array_equal(a == 0, b == 0)
        assert np.abs(a - b).max() <= 1
        # The bound: half a DN step through a monotone interpolation.
        both = (a != 0) & (b != 0)
        delta = np.abs(new["lst_p95"].values - old["lst_p95"].values)[both]
        assert delta.max() <= DN_SCALE_K / 2 + 1e-5

    def test_zero_offsets_reproduce_the_celsius_path_to_float32_rounding(self):
        rng = np.random.default_rng(22)
        ds = _dataset(rng, t=60, shape=(16, 12))
        zero = xr.DataArray(
            np.zeros(ds.sizes["time"], dtype=np.float32), dims=("time",), coords={"time": ds.time}
        )
        n_valid = xr.DataArray(
            np.full(ds.sizes["time"], 10_000), dims=("time",), coords={"time": ds.time}
        )
        new = compute_annual_composite(ds, offsets=(zero, n_valid))
        old = _composite_graph(convert_to_celsius(apply_qa_mask(ds)["lwir11"]))
        # With no offset the DN path is exact and the old path carries float32
        # rounding of each sample; they agree to float32 resolution.
        np.testing.assert_allclose(new["lst_p95"].values, old["lst_p95"].values, atol=1e-4)
        np.testing.assert_array_equal(new["qa_count"].values, old["qa_count"].values)

    def test_float_input_still_takes_the_float_kernel(self):
        # The benchmarks and the equivalence oracle build a Celsius stack and
        # hand it to _composite_graph directly; that branch is unchanged.
        rng = np.random.default_rng(23)
        celsius = convert_to_celsius(apply_qa_mask(_dataset(rng, t=30))["lwir11"])
        out = _composite_graph(celsius)
        assert out["lst_p95"].dtype == np.float32
        expected = nanquantile_last(np.moveaxis(celsius.values, 0, -1), 0.95)
        valid = np.isfinite(expected)
        np.testing.assert_array_equal(
            out["lst_p95"].values[valid], expected.astype(np.float32)[valid]
        )
