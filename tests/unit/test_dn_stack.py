"""The coarse stack's lossless uint16 DN representation (issue #125).

Only the lossless half of the DN work from #136 lives in this repository: the
carrier and the reconstruction. Two things hold it. The DN stack keeps exactly
the samples the Celsius path kept, and the Celsius view of that stack is bit
identical to the array the estimator has always read. The quantisation-bearing
half of #136 -- the whole-DN offset shift and the integer P95 -- is deliberately
absent, so nothing here may change a published value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.qa import (
    DN_SENTINEL,
    apply_qa_mask,
    celsius_stack,
    convert_to_celsius,
    dn_clamp_bounds,
    dn_stack,
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

    def test_land_mask_keeps_the_carrier_two_bytes_wide(self):
        """``.where(land, DN_SENTINEL)``, never a bare ``.where(land)``.

        A bare ``where`` fills with NaN and promotes the carrier to float,
        which silently gives back the halving the whole stage exists for.
        """
        ds = _dataset(np.random.default_rng(6))
        land = xr.DataArray(
            np.tile([True, False], (12, 5)),
            dims=("latitude", "longitude"),
            coords={"latitude": ds.latitude, "longitude": ds.longitude},
        )
        masked = dn_stack(ds).where(land, DN_SENTINEL)
        assert masked.dtype == np.uint16
        direct = convert_to_celsius(apply_qa_mask(ds)["lwir11"]).where(land)
        np.testing.assert_array_equal(celsius_stack(masked).values, direct.values)

    def test_the_quantising_half_of_136_is_absent(self):
        """This repository carries the carrier, not the published-value change."""
        from landsat_lst import qa

        for name in ("debias_dn", "offset_dn_shift", "dn_to_celsius"):
            assert not hasattr(qa, name), f"{name} changes published values (see #136)"

    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_round_trip_holds_over_random_inputs(self, seed):
        ds = _dataset(np.random.default_rng(seed))
        old = convert_to_celsius(apply_qa_mask(ds)["lwir11"])
        new = celsius_stack(dn_stack(ds))
        np.testing.assert_array_equal(new.values, old.values)
