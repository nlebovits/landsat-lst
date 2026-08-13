"""Unit tests for season-aware per-scene normalization (issue #46, ADR-007).

The correction removes each scene's bulk deviation from a per-pixel *monthly*
climatology. Two properties matter and are tested here. It must remove an
injected per-scene bias, and it must leave the seasonal cycle intact -- an
annual reference would eat the season and cool the composite badly (measured:
40.6 C to 29.8 C).

The rejection policy is discard, never clamp. A scene whose offset exceeds the
cap must be absent from the output stack and absent from qa_count, not present
with a bounded correction.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.normalization import offset_diagnostics, scene_offsets, seasonal_debias
from landsat_lst.pipeline import compute_annual_composite

# Comfortably above destripe_min_scene_pixels (500) so grid size never drives
# rejection except where a test means it to.
GRID = 40


def _times(years=(2021, 2022, 2023, 2024, 2025), per_month=2) -> np.ndarray:
    """Production-shaped window: 5 years, 2 observations per calendar month.

    Window length matters to the estimator. Each month's reference is a median
    over ``len(years) * per_month`` samples, and a scene's own bias leaks into
    its own reference, so a short window attenuates the recovered offset.
    """
    days = [5, 20][:per_month]
    stamps = [f"{y}-{m:02d}-{d:02d}" for y in years for m in range(1, 13) for d in days]
    return pd.to_datetime(stamps).values


def _seasonal_stack(
    *,
    bias: np.ndarray | None = None,
    seed: int = 42,
    amplitude: float = 15.0,
) -> xr.DataArray:
    """Synthetic LST with a real seasonal cycle plus optional per-scene bias.

    Mirrors the prototype's self-test: a +-15 C sinusoid, a fixed spatial
    pattern, and small noise, so the seasonal signal is large relative to the
    bias being recovered.
    """
    rng = np.random.default_rng(seed)
    times = _times()
    doy = pd.DatetimeIndex(times).dayofyear.values.astype("float64")

    season = amplitude * np.sin(2 * np.pi * (doy - 15) / 365)
    spatial = rng.normal(0, 3, (GRID, GRID))
    noise = rng.normal(0, 0.3, (len(times), GRID, GRID))

    data = 25.0 + season[:, None, None] + spatial[None, :, :] + noise
    if bias is not None:
        data = data + bias[:, None, None]

    return xr.DataArray(
        data.astype("float64"),
        dims=["time", "latitude", "longitude"],
        coords={
            "time": times,
            "latitude": np.linspace(-33.4, -34.4, GRID),
            "longitude": np.linspace(-61.1, -60.1, GRID),
        },
    )


def _debias(lst, *, max_offset_c=15.0, min_scene_pixels=500):
    return seasonal_debias(lst, max_offset_c=max_offset_c, min_scene_pixels=min_scene_pixels)


class TestSceneOffsets:
    """The offset pass is the expensive half of de-striping on a real tile."""

    def test_reads_the_stack_once(self):
        """Both reductions share one traversal.

        They read the same stack, and the valid-pixel count is trivial next to
        the monthly median. Computing them separately would walk every scene a
        second time to collect it -- on a 5-year tile that is a full re-read of
        ~2,900 scenes for almost nothing.
        """
        import dask

        from landsat_lst import normalization

        calls = []
        real_compute = dask.compute

        def counting_compute(*args, **kwargs):
            calls.append(args)
            return real_compute(*args, **kwargs)

        with patch.object(normalization.dask, "compute", counting_compute):
            offset, n_valid = scene_offsets(_seasonal_stack())

        assert len(calls) == 1, "the stack must be traversed once, not per reduction"
        assert len(calls[0]) == 2, "both reductions belong to the same graph"
        assert offset.sizes["time"] == n_valid.sizes["time"]


class TestSeasonalDebias:
    """Core normalization behavior."""

    def test_recovers_injected_per_scene_bias(self):
        """Estimated offsets track the bias actually injected into each scene."""
        rng = np.random.default_rng(7)
        times = _times()
        bias = rng.normal(0, 2.0, len(times))
        lst = _seasonal_stack(bias=bias)

        _, offset, _ = _debias(lst)

        corr = np.corrcoef(offset.values, bias)[0, 1]
        assert corr > 0.75

    def test_removes_injected_bias(self):
        """Residual scene-to-scene spread collapses after correction."""
        rng = np.random.default_rng(7)
        times = _times()
        bias = rng.normal(0, 2.0, len(times))
        lst = _seasonal_stack(bias=bias)

        debiased, _, _ = _debias(lst)

        # Compare like with like: spread of scene means around the season.
        before = lst.mean(dim=["latitude", "longitude"]).values
        after = debiased.mean(dim=["latitude", "longitude"]).values
        assert np.std(after - np.mean(after)) < np.std(before - np.mean(before))

    def test_preserves_seasonal_amplitude(self):
        """The season survives at >=0.9x.

        This is what separates the monthly reference from the rejected annual
        one, which flattened the cycle and cooled the composite by ~11 C.
        """
        lst = _seasonal_stack()

        debiased, _, _ = _debias(lst)

        before = float(lst.mean(dim=["latitude", "longitude"]).std())
        after = float(debiased.mean(dim=["latitude", "longitude"]).std())
        assert after > 0.9 * before

    def test_offset_is_spatially_uniform(self):
        """The correction shifts a scene's baseline without touching structure.

        Within-scene contrasts must be identical before and after, since the
        product is an indicator of *relative* heat.
        """
        lst = _seasonal_stack()

        debiased, _offset, keep = _debias(lst)

        kept = lst.isel(time=np.flatnonzero(keep.values))
        before = (kept - kept.mean(dim=["latitude", "longitude"])).values
        after = (debiased - debiased.mean(dim=["latitude", "longitude"])).values
        np.testing.assert_allclose(before, after, atol=1e-9)


class TestRejection:
    """Discard policy: scenes are dropped, never clamped."""

    def test_extreme_offset_scene_is_discarded_not_clamped(self):
        """A -73 C scene is absent from the stack, not shifted by -15 C.

        Clamping would leave ~58 C of uncorrected bias in the composite.
        """
        times = _times()
        bias = np.zeros(len(times))
        bias[5] = -73.0
        lst = _seasonal_stack(bias=bias)

        debiased, offset, keep = _debias(lst)

        assert not bool(keep.values[5])
        assert debiased.sizes["time"] == len(times) - 1
        assert times[5] not in debiased.time.values
        # The offset was measured and reported, just not applied.
        assert offset.values[5] < -50

    def test_sparse_scene_is_discarded(self):
        """A scene with too few valid pixels is dropped, not passed through."""
        lst = _seasonal_stack()
        # Blank all but a 10x10 corner: 100 valid pixels, below the 500 floor.
        lst[3, 10:, :] = np.nan
        lst[3, :, 10:] = np.nan

        debiased, _, keep = _debias(lst)

        assert not bool(keep.values[3])
        assert debiased.sizes["time"] == lst.sizes["time"] - 1

    def test_clean_stack_rejects_nothing(self):
        """A fixed cap rejects nothing when the data is clean."""
        lst = _seasonal_stack()

        debiased, _, keep = _debias(lst)

        assert bool(keep.values.all())
        assert debiased.sizes["time"] == lst.sizes["time"]

    def test_all_rejected_raises(self):
        """An empty composite is a failure worth surfacing, not silent nodata."""
        lst = _seasonal_stack()

        with pytest.raises(ValueError, match=r"All .* scenes rejected"):
            _debias(lst, min_scene_pixels=GRID * GRID + 1)

    def test_cap_is_configurable(self):
        """Tightening the cap rejects more scenes."""
        rng = np.random.default_rng(3)
        times = _times()
        bias = rng.normal(0, 6.0, len(times))
        lst = _seasonal_stack(bias=bias)

        _, _, loose = _debias(lst, max_offset_c=15.0)
        _, _, tight = _debias(lst, max_offset_c=2.0)

        assert int(tight.values.sum()) < int(loose.values.sum())


class TestOffsetDiagnostics:
    """The rejection fraction is the number that calibrates the cap."""

    def test_reports_rejection_fraction(self):
        times = _times()
        bias = np.zeros(len(times))
        bias[0] = -73.0
        lst = _seasonal_stack(bias=bias)

        _, offset, keep = _debias(lst)
        stats = offset_diagnostics(offset, keep)

        assert stats["n_scenes"] == len(times)
        assert stats["n_kept"] == len(times) - 1
        assert stats["rejected_frac"] == pytest.approx(1 / len(times), abs=1e-4)
        assert stats["min"] < -50
        assert "p1" in stats and "p50" in stats and "p99" in stats


class TestCompositeIntegration:
    """De-striping wired into compute_annual_composite."""

    @staticmethod
    def _dataset(bias: np.ndarray | None = None) -> xr.Dataset:
        """Clear-sky Landsat-like dataset in raw DN, warm enough to be plausible."""
        times = _times()
        n = len(times)
        lwir = np.full((n, GRID, GRID), 44000.0, dtype=np.float32)
        if bias is not None:
            # DN scale is 0.00341802 K per count.
            lwir = lwir + (bias[:, None, None] / 0.00341802).astype(np.float32)
        return xr.Dataset(
            {
                "lwir11": (["time", "latitude", "longitude"], lwir),
                "qa_pixel": (
                    ["time", "latitude", "longitude"],
                    np.zeros((n, GRID, GRID), dtype=np.uint16),
                ),
            },
            coords={
                "time": times,
                "latitude": np.linspace(-33.4, -34.4, GRID),
                "longitude": np.linspace(-61.1, -60.1, GRID),
            },
        )

    def test_qa_count_excludes_discarded_scenes(self, monkeypatch):
        """Counts describe the evidence actually behind each P95 pixel."""
        monkeypatch.setattr(settings, "destripe", True)
        times = _times()
        bias = np.zeros(len(times))
        # Both January 2021 observations are wrecked.
        bias[0] = -73.0
        bias[1] = -73.0

        result = compute_annual_composite(self._dataset(bias=bias))

        jan = result["qa_count"].sel(month=1).values
        feb = result["qa_count"].sel(month=2).values
        # 5 years x 2 obs = 10 per month; January loses 2 of its 10.
        assert int(jan.max()) == 8
        assert int(feb.max()) == 10

    def test_destripe_disabled_keeps_every_scene(self, monkeypatch):
        """settings.destripe=False reproduces the un-normalized behavior."""
        monkeypatch.setattr(settings, "destripe", False)
        times = _times()
        bias = np.zeros(len(times))
        bias[0] = -73.0
        bias[1] = -73.0

        result = compute_annual_composite(self._dataset(bias=bias))

        assert int(result["qa_count"].sel(month=1).max()) == 10

    def test_land_mask_restricts_composite(self, monkeypatch):
        """A supplied land mask blanks ocean pixels in the composite."""
        monkeypatch.setattr(settings, "destripe", True)
        data = self._dataset()
        mask = xr.DataArray(
            np.r_[np.ones((GRID // 2, GRID)), np.zeros((GRID // 2, GRID))].astype(bool),
            dims=["latitude", "longitude"],
            coords={"latitude": data.latitude, "longitude": data.longitude},
        )

        result = compute_annual_composite(data, land_mask=mask)

        assert int(result["qa_count"].values[:, GRID // 2 :, :].max()) == 0
        assert int(result["qa_count"].values[:, : GRID // 2, :].max()) > 0
        assert result["lst_p95"].values[GRID // 2 :, :].max() == settings.nodata


class TestDaskDebias:
    """The lazy path must match the eager one."""

    def test_dask_matches_eager(self):
        import dask.array as da

        lst = _seasonal_stack()
        lazy = lst.copy(data=da.from_array(lst.values, chunks=(6, 20, 20)))

        eager_out, eager_off, eager_keep = _debias(lst)
        lazy_out, lazy_off, lazy_keep = _debias(lazy)

        np.testing.assert_allclose(eager_off.values, lazy_off.values, atol=1e-9)
        np.testing.assert_array_equal(eager_keep.values, lazy_keep.values)
        np.testing.assert_allclose(eager_out.values, np.asarray(lazy_out.values), atol=1e-9)

    def test_dask_discards_extreme_scene(self):
        import dask.array as da

        times = _times()
        bias = np.zeros(len(times))
        bias[4] = -73.0
        lst = _seasonal_stack(bias=bias)
        lazy = lst.copy(data=da.from_array(lst.values, chunks=(6, 20, 20)))

        debiased, _, keep = _debias(lazy)

        assert not bool(keep.values[4])
        assert debiased.sizes["time"] == len(times) - 1


def _smooth_stack(*, bias: np.ndarray | None = None, seed: int = 11) -> xr.DataArray:
    """Seasonal stack whose spatial field is smooth rather than white noise.

    Subsampling is only expected to be faithful when the field it samples has
    spatial structure, which real LST does. A white-noise field would make
    decimation a small random sample and the median correspondingly noisy, so a
    test built on one would measure sampling error rather than the property
    under test.
    """
    rng = np.random.default_rng(seed)
    times = _times()
    doy = pd.DatetimeIndex(times).dayofyear.values.astype("float64")

    yy, xx = np.meshgrid(
        np.linspace(0, 2 * np.pi, GRID), np.linspace(0, 2 * np.pi, GRID), indexing="ij"
    )
    spatial = 4.0 * np.sin(yy) * np.cos(xx)
    season = 15.0 * np.sin(2 * np.pi * (doy - 15) / 365)
    noise = rng.normal(0, 0.2, (len(times), GRID, GRID))

    data = 25.0 + season[:, None, None] + spatial[None, :, :] + noise
    if bias is not None:
        data = data + bias[:, None, None]
    return xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": times,
            "latitude": np.linspace(-33.4, -34.4, GRID),
            "longitude": np.linspace(-61.1, -60.1, GRID),
        },
    )


class TestCoarseOffsetEstimation:
    """Offsets estimated from a coarser grid (issue #46).

    The offset is one scalar per scene, so it does not need a full-resolution
    climatology. Estimating it from the source COGs' overviews cuts bytes read,
    which post-load subsampling cannot do.
    """

    def test_strided_offsets_match_full_resolution(self):
        """A 4x-decimated grid recovers the same per-scene offsets."""
        rng = np.random.default_rng(5)
        times = _times()
        bias = rng.normal(0, 3.0, len(times))
        lst = _smooth_stack(bias=bias)
        coarse = lst.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))

        full_offset, _ = scene_offsets(lst)
        coarse_offset, _ = scene_offsets(coarse)

        np.testing.assert_allclose(coarse_offset.values, full_offset.values, atol=0.15)

    def test_offset_source_drives_the_correction(self):
        """Offsets come from offset_source; the output keeps lst's own grid."""
        times = _times()
        bias = np.zeros(len(times))
        bias[7] = -40.0
        lst = _smooth_stack(bias=bias)
        coarse = lst.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))

        debiased, offset, keep = seasonal_debias(
            lst,
            max_offset_c=15.0,
            min_scene_pixels=1,
            min_offset_samples=1,
            offset_source=coarse,
        )

        # Output resolution is lst's, not the coarse grid's.
        assert debiased.sizes["latitude"] == GRID
        assert debiased.sizes["longitude"] == GRID
        # The coarse grid still saw the wrecked scene.
        assert not bool(keep.values[7])
        assert offset.values[7] < -30

    def test_misaligned_offset_source_raises(self):
        """A source covering different scenes is rejected, not broadcast."""
        lst = _smooth_stack()
        coarse = lst.isel(time=slice(0, 10), latitude=slice(None, None, 4))

        with pytest.raises(ValueError, match="does not share"):
            seasonal_debias(lst, max_offset_c=15.0, min_scene_pixels=1, offset_source=coarse)

    def test_sparse_guard_uses_the_estimation_grid(self):
        """With a coarse source the floor is min_offset_samples, not min_scene_pixels.

        A coarse valid-pixel count cannot be scaled back to a native one:
        averaging spreads data across nodata, so coarse loading over-reports
        coverage. Each grid therefore states its own floor.
        """
        lst = _smooth_stack()
        coarse = lst.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))
        n_coarse = coarse.sizes["latitude"] * coarse.sizes["longitude"]

        # min_scene_pixels far above the coarse pixel count is ignored, so
        # nothing is rejected for sparseness.
        _, _, keep = seasonal_debias(
            lst,
            max_offset_c=99.0,
            min_scene_pixels=10 * n_coarse,
            min_offset_samples=1,
            offset_source=coarse,
        )
        assert bool(keep.values.all())

        # The coarse floor is what bites.
        with pytest.raises(ValueError, match=r"All .* scenes rejected"):
            seasonal_debias(
                lst,
                max_offset_c=99.0,
                min_scene_pixels=1,
                min_offset_samples=n_coarse + 1,
                offset_source=coarse,
            )

    def test_native_path_is_unchanged(self):
        """Without offset_source, min_offset_samples has no effect."""
        lst = _smooth_stack()

        base = seasonal_debias(lst, max_offset_c=15.0, min_scene_pixels=500)
        with_arg = seasonal_debias(
            lst, max_offset_c=15.0, min_scene_pixels=500, min_offset_samples=10**9
        )

        np.testing.assert_array_equal(base[2].values, with_arg[2].values)
