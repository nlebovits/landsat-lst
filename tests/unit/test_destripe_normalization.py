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

from contextlib import ExitStack
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

    **Sub-second stamps, deliberately.** Real Landsat solar-day timestamps carry
    them, and the offsets are joined to a stack by coordinate *value*. A
    whole-second fixture cannot see a serializer that truncates the axis: the
    cache round-tripped, the join found every stamp, and the composite still
    failed on every shard of S30W065 against real data.
    """
    days = [5, 20][:per_month]
    dates = [(y, m, d) for y in years for m in range(1, 13) for d in days]
    stamps = [
        f"{y}-{m:02d}-{d:02d}T14:07:{i % 60:02d}.{(123_456 + 977 * i) % 1_000_000:06d}"
        for i, (y, m, d) in enumerate(dates)
    ]
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

    def test_graph_form_reads_the_stack_once(self):
        """Both reductions share one traversal, in the graph form.

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

        with (
            patch.object(settings, "destripe_bounded_units", False),
            patch.object(normalization.dask, "compute", counting_compute),
        ):
            offset, n_valid = scene_offsets(_seasonal_stack())

        assert len(calls) == 1, "the stack must be traversed once, not per reduction"
        assert len(calls[0]) == 2, "both reductions belong to the same graph"
        assert offset.sizes["time"] == n_valid.sizes["time"]

    def test_unit_form_reads_the_stack_exactly_twice(self):
        """C1 trades one read for bounded memory, and trades exactly one.

        Sharding the climatology over space and the offsets over scene means
        the two phases cannot share a traversal: they are parallel in
        orthogonal axes, which is the whole reason the single graph had to
        materialize the stack. Two passes is therefore the accepted cost, and
        pinning it is what stops it quietly becoming three -- a per-scene read
        inside phase A, or a re-read of the climatology, would not otherwise
        show up until a tile ran hours long.

        Counted as block *executions* via ``map_blocks``, the same trick
        ``tests/integration/test_cog.py`` uses on export. Counting task keys
        would answer a different question, since fusion renames keys freely.
        """
        import dask.array as da

        reads: list[int] = []

        def tally(block):
            # dask probes the function with a zero-size block to infer meta.
            # That is instrument overhead, not a read.
            if block.size:
                reads.append(1)  # list.append is atomic; the scheduler is threaded
            return block

        eager = _seasonal_stack()
        source = da.from_array(eager.values, chunks=(6, 20, 20))
        counted = source.map_blocks(tally, dtype=source.dtype)
        lst = eager.copy(data=counted)
        n_blocks = len(source.to_delayed().ravel())

        with patch.object(settings, "destripe_bounded_units", True):
            offset, n_valid = scene_offsets(lst)

        passes = len(reads) / n_blocks
        assert passes == pytest.approx(2.0, abs=0.01), (
            f"bounded units must read the stack twice, measured {passes:.2f}"
        )
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


class TestOffsetGraphFormulation:
    """The month-loop graph must reproduce the groupby formulation bit for bit.

    The 2026-08-15 reformulation exists because the groupby shuffle's graph
    grows superlinearly in scene count (>48.5 GB at 2,930 scenes, measured)
    while the month-loop holds the same values in 13.8 GB. These tests are the
    claim that lets it ship without an ALGORITHM_VERSION bump: same input,
    identical offsets, identical valid counts -- including under missing data,
    missing months, and NaN-only slices.
    """

    @staticmethod
    def _pair(lst):
        import dask

        from landsat_lst.normalization import _offset_graph_groupby, offset_graph

        new_o, new_n = dask.compute(*offset_graph(lst))
        old_o, old_n = dask.compute(*_offset_graph_groupby(lst))
        return (new_o, new_n), (old_o, old_n)

    def _assert_identical(self, lst):
        (new_o, new_n), (old_o, old_n) = self._pair(lst)
        np.testing.assert_array_equal(np.asarray(new_o.values), np.asarray(old_o.values))
        np.testing.assert_array_equal(np.asarray(new_n.values), np.asarray(old_n.values))
        assert list(new_o.time.values) == list(old_o.time.values)

    def test_bitwise_equal_on_chunked_input_with_gaps(self):
        """Random NaN holes, one all-NaN scene, one never-valid pixel."""
        import dask.array as da

        rng = np.random.default_rng(7)
        lst = _seasonal_stack()
        values = lst.values.copy()
        values[rng.random(values.shape) < 0.3] = np.nan
        values[4] = np.nan  # a scene with no data at all
        values[:, 3, 5] = np.nan  # a pixel with no data in any scene
        lst = lst.copy(data=values)
        lazy = lst.copy(data=da.from_array(values, chunks=(6, 20, 20)))

        self._assert_identical(lazy)

    def test_bitwise_equal_on_eager_input(self):
        self._assert_identical(_seasonal_stack())

    def test_bitwise_equal_when_months_are_missing(self):
        """A window covering three calendar months only."""
        import dask.array as da

        rng = np.random.default_rng(3)
        stamps = pd.to_datetime([f"{y}-{m:02d}-15" for y in (2021, 2022) for m in (1, 2, 3)]).values
        data = rng.normal(20.0, 5.0, (len(stamps), GRID, GRID))
        data[rng.random(data.shape) < 0.2] = np.nan
        lst = xr.DataArray(
            data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": stamps,
                "latitude": np.linspace(-33.4, -34.4, GRID),
                "longitude": np.linspace(-61.1, -60.1, GRID),
            },
        )
        self._assert_identical(lst.copy(data=da.from_array(data, chunks=(2, 20, 20))))

    def test_graph_carries_no_shuffle_layers(self):
        """The property the reformulation exists for.

        The groupby shuffle is the measured construction cliff; its key
        prefixes must not reappear in the graph the scheduler runs.
        """
        import dask.array as da

        from landsat_lst.normalization import offset_graph
        from landsat_lst.profiling import graph_stats

        lst = _seasonal_stack()
        lazy = lst.copy(data=da.from_array(lst.values, chunks=(6, 20, 20)))
        offset, n_valid = offset_graph(lazy)
        stats = graph_stats(xr.Dataset({"o": offset, "n": n_valid}), optimize=True)
        shuffled = [p.prefix for p in stats.by_prefix if "shuffle" in p.prefix]
        assert shuffled == [], f"shuffle layers back in the offset graph: {shuffled}"


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


class TestOffsetCacheIntegration:
    """The cache short-circuits the estimator without changing its answer."""

    def _cache(self, tmp_path, **overrides):
        from landsat_lst.offsets import OffsetCache, OffsetKey
        from landsat_lst.storage import LocalStorage

        key = OffsetKey.build(tile="N40W075", window="2021-2025", factor=2, scene_ids=("a", "b"))
        return OffsetCache(storage=LocalStorage(output_dir=tmp_path), key=key, **overrides)

    def test_a_hit_returns_without_computing(self, tmp_path):
        """The whole point: 27 minutes becomes a kilobyte read."""
        stack = _seasonal_stack()
        cache = self._cache(tmp_path)

        first = scene_offsets(stack, cache=cache)
        assert cache.last_read_hit is False

        with patch("landsat_lst.normalization.offset_graph") as never:
            second = scene_offsets(stack, cache=cache)
        never.assert_not_called()
        assert cache.last_read_hit is True
        np.testing.assert_allclose(second[0].values, first[0].values)
        np.testing.assert_array_equal(second[1].values, first[1].values)

    def test_a_cached_estimate_still_gets_the_live_rejection_rule(self, tmp_path):
        """Only the estimate is cached, never the decision.

        This is what makes a cap sweep cheap: the same stored offsets are
        re-judged against each candidate cap, so a sweep pays the estimator once
        rather than once per candidate.
        """
        bias = np.zeros(len(_times()))
        bias[0] = 40.0
        stack = _seasonal_stack(bias=bias)
        cache = self._cache(tmp_path)

        scene_offsets(stack, cache=cache)  # warm

        generous = seasonal_debias(stack, max_offset_c=100.0, min_scene_pixels=0, cache=cache)
        strict = seasonal_debias(stack, max_offset_c=5.0, min_scene_pixels=0, cache=cache)

        assert cache.last_read_hit is True
        assert int(generous[2].sum()) > int(strict[2].sum())

    def test_a_broken_cache_falls_back_to_computing(self, tmp_path):
        """A cache failure costs 27 minutes; raising would cost the run."""
        stack = _seasonal_stack()
        cache = self._cache(tmp_path)
        cache.storage.read_text = lambda _key: (_ for _ in ()).throw(OSError("gone"))
        cache.storage.write_text = lambda *_a, **_k: (_ for _ in ()).throw(OSError("gone"))

        offset, n_valid = scene_offsets(stack, cache=cache)
        assert offset.sizes["time"] == len(_times())
        assert int(n_valid.sum()) > 0


class TestRejectionRuleIsShared:
    """`landsat-lst offsets` must apply the identical rule a tile applies."""

    def test_keep_mask_matches_seasonal_debias(self):
        from landsat_lst.normalization import scene_keep_mask

        bias = np.zeros(len(_times()))
        bias[0] = 40.0
        stack = _seasonal_stack(bias=bias)

        offset, n_valid = scene_offsets(stack)
        direct = scene_keep_mask(offset, n_valid, max_offset_c=15.0, floor=0)
        _, _, via_debias = seasonal_debias(stack, max_offset_c=15.0, min_scene_pixels=0)

        np.testing.assert_array_equal(direct.values, via_debias.values)

    def test_floor_follows_the_grid_the_offset_rests_on(self, monkeypatch):
        """A coarse count cannot be scaled into a native one, so the floors swap."""
        from landsat_lst.normalization import rejection_floor

        monkeypatch.setattr(settings, "destripe_min_scene_pixels", 500)
        monkeypatch.setattr(settings, "destripe_min_offset_samples", 200)

        assert rejection_floor(offset_source_given=False) == 500
        assert rejection_floor(offset_source_given=True) == 200


class TestBoundedUnitsMatchTheGraph:
    """The bounded-unit form must be the same estimator, not a similar one.

    E1 established this on a real 300-scene fixture (max |delta| = 0, identical
    NaN patterns, identical valid counts, stable across two block sizes and two
    median kernels). These pin it on synthetic stacks so a regression fails in
    CI rather than five hours into a tile.
    """

    def _pair(self, lst, **kwargs):
        import dask

        from landsat_lst.normalization import offset_graph, offsets_as_units

        graph_o, graph_n = dask.compute(*offset_graph(lst))
        # ExitStack, not a bare setattr inside the patch block: a plain setattr
        # is never restored, and it leaked destripe_compute_panel into every
        # later test in the same process.
        with ExitStack() as stack:
            stack.enter_context(patch.object(settings, "destripe_bounded_units", True))
            for key, val in kwargs.items():
                stack.enter_context(patch.object(settings, key, val))
            unit_o, unit_n = offsets_as_units(lst)
        return (graph_o, graph_n), (unit_o, unit_n)

    def _assert_identical(self, lst, **kwargs):
        (g_o, g_n), (u_o, u_n) = self._pair(lst, **kwargs)
        np.testing.assert_array_equal(
            np.asarray(u_o.values, dtype="float64"),
            np.asarray(g_o.values, dtype="float64"),
        )
        np.testing.assert_array_equal(
            np.asarray(u_n.values, dtype="int64"), np.asarray(g_n.values, dtype="int64")
        )
        assert list(u_o.time.values) == list(g_o.time.values)

    def test_bitwise_equal_on_eager_input(self):
        self._assert_identical(_seasonal_stack())

    def test_bitwise_equal_on_chunked_input_with_gaps(self):
        """Random holes, an all-NaN scene, and a pixel valid in no scene."""
        import dask.array as da

        rng = np.random.default_rng(7)
        lst = _seasonal_stack()
        values = lst.values.copy()
        values[rng.random(values.shape) < 0.3] = np.nan
        values[4] = np.nan
        values[:, 3, 5] = np.nan
        self._assert_identical(lst.copy(data=da.from_array(values, chunks=(6, 20, 20))))

    def test_bitwise_equal_when_months_are_missing(self):
        stamps = pd.to_datetime([f"{y}-{m:02d}-15" for y in (2021, 2022) for m in (1, 2, 3)]).values
        rng = np.random.default_rng(3)
        data = rng.normal(20.0, 5.0, (len(stamps), GRID, GRID))
        data[rng.random(data.shape) < 0.2] = np.nan
        lst = xr.DataArray(
            data,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": stamps,
                "latitude": np.linspace(-33.4, -34.4, GRID),
                "longitude": np.linspace(-61.1, -60.1, GRID),
            },
        )
        self._assert_identical(lst)

    @pytest.mark.parametrize("panel", [8, 16, GRID])
    def test_panel_size_changes_nothing(self, panel):
        """E3 measured 256 as 1.8x faster; it must also be 1.0x different."""
        self._assert_identical(_seasonal_stack(), destripe_compute_panel=panel)

    @pytest.mark.parametrize("batch", [1, 3, 512])
    def test_scene_batch_changes_nothing(self, batch):
        """Phase B carries no state across scenes, so the batch is I/O only."""
        self._assert_identical(_seasonal_stack(), destripe_scene_batch=batch)

    def test_partial_edge_blocks(self):
        """A grid divisible by no block edge, which production also is not."""
        lst = _seasonal_stack().isel(latitude=slice(0, 37), longitude=slice(0, 23))
        self._assert_identical(lst, destripe_compute_panel=16)

    def test_block_edge_shrinks_as_the_window_grows(self):
        """Unit memory is meant to stay flat as scenes accumulate."""
        from landsat_lst.normalization import _io_block_edge

        lst = _seasonal_stack()
        wide = _io_block_edge(lst.isel(time=slice(0, 10)), 4.0)
        deep = _io_block_edge(lst, 4.0)
        assert deep <= wide
        assert wide & (wide - 1) == 0  # power of two

    def test_scene_offsets_dispatches_on_the_setting(self):
        """Both paths reachable, and the flag is what chooses."""
        from landsat_lst.normalization import scene_offsets

        lst = _seasonal_stack()
        with patch.object(settings, "destripe_bounded_units", True):
            unit_o, _ = scene_offsets(lst)
        with patch.object(settings, "destripe_bounded_units", False):
            graph_o, _ = scene_offsets(lst)
        np.testing.assert_array_equal(
            np.asarray(unit_o.values, dtype="float64"),
            np.asarray(graph_o.values, dtype="float64"),
        )
