"""Unit tests for the benchmark harness's pure logic.

Nothing here spawns a subprocess or builds a graph. The measurement itself is
exercised by ``tests/benchmark``; what these pin is the arithmetic and the
verdict, which is the part a reader of ``docs/findings-memory-model.md`` will
be trusting.
"""

from __future__ import annotations

import pytest

from landsat_lst.benchmarks import (
    CI_GEOMETRY,
    GRAPH_BOTH,
    MIN_PEAK_SPREAD,
    Geometry,
    Measurement,
    benchmark_key,
    sweep_report,
)

pytestmark = pytest.mark.unit


def _point(scenes: int, peak_mb: float, floor_mb: float = 1000.0, tasks: int = 0) -> Measurement:
    return Measurement(
        geometry=Geometry(scenes=scenes, graph=GRAPH_BOTH),
        peak_rss_mb=peak_mb,
        floor_mb=floor_mb,
        offset_tasks=tasks or scenes * 100,
    )


class TestGeometry:
    def test_side_is_blocks_times_chunk(self):
        assert Geometry(scenes=10, blocks=8, chunk=512).side == 4096

    def test_ci_geometry_streams(self):
        """More blocks than threads, or dask never streams and peak is a baseline."""
        assert CI_GEOMETRY.blocks**2 > CI_GEOMETRY.threads

    def test_label_names_every_lever(self):
        label = Geometry(scenes=24, blocks=4, chunk=256, threads=2).label
        assert "24sc" in label
        assert "1024x1024" in label
        assert "c256" in label
        assert "t2" in label


class TestMeasurement:
    def test_ok_is_false_when_an_error_is_set(self):
        assert _point(10, 100.0).ok
        assert not Measurement(geometry=CI_GEOMETRY, error="boom").ok

    def test_peak_over_floor_is_the_ratio(self):
        assert _point(10, 3000.0, floor_mb=1000.0).peak_over_floor == pytest.approx(3.0)

    def test_ratios_are_zero_rather_than_dividing_by_zero(self):
        """A failed child reports zeros; a benchmark must not die reading them."""
        blank = Measurement(geometry=CI_GEOMETRY, error="failed")
        assert blank.peak_over_floor == 0.0
        assert blank.native_passes == 0.0

    def test_native_passes_counts_block_executions(self):
        m = _point(10, 100.0)
        m.source_blocks, m.source_reads = 48, 96
        assert m.native_passes == pytest.approx(2.0)

    def test_as_dict_is_json_safe_and_carries_the_derived_numbers(self):
        import json

        m = _point(10, 2000.0, floor_mb=1000.0)
        m.source_blocks, m.source_reads = 10, 10
        payload = m.as_dict()

        assert payload["peak_over_floor"] == 2.0
        assert payload["native_passes"] == 1.0
        assert payload["geometry"]["scenes"] == 10
        json.dumps(payload)  # must not raise


class TestSweepReport:
    def test_too_few_points_refuses_to_fit(self):
        assert sweep_report([_point(50, 1000.0)])["verdict"] == "insufficient"

    def test_failed_points_do_not_count_toward_a_fit(self):
        results = [_point(50, 1000.0), Measurement(geometry=CI_GEOMETRY, error="oom")]
        assert sweep_report(results)["verdict"] == "insufficient"

    def test_flat_peak_refuses_to_extrapolate(self):
        """ADR-011: a number from the wrong regime is worse than no number."""
        results = [_point(n, 1000.0 + n * 0.01) for n in (50, 100, 200, 400)]

        report = sweep_report(results)

        assert report["verdict"] == "not_streaming"
        assert report["streaming_regime"] is False
        assert "projected_peak_mb" not in report
        assert report["peak_spread"] < MIN_PEAK_SPREAD

    def test_constant_ratio_yields_a_correction_factor(self):
        # Peak and floor both linear in scenes: the ratio holds.
        results = [_point(n, peak_mb=20.0 * n, floor_mb=10.0 * n) for n in (50, 100, 200, 400)]

        report = sweep_report(results)

        assert report["verdict"] == "constant_ratio"
        assert report["streaming_regime"] is True
        assert report["mean_peak_over_floor"] == pytest.approx(2.0)
        assert report["projected_peak_mb"] > 0

    def test_growing_ratio_localizes_a_leak(self):
        # Peak grows quadratically against a linear floor.
        results = [_point(n, peak_mb=0.5 * n * n, floor_mb=10.0 * n) for n in (50, 100, 200, 400)]

        report = sweep_report(results)

        assert report["verdict"] == "growing_ratio"
        assert report["ratio_growth"] > 1.5

    def test_projection_flags_a_configuration_that_will_not_fit(self):
        results = [_point(n, peak_mb=100.0 * n, floor_mb=10.0 * n) for n in (50, 100, 200, 400)]

        report = sweep_report(results, target=2930)

        # 100 MB per scene x 2,930 scenes is 286 GB against a 64 GiB VM.
        assert report["projected_fits_vm"] is False

    def test_task_projection_survives_a_non_streaming_sweep(self):
        """Task count is arithmetic over the graph, so it fits in any regime."""
        results = [_point(n, 1000.0, tasks=n * 100) for n in (50, 100, 200)]

        report = sweep_report(results, target=1000)

        assert report["verdict"] == "not_streaming"
        assert report["projected_offset_tasks"] == pytest.approx(100_000, rel=0.01)


class TestBenchmarkKey:
    def test_key_is_outside_the_run_prefix(self):
        """A sweep is not a tile and must never appear in a run manifest."""
        key = benchmark_key("scaling-20260814T120000Z")
        assert key.startswith("_benchmarks/")
        assert not key.startswith("_runs/")

    def test_key_is_stable_for_a_run_id(self):
        assert benchmark_key("abc") == benchmark_key("abc")
