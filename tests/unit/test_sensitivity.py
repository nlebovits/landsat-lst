"""The valid-area sensitivity machinery: shape, honesty, and no hidden results.

#120 requires the 1/9, 5/9, 9/9 check to exist and to be bounded before it is
run. These tests check that the machinery reports what it measured and refuses
to imply more -- they are not the sensitivity result, which is a measurement
against real crops and is not produced here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.sensitivity import (
    STABILITY_BOUNDS,
    THRESHOLDS,
    SensitivityReport,
    run_threshold_sweep,
)

pytestmark = pytest.mark.unit

FACTOR = 3


def _stack(values: np.ndarray) -> xr.DataArray:
    scenes, rows, cols = values.shape
    return xr.DataArray(
        values,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": np.array(
                [
                    np.datetime64("2021-06-01T13:45:12.482915") + np.timedelta64(i * 34, "D")
                    for i in range(scenes)
                ]
            ),
            "latitude": 40.0 - np.arange(rows) / 3600,
            "longitude": -75.0 + np.arange(cols) / 3600,
        },
    )


class TestPreRegistration:
    """The arms and the bounds are declared, not chosen after looking."""

    def test_the_arms_are_the_three_the_decision_names(self):
        assert THRESHOLDS == (1, 5, 9)

    def test_the_default_sits_between_the_extremes(self):
        assert THRESHOLDS[0] < settings.min_valid_source_cells < THRESHOLDS[-1]

    def test_the_bounds_are_stated_as_constants(self):
        """Widening one has to show up in a diff, which is the point of them."""
        assert set(STABILITY_BOUNDS) == {
            "min_rank_correlation",
            "min_hotspot_agreement",
            "max_abs_delta_c",
        }
        assert STABILITY_BOUNDS["min_rank_correlation"] == 0.99


class TestSweep:
    """One stack, three arms, differing only in the valid-area rule."""

    def test_every_arm_is_reported(self):
        arms, _ = run_threshold_sweep(_stack(np.full((6, 9, 9), 30.0, "float32")), crop="c")
        assert [row["min_valid_cells"] for row in arms["c"]] == list(THRESHOLDS)

    def test_a_looser_threshold_never_resolves_fewer_cells(self):
        """Coverage is monotone in the threshold. Anything else is a defect."""
        rng = np.random.default_rng(7)
        values = rng.normal(30.0, 5.0, (8, 9, 9)).astype("float32")
        values[rng.random(values.shape) < 0.5] = np.nan

        arms, _ = run_threshold_sweep(_stack(values), crop="c")
        coverage = [row["coverage"] for row in arms["c"]]

        assert coverage == sorted(coverage, reverse=True)

    def test_a_fully_clear_stack_agrees_across_every_arm(self):
        """With nine of nine valid everywhere, the rule cannot bite."""
        arms, comparisons = run_threshold_sweep(
            _stack(np.full((6, 9, 9), 30.0, "float32")), crop="c"
        )
        assert {row["coverage"] for row in arms["c"]} == {1.0}
        assert all(row["max_abs_delta_c"] == 0.0 for row in comparisons["c"])

    def test_comparisons_are_against_the_default_and_exclude_it(self):
        _, comparisons = run_threshold_sweep(_stack(np.full((6, 9, 9), 30.0, "float32")), crop="c")
        arms_compared = [row["min_valid_cells"] for row in comparisons["c"]]
        assert settings.min_valid_source_cells not in arms_compared
        assert set(arms_compared) == set(THRESHOLDS) - {settings.min_valid_source_cells}

    def test_gained_and_lost_cells_are_counted_separately(self):
        """A threshold moves coverage in one direction; say which."""
        values = np.full((6, 9, 9), 30.0, dtype="float32")
        # One block loses four cells: valid under 1/9 and 5/9, not under 9/9.
        values[:, 0:2, 0:2] = np.nan

        _, comparisons = run_threshold_sweep(_stack(values), crop="c")
        strict = next(r for r in comparisons["c"] if r["min_valid_cells"] == 9)

        assert strict["cells_lost"] == 1
        assert strict["cells_gained"] == 0


class TestReport:
    """The verdict is mechanical, and it never picks a threshold."""

    def test_an_empty_report_has_no_verdict(self):
        assert SensitivityReport().stable() is None

    def test_agreement_inside_the_bounds_reads_as_stable(self):
        report = SensitivityReport(
            crops=["c"],
            comparisons={
                "c": [
                    {
                        "min_valid_cells": 1,
                        "rank_correlation": 0.999,
                        "hotspot_agreement": 0.98,
                        "max_abs_delta_c": 0.2,
                    }
                ]
            },
        )
        assert report.stable() is True

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("rank_correlation", 0.5),
            ("hotspot_agreement", 0.1),
            ("max_abs_delta_c", 9.0),
        ],
    )
    def test_any_single_bound_breached_reads_as_unstable(self, field, value):
        row = {
            "min_valid_cells": 1,
            "rank_correlation": 0.999,
            "hotspot_agreement": 0.98,
            "max_abs_delta_c": 0.2,
        }
        row[field] = value
        assert SensitivityReport(crops=["c"], comparisons={"c": [row]}).stable() is False

    def test_the_written_report_carries_the_bounds_it_was_judged_against(self, tmp_path):
        """A number without its threshold is not evidence of anything."""
        report = SensitivityReport(crops=["c"], arms={"c": []}, comparisons={"c": []})
        payload = json.loads(report.write(tmp_path / "s.json").read_text())

        assert payload["thresholds"] == list(THRESHOLDS)
        assert payload["default"] == settings.min_valid_source_cells
        assert payload["stability_bounds"] == STABILITY_BOUNDS

    def test_the_module_ships_no_measured_result(self):
        """It is machinery. A default-constructed report holds nothing."""
        report = SensitivityReport()
        assert report.crops == [] and report.arms == {} and report.comparisons == {}
