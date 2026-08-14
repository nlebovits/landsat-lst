"""Unit tests for the strings ``watch`` and ``explain`` both print.

Every function here returns text, so every test is an equality against text.
That is the point of the module: the two commands compose the same figures
from the same calls rather than from two copies that drift apart.
"""

import pytest

from landsat_lst.render import (
    bar,
    format_duration,
    format_gib,
    format_money,
    format_money_range,
    format_rate,
    phase_rows,
    provenance_tag,
    sparkline,
    strip_ansi,
    truncate,
)

pytestmark = pytest.mark.unit


def test_format_duration():
    assert format_duration(None) == "-"
    assert format_duration(0) == "0s"
    assert format_duration(14.6) == "14s"
    assert format_duration(501) == "8m21s"
    assert format_duration(3720) == "1h02m"


def test_format_gib():
    assert format_gib(None) == "-"
    assert format_gib(35942) == "35.1G"
    assert format_gib(0) == "0.0G"


class TestFormatRate:
    def test_unmeasurable_is_not_zero(self):
        """``-`` is a fresh graph. ``0/s`` is a stalled one. They differ."""
        assert format_rate(None) == "-"
        assert format_rate(0) == "0/s"

    def test_scales_precision_to_the_magnitude(self):
        assert format_rate(0.25) == "0.25/s"
        assert format_rate(12.4) == "12.4/s"
        assert format_rate(1840.0) == "1840/s"


class TestFormatMoney:
    def test_a_cost_under_a_cent_is_not_free(self):
        assert format_money(0.004) == "<$0.01"

    def test_renders_dollars(self):
        assert format_money(None) == "-"
        assert format_money(0) == "$0.00"
        assert format_money(1234.5) == "$1,234.50"

    def test_a_range_collapses_when_both_ends_agree(self):
        assert format_money_range(0.42, 0.42) == "$0.42"
        assert format_money_range(0.13, 0.42) == "$0.13-$0.42"
        assert format_money_range(None, 0.42) == "-"


class TestSparkline:
    def test_one_point_is_not_a_trend(self):
        assert sparkline([]) == ("", 0.0)
        assert sparkline([12.0]) == ("", 0.0)
        assert sparkline([12.0, None]) == ("", 0.0)

    def test_scales_from_zero_not_from_the_minimum(self):
        """The acceptance case: a flat series must not draw like a climb."""
        flat, _ = sparkline([34.9, 35.0, 35.1])
        climb, _ = sparkline([6.0, 20.0, 35.1])

        assert set(flat) == {"█"}
        assert climb[0] != climb[-1]

    def test_reports_the_value_the_top_block_means(self):
        line, top = sparkline([1.0, 4.0])

        assert top == 4.0
        assert line[-1] == "█"

    def test_decimates_by_the_maximum_so_a_spike_survives(self):
        line, top = sparkline([99.0, 1.0, 1.0, 1.0, 1.0, 1.0], width=2)

        assert len(line) == 2
        assert top == 99.0
        assert line[0] == "█"
        assert line[1] == "▁"

    def test_an_all_zero_series_draws_the_lowest_block(self):
        line, top = sparkline([0.0, 0.0, 0.0])

        assert top == 0.0
        assert set(line) == {"▁"}


class TestBar:
    def test_fills_to_the_fraction(self):
        assert bar(0.5, width=10) == "█████░░░░░"
        assert bar(0.0, width=4) == "░░░░"
        assert bar(1.0, width=4) == "████"

    def test_clamps_rather_than_raising(self):
        assert bar(1.4, width=4) == "████"
        assert bar(-2.0, width=4) == "░░░░"

    def test_no_number_draws_nothing(self):
        assert bar(None, width=4) == ""


class TestPhaseRows:
    def test_orders_by_the_pipeline_not_by_size(self):
        rows = phase_rows({"destriping": 1620.0, "loading": 240.0})

        assert [name for name, _, _ in rows] == ["loading", "destriping"]

    def test_scales_bars_against_the_longest_phase(self):
        rows = {name: drawn for name, _, drawn in phase_rows({"a": 10.0, "b": 5.0}, width=10)}

        assert rows["a"] == "██████████"
        assert rows["b"] == "█████░░░░░"

    def test_marks_the_phase_the_tile_is_in(self):
        rows = phase_rows({"loading": 60.0, "destriping": 120.0}, current="destriping")
        marked = {name: drawn for name, _, drawn in rows}

        assert "←" in marked["destriping"]
        assert "←" not in marked["loading"]

    def test_reports_each_duration(self):
        assert phase_rows({"loading": 501.0})[0][1] == "8m21s"

    def test_a_phase_this_code_has_never_heard_of_still_reports(self):
        """A run in flight during a deploy must not lose its time."""
        rows = phase_rows({"loading": 10.0, "coverage_check": 30.0})

        assert [name for name, _, _ in rows] == ["loading", "coverage_check"]

    def test_nothing_recorded_draws_nothing(self):
        assert phase_rows({}) == []


class TestProvenanceTag:
    def test_escapes_the_bracket_rich_would_eat(self):
        assert provenance_tag("measured", "imds") == "\\[measured: imds]"

    def test_drops_empty_labels(self):
        assert provenance_tag("spot", None) == "\\[spot]"
        assert provenance_tag() == ""


class TestTruncate:
    def test_leaves_short_text_alone(self):
        assert truncate("No scenes found", 44) == "No scenes found"

    def test_marks_what_it_cut(self):
        assert truncate("abcdef", 4) == "abc…"

    def test_no_room_at_all_gives_nothing(self):
        assert truncate("abcdef", 0) == ""


class TestStripAnsi:
    """A captured log is a tee of a real terminal, escapes and all."""

    def test_removes_colour_codes(self):
        assert strip_ansi("\x1b[31mAPIError\x1b[0m: 500") == "APIError: 500"

    def test_leaves_plain_text_alone(self):
        assert strip_ansi("APIError: 500") == "APIError: 500"

    def test_keeps_the_content_of_a_rich_traceback(self):
        """The box drawing is literal text and survives; only the colour goes."""
        line = "\x1b[31m│\x1b[0m   resp = <Response [500]>"

        assert strip_ansi(line) == "│   resp = <Response [500]>"
