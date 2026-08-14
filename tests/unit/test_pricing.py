"""Unit tests for cost estimation from a tile's billed time.

Nothing here reaches a pricing API. The point of the module under test is that
rates are committed data, so these tests read the same committed data and check
the arithmetic and, above all, the label. A cost that reads as derived when it
was substituted, or as a point when spot made it a band, is the failure worth
catching.
"""

from __future__ import annotations

import json

import pytest

from landsat_lst.config import settings
from landsat_lst.pricing import (
    DISCLAIMER,
    FLEET_TILES,
    PRICING_PATH,
    CostEstimate,
    CostRange,
    Lifecycle,
    billed_seconds,
    fleet_cost,
    instance_memory_gib,
    lifecycle_for_policy,
    tile_cost,
)
from landsat_lst.provenance import Provenance
from landsat_lst.tiling import LAND_TILES

pytestmark = pytest.mark.unit

HOUR_S = 3600.0
R6I_2XL_USD_HOUR = 0.504
M6I_4XL_USD_HOUR = 0.768
SPOT_LOW = 0.30
SPOT_HIGH = 0.75


@pytest.fixture
def table() -> dict:
    """The shipped price table, parsed."""
    return json.loads(PRICING_PATH.read_text())


def test_table_covers_every_configured_vm_type(table: dict) -> None:
    region = table["regions"][settings.coiled_region]
    assert set(settings.coiled_vm_types) <= set(region)


def test_table_prices_the_calibrated_and_oom_vms(table: dict) -> None:
    region = table["regions"][settings.coiled_region]
    assert {"r6i.4xlarge", "r6i.xlarge"} <= set(region)


def test_table_records_an_as_of_date(table: dict) -> None:
    assert table["as_of"].count("-") == 2


def test_published_figures_are_labelled_published(table: dict) -> None:
    assert table["billing"]["provenance"] == Provenance.PUBLISHED.value


def test_the_spot_band_is_labelled_assumed(table: dict) -> None:
    assert table["spot_band"]["provenance"] == Provenance.ASSUMED.value


def test_the_spot_band_records_the_quotes_behind_it(table: dict) -> None:
    band = table["spot_band"]
    fractions = [s["fraction"] for s in band["samples"]]
    assert len(fractions) >= 3
    assert min(fractions) >= band["low_fraction"]
    assert max(fractions) <= band["high_fraction"]


def test_configured_vm_types_agree_on_memory(table: dict) -> None:
    region = table["regions"][settings.coiled_region]
    sizes = {region[vm]["memory_gib"] for vm in settings.coiled_vm_types}
    assert sizes == {64.0}


def test_fleet_tiles_tracks_the_land_tile_set() -> None:
    assert len(LAND_TILES) == FLEET_TILES


def test_a_short_run_bills_the_minute_minimum() -> None:
    assert billed_seconds(10.375) == 60.0


def test_a_long_run_bills_its_wall_clock() -> None:
    assert billed_seconds(1234.5) == 1234.5


def test_on_demand_hour_costs_the_list_rate() -> None:
    estimate = tile_cost(duration_s=HOUR_S, instance_type="r6i.2xlarge", lifecycle="on-demand")
    assert estimate is not None
    assert estimate.usd.is_point
    assert estimate.usd.low == pytest.approx(R6I_2XL_USD_HOUR)
    assert estimate.provenance is Provenance.DERIVED


def test_duration_scales_the_cost_linearly() -> None:
    estimate = tile_cost(duration_s=HOUR_S / 2, instance_type="m6i.4xlarge", lifecycle="on-demand")
    assert estimate is not None
    assert estimate.usd.low == pytest.approx(M6I_4XL_USD_HOUR / 2)


def test_a_ten_second_tile_is_priced_at_a_full_minute() -> None:
    estimate = tile_cost(duration_s=10.375, instance_type="r6i.2xlarge", lifecycle="on-demand")
    assert estimate is not None
    assert estimate.billed_s == 60.0
    assert estimate.duration_s == 10.375
    assert estimate.usd.low == pytest.approx(R6I_2XL_USD_HOUR / 60)


def test_spot_prices_to_a_band() -> None:
    estimate = tile_cost(duration_s=HOUR_S, instance_type="m6i.4xlarge", lifecycle="spot")
    assert estimate is not None
    assert estimate.usd.low == pytest.approx(M6I_4XL_USD_HOUR * SPOT_LOW)
    assert estimate.usd.high == pytest.approx(M6I_4XL_USD_HOUR * SPOT_HIGH)
    assert estimate.provenance is Provenance.ASSUMED
    assert estimate.lifecycle is Lifecycle.SPOT


def test_an_unreported_lifecycle_spans_spot_to_on_demand() -> None:
    estimate = tile_cost(duration_s=HOUR_S, instance_type="r6i.2xlarge", lifecycle=None)
    assert estimate is not None
    assert estimate.usd.low == pytest.approx(R6I_2XL_USD_HOUR * SPOT_LOW)
    assert estimate.usd.high == pytest.approx(R6I_2XL_USD_HOUR)
    assert estimate.lifecycle is Lifecycle.UNKNOWN
    assert estimate.provenance is Provenance.ASSUMED


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("on-demand", Lifecycle.ON_DEMAND),
        ("spot", Lifecycle.SPOT),
        ("spot_with_fallback", Lifecycle.UNKNOWN),
    ],
)
def test_policy_maps_to_a_lifecycle(policy: str, expected: Lifecycle) -> None:
    assert lifecycle_for_policy(policy) is expected


def test_an_unreported_lifecycle_follows_an_unambiguous_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "coiled_spot_policy", "on-demand")
    estimate = tile_cost(duration_s=HOUR_S, instance_type="r6i.2xlarge", lifecycle=None)
    assert estimate is not None
    assert estimate.usd.is_point
    assert estimate.provenance is Provenance.DERIVED


def test_an_unrecognised_lifecycle_string_counts_as_unreported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "coiled_spot_policy", "spot")
    estimate = tile_cost(duration_s=HOUR_S, instance_type="r6i.2xlarge", lifecycle="preempted")
    assert estimate is not None
    assert estimate.lifecycle is Lifecycle.SPOT


def test_unknown_instance_type_substitutes_and_says_so() -> None:
    estimate = tile_cost(duration_s=HOUR_S, instance_type="c7g.16xlarge", lifecycle="on-demand")
    assert estimate is not None
    assert estimate.instance_type == settings.coiled_vm_types[0]
    assert estimate.provenance is Provenance.ASSUMED


def test_unknown_region_refuses_rather_than_borrowing_rates() -> None:
    assert (
        tile_cost(
            duration_s=HOUR_S,
            instance_type="r6i.2xlarge",
            lifecycle="on-demand",
            region="ap-southeast-4",
        )
        is None
    )


def test_an_empty_instance_type_never_raises() -> None:
    estimate = tile_cost(duration_s=1.0, instance_type="", lifecycle=None)
    assert estimate is not None
    assert estimate.usd.low > 0.0


def _estimate(low: float, high: float, provenance: Provenance) -> CostEstimate:
    return CostEstimate(
        usd=CostRange(low, high, provenance),
        usd_hour=CostRange(R6I_2XL_USD_HOUR, R6I_2XL_USD_HOUR, Provenance.PUBLISHED),
        duration_s=HOUR_S,
        billed_s=HOUR_S,
        instance_type="r6i.2xlarge",
        region=settings.coiled_region,
        lifecycle=Lifecycle.ON_DEMAND,
        provenance=provenance,
    )


def test_fleet_cost_multiplies_the_observed_mean() -> None:
    fleet = fleet_cost(
        [_estimate(1.0, 1.0, Provenance.DERIVED), _estimate(3.0, 3.0, Provenance.DERIVED)],
        tiles=700,
    )
    assert fleet is not None
    assert fleet.mean_usd_per_tile.low == pytest.approx(2.0)
    assert fleet.usd.low == pytest.approx(1400.0)
    assert fleet.observed_tiles == 2


def test_fleet_cost_keeps_the_band_from_its_inputs() -> None:
    fleet = fleet_cost([_estimate(1.0, 4.0, Provenance.ASSUMED)], tiles=10)
    assert fleet is not None
    assert fleet.usd.low == pytest.approx(10.0)
    assert fleet.usd.high == pytest.approx(40.0)


def test_fleet_cost_defaults_to_the_land_tile_count() -> None:
    fleet = fleet_cost([_estimate(1.0, 1.0, Provenance.DERIVED)])
    assert fleet is not None
    assert fleet.tiles == FLEET_TILES


def test_extrapolating_past_the_observed_tiles_is_assumed() -> None:
    fleet = fleet_cost([_estimate(1.0, 1.0, Provenance.DERIVED)], tiles=700)
    assert fleet is not None
    assert fleet.provenance is Provenance.ASSUMED


def test_a_fully_observed_fleet_keeps_its_inputs_provenance() -> None:
    fleet = fleet_cost(
        [_estimate(1.0, 1.0, Provenance.DERIVED), _estimate(3.0, 3.0, Provenance.DERIVED)],
        tiles=2,
    )
    assert fleet is not None
    assert fleet.provenance is Provenance.DERIVED


def test_one_assumed_tile_weakens_the_fleet() -> None:
    fleet = fleet_cost(
        [_estimate(1.0, 1.0, Provenance.DERIVED), _estimate(3.0, 3.0, Provenance.ASSUMED)],
        tiles=2,
    )
    assert fleet is not None
    assert fleet.provenance is Provenance.ASSUMED


def test_no_observations_gives_nothing_rather_than_zero() -> None:
    assert fleet_cost([], tiles=700) is None


def test_instance_memory_is_reported_for_a_known_type() -> None:
    assert instance_memory_gib("r6i.2xlarge") == pytest.approx(64.0)
    assert instance_memory_gib("r6i.4xlarge") == pytest.approx(128.0)


def test_instance_memory_is_none_for_an_unknown_type() -> None:
    assert instance_memory_gib("c7g.16xlarge") is None


def test_disclaimer_names_what_it_omits() -> None:
    for omission in ("Coiled", "S3", "provisioning"):
        assert omission in DISCLAIMER
