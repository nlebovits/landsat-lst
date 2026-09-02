"""Guards on the fleet cost model's anchor, its buckets, and its ranges.

Three things are pinned here and each has already been got wrong once.

**The anchor's reconciliation.** The model is usable because three figures from
one run agree: the fleet shape prices to $16.38 on-demand against $7.28
reported (0.445, matching the 0.44 sample in ``pricing.json``), and the same
shape is 317.73 vCPU-hours against 268.11 credits reported (0.844, inside the
0.6-1.25 band in ``quota.py``). Editing the shape without editing the anchors
breaks that agreement silently and turns every projection into arithmetic about
nothing.

**The buckets.** No billing export for that run exists in any commit of this
repository, so the anchor is ``user_reported`` and a test says so. An earlier
draft labelled it measured, which reads as an invoice nobody can produce.

**The ranges.** Capture and the credit price are unmeasured, so the model emits
intervals and a formula. A scalar would price an unverified premise as settled.

Bands, never exact values, following the repository's benchmark discipline: a
test that fails on rounding gets disabled within a month.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load scripts/fleet_cost_model.py (scripts/ is not a package).
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fleet_cost_model.py"
_spec = importlib.util.spec_from_file_location("fleet_cost_model", _SCRIPT)
fcm = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fcm
_spec.loader.exec_module(fcm)

pytestmark = pytest.mark.unit


class TestReconciliation:
    """The three figures from one run have to keep agreeing."""

    def test_the_implied_spot_factor_sits_inside_the_published_band(self):
        t = fcm.reference_totals()
        lo, hi = fcm.SPOT_BAND
        assert lo <= t["implied_spot_factor"] <= hi

    def test_the_implied_spot_factor_matches_the_pricing_json_sample(self):
        # pricing.json samples r6i.4xlarge at 0.44 on the same day; the run
        # implies 0.445. Agreement within 0.05 is the corroboration.
        assert abs(fcm.reference_totals()["implied_spot_factor"] - 0.44) < 0.05

    def test_the_implied_credit_rate_sits_inside_the_observed_band(self):
        lo, hi = fcm.CREDITS_PER_VCPU_HOUR_BAND
        assert lo <= fcm.reference_totals()["implied_credits_per_vcpu_hour"] <= hi


class TestDecomposition:
    """Boot, useful compute, and idle, summing to the reported shape."""

    def test_useful_compute_is_a_minority_of_billed_vm_time(self):
        # The consolidation argument rests on this. If a change makes useful
        # compute the majority, the argument changed and the document citing
        # it has to change too.
        d = fcm.decompose_reference_run()
        assert d.useful_vm_min / d.total_vm_min < 0.5

    def test_no_term_is_negative(self):
        d = fcm.decompose_reference_run()
        assert min(d.boot_vm_min, d.useful_vm_min, d.idle_vm_min) >= 0.0

    def test_a_recovery_round_is_mostly_boot(self):
        # offsets_round_2 lived 7 minutes against a 5-minute boot. A model that
        # let a short round claim the full compute budget would understate what
        # retries cost.
        d = fcm.decompose_reference_run()
        r2 = next(c for c in d.per_cluster if c["cluster"] == "offsets_round_2")
        assert r2["vm_min_boot"] > r2["vm_min_useful"]


class TestProvenance:
    """Five buckets, and the anchor is not in the strongest one."""

    def test_every_registered_quantity_carries_a_known_bucket(self):
        assert {q.bucket for q in fcm.REGISTER} <= set(fcm.BUCKETS)

    def test_the_billing_anchors_are_user_reported_not_measured(self):
        # No invoice, no cost export, no billing artifact exists in any commit.
        # Labelling these measured claims an artifact nobody can produce.
        buckets = {q.name: q.bucket for q in fcm.REGISTER}
        assert buckets["reference_billed_aws_usd"] == "user_reported"
        assert buckets["reference_billed_credits"] == "user_reported"
        assert buckets["reference_fleet_shape"] == "user_reported"

    def test_unknown_quantities_carry_no_value(self):
        # An unknown given a value is an unknown priced at that value.
        for q in fcm.REGISTER:
            if q.bucket == "unknown":
                assert q.value is None, q.name

    def test_the_three_blocking_unknowns_are_registered(self):
        unknown = {q.name for q in fcm.REGISTER if q.bucket == "unknown"}
        assert {
            "queues_surplus_holds",
            "credit_unit_price_usd",
            "usable_credit_quota",
        } <= unknown

    def test_every_quantity_names_what_would_strengthen_it(self):
        assert all(q.upgrade.strip() for q in fcm.REGISTER)

    def test_a_bad_bucket_is_refused(self):
        with pytest.raises(ValueError):
            fcm.Quantity("x", 1, "u", "probably", "s", "u")


class TestScaling:
    """Equivalents follow each stage's own byte model."""

    def _counts(self):
        return {"S30W065": 1000, "A": 1000, "B": 500}

    def test_composite_equivalents_ignore_land_fraction(self):
        # The native footprint is read regardless of land.
        counts = self._counts()
        dry = fcm.tile_equivalents(counts, {}, "S30W065")
        wet = fcm.tile_equivalents(counts, dict.fromkeys(counts, 0.1), "S30W065")
        assert dry["composite_equivalents"] == wet["composite_equivalents"]

    def test_offsets_equivalents_respond_to_land_fraction(self):
        counts = self._counts()
        ref = dict.fromkeys(counts, 1.0)
        less = {"S30W065": 1.0, "A": 0.0, "B": 0.0}
        assert (
            fcm.tile_equivalents(counts, less, "S30W065")["offsets_equivalents"]
            < fcm.tile_equivalents(counts, ref, "S30W065")["offsets_equivalents"]
        )


class TestLayers:
    """Compute and provisioning are reported apart, and stay apart."""

    def _equiv(self):
        return fcm.tile_equivalents({"S30W065": 4138, "A": 4138, "B": 2000}, {}, "S30W065")

    def test_the_composite_dominates_compute(self):
        # 88% on the reference shape. Every further saving after consolidation
        # has to come from that pass rather than from scheduling.
        assert fcm.layers(self._equiv())["compute_composite_share"] > 0.8

    def test_consolidation_touches_provisioning_only(self):
        # Capture must never move the compute line. If it does, the model has
        # started discounting work that has to happen.
        equiv = self._equiv()
        base = fcm.layers(equiv)["compute_usd_on_demand"]
        for scenario in fcm.CAPTURE_BANDS:
            assert fcm.layers(equiv)["compute_usd_on_demand"] == base
            assert fcm.project(equiv, scenario=scenario)["capture_band"]["low"] >= 0.0


class TestProjection:
    """Intervals, never scalars, and no credit ever priced."""

    def _equiv(self):
        return fcm.tile_equivalents({"S30W065": 4138, "A": 4138, "B": 2000}, {}, "S30W065")

    def test_every_interval_is_ordered_and_positive(self):
        for scenario in fcm.CAPTURE_BANDS:
            p = fcm.project(self._equiv(), scenario=scenario)
            for key in ("aws_usd_spot", "coiled_credits", "aws_usd_on_demand_basis"):
                assert 0 < p[key]["low"] <= p[key]["high"], (scenario, key)

    def test_more_capture_costs_less(self):
        equiv = self._equiv()
        none = fcm.project(equiv, scenario="queues_surplus_false")
        band = fcm.project(equiv, scenario="design_band")
        assert band["aws_usd_spot"]["high"] < none["aws_usd_spot"]["high"]

    def test_the_failed_premise_scenario_captures_nothing(self):
        # If queues_surplus is false the design buys nothing, and the model has
        # to be able to say so rather than assuming the premise holds.
        p = fcm.project(self._equiv(), scenario="queues_surplus_false")
        assert p["capture_band"] == {"low": 0.0, "high": 0.0}

    def test_credits_are_a_quantity_and_never_dollars(self):
        # The dollar price of a credit is unknown. A model that invented one
        # would fold an unbounded term into a headline figure.
        p = fcm.project(self._equiv(), scenario="conservative")
        assert p["coiled_usd"] is None
        assert p["coiled_credits"]["low"] > 0

    def test_no_scenario_claims_the_build_fits_the_ceiling(self):
        # $3,000 is an approval ceiling. A model output is not a finding about
        # it, least of all while the credit term is unpriced.
        for scenario in fcm.CAPTURE_BANDS:
            verdict = fcm.project(self._equiv(), scenario=scenario)["vs_approval_ceiling"]
            assert "not a target" in verdict
            assert "unpriced credit term" in verdict

    def test_consolidation_never_beats_the_compute_floor(self):
        equiv = self._equiv()
        floor = fcm.layers(equiv)["compute_usd_on_demand"]
        best = fcm.project(equiv, scenario="design_band")
        assert best["aws_usd_on_demand_basis"]["low"] >= floor


class TestRegime:
    """The 30 m stamp, so a figure cannot be carried across #121 silently."""

    def test_the_report_names_the_pipeline_it_describes(self):
        report = fcm.build_report(fcm.RESULTS / "scene_counts.json", fcm.RESULTS)
        assert "30 m" in report["pipeline_regime"]
        assert "#121" in report["regime_note"]

    def test_the_report_lists_its_blocking_unknowns(self):
        report = fcm.build_report(fcm.RESULTS / "scene_counts.json", fcm.RESULTS)
        names = {u["name"] for u in report["blocking_unknowns"]}
        assert "credit_unit_price_usd" in names
        assert all(u["settled_by"].strip() for u in report["blocking_unknowns"])
