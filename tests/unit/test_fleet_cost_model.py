"""Guards on the fleet cost model's anchor, its buckets, and its ranges.

Three things are pinned here and each has already been got wrong once.

**The anchor's reconciliation.** The model is usable because figures from one
run agree. The measured fleet shape prices to $14.71 on-demand against $7.28
reported, a spot factor of 0.495 inside the 0.30-0.75 band, and it is 279.7
vCPU-hours against 268.1063 credits billed, a rate of 0.959. Editing the shape
without editing the anchors breaks that agreement silently and turns every
projection into arithmetic about nothing.

**The buckets.** The three Coiled cluster event log exports were recovered on
2026-09-02 and are retained by digest in ``cluster_records.json``, so the fleet
shape and the credit total are ``measured`` and read per worker rather than per
cluster. No AWS invoice or cost export exists in any commit, so the dollar
anchor stays ``user_reported`` and a test says so. Promoting it would read as an
invoice nobody can produce.

**The ranges.** Capture and the credit price are unmeasured, so the model emits
intervals and a formula. A scalar would price an unverified premise as settled.

Bands, never exact values, following the repository's benchmark discipline: a
test that fails on rounding gets disabled within a month.
"""

from __future__ import annotations

import importlib.util
import json
import re
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
        # pricing.json samples 0.35 / 0.44 / 0.71 on the same day. The measured
        # shape implies 0.495, which is the 0.44 sample within 0.1 and no
        # tighter: the numerator is still a spoken figure.
        assert abs(fcm.reference_totals()["implied_spot_factor"] - 0.44) < 0.1

    def test_the_implied_credit_rate_sits_inside_the_observed_band(self):
        lo, hi = fcm.CREDITS_PER_VCPU_HOUR_BAND
        assert lo <= fcm.reference_totals()["implied_credits_per_vcpu_hour"] <= hi

    def test_the_credit_rate_band_is_derived_from_the_three_clusters(self):
        # quota.py's 0.6-1.25 divided credits by vms x cluster lifetime, which
        # charges every worker the wall clock of the slowest. Per worker the
        # three clusters agree to about a tenth of a credit, which is why
        # CREDITS_PER_VCPU_HOUR = 1.0 is usable rather than a coin flip.
        lo, hi = fcm.CREDITS_PER_VCPU_HOUR_BAND
        assert hi - lo < 0.2
        assert lo > 0.6

    def test_one_credit_per_vcpu_hour_stays_conservative(self):
        # quota.py prices a run at 1.0 credits per vCPU-hour. It has to sit at
        # or above what the run billed, because a rate that understates lets an
        # unaffordable run start.
        t = fcm.reference_totals()
        assert t["vcpu_hours"] * 1.0 >= t["credits_billed"]
        assert t["vcpu_hours"] * 1.0 / t["credits_billed"] < 1.10


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

    def test_a_recovery_round_spends_the_most_of_itself_booting(self):
        # Five of offsets_round_2's fourteen workers lived 1.89 minutes and
        # never reached compute, so 41% of its billed VM-time is boot against
        # 24% for the composite round. A model that let a short round claim the
        # full compute budget would understate what retries cost.
        d = fcm.decompose_reference_run()
        by_stage = {c["cluster"]: c for c in d.per_cluster}
        share = {k: v["vm_min_boot"] / v["vm_min_total"] for k, v in by_stage.items()}
        assert share["offsets_round_2"] > share["composite_round_1"]
        assert share["offsets_round_2"] > share["offsets_round_1"]

    def test_billed_vm_time_is_read_per_worker_not_per_cluster(self):
        # The whole point of the recovered records. Charging every worker its
        # cluster's wall clock overstates the run by a third, and it overstates
        # the composite fleet most, which is the fleet carrying the cost.
        d = fcm.decompose_reference_run()
        for c in d.per_cluster:
            assert c["vm_min_total"] <= c["vm_min_if_all_lived_the_wall_clock"]
        comp = next(c for c in d.per_cluster if c["cluster"] == "composite_round_1")
        assert comp["vm_min_total"] < 0.7 * comp["vm_min_if_all_lived_the_wall_clock"]


class TestProvenance:
    """Five buckets, and the anchor is not in the strongest one."""

    def test_every_registered_quantity_carries_a_known_bucket(self):
        assert {q.bucket for q in fcm.REGISTER} <= set(fcm.BUCKETS)

    def test_the_aws_dollar_anchor_stays_user_reported(self):
        # The recovered records are Coiled event logs and Coiled billing
        # activity. Neither carries an AWS dollar figure, and no invoice or cost
        # export exists in any commit. Promoting this claims an artifact nobody
        # can produce.
        buckets = {q.name: q.bucket for q in fcm.REGISTER}
        assert buckets["reference_billed_aws_usd"] == "user_reported"

    def test_the_recovered_records_are_measured(self):
        buckets = {q.name: q.bucket for q in fcm.REGISTER}
        assert buckets["reference_fleet_shape"] == "measured"
        assert buckets["reference_billed_credits"] == "measured"
        assert buckets["reference_worker_lifetimes"] == "measured"

    def test_the_retained_artifact_carries_no_instance_address(self):
        # The exports the records come from carry private VPC addresses. They
        # are not in the repository and neither is any address they held.
        text = fcm.CLUSTER_RECORDS_FILE.read_text()
        assert not re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)

    def test_the_retained_artifact_carries_the_export_digests(self):
        # Provenance without publication: the CSVs stay private and the model
        # stays checkable against them.
        doc = json.loads(fcm.CLUSTER_RECORDS_FILE.read_text())
        for cluster in doc["clusters"]:
            src = cluster["source_export"]
            assert len(src["sha256"]) == 64
            assert src["bytes"] > 0
            assert src["retained_in_repo"] is False

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
        # 84% on the measured shape, down from 88% on the transcribed one
        # because the composite fleet is where the staggered exits were. Every
        # further saving after consolidation still has to come from that pass
        # rather than from scheduling.
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


class TestWaveTimings:
    """INV-35: what the driver's wave record would let the model measure.

    At f4d1e93 the record carries ``submitted_at`` and neither completion
    stamp, so the idle term is assumed. These tests pin what happens when the
    stamps arrive, and pin the limit: three stamps bound idle, and only
    per-unit durations measure it.
    """

    def _stamped(self, **over):
        record = {
            "stage": "offsets",
            "wave": 1,
            "units": 120,
            "max_workers": 60,
            "submitted_at": 1000.0,
            "first_completion_at": 1600.0,
            "last_completion_at": 2200.0,
        }
        record.update(over)
        return record

    def test_a_record_without_the_stamps_yields_nothing(self):
        # The f4d1e93 shape, which is what a run would write today.
        bare = {"stage": "offsets", "wave": 1, "units": 120, "max_workers": 60}
        bare["submitted_at"] = 1000.0
        assert fcm.wave_envelope(bare, boot_s=300.0) is None

    def test_billed_vm_time_comes_from_the_envelope(self):
        e = fcm.wave_envelope(self._stamped(), boot_s=300.0)
        # 60 workers x (2200 - 1000) seconds.
        assert e["billed_vm_s"] == pytest.approx(72_000.0)
        assert e["boot_vm_s"] == pytest.approx(18_000.0)
        assert e["drain_tail_s"] == pytest.approx(600.0)

    def test_queue_depth_is_reported_because_capture_turns_on_it(self):
        assert fcm.wave_envelope(self._stamped(), boot_s=300.0)["queue_depth"] == 2

    def test_idle_is_bounded_and_never_claimed_measured(self):
        # Three stamps cannot separate a worker running its next unit from a
        # worker waiting. Claiming a measured idle from them would be the
        # model's own version of the mistake it is auditing.
        e = fcm.wave_envelope(self._stamped(), boot_s=300.0)
        assert e["idle_vm_s"] is None
        assert e["idle_bucket"] == "assumed"
        assert e["idle_upper_bound_vm_s"] == pytest.approx(54_000.0)

    def test_a_malformed_envelope_is_refused_rather_than_guessed(self):
        assert fcm.wave_envelope(self._stamped(max_workers=0), boot_s=300.0) is None
        assert fcm.wave_envelope(self._stamped(last_completion_at=900.0), boot_s=300.0) is None

    def test_the_rollup_says_assumed_when_no_run_has_written_stamps(self):
        rollup = fcm.measured_idle([], boot_s=300.0)
        assert rollup["waves_with_timings"] == 0
        assert rollup["idle_bucket"] == "assumed"
        assert "INV-35" in rollup["note"]

    def test_the_rollup_consumes_stamped_records_when_they_exist(self):
        rollup = fcm.measured_idle([self._stamped(), self._stamped(wave=2)], boot_s=300.0)
        assert rollup["waves_with_timings"] == 2
        assert rollup["billed_vm_s"] == pytest.approx(144_000.0)
        assert rollup["idle_bucket"] == "assumed"

    def test_the_report_carries_the_idle_bucket(self):
        report = fcm.build_report(fcm.RESULTS / "scene_counts.json", fcm.RESULTS)
        assert report["measured_idle"]["idle_bucket"] == "assumed"
