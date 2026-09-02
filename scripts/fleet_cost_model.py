"""All-in fleet cost model for issue #108, in ranges, with every input labelled.

Anchored on the S30W065 acceptance run of 2026-08-23 and reconciled against
``docs/adr/017-fleet-consolidation.md`` as the driver that is actually being
integrated. Usage::

    python scripts/fleet_cost_model.py                # table to stdout
    python scripts/fleet_cost_model.py --json         # pure JSON
    python scripts/fleet_cost_model.py --quantities   # the provenance register

The model separates three layers and never lets them blur, because they have
different evidence behind them and respond to different decisions:

1. **Compute.** Work the pipeline must do whatever schedules it. Scheduling
   cannot remove it, so it is the floor every scenario shares.
2. **Provisioning and idle.** Boot, barrier wait, and stragglers. This is the
   only layer fleet consolidation touches, and how much of it consolidation
   removes is called *capture*.
3. **Approval.** Retry variance, contingency, storage, and the comparison
   against the $3,000 figure in issue #108. That figure is an approval ceiling.
   It is not a target, and a model output is not a finding about it.

Five provenance buckets, and a number never carries two:

``measured``
    A retained artifact in this repository *is* the observation. Vendor list
    prices in ``pricing.json``, the 700 STAC scene counts.
``derived``
    Arithmetic over other quantities. The arithmetic is shown.
``assumed``
    A modelling choice. Carries a value, a range, and a sensitivity.
``user_reported``
    A person transcribed it from an external system and no export was kept.
    Every billing anchor in this model is in this bucket, including the fleet
    shape, because the Coiled billing activity it came from was not retained.
``unknown``
    Cannot be settled from this repository. Never given a value, never
    defaulted to zero, and never multiplied into a headline.

**Every figure here describes the current source-grid 30 m pipeline.** Draft
PR #121 aggregates to a nominal ~100 m delivered grid before the percentile,
which changes the term carrying most of the cost. Its own gate 5 says the
100 m wall time, AWS cost, and credits are unmeasured. A number from this model
carried across that change is stale, and the model says so in its output rather
than leaving a reader to remember.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "cost-model"

#: Which pipeline every number below describes. Stamped into every output.
PIPELINE_REGIME = "source-grid 30 m, pre-#121"
REGIME_NOTE = (
    "Draft PR #121 delivers a nominal ~100 m grid, aggregating 3x3 source cells "
    "before the percentile. The source read is deliberately unchanged, so the "
    "composite stage's bytes stand, but its working set, its task count, and its "
    "output size all move, and PR #121 gate 4 leaves R_COMPOSITE_MB_S unrevised. "
    "Recalibrate against a 100 m run before quoting any figure here after #121 lands."
)

# ---------------------------------------------------------------------------
# The provenance register. Every quantity the model uses is declared once here
# and the code reads its value from the register, so a figure cannot appear in
# an output without a bucket attached to it.
# ---------------------------------------------------------------------------

BUCKET_MEANINGS = {
    "measured": "A retained artifact in this repository is the observation itself.",
    "derived": "Arithmetic over other quantities, never stronger than its weakest input.",
    "assumed": "A modelling choice, carrying a value, a range, and a sensitivity.",
    "user_reported": "Transcribed from an external system with no export retained.",
    "unknown": "Cannot be settled from this repository. Never given a value.",
}

BUCKETS = tuple(BUCKET_MEANINGS)


@dataclass(frozen=True)
class Quantity:
    """One input, its bucket, where it came from, and what would strengthen it."""

    name: str
    value: Any
    unit: str
    bucket: str
    source: str
    upgrade: str

    def __post_init__(self) -> None:
        if self.bucket not in BUCKETS:
            raise ValueError(f"{self.name}: unknown bucket {self.bucket!r}")


REGISTER: list[Quantity] = []


def q(name: str, value: Any, unit: str, *, bucket: str, source: str, upgrade: str) -> Any:
    """Declare a quantity, register it, and return its value."""
    REGISTER.append(Quantity(name, value, unit, bucket, source, upgrade))
    return value


# --- The reference run ------------------------------------------------------

#: (name, vms, vcpus_each, wall_minutes, on_demand_usd_hour)
#:
#: USER-REPORTED, not measured, and the demotion is deliberate. This shape came
#: from Coiled billing activity that nobody exported. It survives as prose in
#: ``quota.py`` and as class constants in
#: ``tests/unit/test_driver_state_machine.py``, both of which retain the
#: transcription rather than the observation. A search of every commit in this
#: repository finds no invoice, no cost export, and no billing artifact.
FLEET_SHAPE: list[tuple[str, float, int, float, float]] = q(
    "reference_fleet_shape",
    [
        ("offsets_round_1", 15.0, 8, 31.0, 0.504),  # r6i.2xlarge
        ("offsets_round_2", 14.0, 8, 7.0, 0.504),  # r6i.2xlarge, a recovery round
        ("composite_round_1", 35.0, 16, 26.0, 0.768),  # m6i.4xlarge
    ],
    "(vms, vcpus, minutes, usd/h) per cluster",
    bucket="user_reported",
    source="S30W065 2026-08-23, transcribed into quota.py and test_driver_state_machine.py; "
    "no billing export retained",
    upgrade="Attach the Coiled billing activity export for that run under results/.",
)

BILLED_AWS_USD = q(
    "reference_billed_aws_usd",
    7.28,
    "USD",
    bucket="user_reported",
    source="$2.19 + $4.55 + $0.54 spot, spoken; appears in no commit of this repository",
    upgrade="Attach the AWS Cost Explorer export or invoice line for 2026-08-23.",
)

BILLED_CREDITS = q(
    "reference_billed_credits",
    268.11,
    "Coiled credits",
    bucket="user_reported",
    source="quota.py docstring, per cluster 67.81 / 16.15 / 184.15; no billing export retained",
    upgrade="Attach the Coiled billing activity export for that run under results/.",
)

BOOT_MIN = q(
    "vm_boot_minutes",
    5.0,
    "minutes",
    bucket="user_reported",
    source="budgets.VM_BOOT_S = 300; '4-5 min observed' across the S30W065 runs of 2026-08-21/22",
    upgrade="Retain the driver logs those boots were read from, or time a boot in the probe run.",
)

OFFSETS_COMPUTE_MIN = q(
    "offsets_unit_compute_minutes",
    6.0,
    "minutes",
    bucket="user_reported",
    source="ADR-016: offsets-side shards computed for about six minutes each",
    upgrade="Publish per-unit start and finish stamps from a shard's own state objects.",
)

COMPOSITE_OBSERVED_MIN = q(
    "composite_fastest_vm_minutes",
    20.0,
    "minutes",
    bucket="user_reported",
    source="fastest composite VM finished at 20 min; the fleet's spread was 20-32 min",
    upgrade="Publish per-unit timings; the spread to 32 is unattributed between work and wait.",
)

COMPOSITE_COMPUTE_MIN = q(
    "composite_unit_compute_minutes",
    COMPOSITE_OBSERVED_MIN - BOOT_MIN,
    "minutes",
    bucket="assumed",
    source="the fastest VM's wall clock minus one boot, treating the 20-32 spread as "
    "straggler and barrier time rather than work",
    upgrade="Per-unit timings would make this derived. If half the spread is work, the "
    "composite term rises about 40% and it already carries 88% of compute.",
)

# --- Prices and bands -------------------------------------------------------

SPOT_BAND = q(
    "spot_fraction_band",
    (0.30, 0.75),
    "fraction of on-demand",
    bucket="assumed",
    source="pricing.json spot_band, provenance 'assumed'; samples 0.35 / 0.44 / 0.71 on 2026-08-14. "
    "Valid because settings.shard_spot_policy is 'spot'; under spot_with_fallback the "
    "upper bound is 1.00",
    upgrade="Not reducible. Spot is priced per type, per AZ, per minute, and a finished run "
    "cannot be repriced against what it paid.",
)

IMPLIED_SPOT_NOTE = (
    "The 0.445 this run implies is derived from two user-reported figures and is "
    "reported as a point inside the band, never as the band's replacement."
)

CREDITS_PER_VCPU_HOUR_BAND = q(
    "credits_per_vcpu_hour_band",
    (0.6, 1.25),
    "credits per vCPU-hour",
    bucket="assumed",
    source="quota.py observed band across the run's three clusters; the spread is staggered "
    "VM lifetimes rather than a different rate",
    upgrade="A billing export with per-VM lifetimes would make this derived and narrow.",
)

S3_USD_PER_GB_MONTH = q(
    "s3_standard_usd_per_gb_month",
    0.023,
    "USD/GB-month",
    bucket="measured",
    source="AWS S3 Standard list price, us-west-2",
    upgrade="Nothing. It is a published list price.",
)

OUTPUT_GB_PER_TILE = q(
    "output_gb_per_tile",
    1.5,
    "GB compressed",
    bucket="assumed",
    source="compressed COG pair at the 30 m grid; no shipped tile was measured",
    upgrade="Measure a shipped tile. #121 cuts the uncompressed output 9x, so this is one "
    "of the inputs that goes stale first. Immaterial either way at ~$24/month.",
)

# --- Consolidation ----------------------------------------------------------

#: Capture is the fraction of provisioning and idle that consolidation removes,
#: reported as an interval per scenario because no scenario's capture has been
#: observed. ``queues_surplus_false`` is the case where the premise fails.
CAPTURE_BANDS: dict[str, tuple[float, float]] = q(
    "consolidation_capture_bands",
    {
        "queues_surplus_false": (0.0, 0.0),
        "conservative": (0.50, 0.80),
        "design_band": (0.85, 0.95),
    },
    "fraction of provisioning + idle removed",
    bucket="assumed",
    source="ADR-017: capture = 1 - 1/R for queue depth R. At 700 tiles the first offsets "
    "wave buffers about 10,500 units against a 64-VM cap, R about 164, boot capture "
    "0.994. Idle capture is lower because a tail wave, a final round's stragglers, "
    "a competing composite stage, and tiles not becoming ready together all remain.",
    upgrade="The ~$0.50 queues_surplus probe in ADR-017, then a capped calibration run that "
    "counts boots and idle VM-minutes.",
)

QUEUES_SURPLUS = q(
    "queues_surplus_holds",
    None,
    "boolean",
    bucket="unknown",
    source="ADR-017 backend contract: read from documented Coiled semantics and never "
    "verified against real Coiled by this project",
    upgrade="The ~$0.50 one-wave probe: more units than workers, count the workers that "
    "start and how units distribute. Until it passes, every capture figure is an "
    "assumption resting on an unverified premise.",
)

P_CREDIT = q(
    "credit_unit_price_usd",
    None,
    "USD per Coiled credit",
    bucket="unknown",
    source="no marginal credit price is recorded for this account anywhere in the repository",
    upgrade="One question to the account owner, or one invoice line.",
)

USABLE_CREDIT_QUOTA = q(
    "usable_credit_quota",
    None,
    "Coiled credits per period",
    bucket="unknown",
    source="settings.coiled_credit_quota is 400, transcribed from the kill message of "
    "2026-08-22. That is the quota that was hit on one day, not a statement of what "
    "this workspace can be granted, and the period boundary is not observable either",
    upgrade="Read the workspace usage endpoint, or ask the account owner for the quota and "
    "its reset period.",
)

RETRY_UPLIFT = q(
    "excess_retry_uplift",
    {"queues_surplus_false": 0.15, "conservative": 0.15, "design_band": 0.05},
    "fraction",
    bucket="assumed",
    source="deliberately small: the reference run already contains a recovery round "
    "(offsets_round_2) and a spot reclaim, so the base rate of retrying is inside "
    "the anchor and an uplift on top of it would count it twice",
    upgrade="Count recovery rounds across a multi-tile calibration run.",
)

CONTINGENCY = q(
    "contingency",
    {"queues_surplus_false": 0.25, "conservative": 0.25, "design_band": 0.15},
    "fraction",
    bucket="assumed",
    source="unmodelled all-in slack: driver hours, S3 request charges, a stage re-run "
    "after a bug, price drift since pricing.json as_of 2026-08-14",
    upgrade="Judgement. Keep it, and keep it named.",
)

APPROVAL_CEILING_USD = q(
    "approval_ceiling_usd",
    3000.0,
    "USD",
    bucket="measured",
    source="issue #108 acceptance text; an approval ceiling set by the project owner",
    upgrade="Nothing. It is a stated constraint, not an estimate, and it is neither a "
    "target nor a prediction.",
)

# --- Scaling ----------------------------------------------------------------

SCALING_BASIS = q(
    "composite_scaling_basis",
    "items",
    "basis",
    bucket="assumed",
    source="scene_counts.json counts STAC items; budgets._scene_count uses solar-day time "
    "steps, and the reference run's observed composite matches the solar-day basis "
    "within 7% against 3.7x off the item basis. The model scales by a ratio against "
    "the same reference, so a constant items-per-group factor cancels exactly",
    upgrade="A per-tile solar-day count for all 700 tiles. It is a STAC metadata job, not a "
    "compute job, and it removes the largest uncertainty in the term carrying 88% "
    "of compute. What does not cancel is the variation: WRS-2 path overlap grows "
    "toward the poles, so item-scaling overstates high-latitude tiles. The residual "
    "bias is conservative and the model cannot bound it.",
)

LAND_FRACTION_CACHE = "land_fractions.json"


@dataclass(frozen=True)
class Interval:
    """A low and a high. The model reports these wherever an input is not measured."""

    low: float
    high: float

    def scaled(self, factor: float) -> Interval:
        return Interval(self.low * factor, self.high * factor)

    def as_dict(self, ndigits: int = 0) -> dict[str, float]:
        return {"low": round(self.low, ndigits), "high": round(self.high, ndigits)}


@dataclass
class Decomposition:
    """Billed VM-minutes split into what was useful and what was not."""

    boot_vm_min: float = 0.0
    useful_vm_min: float = 0.0
    idle_vm_min: float = 0.0
    boot_usd_od: float = 0.0
    useful_usd_od: float = 0.0
    idle_usd_od: float = 0.0
    per_cluster: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_vm_min(self) -> float:
        return self.boot_vm_min + self.useful_vm_min + self.idle_vm_min

    @property
    def total_usd_od(self) -> float:
        return self.boot_usd_od + self.useful_usd_od + self.idle_usd_od


def _unit_compute_min(cluster: str, wall_min: float) -> float:
    """Compute minutes for one VM in a cluster, never longer than it lived."""
    each = COMPOSITE_COMPUTE_MIN if cluster.startswith("composite") else OFFSETS_COMPUTE_MIN
    return min(each, max(0.0, wall_min - BOOT_MIN))


def decompose_reference_run() -> Decomposition:
    """Split the reference run into boot / useful compute / idle, per cluster.

    Idle is the residue, deliberately: it is everything the fleet was billed for
    that was neither booting nor computing, which is queueing, barrier wait, and
    stragglers. Naming it as a residue rather than modelling it keeps the three
    terms summing to what was reported.
    """
    d = Decomposition()
    for name, vms, _cpus, wall_min, usd_h in FLEET_SHAPE:
        boot = vms * min(BOOT_MIN, wall_min)
        useful = vms * _unit_compute_min(name, wall_min)
        idle = max(0.0, vms * wall_min - boot - useful)
        rate = usd_h / 60.0
        d.boot_vm_min += boot
        d.useful_vm_min += useful
        d.idle_vm_min += idle
        d.boot_usd_od += boot * rate
        d.useful_usd_od += useful * rate
        d.idle_usd_od += idle * rate
        d.per_cluster.append(
            {
                "cluster": name,
                "vms": vms,
                "wall_min": wall_min,
                "vm_min_total": vms * wall_min,
                "vm_min_boot": round(boot, 1),
                "vm_min_useful": round(useful, 1),
                "vm_min_idle": round(idle, 1),
                "usd_on_demand": round(vms * wall_min * rate, 2),
            }
        )
    return d


def reference_totals() -> dict[str, Any]:
    """VM-hours, vCPU-hours, on-demand dollars, and the two implied factors.

    The two implied factors are derived from user-reported numerators, so they
    corroborate the anchor rather than measure it. Both land on values recorded
    independently: 0.445 against the 0.44 sample in ``pricing.json``, and 0.844
    inside the 0.6-1.25 band in ``quota.py``. Two mistyped figures would have to
    land there by accident, which is why the anchor is usable. It is still not
    a measurement.
    """
    vm_h = sum(n * m / 60.0 for _, n, _, m, _ in FLEET_SHAPE)
    vcpu_h = sum(n * c * m / 60.0 for _, n, c, m, _ in FLEET_SHAPE)
    usd_od = sum(n * m / 60.0 * p for _, n, _, m, p in FLEET_SHAPE)
    return {
        "vm_hours": round(vm_h, 2),
        "vcpu_hours": round(vcpu_h, 2),
        "usd_on_demand": round(usd_od, 2),
        "usd_billed_spot": BILLED_AWS_USD,
        "implied_spot_factor": round(BILLED_AWS_USD / usd_od, 3),
        "credits_billed": BILLED_CREDITS,
        "implied_credits_per_vcpu_hour": round(BILLED_CREDITS / vcpu_h, 3),
        "corroboration": IMPLIED_SPOT_NOTE,
    }


# ---------------------------------------------------------------------------
# Scaling one reference tile to the 700-tile build.
# ---------------------------------------------------------------------------


def load_scene_counts(path: Path) -> dict[str, int]:
    counts = json.loads(path.read_text())
    return {k: int(v) for k, v in counts.items() if int(v) > 0}


def land_fractions(tiles: list[str], cache_dir: Path) -> dict[str, float]:
    """Land fraction per tile from Natural Earth 10m, buffered 25 km.

    Prefers the committed cache and falls back to computing from the polygons,
    writing the cache when it does. Returns an empty mapping when neither is
    available, and the caller then reports the scenes-only weighting rather than
    inventing one.
    """
    cache = cache_dir / LAND_FRACTION_CACHE
    if cache.exists():
        stored = json.loads(cache.read_text())
        if all(t in stored for t in tiles):
            return {t: float(stored[t]) for t in tiles}
    try:
        import geopandas as gpd  # noqa: PLC0415 -- optional dependency
        from shapely.geometry import box  # noqa: PLC0415
        from shapely.ops import unary_union  # noqa: PLC0415
    except ImportError:
        return {}
    cached = cache_dir / "ne_10m_land_buf25km.gpkg"
    if not cached.exists():
        return {}
    land = gpd.read_file(cached)
    sindex = land.sindex
    out: dict[str, float] = {}
    for name in tiles:
        lat = int(name[1:3]) * (1 if name[0] == "N" else -1)
        lon = int(name[4:7]) * (1 if name[3] == "E" else -1)
        cell = box(lon, lat, lon + 5, lat + 5)
        cand = land.iloc[list(sindex.query(cell))]
        if len(cand) == 0:
            out[name] = 0.0
            continue
        inter = unary_union([g.intersection(cell) for g in cand.geometry])
        out[name] = min(1.0, inter.area / cell.area)
    cache.write_text(json.dumps({k: round(v, 6) for k, v in sorted(out.items())}, indent=1) + "\n")
    return out


def tile_equivalents(counts: dict[str, int], lf: dict[str, float], ref: str) -> dict[str, Any]:
    """Reference-tile equivalents, per stage, by the byte model each stage obeys.

    The offsets stage reads the coarse stack twice, once discounted by land
    (phase A skips blocks with no land) and once at full footprint (phase B), so
    it scales as ``scenes * (land_fraction + 1)``. The composite stage reads the
    native stack once at full footprint whatever the land fraction, so it scales
    as ``scenes`` alone. Weighting the whole build by ``scenes * land_fraction``
    applies an offsets-only discount to the stage that dominates the bill.
    """
    ref_scenes = counts[ref]
    ref_lf = lf.get(ref, 1.0)
    off_ref = ref_scenes * (ref_lf + 1.0)
    off = sum(c * (lf.get(t, 1.0) + 1.0) for t, c in counts.items()) / off_ref
    comp = sum(counts.values()) / ref_scenes
    return {
        "offsets_equivalents": round(off, 1),
        "composite_equivalents": round(comp, 1),
        "reference_scenes": ref_scenes,
        "reference_land_fraction": round(ref_lf, 3),
        "total_scenes": sum(counts.values()),
        "tiles": len(counts),
        "scaling_basis": SCALING_BASIS,
    }


def stage_shares() -> dict[str, dict[str, float]]:
    """Reference on-demand dollars per stage per term, the projection's basis."""
    shares = {
        "offsets": {"boot": 0.0, "useful": 0.0, "idle": 0.0},
        "composite": {"boot": 0.0, "useful": 0.0, "idle": 0.0},
    }
    for name, vms, _c, wall, usd_h in FLEET_SHAPE:
        rate = usd_h / 60.0
        boot = vms * min(BOOT_MIN, wall) * rate
        useful = vms * _unit_compute_min(name, wall) * rate
        idle = max(0.0, vms * wall * rate - boot - useful)
        tgt = shares["composite" if name.startswith("composite") else "offsets"]
        tgt["boot"] += boot
        tgt["useful"] += useful
        tgt["idle"] += idle
    return shares


def layers(equiv: dict[str, Any]) -> dict[str, Any]:
    """The two physical layers, before any approval-layer uplift.

    Compute is what the pipeline must do whatever schedules it. Provisioning is
    the only layer consolidation touches. Reporting them apart is what stops a
    scheduling change from being read as a science saving, or the reverse.
    """
    s = stage_shares()
    e_off = equiv["offsets_equivalents"]
    e_comp = equiv["composite_equivalents"]
    compute_off = s["offsets"]["useful"] * e_off
    compute_comp = s["composite"]["useful"] * e_comp
    prov_off = (s["offsets"]["boot"] + s["offsets"]["idle"]) * e_off
    prov_comp = (s["composite"]["boot"] + s["composite"]["idle"]) * e_comp
    compute = compute_off + compute_comp
    return {
        "compute_usd_on_demand": round(compute, 0),
        "compute_offsets_usd": round(compute_off, 0),
        "compute_composite_usd": round(compute_comp, 0),
        "compute_composite_share": round(compute_comp / compute, 3),
        "provisioning_usd_on_demand": round(prov_off + prov_comp, 0),
        "provisioning_offsets_usd": round(prov_off, 0),
        "provisioning_composite_usd": round(prov_comp, 0),
        "note": (
            "Consolidation acts on the provisioning line alone. It amortizes "
            "provisioning across tiles and does not make one full tile faster."
        ),
    }


def reference_vcpu_hours() -> tuple[float, float]:
    """``(offsets, composite)`` vCPU-hours in the reference run."""
    off = sum(n * c * m / 60.0 for nm, n, c, m, _ in FLEET_SHAPE if not nm.startswith("composite"))
    comp = sum(n * c * m / 60.0 for nm, n, c, m, _ in FLEET_SHAPE if nm.startswith("composite"))
    return off, comp


def project(equiv: dict[str, Any], *, scenario: str) -> dict[str, Any]:
    """One scenario's AWS interval and credit interval. No all-in scalar.

    The AWS interval spans both uncertain inputs at once: capture at its band's
    ends, and the spot fraction at the band's ends. Reporting a point would
    price two unmeasured quantities as if they were settled.
    """
    cap_lo, cap_hi = CAPTURE_BANDS[scenario]
    retry = RETRY_UPLIFT[scenario]
    cont = CONTINGENCY[scenario]
    uplift = (1.0 + retry) * (1.0 + cont)

    lay = layers(equiv)
    compute = float(lay["compute_usd_on_demand"])
    provisioning = float(lay["provisioning_usd_on_demand"])

    # Highest cost takes the least capture; lowest takes the most.
    od_high = (compute + provisioning * (1.0 - cap_lo)) * uplift
    od_low = (compute + provisioning * (1.0 - cap_hi)) * uplift

    spot_lo, spot_hi = SPOT_BAND
    storage = OUTPUT_GB_PER_TILE * equiv["tiles"] * S3_USD_PER_GB_MONTH
    aws = Interval(od_low * spot_lo + storage, od_high * spot_hi + storage)

    # Credits track vCPU-hours, so the same capture applies to the same share.
    off_vcpu, comp_vcpu = reference_vcpu_hours()
    scaled_vcpu_h = (
        off_vcpu * equiv["offsets_equivalents"] + comp_vcpu * equiv["composite_equivalents"]
    )
    overhead_frac = provisioning / (provisioning + compute)
    rate_lo, rate_hi = CREDITS_PER_VCPU_HOUR_BAND
    credits = Interval(
        scaled_vcpu_h * (1.0 - overhead_frac * cap_hi) * uplift * rate_lo,
        scaled_vcpu_h * (1.0 - overhead_frac * cap_lo) * uplift * rate_hi,
    )

    return {
        "scenario": scenario,
        "capture_band": {"low": cap_lo, "high": cap_hi},
        "retry_uplift": retry,
        "contingency": cont,
        "aws_usd_on_demand_basis": Interval(od_low, od_high).as_dict(),
        "aws_usd_spot": aws.as_dict(),
        "storage_usd_month": round(storage, 0),
        "coiled_credits": credits.as_dict(),
        "coiled_usd": None,  # unknown: no credit price is manufactured
        "all_in_usd_formula": (
            f"${aws.low:,.0f}-${aws.high:,.0f} AWS"
            f" + {credits.low:,.0f}-{credits.high:,.0f} credits x P_credit"
        ),
        "vs_approval_ceiling": ceiling_verdict(aws),
    }


def ceiling_verdict(aws: Interval) -> str:
    """Where the AWS interval sits relative to #108's ceiling, with no verdict.

    The comparison is reported because the ceiling is a stated constraint. It is
    never reported as a pass, because the all-in cost carries an unpriced credit
    term and a model output cannot clear a ceiling on a term it has not priced.
    """
    ceiling = APPROVAL_CEILING_USD
    if aws.low > ceiling:
        where = "the AWS term alone lies above the ceiling across the whole interval"
    elif aws.high < ceiling:
        where = "the AWS term lies below the ceiling across the whole interval"
    else:
        where = "the AWS interval straddles the ceiling"
    return (
        f"{where}. The all-in figure adds an unpriced credit term, so no scenario "
        f"here demonstrates that the build fits ${ceiling:,.0f}, and none is offered "
        f"as one. The ceiling is an approval constraint, not a target."
    )


def build_report(scene_counts_path: Path, cache_dir: Path, ref: str = "S30W065") -> dict[str, Any]:
    counts = load_scene_counts(scene_counts_path)
    cached = (cache_dir / LAND_FRACTION_CACHE).exists()
    lf = land_fractions(sorted(counts), cache_dir)
    if not lf:
        source = "unavailable -- offsets weighted at land_fraction=1.0"
    elif cached:
        source = f"{LAND_FRACTION_CACHE} (committed; derived from NE 10m, 25 km buffer)"
    else:
        source = "ne_10m_land_buf25km.gpkg (computed; cache written)"
    equiv = tile_equivalents(counts, lf, ref)
    equiv["land_fraction_source"] = source
    if lf:
        equiv["mean_land_fraction"] = round(sum(lf.values()) / len(lf), 3)
    dec = decompose_reference_run()
    return {
        "pipeline_regime": PIPELINE_REGIME,
        "regime_note": REGIME_NOTE,
        "reference_run": {
            "tile": ref,
            "date": "2026-08-23",
            "anchor_bucket": "user_reported",
            "totals": reference_totals(),
            "decomposition": {
                "boot_vm_min": round(dec.boot_vm_min, 0),
                "useful_vm_min": round(dec.useful_vm_min, 0),
                "idle_vm_min": round(dec.idle_vm_min, 0),
                "useful_fraction_of_vm_time": round(dec.useful_vm_min / dec.total_vm_min, 3),
                "boot_usd_on_demand": round(dec.boot_usd_od, 2),
                "useful_usd_on_demand": round(dec.useful_usd_od, 2),
                "idle_usd_on_demand": round(dec.idle_usd_od, 2),
                "per_cluster": dec.per_cluster,
            },
        },
        "fleet_scale": equiv,
        "layers": layers(equiv),
        "projection": [project(equiv, scenario=s) for s in CAPTURE_BANDS],
        "blocking_unknowns": [
            {
                "name": qty.name,
                "source": qty.source,
                "settled_by": qty.upgrade,
            }
            for qty in REGISTER
            if qty.bucket == "unknown"
        ],
    }


DERIVED_UPGRADE = (
    "Nothing on its own. It inherits the bucket of its weakest input, so it "
    "strengthens when that input does."
)


def derived_quantities(report: dict[str, Any]) -> list[Quantity]:
    """Quantities the model computes, each with its arithmetic in the source.

    Registered separately from the inputs because they are outputs, and because
    every one of them inherits a user-reported or assumed input. A derived
    figure is never stronger than what it was derived from, and reading one as
    a measurement is the mistake this register exists to stop.
    """
    t = report["reference_run"]["totals"]
    d = report["reference_run"]["decomposition"]
    f = report["fleet_scale"]
    lay = report["layers"]
    total_vm_min = d["boot_vm_min"] + d["useful_vm_min"] + d["idle_vm_min"]
    out = [
        Quantity(
            "reference_vm_hours",
            t["vm_hours"],
            "VM-hours",
            "derived",
            "sum(vms x minutes) / 60 over the fleet shape: (15x31 + 14x7 + 35x26) / 60. "
            "Inherits user_reported.",
            DERIVED_UPGRADE,
        ),
        Quantity(
            "reference_vcpu_hours",
            t["vcpu_hours"],
            "vCPU-hours",
            "derived",
            "sum(vms x vcpus x minutes) / 60: (15x8x31 + 14x8x7 + 35x16x26) / 60. "
            "Inherits user_reported.",
            DERIVED_UPGRADE,
        ),
        Quantity(
            "reference_usd_on_demand",
            t["usd_on_demand"],
            "USD",
            "derived",
            "VM-hours per cluster x the pricing.json list rate. Inherits measured "
            "prices and a user_reported shape.",
            DERIVED_UPGRADE,
        ),
        Quantity(
            "implied_spot_factor",
            t["implied_spot_factor"],
            "fraction of on-demand",
            "derived",
            f"{BILLED_AWS_USD} / {t['usd_on_demand']}. Both inputs user_reported, so "
            "this corroborates the anchor against pricing.json's 0.44 sample rather "
            "than measuring a spot rate.",
            DERIVED_UPGRADE,
        ),
        Quantity(
            "implied_credits_per_vcpu_hour",
            t["implied_credits_per_vcpu_hour"],
            "credits per vCPU-hour",
            "derived",
            f"{BILLED_CREDITS} / {t['vcpu_hours']}. Inherits user_reported, and sits "
            "inside quota.py's 0.6-1.25 band.",
            DERIVED_UPGRADE,
        ),
        Quantity(
            "reference_useful_fraction_of_vm_time",
            d["useful_fraction_of_vm_time"],
            "fraction",
            "derived",
            f"{d['useful_vm_min']:.0f} useful VM-min / {total_vm_min:.0f} total. Idle "
            "is the residue after boot and compute, so it carries the error in both "
            "compute assumptions.",
            DERIVED_UPGRADE,
        ),
        Quantity(
            "total_scenes_700_tiles",
            f["total_scenes"],
            "STAC items",
            "measured",
            "sum over results/cost-model/scene_counts.json, a retained artifact of a "
            "STAC query. Items, not solar-day groups: see composite_scaling_basis.",
            "A per-tile solar-day count would replace the basis assumption.",
        ),
        Quantity(
            "mean_land_fraction",
            f.get("mean_land_fraction"),
            "fraction",
            "derived",
            "mean over results/cost-model/land_fractions.json, computed from Natural "
            "Earth 10m buffered 25 km against each 5-degree cell.",
            DERIVED_UPGRADE,
        ),
        Quantity(
            "offsets_equivalents",
            f["offsets_equivalents"],
            "reference tiles",
            "derived",
            "sum(scenes x (land_fraction + 1)) / reference, because phase A skips "
            "blocks with no land and phase B reads the full footprint.",
            DERIVED_UPGRADE,
        ),
        Quantity(
            "composite_equivalents",
            f["composite_equivalents"],
            "reference tiles",
            "derived",
            "sum(scenes) / reference scenes. No land discount: the native stack is "
            "read at full footprint whatever the land fraction.",
            DERIVED_UPGRADE,
        ),
        Quantity(
            "compute_usd_on_demand",
            lay["compute_usd_on_demand"],
            "USD on-demand",
            "derived",
            "per-stage useful dollars x that stage's equivalents. Work the pipeline "
            "must do whatever schedules it, so consolidation cannot touch it.",
            DERIVED_UPGRADE,
        ),
        Quantity(
            "provisioning_usd_on_demand",
            lay["provisioning_usd_on_demand"],
            "USD on-demand",
            "derived",
            "per-stage (boot + idle) dollars x that stage's equivalents. The only "
            "layer consolidation acts on.",
            DERIVED_UPGRADE,
        ),
        Quantity(
            "compute_composite_share",
            lay["compute_composite_share"],
            "fraction",
            "derived",
            "composite compute / total compute. After consolidation every further "
            "dollar has to come from this pass rather than from scheduling.",
            DERIVED_UPGRADE,
        ),
    ]
    for proj in report["projection"]:
        band = proj["capture_band"]
        out.append(
            Quantity(
                f"aws_usd_spot_{proj['scenario']}",
                proj["aws_usd_spot"],
                "USD interval",
                "derived",
                "(compute + provisioning x (1 - capture)) x (1 + retry) x "
                f"(1 + contingency) x spot, at capture {band['low']}-{band['high']} "
                f"and spot {SPOT_BAND[0]}-{SPOT_BAND[1]}, plus storage. Inherits three "
                "assumed inputs and a user_reported anchor.",
                DERIVED_UPGRADE,
            )
        )
        out.append(
            Quantity(
                f"coiled_credits_{proj['scenario']}",
                proj["coiled_credits"],
                "credit interval",
                "derived",
                "scaled vCPU-hours x (1 - overhead share x capture) x uplift x the "
                "0.6-1.25 credit rate band. Never multiplied by a dollar price.",
                DERIVED_UPGRADE,
            )
        )
    return out


def quantities(report: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The full register: declared inputs first, then what the model derives."""
    seen: dict[str, dict[str, Any]] = {}
    entries = list(REGISTER)
    if report is not None:
        entries += derived_quantities(report)
    for qty in entries:
        record = asdict(qty)
        if isinstance(record["value"], tuple):
            record["value"] = list(record["value"])
        seen[qty.name] = record
    return list(seen.values())


def main() -> None:
    ap = argparse.ArgumentParser(description="All-in fleet cost model, in ranges.")
    ap.add_argument("--json", action="store_true", help="emit pure JSON")
    ap.add_argument("--quantities", action="store_true", help="emit the provenance register")
    ap.add_argument("--scene-counts", type=Path, default=RESULTS / "scene_counts.json")
    ap.add_argument("--cache-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    report = build_report(args.scene_counts, args.cache_dir)

    if args.quantities:
        print(
            json.dumps(
                {
                    "pipeline_regime": PIPELINE_REGIME,
                    "regime_note": REGIME_NOTE,
                    "buckets": BUCKET_MEANINGS,
                    "quantities": quantities(report),
                },
                indent=2,
            )
        )
        return
    if args.json:
        print(json.dumps(report, indent=2))
        return

    r = report["reference_run"]
    t = r["totals"]
    print(f"Pipeline regime: {report['pipeline_regime']}")
    print(f"Reference run: {r['tile']} {r['date']} -- anchor is USER-REPORTED")
    print(f"  {t['vm_hours']} VM-h, {t['vcpu_hours']} vCPU-h, ${t['usd_on_demand']} on-demand list")
    print(f"  reported ${t['usd_billed_spot']} spot -> spot factor {t['implied_spot_factor']}")
    print(
        f"  reported {t['credits_billed']} credits -> {t['implied_credits_per_vcpu_hour']}/vCPU-h"
    )

    d = r["decomposition"]
    print(
        f"\nWhere the money went (on-demand basis, ${d['boot_usd_on_demand'] + d['useful_usd_on_demand'] + d['idle_usd_on_demand']:.2f}):"
    )
    print(f"  boot   ${d['boot_usd_on_demand']:>6.2f}   {d['boot_vm_min']:>6.0f} VM-min")
    print(
        f"  useful ${d['useful_usd_on_demand']:>6.2f}   {d['useful_vm_min']:>6.0f} VM-min   <- {d['useful_fraction_of_vm_time']:.0%} of billed VM-time"
    )
    print(f"  idle   ${d['idle_usd_on_demand']:>6.2f}   {d['idle_vm_min']:>6.0f} VM-min")

    f = report["fleet_scale"]
    lay = report["layers"]
    print(f"\nFleet scale ({f['tiles']} tiles, {f['total_scenes']:,} scenes) -- MEASURED counts")
    print(f"  offsets equivalents   {f['offsets_equivalents']}")
    print(f"  composite equivalents {f['composite_equivalents']}")
    print("\nLayers (on-demand, before uplift):")
    print(
        f"  compute       ${lay['compute_usd_on_demand']:>8,.0f}   composite is {lay['compute_composite_share']:.0%} of it"
    )
    print(
        f"  provisioning  ${lay['provisioning_usd_on_demand']:>8,.0f}   the only layer consolidation touches"
    )

    print("\nAll-in cost is a formula over intervals, never a scalar:")
    for p in report["projection"]:
        cap = p["capture_band"]
        print(f"  {p['scenario']:<22} capture {cap['low']:.2f}-{cap['high']:.2f}")
        print(f"    {p['all_in_usd_formula']}")

    print("\nUNKNOWN and blocking:")
    for u in report["blocking_unknowns"]:
        print(f"  - {u['name']}")
    print(f"\n{report['regime_note']}")


if __name__ == "__main__":
    main()
