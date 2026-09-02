# All-in fleet cost model (issue #108)

Anchored on the S30W065 acceptance run of 2026-08-23, reconciled against
`docs/adr/017-fleet-consolidation.md` and the driver at f4d1e93, which is the
design being integrated. Reproduce with:

```bash
python scripts/fleet_cost_model.py
python scripts/fleet_cost_model.py --json \
  > results/cost-model/cost_model_inputs.json
python scripts/fleet_cost_model.py --quantities
```

The inputs sit beside the script so it reproduces with no network and no
geopandas: `scene_counts.json` holds 700 scene counts from a STAC query, and
`land_fractions.json` holds the 700 floats the Natural Earth computation
reduces to. Committing derived floats rather than depending on a 10 MB download
is deliberate, because `results/probe/` is referenced throughout the code and
exists in no commit.

## 0. Blocking premise, read this before any number below

**`queues_surplus` is unverified, and every figure in this document rests on
it.** ADR-017's backend contract requires that `coiled.batch_run`, given more
units than workers, runs every unit on at most `max_workers` concurrent workers
and hands a worker the next unit when it finishes one. ADR-017 says in its own
words that this is read from documented Coiled semantics and has never been
verified against real Coiled by this project. The independent design audit
carries the same finding as GAP-Q.

If it is false, three things follow at once.

1. `max_workers` bounds nothing, so the cap invariants are unenforceable
   however correct the driver's arithmetic is. A 10,500-unit first wave starts
   10,500 VMs.
2. Capture collapses to zero, which is the `queues_surplus_false` row in
   section 5. Its cost equals doing no consolidation at all.
3. Every cost figure that assumes a bounded fleet is wrong, including the upper
   ends of the intervals below, which are computed against a 64-VM cap.

**No capture figure and no all-in estimate in this document may be quoted as
settled until the probe passes.** The probe is one wave, a handful of trivial
units, more units than workers, counting the workers that start and how the
units distribute. It costs about $0.50. It is not authorized and has not been
run.

## 0a. What this model describes, and when it goes stale

**Every figure here describes the current source-grid 30 m pipeline.** Draft
PR #121 delivers a nominal ~100 m grid, aggregating 3x3 source cells before the
percentile. Three things follow.

- The source read does not change. `budgets._native_bytes` and
  `projection.native_pass_gb` stay on the source shape on purpose, so the
  composite stage's byte count stands.
- Everything downstream of the read does change. The percentile's working set
  falls nine-fold, the composite graph grows from 828 to 2,094 tasks at 24
  scenes, and the uncompressed output falls exactly 9x.
- PR #121's own gate 4 leaves `R_COMPOSITE_MB_S` at 45.5, a decode rate
  measured before the change, and says its numbers must not be scaled by the
  pixel-count ratio. Gate 5 says the 100 m wall time, AWS cost, and Coiled
  credits are unmeasured.

The composite term carries 84% of this model's compute. A cost number carried
across #121 without recalibration is stale, and the model stamps
`pipeline_regime` into every output so a reader cannot miss which pipeline a
figure belongs to.

## 1. The five buckets

Every quantity carries exactly one. `quantities.json` holds one record per
quantity, with its value, unit, bucket, source, and what would move it to a
stronger bucket.

| Bucket | Meaning | Count |
|---|---|---|
| Bucket | Was | Now |
|---|---|---|
| `measured` | 3 | 8 |
| `derived` | 18 | 19 |
| `assumed` | 8 | 7 |
| `user_reported` | 6 | 4 |
| `unknown` | 3 | 3 |

**Half the billing anchor stopped being a transcription on 2026-09-02.** The
three Coiled cluster event log exports for the reference run were recovered and
are retained by SHA-256 digest in `cluster_records.json`. The exports carry
private VPC instance addresses, so they stay out of the repository and their
digests, byte counts, row counts, and retrieval date stand in for them. What the
records settle is the fleet shape, every worker's own billed window, the
scheduler instance beside each fleet, and the per-cluster credit split. Those
move to `measured`, along with three quantities the records make visible for the
first time: `reference_worker_lifetimes`, `reference_scheduler_vm_hours`, and
`vm_infrastructure_boot_minutes`.

**The AWS dollar anchor did not move.** The recovered records are Coiled event
logs and Coiled billing activity, and neither carries an AWS dollar figure. A
search of every commit in this repository still finds no invoice and no cost
export, so `$7.28` stays `user_reported` and `budgets.VM_BOOT_S` and ADR-016's
six-minute offsets shard stay there with it.

### What reading the shape per worker changed

The transcribed shape charged all 64 workers the wall clock of their cluster.
The event logs show workers exiting as they finish, so the fleet's billed time
is smaller and unevenly so.

| Basis | VM-hours | vCPU-hours | On-demand | Credits per vCPU-h |
|---|---|---|---|---|
| Transcribed shape | 24.55 | 317.73 | $16.38 | 0.844 |
| Workers charged the cluster wall clock | 30.99 | 394.76 | $20.39 | 0.679 |
| **Measured, per worker** | **22.82** | **279.66** | **$14.71** | **0.959** |

The composite fleet is where the divergence lives: 35 workers over 32.2 minutes
of wall clock billed 728 VM-minutes rather than 1,128, because its exits spread
from 13.9 to 31.6 minutes. The recovery round is the other extreme, with five of
its fourteen workers living 1.89 minutes.

The two implied factors part company as a result. `268.1063 / 279.66 = 0.959`
divides a measured numerator by a measured denominator, so it measures the
credit rate. `7.28 / 14.71 = 0.495` still divides a spoken figure by that
denominator, so it corroborates the AWS anchor against the 0.44 sample in
`pricing.json` and does not measure a spot rate.

### The credit rate band collapsed, and the total is why it could

The billed total 268.1063 reproduces what `quota.py` already carried, so the
total settles nothing on its own. The shape does. `quota.py` divided each
cluster's credits by `vms x cluster lifetime` and got 0.977, 0.605, and 0.617,
which is where the 0.6 to 1.25 band came from. Per worker the same three
clusters give **1.003, 0.948, and 0.902**, a spread of a tenth of a credit
rather than a factor of two. `quota.py`'s own note guessed the cause correctly
and could not act on it: the width was the staggered exits, not a varying rate.

`CREDITS_PER_VCPU_HOUR = 1.0` therefore holds, and holds better than it did.
It prices this run at 279.66 credits against 268.1063 billed, **4.3% high**
where the transcribed shape made it 18.5% high. High is the safe direction: a
rate that understates lets an unaffordable run start.

## 1a. Coverage of issue #108's cost gates

The independent design audit records INV-34, INV-35, and INV-36 as
`not-implemented` against the driver candidate f4d1e93. Two of the three are
this model's to answer, and the third is answered by a sibling document on this
branch.

| Invariant | #108 gate | Status here |
|---|---|---|
| INV-34 | a cost model separating compute, provisioning/idle, credits, retries, and storage, in ranges | met by `scripts/fleet_cost_model.py` |
| INV-35 | per-wave `submitted_at`, `first_completion_at`, `last_completion_at`, and a derived idle | **unmet**, and the model is written to consume it |
| INV-36 | a documented cached-iteration path submitting no production tile | met by `iteration_workflow.md` |

**INV-34.** Section 4 separates compute from provisioning and idle, section 5
reports both as intervals, credits are a separate currency reported as an
interval, and retries, contingency, and storage are named terms with their own
records in `quantities.json`.

**INV-35, and the reason the idle term is `assumed` rather than `measured`.**
`FleetDriver._record_wave` at f4d1e93 writes `run_id`, `stage`, `wave`,
`units`, `tiles`, `max_workers`, `handle_name`, `submitted_at`, `deadline_s`,
and `handle_id`. It writes neither completion stamp. Nothing in the run
therefore records when a wave's first unit landed or when its last one did, so
the idle term in section 2 is the reference run's residue rather than an
observation. `scripts/fleet_cost_model.py` already reads the record INV-35
describes, through `wave_envelope` and `measured_idle`, and
`--wave-records` feeds them a real run's records.

The stamps are necessary and they are not sufficient, which is worth stating
before anyone implements them and expects a measured number.

- They settle the **envelope**. A wave's workers start together and the last
  one stops when the wave's last unit lands, so billed VM-seconds is
  `workers x (last_completion_at - submitted_at)`. That is an observation, and
  it replaces a transcribed wall clock.
- They do not settle the **split**. Idle is billed time minus boot minus useful
  compute, and no stamp says how long a unit ran. Three stamps bound idle above
  and cannot separate a worker running its next unit from a worker waiting.

So the idle term becomes `measured` only when per-unit durations exist
alongside the wave stamps, which is the same observation `unresolved-inputs.md`
asks for under U4. Until a real run writes both, the idle term is `assumed`,
the model says so in its own output, and
`test_idle_is_bounded_and_never_claimed_measured` stops it drifting.

**INV-36.** `iteration_workflow.md` on this branch routes each question to its
cheapest artifact and marks the cloud rows. Every row it marks "Cloud? no"
reads a cached offset record, a fixture, band slabs, or a local graph, and none
reaches `submit_shard_stage`.

## 2. Where the reference run's money went

Boot comes from `budgets.VM_BOOT_S`. Offsets compute comes from ADR-016. The
composite's useful compute is assumed at the fastest VM's 20 minutes minus one
boot, treating the fleet's 20 to 32 minute spread as straggler and barrier time
rather than work. Idle is the residue, so the three terms sum to the reported
shape.

The split runs worker by worker, which is what the measured lifetimes buy: a
worker that lived 1.89 minutes is charged 1.89 minutes of boot and no compute,
where a fleet-wide wall clock would have charged it the full round.

| Term | VM-min | On-demand USD | Share of VM-time |
|---|---|---|---|
| Boot | 304 | $3.33 | 22% |
| Useful compute | 592 | $6.95 | **43%** |
| Idle, barrier, straggler | 473 | $4.43 | 35% |
| Total | 1,369 | $14.71 | 100% |

Only 43% of billed VM-time was useful compute. That number is what issue #108
exists to move, and a model with no term for the other 57% could not price the
thing being fixed. Every term fell against the transcribed shape and the split
between them barely moved, which is the useful result: the case for
consolidation rested on a proportion, and the proportion survived measurement.

| Cluster | Workers | Wall clock | Billed VM-min | If all lived the wall clock | Boot | Useful | Idle |
|---|---|---|---|---|---|---|---|
| 1954303 offsets round 1 | 15 | 35.2 min | 506.9 | 527.8 | 75.0 | 90.0 | 341.9 |
| 1954376 offsets round 2 | 14 | 14.5 min | 134.3 | 203.6 | 54.5 | 54.0 | 25.8 |
| 1954375 composite round 1 | 35 | 32.2 min | 728.1 | 1,128.0 | 175.0 | 448.1 | 105.0 |

Each cluster also ran a scheduler instance, 1.36 VM-hours across the three. The
exports give no instance type for it, so the model counts that VM-time and
declines to price it.

**The idle line is `assumed`, not measured, and the measurement sharpened why.**
It is what is left after boot and an assumed compute figure, so it absorbs the
error in both, and the denominator it is subtracted from is now exact. The
strain shows in offsets round 1, which books 342 of its 507 VM-minutes as idle
because ADR-016's six-minute per-unit figure is applied to VMs that lived 34
minutes processing many units each. That is a claim about the numerator, not
about the fleet. Section 1a says what would settle it and why the wave stamps
alone are not enough.

## 3. Scaling to 700 tiles

The two stages read different bytes, so they get different equivalents.

- **Offsets** reads the coarse stack twice, phase A discounted by land because
  blocks with no land are skipped, phase B at full footprint. It scales as
  `scenes x (land_fraction + 1)`.
- **Composite** reads the native stack once at full footprint whatever the land
  fraction. It scales as `scenes` alone.

| Quantity | Value | Bucket |
|---|---|---|
| Reference scenes (S30W065) | 4,138 | measured |
| Total scenes, 700 tiles | 2,566,229 | measured |
| Mean land fraction, NE 10m, 25 km buffer | 0.757 | derived |
| Offsets equivalents | 559.9 | derived |
| Composite equivalents | 620.2 | derived |

Weighting the whole build by `scenes x land_fraction` would apply an
offsets-only discount to the stage that dominates the bill.

## 4. Three layers, kept apart

| Layer | On-demand USD | What it is |
|---|---|---|
| Compute | 4,235 | Work the pipeline must do whatever schedules it |
| Provisioning and idle | 4,561 | Boot, barrier wait, stragglers |
| Approval | uplifts, storage, ceiling | Retry variance, contingency, the #108 constraint |

Compute splits 84% composite ($3,558) and 16% offsets ($677). Both layers fell
about 10% against the transcribed shape, and the composite share fell from 88%
to 84% because the composite fleet is the one whose exits were most staggered.

**Consolidation acts on the provisioning layer alone.** It amortizes
provisioning across many tiles. It does not make one full tile faster, and
ADR-017 says the same thing from the design side: a fleet run of one tile is
ADR-016 with extra indirection and no saving.

## 5. Results, as intervals

There is no all-in dollar scalar, and there cannot be one while the credit
price is unknown. Collapsing the formula to a number requires pricing the
unknown term at zero.

| Scenario | Capture | AWS term (spot 0.30-0.75) | Credits (0.902-1.003 per vCPU-h) |
|---|---|---|---|
| `queues_surplus_false` | 0.00 | **$3,817 - $9,507** | 218,207 - 242,641 |
| `conservative` | 0.50 - 0.80 | **$2,244 - $7,049** | 127,690 - 179,732 |
| `design_band` | 0.85 - 0.95 | **$1,641 - $4,479** | 93,003 - 113,985 |

All-in cost is `AWS + credits x P_credit`, and `P_credit` is unknown.

The measured records moved both columns, and moved them in opposite ways. The
AWS intervals fell about 11% at both ends, because the fleet billed less VM-time
than the transcribed shape claimed. The credit intervals narrowed from a factor
of 2.1 wide to a factor of 1.1, and their floors rose by about a third: the
`design_band` credit floor went from 70,373 to 93,003. Better evidence is not
the same as a cheaper answer, and here it raised the credit floor while lowering
the AWS one.

No scenario demonstrates that the build fits the $3,000 approval ceiling. Only
`design_band` puts the AWS term below it at the bottom of the spot band, the
other two straddle it or clear it, and every row carries a credit term that
cannot be converted to dollars while `credit_unit_price_usd` is unknown.

Each interval spans both unmeasured inputs at once: capture at its band's ends,
and the spot fraction at its band's ends. The spot band 0.30 to 0.75 is valid
because `settings.shard_spot_policy` is `"spot"`. Under `spot_with_fallback`
the upper bound becomes 1.00.

### 5a. Capture, and the premise underneath it

Capture is the share of provisioning that consolidation removes. ADR-017 gives
it as `1 - 1/R` for queue depth `R = ceil(units / max_workers)`. At 700 tiles
the first offsets wave buffers about 10,500 units against a 64-VM cap, so
`R` is about 164 and boot capture is `1 - 64/10,500 = 0.994`. Idle capture
rises with it, because a worker takes the next queued unit instead of waiting
out a barrier.

Capture is not 0.994 all the same. A tail wave, stragglers in a final round, a
composite stage competing for the same cap, and tiles that do not become ready
together all survive. The `design_band` row reads 0.85 to 0.95 for that reason.

**The premise under all of it is unverified.** Section 0 states it in full.
`queues_surplus` has never been checked against real Coiled by this project, no
number anywhere in this repository measures it, and if it is false capture
collapses to zero.

Read `queues_surplus_false` as the outcome if the premise fails, not as a
pessimistic scenario. Its cost equals no consolidation at all, and by then the
driver, the deadlines, and the capacity model have all been changed.

### 5b. The arithmetic, term by term

The `conservative` scenario's high end, in order, each factor named. Rates are
`pricing.json` on-demand list, us-west-2.

| Step | Operation | Result (USD, on-demand) |
|---|---|---|
| Compute | offsets useful + composite useful, at their equivalents | 4,235 |
| Provisioning | boot + idle, at their equivalents | 4,561 |
| Consolidation | provisioning x (1 - 0.50 capture) | 2,281 |
| Subtotal | 4,235 + 2,281 | 6,516 |
| Excess retries | x 1.15 | 7,493 |
| Contingency | x 1.25 | 9,366 |
| Spot, high end of band | x 0.75 | 7,025 |
| Storage | + 1.05 TB x $0.023/GB-month | **7,049** |

The low end of the same row takes capture 0.80 and spot 0.30. Credits follow
the same VM-time through the 0.902 to 1.003 measured rate band, and are never
multiplied by a dollar price.

The retry uplift is deliberately small. The reference run already contains a
recovery round, `offsets_round_2`, and a spot reclaim, so the base rate of
retrying sits inside the anchor. An uplift on top would count it twice, and
what is left to cover is variance beyond the one tile that was reported.

## 6. The $3,000 figure

Issue #108 sets $3,000 as an approval ceiling and says in the same sentence
that it is not a target to spend up to. This model does not treat it as one,
and it does not report a verdict against it.

Two reasons, and the first is sufficient on its own.

1. **The all-in cost carries an unpriced term.** A model cannot clear a ceiling
   on a total it has not computed. Every scenario above reports where its AWS
   interval sits relative to $3,000 and then says plainly that no scenario
   demonstrates the build fits.
2. **The intervals straddle it.** `conservative` runs $2,505 to $7,892 on the
   AWS term alone, so the AWS term's own uncertainty is wider than the distance
   to the ceiling.

An earlier draft of this document contained a break-even table reading the
ceiling backwards, asking what credit price would let each capture level clear
$3,000, and concluded that the build "does not fit". That is the ceiling used
as a target with the sign flipped. It has been removed, and
`contradictions.md` records where it was.

## 7. What consolidation buys

- **The saving is bounded by the provisioning layer, which is 52% of the
  on-demand total.** Compute is the other 48% and is the floor no scheduling
  reaches. Comparing at identical uplifts, capture 0.50 takes the AWS term down
  26%, and capture 0.85 to 0.95 takes it down 44% to 49%.
- **After consolidation the composite native pass dominates.** 84% of compute
  is that pass. Every further dollar has to come from the pass rather than from
  scheduling, which is what makes #121's aggregation the next cost question and
  not a separate one.
- **Consolidation does not make a single tile faster.** It amortizes
  provisioning across tiles. A one-tile fleet run saves nothing.

## 8. The largest modelling risk: items against solar-day time steps

The composite term dominates everything after consolidation, and its scaling
basis is unresolved.

| Basis for S30W065's composite | Predicted VM-h at 45.5 MB/s |
|---|---|
| 4,138 items (`scene_counts.json`) | 32.7 |
| 4,403 items (frozen plan) | 34.8 |
| 1,031 solar-day time steps | **8.2** |
| Reported (35 VMs x 15 min) | **8.8** |

The observation matches the solar-day basis within 7% and is 3.7x off the item
basis. Two pieces of evidence settle which is right, and neither needs a cloud
run. `budgets._scene_count` returns `len(plan.scene_times)` under the docstring
"Time steps, not items: that axis is what every phase iterates", and every
stage deadline in the sharded driver is computed on that basis. `odc-stac`
groups before it stacks, so `groupby="solar_day"` gives the dask array one time
entry per group.

**This model scales on item counts anyway, and does so knowingly.**
`scene_counts.json` counts items, and no per-tile solar-day count exists for
the other 699 tiles. It is defensible because the model scales by a *ratio*
against the same reference tile, so a constant items-per-group factor cancels
exactly. What does not cancel is variation in that factor, and it does vary:
WRS-2 path overlap grows toward the poles, so high-latitude tiles pack more
items into each group. S30W065 sits at 4.01 items per solar day. The residual
bias overstates high-overlap tiles, which is the safe direction, and the model
cannot bound it.

Producing a per-tile solar-day count is a STAC metadata job rather than a
compute job, and it would remove the largest uncertainty in the term carrying
84% of compute.

Worth flagging alongside it: `projection.tile_projection(scenes=2930)` is
documented against N40W075's 2,912 *items*, while `budgets._scene_count` uses
*time steps*, and the same `R_COMPOSITE_MB_S` was calibrated through both.

## 9. The credit unknowns

Two, and they are separate.

**The dollar price of a credit is unknown.** Nothing in this repository records
a marginal credit price for this account. At $0.01 per credit the
`conservative` row adds $968 to $2,557. At $0.05 it adds $4,838 to $12,787,
which is more than the AWS bill. The all-in figure is unbounded above until the
price is known, so establishing it is worth more than any engineering change in
this issue.

**The usable quota is unknown.** `settings.coiled_credit_quota` is 400,
transcribed from the kill message of 2026-08-22. That is the quota that was hit
on one day. It is not a statement of what this workspace can be granted, and
`quota.py` says the period boundary is not observable either. The build needs
credits in the tens or hundreds of thousands on every scenario above, so the
gap is large whatever the exact quota is. Calling it "250x short" treats a
transcribed kill message as a measured allowance, and this model does not.

Both are release blockers on their own terms. A workspace at its quota does not
degrade. It gets a healthy fleet killed mid-stage and its cluster creates
rejected with an empty `ServerError`, which is what happened on 2026-08-22 and
why `quota.preflight_credits` exists.

## 10. Storage and transfer

700 tiles at an assumed 1.5 GB compressed is about 1.05 TB, and S3 Standard at
$0.023/GB-month is about $24 a month. Band slabs and intermediates are
same-region S3 to EC2, so transfer is free, and request charges are cents.
Egress is zero unless the product is downloaded out of region, which is a
publication decision rather than a build cost. #121 cuts the uncompressed
output 9x, so this line goes stale with the rest, and at $24 a month it does
not change any decision.

## 11. What must be recorded, and was not

The acceptance run's measurements survive only as prose in docstrings, a config
description, an ADR, `CLAUDE.md`, and one test's class constants.
`results/probe/` is referenced throughout the code and exists in no commit,
current or historical. `$7.28` appears nowhere in this repository.

Issue #108's cloud gate requires a calibration run that reconciles measured
wall time, VM lifetimes, AWS cost, and Coiled credits against this model. That
reconciliation is impossible against numbers nobody kept and a credit price
nobody has asked for. Without them the next cost review starts where this one
did, reconstructing an invoice from a test fixture.

Three observations would settle most of it, and `unresolved-inputs.md` states
each one exactly.
