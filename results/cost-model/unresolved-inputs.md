# Unresolved inputs

Three are `unknown` and block a conclusion. Four more are `assumed` and each
carries enough weight to move the answer. For every one, the observation that
would settle it.

---

## Blocking

### U1. Does `queues_surplus` hold

**Status:** unknown. Registered as `queues_surplus_holds`, value `null`.

**Why it blocks.** The entire economic case for issue #108 rests on it. ADR-017
requires that `coiled.batch_run`, given more units than workers, runs every
unit on at most `max_workers` concurrent workers and hands a worker the next
unit when it finishes one. ADR-017 says in its own words that this is read from
documented Coiled semantics and has never been verified against real Coiled by
this project. If it is false, the surplus starts its own VMs and breaks the
cost cap, or the submission is refused, or the surplus is dropped. Capture then
collapses to zero and the `queues_surplus_false` row is the answer.

**What would settle it.** ADR-017 names the probe. One wave, a handful of
trivial units, more units than workers. Count the workers that start and record
how the units distribute across them. Cost is about $0.50. It is not
authorized and has not been run.

**What it moves.** Everything. The AWS interval spans $1,828 to $10,667 across
the capture range, so this one boolean is worth more than a factor of five on
the term that has a price at all.

### U2. The dollar price of a Coiled credit

**Status:** unknown. Registered as `credit_unit_price_usd`, value `null`.

**Why it blocks.** All-in cost is `AWS + credits x P_credit`. Nothing in this
repository records a marginal credit price for this account. Collapsing the
formula to a scalar prices the unknown term at zero, and the term is large: at
$0.01 per credit the `conservative` scenario adds $968 to $2,557, and at $0.05
it adds $4,838 to $12,787, which exceeds the AWS bill. The all-in figure is
unbounded above until the price is known.

**What would settle it.** One question to the account owner, or one invoice
line showing dollars against credits for a known period.

### U3. The usable credit quota, and its period

**Status:** unknown. Registered as `usable_credit_quota`, value `null`.

**Why it blocks.** `settings.coiled_credit_quota` is 400, transcribed from the
kill message of 2026-08-22. That is the quota that was hit on one day rather
than a statement of what the workspace can be granted, and `quota.py` records
that nothing observable says when the quota period resets. Every scenario needs
credits in the tens or hundreds of thousands, so the gap is large whatever the
true figure. A workspace at its quota does not degrade: it gets a healthy fleet
killed mid-stage and its cluster creates rejected with an empty `ServerError`.

**What would settle it.** Read the workspace usage endpoint
`/api/v2/user/account/{workspace}/usage`, which `quota.py` already calls, and
record both `has_quota` and any remaining figure with the date. Ask the account
owner for the quota and its reset period.

---

## Assumed, and heavy enough to matter

### U4. The composite unit's useful compute minutes

**Status:** assumed at 15 minutes, from the fastest VM's 20 minus one boot.

**Why it matters.** The composite carries 88% of compute. The fleet's spread
was 20 to 32 minutes and the model attributes all of it to straggler and
barrier time rather than to work. If half the spread is work, the composite
term rises about 40% and the whole build's compute floor rises with it.

**What would settle it.** Publish per-unit start and finish stamps from each
shard's own state objects. The driver already writes state per attempt, so this
is retention rather than new instrumentation.

### U5. Items against solar-day time steps

**Status:** assumed. The model scales the composite on item counts.

**Why it matters.** The reference run's observed composite matches a solar-day
basis within 7% and is 3.7x off the item basis, and `budgets._scene_count`
already uses time steps. The model survives it because it scales by a ratio
against the same reference, so a constant items-per-group factor cancels
exactly. What does not cancel is variation: WRS-2 path overlap grows toward the
poles, so item-scaling overstates high-latitude tiles. The bias is in the safe
direction and the model cannot bound it.

**What would settle it.** A per-tile solar-day group count for all 700 tiles.
It is a STAC metadata job rather than a compute job, it needs no cloud run, and
it removes the largest uncertainty in the term carrying 88% of compute.

### U6. The capture band itself

**Status:** assumed, 0.85 to 0.95 in `design_band`, conditional on U1.

**Why it matters.** It is the only lever consolidation pulls, acting on 52% of
the on-demand total.

**What would settle it.** After U1 passes, a capped multi-tile calibration run
that counts boots and idle VM-minutes directly, which is exactly the run issue
#108's cloud gate describes.

### U6a. Per-wave provisioning and idle (INV-35)

**Status:** assumed. The idle term is the reference run's residue.

**Why it matters.** Provisioning and idle is 52% of the on-demand total and is
the only layer consolidation acts on. It is the quantity issue #108 exists to
reduce, and no run has observed it. `FleetDriver._record_wave` at f4d1e93
writes `submitted_at` and neither completion stamp, so #108's cloud gate,
reconciling measured wall time and VM lifetimes against the model, cannot be
met as the driver is built.

**What would settle it, in two parts.** The wave record needs
`first_completion_at` and `last_completion_at` beside `submitted_at`, which
makes billed VM-time per wave an observation. That alone bounds idle rather
than measuring it, because no stamp says how long a unit ran. Per-unit
durations, which is U4, are the second half. Both together move the idle term
from assumed to derived over measured inputs.

The model already consumes the first half through `wave_envelope`,
`measured_idle`, and `--wave-records`, so a run that writes the stamps needs no
change here.

### U7. Compressed output size per tile

**Status:** assumed at 1.5 GB.

**Why it matters.** Least of the seven. It drives about $24 a month of storage
and changes no decision. It is listed because #121 cuts the uncompressed output
9x, so it is one of the first inputs to go stale, and because a reader should
not have to guess whether 1.5 GB was measured. It was not.

**What would settle it.** Measure a shipped tile.

---

## The staleness boundary

Every figure in this model describes the source-grid 30 m pipeline. Draft
PR #121 delivers a nominal ~100 m grid and aggregates before the percentile.
The source read is deliberately unchanged, so the composite's bytes stand, but
its working set, its task count, and its output size all move. PR #121's gate 4
leaves `R_COMPOSITE_MB_S` at 45.5 and states that its numbers must not be
scaled by the pixel-count ratio. Its gate 5 states that the 100 m wall time,
AWS cost, and Coiled credits are unmeasured and that #108 is not updated by it.

So the recalibration is a fourth blocking item on any post-#121 number, and it
cannot be done by arithmetic on the numbers in this model.
