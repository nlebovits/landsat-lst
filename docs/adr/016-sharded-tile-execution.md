# ADR-016: Sharded tile execution across many VMs

**Status:** Accepted
**Date:** 2026-08-21
**Authors:** @nlebovits

## Context

A tile does not fit in an hour on one VM, and it never will.

`landsat_lst.projection.tile_projection`, built on the rates the 2026-08-21 I/O ladder
measured (158.4 MB/s coarse at chunk 1024, 70–85 MB/s native at chunk 512), prices a
2,930-scene five-degree tile at roughly 950 GB read twice for the offsets and a further
3.8 TB for the native composite. On one VM that is hours, and the number is a projection
from a *measured* per-VM rate rather than a hope: the way to get under an hour is more
VMs, not a faster one. Buying a bigger instance moves an I/O-bound read by the ratio of
its network share, which is not the factor needed.

Two constraints shape everything below.

**There is no intra-tile checkpoint** (ADR-015 says so explicitly). A failure at minute
170 costs all 170. Sharding is the checkpoint that was never built: a lost shard costs
one shard.

**`coiled.batch_run` has no dependency mechanism.** It starts an array of tasks. It
cannot express "start these when those finish", and ADR-014 already established that
nothing it reports about a task is worth acting on — the exit code recorded is the tee
wrapper's, task stdout never reaches `coiled logs`, and a task can exit non-zero after
its artifact landed.

## Decision

One tile is cut into stages, each stage into shards, and the stages are sequenced by a
**local driver that polls S3 between them**. A shard is complete when its artifact is
listed. Never an exit code, never a state object.

```
resolve (1)  ->  climatology (N)  ->  offsets (M)  ->  [merge, in the driver]
             ->  composite (B)    ->  export (1)
```

### Row bands only, never column bands

`odc-stac` groups scenes by `solar_day`, and the solar-time shift it applies comes from
the geobox **centroid longitude**. Two column bands of one tile have different centroid
longitudes, so they can group the same items onto *different time axes* — and the
offsets, estimated once for the tile, would then no longer line up with the scenes they
were estimated for. Cutting along rows leaves the centroid longitude fixed.
`landsat_lst.shards` offers no column split and nothing should add one.

### S3 barriers, not a Coiled DAG

The driver submits a stage, then lists the tile's shard prefix every
`shard_driver_poll_s` until every expected key is present. On `shard_barrier_timeout_s`
it resubmits **only the indexes that are still missing**, as a fresh small array, and it
does that at most `shard_barrier_rounds` times in total before failing the tile and
naming the keys that never appeared.

Resubmitting an index that is merely slow is safe because every shard's output key is a
pure function of its index, and every shard checks its own output first and exits if it
is already there. That idempotency is load-bearing, not a nicety: the driver cannot
distinguish a slow shard from a dead one and must never need to.

### The driver and its shards must share one namespace

The barrier's premise is that both halves read and write the same place. The first
acceptance run did not. `landsat-lst shard process --tile S30W065` ran with the default
`storage_backend=local`, so the driver listed a directory on the laptop while the VMs —
which inherit `LST_STORAGE_BACKEND=s3` from `_worker_environ` — published to the bucket.
`plan.json` was on S3 within 3.5 minutes and the resolve barrier never closed.

`require_shared_storage` now refuses, before anything is submitted, when the driver is
bound to Coiled and the backend is not S3. It does not quietly switch: a caller who asked
for local storage and got their COGs in a bucket would be a second surprise on top of the
first. A caller that injects its own submitter is driving something local deliberately,
and both halves then share `LocalStorage`; that case is allowed.

A barrier that cannot see its artifacts fails as a *hang*, which is the most expensive
shape a failure can take — nothing is wrong until the deadline, and the deadline is hours.

### A stage already in flight is adopted, never restarted

Artifacts alone cannot distinguish "still booting" from "nobody has started this": a shard
publishes nothing until it finishes. A resumed driver arriving during the first driver's
boot therefore saw zero climatology artifacts, concluded the stage had not started, and
resubmitted — which Coiled refused outright:

```
RuntimeError: Unable to add batch jobs to existing cluster
'lst-shard-S30W065-2021-2025-20260821T194111Z-S30W065-climato'
```

Two things are wrong in that name at once. It carries no round marker, and it was
truncated mid-stage, so appending one would have been eaten. `stage_cluster_name` hashes
the run id to eight characters instead — it already contains the tile and window, both of
which appear in the name anyway — and puts the round last in a name far short of the
limit.

A unique name alone would only turn a refusal into duplicated work, so the driver also
publishes a **submission record** at `_shards/{run_id}/{tile}/state/{stage}.submission.{round}.json`
*before* it submits. A record younger than `shard_barrier_timeout_s` means somebody's
cluster is in flight: watch it, do not submit. Past that deadline the cluster is gone, and
the next round starts covering only the indexes that never landed.

The record is written before the submission rather than after on purpose. A driver that
dies in between leaves a record for a cluster that never ran, costing the next driver one
barrier timeout; the other order leaves a live cluster nothing mentions, which is exactly
the collision. One wasted wait beats one refused submission and a stage of duplicated
reads.

The round budget is counted across drivers, not per driver — otherwise each resume would
hand the stage a fresh budget and the cap would mean nothing.

### Artifact-is-completion, and a driver with no state

Nothing the driver holds survives its own process, because nothing needs to. Position in
the stage sequence is derived from one listing: `landsat-lst shard resume <run-id>
<tile>` reconstructs it and continues. This is the same discipline `submit_batch` follows
for a fleet run — "the submitting shell is disposable" — pushed one level down. The
difference is that here the shell must stay open *while* the tile runs, since it is the
thing sequencing the stages; what it must not be is the thing remembering them.

A driver on Coiled would remove even that requirement. It needs a token that outlives a
multi-hour run, which is a separate piece of work.

### The shard prefix is outside `_runs/`

Everything a tile's shards publish lives under `_shards/{run_id}/{tile}/`.
`runs.classify` treats every key under `_runs/{run_id}/` as a tile attempt, and seven
shards of one tile share one tile name: filed there, they would appear in a manifest as
seven attempts of `N40W075` and `watch` would subtract them from its pending count. The
same reasoning already produced `capture_task_log(key=...)` for the synthetic sweep;
`TileHeartbeat` gains the matching override here.

### The merge runs in the driver

The offset partials total a few hundred kilobytes and merge into ~600 floats. A VM would
spend longer booting than working. What it writes is the **ordinary ADR-012 cache record
at the canonical `_offsets/` key** — not a shard artifact in a shard-shaped place. That
is the seam: every composite shard reads it back exactly as a single-VM tile would, and
because only the estimate is ever cached (ADR-012), the rejection is applied tile-wide
and identically by all of them. A later whole-tile run over the same scenes finds the
record and skips its own offset pass.

### Fleet sizing comes from measured rates

`shards.stage_shard_counts` takes its widths from `projection.tile_projection` when the
`shard_*_vms` settings are 0: the phase's projected VM-hours divided by its share of the
sixty-minute budget. Every width is clamped to the work available, because a shard with
no block is a VM that boots, reads a plan, finds nothing, and bills a minute. This is
ADR-011's rule — price a configuration before you run it — applied to the fleet itself.

The composite stage is the one that overrides its VM type and chunk. A whole-tile
composite stops at `load_chunk_size=512` because the single-time-chunk rechunk holds
`chunk² × scenes × 4 B` (3.1 GB at 512 over 2,930 scenes, an infeasible 12.3 GB at 1024).
A row band holds a fraction of the rows, so the same per-task working set buys the larger
request the probe measured as *the* throughput lever. Because the plan digest covers
`load_chunk_size`, `shard_tasks.apply_shard_settings` applies the override in every shard
process **and in the planner**, so all of them hash the same number; setting it only in
the composite shard would make that shard refuse a plan its own planner had cut.

## Consolidation: one fleet per side, not one per phase

**Amendment, 2026-08-22.** The first working acceptance run measured the thing the design
had assumed away. Offsets-side shards **computed for about six minutes each while their
stages held fleets for about thirty**. The work was fine; the lifecycle was the cost. Every
stage boundary was being paid for twice — once as the driver's poll, and again as a whole
fleet's boot and queue wait — and the offsets side had three of them.

Nothing about the estimator, the work-unit bodies, the artifact keys, or any equivalence
guarantee changes below. This is lifecycles only.

### The offsets side is one task type

`resolve`, `climatology`, and `offsets` became sub-phases of one task
(`shard_tasks.run_offsets_stage`) rather than three fleets:

- **Shard 0 resolves** — and only when no plan exists yet, so a retry, and every shard of a
  resumed run, reaches its own work without needing the window. Exactly one process may
  resolve; two would query a live catalog twice and assemble the tile from two scene sets.
- **Every shard waits for that plan**, bounded by `shard_plan_wait_s`.
- **Every shard reduces its climatology blocks**, then waits at an **in-process phase-A
  barrier** for its peers'. Phase B measures each scene against the *whole* climatology, so
  this barrier is not negotiable — but the process that needs it is already booted, and a
  second fleet existed only to re-establish that fact.
- **Every shard estimates its scenes' offsets.**

The sub-phase bodies are reused verbatim. `run_climatology_shard` and `run_offsets_shard`
gained an optional `ctx=` so the fused task loads the plan once, and nothing else moved. Each
sub-phase still checks its own outputs first, so a retried fused task walks back to where it
died: the resolve is a plan read, the blocks it published are skipped, and it resumes at the
barrier.

One consequence worth stating plainly. The fused fleet's **width has to be fixed before the
plan exists**, because shard 0 of that fleet is what writes the plan. `shards.offsets_fleet_units`
decides it, and it travels to the planner as `--units` so the plan is cut to the fleet that
will run it. Re-deriving it on the VM would let the two disagree, and the symptom would be a
stage waiting forever for a partial nobody was asked to write. A shard whose index falls past
a clamped count skips that phase and says so.

### The composite fleet boots on the offsets stage's time

The driver starts it from **inside** the offsets barrier, as soon as phase B is demonstrably
producing (`shard_composite_overlap`, default: the first partial). Evidence, not a timer —
a partial means the stage is running and producing, which is what separates overlapping from
gambling a fleet's boot on a stage that may be about to fail.

That start goes through `ensure_started`, so the ordinary composite barrier afterwards sees a
fresh submission record and **adopts** it. The same machinery that stops two drivers colliding
turns out to be exactly what an early start needs.

A composite shard that arrives before the merge now **waits** for the offset record
(`shard_offsets_record_wait_s`) instead of refusing. Refusing would burn the boot the overlap
exists to save.

### The export is claimed, not submitted

The export is one task at the end of a wide stage. After writing its bands, each composite
shard checks whether every band now exists; the one that finds them all writes
`state/export.claim.json` and runs the export itself, already booted and warm.

**The claim is not a lock, and first-writer-wins is not needed.** The export is idempotent at
the canonical COG keys, so two workers racing produce the same two objects: a lost race costs
duplicated work, never a corrupted tile. That is why it is a plain write with no
compare-and-set — S3 offers none, and synthesizing one out of listings would add a failure
mode to save a few minutes of one VM. The claim makes duplication rare; it does not make it
impossible, and it does not need to.

The driver keeps a belt: if the COGs are still absent `shard_export_claim_fallback_s` after
every band exists, it submits the old export stage. That covers the claim that is written and
never executed, because the claiming VM was preempted in between.

### What did not change

Barrier and round semantics, the round budget counted across drivers, the submission records,
the cluster naming, `shard_spot_policy`, every S3 key, and every equivalence test. Tile
completion is still both COGs at the canonical keys.

Observability was extended rather than trimmed: the fused task reports `shard_resolve`,
`shard_plan_wait`, `shard_barrier_wait`, and the estimator's own phases separately, and the
composite reports `shard_offsets_wait`. Time spent waiting at an in-process barrier is now the
majority of what a consolidated run can waste, so it has to be attributable — otherwise the
next round of tuning is guesswork again.

## The driver is a state machine, and its deadlines are budgets

**Amendment, 2026-08-23.** A night's fleet died at the Coiled workspace credit quota
(`Scheduler Stopped -> Instance Stopped: You have reached the workspace quota of 400
Coiled credits`), and the same gate had earlier produced an empty `ServerError` on a
cluster create. Neither was a driver bug. The driver's part was worse: it *masked* both,
and the post-mortem found a real defect underneath.

### Every deadline is derived, none is typed

`shard_barrier_timeout_s` was a hand-entered 7200 for every stage — a guess that ages
badly and that nobody recomputes when the window, the fleet width, or a measured rate
moves. `landsat_lst.budgets` replaces it with the model `projection.py` already uses for
cost: bytes over measured rates, per *shard*, plus the named fixed costs a shard pays
before it reads anything (`VM_BOOT_S`, `RESOLVE_S`), each carrying its provenance.

A stage's deadline is its budget times `shard_budget_safety` — one named factor, not one
per stage, so widening it is a conversation rather than a silent edit. The setting stays
as an explicit override defaulting to `None` (= derived), for the case where somebody is
holding a stopwatch during an incident.

Two consequences worth stating. The *widest* shard sets each stage's budget, because a
barrier waits for the slowest one; the shares come from the same `balance_by_land` split
the shards use, since an even division understates the widest group on a coastal tile.
And the composite budget includes an `offsets_tail` phase, because that fleet is started
from inside the offsets barrier and spends real time polling for a record the driver has
not written yet.

### Every round gets its own deadline

The observed defect. A barrier's deadline was measured from the first submission, so
round 2 opened at T+46min against a deadline that had expired at T+45, watched for
nothing, and failed instantly. `StageMachine` computes a fresh deadline **when a round
opens** — `clock.now() + deadline_s` on submit, `record.submitted_at + deadline_s` on
adopt. Pinned by a test that makes a never-landing shard cost *two* budgets of wall
clock; an inherited deadline spends one and quits.

### Errors are classified, not swallowed

A control-plane failure is `terminal` or `transient`. Terminal — quota, credits, billing,
auth — stops the tile now and surfaces the reason: an exhausted quota will not clear
inside a backoff, and burning the remaining rounds against it turns a two-line
explanation into a 45-minute silence. Everything else, **including an error with no
message at all**, is transient and retried with backoff up to `shard_submit_retries`,
then reported. Guessing "terminal" for the unknown case would reintroduce the failure
where an empty `ServerError` killed the driver outright.

At barrier level the driver also probes the round's cluster (`coiled.list_clusters`).
A cluster reported `error` or `stopped` while artifacts are missing raises
`ShardFleetKilled` with the reason attached, rather than waiting out the barrier — a
killed fleet produces no artifacts and never will. The probe can only end a barrier
*sooner*; it never declares success, and a dead report is re-checked against the bucket
first, because a fleet whose last task uploaded and then stopped is a finished stage.

### Nothing submits before the quota is checked

`quota.preflight_credits` is state zero. Three sources, best first: the workspace usage
endpoint `coiled login` itself reads (`has_quota` is authoritative for exhausted-or-not);
`coiled_credit_quota` minus the billing-activity debits in a recent window; and, if
neither answers, a refusal that prints the team page and demands `--ack-quota`. The run's
credit estimate comes from the same budget model, so geometry moves it.

The fallback's approximation is documented rather than hidden: nothing observable says
when the quota period resets, so "the period" is the last `coiled_credit_period_days`.
Too short under-counts spend and lets an unaffordable run start; too long over-counts and
refuses an affordable one. `CREDITS_PER_VM_HOUR` (2.7823) is calibrated from the observed
event amounts on the assumption that one event is roughly one VM-hour, which the event
stream does not state. **A preflight pass means "not obviously unaffordable", never a
promise.**

### The credit model, calibrated against an invoice

**Amendment, 2026-08-23.** The first run that could check the estimate billed **268.11
credits** against an estimate of 75. The model was per *VM*-hour; Coiled bills per
*vCPU*-hour, so it could not see that a 16-vCPU composite VM costs twice an 8-vCPU offsets
VM for the same wall clock — and being 3.6x low is the direction that lets an unaffordable
run start.

`estimate = Σ over stages (fleet_size × vcpus × per-VM wall hours) × CREDITS_PER_VCPU_HOUR`.
Wall hours are *per VM* because that is what a fleet bills: every VM pays its own boot and
then its share of the stage's work. The per-cluster rates from that run were 1.09, 1.24,
and 0.62–0.99; the spread is staggered VM lifetimes rather than a different rate, since a
fleet's VMs do not boot or finish together. 1.0 sits inside the band and prices the run's
own shape ~19% high — pinned from both sides by a regression test that takes the billed
fleet shape as its input.

`settings.coiled_credit_safety` rises to 2.0 to carry the band's width. The estimate
itself stays raw so it remains comparable to an invoice.

### Identity, before credits, before anything

An AWS SSO session expires within hours, which is less than a tile takes. Three times the
driver spent a STAC query, a plan, and a fleet's boot before discovering that nothing it
wrote could reach S3. `quota.preflight_identity` calls STS with a 5-second timeout and no
retries — an expired token does not un-expire — and refuses with the exact command,
reading `AWS_PROFILE` for the profile and falling back to the configured one.

It runs *first*, before the credit check, because a session that cannot call STS cannot
read a Coiled balance either, and "log in again" is a better message than "the balance
could not be read".

### Then permission, which STS does not answer

A valid identity is not a permitted one. On 2026-09-02 the default chain on the submitting
machine resolved `arn:aws:iam::392361759182:user/vercel-data-access`, which reads the
publication bucket and cannot write it. All four profiles on that machine cleared the
identity gate; two of them could not run a tile. The failure that gate misses looks like a
fleet that booted, staged nothing, and left the driver's barrier waiting on shards that
never published.

`quota.preflight_write_access` writes one small object under
`{s3_prefix}/_preflight/`, reads it back, and deletes it. Each of the three answers a
question the others do not. Reading alone clears the very identity that prompted the gate.
An object that will not read back means something between the shell and the bucket rewrote
it. A run that writes and cannot delete leaves artifacts behind, and a later listing reads
leftovers as work that finished. The refusal names the operation, the ARN, where those
credentials came from, and the bucket and prefix tried, because "access denied" on its own
sends people to the wrong console.

It probes, and never infers from IAM policy or bucket ACLs. Those do not compose into an
answer a caller can act on, and a wrong inference is worse than no gate.

**A run has two writers, and they are not always one identity.** The driver's `S3Storage`
builds its client from the default chain. Workers hold no instance role, so every write a
shard performs runs as the credentials `job._worker_environ` freezes: this shell's `AWS_*`
variables when they are set, and `settings.aws_profile` when they are not. The
`forward_aws_credentials=False` on the `batch_run` call turns off Coiled's own forwarding
and does not change this. With no `AWS_ACCESS_KEY_ID` exported, the two resolve to
different identities, so the probe runs against both.

`writer_specs` collapses them where the *source* is provably shared: `AWS_ACCESS_KEY_ID`
exported, or `AWS_PROFILE` equal to `settings.aws_profile`. Never by comparing resolved
credentials, because an SSO profile hands each session its own temporary access key and
key comparison then reports two identities where there is one, probing the bucket twice
for no answer.

### A clock seam, and what it bought

Every wait, poll, deadline, and backoff goes through an injectable `Clock`. That is not
mainly a speed argument: both defects above are time arithmetic, and time arithmetic that
cannot be tested is time arithmetic nobody checks.
`tests/unit/test_driver_state_machine.py` runs 45 scenarios — fresh run, already
complete, adoption, restart at every boundary, worker failure, partial completion across
rounds, the fresh-deadline regression, exhausted rounds, stale records, deterministic
failure, transient and terminal API failures, a killed fleet, both export paths, and the
preflight — **in under a second**, and asserts that no scenario waits for real.

## The three latent defects the seams pass fixed

Stage 3's first PR opened five narrow seams in existing code. Three of them closed
defects that were already present and would have surfaced as silent wrongness rather than
as failures.

1. **`load_scenes` derived its grid from a bbox.** A row band deriving its own grid would
   have been misregistered against the tile it is part of, the same fractional-pixel bug
   ADR-008 fixed between neighbouring tiles. `geobox=` makes a band's pixels *the tile's*
   pixels.

2. **The land mask was rasterized from a transform rebuilt out of bounds.**
   `rasterio.transform.from_bounds` divides a span by a pixel count, so a tile got
   `5/18000` and a band gets `(rows/3600)/rows`. Those differ in the last bits — enough to
   move a polygon edge across a pixel centre and leave a one-pixel land/ocean
   disagreement at every band seam. `masks.get_land_mask_for_geobox` rasterizes against
   the geobox's own affine, and the whole-tile pipeline now uses it too.

3. **A warm offset cache silently widened the composite.** `OffsetCache.read` rebuilt the
   offsets as float64, and `debiased = lst - offset` takes the wider type, so whether
   every intermediate the P95 holds was 4 or 8 bytes wide came down to whether a lookup
   hit. It reads back float32 now. The stored JSON is unchanged, so no `ALGORITHM_VERSION`
   bump.

`debias_with_offsets` also joins offsets to a stack **by time coordinate, never by
position**. A spatial subset can lose a time step, and index alignment would then apply
scene *k*'s offset to scene *k+1* from the first gap onward: a plausible, entirely wrong
correction that nothing downstream inspects.

## Consequences

- A tile's wall clock becomes a fleet width rather than a fixed cost, and a lost shard
  costs a shard.
- The driver's shell must stay open for the duration of one tile. It may be killed and
  resumed at any point.
- Two stages read the coarse stack (ADR-015's accepted two passes) and one reads the
  native stack once (ADR-013 holds *per shard*: a band writes both its products in one
  `dask.compute`).
- The export VM needs real scratch disk: every band slab, a full-tile intermediate, and
  the COG being translated out of it, all at once. `shard_export_disk_gb` defaults to 100.
- Correctness is pinned with **zero tolerance** by
  `tests/integration/test_shard_merge_equivalence.py`: sharded offsets against whole-tile
  offsets, concatenated band composites against the whole-tile composite, and a merged
  COG's arrays *and* every `STATISTICS_*` tag against a single-process export. A tolerance
  would let the failure mode hide — a horizontal seam at a band boundary is a few
  hundredths of a degree over a few rows.

## Alternatives considered

**A dask cluster spanning the tile.** ADR-010 records three validation runs killed by
exactly this in one day. A multi-hour tile graph inside another cluster escaped to the
shared scheduler, then crushed it, then starved the worker heartbeat until Coiled tore
the VM down mid-tile.

**Coiled Workflows or another DAG runner.** A dependency mechanism is what is missing,
and one exists. It is also a second orchestration system to operate, and the barrier it
would replace is thirty lines that only ever ask whether a key exists. Revisit if the
driver grows a second reason to be clever.

**Checkpointing `ref` and resuming a single VM.** Halves the worst case (ADR-015 says as
much) and does nothing about the wall clock, which is the requirement.

## References

- Issue #96, Stage 3 design
- [ADR-008](008-global-mosaic-topology.md) — one shared grid, which a band must not leave
- [ADR-010](010-coiled-batch-for-distributed-runs.md) — Batch, never Functions
- [ADR-012](012-cached-scene-offsets.md) — the offset cache the merge writes
- [ADR-013](013-single-native-pass.md) — one native pass, held per shard
- [ADR-014](014-run-self-explanation.md) — a task is only visible through what it publishes
- [ADR-015](015-bounded-work-unit-offsets.md) — the bounded units the shards are cut from
