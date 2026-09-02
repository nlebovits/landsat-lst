# ADR-018: One work array per stage per wave, across many tiles

**Status:** proposed (local gates only; no cloud work implied)
**Supersedes nothing.** Extends [ADR-016](016-sharded-tile-execution.md), which
cut one tile across many VMs. This cuts *many tiles* across one fleet.

## Problem

ADR-016 pays a fleet's boot per tile per stage. On S30W065 an offsets-side
shard computed for about six minutes while its stage held a fleet for about
thirty: `budgets.VM_BOOT_S` is 300 s, paid concurrently by every VM in the
array, once per stage, once per tile. Across 700 tiles that provisioning idle
is repeated 700 times on each of two stages, and it is the dominant term in
both the bill and the wall clock. Nothing about the *work* changed; only how
often we pay to start it.

## What was rejected, and why

**A persistent work array fed units as tiles become ready.** This is the
shape the boot argument points at: one array, alive for the whole build,
handed new units as they appear. `coiled.batch_run` has no such mechanism —
it starts an array over a fixed `map_over_values` and returns. Reusing a
cluster name is not a way in either: that path is exactly the observed
`Unable to add batch jobs to existing cluster '...-climato'` failure, which is
why `stage_cluster_name` carries a round. Building the missing mechanism means
a long-lived worker polling a queue, which is a shared scheduler wearing
another name, and ADR-010 records three runs killed by exactly that in one
day.

**A Coiled DAG.** Same answer as ADR-016. There is none.

## Decision

**Batched multi-tile waves.** One Coiled Batch array per *stage per wave*,
whose `map_over_values` carries units from every tile that is ready for that
stage when the wave flushes. A wave with more units than workers is where the
saving comes from: Coiled queues the surplus onto workers that have already
booted, so a boot is paid once per VM per wave instead of once per VM per
stage per tile.

The ordering between stages remains a poll loop against S3, per tile. The
driver is now a single non-blocking loop over per-tile tracks rather than a
blocking sequence for one tile.

### Submission count does not scale with tile count

A wave flushes when any of three conditions holds:

1. the buffered units fill the remaining VM headroom (`fleet_max_vms`),
2. `fleet_wave_window_s` has passed since the first unit was buffered,
3. no track that has not yet demanded could still join this stage (quiescence).

So submissions are bounded by `ceil(total_units / cap)` and by
`elapsed / window`. Neither term mentions the tile count. Twelve tiles and
four tiles produce the same number of submissions when their demands
coincide, which is the property the test suite asserts directly rather than
inferring from a wall-clock measurement.

### Per-tile barriers advance independently

Each tile owns a `TileTrack`: a non-blocking state machine over
`offsets -> merge -> composite -> export`, stepped once per poll. A track that
is watching, exhausted, or failed is stepped and returns nothing; it cannot
hold the loop. A tile that fails — barrier rounds exhausted, no plan, a dead
fleet — is recorded and the loop continues. Only a *terminal* control-plane
failure (quota, credits, billing, auth: `shard_driver.classify_failure`)
aborts the whole run, because that one is not tile-specific and retrying it
anywhere costs the same silence.

### Everything durable stays where it was

- Work-unit bodies are untouched. `shard_tasks.run_shard` is called with the
  same `(stage, run_id, tile, index, job, units)` it is called with today; the
  only new thing on the VM is a token that says which `(tile, index)` this task
  owns, resolved before the body runs.
- Artifact keys are untouched. `shard_root(run_id, tile)` already namespaces by
  tile, so a run id shared by many tiles keeps their plans, blocks, partials
  and bands disjoint. `shards.py` still owns the grammar; the fleet adds keys
  there rather than deriving any elsewhere.
- The offset record is untouched: the merge still runs in the driver and writes
  the ordinary ADR-012 record at the canonical key.
- Completion is still bytes in the bucket, and a shard still checks its own
  output first, so a unit resubmitted in a later wave is waste rather than
  corruption.
- Submission records stay **per tile** (`stage_submission_key`), even though one
  wave writes many of them against one cluster. That is what keeps round
  budgets counted across drivers, keeps adoption per tile, and keeps
  `resume` able to tell a booting tile from an unstarted one.
- `shard_spot_policy` is passed through unchanged. There is no on-demand
  fallback anywhere in this path, silent or otherwise.

### The fleet manifest

`_shards/{run_id}/fleet.json` lists the tiles in the run and each tile's job
parameters. It is written before the first submission and is the one thing
`resume_fleet` needs: the tile list cannot be recovered from a listing that
contains only the tiles that got far enough to write something. The VM reads
its own tile's parameters from it, which is why tiles with different windows
can share a fleet rather than being forbidden from it.

## Queue depth is the design, and it sets the deadline

The saving comes from queue depth and nothing else. With `R = ceil(units /
max_workers)` serial rounds, the share of provisioning a wave avoids —
*capture* — is `1 - 1/R`. A wave with `R = 1` saves nothing at all; it is a
per-tile submission wearing a different name.

The first version of this driver did not follow that through, and the defect
was fatal rather than inefficient. It gave every wave a deadline of **one
shard's** budget (`budgets.stage_budget(...).deadline_s`), independent of `R`.
So any wave deep enough to save a boot expired long before it could finish; the
driver retired it and handed its width back as headroom while its workers were
still running; the next submission went out into capacity that existed only on
paper. Reproduced at `R = 4` against a cap of 4: the driver reported four free
slots with every unit still outstanding, then ran eight workers. Real capture
could not exceed 0.50, and at 700 tiles the first wave (~10,500 units, `R ≈
164`) would expire at roughly one percent of its runtime, every tile would
re-demand, `shard_barrier_rounds = 2` would be exhausted, and the build would
fail wholesale. The design was sound only up to about `cap × safety` units —
roughly nine tiles.

`budgets.wave_deadline_s` now derives a wave's deadline from three named terms,
because a wave's wall clock has three distinct sources and lumping them hides
which one a late wave overran:

    deadline = (provisioning + (R + 1) × unit_work) × safety

- **provisioning** is paid once per worker, not once per unit: the workers
  start together and the queue drains through them.
- **queued execution** is `R × unit_work`. A unit that has not started cannot
  be overrunning its own execution budget, so the budget has to cover the units
  ahead of it in the queue as well as its own.
- **tail** is one further `unit_work`. Units are not equal, and a greedy
  queue's makespan exceeds the ideal `R` rounds by at most one unit's duration
  (the standard list-scheduling bound). Budgeting the ideal exactly would make
  the last straggler in every wave look late.

The same horizon is written into each tile's submission record, so a tile whose
unit is scheduled last in the queue is not judged against one round's worth of
time by a later driver.

## Capacity is released on evidence, never on a clock

A deadline is not proof that anything stopped. A wave is retired — its width
returned to the cap — only when its units' artifacts are listed, or when the
backend confirms the submission dead. An expired wave whose units have not
landed keeps its capacity and is logged as overdue.

Held width is counted **per unit, not per wave**: a worker that has published
its unit's artifact has moved on, so a 60-unit wave with 45 artifacts on the
ground holds at most 15. Counting the requested width until the whole wave
settles is not conservative but wrong in a way that deadlocks — a first wave
wider than the cap would never return any headroom, and a run would stall with
most of its work done. That was reproduced too, and is what
`FleetDriver.wave_held` exists to prevent.

The barrier remains a separate question and still expires on time: a tile
re-demands when *its record* ages out, whether or not the wave holding its
capacity has been released. Expiry therefore delays a resubmission rather than
permitting an over-run, which is the safe way round. A wave that neither
settles nor can be confirmed dead holds its width until the poll ceiling ends
the run — deliberately, because a loud stall costs less than a silent doubling
of the bill.

The rule is easy to state and easy to breach, because there are several facts
about the *driver* that look like facts about a VM. Three were found in review
and each is pinned by a test that fails when the shortcut is put back:

- **A settled track is not a stopped worker.** An earlier draft also retired a
  wave once every tile it carried had settled that stage. A tile settles when
  it finishes, and it settles just as surely when it exhausts its barrier
  rounds — at which point the artifacts are missing *because* the workers are
  still running. Retirement now reads artifacts only, and a failed tile's units
  keep their width until they land or the probe answers.
- **A track that has stopped looking is not a stage that has finished.** The
  composite wave is demanded from inside the offsets barrier, and the track will
  not revisit the composite stage until it has merged. So the driver refreshes
  the evidence for every `(tile, stage)` a live wave references, once per poll,
  whether or not that track would have looked.
- **A wave that never started must not be counted.** A submission that raised
  through every retry has no workers, no handle to probe, and artifacts that
  can never land. Counting it is the mirror failure: not an over-run but a
  deadlock, capacity spent for the rest of the run with nothing able to release
  it.

Adoption follows the same rule, and this is where the first draft put the
defect back one process later. It adopted only waves whose deadline had not
passed, so a resumed driver ignored a merely late wave and submitted into
headroom that existed only on paper. The wave record therefore carries its unit
tokens, a resumed driver adopts every recorded wave, and `_retire` settles them
on the ordinary terms. That terminates because an overdue wave is always probed
regardless of the `probe_waves` setting: a backend that answers retires an
ancient record on the first poll, and one that cannot answer holds the capacity
loudly.

## The poll loop's request rate

Every tile's barrier asks the same question against prefixes under one run
prefix, so asking per tile made LIST calls linear in tile count: about 1,400
serial listings a cycle at 700 tiles, measured at roughly 70 s against a 30 s
poll. Beyond about 300 tiles the loop silently stops keeping up with itself.
`PollIndex` lists each shared prefix once per cycle and serves every tile from
it, and caches record bodies against the modification time the listing already
returns.

What that buys, stated exactly rather than generously: the number of LIST
operations per poll stops depending on how many tiles are being driven and
starts depending on how many keys the run has published, because a paginated
listing returns 1,000 keys a request. At 700 tiles and roughly 50 keys a tile
that is about 35 requests a poll against 1,400. The exponent on the tile count
is what changed, not the presence of one, and calling it a constant would be
the kind of wrong that only appears at the scale nobody tests at.

Sharing the request is only half of it. A cached listing that every tile filters
end to end is still `O(tiles × keys)` of CPU per poll, which at 700 tiles is
tens of millions of prefix comparisons inside a 30 s budget. Keys are held
sorted, so one tile's slice is a binary search plus what it matched.

## The backend contract

The boot amortization above is a property of the *submission substrate*, not of
this pipeline. Whether AWS Batch, an ECS service, or a plain EC2 Spot fleet
delivers it more cheaply is a question worth being able to ask without
rewriting the state machine — so the driver depends on
`fleet_backend.FleetBackend`, and `CoiledFleetBackend` is one implementation
rather than the implementation. Nothing Coiled-shaped crosses this boundary: a wave
comes back as a `WaveHandle` (an opaque id, a name, a worker count), and the
only questions the driver asks about it are "is this dead" and "was that
failure worth retrying", both answered by the backend.

`fleet_backend.BACKEND_CONTRACT` is what any substrate has to provide, and it
is **declared, not inferred**: a backend publishes a `guarantees` set and
`FleetDriver` refuses one that does not cover the contract. An AWS Batch
evaluation therefore has a checklist rather than a reading exercise.

| Guarantee | What the state machine needs |
|---|---|
| `queues_surplus` | With `len(units) > max_workers`, *every* unit runs on at most `max_workers` concurrent workers, a worker taking the next unit when it finishes. This is the entire saving; a substrate that refuses, or drops the surplus, fails the contract rather than merely performing worse. |
| `fire_and_forget` | `submit` returns promptly with a handle. The driver has other tiles to step. |
| `at_least_once` | A unit may run more than once. Units are idempotent at their artifact keys, so this is permitted rather than tolerated. What is not allowed is silently skipping a unit in a way the driver cannot notice — which it notices by listing the artifact, never by an exit code. |
| `no_dependencies_needed` | Stage ordering is the driver's poll loop (ADR-010, ADR-016). A substrate with a DAG feature is fine; its DAG feature is unused. |
| `unique_wave_names` | `wave_name` is unique per `(run_id, stage, wave)` and stable, so two drivers agree and a resumed one does not rebuild a name still in flight. |
| `opaque_handle` | The handle id is JSON-serializable and stable: it is persisted in the wave record and handed back to `probe` by a later process, possibly on another machine. |
| `probe_is_advisory` | `probe` may say "dead" or "unknown", never "succeeded". Completion is bytes in the bucket, so a probe can only ever end a barrier sooner. |
| `classified_failures` | Control-plane errors map to terminal or transient, and an *unrecognized* error maps to transient. Guessing terminal for the unknown case turns every ordinary blip into a dead run. |
| `no_silent_cost_substitution` | However cheap capacity is requested, the expensive class must not be silently substituted. |

### `queues_surplus` is documented semantics, not a measurement — GATE

The entire economic case rests on one behaviour: that `coiled.batch_run` given
`len(units) > max_workers` runs every unit on at most `max_workers` workers,
handing a worker the next unit when it finishes one. **That is read from
documented semantics and has never been verified against real Coiled by this
project.** No number in this ADR is a measurement of it.

If the substrate instead starts one worker per value, or refuses the
submission, or silently drops the surplus, then capture is zero and this design
buys nothing — while still having changed the driver, the deadlines and the
capacity model.

So: **a ~$0.50 cloud probe — one wave, a handful of trivial units, more units
than workers, observing how many workers start and how the units are
distributed — is a required gate before any capture assumption is trusted, and
before any cost estimate derived from capture is quoted.** It is **not
authorized and has not been run**; it is recorded here as the gate it is. Until
it passes, treat every capture figure in this document as an assumption with an
unverified premise.

### What stays Coiled-specific, and why

Four behaviours are Coiled's rather than the contract's, and all four live
inside `CoiledFleetBackend`:

- **Cluster-name collision.** `batch_run` refuses a name matching a running
  cluster, which is why the wave number is in the name and the run id is
  hashed (spelling it out got the marker truncated away once). The contract
  asks only for uniqueness; a substrate that tolerates duplicate names is free
  to ignore the rest.
- **An error with no message is ordinary.** An empty `ServerError` — the credit
  quota, as it turned out — killed a driver outright on 2026-08-22. That is why
  the Coiled mapping sends unknown to transient. The contract fixes only the
  *direction*, not the marker list.
- **Cost is credits and identity is AWS SSO.** The preflight is those two gates
  in that order, because a session that cannot call STS cannot read a Coiled
  balance either.
- **Workers always write S3.** Hence `validate_storage`; a driver polling a
  local directory would hang on a barrier whose artifacts are in a bucket.

Outside the backend, three couplings remain and are not worth extracting
today: `batch.py` still owns the Coiled task command (including the
`COILED_BATCH_TASK_INPUT` variable name — though the *token grammar* it carries
is the driver's and is backend-neutral), `quota.py` prices in Coiled credits,
and `budgets.VM_BOOT_S` is measured on Coiled VMs. The first is what a second
backend would reimplement; the other two are numbers a second backend would
re-measure.

## Consequences

- A wave is as slow as its slowest unit for the purpose of retiring VM
  headroom, so a long tail of one tile can hold capacity that later tiles
  want. The cap is enforced against *live* waves, and a wave is retired only
  when its units settle or the backend reports the submission gone, so a tail
  costs real headroom for as long as it really runs. That is the deliberate
  trade: an earlier draft retired on the deadline instead, which returned the
  tail's width on paper while its workers were still billing.
- **The cap invariants rest on an unverified external premise.** Every
  statement here about peak concurrency assumes `coiled.batch_run` runs an
  over-subscribed array on at most `max_workers` VMs. If that is false,
  `max_workers` bounds nothing and the cap is a number in a log. See the gate
  above; it is not closed.
- Provisioning idle is recorded rather than inferred, in two halves, because
  one half is not enough. Each wave record carries `submitted_at`,
  `first_completion_at` and `last_completion_at`: poll-resolution observations
  against the bucket, bounded by `fleet_poll_s`, which can only be late. Those
  three bound billed VM time from above, `workers × (last − submitted)`, and
  they cannot go further: a worker waiting for its next unit and a worker
  running one are indistinguishable from the bucket. So each unit also writes
  its own start and end, at `shards.unit_timing_key`, and idle is billed minus
  boot minus the sum of those durations. The unit records live under
  `_shards/timings/{run_id}/`, deliberately outside the run prefix the driver
  lists every poll: one object per unit is tens of thousands of keys, and the
  poll listing exists to answer a question about barriers. Both writes are
  best-effort. A lost duration costs a term in a cost model; nothing about
  instrumentation may cost a composite.
- Boot amortization is a property of *many units per wave*. A fleet run of one
  tile is exactly ADR-016 with extra indirection and no saving; that is
  expected and is not a reason to route single-tile work through here.
- This amortizes across tiles. It does not make a single full-tile rerun fast,
  and must not be sold as if it did.
