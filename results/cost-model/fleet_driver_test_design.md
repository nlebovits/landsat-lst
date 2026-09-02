# Adversarial test design — fleet-consolidated driver (issue #108)

Status: design document. Written against `main` @ c84448b while the
consolidated driver is implemented separately. Nothing here edits
`shard_driver.py`, `shards.py`, or `budgets.py`.

## Why this document exists

`tests/unit/test_driver_state_machine.py` already drives the **per-tile**
driver through 65 tests in under two seconds: restart at every boundary,
capped rounds, partial completion, a deterministically broken shard, a killed
fleet, terminal quota and auth, the export claim. That suite is the model to
extend, not to replace.

Consolidation changes the *scope* of the state machine, not its states. One
work array per stage now carries shards belonging to many tiles, and the
driver holds many per-tile barriers at once. Every bug that suite was written
to catch has a fleet-level twin that it cannot see, because each of its
scenarios has exactly one tile in it:

| Single-tile defect the suite pins | Fleet-level twin it cannot reach |
|---|---|
| Round 2 measured its deadline from round 1's submission | Tile B's barrier measured from tile A's submission |
| An empty `ServerError` killed the driver | One tile's terminal error kills 699 healthy tiles |
| Round 2 resubmitted the whole stage, not the missing index | Round 2 resubmits every tile's shards, not the missing ones |
| A stale record started a new round rather than adopting | Two drivers each submit the same tile-shard |
| Retry rounds counted across drivers, not per driver | Retry rounds counted per array, not per tile-stage |

The suite's own closing test asserts the whole thing runs without real
waiting. That property is what makes an adversarial suite affordable, and it
is a hard requirement on everything below: **every scenario here must run
against an injected clock and a scripted fleet, never against a real barrier.**

## Invariants under test

These are the properties the scenarios exist to defend. A scenario that does
not map to one of these is decoration.

- **I1 — Submission count is bounded by stages and rounds, never by tiles.**
  This is the entire objective of #108. If it does not hold, nothing else
  matters.
- **I2 — Per-tile barriers advance independently.** A tile that is ready
  proceeds regardless of what any other tile is doing.
- **I3 — Completion is bytes in the bucket.** Never an exit code, never a
  state object. Already the rule at tile level (CLAUDE.md); it must survive
  the move to fleet level unchanged.
- **I4 — A shard re-run at an existing key is a no-op exit.** What makes
  resubmission safe, and therefore what makes every recovery path safe.
- **I5 — Max in-flight VMs is a hard cap.** Not a target, not an average.
- **I6 — No silent on-demand fallback.** A spot-configured fleet that cannot
  get spot capacity waits or fails; it never quietly bills on-demand.
- **I7 — The retry budget is persisted and counted across driver restarts,
  per tile-stage.** Per-driver counting hands every resume a fresh budget;
  per-array counting lets one sick tile consume the whole build's budget.
- **I8 — Terminal control-plane failures stop the build; transient ones are
  retried.** Including an error with no message, which must be transient
  (an empty `ServerError` killed the driver once).

## Harness extensions required

The existing fixtures are single-tile: `RUN_ID`, `TILE`, `make_plan()`,
`FakeFleet`, `ScriptedFleet`, `TickingStorage`, `FakeClock`. Four additions
carry them to fleet scope, and they belong in `tests/unit/shard_fixtures.py`
beside the ones they extend.

1. **`make_plans(tiles)`** — a plan per tile, with *differing* shard counts.
   Uniform widths hide index-mapping bugs: if every tile has three shards,
   an off-by-one that maps global index 4 to (tile 1, shard 1) instead of
   (tile 1, shard 0) still lands somewhere plausible. Use widths like
   `{A: 2, B: 3, C: 5}` so a mis-mapping lands out of range or on the wrong
   tile and the assertion fires.
2. **`ScriptedFleet` keyed by `(tile, stage, index)`** rather than
   `(stage, index)`. `lands_after`, `never`, and `heal` all need the tile
   component; without it "one slow tile" is not expressible.
3. **`submitted_indexes: list[tuple[str, frozenset]]`** per submission — the
   stage and the exact set of `(tile, index)` pairs carried. Counting
   submissions proves I1; only the *contents* prove that round 2 carried
   the missing work and nothing else.
4. **`peak_in_flight`** — the fleet tracks concurrent VM count and its
   maximum over the run. Asserting on the peak is the only way to test I5;
   asserting on the requested width tests the driver's intent, not its
   behaviour under a straggler.

Storage stays `LocalStorage` on `tmp_path`. Nothing here needs S3, and a
scenario that does is a scenario that will be skipped within a month.

---

## Scenarios

Numbering continues the existing suite's convention. Each carries the defect
it exists to catch, because a test whose rationale is not written down is a
test the next person deletes.

### Group A — consolidation actually consolidates

**A1. One submission per stage carries every tile's shards.**
Given 12 tiles with differing shard widths, when the build runs to
completion, then the count of offsets submissions is 1 and composite
submissions is 1 (plus at most `shard_barrier_rounds - 1` recovery rounds),
and the union of `(tile, index)` pairs carried equals every shard in every
plan, each exactly once.
*Catches:* a "consolidated" driver that loops `drive_tile` internally. This
is the acceptance gate of #108 and must fail loudly against the old driver.

**A2. Submission count does not grow with tile count.**
Run the same scenario at 3, 12, and 50 tiles; assert the submission count is
identical across all three. A constant, not a slope.
*Catches:* per-batch chunking that reintroduces O(tiles) submissions with a
larger constant — which would pass A1 at 12 tiles and fail the objective at
700.

**A3. Boot is paid per VM, not per tile-stage.**
Assert total VM launches across the build is bounded by
`max_in_flight × (stages + recovery rounds)`, not by `tiles × stages`.
*Catches:* the actual cost regression. A1 can pass while each shard still
lands on its own freshly-booted VM, which buys nothing.

### Group B — barrier independence

**B1. Interleaved progress: tiles at different stages simultaneously.**
Given tiles A, B, C where A's offsets land immediately, B's after 5 polls,
C's after 12, when the build runs, then A reaches composite while B is still
in offsets, and the composite submission carries A's shards before C's
offsets have landed.
*Assert on ordering, not just completion:* record `(clock.now(), tile,
stage)` at each transition and assert A's composite transition strictly
precedes C's offsets completion.
*Catches:* a driver that walks tiles in lockstep — the most natural wrong
implementation, and one that completes correctly and slowly, so a
completion-only assertion passes.

**B2. One slow tile does not stall ready tiles.**
Given 10 tiles where tile 7 lands after 200 polls and the rest after 1, then
9 tiles reach `export` before tile 7 leaves `offsets`, and the wall clock at
which tile 1 completes is independent of tile 7's delay (run at delay 200 and
delay 400; assert tile 1's completion time is unchanged).
*Catches:* a global barrier disguised as per-tile barriers. The two-delay
comparison is the load-bearing part — a single run cannot distinguish
"independent" from "coincidentally fast enough".

**B3. One permanently failing tile does not fail the build.**
Given tile 4 whose composite shard 1 never lands, when the build runs, then
tile 4 exhausts its rounds and is reported failed **by name**, every other
tile completes, and the driver's exit is a summary carrying 1 failure and
N-1 successes rather than an exception.
*Catches:* `ShardStageFailed` propagating out of one tile and aborting the
build. At 700 tiles this is the difference between losing a tile and losing
a day.

**B4. A failed tile releases its capacity.**
Following B3, assert that after tile 4 is abandoned, in-flight capacity is
reused by waiting tiles — the peak in-flight after the failure equals the
cap, not the cap minus tile 4's width.
*Catches:* leaked reservations. Invisible at 3 tiles, fatal at 700.

### Group C — recovery and idempotence

**C1. Driver death and restart at every barrier, many tiles mid-flight.**
Parametrise over the boundary set, but with the build in a *heterogeneous*
state: tiles spread across `nothing`, `resolve`, `offsets`, `composite`,
`export` simultaneously. Kill the driver, restart, assert completion and that
the restarted driver submitted **only** work absent from the bucket.
*Catches:* a resume that reconstructs from a checkpoint rather than from a
listing. The existing test 4 does this with one tile, where "reconstruct the
position" is a single index; with many tiles it is a mapping, and a mapping is
where this breaks.

**C2. Preemption after artifact upload.**
Given a shard whose artifact is written and whose task then exits non-zero
(the spot-reclaim case, and the `$7.28` run contains one), when the driver
next checks, then the shard is complete, no resubmission occurs, and the
tile advances.
*Catches:* an exit-code-driven driver. I3 in its sharpest form: the exit code
and the bucket disagree, and the bucket is right.

**C3. Duplicate submission is harmless.**
Given the same `(tile, stage, index)` submitted twice — a resume adopting a
live stage it could not see, or two rounds racing — assert the artifact is
written once, its content is unchanged, and the tile completes.
Additionally assert the *second* shard process exits early having found its
own key present, rather than recomputing.
*Catches:* both halves of I4. The content assertion catches corruption; the
early-exit assertion catches paying twice, which at 700 tiles is a budget
item, not a nicety.

**C4. Capped retries counted across restarts, per tile-stage.**
Given `shard_barrier_rounds = 2` and a tile whose shard never lands, when the
driver is killed and restarted three times, then that tile's total
submissions for that stage is 2, not 6 — and, separately, a *healthy* tile
that has consumed 0 rounds still has its full budget available.
*Catches:* per-driver counting (each resume gets a fresh budget — the
existing suite pins this at tile level and the fleet version must not lose it)
and per-array counting (one sick tile burns the build's budget).

**C5. Partial submission failure.**
Given a submission API that accepts 6 of 10 tiles' shards and raises for the
rest, assert the 6 accepted are not resubmitted, the 4 rejected are retried,
and no tile is double-counted against its round budget for work that was
never accepted.
*Catches:* the all-or-nothing assumption. A per-tile driver's submission is
atomic; a consolidated one's is not, and a failure mid-array is a state the
old suite has no vocabulary for.

### Group D — caps and money

**D1. Max in-flight is never exceeded, at any instant.**
Instrument the fleet to record concurrent VM count at every clock advance;
assert `peak_in_flight <= settings.<cap>` across a build with 50 tiles and a
cap well below the total shard count.
*Catches:* a cap enforced at submission time but not across rounds — round 2
submitting while round 1's stragglers still hold VMs is exactly how a cap
gets doubled.

**D2. The cap binds across stages, not per stage.**
Given the overlap behaviour (composite starts inside the offsets barrier),
assert the peak counts offsets and composite VMs *together*.
*Catches:* two caps that each hold and jointly do not. The overlap is a
deliberate feature; a per-stage cap makes it a deliberate cost overrun.

**D3. No on-demand fallback.**
Given a fleet whose spot request is refused, assert the driver retries,
reports, or fails — and assert on the submitted request itself, that no
submission ever carries an on-demand or `spot_with_fallback`-widened
specification when the configuration says spot.
*Catches:* the silent 2-3x. Asserting on the *request* rather than on an
outcome is the point: a fallback that never triggers in the test still ships.

**D4. Terminal control-plane failure stops the build.**
Quota, credits, billing, auth → the build fails immediately, naming the
reason, having submitted nothing further. Assert no tile advances afterward.
*Catches:* burning a fleet's boots against a quota that will reject
everything. The per-tile suite pins the classification; the fleet version must
pin the *blast radius*.

**D5. An error with no message is transient.**
Explicitly assert that `ServerError("")` is retried with backoff.
*Catches:* the regression that killed the driver on 2026-08-22. It is pinned
at tile level today and must stay pinned when the classification moves.

**D6. Preflight runs once for the build, before anything submits.**
Assert identity is checked before credits, both before the first submission,
and neither is re-run per tile.
*Catches:* 700 STS calls, and — worse — a build that passes preflight, runs
four hours, and then re-checks into an expired SSO session mid-flight. The
session expires within hours; a consolidated build lasts longer than one.
This scenario is the reason a build-scoped credential-refresh strategy has to
be a design decision rather than an emergent one.

### Group E — scientific equivalence

**E1. Consolidated output is byte-identical to the per-tile driver's.**
For one small fixture tile, run both drivers and compare every artifact byte
for byte, including the merged offset record at the canonical `_offsets/` key.
*Catches:* the acceptance criterion of #108 that no state-machine test
touches. Consolidation is a scheduling change; if any value moves, it is a
bug, and `offsets.ALGORITHM_VERSION` discipline says so.

---

## What this design deliberately does not test

- **Real Coiled, real S3, real timing.** Every scenario above is reachable
  with an injected clock and local storage. A suite that needs the cloud is a
  suite that runs once.
- **Whether a stage that fits its budget finishes.** `budgets.py` is explicit
  that it bounds how long a driver waits, not how long work takes. A test
  asserting the latter would be asserting on the rate model, which belongs in
  the benchmark tier.
- **Throughput.** Wall-clock speedup is a property of the real fleet.
  A1–A3 pin the *structural* precondition for it (submission and boot counts);
  they must not be dressed up as a performance claim.

## Credential-lessness

Per CLAUDE.md, anything reaching `quota.preflight_identity`, `read_balance`,
or a real `coiled`/`boto3` call reads the machine rather than the code. Group
D scenarios touch all three, so they must stub them, and the suite must be
validated with:

```bash
FAKEHOME=$(mktemp -d)
env HOME="$FAKEHOME" AWS_ACCESS_KEY_ID= AWS_SECRET_ACCESS_KEY= \
    AWS_PROFILE= COILED_TOKEN= \
    .venv/bin/python -m pytest tests/unit -n auto -q
```

This has escaped twice. Group D is precisely the territory where it escapes.

## Memory footprint

Any scenario reaching `run_composite_shard` or `run_export_merge` must call
`shard_fixtures.stub_tile_geoboxes` (CLAUDE.md: a toy plan does not make a toy
grid — the production 18,000-column geobox is derived from the real tile
name). The fleet scenarios multiply tile count, so the failure mode that
OOM-killed a 7 GB CI worker at one tile per test is strictly worse here.
E1, which actually computes, is the scenario to watch; A–D should stub the
loader throughout and assert on scheduling, never on pixels.
