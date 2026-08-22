# ADR-012: Cache scene offsets on their inputs, and give the offset pass its own command

**Status:** Accepted
**Date:** 2026-08-13
**Authors:** @nlebovits

## Context

Iterating on the pipeline cost roughly two hours per attempt, and three days of work on
issue #31 produced zero completed tiles. The blocker was never a single bug. Every
experiment re-entered the pipeline at the top and repaid its longest compute, including
the experiments where that compute provably could not change.

The numbers come from run `2021-2025-sample300-20260813T123249Z` on `N40W075`, a 300-scene
sample at chunk 512, offset factor 2, 4 threads on an `r6i.4xlarge`:

| Phase | Time |
|-------|------|
| `stac_query` | 28.3s |
| `loading` | 36.1s |
| `destriping` (598,604 tasks) | 1616.3s |
| `compositing` (nothing reported) | 554s and counting |

The offset pass is 27 of those minutes and its entire output is one scalar offset and one
valid-pixel count per scene: roughly 600 float64 values, a few kilobytes. `scene_offsets`
was already split out of `seasonal_debias` with the docstring reason "computing the offsets
is the expensive part, and a sweep over candidate caps should pay it once." That was true
within one process and false across two, because nothing persisted.

The work that recomputes offsets it cannot change: sweeping `destripe_max_offset_c`, moving
`destripe_min_scene_pixels` or `destripe_min_offset_samples`, any change to compositing, the
land mask, the encoding, or the COG writer, and every rerun after a crash, a spot
preemption, or a timeout.

## Decision

### Cache the estimate, keyed on every input that moves it

`scene_offsets` takes an optional `OffsetCache`. A hit returns immediately; a miss computes
and writes. Records live at
`_offsets/{tile}/{window}/f{factor}/v{algorithm_version}-{digest}.json`, a sibling of the
collection prefixes and invisible to the catalog.

The cache is **keyed, not versioned**. A stale record is unreachable from a changed
configuration rather than detected and discarded, so there is no freshness check to get
wrong. The digest covers:

- **the scene ids**, sorted. `--max-scenes` changes which scenes are pooled, and each
  offset is measured against a monthly climatology built from the whole set, so a different
  scene list is a different answer for every scene in it.
- **`destripe_offset_resolution_factor`**, which decides the grid the median rests on.
- **`lst_valid_min` and `lst_valid_max`**, because `convert_to_celsius` applies the clamp
  before the median sees a pixel.
- **`ALGORITHM_VERSION`**, the manual escape hatch for code changes a hash cannot see: the
  reduction in `offset_graph`, the QA bits `create_qa_mask` masks, the DN conversion.

### Cache the estimate, never the decision

`seasonal_debias` applies `max_offset_c` and the sparse floor to whatever the cache returns.
This is what makes a cap sweep cheap: the same stored offsets are re-judged against each
candidate, so the sweep pays the estimator once rather than once per candidate. Caching
`debiased` instead would have cached the rejection along with it and bought nothing.

### A cache failure must never fail a tile

Every read and write is best-effort: logged and swallowed, the tile recomputes. Same rule
the heartbeat follows, for the same reason. Losing the cache costs 27 minutes. Failing the
tile costs the run. Read misses cover a cold cache, an unreadable object, a malformed
record, and a record whose time axis does not match the stack in hand — the last of which
is logged as a defect, since the digest should have made it impossible.

### Two switches, not one

`--no-offset-cache` disables both halves and leaves the stored record untouched, for
validating a change to the estimator itself. `--force` on `landsat-lst offsets` skips the
lookup but still writes, for rebuilding an estimate whose inputs have not changed. These are
different intentions and conflating them would make one of them unavailable.

Note that `--force` on `composite` and `process` is unrelated: it is about the output COGs.
Re-deriving an unchanged input in order to overwrite a corrupt object would be 27 minutes
bought for nothing.

### Give the offset pass its own command

```bash
landsat-lst offsets   -t N40W075   # estimate and persist; reports the rejection fraction
landsat-lst composite -t N40W075   # one tile to COGs, reading whatever is cached
landsat-lst process   --distributed  # unchanged: the fleet driver
```

`offsets` also answers a question nothing else answered cheaply: a tile's rejection
fraction. `destripe_max_offset_c` was calibrated to 15.0 °C on mid-latitude cropland at
Pergamino, where it rejects 21.8%, and ADR-007 asks for that to be re-checked on a humid
tropical tile before the global build. That check no longer needs a composite.

## Consequences

A compositing experiment on a warm cache drops from about 35 minutes to about 8. A cap
sweep costs 27 minutes once instead of 27 minutes per candidate. A preempted tile's retry
skips the offset pass its first attempt already paid for, since `use_offset_cache` defaults
to on and the batch task forwards it.

Local `write_text` became atomic (temp file plus `os.replace`). S3 gives that for free, and
a truncated offset record that parses as far as it goes would be worse than no record.

**The stored time axis is serialized at nanosecond precision, and that is not cosmetic.**
The record was written with `np.datetime_as_string(..., unit="s")`, which is a *truncation*
of a real Landsat solar-day stamp rather than a spelling of it. That was harmless while
offsets were aligned to a stack by position. ADR-016 made the stamps load-bearing —
`debias_with_offsets` joins by coordinate value, because a row band's stack can lose a time
step and index alignment would then apply scene *k*'s offset to scene *k+1* — and an axis
rebuilt from truncated stamps is a different axis. Every composite shard of S30W065 failed
with `lst carries a time step the offsets do not ... ("not all values found in index
'time'")`.

Records written at the old precision are still read. Where the second-precision rendering
of the live axis matches the stored list element for element **and holds no duplicates**,
the record is accepted and returned on the loaded axis at full precision. A duplicate means
the record is consistent with more than one axis, and that is a miss: recompute rather than
serve numbers that might belong to a different scene set. Support for this is permanent
rather than a migration window — the records are correct answers that cost half an hour of
compute each — and `ALGORITHM_VERSION` is **not** bumped, because no value ever changed.

The reason this survived so long is a testing one worth keeping: every synthetic fixture in
the repo used whole-second timestamps, so no test could distinguish a serializer that
truncated from one that did not. Fixtures whose offsets round-trip through JSON now carry
sub-second components.

**The records were only half of it.** A sharded tile freezes its time axis in
`plan.scene_times` too, and `shard_tasks._time_coord` rebuilds the offset axis from *that*.
A plan written before the fix therefore hands the join a truncated axis whatever the record
says — which is why the packing probe failed every arm after the record fix landed, and why
the S30W065 rerun would have failed again: a resume reads the legacy plan rather than
writing a new one.

`load_context` recovers the lost fraction from `items.json`, which was never truncated: the
loss happened on the way *into* the plan. Two properties make the recovery a derivation
rather than a guess. `odc-stac` sets each group's coordinate to
`group[0].nominal_datetime`, so the value is always some item's timestamp. And several
items inside one second is the *ordinary* case, not an exotic one — `items.json` holds one
entry per scene where the axis holds one per solar-day group, and adjacent WRS rows of one
overpass are seconds apart — but items within a second are necessarily the same date and so
the same group, whose representative is the earliest of them, because `odc-stac` sorts each
group by `nominal_datetime` before taking the first. Taking the minimum is exact.

What remains a hard error is ambiguity nothing can resolve: two entries in the *stored axis*
truncating to the same second, or a stamp no item matches at all. The digest is unaffected,
because it covers the scene ids and the settings and never the stamps — which is precisely
what lets a legacy plan verify against a current process.

**A sampled window cannot validate a rejection fraction.** 93 of 300 sampled scenes survived
de-striping, a 69% rejection rate against 21.8% measured at Pergamino. Spreading 300 scenes
across five years leaves each calendar month roughly 25 scenes to build its climatology
instead of 244, and the noisier reference inflates apparent offsets. The sample window keys
separately, so this cannot contaminate a real tile's cache, but it also means the number
`landsat-lst offsets --max-scenes 300` prints is not the number a full run would print. The
command says so where it prints it.

### What this does not do

The composite and export phases stay fused. Splitting them needs an intermediate on the
order of a gigabyte per tile, and it would buy only the ability to re-run an encoding
change cheaply, which is not where the time goes. Issue #77's premise that item 2 "buys
nothing on its own, because nothing can start halfway" turned out to be wrong: with the
cache keyed on inputs, `process` re-entered from the top *is* starting halfway.

Instrumenting the composite phase (issue #77 item 4) exposed a doubled compute that this
ADR does not fix. `process_tile` runs an eager `.values` over `qa_count` for the coverage
log, and `cog_export` then walks the same native stack again to write the COGs. Two full
passes per tile. It now has its own phase and a `GraphProgress` so a watcher can see it;
sharing the two passes is separate work.

## Alternatives considered

**Cache the debiased stack.** Gigabytes per tile instead of kilobytes, and it would freeze
the rejection decision into the artifact, killing the cap sweep that motivated the cache.

**Version the cache rather than key it.** A version field needs a freshness check, and a
freshness check that misses is exactly the failure mode — a stale result arriving fast and
looking correct. Keying makes staleness unreachable rather than detectable.

**Hash the whole settings object.** Simple and wrong: every unrelated setting would
invalidate the cache, including `cog_blocksize` and the Coiled knobs. Only inputs the
estimator reads belong in the key.

## References

- Issue #77, items 1, 2, and 4
- ADR-007 (scene normalization), ADR-010 (Coiled Batch), ADR-011 (static planning)
- `docs/findings-offset-subsampling.md`
