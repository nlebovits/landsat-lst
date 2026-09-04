# ADR-020: Stage phase A's coarse observations for phase B

- Status: accepted
- Date: 2026-09-04
- Issue: [#125](https://github.com/nlebovits/landsat-lst/issues/125)
- Supersedes nothing. Refines [ADR-015](015-bounded-work-unit-offsets.md).

## Context

ADR-015 split the offset estimator into two bounded phases because its two
medians reduce along orthogonal axes. Phase A shards over space and reduces
every scene into a monthly climatology. Phase B shards over scene and takes one
spatial median each. That ADR accepted two passes over the source stack as the
cost of the split, and pinned it with
`test_unit_form_reads_the_stack_exactly_twice` so it could not quietly become
three.

Two passes was the right call for memory. It was never examined for time. The
retained anchor run `shard-S30W065-2021-2025-20260823T102135Z` measures phase B
at 488 to 687 seconds per shard against 2.6 seconds of compute, so phase B is
read time and nothing else. Priced in billed credits it looks small, 3.8% of a
tile, because the composite stage runs on roughly 36 sixteen-core VMs and
dominates vCPU-hours. Priced in critical-path wall clock it is 17.4%, because
the offsets stage runs on 15 VMs and the composite's much larger vCPU total
compresses into a comparable elapsed time.

The passes cannot share a traversal, and re-sharding phase B spatially would
change the estimator, because a spatial median does not decompose across blocks.
So the only way to stop reading the sources twice is to carry the observations
forward.

## Decision

Phase A publishes what it decodes; phase B reads that instead of the sources.

**The carried form is `uint16` DN with 0 for no observation.** It is half the
bytes of the source pass, which carries two `uint16` bands where the stage
carries one, and half the bytes of the float32 Celsius array phase A used to
hold. `qa.celsius_stack` rebuilds the estimator's float32 input from it bit for
bit, so no estimator, threshold, or output changes. This is the lossless half of
the DN representation from #136; the quantising half, which shifts offsets by
whole DN and takes an integer P95, is deliberately excluded and a test asserts
its absence.

**Only blocks phase A actually reads are staged.** A land-free block is never
read, never staged, and reconstructed as all-NaN, which is what the land-masked
source already yields there. It therefore contributes nothing to a spatial
median and exactly zero to `n_valid`. A block the plan marks as holding land
whose object is missing raises, because reading it as no-observation would thin
`n_valid` in silence.

**The stage is keyed by the `OffsetKey`.** Its prefix carries the digest and
`offsets.ALGORITHM_VERSION`, so a stage written under a different scene set,
offset factor, clamp, or estimator version is at a prefix this run never lists.
Staleness is unreachable rather than merely unlikely.

**The stage is scratch with an owner.** The driver sweeps it when the merged
record lands, because that record is the durable output, and when it gives a
tile up, because a tile that failed will never merge. An object left under the
run prefix is an object a later listing reads as finished work.

## Alternatives rejected

**Reuse only what a shard staged for itself.** A shard's phase-A blocks and its
own phase-B scenes intersect in 1/N of the bytes, which is 6.7% at 15 shards.
It needs no exchange and saves almost nothing.

**Stage the float32 Celsius array.** Twice the bytes for the same information,
and it would have made the transfer cost more than the source read it replaces.

**Fuse the phases into one graph.** The climatology must be complete before any
anomaly can be computed, so a fused graph holds the whole stack across the
barrier. That is the construction cliff ADR-015 exists to avoid.

**Re-shard phase B over space.** Exact, decomposable spatial medians would
require a distributed selection the estimator does not have. It changes the
estimator, which #125 forbids.

## Consequences

- Phase A gains a write. It rides inside the read it was already doing, in
  bounded scene groups, so peak memory is the float32 block it already held
  plus a bounded DN buffer measured at roughly 250 MB, which does not scale.
- The run holds transient S3 scratch: 167 GB for S30W065, half a source pass.
- `StorageBackend` gains `delete_prefix`, its first deleting operation.
- `climatology_by_blocks` and `offsets_by_scene` gain an optional reader each.
  Unset is the direct read, byte for byte, and a test pins that.
- `settings.destripe_stage_coarse` turns the whole thing off.
