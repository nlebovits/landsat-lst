# ADR-015: Bounded work units for the offset pass

**Status:** Accepted
**Date:** 2026-08-16
**Authors:** @nlebovits

## Context

The offset pass could not run a production window. Not slowly — at all.

`offset_graph` estimates one scalar per scene from a per-pixel monthly climatology. Two
medians do the work, and they reduce along **orthogonal axes**: the climatology is a median
over *time* per pixel, the offset is a median over *space* per scene. Direct layer
inspection at production geometry (300 scenes, 9000², chunk 256) found the consequence:

```
TOTAL 1,857,873          (lwir11 only)
  rechunk-merge   777,600   = 2 x (1296 blocks x 300 scenes)
  getitem         388,801
  sub             388,800
  shuffle         116,665
  nanmedian        15,852
```

**Two** rechunk layers of 388,800 tasks each. One gathers time, one gathers space. No
chunking satisfies both, so the scheduler materializes the stack whatever it is told.

Three failure modes followed, the first fatal on its own:

- Graph **construction** exceeded 26 GB above 2,000 scenes, and hit 48.5 GB unguarded.
  Chunk-size independent.
- **Execution** held a scene-independent ~21 GB plateau plus 14–28 MB per scene.
- The fused graph reached ~20M tasks, or 26–48 GB of scheduler state at the measured
  2,378 bytes per task.

PR #99's month-loop reformulation removed the groupby shuffle and cut memory from 46.5 GB
to 17.4 GB. It did not remove the two rechunks, because they are the estimator's shape
rather than the spelling of it.

## Decision

Split the pass along the axis each median is parallel in, and run each half as bounded work
units in plain numpy. No dask graph spans the window.

**Phase A — climatology, sharded over space.** `climatology_by_blocks` walks spatial blocks.
Each block is read with one small dask graph and reduced in memory. The block edge comes
from `_io_block_edge`: the largest power of two whose stack fits `destripe_unit_memory_gb`,
so unit memory stays flat as the window deepens rather than scaling with it.

**Phase B — offsets, sharded over scene.** `offsets_by_scene` takes one batch of scenes at a
time, subtracts that scene's month, and reduces over space. The loop carries no state, so a
scene's cost is independent of how many others exist.

`scene_offsets` dispatches on `settings.destripe_bounded_units` (default true). The graph
form is retained, unused in production, as the equivalence oracle.

### Two parameters that are not the same parameter

The **I/O block** and the **compute panel** are separate on purpose, because reads and
kernels want opposite things. Reads want few and large. The kernel wants a working set that
stays in cache. Measured on a 2250² fixture at 300 scenes:

| panel | wall |
|---|---|
| **256** | **65.8 s** |
| 512 | 122.3 s |
| 1024 | 117.6 s |
| 2048 | 116.0 s |

256 is 1.8x faster than anything larger, and the larger sizes are flat within 5% of each
other — a cache cliff, not a trend. Every panel size produced an identical climatology
checksum.

### Alignment to source chunks is not an optimization

`_scene_batches` groups **whole** source time-chunks rather than counting scenes. With the
shipped `TIME_CHUNK = 10` and a batch of 8 the boundaries never line up, every boundary
chunk materializes twice, and the pass pays roughly a quarter of an extra read of the whole
stack. `_io_block_edge` refuses to go below the spatial chunk edge for the same reason.

This was found by a test, not by reading the code: `test_unit_form_reads_the_stack_exactly_twice`
measured **2.51** passes before the fix and 2.00 after.

## Equivalence

Not argued. Measured, on a real 300-scene fixture (S30W065, factor 8, 95.66% NaN):

| variant | max abs Δ | exact | flips | n_valid |
|---|---|---|---|---|
| `nanmedian`, 1024 blocks | **0** | yes | 0 | identical |
| sort-and-index kernel, 1024 blocks | **0** | yes | 0 | identical |
| `nanmedian`, 512 blocks | **0** | yes | 0 | identical |

`np.array_equal(equal_nan=True)` is true against the shipped `offset_graph`, on all 300
scenes, at two block sizes, from two median kernels. 2250 divides by neither block size, so
partial edge blocks were exercised. Re-verified through the shipped `scene_offsets` on the
same fixture: max |Δ| 0.0.

**`offsets.ALGORITHM_VERSION` is therefore not bumped.** The version exists to invalidate
caches when values change. Values do not change, and bumping would force every cached tile
to recompute for an identical answer.

## Measured behaviour at production geometry

One VM, `r6i.2xlarge`, us-west-2, N40W075, 150 scenes, 9000² grid, forced to the production
512 edge and 324 blocks:

```
stac_query               28 s
land_mask                13 s
destripe_climatology  1,823 s   324/324 blocks   26.7 MB/s
destripe_offsets      1,082 s   150/150 scenes   44.9 MB/s
TOTAL                 2,950 s = 49 min           peak RSS 14.66 GB
```

**Per-block cost falls and then flattens** — 8.39 s over blocks 1–42, then 6.11, 5.26, 4.62,
5.14. The early cost is warm-up and it amortizes. There is no accumulating per-block penalty
at the production block count.

**Phase A memory climbs to a bound rather than sitting flat.** RSS went 2.31 → 5.87 GB
across 324 blocks, about 11.4 MB per block. That is not a leak: `ref` is
`12 x 9000² x 4 B = 3.89 GB` of lazily-allocated pages being touched block by block, and
3.89 GB ÷ 324 = 12.0 MB per block. It converges to base + one block + 3.89 GB.

**Phase B memory is scene-count independent by construction.** `_scene_batches` groups whole
`TIME_CHUNK` chunks, so the batch is 10 scenes at 150 scenes and 10 at 2,930. The accounted
resident set is identical at both: ref 3.888 + batch 3.240 + anomaly 0.324 = 7.452 GB.

**Graph construction is not a cost.** A 2,930-scene source array holds 379,728 chunks and
the pass performs 617 slices against it. Measured: 0.020 s to optimize a phase A block,
0.022 s for a phase B batch, 0.14 GB peak, **13.5 s total** — 0.024% of a tile. The cost
scales sublinearly, because per-slice cost tracks the slice's own task count and not the
parent array's.

## Consequences

- The pass reads the stack **twice** where the graph form read it once. That is the price of
  sharding in orthogonal axes and it is pinned by a test, so it cannot quietly become three.
- Progress is reported as `blocks_done/blocks_total` and `scenes_done/scenes_total` rather
  than a dask task fraction. A block index localizes a stall; a fraction over a fused graph
  does not.
- There is no `GraphProgress` on this path, because there is no single graph to count.
- A failure still costs the whole tile. There is no intra-tile checkpoint, and at a
  projected 15.75 h per production tile a spot reclaim at hour 9 loses everything.
  Checkpointing `ref` after phase A would halve the worst case and is not implemented.

## Assumptions this ADR does not establish

Labelled, because the projections built on them are not measurements:

| | status |
|---|---|
| Phase A holds 26.7 MB/s at 20x larger per-block reads (2.93 GB vs 150 MB) | **assumed** |
| Native composite runs at phase B's rate at 18000² | **assumed**, never measured |
| Per-VM throughput holds beyond 8 concurrent VMs | **assumed**; flat 1→8 measured |
| Bit-exactness holds at 2,930 scenes and factor 2 | **assumed**; measured at 300 / factor 8 |
| `notnull` and `isfinite` agree | **latent difference** — they diverge on ±inf, which
  `convert_to_celsius`'s clamp currently makes unreachable |

### The thread cap, reconciled

Every rate above was measured through `compute_tile_offsets`, which `job._thread_cap` does
**not** wrap, so all of them ran uncapped while production capped dask at
`settings.dask_max_threads = 1`. That gap was found in review, before merge, and measured
rather than assumed. Four arms, four VMs, `r6i.2xlarge`, production geometry, 40 scenes,
identical work:

| threads | wall s | peak RSS | CPU fraction | speedup |
|---|---|---|---|---|
| 1 | 1776.4 | 14.60 GB | 0.662 | 1.00 |
| 2 | 1377.3 | 14.66 GB | 0.464 | 1.29 |
| **4** | **1234.7** | **14.15 GB** | 0.276 | **1.44** |
| 8 | 1444.6 | 14.25 GB | 0.149 | 1.23 |

All four returned identical offsets (`n_kept` 15/40, `rejected_frac` 0.625, `std` 2.3).

**Peak RSS is flat within 3.6% and shows no trend in thread count.** The reason `1` was
correct before is the reason it is wrong now: the fused graph made each thread hold a chunk
across the whole time axis, and a bounded block read makes a thread cost one chunk. The
`threads * chunk**2 * scenes * 4` term the old default defended against no longer exists.

`dask_max_threads` is therefore **4**, and `coiled_job_timeout` moves from 6 hours to 24.
Six hours would have killed every production tile mid-climatology.

## References

- Issue #93, and `results/batch1-investigation/batch4/` for E1–E4, the calibration, the
  scaling runs, and the graph probe.
- ADR-007 (the estimator), ADR-010 (one VM per tile), ADR-012 (the offset cache),
  ADR-013 (one native pass), ADR-014 (run self-explanation).
