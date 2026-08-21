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
