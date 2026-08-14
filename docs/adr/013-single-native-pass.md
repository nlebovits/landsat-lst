# ADR-013: One pass over the native stack per tile

**Status:** Accepted
**Date:** 2026-08-14
**Authors:** @nlebovits

## Context

A tile read the full native stack three times. Two of those reads were avoidable and one of
them recomputed an array the first had already materialized and discarded.

The three walks all descended from the same de-biased, land-masked stack, so each one
re-read every scene in the window:

1. **Coverage check** (`pipeline.py`) — `composite["qa_count"].sum(dim="month").values`
   forced an eager compute over every scene. The result fed one log line and was dropped.
2. **LST export** (`cog.py`) — `_write_intermediate` on `lst_p95`.
3. **QA export** (`cog.py`) — `_write_intermediate` on `qa_count`, recomputing what walk 1
   had already produced.

`lst_p95` comes from `lst.quantile(...)`, `qa_count` from
`valid_mask.groupby("time.month").sum()`. Both need every scene, and dask held nothing
between the three computes.

This was invisible until the phase split in issue #79. The old single `compositing` label
spanned graph construction, the coverage reduction, and the handoff to export, so nine
minutes of silence inside it could not be attributed to any of them.

### Measured, not read off the code

A `dask.callbacks.Callback` counting source-block productions on a synthetic tile
(`profiling.synthetic_dataset`, no network, no cloud) settled the pass count exactly:

| Shape | Passes | Peak RSS | Wall |
|-------|--------|----------|------|
| Three sequential computes (before) | **3.0x** | 1.30 GB | 126s |
| Fused export writes only | 1.0x | **10.88 GB** | 110s |
| Fused writes + shared rechunk (after) | **1.0x** | 1.60 GB | 123s |
| One blockwise kernel (rejected) | 1.0x | 1.30 GB | **1197s** |

4096 x 4096 x 120 scenes, chunk 512, 4 threads. Wall clock is not the headline here: the
synthetic source is `dask.array.random`, so a "read" is CPU-bound generation rather than the
network I/O a real tile pays. The pass count is what transfers. Counting task-key names
instead of executions does not work at all: graph fusion renames keys, and the count then
silently reads zero.

The middle row is the trap. Handing both deferred writes to one `dask.compute` does collapse
the reads to one pass, and it destroys streaming while doing it — 8x the peak memory, which
at production geometry is not a regression but an OOM. The two products consumed differently
chunked views of the same stack: `quantile` rechunks time to a single chunk, `groupby` does
not. With two incompatible consumers the scheduler has no block-by-block order that serves
both, so it fans out and holds the whole stack.

## Decision

### Build both products on one time-contiguous view

`_composite_graph` rechunks time to a single chunk before building either expression. The
P95 forced that rechunk anyway, so it costs nothing against the memory floor
(`threads * chunk**2 * scenes * 4`); what it buys is a single shared consumer structure, and
with it a scheduler order that streams.

### Write both intermediates in one compute

`rio.to_raster(compute=False)` returns a deferred store and writes the header eagerly, so
the file exists before the compute runs. `cog_export` defers both products and hands them to
one `dask.compute`. Shared source blocks are retired once.

`export_lst_cog` and `export_qa_cog` still work standalone and still cost a pass each.
`tests/integration/test_cog.py` pins both directions: one call to `cog_export` is one pass,
two separate calls are two.

### Delete the coverage check and recover its numbers from the raster

`load_scenes` runs with `fail_on_error=False`, so a read failure fills a scene with nodata
rather than aborting the tile. Occasional fill is the point; mass fill is a broken run, and
a low median or a high zero fraction is what tells them apart. That check stays.

It no longer costs a pass. The windowed walk that computes the `STATISTICS_*` band tags over
the written QA intermediate now also accumulates a histogram of the per-pixel sum across the
twelve monthly bands, and `valid_coverage_obs_per_pixel` reports the same four numbers from
it — the same key, the same values, including the exact median as an order statistic. The
same restructuring took that walk from one pass per band to one pass total.

Deriving the numbers from the written raster moves the check after the write rather than
before it. That is acceptable: nothing acted on them programmatically, and reading the
shipped artifact is a stronger check than reading what we intended to ship.

## Consequences

- One pass over the native stack per tile, down from three. On the 300-scene `N40W075`
  sample a pass was roughly 180s, projecting to roughly 30 min at the production 2,930
  scenes; across 700 tiles each avoided pass is on the order of 350 VM-hours.
- Peak RSS rises about 25% on the synthetic tile (1.30 to 1.60 GB) because one compute now
  serves two writers. The floor terms are unchanged.
- `landsat-lst plan` reports different composite task counts. The rechunk is explicit now
  instead of implicit inside `quantile`, and the graph it counts is the graph that runs.
- The `coverage_check` phase is gone from `progress.PHASES`. Naming it is what made it
  measurable, and then deletable.
- QA `STATISTICS_VALID_PERCENT` is always 100 by construction (`nodata=None`, because zero
  observations is data). It cannot serve as the mass-fill check, and the issue text that
  proposed it as one was wrong. The LST band's `VALID_PERCENT` and the coverage line do that
  job.

## Alternatives considered

**One blockwise kernel producing both outputs.** A single `map_blocks` over the rechunked
stack, returning the P95 and the twelve counts stacked together, guarantees one consumer per
source block by construction. It was prototyped and it works. On the same synthetic tile it
reached 1.0x at the lowest peak of the three shapes, 1.30 GB, and took **1197s against 123s**
— roughly ten times the wall clock. Hand-rolling the twelve monthly masks and the quantile
per block gives up the optimized reductions that xarray and dask dispatch to, and no amount
of memory saved pays for a 10x slowdown on the phase this ADR exists to shorten.

It is also the riskier change numerically. `np.nanquantile` per block matches xarray's numpy
path exactly, while the current dask path differs from that path by ~7e-15, which rounds to
~4e-6 °C in float32. Harmless, but a shift for nothing: the shared rechunk reaches the same
1.0x with bit-identical output.

**`.persist()` the masked stack.** The whole five-year stack in memory. Not viable at
production geometry.

**Keep the coverage check and accept two passes.** Rejected. The information was already
being computed a second time by the exporter.

Related: #80, #77, #79.
