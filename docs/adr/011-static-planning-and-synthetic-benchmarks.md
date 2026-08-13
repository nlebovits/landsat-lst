# ADR-011: Plan tile cost statically, and benchmark on synthetic geometry

**Status:** Accepted
**Date:** 2026-08-13
**Authors:** @nlebovits

## Context

Every question we asked about pipeline cost cost a Coiled submission and twenty minutes.
Run `2021-2025-sample300-20260813T123249Z` on `N40W075` is the pattern. Sixty seconds in,
the heartbeat reported that the de-striping graph held 598,604 tasks for a 300-scene
sample. Nothing about that number depends on pixel values. It follows from array shape
and chunking, both of which are known before a run starts. We paid a cloud submission for
arithmetic.

The same held for memory. We spent a morning reconstructing a peak-RSS curve by hand from
heartbeat samples, watching it climb in steps to 78.6 GB at 49.8% complete. Ten validation
attempts produced zero completed tiles, and the blocker was never one configuration. It
was that no configuration could be priced without paying for it, so every lever got tested
serially at twenty minutes a turn.

Two gaps made this worse than it needed to be.

**Progress reporting counts tasks, not work.** `TileHeartbeat.phase_seconds` attributes
hours to phases and `GraphProgress` publishes a task fraction. Neither says *which* tasks.
As `progress.py` puts it: "Counting tasks is not counting work." Reading `4182/18600` is
identical whether the hour sits in `median-aggregate`, in `open_rasterio`, or in a rechunk
shuffle.

**The memory benchmark measured the wrong regime.** `scripts/measure_memory_scaling.py`
sweeps window length against a 0.25 degree AOI over Philadelphia, resting on the claim in
its own docstring that "the **ratios** are what transfer". They do not. Below roughly one
degree the whole time stack fits in RAM and dask never streams. A five degree tile streams
from its first block. The script measures a regime production never runs in, and its
ratios describe a different pipeline.

## Decision

Three layers of observability, cheapest first. Each answers a question the layer above it
cannot, and none of the first two touches the network.

### Layer 1: static graph inspection

`landsat_lst.profiling.graph_stats` reads a lazy collection's `__dask_graph__()` and
reports task count, layer count, and a breakdown by key prefix. `predict_peak` states a
memory **floor** for a configuration from three terms the pipeline demonstrably holds: the
per-block time stacks in flight (`threads * chunk**2 * scenes * itemsize`), the resident
per-pixel monthly climatology (`12 * height * width * itemsize`), and a process baseline.

`landsat-lst plan -t N40W075` builds both of a tile's graphs and prints them.
`--sweep` crosses chunk size with thread count. Graphs are built once per chunk size,
since task count follows from chunking and thread count changes only how many blocks are
in flight. A config sweep becomes a table instead of a day.

The floor is not a forecast, and the report says so. On the 300-scene N40W075 sample it
lands far below the 78.6 GB observed. A floor still earns its keep: a configuration that
cannot fit even the floor is disqualified for free.

### Layer 2: synthetic-geometry benchmarking

`scripts/synthetic_scaling.py` builds a stack from `dask.array.random` at production
chunking with a real time axis, then runs the real `compute_annual_composite` against it.
Same graph, same memory curve, no STAC query and no egress. It sweeps scene count in fresh
subprocesses, fits peak RSS and task count against scene count, and extrapolates to the
2,930 scenes a five-year land tile pulls.

Fresh subprocesses are load-bearing. `getrusage` reports a high-water mark for the whole
process, so a second configuration inside the first one's interpreter inherits its peak and
draws a flat curve whatever the truth is.

Spatial extent is a knob, defaulting to 8 x 8 production chunks rather than the full
18,000 squared tile. This is not the flaw the old script had. Peak memory is
`threads * chunk**2 * scenes * itemsize`, which does not grow with tile width once there
are more blocks than threads; wall clock does. The default keeps every regime-defining
property (production chunk edge, sixteen times more blocks than in-flight threads, real
time depth) and finishes in minutes.

`scripts/measure_memory_scaling.py` is deprecated in the same change. Two tools that
disagree about which regime matters are worse than one tool that is right.

### Layer 3: per-key profiling in real runs

`profiling.profile_compute` wraps the de-striping compute in `dask.diagnostics` and dumps a
summary to `_runs/{run_id}/{tile}.{label}.profile.json`, beside the heartbeat an operator
is already reading.

| Profiler | Answers |
|----------|---------|
| `Profiler` | Per-task start and stop by key, so an hour attributes to a task prefix |
| `ResourceProfiler` | The RSS curve we currently reconstruct by hand from heartbeats |
| `CacheProfiler` | Bytes held in memory, the direct answer to "why is RSS climbing" |

Gated on `settings.profile_dask`, default off. `CacheProfiler` is gated separately on
`settings.profile_dask_cache`, because it retains one record per task and the graph it
would instrument reached 598,604 of them on a run already near its memory ceiling.

Every write is best-effort, matching the existing rule that instrumentation never fails a
tile. Summaries are aggregated by key prefix and the resource curve is strided, so a
profile cannot outgrow what it profiles.

## Consequences

- Task count is knowable in seconds. A full production graph is still a lot of arithmetic:
  expect roughly half a minute and several GB of RSS at 2,930 scenes. That is the cost of
  an exact count rather than an estimate, and it beats twenty minutes by two orders of
  magnitude.
- `normalization.scene_offsets` is split. `offset_graph` builds the lazy pair and
  `scene_offsets` computes it, so a planner can inspect the graph without running it.
- `pipeline.TIME_CHUNK` is named rather than inlined. A synthetic stack chunked differently
  from the real one builds a different graph, and the whole value of planning against it is
  that it does not.
- Synthetic source layers are one task per block, matching a per-block read. What they
  cannot reproduce is the extra layers odc-stac wraps each band in, so a planned total runs
  slightly under a real one. Every reduction above the leaves is identical, and that is
  where the tasks live.
- The composite phase is planned with de-striping disabled, because
  `compute_annual_composite` computes offsets eagerly. Its scene count is therefore an
  upper bound: a real run has already discarded roughly 22% of scenes by then.
- Ratios measured on a sub-degree AOI do not transfer to a tile. Any future benchmark
  either runs at production chunking with more blocks than threads, or states plainly which
  regime it measures.
