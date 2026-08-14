# How far a real peak lands above the predicted floor

**Status:** Harness built, measurement not yet run. The verdict section below is empty on purpose.
**Date:** 2026-08-14
**Tracking:** [#94](https://github.com/nlebovits/landsat-lst/issues/94), [ADR-011](adr/011-static-planning-and-synthetic-benchmarks.md)

## The question

`landsat-lst plan` reports a memory **floor** in three terms: the concurrent per-block time stacks
(`threads * chunk**2 * scenes * itemsize`), the resident monthly climatology
(`12 * height * width * itemsize`), and a process baseline. The floor is honest arithmetic and it
earns its place, because a configuration that cannot fit even this is disqualified without
spending twenty minutes finding out.

What it is not is a forecast. On run `2021-2025-20260814T102049Z` (N40W075, 2021-2025, 2,930
scenes, offset factor 2 on a 9,000 squared grid, chunk 512, four threads) the floor came to about
17 GB. The tile OOMed on a 64 GiB VM at 46.5 GB and still climbing, 2% into the de-striping graph.
An earlier 300-scene sample peaked at 78.6 GB against a floor of a few GB.

The model is not naive about the part it covers: `stack_bytes` is precisely what a monthly median
holds when each block carries its whole time axis. What it does not model is the `groupby` rechunk
shuffle, the anomaly broadcast that materializes a second stack, and the spatial median holding
full planes. The size of that gap decides whether `plan` can be trusted at all.

## Three tiers, and why the middle one is not optional

Confusing the tiers is its own failure mode. Ten validation attempts on 2026-08-13 produced zero
completed tiles, at twenty minutes a turn, because every lever was tested serially in the cloud.
The failure was not spending money. It was spending time on the wrong tier.

| Tier | Instrument | Cost | Answers |
|---|---|---|---|
| Instant, local | `graph_stats`, `predict_peak`, `landsat-lst plan` | seconds, no data moves | task counts, the memory floor |
| Cloud, sampled | `landsat-lst benchmark --distributed` | ~20 min, well under a dollar | peak RSS at production geometry |
| Cloud, full window | `landsat-lst process --distributed` | hours, real money | the product |

The middle tier runs on a VM and not on the dev box for three reasons. The dev box carries 54 GB
against a 64 GiB VM, so the ceiling under test is unreachable. Synthetic data means the VM does no
I/O, so the sweep is minutes rather than the hours a real load takes. And the answer is about
production hardware, which is the only hardware whose peak RSS matters.

Only the graph-inspection tier runs locally, and it stays bounded there: an unbounded
`plan --sweep` has crashed the machine before. `landsat-lst benchmark` refuses more than 200 scenes
locally for the same reason, and `plan` carries `--max-tasks`.

## Running it

```bash
export LST_COILED_RETRIES=0        # a retry would destroy the evidence
landsat-lst benchmark --distributed
landsat-lst benchmark --fetch <run-id>
```

Budget about 25 minutes of compute plus VM start, and under a dollar. Three
points measured on the dev box at the sweep's own geometry (4096 squared, chunk
512, four threads) fit `5.9 + 0.876 * scenes` seconds, which puts the default
`(50, 100, 200, 400, 800)` sweep at 23 minutes:

| scenes | wall | peak RSS | floor | ratio |
|---:|---:|---:|---:|---:|
| 25 | 27.8s | 3.95 GB | 2.85 GB | 1.39 |
| 50 | 39.6s | 6.17 GB | 2.95 GB | 2.09 |
| 100 | 93.5s | 9.64 GB | 3.14 GB | 3.07 |

`SWEEP_JOB_TIMEOUT` is one hour rather than the six-hour tile ceiling, which
leaves better than 2x headroom over that fit. At $0.504/hr for `r6i.2xlarge` and
$0.768/hr for `m6i.4xlarge`, spot floor to on-demand fallback, 25 minutes costs
$0.06 to $0.32 and the full hour still lands under a dollar.

**These three points are a pre-flight expectation, not the finding.** They were
measured on a laptop with 54 GB against a 64 GiB VM, which is the substitution
this whole document exists to warn against. They suggest `growing_ratio` and a
peak near 85 GB at 2,930 scenes, consistent with the 46.5 GB and climbing that
the OOM was caught at. Suggestive is not measured. The verdict below stays empty
until a VM produces it.

### Watching it

`landsat-lst watch` will not work here. It lists `_runs/{run_id}/` and classifies every object as
a tile attempt, and a sweep is not a tile. Neither does the Coiled dashboard, which describes a
dask scheduler a batch task never registers with, nor `coiled logs`, which does not carry task
stdout. Poll `--fetch` instead:

```bash
watch -n 60 'landsat-lst benchmark --fetch <run-id>'
```

The VM republishes the whole object after every scene count, carrying `status` and `completed`, so
a partial read shows the points that have landed and names the ones still to run. It uploads its
own stdout and stderr to `_benchmarks/{run_id}/sweep.log` on the way out either way. Both together
are the only channel a batch task has, which is the same conclusion the tile path reached in issue
\#68 and ADR-014.

The sweep runs `landsat_lst.benchmarks.measure` once per scene count, each in a fresh subprocess:
`getrusage` reports a high-water mark for the life of a process, so a second configuration measured
inside the first one's interpreter would inherit its peak and draw a flat curve whatever the truth
was. Retries are pinned to zero regardless of `settings.coiled_retries`, because on a diagnostic
run the failure is the product.

## The three ways this can land

The interpretation is the deliverable, not the numbers. `sweep_report` returns one of these as its
`verdict`.

**`constant_ratio`** — the ratio holds across the sweep. Give `predict_peak` this correction factor
and `plan` becomes predictive rather than one-sided.

**`growing_ratio`** — the ratio grows with scene count. Something scales that the model treats as
fixed, which localizes the leak to the `groupby` shuffle or the anomaly broadcast, and is direct
evidence for [#93](https://github.com/nlebovits/landsat-lst/issues/93).

**`not_streaming`** — peak RSS did not move with scene count. The configuration is too small to
stream, the stack fits in RAM, and a line fitted through the process baseline describes the
interpreter rather than the pipeline. Per ADR-011 no projection is printed; raise `--blocks` or
`--scenes` rather than trusting the result.

## Verdict

Not yet measured. Run the sweep and record the table, the fitted slope, and the verdict here.

## What the regression tier catches meanwhile

`tests/benchmark/` runs on every nightly build at reduced geometry, asserting on shape rather than
magnitude. It cannot reproduce a production peak and does not try. Two numbers proved out while the
guards were being written, both at 24 scenes on a 1,024 squared grid with chunk 256 and two threads:

- Deleting the shared time rechunk from `_composite_graph` takes the composite from **828 tasks to
  1,326** and composite-only peak RSS from **308 MB to 842 MB**, a factor of 2.73. Both guards fail
  on it.
- The read tally stays at **1.0 passes either way**, because both consumers descend from the same
  source keys and within one `dask.compute` each key is produced once whatever is downstream. An
  earlier draft of the guard asserted the pass count, passed with the rechunk deleted, and would
  have shipped the regression it was written to catch. The pass count is the right guard for
  `cog_export`, where the two products are separate computes, and the wrong one here.
