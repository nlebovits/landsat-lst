# How far a real peak lands above the predicted floor

**Status:** Measured. Verdict `growing_ratio`; `predict_peak` is not correctable by a factor.
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
| Cloud, sampled | `landsat-lst benchmark --distributed` | ~12 min, a dime | peak RSS at production geometry |
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

Budget about twelve minutes and a dime. `SWEEP_JOB_TIMEOUT` is one hour rather than the six-hour
tile ceiling, which leaves 5x headroom; at $0.504/hr for `r6i.2xlarge` and $0.768/hr for
`m6i.4xlarge`, spot floor to on-demand fallback, even the full hour lands under a dollar.

A pre-flight estimate off the dev box put this at 23 minutes, from a fit of `5.9 + 0.876 * scenes`
seconds through 25, 50, and 100 scenes. The VM ran it in twelve, and its 800-scene point died
where the laptop's fit had no opinion at all. The estimate was useful for sizing the timeout and
worthless for anything else, which is the whole argument for the middle tier.

### Watching it

`--distributed` follows by default, so submitting and watching are one command. Output appends as
things happen, so scrollback keeps the whole run:

```
     50 scenes: running...
     50 scenes: 6.2 GB peak, 2.1x floor, 19,943 tasks, 0.7 min
    100 scenes: running...
    100 scenes: 9.6 GB peak, 3.1x floor, 35,782 tasks, 1.6 min
    200 scenes: running...
```

Ctrl-C detaches and leaves the VM running. Re-attach with
`landsat-lst benchmark --follow <run-id>`, or pass `--no-follow` at submit to get the run id back
and nothing else.

Nothing here is a true tail, and it cannot be. `landsat-lst watch` lists `_runs/{run_id}/` and
classifies every object there as a tile attempt, so it cannot see a sweep. The Coiled dashboard
describes a dask scheduler a batch task never registers with, and `coiled logs` does not carry
task stdout. The only channel is what the VM publishes, so it republishes the whole result object
when it starts each scene count and again when that point lands, and uploads its own stdout to
`_benchmarks/{run_id}/sweep.log` on exit either way. One line per transition is the real
resolution of the work: the largest point that completed ran for 3.2 minutes. Same conclusion the tile path
reached in issue \#68 and ADR-014.

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

## Verdict: `growing_ratio`

Run `scaling-20260814T140131Z`, one `r6i.2xlarge`-class VM, 4096 squared px at chunk 512 and
four threads, twelve minutes wall clock and under a dime.

| scenes | peak RSS | floor | ratio | offset tasks | wall |
|---:|---:|---:|---:|---:|---:|
| 50 | 6.4 GB | 2.9 GB | **2.16** | 19,943 | 33s |
| 100 | 10.0 GB | 3.1 GB | **3.18** | 35,782 | 50s |
| 200 | 16.3 GB | 3.5 GB | **4.63** | 83,069 | 96s |
| 400 | 27.5 GB | 4.3 GB | **6.37** | 132,435 | 193s |
| 800 | **OOM** | — | — | — | died at 337s, `exit -9` |

**`predict_peak` cannot be corrected with a factor.** The ratio nearly triples across an 8x span
of scene count, 2.16 to 6.37. There is no constant to multiply the floor by, so `plan` stays what
it already was: a disqualifier, not a forecast. A configuration that cannot fit the floor is ruled
out for free; one that fits tells you nothing.

**The growth is accelerating, not settling.** Peak rises by 1.57x, then 1.64x, then 1.68x per
doubling of scene count — exponents of 0.65, 0.71, 0.75. A model whose error grows with the axis
it is being extrapolated along is worse than no model at the far end.

**800 scenes killed the VM.** `exit -9` is SIGKILL at 337 seconds. That is 27% of the 2,930 scenes
a five-year land tile pulls, on the instance type production runs, and it could not finish. This
is the single most useful number the sweep produced, and it is a failure rather than a
measurement.

**A production tile needs somewhere between 123 GB and 179 GB.** The linear fit the tool prints
says 179; a power law through the top pair says 123. Both are extrapolations past a point that
OOMed, so treat the range as a lower bound on the problem rather than a target. Either way a
64 GiB VM is not close, which is exactly consistent with the production run that OOMed at 46.5 GB
and still climbing, 2% into the de-striping graph.

### What this means

The gap is not overhead to be trimmed. At 400 scenes the pipeline holds 6.4x its own floor, and
the multiple grows with the window. Something scales that the three-term model treats as fixed,
which points at the `groupby` rechunk shuffle or the anomaly broadcast materializing a second
stack — direct evidence for [#93](https://github.com/nlebovits/landsat-lst/issues/93).

It also means **#93 alone may not be sufficient**. Halving the offset pass takes a 123 GB tile to
roughly 62 GB, which is at the ceiling rather than under it.

### The configuration levers were already pulled, and did not hold

Worth stating plainly, because it is easy to remember this the other way round. Commit `7fda25c`
established that peak is roughly `threads * chunk**2 * scenes * 4`, that halving
`load_chunk_size` cuts that term, and that doing so measured **2.9x slower** locally through
per-chunk overhead — four times the graph nodes and four times the range requests. So the project
capped threads instead, deliberately, and chunk stayed at 512.

The memory half of that argument was arithmetic on the floor. The only thing measured was the
slowdown. And the tile that OOMed ran at **chunk 512 with four threads**: the cap was in effect,
and it died anyway, at 46.5 GB against a 17 GB floor.

That is not a wrong formula. `stack_bytes` is precisely what concurrent block stacks cost. It is
the wrong *dominant term* — at 400 scenes the pipeline holds 6.37x its floor, so the part the
levers control is a shrinking minority of the total.

**So the next measurement is a fork, not a confirmation.** Re-run this sweep at `--threads 1
--chunk 256`, twelve minutes and a dime:

- Ratio falls toward 1 → the excess is concurrent block stacks after all, the levers work, and
  #93 is an optimization.
- Ratio holds near 6.4x → the excess is the shuffle or the broadcast, both levers are dead ends,
  and #93 is structural work no setting can avoid.

Do this before starting #93. It costs a dime and it decides how much #93 has to accomplish.

The laptop pre-flight projected ~85 GB and `growing_ratio`. It called the verdict correctly and
understated the magnitude by a third to a half. That is the substitution this document warns
against, now measured rather than asserted.

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
