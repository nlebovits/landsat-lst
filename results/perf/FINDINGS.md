# Candidate 1: build the time-contiguous view already in `(latitude, longitude, time)`

Result: **accepted** on the wall-clock arm of the contract's criterion 4, with
bit-identical output on both fixtures.

Protocol: `docs/perf/composite-experiment-contract.md`. Reproduction:
`results/perf/REPRODUCE.md`. Raw rows: `composite-experiment.jsonl`, one object
per run. Derived table: `summary.json`.

## What changed

`_composite_graph` transposes the stack to put time last *before* the
single-time-chunk rechunk, so the concatenate ADR-013 already requires writes the
layout `nanquantile_last` wants. `groupby("time.month").sum()` then returns
`(latitude, longitude, month)` and is transposed back to the shipped
`(month, latitude, longitude)`.

## Measurement

Acceptance configuration: chunk 512, 4 threads, 300 scenes, median of 3 runs in
3 fresh subprocesses per arm, arms interleaved A/B/A/B/A/B.

| arm | fixture | wall s (median) | spread | peak VmHWM MB | composite tasks | native passes |
|---|---|---|---|---|---|---|
| baseline | S30W065 | 73.99 | 1.00x | 11,658 | 7,473 | 1.00 |
| treatment | S30W065 | 16.90 | 1.15x | 9,664 | 7,473 | 1.00 |
| baseline | N40W075 | 82.20 | 1.01x | 12,373 | 7,711 | 1.00 |
| treatment | N40W075 | 20.08 | 1.03x | 10,083 | 7,711 | 1.00 |

All four rows are `[retained-real]`: 300 real Landsat scenes per fixture, real
QA-driven NaN density, production chunk edge and time chunk, production dtype
chain. Grid is the fixtures' 2,250 squared, not production's 18,000 squared, and
the leaves are memmap pages rather than S3 reads.

| criterion (contract section 6) | S30W065 | N40W075 |
|---|---|---|
| 1. bit-identical `lst_p95`, `qa_count`, encoded uint16 | pass | pass |
| 2. `native_passes` stays 1.0 | pass | pass |
| 3. `composite_tasks` inside 1.4x | pass (1.00x) | pass (1.00x) |
| 4a. wall clock >= 1.50x | pass (4.38x) | pass (4.09x) |
| 4b. peak RSS >= 1.35x | not met (1.21x) | not met (1.23x) |

Criterion 4 is an OR and the wall arm clears it on both fixtures. The memory arm
does not, and no memory claim is made: the contract predicted "down or flat" and
that is what happened.

## Contention

The machine was **not** quiescent. Seven or eight other agents held jobs on the
same 16-core laptop throughout; `/proc/loadavg` captured on both sides of every
timed section ranged 9.52 to 15.61, median 13.97, and is recorded per run.

Two things make the rows usable anyway. The arms were **interleaved**, so drift
is common to both rather than an advantage for whichever ran second. And every
row's min-max spread came in at 1.00x-1.15x, inside the contract's 1.30x validity
limit, so no row is invalidated and stop condition 7 did not fire.

The speedup is nonetheless quoted from a contended machine.
`scripts/perf/rerun_wall.sh` recovers only that number on a quiet one, in about
fifteen minutes, without re-deriving anything else. Peak RSS, the native-pass
tally, the fused task count and the checksums are robust to CPU contention.

## What this cannot establish

Contract section 9 stands in full, and in particular: **this is not a production
speedup and the ratio must never be multiplied into a shard's wall clock.** E1
measures a real shard at 1.64-1.69 busy cores of 16, so production wall clock is
dominated by something the local tier has no instance of, and the compute
fraction of that wall clock is `[unknown]`. Section 7's own arithmetic already
shows the 42 s layout conversion cannot hold on an m6i.4xlarge: 36 blocks times
42.05 s would exceed the shard's entire measured 1,332.7 cpu-s. The mechanism is
real; its magnitude on production hardware is unmeasured.

## Deviations from the contract, recorded

1. **Python 3.14.3, not 3.12.2.** numpy 2.5.2, dask 2026.7.1 and xarray 2026.7.0
   match section 2 exactly, which is what the bit-exactness claim rests on, and
   both arms ran in one interpreter version.
2. **Machine not quiescent** (section 2, stop condition 7). Handled as above.
3. **8 threads was refused.** Section 3's launch bound uses E5's synthetic 3.9x
   multiplier over the per-block time stack; the first real-fixture baseline
   measured 9.6x (12,096 MB against a 1,258 MB stack). The guard uses the
   measured number, which projects 24.2 GB at 8 threads against the 20 GB cap.
   The 1- and 2-thread rows are in
   `composite-experiment-precorrection.jsonl`; they predate the per-run
   contention capture and are context, not evidence.
4. **`results/perf/` is committed**, where section 5 assumed it stays ignored.
5. **Two rows were discarded** after a `git stash` of mine raced the runner for
   `pipeline.py`. `summarize.py` now refuses any set whose arms do not each map
   to exactly one `pipeline.py` sha256, so that failure cannot pass silently.
