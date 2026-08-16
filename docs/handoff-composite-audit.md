# Handoff: audit and improve the native composite

You are picking up where the de-striping work left off. That work is merged
(PR #100, ADR-015) and closed. This document is the full context for the next
piece: the **native composite**, which is now the largest unexamined cost in the
pipeline.

Read this first, then `CLAUDE.md`, then the code pointers below. The measured
numbers here were produced between 2026-08-15 and 2026-08-16 and are cited so
you can check them rather than trust them.

---

## 1. The one-sentence problem

A production tile reads **3,797 GB** through the native composite and nobody has
ever measured how fast that goes, where the time is spent, or what it peaks at.

`18000² × 2930 scenes × 2 bands × 2 B = 3,797 GB`. That is **four times** the
949.3 GB the offset pass reads, and the offset pass alone took 15.75 hours per
tile.

## 2. What is already known, and how well

Provenance matters more than the numbers. Tag everything you produce the same
way: **[M]** measured, **[D]** derived from measurement, **[A]** assumed.

| fact | value | status |
|---|---|---|
| Native grid | 18000² (factor 1); offset grid is 9000² (factor 2) | **[M]** `geobox_for_bbox` |
| Native volume per tile | 3,797 GB | **[D]** |
| Native throughput, 1.5° window, 21 scenes | 13.87 MB/s decoded, 8.47 MB/s wire | **[M]** U4, 2026-08-15 |
| Native compression | wire/decoded 1.64 (native) vs 0.74 (factor 2) | **[M]** U4 |
| Bytes per scene, native vs factor 2 | decoded 3.99×, **wire only 1.80×** | **[M]** U4 |
| Native throughput at full 18000² footprint | — | **[A] never measured** |
| Composite peak RSS at production geometry | — | **[A] never measured** |
| Composite wall time per tile | ~23.5 h if it runs at phase B's 44.9 MB/s | **[A]** borrowed rate |

**The last three rows are your job.** Everything downstream of them — cost per
tile, 700-tile budget, worker count — currently rests on a rate borrowed from a
different pass at a different resolution.

### Why the wire ratio is the interesting one

Quadrupling pixels only 1.8×'d the bytes on the wire, because full-resolution
thermal data compresses well and the factor-2 overview does not. Native
*decodes* faster per byte than the coarse pass (13.87 vs 11.14 MB/s). Do not
assume native is proportionally slower just because it is 4× the pixels.

## 3. Code pointers

| what | where |
|---|---|
| Composite entry | `pipeline.compute_annual_composite` (`pipeline.py:327`) |
| The graph under audit | `pipeline._composite_graph` (`pipeline.py:423`) |
| Export, both products in one compute | `cog.cog_export` (`cog.py:423`), `export_lst_cog` (`411`), `export_qa_cog` (`417`) |
| Deferred writes | `cog._write_intermediates` (`cog.py:151`) |
| Coverage diagnostic | `cog._log_coverage` (`cog.py:394`) |
| Phase instrumentation | `timed_section("composite_graph")` at `pipeline.py:419` |

**ADR-013 governs this code and you must not break it.** `_composite_graph`
rechunks time to a single chunk before building either output, and `cog_export`
writes both intermediates in **one** `dask.compute`. Either change alone gives
back a full native pass: two differently-chunked consumers materialize every
source block twice, and peak went **10.88 GB against 1.30 GB** on a 4096²×120
synthetic tile when the shared rechunk was removed.
`tests/integration/test_cog.py` pins the pass count from both sides.

Note the subtlety recorded in `CLAUDE.md`: the **read tally does not catch the
`_composite_graph` regression** (both consumers descend from the same source
keys within one `dask.compute`), so pass count is the right guard for
`cog_export` and the wrong one here. An earlier draft asserted it, passed with
the fix deleted, and would have shipped the regression.

## 4. How the de-striping work was done, and what it cost to learn

Copy the method, not the conclusion.

**What worked:**

- **Inspect the graph layer by layer before theorizing.** The de-striping root
  cause — two medians reducing along orthogonal axes, producing two rechunk
  layers of 388,800 tasks each — came from direct layer inspection at production
  geometry, not from reading the code.
- **Measure the graph without data.** `dask.array.zeros` at production shape and
  chunking answers construction-cost questions for free. See
  `results/batch1-investigation/batch4/graph_probe_child.py`. It closed a
  suspected bottleneck in minutes at zero cost (13.5 s of a 15.75 h tile).
- **Every local measurement runs in a fresh guarded subprocess.** `RLIMIT_AS`
  30 GB, a watchdog summing VmRSS over the process **tree** at 26 GB, a timeout,
  and outcomes recorded rather than raised. `getrusage` is a high-water mark, so
  a second config in the first's interpreter draws a flat curve whatever the
  truth is. An unguarded build has taken this desktop down twice.
- **Repeat a control arm last.** A `baseline_repeat` arm revealed that an entire
  apparent GDAL speedup was first-run warming. A trailing 1-VM group proved the
  scaling curve was not drift.
- **Tag provenance relentlessly.** The single most expensive error in the
  de-striping work was using 4500² instead of 9000² for the offset grid, which
  made every volume, wall-clock and cost number **4× optimistic** for days. It
  was caught by telemetry (`blocks: 1/324` implies 9000², not 4500²), not by
  review. Derive geometry from `geobox_for_bbox`, never from memory.

**What wasted time and money:**

- **A benchmark with no instrumentation.** A 7.81-hour run died leaving an empty
  log and a null exit code, cost 62.56 credits, and established nothing. It went
  through the interactive `landsat-lst offsets` CLI, which carries no heartbeat
  and uploads no log. **Use the instrumented path.**
- **Assuming the cheap explanation.** No OOM-killer and no spot-reclaim message
  appears in that run's system log. Both cheap explanations were unsupported.
- **Wrapping a command in `/usr/bin/time`**, which does not exist in the Coiled
  container. Killed a run in 0.3 s while the shell pipeline reported exit 0.
- **Writing to the wrong S3 prefix.** `S3Storage._full_key` prepends
  `nlebovits/landsat-lst/`; raw boto3 to a bucket-root prefix gets `AccessDenied`.
  Use `get_storage()`.
- **Concurrency experiments whose tasks never overlapped.** A 4-VM group scored
  `overlap_fraction 0.0` — one task finished before another started, so its
  aggregate described sequential execution. Always report the concurrent
  interval, never just the union.

## 5. Instrumentation you inherit — use it

- `progress.TileHeartbeat` republishes phase, unit counts, phase seconds, and an
  **RSS series** to `_runs/{run_id}/{tile}.{attempt}.json` every 60 s.
- `progress.capture_task_log` tees stdout/stderr at the file-descriptor level
  (so GDAL and rasterio are caught) and uploads on exit either way.
- `runs.resolve_attempt` numbers artifacts; resolve it **once** per process.
- `landsat-lst watch <run-id>` and `landsat-lst explain <run-id> <tile>` render
  both.
- A working instrumented harness: `results/batch1-investigation/batch4/diag_child.py`
  plus `diag_submit.py`. Adapt it; do not rebuild it.
- **Phases are split finer than the work.** `composite_graph` is currently one
  `timed_section` spanning graph construction, the P95, the `qa_count`
  climatology, and the handoff to export. That is exactly the granularity
  problem ADR-013 already had to solve once. Expect to split it before you can
  attribute anything.

## 6. Suggested shape of the work

Not a plan to follow blindly — the de-striping investigation changed direction
three times as evidence arrived.

1. **Characterize before optimizing.** Where does the composite spend time,
   memory and I/O at 18000²? Sample a real tile with `--max-scenes` through the
   instrumented path. The offset cache (ADR-012) means you can warm offsets once
   and iterate on the composite alone.
2. **Establish the native rate at full footprint.** U4 measured a 1.5° window.
   The offset pass ran 3× faster at full tile footprint than at the small
   window, so do not extrapolate.
3. **Find the bottleneck, then make the smallest change that addresses it.** The
   de-striping answer was a structural split, not a framework. `quantile` needs
   the whole time series per pixel, which is a real constraint — check whether
   an approximate or streaming P95 is even admissible against the accuracy gates
   before assuming it is available.
4. **Guard whatever you change** with tests on the tiers that run: integration
   on every CI build, benchmark nightly. See `tests/benchmark/test_bounded_units.py`
   for the pattern — wide bands, sized against a known regression.

## 7. Standing constraints

- **Never use Earth Search locally** (hook-enforced); Planetary Computer for
  local, Earth Search on AWS.
- **Never `git commit --no-verify`.**
- **Coiled Batch, never Coiled Functions.** One VM per tile (ADR-010).
- `dask_max_threads` is now **4** and `coiled_job_timeout` **24 hours**, both set
  from measurement in PR #100. If the composite behaves differently, re-measure
  rather than assume they transfer.
- Accuracy gates (issue #93: max |Δ| ≤ 0.5 °C, zero keep/reject flips) are never
  relaxed to make an approach pass.
- Ask before spending: the Coiled quota is a real, finite budget and was
  exhausted once during the de-striping work.

## 8. Open assumptions inherited from de-striping

These are not yours to close, but they bound any end-to-end projection you make:

| | status |
|---|---|
| Phase A holds 26.7 MB/s at 20× larger per-block reads | **[A]** |
| Per-VM throughput beyond 8 concurrent VMs | **[A]**; flat 1→8 **[M]** |
| C1 bit-exactness at 2,930 scenes / factor 2 | **[A]**; **[M]** at 300 / factor 8 |
| No intra-tile checkpoint exists | **[M]** — a failure costs the whole tile |

## 9. Where the evidence lives

- `docs/adr/015-bounded-work-unit-offsets.md` — the de-striping decision and its
  measured evidence.
- `docs/adr/013-single-native-pass.md` — **read before touching the composite.**
- `docs/adr/012-cached-scene-offsets.md` — how to avoid paying for offsets while
  iterating on the composite.
- `docs/adr/010-coiled-batch-for-distributed-runs.md`, `014-run-self-explanation.md`.
- `results/batch1-investigation/batch4/` — every harness and result from the
  de-striping work: E1–E4, the GDAL sweep, the calibration, both scaling runs,
  the graph probe, and the thread benchmark.
- `docs/findings-memory-model.md`, `docs/findings-offset-subsampling.md`,
  `docs/findings-cloud-cover-filter.md`.
- PR #100 for the diff and its review notes.
