# Pre-registered experiment contract: composite performance, local tier

**Status:** Pre-registered. Written before any change to `_composite_graph`.
**Date:** 2026-09-02
**Branch:** `perf/composite-investigation` (base `c84448b`, `origin/main`)
**Scope:** Make the existing 30 m composite faster and/or less memory-intensive with
**zero** change to scientific output. 1/3600-degree, 18,000 x 18,000 output stays.
No cloud run of any kind is authorised by this document.

Out of scope and not to be touched: the offset estimator, offsets caching, rejection
rules, QA masking, solar-day fusion, the P95 definition, `qa_count` semantics, GED
policy, the land mask, encoding, scene population, fleet consolidation, de-striping,
and PR #121's 100 m architecture.

---

## Evidence labels

Every numerical statement below carries exactly one:

| label | meaning |
|---|---|
| `[retained-real]` | measured on representative retained real data |
| `[synthetic]` | measured, but on generated arrays — real shapes, fake pixels |
| `[structural]` | a graph/layout property, no data movement or exact by construction |
| `[projection]` | derived from a measured rate, not itself measured |
| `[assumption]` | believed, not measured |
| `[unknown]` | named because it matters and nobody has measured it |

**Synthetic wall time is never production evidence.** It bounds a compute-side ratio
and nothing else.

---

## 1. Retained-evidence audit

### E1 — `results/probe/composite_packing.json`

Real production composite shards on S30W065, 1,031 solar-day steps, chunk 512, real
S3 reads, m6i.4xlarge and r6i.4xlarge, 2026-08-22.

- **Reproducible locally today: no.** Needs Coiled, AWS credentials, and Earth Search.
  It is a frozen artifact; read it, never re-run it under this contract.
- **Establishes** `[retained-real]`:
  - Per-VM decoded throughput is ~45-47 MB/s and is **invariant to concurrency**:
    46.79 (k=1, 16 threads), 46.96 (k=2, 8 threads each), 44.98 (k=3, 5 threads each).
    Two children each decoding 38.0 GB took 1,618.7 s against one child's 812.2 s —
    packing is exactly serial, so intra-VM packing buys nothing.
  - `cpu_cores_busy` 1.64-1.69 of 16 at k=1, from `cpu_s` 1,332.7 over 807.8 s wall.
  - Peak child `VmHWM` 33.21-39.40 GB against a `predict_peak` floor of
    16 x 512² x 1031 x 4 B = 17.3 GB plus 0.44 GB climatology `[structural]` — a ~2x gap
    the three-term floor model does not explain.
  - Packing OOM-kills: `min_headroom_frac` 0.018 at k=3 and 0.006 at k=4, one signal-9
    child in each.
- **Cannot establish:** what the idle 14 threads are blocked on; whether any code change
  moves the number; anything about a tile other than S30W065.

### E2 — `results/probe/io_ladder_v{1,2,3}.json`, `composite_rate_m6i4xl.json`

Cloud read ladders, 12-24 scenes, 2026-08-21.

- **Reproducible locally today: no.** Same reason as E1.
- **Establishes** `[retained-real]`: request size, not concurrency, is the read lever.
  chunk 256 is flat at 11.77-24.32 MB/s across 4/16/32/64/128/256 io threads; chunk 512
  reads 49.97-70.47 MB/s at 8-16 threads; chunk 1024 reads 94.60 MB/s at 16 threads and
  **158.39 MB/s at 8 threads** — the best arm in any ladder. `cpu_cores_busy` on those
  arms is 2.11-3.92, still a small fraction of 16.
- **Cannot establish:** that the ladder rates survive to 1,031-scene depth. They do not
  fully: the real composite at chunk 512 (E1) got 46 MB/s where the 24-scene ladder at
  chunk 512 got 70. Depth costs something the ladder never saw.

### E3 — `results/fixtures/{N40W075,S30W065}_2021-2025_n300_f8`

300 real Landsat scenes each, two `uint16` bands, uncompressed `.npy`, ~6.08 GB per
fixture, memory-mapped by `landsat_lst.fixture.load_fixture` at `TIME_CHUNK = 10` and
`settings.load_chunk_size`.

- **Reproducible locally today: yes, with no network.** This is the only real-pixel
  evidence this contract can generate.
- **Establishes:** value equivalence on real pixels, with real QA-driven NaN density and
  real per-scene coverage; per-block compute behaviour at the production chunk edge and
  time chunk.
- **Cannot establish, and this is the load-bearing caveat:**
  - **The grid is coarse.** Offset factor 8 gives 2,250² per tile against production's
    18,000². 25 blocks at chunk 512, against 36 per production shard band and 1,225 per
    tile.
  - **The window is shallow.** 300 scenes against ~1,031 for S30W065's real window and
    ~2,930 for a full production tile. The dominant memory term is linear in scenes, so a
    fixture block holds 300 MB where a production block holds 1.03-3.01 GB. **Absolute GB
    numbers do not transfer. Ratios at fixed geometry do.**
  - **There is no read path.** A memmap leaf is a page-cache hit; a production leaf is
    odc-stac -> GDAL -> DEFLATE -> S3. The fixture pays no decode CPU and no network
    latency, which is precisely where E1 says the wall clock goes.

### E4 — ADR-013's table and `tests/benchmark/`

`[synthetic]`, `dask.array.random`, 4096² x 120, chunk 512, 4 threads: 3.0x passes at
1.30 GB before; 1.0x at 10.88 GB with fused writes and no shared rechunk; 1.0x at
1.60 GB after. `results/benchmark/trend.jsonl` holds the CI series.

- **Reproducible locally today: yes**, via `landsat_lst.benchmarks.measure`.
- **Establishes:** the composite already costs **1.0** native pass. The
  "redundant passes" candidate area is closed unless the tally regresses.
- **Cannot establish:** any wall-clock claim. Its own text says so: a synthetic "read" is
  CPU-bound generation, not network I/O.

### E5 — probes run while writing this contract (2026-09-02, this laptop)

AMD Ryzen 7 7840HS, 16 logical cores, 54 GiB RAM, Python 3.12.2, numpy 2.5.2,
dask 2026.7.1, xarray 2026.7.0. **Single runs on a contended machine** — treated as
direction-finding, not as the baseline. The baseline is defined in §2 and must be
re-measured.

- P1 `[structural]`: `xr.apply_ufunc(nanquantile_last, ..., input_core_dims=[["time"]])`
  hands the kernel a **C-contiguous `(latitude, longitude, time)` block**. Verified by
  spying on the kernel through `_composite_graph` and printing shape, strides, and
  `flags['C_CONTIGUOUS']`. So a full per-block layout conversion of the time stack is
  materialized between the `chunk({"time": -1})` rechunk and the kernel.
- P2 `[synthetic]`, float32, C=512, T=1031, single-threaded, `VmHWM` reset between
  phases: the layout conversion costs **42.05 s**; `nanquantile_last` on its output costs
  **2.17 s**. The transpose is **19.4x** the kernel it feeds. At T=300 it is 12.65 s
  against 0.51 s.
- P3 `[synthetic]`: `_composite_graph(...).compute()` with the source pre-transposed to
  `(latitude, longitude, time)` before the rechunk, against the shipped form, identical
  checksums throughout:

  | geometry | current | pre-transposed | speedup | peak VmHWM |
  |---|---|---|---|---|
  | C=512, T=1031, 1 block, 1 thread | 44.4 s | 20.0 s | 2.22x | 3,859 -> 4,040 MB |
  | C=512, T=300, 4 blocks, 2 threads † | 34.3 s | 10.2 s | 3.36x | 2,267 -> 1,732 MB |
  | C=512, T=300, 4 blocks, 4 threads | 15.6 s | 5.8 s | 2.69x | 3,365 -> 3,052 MB |

  † taken while a second job held the machine; the 2-thread row is contended and is shown
  only to bracket the range. **No thread-scaling conclusion is drawn from this table** —
  the contended row makes the shipped form's 2-to-4-thread ratio unreadable. Thread scaling
  is a recorded metric in §5, not an established fact.
- P4 `[structural]`, small real-shaped synthetic, exact comparison: `lst_p95` and
  `qa_count` are **bit-identical** across spatial chunk edges 64 / 128 / 256, and
  **bit-identical** between the shipped form and the pre-transposed form.
- P5 `[structural]`: the stack is float32 end to end — `apply_qa_mask`'s `.where` promotes
  `uint16` to float32 (itemsize <= 2), and NEP 50 weak promotion keeps
  `convert_to_celsius` in float32. `predict_peak`'s `itemsize=4` is correct.

### Audit verdict

The wall-clock question and the memory question have **different** evidence bases and must
not be mixed. E1/E2 are the only evidence about production wall clock, they are frozen,
and they say the shard is not CPU-bound in the cloud. E3/E5 are the only evidence this
contract can generate, and they can only speak to the compute side. Any claim this
experiment makes about production is therefore a `[projection]` conditioned on an
`[unknown]`: the compute fraction of a real shard's wall clock.

---

## 2. Exact baseline

- **Commit:** `c84448b` (`origin/main`), the merge-base of `perf/composite-investigation`.
  Record `git rev-parse HEAD` in every result row.
- **Worktree:** `.claude/worktrees/perf-integration`. Never write to the shared checkout.
- **Environment:** `uv sync --frozen` against the committed `uv.lock`. Recorded and
  asserted: Python 3.12.2, numpy 2.5.2, dask 2026.7.1, xarray 2026.7.0. A different numpy
  invalidates every bit-exactness claim and the run must be re-baselined.
- **Settings:** `settings.load_chunk_size = 512` (production composite chunk),
  `pipeline.TIME_CHUNK == 10` asserted at start-up exactly as `tests/benchmark/conftest.py`
  does, `settings.destripe` irrelevant because offsets are pinned (§3), storage backend
  local, no network reachable during a run.
- **Scheduler:** `dask.config.set(scheduler="threads", num_workers=<threads>)`. No
  distributed scheduler, no cluster.
- **Machine:** record `model name` from `/proc/cpuinfo`, logical core count, and
  `MemTotal`. The numbers in E5 come from an AMD Ryzen 7 7840HS laptop; a different
  machine changes every absolute number and must re-run the baseline arm.
- **Quiescence:** no other agent's job running. E5's numbers were taken under contention
  and are explicitly not the baseline.

---

## 3. Representative local workload

**Fixture:** `results/fixtures/S30W065_2021-2025_n300_f8` for the acceptance measurement
(it is the tile E1 was probed on and the shard work's acceptance tile).
`results/fixtures/N40W075_2021-2025_n300_f8` as the confirmatory second tile — every
acceptance claim must hold on both.

**Path under test**, built once and reused, so that only `_composite_graph` varies:

```
load_fixture(spec)                       # memmap, TIME_CHUNK=10, chunk 512
  -> apply_qa_mask(...)["lwir11"]
  -> convert_to_celsius(...)
  -> debias_with_offsets(lst, *offsets)  # offsets PINNED, see below
  -> _composite_graph(lst)
  -> dask.compute(...)                   # the timed section
```

**Offsets are pinned, not estimated.** Compute the offset vector for each fixture **once**,
write it to `results/perf/offsets-<tile>.json`, and load it on every subsequent run. The
estimator is out of scope and its own chunk sensitivity would confound the composite
measurement. Record the file's sha256 in every result row.

**Configuration:**

| lever | value | why |
|---|---|---|
| chunk | 512 | `settings.shard_composite_chunk`, production's composite chunk |
| TIME_CHUNK | 10 | production; asserted |
| threads | 4 (acceptance point) | 25 blocks > 4 threads, so dask streams |
| threads sweep | 1, 2, 4, 8 | recorded, not an acceptance point |
| scenes | 300 | all the fixture holds |
| dtype | float32 | `[structural]`, P5 |

**Bounding, per the local-run rule.** Per-block time stack = 512² x 300 x 4 B = 300 MB.
E5 measured a per-thread multiplier of 2.6-3.9x over that. At 4 threads the projected
working set is 3.1-4.7 GB `[projection]`, and at 8 threads 6.2-9.4 GB. **Hard cap: refuse
to launch any configuration whose projected peak exceeds 20 GB.** Compute the projection
before every launch; never run a sweep that has not been bounded first.

**Why it is representative:** production chunk edge, production time chunk, production
dtype chain, real Landsat radiometry, real QA-driven NaN density and per-scene coverage
holes, and more blocks than threads so the scheduler streams rather than holding the stack.

**Where it diverges from production, explicitly:**

1. 2,250² not 18,000² — 25 blocks, not 36 per shard band or 1,225 per tile.
2. 300 scenes not ~1,031 (S30W065's real window) or ~2,930 (a full tile). The dominant
   memory term is linear in scenes; fixture blocks are 3.4x-9.8x smaller.
3. Memmap leaves, not odc-stac/GDAL/S3 leaves. **No decode CPU and no network latency.**
4. No land mask and no GED gap mask. Both are output-side elementwise `.where` on the
   2-D products and neither touches the native stack, so they do not change the graph
   under test — but their absence is a deviation and is recorded as one.
5. A single band-equivalent, not a shard: no upload, no COG write, no `cog_export`.
   ADR-013's export property is separately pinned by `tests/integration/test_cog.py` and
   `tests/benchmark/test_native_passes.py` and is not re-litigated here.

---

## 4. Correctness comparison

**Ground truth.** Run the baseline (§2) once per fixture, and store `lst_p95` (float32)
and `qa_count` (uint8) as `.npy` under `results/perf/truth-<tile>-<sha>.npy`, together
with the sha256 of each array's `tobytes()`.

**Assertion — exact, no tolerance:**

```python
assert np.array_equal(new["lst_p95"].values, truth_lst, equal_nan=True)
assert np.array_equal(new["qa_count"].values, truth_qa)
assert new["lst_p95"].dtype == np.float32 and new["qa_count"].dtype == np.uint8
```

plus the encoded product, because that is what a user receives:

```python
assert np.array_equal(encode_lst_uint16(new["lst_p95"]).values,
                      encode_lst_uint16(truth_lst_da).values)
```

**Why exact and not `allclose`.** `tests/unit/test_kernels.py` states the standing rule:
these kernels "replace numpy's slowest paths … The replacement is admissible only because
it changes no bit of output; every test here asserts exact equality, never closeness."
Both candidates below are layout and blocking changes, not arithmetic changes, so the
same standard applies and **the tolerance is zero.** The float comparison is the binding
one: a difference below the 0.01 °C DN step of `encoding.LST_SCALE` would be invisible in
the encoded product but must still fail.

**How this comparison could fail if a change were wrong — it is not circular:**

- Candidate 1 changes which axis the sort sees. A wrong `moveaxis`, or a dimension order
  that swaps `latitude` with `longitude`, produces quantiles drawn from the wrong axis or
  a transposed image. Both differ from truth at essentially every pixel; `array_equal`
  over 2,250² float32 values catches either immediately.
- Candidate 2 changes block boundaries. If any reduction in the composite were secretly
  chunk-dependent — a partial-sum ordering in the `groupby`, an off-by-one at a block
  edge, a month bucket split across chunks — the arrays differ **at or near block
  boundaries**, which the exact comparison localizes. P4 shows the shipped form is
  chunk-invariant today; the test's job is to detect a change that breaks that.
- NaN placement is compared, not just finite values (`equal_nan=True` on an exact
  comparison still requires NaNs in the same positions). A change that silently converted
  a nodata sentinel into a finite value, or lost the `LST_MIN_TRUSTED_C` anomaly gate,
  fails.
- `qa_count` is compared independently of `lst_p95`, so a change that fixed one product
  while corrupting the other cannot pass.

There is no epsilon in which an error can hide. A candidate that cannot pass this exactly
is rejected, not re-tuned.

---

## 5. Metrics

One JSON object per run, appended to `results/perf/composite-experiment.jsonl`
(`results/` is gitignored; that is deliberate — this is evidence, not product).

| metric | how it is captured | tool |
|---|---|---|
| `wall_s` | `time.monotonic()` around **`dask.compute` only** | child program |
| `graph_build_s` | separate `time.monotonic()` around graph construction, which computes nothing and must not be conflated with the timed section | child program |
| `peak_rss_mb` | `VmHWM` from `/proc/self/status`, `getrusage(RUSAGE_SELF).ru_maxrss` as the no-procfs fallback — verbatim `landsat_lst.benchmarks._child_source._peak_mb` | **a fresh subprocess per configuration** |
| `cpu_s`, `cores_busy` | `getrusage` utime+stime; `cores_busy = cpu_s / wall_s` — the quantity E1 reports as 1.64-1.69 of 16 | child program |
| `source_reads`, `source_blocks`, `native_passes` | `map_blocks(_tally)` on the source array counting **block executions**; `native_passes = source_reads / source_blocks` | the `benchmarks`-style tally |
| `composite_tasks` | `profiling.graph_stats(composite, optimize=True).tasks` — the **fused** count, never the raw one | `landsat_lst.profiling` |
| `floor_mb` | `profiling.predict_peak(...).total_bytes` | `landsat_lst.profiling` |
| `lst_sha256`, `qa_sha256` | `hashlib.sha256(arr.tobytes()).hexdigest()` | child program |
| provenance | git sha, numpy/dask/xarray versions, cpu model, `MemTotal`, fixture name, offsets-file sha256, chunk, threads, scenes | child program |

**The fresh-subprocess rule is not optional.** `getrusage` reports a high-water mark for
the life of a process, so a second configuration measured inside the first one's
interpreter inherits its peak and draws a flat curve whatever the truth is. `VmHWM` comes
from the `mm` that `execve` created and is the child's own truth; a child forked from a fat
parent still reports the parent's `ru_maxrss`, which is why `VmHWM` is primary.

**Repetition.** Every configuration runs **3 times in 3 fresh subprocesses**. Report the
median and the min-max spread. A spread above 1.3x on `wall_s` invalidates that
configuration's row — re-run on a quiescent machine.

**Counting task keys instead of block executions does not work.** Fusion renames keys and
the count then silently reads zero. This is ADR-013's finding and it stands.

---

## 6. Acceptance threshold — decided before measurement

A candidate is **accepted** only if, on **both** fixtures, at the acceptance
configuration (chunk 512, 4 threads, 300 scenes), median of 3:

1. **Output is bit-identical** to truth, on `lst_p95`, `qa_count`, and the encoded uint16
   product. Non-negotiable, and checked first.
2. **`native_passes` stays at 1.0** (`pytest.approx`), preserving ADR-013.
3. **`composite_tasks` stays inside 1.4x of the baseline count**, matching
   `tests/benchmark/test_graph_size.COMPOSITE_BAND`.
4. **AND at least one of:**
   - **wall clock: >= 1.50x speedup** of the timed section, or
   - **peak RSS: >= 1.35x reduction** at identical geometry.

**Why 1.50x on wall.** `tests/benchmark/test_graph_size.COMPOSITE_BAND = 1.4` is the
smallest composite move this repo already treats as build-failing — it was sized against a
known 1.60x structural regression (deleting the shared rechunk took the composite from 828
to 1,326 tasks). A change under 1.4x is inside the band the repo has decided not to act on.
1.50x clears that band with margin for the 1.3x run-to-run spread this contract tolerates.

**Why 1.35x on memory.** `tests/benchmark/test_bounded_units.MEMORY_BAND = 1.35` is the
tightest memory band in the repo, chosen because the property under it is that memory does
not grow at all and the band only covers allocator noise. A memory claim smaller than the
noise band the repo already accepts is not a claim.

**A candidate that is bit-identical and passes 1-3 but clears neither threshold in 4 is
rejected and written up as a negative result.** It is not re-tuned until it passes.

---

## 7. Candidates — two, ranked

Both were selected from the evidence above, not from a menu. Both are locally falsifiable.

### Candidate 1 (rank 1) — build the time-contiguous view already in `(latitude, longitude, time)` order

**Mechanism.** `_composite_graph` rechunks a `(time, latitude, longitude)` array to
`{"time": -1}`, and `xr.apply_ufunc(..., input_core_dims=[["time"]])` then requires
`(latitude, longitude, time)`. P1 `[structural]` verifies the kernel receives a
**C-contiguous** block, so a full per-block layout conversion of the whole time stack is
materialized between the rechunk and the kernel: a strided gather with a
`chunk² x 4 B` stride, which is cache- and TLB-hostile. Transposing **before** the
rechunk makes the concatenate that ADR-013 already requires write directly into the order
the kernel wants; the reorders that replace it operate on `TIME_CHUNK`-sized slabs small
enough to stay in cache. `qa_count` comes out `(latitude, longitude, month)` and is
transposed back — 12 x H x W uint8, negligible.

**Why the evidence points here.** P2 `[synthetic]`: at production depth (C=512, T=1031)
the conversion costs 42.05 s against the kernel's 2.17 s — **19.4x** the work it feeds.
P3 `[synthetic]`: removing it gave 2.22x-3.36x end-to-end with identical checksums. It is
also a candidate explanation for part of the ~2x gap between E1's 33-39 GB measured peak and
its 17.7 GB floor, since the conversion holds a second full-block allocation per in-flight
block. Whether it explains any of E1's 1.64-1.69 busy cores is `[unknown]` and this
experiment cannot settle it.

**Predicted effect and direction.** Timed-section wall clock **down** by 2x-3x on the
representative workload `[projection]` from P3. Peak RSS **down or flat** — P3 saw
3,365 -> 3,052 MB at 4 threads but 3,859 -> 4,040 MB at 1 thread, so memory is not the
claim. Output bit-identical `[structural]`, P4. `composite_tasks` up by at most one layer,
likely fused away.

**The single falsifying measurement.** Median of 3 fresh-subprocess runs of the
representative workload, both fixtures, chunk 512 / 4 threads: **speedup below 1.50x
falsifies it.** (Any bit difference fails it outright, before speed is considered.)

**Residual risk.** P2 and P3 are `[synthetic]`, on a Ryzen laptop. A server's memory
subsystem may make the strided copy far cheaper — indeed 36 blocks x 42.05 s = 1,512 cpu-s
would exceed E1's entire measured 1,332.7 cpu-s for the shard, so the conversion **cannot**
cost 42 s on an m6i.4xlarge `[structural]`. The mechanism is real; its magnitude on
production hardware is `[unknown]`. This is why the acceptance measurement is on the real
fixture and why §9 forbids projecting the ratio into a production wall clock.

### Candidate 2 (rank 2) — raise the composite chunk edge to 1024 and pay for it with threads

**Mechanism.** The composite's per-task working set is `chunk² x scenes x itemsize`,
forced by the single-time-chunk rechunk. `settings.shard_composite_chunk`'s own docstring
names 1024 as infeasible: 4.32 GB per task at 1,031 scenes, x 16 threads = 69 GB on a
64 GiB VM. But E1 `[retained-real]` shows those 16 threads buy no throughput — decoded
MB/s is invariant across k=1/16t, k=2/8t, k=3/5t at 1.64-1.69 busy cores — and E2
`[retained-real]` shows chunk 1024 reads 94.6-158.4 MB/s against chunk 512's 50.0-70.5.
So trade threads for chunk: 8 threads x 4.32 GB = 34.6 GB, which is what the shipped
chunk-512 / 16-thread configuration **already** peaks at (33.2-39.4 GB, E1).

**Why the evidence points here.** It is the only lever E2 identifies, and E1 supplies the
budget to pay for it. It composes with candidate 1: any per-block transient candidate 1
removes is multiplied by `chunk²`, so landing 1 first widens the room for 2.

**Predicted effect and direction.** Production decoded rate **up** toward E2's chunk-1024
band, i.e. `R_COMPOSITE_MB_S` from 45.5 toward 90-150 `[projection]`, halving the composite
stage's budget in `landsat_lst.budgets`. Peak RSS **flat** at (chunk 1024, 8 threads)
relative to (chunk 512, 16 threads). Output bit-identical `[structural]`, P4 — the P95 sees
the whole time axis in one block whatever the spatial edge, and the `groupby` sum order
along time is unchanged.

**The single falsifying measurement.** On the representative workload: **peak VmHWM at
(chunk 1024, threads/2) exceeds peak VmHWM at (chunk 512, threads) by more than 1.0x.**
The entire trade is that the memory comes out even; if it does not, the chunk cannot be
raised and the candidate dies locally. Run it at (512, 8) vs (1024, 4) — projected 6.2-9.4
GB and 9.9-15.0 GB respectively `[projection]`, both inside the 20 GB cap. Do **not** run
(1024, 8) locally at 300 scenes without re-bounding.

**Honest limitation.** Candidate 2's *payoff* is an I/O effect and the local tier has no
I/O. Locally it can only be **falsified**, never confirmed. Surviving falsification means
"escalate to one cloud probe arm", which this contract does not authorize. Do not claim a
speedup for candidate 2 from any local number.

### Explicitly considered and rejected before measurement

- **Redundant passes.** Closed by ADR-013 and E4: the composite is already 1.0 pass. The
  tally stays as a guard, not as a candidate.
- **Intra-VM worker packing.** Falsified by E1 `[retained-real]` before this contract
  existed: aggregate decoded MB/s is flat across k, and k=3 and k=4 both OOM-killed a
  child at `min_headroom_frac` 0.018 and 0.006.
- **A faster percentile kernel.** Already done in `kernels.py`; P2 measures it at 2.17 s
  against a 42.05 s neighbour. Re-optimizing a 5% term is not on the table.
- **Sorting along axis 0 to skip the transpose.** Measured `[synthetic]` and rejected:
  25.93 s against 12.10 s at C=512/T=300, identical peak. Sorting the slowest axis is worse
  than the copy that avoids it. This is why candidate 1 moves the copy rather than deleting
  it.

---

## 8. Stop conditions

The implementer stops, reports, and changes nothing further when any of these holds:

1. **Any bit difference** appears in `lst_p95`, `qa_count`, or the encoded uint16 product.
   Revert; do not re-tune toward passing.
2. `native_passes` leaves 1.0, or `composite_tasks` leaves the 1.4x band.
3. A candidate's single falsifying measurement (§7) fires. Write the negative result; do
   not open a second front on the same candidate.
4. Both candidates have been measured. **Two is the cap**; a third idea is a new contract.
5. The projected peak of a configuration exceeds **20 GB**. Do not launch it.
6. Any step would require a network call, AWS credentials, Coiled, GEE, or a production
   tile. This contract authorizes none of them.
7. The measured spread across 3 fresh subprocesses exceeds 1.3x on `wall_s` and does not
   settle on a quiescent machine — the machine is not fit to measure, so nothing measured
   on it counts.
8. numpy, dask, or xarray differs from §2. Re-baseline first; bit-exactness claims do not
   survive a numpy change.
9. A candidate passes locally but its mechanism turns out to depend on a cloud property
   (candidate 2 is already in this position). Stop at "survived falsification" and hand
   back; do not upgrade it to a production claim.

---

## 9. Limitations — what this cannot establish

1. **It cannot establish a production speedup.** E1 measures a shard at 1.64-1.69 busy
   cores of 16, so production wall clock is dominated by something the local tier has no
   instance of. A candidate that halves composite CPU moves production wall clock by an
   amount equal to the compute fraction of that wall clock, and that fraction is
   `[unknown]`. **Never multiply a local ratio into an 812 s shard.**
2. **It cannot establish a production peak RSS.** The fixture holds 300 scenes on a
   2,250² grid; a production shard holds 1,031 on 18,000². The dominant term is linear in
   scenes and the block count differs by 1.4x per band. Ratios at fixed geometry transfer;
   gigabytes do not.
3. **It cannot measure the read path at all** — no GDAL, no DEFLATE, no S3, no request
   sizing, no latency, no per-VM throughput cap. E1 and E2 are frozen and cannot be
   extended under this contract, so every claim about I/O remains theirs.
4. **It cannot revalidate `R_COMPOSITE_MB_S = 45.5`**, which is what `landsat_lst.budgets`
   and `landsat_lst.projection` price the whole fleet from. Changing that number requires a
   probe run.
5. **It cannot see hardware effects.** The strided copy at the heart of candidate 1 is a
   memory-subsystem phenomenon, and a Ryzen 7 7840HS laptop is not an m6i.4xlarge. §7
   records the arithmetic showing the 42 s figure cannot hold on the production VM.
6. **It cannot see the export, the upload, the merge, or the pyramid.** ADR-013's export
   property and the `BIGTIFF=IF_SAFER` / `qa_count` nodata rules are pinned elsewhere and
   are untouched here.
7. **It cannot see a second tile's climate.** Both fixtures are 300-scene samples; a
   sampled window already misreports the rejection fraction (69% against 21.8% at
   Pergamino), and nothing here should be read as a statement about a real window.
8. **It says nothing about the offsets stage**, which ADR-015 restructured separately and
   which is out of scope.
