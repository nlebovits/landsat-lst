# Findings: precision and dtype audit of the composite path (#136)

Issue #136 asks what numerical range and precision each intermediate on the LST
composite path needs to produce the published product, and whether a narrower
representation can put a 512-row composite shard on a 32 GiB worker. This is the
audit. No candidate was implemented and nothing ran in the cloud.

Provenance tags: **[M]** measured, **[D]** derived from a measurement, **[A]**
assumed. Every number carries its source. Line numbers refer to `main` at
`32afd4b` with numpy 2.5.2 and xarray 2026.7.0, the versions the lock installs.

## Verdict

1. **The ticket's premise is wrong. The production stack is float32, not
   float64.** `Dataset.where` on a `uint16` band promotes to float32, because
   xarray's `maybe_promote` picks float32 for integers of two bytes or fewer
   (`xarray/core/dtypes.py:99`). The offset cache has returned float32 since
   commit `b67977e` on 2026-08-21 (`offsets.py:230`), so `lst - offset` stays
   float32. The S30W065 run that measured 31 to 44 GB ran on 2026-08-23, after
   that commit. The only float64 on the path is the two-dimensional output of the
   P95 kernel, narrowed to float32 at `pipeline.py:624`.

2. **The float64 claim in #129 came from the probe, not from production.**
   `probe_composite_local.py` builds its synthetic offsets with
   `np.where(rejected, 99.0, 0.0)`, a float64 array, and the subtraction widened
   its whole stack. That is the `2.01 GiB ... float64` allocation in its log, and
   `findings-composite-shard-bottleneck.md` line 110 generalized it to
   `apply_qa_mask`. Every local probe point in that document is a float64
   pipeline that production does not run. The cloud packing probe in the same
   document ran the real shard code and is unaffected. Both files live only in
   the untracked `issue-129-composite-shard` worktree; the correction belongs
   wherever they land.

3. **The narrowest representation the 0.01 C contract justifies is a uint16 stack
   in native DN units, with each scene's offset rounded to one DN.** The DN step
   is 0.00341802 C, so the rounding error is at most 0.0017 C per sample, and
   the P95's linear interpolation is monotone in its inputs, so the P95 moves by
   at most 0.0017 C [D]. Measured on two real 512-square windows of the S30W065
   fixture: max |delta| 0.0016 C, no pixel moved more than one output DN, and
   `qa_count` was equal [M]. Because `encode_lst_uint16` truncates rather than
   rounds (`encoding.py:71`), an error that small still flips the last DN on 3.9%
   and 7.3% of pixels, all by exactly one DN of 0.01 C. That is the cost of two
   bytes per element instead of four.

4. **float32 versus float64 is not bit-identical after encoding either.** The
   control arm, a float64 stack, differs from the shipped float32 path on 14 and
   20 pixels of 262,144 (5e-5 and 8e-5), by one DN, with max |delta| 5.5e-6 C
   [M]. Truncation turns any nonzero difference near a boundary into a DN flip.
   The 0.01 C contract already tolerates flips at this rate; the question for the
   operator is whether it tolerates them at 4 to 7%.

5. **int16 at 0.01 C is worse, and float16 fails.** Rounding the debiased value
   to 0.01 C gives 0.005 C of error per sample and flips 21.5% of pixels [M].
   float16 has a 0.03 C step at 50 C, three output DN [D].

6. **The modeled saving is a halving of every stack-proportional term, which is
   nearly all of the measured peak. It is a floor, not a forecast.** At the
   measured 912 kept scenes, a 512-row band holds 34.4 GB of float32 pieces if
   every column chunk is read before any P95 retires, and one fused P95 task
   holds two to three copies of a 0.96 GB chunk [D]. The measured peaks, 31.9 to
   43.9 GB across 33 bands [M], sit at that band-stack bound. A two-byte stack
   models 17.2 GB plus 1.4 to 2.9 GB of in-flight copies, so 19 to 23 GB against
   a 27 to 28 GiB target. Only the corrected local probe, then one Coiled shard,
   can turn that into a number.

7. **The candidate is not a dtype swap. It is a kernel and a graph change.** An
   integer stack has no NaN, so the P95 kernel needs a sentinel-first sort and a
   shifted gather, `notnull` becomes a sentinel compare, and the Celsius
   conversion moves from before the stack to after the quantile. The scene keep
   mask, the QA bits, the clamp, the land and GED masks, and the offset values
   are unchanged. `offsets.ALGORITHM_VERSION` does not move, because the cached
   estimate does not change; the composite's bit-exactness tests do.

## 1. The numerical path on `main`

Every stage, with the expression that sets its dtype. "Per chunk" is one 512 by
512 spatial chunk at T scenes; T = 912 is the measured kept depth of a S30W065
band [M], T = 1,031 the band's axis before rejection.

| # | Stage | Function, site | dtype in, out | Why | Valid range | Precision downstream needs | B/elem | Nodata needs float? | Narrower safe? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | STAC load `ST_B10`, `QA_PIXEL` | `load_scenes`, `pipeline.py:301` | disk, `uint16` | odc-stac infers from `raster:bands`; no `dtype=` passed | DN 1 to 65,535; 0 is fill | integer, exact | 2 | no, 0 is fill | already minimal |
| 2 | QA mask | `create_qa_mask`, `qa.py:37` | `uint16`, `bool` | bit tests | 0 or 1 | exact | 1 | no | already minimal |
| 3 | QA apply | `apply_qa_mask`, `qa.py:51` | `uint16`, **`float32`** | `Dataset.where` promotes; `dtypes.py:99` | DN grid, NaN | none added; mask only | 4 | yes, as written (NaN) | yes: keep uint16 with 0 as the mask sentinel. Also masks `qa_pixel` to float32 for nothing; dropped one line later |
| 4 | Celsius | `convert_to_celsius`, `qa.py:79-85` | `float32`, `float32` | Python-float scalars are weak under NEP 50 | -50 to 80 C after clamp | the P95 needs the DN order and the DN values; conversion is affine, so it commutes with the quantile | 4 | as written, yes | yes: defer the affine map to after the P95; clamp on DN bounds |
| 5 | Land mask | `pipeline.py:493` `.where(land_mask)` | `float32`, `float32` | mask is `bool` (`masks.py:174`) | unchanged | none | 4 | as written | follows stage 3 |
| 6 | Offsets from cache | `OffsetCache.read`, `offsets.py:230` | JSON float64, `float32` | explicit `dtype=np.float32`, with the reason in its docstring | -15 to 15 C after rejection | 0.0017 C is enough (section 3) | 4 per scene | n/a, one value per scene | yes: round to whole DN, int16 |
| 7 | Debias | `debias_with_offsets`, `normalization.py:647` | `float32` - `float32`, `float32` | broadcast subtract along time | -65 to 95 C | order and value to 0.0017 C | 4 | as written | yes: integer subtract on DN; range 19,347 to 59,367 on both windows, inside uint16 with 0 free [M] |
| 8 | Solar-day merge | `groupby="solar_day"`, `pipeline.py:306` | `uint16`, `uint16` | odc-stac mosaics items in the load | as stage 1 | exact | 2 | no | already minimal |
| 9 | Rechunk time to one chunk | `pipeline.py:575` | `float32`, `float32` | `chunk({"time": -1})`; the P95 needs the full series | | | 4 | | follows stage 3 |
| 10 | Valid mask | `pipeline.py:578` `lst.notnull()` | `float32`, `bool` | NaN test | | exact | 1 | | becomes `!= 0` on an integer stack |
| 11 | `qa_count` | `pipeline.py:579-586` | `bool`, `int64`, **`uint8`** | `groupby("time.month").sum()` returns int64; explicit `astype(np.uint8)` | 0 to 255 | exact integer | 8 then 1 | no | int64 is 12 x 512^2 x 8 B = 25 MB per chunk, 2.6% of one float32 copy; uint16 would do, but the saving is below noise |
| 12 | `total_valid` | `pipeline.py:590` | `bool`, `int64` | `sum(dim="time")` | 0 to T | exact | 8 | no | 2 MB per chunk; ignore |
| 13 | P95 | `nanquantile_last`, `pipeline.py:601-609`, `kernels.py:91-105` | `float32` in, **`float64`** out | `np.sort` in the input dtype; `h`, `t`, `out` in float64 | -65 to 95 C | the lerp's float64 is 2-D, 2 MB per chunk | 4 for the sort copy, 8 for the 2-D result | | the sort copy follows the stack dtype; the float64 2-D result is free |
| 14 | Narrow | `pipeline.py:624` `astype(np.float32)` | `float64`, `float32` | explicit | | float32 ulp at 95 C is 7.6e-6 C, 1/1300 of one output DN | 4 | | lossless to the contract (section 2) |
| 15 | Ocean zero and masks | `pipeline.py:830-846` | `float32`, `float32`; `uint8` | `.where`, explicit `astype(np.uint8)` on counts | | | | | already minimal |
| 16 | Encode | `encode_lst_uint16`, `encoding.py:60-71` | `float32`, `uint16` | `(c + 50) / 0.01`, out-of-range to 0, then `astype` **truncates** | DN 1 to 65,535 | 0.01 C | 2 | no, 0 is fill | this is the contract |
| 17 | COG statistics | `cog.py:302` | `uint16`, `float64` | moments accumulate in float64 | | | 8, streaming | | not a stack term |

The shard path (`run_composite_shard`, `shard_tasks.py:979-1070`) calls the same
functions on a row slice of the tile geobox, with offsets from `_tile_offsets`
at `shard_tasks.py:937-976`, so the chain is identical. The whole-tile path
estimates offsets in float32 as well (`normalization.py:311,401`).

## 2. What the contract discards

The product is `uint16` at 0.01 C with `offset -50`, and the encoder truncates
toward zero (`encoding.py:71`; verified: 25.007 C encodes as DN 7500, which
decodes as 25.00 C [M]). So:

- Any two intermediates that differ by less than the distance to the next 0.01 C
  boundary encode identically. There is no rounding, so the boundary sits at the
  value itself: a value 1e-6 C above 25.00 and one 1e-6 C below encode to
  different DN.
- float32 carries 24 significant bits. At 95 C the spacing is 7.6e-6 C; at the
  DN grid of 0.0034 C the debiased value is represented to better than one part
  in 400 of a DN step [D]. The float64 kernel output narrowed to float32 at
  `pipeline.py:624` therefore loses nothing the encoder can see. The control
  arm below measures the residual: 5e-5 to 8e-5 of pixels flip one DN because of
  float32 rounding in the per-sample Celsius conversion, not the narrowing.
- The source itself is a 0.0034 C grid. Nothing on the path adds information
  below that grid except the offset, one value per scene.

## 3. Error propagation for the candidates

The debiased sample is `DN_t * s + c - o_t` with `s = 0.00341802`, `c = 149 -
273.15`, and `o_t` the scene offset. The P95 is a linear interpolation between
two order statistics of that series per pixel, so it is monotone and
1-Lipschitz in each sample: perturb every sample by at most `e` and the P95
moves by at most `e` [D].

| Candidate | Representation | Per-sample error bound | Predicted flip fraction, E\|err\| / 0.01 [D] |
|---|---|---|---|
| float64 stack (control) | 8 B | float32 rounding of the shipped path, about 4e-6 C | 4e-4 |
| uint16 DN stack | 2 B, `DN_t - round(o_t / s)`, 0 as sentinel | `s / 2 = 0.0017` C, from rounding `o_t` | 0.086 |
| int16 at 0.01 C | 2 B, `round(v / 0.01)`, -32768 as sentinel | 0.005 C | 0.25 |

For the uint16 DN candidate the affine map commutes with the quantile exactly,
so the only error is the offset rounding. The range check is part of the
candidate: `DN - shift` must stay within 1 to 65,535. With the clamp at -50 to
80 C and the offset cap at 15 C the debiased value spans -65 to 95 C, which is
DN 17,261 to 64,073 [D]; both measured windows spanned 19,347 to 59,367 [M].

### Measured on real data [M]

`scripts/experimental/precision_audit_p95.py` loads one 512-square window of
the retained S30W065 fixture (`results/fixtures/S30W065_2021-2025_n300_f8`,
300 solar days at offset factor 8, real DN and real QA), runs the shipped
functions as the baseline (`apply_qa_mask`, `convert_to_celsius`, the float32
subtraction, `_composite_graph`, `encode_lst_uint16`), and compares each
candidate's encoded DN pixel by pixel. Offsets are drawn from the calibrated
distribution (normal, 5.71 C, seeded) and cast to float32 as the cache does;
scenes beyond 15 C are dropped as production drops them. The harness asserts
that its own call of `nanquantile_last` reproduces `_composite_graph` bit for
bit before comparing anything.

| Window (row, col) | Kept scenes | Valid samples | Median valid per pixel | P95 range, C |
|---|---|---|---|---|
| 800, 800 | 295 of 300 | 3.3% | 10 | 34.4 to 73.4 |
| 1600, 300 | 294 of 300 | 4.2% | 14 | 22.9 to 57.7 |

| Candidate | Window | Identical | One DN | More than one DN | Flip fraction | Max \|delta\| C | `qa_count` equal |
|---|---|---|---|---|---|---|---|
| float64 stack | 800, 800 | 262,124 | 20 | 0 | 7.6e-5 | 5.5e-6 | n/a |
| float64 stack | 1600, 300 | 262,127 | 14 | 0 | 5.3e-5 | 3.8e-6 | n/a |
| uint16 DN stack | 800, 800 | 243,014 | 19,130 | 0 | 0.073 | 0.0015 | yes |
| uint16 DN stack | 1600, 300 | 251,930 | 10,211 | 0 | 0.039 | 0.0016 | yes |
| int16 at 0.01 C | 800, 800 | 205,659 | 56,485 | 0 | 0.215 | 0.0050 | yes |
| int16 at 0.01 C | 1600, 300 | 205,800 | 56,341 | 0 | 0.215 | 0.0050 | yes |

No candidate produced a fill disagreement, a difference above one DN, or a
different `qa_count`. The uint16 DN flip fraction fell below its prediction
(0.086) on both windows, because the prediction assumes a uniform position
relative to the boundary and the P95 of an integer-valued series lands on
the grid more often than that. The int16 candidate matched its prediction.

Both windows are sparse (median 10 to 14 valid samples per pixel) because the
fixture holds 300 of the window's roughly 2,900 scenes and the QA mask is
strict. A production pixel has an order of magnitude more samples, so the P95
interpolates between closer neighbors and the same bound holds. Two windows
and two seeds are not a survey; they are enough to reject the 0.01 C step and
to bound the DN step.

The script takes 72 s and 5.4 GB peak RSS on a laptop [M], because xarray
materializes several full-window temporaries. It is not a memory measurement
of anything.

## 4. Working-set model, and what the measured peaks say

Per element, one fused P95 task holds: the merged chunk (all T pieces
concatenated), a contiguous transposed copy for the last-axis sort (observation
2026-09-02: the transpose copy, not the sort, dominates the kernel's wall time),
and the sort output. Two of the three coexist at any instant, three at the
handover [A]. Beside them the 2-D arrays (`n`, `h`, `a`, `b`, `out`) are 2 to
8 MB each and the month sums 25 MB.

The band-wide term matters more. `_composite_graph` fuses the read with the
elementwise chain, so a read task retires as a float32 Celsius piece of 10 x 512
x 512 x 4 B = 10.5 MB, and that piece is held until its column's merge can run.
The cloud packing probe measured reads delivering a chunk every 22 s against 37
s of fused compute per chunk [M, `findings-composite-shard-bottleneck.md`], so
reads run ahead and pieces accumulate toward the whole band.

| Term | float32, T = 912 | uint16, T = 912 | float32, T = 1,031 | uint16, T = 1,031 |
|---|---|---|---|---|
| One chunk copy | 0.96 GB | 0.48 GB | 1.08 GB | 0.54 GB |
| One fused task, 3 copies | 2.9 GB | 1.4 GB | 3.2 GB | 1.6 GB |
| Whole band held, 36 columns | 34.4 GB | 17.2 GB | 38.9 GB | 19.5 GB |
| Band plus 2 in-flight tasks | 40.2 GB | 20.1 GB | 45.4 GB | 22.7 GB |
| 16 tasks in flight, no band term | 45.9 GB | 23.0 GB | 51.9 GB | 25.9 GB |

Measured [M], run `shard-S30W065-2021-2025-20260823T102135Z`,
`results/probe/composite_band_phase_seconds.json`: 33 bands of 512 rows on
`m6i.4xlarge`, 912 kept scenes, peak RSS min 31.9, median 39.7, max 43.9 GB;
the 1,024-row band 0 peaked at 59.4 GB and finished. The float32 column of the
model brackets those peaks, and band 0 at 59 GB against a 69 to 78 GB band term
says the pile does not reach the whole band before P95 tasks start retiring it.

So the halving is credible on the model, and the modeled peak of 19 to 23 GB at
T = 912 sits under the 27 to 28 GiB target with 4 to 9 GB to spare. Three things
the model does not hold: the Python and GDAL baseline, read buffers on 16
threads, and the dask scheduler's own retention. CLAUDE.md's rule applies: a
floor a configuration cannot fit disqualifies it for free, and a floor it fits
is not a forecast. The prior three-term model landed at 17 GB for a run that
OOMed at 46.5 GB.

Two points the ticket excludes but the model exposes, recorded and not pursued:
the number of fused tasks in flight is the dask thread count (16 on that VM),
and the read-ahead that fills the band term is the scheduler's ordering. Both
are concurrency, which the stop rules forbid.

## 5. What a candidate would have to change

Recorded for scoping. None of it is implemented.

- `apply_qa_mask` and `convert_to_celsius` keep the stack `uint16`, with 0 as
  the sentinel for QA-masked, fill, and out-of-clamp samples. The clamp becomes
  a DN bound (`(-50 - c) / s` to `(80 - c) / s`). Masking `qa_pixel` to float32
  stops.
- `debias_with_offsets` subtracts `round(o_t / s)` as an integer, on the kept
  scenes only, and asserts the result stays inside 1 to 65,535 before writing 0
  where the source was 0.
- `_composite_graph` counts validity with `!= 0`, and the P95 kernel takes an
  integer array with a sentinel-first sort and a shifted gather (the shape of
  `_lerp_last` in the script). The lerp stays in float64 and the affine map to
  Celsius runs on the 2-D result. `output_dtypes` stays float64; the narrow to
  float32 at `pipeline.py:624` stays.
- The whole-tile path's offset estimator is untouched: `climatology_by_blocks`
  and `offsets_by_scene` read the stack through `_read_values(..., dtype)` and
  would need their own conversion if the loaded stack is no longer Celsius.
  That is the seam the candidate has to respect, and the reason the change is
  larger than one `astype`.
- Tests: `tests/unit/test_kernels.py` pins bit-exactness against
  `np.nanquantile` on float input; the integer kernel needs its own oracle
  (this audit's harness is one). `tests/integration/test_shard_merge_equivalence.py:527`
  pins `lst_p95` float32 and is unaffected.
- The probe must build float32 offsets before any candidate is measured with
  it, or the baseline it reports is the float64 pipeline again.

## 6. The candidate, implemented and measured

The operator accepted the one-DN behavior on 2026-09-03 and asked for one
bounded candidate: the uint16 DN stack, nothing else. It is in
[ADR-019](adr/019-composite-stack-in-native-dn.md). What it changed:

- `qa.py`: `dn_stack`, `dn_clamp_bounds`, `celsius_stack`, `debias_dn`,
  `offset_dn_shift`, `dn_to_celsius`, and the DN constants.
- `kernels.py`: `quantile_last_sentinel`, the integer P95 kernel.
- `normalization.py`: `debias_with_offsets` dispatches on dtype;
  `seasonal_debias` takes `apply_to`, so the estimator reads the float32
  Celsius view and the correction lands on the DN stack.
- `pipeline.py`: `compute_annual_composite` builds the DN stack;
  `_composite_graph` dispatches on dtype and converts the 2-D P95 to Celsius.
- `tests/unit/test_dn_stack.py`: 40 tests. The sentinel kernel is pinned bit
  for bit against `np.nanquantile` on the float64 image of the stack; the
  whole composite is pinned at most one encoded DN from the float32 path with
  `qa_count` equal; the Celsius view is pinned bit-identical to the old stack.
- `scripts/probe_composite_local.py`: ported from the #129 worktree, offsets
  cast to float32, and a `--keep-dir` so two arms can be diffed.

The full unit suite passes credential-less (1,591), the composite integration
tests pass (32), and `tests/benchmark` passes (16) on the float oracle.

### Corrected probe, both arms [M]

16 threads, 1,031 scenes with 923 kept, chunk 512, 512 rows, laptop with 16
logical cores and 54 GB. Each point is a fresh subprocess under `RLIMIT_AS`
and an RSS watchdog. `results/issue-136/probe/{baseline_f32,candidate_u16}.jsonl`.

| Column chunks | float32 peak, GB | uint16 peak, GB | Ratio | float32 wall, s | uint16 wall, s | float32 CPU, s | uint16 CPU, s |
|---|---|---|---|---|---|---|---|
| 4 | 9.45 | 5.04 | 0.53 | 48.2 | 37.2 | 205 | 162 |
| 8 | 16.92 | 7.75 | 0.46 | 62.2 | 52.2 | 419 | 317 |
| 16 | 24.24 | 10.18 | 0.42 | 101.2 | 79.3 | 804 | 625 |

Task counts are identical per point (3,216, 6,224, 12,240). The fused P95
task holds 44 to 48% of task time on the candidate against 51 to 56% on the
baseline, and the `where` chain 19 to 21% against 22 to 26%.

The float32 baseline at 8 column chunks is 16.9 GB where the float64 probe
in #129 reported 23.2 GB at the same point and was killed at 26 GB with 16
threads: the 2x the audit predicted for the mis-shaped probe.

### Extrapolation to a production band [D]

Peak grows 1.23 GB per column chunk on the baseline and 0.43 GB on the
candidate between 4 and 16 chunks. A production band has 36:

| | float32 | uint16 |
|---|---|---|
| Linear to 36 chunks | 48.9 GB | 18.8 GB |
| 16-chunk ratio applied to the measured worst case, 43.9 GB | | 18.4 GB |
| 16-chunk ratio applied to the measured median, 39.7 GB | | 16.7 GB |

The float32 extrapolation overshoots the VM's 31.9 to 43.9 GB because the
synthetic source delivers pieces faster than S3 does, so more of the band
piles up before the P95 retires it. That makes the candidate's 18.8 GB the
pessimistic figure. Against a 32 GiB VM with about 30 GiB usable, and the
ticket's 27 to 28 GiB target, the model leaves 9 to 10 GB of headroom.

### Product differences, implemented path [M]

| Data | Pixels compared | Identical | One DN | More than one | Max delta, C | `qa_count` |
|---|---|---|---|---|---|---|
| Fixture window (800, 800), real | 262,144 | 243,014 | 19,130 (7.3%) | 0 | 0.0015 | equal |
| Fixture window (1600, 300), real | 262,141 | 251,930 | 10,211 (3.9%) | 0 | 0.0016 | equal |
| Probe, 4 chunks, synthetic, zero offsets | 1,048,576 | 1,048,176 | 400 (0.04%) | 0 | | equal |
| Probe, 8 chunks, synthetic, zero offsets | 2,097,152 | 2,096,329 | 823 (0.04%) | 0 | | equal |

The implemented path reproduces the audit's `uint16_dn_stack` arm bit for bit
on both windows. With zero offsets the DN path is exact and the residual
0.04% is the float32 per-sample rounding of the old path.

### Verdict on the local question

Yes on the model: one change, peak RSS at 0.42 to 0.53 of the float32 arm,
wall time down 16 to 23%, CPU down 21 to 25%, the product within the accepted
one-DN behavior, `qa_count` exact. The next and only cloud discriminator is
one representative 512-row composite shard on a 32 GiB VM. It has not run.

## 7. The cloud discriminator: stop

The operator greenlit one 512-row composite shard on a 32 GiB worker on
2026-09-03. Band 27, the band with the highest measured peak (43.9 GB on a
64 GiB `m6i.4xlarge`), ran on a `c6i.4xlarge` (16 vCPU, 32 GiB, the same
core count) from the retained production plan, item list, and merged
offsets, under a fresh run prefix. Decision rule, fixed before the run: peak
RSS at or under 28 GiB with both slabs published passes; above 28 GiB, an
OOM, or a material output difference stops.

**Result: stop.** Attempt 1 of run `shard-S30W065-2021-2025-20260903T112514Z-u16gate2`
(Coiled cluster 2000575) was killed at **28,789.5 MB (28.11 GiB)** after
1,391.8 s, 1,313.7 s of it in `exporting`, with no log and no error object,
which is the OOM shape on a 32 GiB VM. No band slab was published. Coiled
started a retry, which was stopped at 422 s and 9.3 GB to bound spend.
Output equivalence could not be measured. The contract, the decision
manifest (validated by `evidence_contract.validate_result`, decision `stop`),
and the bundle from `landsat-lst evidence collect` are under
`results/issue-136/cloud/`; the state objects are retained on S3 under the
run prefix.

| | Baseline, f32, m6i.4xlarge (64 GiB) | Candidate, uint16, c6i.4xlarge (32 GiB) |
|---|---|---|
| Elapsed at last heartbeat, s | 1,365.8 (done) | 1,391.8 (killed) |
| `exporting` phase, s | 1,293.4 | 1,313.7 and counting |
| Sampled RSS at 1,270 to 1,330 s, MB | 32,103, plateau | 25,646 to 27,277, still climbing |
| Sampled RSS slope, 120 to 1,270 s | 1.75 GB/min | 1.45 GB/min (0.83 of baseline) |
| `peak_rss_mb` (VmHWM) | 43,948.75 | 28,789.5, a lower bound |
| Effect on the contract metric | | 0.345 decrease, above the 0.30 minimum |

Two facts from the series [M] that the local probe did not show:

- **RSS climbs linearly through `exporting` in both arms, and halving the
  stack cut the slope by 17%, not 50%.** The growing term is therefore mostly
  not the Celsius or DN stack. The pieces a read task holds before its
  column's merge can run are the raw `uint16` `lwir11` and `qa_pixel` reads,
  four bytes per pixel-scene in both arms, if the elementwise chain is not
  fused into the read task; that is a hypothesis this run cannot test and
  the stop rule forbids chasing.
- **The baseline's sampled plateau was 32.1 GB while its VmHWM was 43.9 GB.**
  An 11.8 GB spike sits between two 60 s heartbeats, at the end of the
  phase. The candidate was killed before reaching that point, so 28.1 GiB is
  a lower bound on its peak, and the true peak on a 64 GiB VM would likely
  sit above 30 GiB.

The uint16 stack still does what the local probe measured on this band:
lower and slower memory growth, and the change is bit-for-bit within the
accepted contract on real data. It does not move the composite stage onto a
32 GiB worker, which was the economic goal. Per the stop rule, no further
memory optimization is stacked on it here.

Arm 1 of the discriminator (run `...20260903T110545Z-u16gate`, cluster
2000548) never reached compute: four attempts failed in the ASTER GED
gap-mask lookup because the worktree ships no `data/ged_gap_mask.npz`. The
mask is applied to the 2-D result after the P95 and cannot move peak RSS, and
the retained 2026-08-23 band predates it, so arm 2 ran with
`LST_GED_GAP_MASK=false` to keep the output comparison like for like.

Cost [M]: Coiled billed 1.56 credits for cluster 2000548 and 8.39 for
2000575, 9.95 in total, against a 20-credit cap; both clusters ran
on-demand although the shard policy is spot. The EC2 charge is derived at
about $0.42 for 0.62 VM-hours at the on-demand list price and is not a
billed number.

## 8. Where the evidence lives

- `scripts/experimental/precision_audit_p95.py`: the harness. Run from the repo
  root with `PYTHONPATH=src`, one window at a time.
- `results/issue-136/precision_audit_window_800_800.json` and
  `precision_audit_window_1600_300.json`: the two windows, with the
  implemented arm, untracked (`results/` is gitignored).
- `results/issue-136/probe/baseline_f32.jsonl`, `candidate_u16.jsonl`, and
  the kept products under `base_c{4,8}/` and `cand_c{4,8}/`.
- `results/issue-136/cloud/`: `contract.json`, `result.json` (decision
  `stop`), `equivalence.json`, `submission.json`, the baseline and candidate
  state objects, and `evidence/evidence.json` from `landsat-lst evidence
  collect --cluster-id 2000548 --cluster-id 2000575`. Retained copies of the
  contract and the decision are under `docs/evidence/issue-136/`.
- S3, run prefixes `_shards/shard-S30W065-2021-2025-20260903T110545Z-u16gate/`
  and `...T112514Z-u16gate2/` under the project prefix: plan and item copies,
  state objects, and logs. No band slab and no COG were written.
- `results/probe/composite_band_phase_seconds.json`: the 35-band RSS anchor.
- `results/fixtures/S30W065_2021-2025_n300_f8/`: the real-data fixture.
- `.claude/worktrees/issue-129-composite-shard/`: the probe and the #129
  findings doc that carry the float64 attribution, both untracked.
