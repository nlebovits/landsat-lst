# ADR-019: The composite stack is uint16 DN, not float32 Celsius

**Status:** Implemented and measured, 2026-09-03. The cloud discriminator
ran and said **stop**: band 27 on a 32 GiB `c6i.4xlarge` was OOM-killed at
28.11 GiB, so the composite stage stays on 64 GiB VMs. The representation
change itself stands on the local measurements and the output contract; the
VM-size move it was meant to enable does not follow from it.
**Tracking:** [#136](https://github.com/nlebovits/landsat-lst/issues/136),
[findings](../findings-composite-precision-audit.md)

## Context

A 512-row composite shard of S30W065 peaked at 31.9 to 43.9 GB RSS on a 64 GiB
`m6i.4xlarge` (33 bands, run `shard-S30W065-2021-2025-20260823T102135Z`). The
fleet's cost lever is the VM size: a 32 GiB worker for the composite stage
needs the worst case under about 28 GiB.

The audit for #136 traced every intermediate. The stack was float32 Celsius
from `apply_qa_mask` onward (xarray promotes `uint16` to float32 under
`where`), and every stack-proportional term of the working set carried four
bytes per element. The published product is `uint16` at 0.01 C. The source is
`uint16` DN at 0.00341802 C per DN, three times finer than the product. Four
bytes carried precision the product cannot express.

## Decision

The composite stack stays in native DN as `uint16`, with 0 as "no
observation", from the load to the P95:

- `qa.dn_stack` keeps exactly the samples the Celsius path kept: QA-clear,
  not fill, and inside the plausibility clamp, where the clamp is the float32
  Celsius clamp read back as a DN range (`qa.dn_clamp_bounds`).
- `qa.debias_dn` applies the scene offset as a whole-DN shift,
  `round(offset / 0.00341802)`, and stays `uint16`.
  `normalization.debias_with_offsets` dispatches on the stack's dtype.
- `kernels.quantile_last_sentinel` sorts the two-byte stack with the sentinel
  first and runs the same two-branch float64 lerp as `nanquantile_last`.
  The affine map to Celsius is applied to the 2-D float64 result
  (`qa.dn_to_celsius`), because a quantile with linear interpolation commutes
  with an affine map exactly.
- `qa_count` counts `stack != 0`; it is exact and unchanged.
- The offset estimator is untouched. When it must read the native stack it
  reads `qa.celsius_stack`, a lazy float32 view that is bit-identical to the
  old Celsius stack, and `seasonal_debias(apply_to=...)` lands the correction
  on the DN stack. `offsets.ALGORITHM_VERSION` does not move.
- `_composite_graph` still accepts a float Celsius stack and takes the old
  kernel on it, so the benchmarks and the equivalence oracle keep working.

## Consequences

The only departure from the float32 path is the offset rounded to a whole DN,
bounded at half a DN, 0.0017 C, through a monotone interpolation. Because the
encoder truncates, that flips the last DN on some pixels:

| Data | Pixels | Identical | One DN | More | `qa_count` |
|---|---|---|---|---|---|
| S30W065 fixture window (800, 800), real DN and QA | 262,144 | 92.7% | 7.3% | 0 | equal |
| S30W065 fixture window (1600, 300) | 262,141 | 96.1% | 3.9% | 0 | equal |
| Probe, synthetic, 923 scenes, zero offsets | 2,097,152 | 99.96% | 0.04% | 0 | equal |

The operator accepted the one-DN behavior for this product on 2026-09-03: one
DN is 0.01 C, below any physical precision of the retrieval, and consistent
with the published encoding. float32 versus float64 already flips 5e-5 to 8e-5
of pixels by the same mechanism.

Measured locally on the corrected probe (float32 offsets, 16 threads, 1,031
scenes with 923 kept, `chunk 512`, laptop):

| Column chunks | float32 peak GB | uint16 peak GB | Ratio | Wall s | CPU s |
|---|---|---|---|---|---|
| 4 | 9.45 | 5.04 | 0.53 | 48.2 to 37.2 | 205 to 162 |
| 8 | 16.92 | 7.75 | 0.46 | 62.2 to 52.2 | 419 to 317 |
| 16 | 24.24 | 10.18 | 0.42 | 101.2 to 79.3 | 804 to 625 |

The task count is unchanged (3,216, 6,224, 12,240). Wall time fell 16 to 23%
and CPU 21 to 25%, because the sort and the transpose move half the bytes.
Extrapolated linearly to the 36 column chunks of a production band, the
float32 arm models 48.9 GB against the 31.9 to 43.9 GB measured on the VM
(reads from S3 are slower than the synthetic source, so fewer pieces pile
up), and the uint16 arm models 18.8 GB. Applying the 16-chunk ratio to the
measured worst case gives 18.4 GB. Both sat under the 27 to 28 GiB target on the model. The
measurement disagreed: on a 32 GiB `c6i.4xlarge` the same band reached
28,789.5 MB after 1,392 s, still climbing at 1.45 GB/min against the
baseline's 1.75, and was killed. The sampled RSS ramp is mostly not
stack-proportional, and the baseline's own sampled plateau (32.1 GB) sits
11.8 GB under its VmHWM (43.9 GB). See the findings doc, section 7.

Costs: two kernels instead of one, a dtype dispatch in `_composite_graph` and
`debias_with_offsets`, an `apply_to` seam in `seasonal_debias`, and 190 lines
in `qa.py`. A float `lwir11` is rounded to DN on entry, which only test
fixtures exercise. The `landsat-lst plan` memory floor still assumes four
bytes per element for the composite; it now overstates that floor by two,
which is the safe direction, and is left alone by this decision.

## Alternatives rejected

- **float32 unchanged.** It was already the production dtype; the audit found
  no float64 to remove.
- **int16 at 0.01 C.** Rounds each sample to the product's own step, 0.005 C
  of error, and flips 21.5% of pixels.
- **float16.** A 0.03 C step at 50 C, three product DN.
- **Rechunking, spilling, thread count, read ordering.** All excluded by the
  ticket's stop rules. The model shows the thread count and the read-ahead
  are the other two levers, recorded and not pursued.
