# Estimating de-striping offsets from coarse overviews

**Status:** Complete. Shipped as `settings.destripe_offset_resolution_factor = 2`.
**Date:** 2026-08-07
**Tracking:** [#46](https://github.com/nlebovits/landsat-lst/issues/46), [ADR-007](adr/007-scene-normalization.md)

## The problem

De-striping needs one scalar per scene, but derives it from a full-resolution per-pixel monthly
climatology. Offsets must be materialized before the time axis can be subset, and nothing is
persisted, so the pipeline reads the stack twice: once for the offsets, once for the P95.
Measured at 9.5 minutes for the offset pass alone on a 1° AOI with 390 solar-day scenes, against
a 5° production tile covering 25 times the area.

Since the output is a single number per scene, spatial detail should buy nothing. The question
was how much resolution could be given up, and how to prove the cheaper estimate is the same
number.

## Subsampling a loaded array does not help

The obvious approach is to stride or coarsen the array after loading, which is the idiom already
used in `zarr_writer.build_overviews()`. It cuts compute but **not** I/O: dask must materialize
each source chunk before discarding most of it, so the bytes still cross the network.

At roughly 20 GB per full-resolution pass over 9.5 minutes, the work is network-bound. Only a
coarser `resolution=` on `stac_load` reduces bytes fetched, because GDAL then serves the request
from the source COGs' internal overviews.

Two facts had to hold, and both were verified before building anything:

**The source COGs carry overviews.** Both `lwir11` and `qa_pixel` expose levels
`[2, 4, 8, 16, 32, 64]`.

**Decimated `qa_pixel` stays valid.** This is a bitfield, so averaged overviews would make
decimated QA arithmetic nonsense. Reading a 1024×1024 window at 16× produced 9 distinct values,
every one present in the full-resolution vocabulary, and no novel values. USGS built these with
nearest or mode. The pipeline still passes `resampling={"lwir11": "average", "qa_pixel":
"nearest"}` explicitly rather than relying on that.

## Method

`scripts/validate_offset_subsampling.py` loads the stack at each factor, computes offsets for all
of them in one `dask.compute` so the graphs share work, and compares scene by scene against the
full-resolution result from the same session.

Three guard rails, since a silent failure would invalidate every number:

- **Paired comparison.** Every coarse offset is compared against the same scene's native offset,
  not against summary statistics.
- **Time axes asserted equal.** All loads use the same items and `groupby="solar_day"`, so they
  must align. Without the assertion, xarray would broadcast and quietly invent agreement.
- **Reference cross-checked** against the committed calibration output. It reproduced to
  **0.0005 °C**, confirming the pipeline is deterministic and the numbers are comparable to the
  shipped 15 °C cap.

Accuracy is scored only over scenes both paths actually keep. Scoring everything reads as failure
where there is none: an early run showed a 17 °C disagreement on a scene holding **2** valid
native pixels, which both paths were discarding as too sparse to estimate.

Acceptance criteria were fixed before results were seen: median |Δ| ≤ 0.05 °C, p99 ≤ 0.25, max
≤ 0.5, |bias| ≤ 0.05, and keep/reject flips ≤ 1% with every flip within 1 °C of the cap. An
offset error lands directly in the P95 of every pixel its scene touches, and the urban contrasts
this product exists to show are a few degrees.

## Results

Pergamino, 2021–2025, 757 STAC items, 390 solar-day scenes.

| factor | grid | MB read | med \|Δ\| | p99 \|Δ\| | max \|Δ\| | flips | passes |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 1 | 3600×3601 | 20223 | *reference* | | | | |
| 2 | 1801×1801 | 5060 | 0.0017 | 0.063 | 0.188 | 0 | **yes** |
| 4 | 901×901 | 1266 | 0.0068 | 0.156 | 0.431 | 0 | **yes** |
| 8 | 451×451 | 317 | 0.0171 | 0.359 | 1.384 | 0 | no |
| 16 | 226×226 | 80 | 0.0359 | 0.757 | 0.923 | 1 | no |
| 32 | 113×113 | 20 | 0.0667 | 1.295 | 2.345 | 2 | no |
| 64 | 57×57 | 5 | 0.1102 | 2.011 | 3.233 | 2 | no |

## A prediction that failed

The plan predicted, in advance and in writing, that error would stay flat and then plateau: LST
is spatially autocorrelated, so effective sample size should be governed by correlation length
rather than pixel count, and full resolution should already hold far fewer effective samples than
its 12.96M pixels suggest. On that reasoning, large factors would cost almost nothing.

The data says otherwise. **Error grows linearly in the factor.** Median |Δ| runs 0.0017, 0.0068,
0.0171, 0.0359, 0.0667, 0.1102 across factors 2 to 64. From factor 4 to 64, a 16× coarsening,
median error rose 0.0068 → 0.1102, a factor of 16.2.

That is what independent sampling predicts and autocorrelation was supposed to beat: standard
error of a median scales as 1/√n, and n falls as 1/factor², giving error ∝ factor. Whatever
correlation structure the anomaly field has, it does not rescue the estimate at these scales.

The practical consequence is that the aggressive factors are not available. Factor 64 would have
read 5 MB instead of 20 GB, but its offsets are wrong by 0.11 °C at the median and 3.2 °C at the
worst, and it flips two keep/reject decisions.

## Decision

**Factor 2**, following the pre-registered rule of shipping one step more conservative than the
largest passing value (4).

The saving is bounded near 2× no matter what, because the P95 still needs its native pass:

| | bytes per composite | vs today |
|---|--:|--:|
| today (two native passes) | 40.4 GB | — |
| factor 2 | 25.3 GB | 1.60× less |
| factor 4 | 21.5 GB | 1.88× less |

Going from 2 to 4 buys another 15% of total I/O while roughly quadrupling offset error. Not worth
it for a correction whose whole justification is that it changes only a scene's baseline.

## Scene-set effects

The two grids apply different sparse guards, so they keep slightly different scene sets. The
sweep reported 14 differences in the sparse guard, but most of those scenes are rejected by the
15 °C cap anyway: the **final** keep-sets differ by 2 out of 390. Native keeps 305, matching the
calibration sweep's cap=15 row exactly; the coarse path keeps 307.

Both extra scenes are empty at native resolution. Their native offset is NaN because no valid
pixel existed to take a median over, while the coarse grid manufactured ≥200 apparently-valid
samples through the same validity dilation described below. Since the composite is built from the
native stack, a scene with no valid native pixels contributes nothing whether kept or not:
subtracting an offset from NaN leaves NaN, and the percentile skips it.

## Coarse loading inflates apparent validity

Worth recording separately, because it broke a design assumption.

The first design scaled the coarse valid-pixel count by factor² so `min_scene_pixels` would keep
one meaning across grids. That is unsound. GDAL's `average` ignores nodata, so a block holding a
single valid fine pixel still yields a valid coarse pixel, and `qa_pixel` is nearest-sampled
independently of it. Measured at Pergamino, a scene with exactly **1** valid native pixel
reported **13** valid pixels at factor 8. Scaling by 64 would have claimed 816 and waved through
a scene the native path rejects outright.

The sparse guard is therefore stated on whichever grid the offset was estimated on: the native
path keeps `destripe_min_scene_pixels`, and a coarse path uses `destripe_min_offset_samples` on
its own grid.

## Reproducing

```bash
uv run python scripts/validate_offset_subsampling.py
uv run python scripts/compare_destripe_composites.py --cogs
```

Outputs land in `results/decision/`. Planetary Computer per CLAUDE.md; Earth Search costs egress
from a laptop.
