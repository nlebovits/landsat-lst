# ADR-007: Normalize each scene against a monthly climatology, and discard scenes we cannot correct

**Status:** Accepted
**Date:** 2026-08-07
**Authors:** @nlebovits

## Context

Landsat Collection 2 Level-2 Surface Temperature is produced one scene at a time. Turning
measured thermal radiance into surface temperature needs an atmospheric correction, and that
correction rests on estimates of column water vapor. Errors in those estimates apply to the
whole scene at once, with a published magnitude around 1 to 5 K and worse in humid air.

WRS-2 footprints are tilted roughly 10 degrees from vertical, and each pixel in a composite
draws on whichever scenes overlap it. The contributing set therefore changes abruptly at
footprint edges, and neighboring scenes carrying different biases make the composite jump
there. The result is diagonal rectangular seams that trace satellite geometry rather than
anything on the ground.

[ADR-005](005-multiyear-monthly-qa-and-destriping.md) established that the seams are
structural: going from one year to three to five did not reduce them, because pooling more
scenes does not cancel a bias that is common to a whole scene. It shipped stricter QA bits and
the -50/80 C plausibility clamp, which removed most seams, and deferred the correction itself.
[`docs/findings-destriping-and-multiyear.md`](../findings-destriping-and-multiyear.md) records
what was tried. Two approaches are dead ends and should not be revisited: lowering the
percentile makes seams worse (P50 was the stripiest, P95 the cleanest), and scene-level cloud
filtering changes nothing (dropping 34 of 170 scenes moved the mean by 0.01 C).

An earlier normalization used a per-pixel *annual* median as the reference. It failed badly:
because the reference absorbed the seasonal cycle, subtracting it removed the season itself and
cooled the composite from 40.6 C to 29.8 C, with a spatial correlation of only 0.44 against the
baseline.

## Decision

**Estimate one scalar offset per scene against a per-pixel monthly climatology, subtract it, and
discard any scene whose offset is implausible.**

The reference is monthly, not annual. A scene is compared against what that pixel normally does
*in that calendar month*, pooled across every year in the window, so the seasonal cycle stays in
the data and only the scene's bulk deviation is removed. Both reductions use medians rather than
means, so residual cloud cannot drag either the reference or the offset.

Because the same constant applies to every pixel in a scene, the correction shifts only that
scene's baseline. It does not alter within-scene contrasts, create or erase hot spots, sharpen
or blur features, or otherwise change spatial structure. That property is asserted directly in
`tests/unit/test_destripe_normalization.py::test_offset_is_spatially_uniform`.

Offsets are estimated over **land pixels only**. `process_tile` builds the Natural Earth land
mask before compositing and passes it into `compute_annual_composite`, so ocean cannot damp the
estimate on coastal tiles. Ocean is thermally stable, and its near-zero anomalies would
otherwise pull a coastal scene's offset toward zero and make it incomparable with an inland
tile's.

### Discard, never clamp

Scenes whose correction cannot be trusted are removed from the stack. Two conditions trigger
removal: an absolute offset above `settings.destripe_max_offset_c`, and fewer valid land pixels
than `settings.destripe_min_scene_pixels` (500), where the offset is too sparsely estimated to
believe.

Clamping was considered and rejected. Bounding a -73 C offset to a -15 C cap would leave roughly
58 C of uncorrected bias in a scene that is almost certainly cloud-contaminated, and would then
present that scene as corrected. The guiding principle for this dataset is to prefer honest
omission over questionable correction.

Discarding has a consequence worth stating: `qa_count` is computed from the surviving stack, so
the published monthly observation counts describe the evidence actually behind each P95 value
rather than raw data availability. A scene that was dropped is not counted.

The prototype's behavior of neutralizing a sparse scene's offset to 0.0 while keeping the scene
is also rejected, for the same reason. An uncorrected scene in a corrected stack is exactly the
artifact the correction exists to remove.

If every scene is rejected, the pipeline raises rather than emitting an empty composite. That is
a failure worth surfacing, not a tile to publish as nodata.

### The cap is provisional

`destripe_max_offset_c` ships at 15.0 C and is **not yet calibrated**. Measured offsets at
Pergamino, across three prototype runs, were:

| Window | std | min | max |
|---|--:|--:|--:|
| 1-year (n=87) | 10.97 | -30.27 | +15.82 |
| 3-year | 12.34 | -66.06 | +16.41 |
| 5-year | 12.88 | -73.02 | +16.92 |

The distribution is asymmetric. The warm side sits near +16 C in every run, while the cold tail
runs to -73 C, which is the signature of undetected cloud rather than atmospheric bias. The
spread is genuinely broad rather than one outlier inflating the standard deviation: a lone -73
among roughly 674 scenes would contribute only about 2.8 to sigma. Much of that width is real
day-to-day weather that a monthly reference cannot absorb.

At sigma near 12.9, a 15 C cap could reject a substantial share of scenes. That is tolerable
under the guiding principle and leaves ample coverage, since the five-year window carries a
median of 13 to 16 valid observations per pixel per month. It is still a number that must be
measured rather than assumed. `scripts/calibrate_destripe_cap.py` loads a window once, computes
the offsets once, and sweeps candidate caps from 5 to 30 C over the result, reporting the
rejection fraction for each. Run it before treating the default as settled.

### Single-pass climatology

The reference is computed once from uncorrected data and is not re-estimated after rejection.
Iterating would tighten the estimate slightly, at the cost of a method that is harder to
describe and another full pass over roughly 1,900 scenes. The median reference is robust enough
that a handful of outlier scenes barely move it.

## Consequences

The correction costs one extra full traversal of the stack. Offsets must be materialized before
the time axis can be subset, so `scene_offsets` forces an eager reduction on top of the later
percentile pass. On a five-year window that is a second read of a large number of scenes, which
is the main compute risk in the distributed run.

`settings.destripe` defaults to True and exists so raw composites stay reachable for
benchmarking. `compute_annual_composite` still applies no land mask unless one is supplied, so
its documented contract from issue #26 is unchanged for existing callers.

Published tiles built before this change carry the seams. They need reprocessing.

Independent validation against MODIS or ECOSTRESS remains a recommended confidence check rather
than a prerequisite, per the decision record. This is a conversation-starting public product
applying a standard correction for a documented per-scene error, and it is presented as an
indicator of relative surface heat rather than a source of precise absolute temperatures.
