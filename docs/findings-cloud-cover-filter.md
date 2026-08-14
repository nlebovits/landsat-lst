# Pricing a scene-level cloud filter

**Status:** Measured. `max_cloud_cover` stays at 100.
**Date:** 2026-08-14
**Tracking:** [#81](https://github.com/nlebovits/landsat-lst/issues/81), [#34](https://github.com/nlebovits/landsat-lst/issues/34)

## The question is cost, and it has not been asked before

A scene-level cloud filter has been rejected here once already. Issue #34 disabled it, and
[findings-destriping-and-multiyear.md](findings-destriping-and-multiyear.md) confirmed the
decision: tightening `max_cloud_cover` from 100 to 70 dropped 34 of 170 scenes and moved the
composite mean by 0.01 °C at a spatial correlation of 0.98, without touching the seams. The
verdict was **redundant** — once pixel-level QA is good, scene-level filtering adds nothing.

That answered a quality question. [#81](https://github.com/nlebovits/landsat-lst/issues/81) asks
a cost question, and the same measurement reads differently under it: a filter that removes a
fifth of the scenes while moving the product by 0.01 °C is not redundant, it is nearly free. Scene
count is the linear term in every phase of a tile, so 20% fewer scenes is 20% less work in the
STAC query, the load, the offset pass, and the composite alike.

So the earlier finding is not overturned. It measured the wrong axis for this purpose, and the
missing number is what a threshold costs in **evidence**: the valid observations behind each P95
value and behind `qa_count`.

## The default is not the no-op it reads as

`settings.max_cloud_cover = 100` is documented as disabling scene-level filtering. It does not.
The STAC query asks for `eo:cloud_cover` **strictly** less than the threshold, so the default
already drops every scene reported at exactly 100% cloud:

| AOI | window | items, no filter | items, `lt 100` | dropped |
|---|---|--:|--:|--:|
| N40W075 (5°) | 2021–2025 | 2,912 | 2,758 | 154 (5.3%) |
| Pergamino (1°) | 2021–2025 | 782 | 757 | 25 (3.2%) |

A true no-op is 101. The behaviour is the right one — a scene whose own metadata reports no
unclouded pixel has nothing to contribute — but the documented contract was wrong, and anyone
reading it would have believed no scene is ever excluded. Fixed in `config.py`, `README.md`, and
[methodology.md](methodology.md).

Every measurement below is therefore stated against the `lt 100` baseline, which is what
production actually queries.

## What a threshold saves, and what it destroys

Measured at Pergamino, 2021–2025, 757 items, 390 solar-day scenes, of which de-striping keeps
305. The valid-pixel counts come from the same session as the offsets, on the native grid, so the
join is exact rather than inferred. `scripts/analyze_cloud_cover_filter.py`:

| threshold | scenes cut | I/O saved | of those, already rejected | observations lost |
|--:|--:|--:|--:|--:|
| 30 | 180 | 46.2% | 46% | 17.1% |
| 40 | 159 | 40.8% | 51% | 11.4% |
| 50 | 142 | 36.4% | 57% | 7.8% |
| 60 | 113 | 29.0% | 63% | 4.8% |
| 70 | 84 | 21.5% | 69% | 2.7% |
| 80 | 60 | 15.4% | 78% | 1.3% |
| 90 | 30 | 7.7% | 90% | 0.3% |
| 95 | 21 | 5.4% | 90% | 0.2% |

"Already rejected" is the share of cut scenes that de-striping discards anyway, under the shipped
15 °C cap and the 500-pixel floor. Those cost nothing to skip earlier, which is the case #81
makes: a threshold that removes scenes the pipeline was going to throw away is free. At 90 it is
nine cut scenes in ten.

"Observations lost" counts only the rest — valid pixel-scenes belonging to scenes de-striping
would have kept. At 90 that is 0.3% of the evidence for 7.7% of the work.

## The scene-level statistic predicts AOI coverage better than expected

`eo:cloud_cover` describes a whole Landsat footprint, roughly 185 km across, while a tile sees
only part of one. A scene reported at 90% cloud could in principle be clear over the AOI. Measured
against the valid-pixel count the QA mask actually produces:

- **corr(`eo:cloud_cover`, valid-pixel fraction) = −0.752.**
- **Zero** scenes at ≥70% cloud that de-striping keeps and that cover more than half the AOI.
  Same at ≥90%.

So at this AOI the proxy holds: heavily clouded scenes really are the ones that contribute little.
The correspondence should be *better* on a 5° tile, not worse, because a footprint sits mostly
inside a 550 km tile, whereas the 1° AOI here is a small window inside the footprint and the
statistic averages over a great deal of ground the AOI never sees.

On the direct cost alone, then, a threshold of 90 looks like an easy trade: 7.7% of the work for
0.3% of the evidence, nine cut scenes in ten already destined for the bin.

## The indirect cost is an order of magnitude larger

Every scene's offset is measured against a per-pixel **monthly climatology built from the
surviving scenes**. Remove scenes and the reference moves, which moves the offsets of the scenes
that stay. #81 flags this, and the 300-scene sample that pushed the rejection rate from 21.8% to
69% is the precedent. A cloud filter is not random sampling, so the size of the effect had to be
measured. `scripts/measure_climatology_thinning.py`, same AOI and window, offsets recomputed on
each subset in one shared pass, scored on the scenes both sets contain:

| threshold | scenes | rejected | med \|Δoffset\| | p99 \|Δoffset\| | max \|Δoffset\| | decision flips |
|--:|--:|--:|--:|--:|--:|--:|
| *none* | 390 | 21.3% | *reference* | | | |
| 90 | 360 | 16.1% | 0.0154 | 1.120 | **3.032** | 0 |
| 80 | 330 | 11.5% | 0.0906 | 1.900 | **4.406** | 0 |
| 70 | 306 | 8.8% | 0.2119 | 1.991 | **4.131** | 0 |

The keep-set itself is stable: **zero decision flips** at every threshold, and the falling
rejection rate is the filter removing scenes de-striping was rejecting anyway. What moves is the
correction applied to the scenes that stay, and it moves by degrees.

Put that beside the bar this project already applies to offsets. The coarse-grid work fixed its
acceptance criteria in advance at p99 ≤ 0.25 °C and max ≤ 0.5 °C, on the reasoning that an offset
error lands directly in the P95 of every pixel its scene touches and the urban contrasts this
product exists to show are a few degrees. Factor 4 was rejected this week for a max of 0.546 °C.

A cloud filter at 90 — the mildest threshold tested, cutting 7.7% of scenes — moves offsets by
**3.03 °C at the worst and 1.12 °C at the p99**. That is six times the bound that disqualified
factor 4, for a fifth of the saving.

The measurement cannot say which set of offsets is *better*. Removing cloudy scenes thins the
climatology, which argues the filtered estimate is noisier; it also removes undetected cloud, the
known source of the cold tail, which argues the opposite. Settling that needs an end-to-end
composite comparison, not an offset comparison. But the direction does not change the conclusion:
the two configurations produce materially different corrections, so a cloud filter is not a
drop-in cost optimization. It is a change to the product.

## Why the earlier measurement missed this

Issue #34's test found a 70% threshold moving the composite mean by 0.01 °C. That is not in
conflict. It was run as a candidate **seam fix**, before season-aware de-striping existed — the
finding's own words are that it "did not reduce the seams". With no monthly climatology in the
pipeline, there was no reference for a smaller scene set to thin, and the interaction measured
above could not arise.

The lesson generalizes past this setting: a dead end recorded against one pipeline does not stay
dead once the pipeline grows a component the test could not see.

## Decision

**`max_cloud_cover` stays at 100.** No threshold tested clears the offset-accuracy bar the
project applies to every other change in this area, and the cheapest one that does anything
useful misses it by 6×.

What would change the answer, in the order worth trying:

1. **An end-to-end composite comparison at threshold 90.** `compare_destripe_composites.py`
   already builds paired P95 maps from one load. If a 3 °C offset shift turns out not to move the
   composite, the trade is back on the table, and this time on the right axis.
2. **A humid tropical tile.** Everything here is mid-latitude cropland where the median scene is
   26.7% cloud. Where the median scene is mostly cloud, both the saving and the thinning are
   larger, and neither is predictable from these numbers.

Neither is a laptop measurement on a production tile, which is the same conclusion #81 reaches
about its own runtime claims: one full-window tile settles it, nothing else does.
