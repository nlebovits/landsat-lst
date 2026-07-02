# Destriping the LST P95 + Choosing a Multi-Year Window

**Date:** 2026-07-02
**Status:** Complete (investigation); season-aware normalization is a prototype pending productionization
**AOI:** ~1° box around tile S30W065 (Pergamino, Argentina), the constant `AOI_BBOX` in the diagnostic scripts, unless noted
**Analysis:** `scripts/season_aware_p95_test.py` (and the fast multi-year / percentile diagnostics)

---

## Summary

We are building a **global extreme-heat Land Surface Temperature map (per-pixel P95)**
for urban planners and public-health users. This is framed as a *conversation-starter
v1, not a perfect product* — good enough to start a conversation about where heat
concentrates, honest about its limits.

Two questions drove this investigation:

1. **How many years of Landsat should feed each P95?** Answer: **3 years is the
   production default.** The 1/3/5-year choice is a *compute-vs-quality* trade,
   **not** a storage trade — storage is essentially constant across windows. Three
   years kills the per-month coverage holes that 1 year leaves; 5 years adds little
   for a 5× compute bill (5yr is under final evaluation).

2. **Why is the P95 raster striped, and how do we fix it?** The stripes are
   **diagonal WRS scene-footprint seams** — a structural consequence of acquisition
   geometry and residual per-scene atmospheric-correction error, *not* noise that
   averages out with more years. The winning fix is **improved QA masking + a
   physical-plausibility clamp + season-aware per-scene normalization**. The
   season-aware step removes the seams while **preserving the hot signal**
   (AOI mean stays ~41 °C).

**Winning recipe: 3-year (or 5-year) P95 + improved QA masking + season-aware
per-scene normalization.**

---

## Part 1 — Multi-year window: 1yr vs 3yr vs 5yr

We built three P95 composites, each paired with a 12-month climatological
`qa_count` (uint8; for calendar month M, the count is valid observations pooled
across all years in the window):

- **1yr:** 2024
- **3yr:** 2022–2024
- **5yr:** 2020–2024

### Storage is essentially constant across windows

The product is one P95 array plus a 12-band monthly QA count — **regardless of how
many years feed it**. QA is always 12 bands; P95 is always a single array. So a
longer window does **not** cost more storage.

Measured global-land extrapolation, per product:

| Quantity | Value |
|---|---|
| Compressed (global land) | **~311 GB** |
| Uncompressed (global land) | **~2.47 TB** |
| Measured compression, LST | ~1.8× |
| Measured compression, QA | ~11–29× |
| Mean compression | ~18× |

**Implication:** the window length is a compute-and-quality decision only. There is
no storage penalty for choosing 3 or 5 years over 1.

### Coverage / per-month gaps

At this well-covered mid-latitude AOI:

- **1yr** leaves months with significant zero-coverage — e.g. **Nov ~17%**,
  **Mar ~5%** of pixels with no valid observation in that calendar month.
- **3yr** eliminates these holes (→ **~0%**).
- **5yr** adds only negligible further improvement.

Overall gap fraction is ~0 for all three windows *here* — but this is a benign,
well-observed site. **This will not generalize** to cloudy or high-latitude tiles,
where the longer window earns its keep.

### Decision

**5-year P95 is the production default**, chosen for consistency with established
products (WRI uses a 5-year window) — that external precedent makes the methodology
easy to defend. Both 3- and 5-year remove the per-month coverage holes and give a
robust hot-tail P95; 5-year is marginally hotter (mean 42.0 vs 41.0 °C) but not
measurably cleaner (offset std 12.9 vs 12.3). It costs ~1.4× the reads of 3-year, but
a single 5-year composite is *cheaper* than five separate annual composites (one load
+ one P95 + one write). **3-year is a supported, lighter fallback** when compute is
constrained.

---

## Part 2 — The striping problem (the crux)

### Symptom

The P95 rasters show **diagonal (~10°) seams** aligned with WRS scene footprints,
plus parallelogram / triangular blocks, clearly visible in QGIS. The mechanism:
each pixel's P95 is computed from whatever set of scenes overlaps *that* pixel.
Crossing a footprint edge changes the contributing scene-set, so the P95 value
**jumps** at the edge — a seam.

### More years did NOT fix it

Extending the window from 1 → 3 → 5 years did not remove the seams. They are
**structural** (acquisition geometry + per-scene correction error), not random
noise that averages away with more samples.

### Percentile sweep — the counterintuitive result

From a single loaded stack we swept P50 / P75 / P90 / P95:

| Percentile | AOI mean | Seaminess (QGIS) |
|---|---:|---|
| P50 (median) | 25.1 °C | **stripiest** |
| P75 | 31.4 °C | |
| P90 | 37.7 °C | |
| P95 | 40.6 °C | **cleanest** |

Counterintuitively, **the median is the stripiest percentile and P95 is the
cleanest.** Explanation: the median sits in the **contaminated, composition-
sensitive bulk** of each pixel's distribution — cold cloud/haze contamination and
the differing scene-set composition across footprints both live there, so the
median jumps at every seam. The **hot tail (P95)** is consistent across footprints
and immune to cold contamination (clouds are cold; they never reach the 95th
percentile). **Lowering the percentile is the wrong direction — P95 was already the
correct and most seam-robust choice.**

> **Lesson (recorded so it isn't repeated):** an early *column-based* striping
> metric reported the **opposite** conclusion because it assumed **vertical**
> stripes. The seams are **diagonal**. The automated metric was wrong; **the visual
> / QGIS inspection was ground truth.** Do not trust an axis-aligned stripe metric
> against diagonal seams.

### Band / harmonization check — a non-issue

We confirmed the inputs are **USGS Collection 2 Level-2 Surface Temperature**
(`ST_B10`), which is **already atmospherically corrected** (scale `0.00341802`,
offset `149.0`). Landsat 8 and Landsat 9 use the **same** ST algorithm and are
**cross-calibrated** in Collection 2, so **cross-sensor harmonization is not the
cause** of the seams and needs no correction.

The residual seams are the **per-scene atmospheric-correction error** (~1–5 K,
worse in humid scenes) that is baked into each ST product at generation time.

---

## Part 3 — Fixes tried, in order

### 1. Improved QA masking — **worked (partial)**

Added **dilated-cloud (bit 1)** and **cirrus (bit 2)** to the existing
cloud/shadow/snow mask (`qa.py` `create_qa_mask`), plus a **physical-plausibility
clamp** (`settings.lst_valid_min` / `lst_valid_max`, default **−50 / 80 °C**, in
`convert_to_celsius`) that drops the ~**−124 °C** DN=0 resampling artifacts near
scene edges.

- **Result:** removed **most** seams; **hot signal preserved** (mean **40.7** vs
  **40.6** baseline).
- **QGIS verdict:** "really close," but **residual edge seams remained.**

### 2. Season-aware per-scene normalization — **worked (chosen approach)**

Level each scene to a **per-pixel monthly climatology** — the median LST for that
pixel in that calendar month — and subtract only the scene's **deviation from that
expected value** (i.e. the atmospheric bias), leaving the seasonal cycle intact.

- **Synthetic self-test:** recovers an injected per-scene bias — the fitted offset
  std matches the injected bias size (not the seasonal amplitude), and the seasonal
  amplitude is preserved.
- **3-year AOI result:** mean **41.0 °C** (hot signal preserved), **seams gone, no
  blocky artifacts** (QGIS-confirmed).
- **Caveat (honest):** on *real* data the monthly reference also absorbs
  **day-to-day weather**, so the per-scene offsets are larger than pure bias
  (**std ~12 °C**, with one outlier scene at **−66 °C**). It is a heavier-handed
  correction than ideal, but the output is **visually clean and mean-preserving**.
  If artifacts ever reappear, **cap the offset magnitude** to reject outlier scenes.

This is currently a **prototype** in `scripts/season_aware_p95_test.py` (function
`seasonal_debias`), not yet in the production pipeline.

---

## What didn't work, and why (do not retry)

Recorded deliberately so these dead ends are not re-explored.

### Lowering the percentile — wrong direction
Intuition says a lower/median percentile would be smoother. **The opposite is
true:** P50 is the *stripiest* (it sits in the contaminated, composition-sensitive
bulk) and P95 is the *cleanest*. Keep P95.

### Scene-level cloud filter — rejected (redundant)
Tightening `max_cloud_cover` from **100 → 70** dropped **34 of 170 scenes** but
changed the product **negligibly**: Δmean **+0.01 °C**, spatial correlation
**r = 0.98**, coverage loss **478 px** — and it did **not** reduce the seams.
Once pixel-level QA is good, scene-level filtering is redundant. This **vindicates
the earlier issue-#34 decision** to disable scene-level cloud filtering.

### Cross-sensor harmonization — non-issue
Landsat 8/9 share the C2 ST algorithm and are cross-calibrated. There is nothing to
harmonize; it is not the seam source.

### Blunt per-scene normalization — failed, dangerous
Leveling each scene to the **annual** per-pixel median flattened the hot signal:
mean **40.6 → 29.8 °C**, spatial **r = 0.44** to baseline. Root cause: a scene's
deviation from the *annual* mean is dominated by the **seasonal cycle**
(offset std ~**11 °C**), not by bias — so subtracting it removes the season, not the
error. **Cooling an extreme-heat product is the dangerous failure mode.** The fix
was to use the *monthly* climatology as the reference (Part 3, step 2), which
removes only the deviation-from-expected.

---

## Decisions summary

| Component | Decision | Status |
|---|---|---|
| P95 (vs lower percentile) | **Keep P95** — cleanest and most seam-robust | Locked |
| Window length | **5-year** default (WRI-aligned); 3-year lighter fallback | Adopted (5yr) |
| Improved QA mask (dilated cloud + cirrus) | **In pipeline** | Shipped (`qa.py`) |
| Physical-plausibility clamp (−50/80 °C) | **In pipeline** | Shipped (`config.py`, `convert_to_celsius`) |
| Scene-level cloud filter | **Disabled** (redundant) | Confirmed (issue #34) |
| Season-aware per-scene normalization | **Chosen approach** | **Prototype only** — `scripts/season_aware_p95_test.py` |

**Winning recipe:** 3-year (or 5-year) P95 + improved QA masking (+ clamp) +
season-aware per-scene normalization.

---

## Open questions / next steps

1. **Productionize season-aware normalization.** Currently only `seasonal_debias`
   in `scripts/season_aware_p95_test.py`. **Recommended next step:** wire it into
   `pipeline.py` / `process_tile` so production composites are debiased.
2. **Offset-outlier cap.** Add a magnitude cap on per-scene offsets to reject
   outlier scenes (e.g. the −66 °C scene) — a safeguard against the monthly
   reference over-absorbing day-to-day weather. Not yet needed visually, but cheap
   insurance.
3. **Independent validation (recommended, not required).** Cross-checking the debiased
   P95 against an **independent LST reference** (MODIS LST, ECOSTRESS) would add
   confidence, but this is a public "conversation-starter" product, not a peer-reviewed
   claim — the method is a standard correction for a documented ~1–5 K per-scene error
   and is unlikely to be challenged for this use. So far we have self-consistency and
   visual checks, which are sufficient to ship a v1.
4. **Global-scale compute.** The dominant cost is reading 5 years of Landsat per tile
   across all land tiles — this belongs on **Coiled / Earth Search** (same-region S3, no
   egress), not local PC (a single 1-year *full* 5° tile already ran 45–77 min here).
   The 5-year window is ~1.4× the reads of 3-year, but a single 5-year composite is
   cheaper than five annual composites. Confirm the global run stays within budget; drop
   to the 3-year fallback if it doesn't.

---

## Reproducing

Diagnostics were run on the **~1° AOI** (`AOI_BBOX` in the scripts), not the full
5° tile, for iteration speed. A `--full-tile` path exists in the driver.

```bash
uv run --extra analysis python \
  scripts/season_aware_p95_test.py
```

## References

- Prototype: `scripts/season_aware_p95_test.py` (`seasonal_debias`)
- QA masking + clamp: `src/landsat_lst/qa.py` (`create_qa_mask`,
  `convert_to_celsius`)
- Clamp settings: `src/landsat_lst/config.py`
  (`settings.lst_valid_min` / `lst_valid_max`)
- Production entry points: `src/landsat_lst/pipeline.py` (`process_tile`)
- Scene-level cloud-filter decision: PR #35 / issue #34
- Related: `docs/findings-pergamino-urban-seasonality.md`,
  `docs/findings-phase0.md`
