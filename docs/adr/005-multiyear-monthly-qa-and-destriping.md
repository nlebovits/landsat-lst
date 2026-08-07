# ADR-005: Multi-Year Composites, Monthly QA Climatology, and De-striping

**Status:** Accepted (multi-year windows, monthly QA, QA masking + physical clamp).
Season-aware de-striping is no longer deferred: it was productionized in
[ADR-007](007-scene-normalization.md), which adds the scene-rejection policy this
ADR left open.
**Date:** 2026-07-02
**Authors:** @nlebovits

## Context

The product is a global extreme-heat **P95 Land Surface Temperature** map for
urban-planning / public-health users — a "conversation-starter v1". Two problems in the
single-year composite blocked that goal:

1. **Coverage holes.** A single year leaves per-month zero-coverage gaps — e.g. ~17% of
   pixels have no valid November observation — and a thin sample for a robust P95.
2. **Scene-footprint striping.** Visible WRS scene seams (~1–5 K steps) survive QA
   masking. These are the per-scene atmospheric-correction residual in USGS Collection-2
   Level-2 Surface Temperature, plus per-scene-edge contaminants (thin/dilated cloud,
   cirrus, DN=0 resampling artifacts) that leak through a too-narrow QA mask.

The narrative — measurements, plots, and the full alternative sweep — lives in
[`docs/findings-destriping-and-multiyear.md`](../findings-destriping-and-multiyear.md).
This ADR records only the decisions.

## Decision

### 1. Multi-year composite windows — 5-year P95 is the production default

`ProcessingJob` gains a multi-year window (`end_year`, `window_label`); P95 is pooled
across **all** scenes in the window. **5 years is the default**, chosen for consistency
with established products (WRI uses a 5-year window), which makes the methodology easy to
defend. Both 3- and 5-year eliminate the per-month zero-coverage holes a single year has
and give a robust P95; 5-year is marginally hotter (mean 42.0 vs 41.0 °C at the AOI) but
not measurably cleaner. Storage is ~identical across windows (see #2), so the window is a
**compute-vs-defensibility** choice, not a storage or quality one. Note a single 5-year
composite is *cheaper* than five separate annual composites (one scene load + one P95 +
one write, versus five), though ~1.4× the reads of 3-year. **3-year remains a supported,
lighter alternative** when compute is constrained.

### 2. `qa_count` becomes a 12-month climatology, uint8

`qa_count` changes from a single annual count (int16→uint16) to a per-calendar-month
climatology: dims `(month, y, x)`, dtype **uint8**, where month *M* = valid observations
in calendar month *M* pooled across the window. It diagnoses seasonal coverage gaps and,
being per-calendar-month, is the **same size regardless of window length**. Global
product footprint: **~311 GB compressed / ~2.47 TB uncompressed** (measured LST ~1.8×,
QA ~18× compression). Monthly-over-single QA costs only ~107 GB compressed globally.

### 3. Improved QA masking + physical clamp

- `create_qa_mask` additionally masks **dilated-cloud (bit 1)** and **cirrus (bit 2)** on
  top of cloud / shadow / snow — the standard Collection-2 clear-sky bit set.
- `convert_to_celsius` clamps to a configurable physical range
  (`lst_valid_min` / `lst_valid_max`, default **−50 / 80 °C**), dropping ~−124 °C DN=0
  resampling artifacts and high-DN saturation junk.

These per-scene-edge contaminants drive scene-footprint striping; masking removes them
without touching the temperature signal.

### 4. De-striping via season-aware per-scene normalization — chosen approach, productionization DEFERRED

The seams that survive masking are the per-scene atmospheric-correction residual
(~1–5 K). The chosen fix de-biases **each scene** relative to a **per-pixel monthly
climatology** (not the annual mean), which preserves the seasonal / hot signal.
Validated on the 3-year AOI: seams removed, no artifacts, mean **41.0 °C** preserved.

**Status: prototype only** in `scripts/season_aware_p95_test.py`. Productionizing into
`pipeline.py` / `process_tile` is **deferred** and tracked in #46. This is a
public "conversation-starter" product, not a peer-reviewed claim, so independent-reference
validation (MODIS / ECOSTRESS) is **recommended for added confidence, not a prerequisite
for shipping**.

## Rejected alternatives

Recorded so they are not revisited:

- **Lower the percentile.** P50 is the *stripiest*; P95 is the most seam-robust — the hot
  tail is consistent across footprints. Keep P95.
- **Scene-level cloud-cover pre-filtering.** Negligible effect once pixel QA is good
  (r = 0.98). Not worth the dropped scenes.
- **L8/L9 cross-sensor harmonization.** Non-issue — C2 L2 ST is already
  cross-calibrated and atmospherically corrected.
- **Blunt normalization to the annual median.** Removes the season → cools the composite
  to 29.8 °C. Rejected in favor of the monthly-climatology reference (#4).

## Consequences

### Positive
- No per-month coverage holes; ~3× more robust P95 for the global v1.
- Monthly QA climatology makes seasonal coverage gaps auditable at ~fixed cost.
- Broader clear-sky masking + physical clamp remove the contaminants that cause most
  visible striping, before any normalization.

### Negative / Open questions
- **De-striping is not yet in the production path.** It must be moved into
  `pipeline.py` / `process_tile` before the global run.
- Season-aware normalization is heavier-handed than pure bias removal: the monthly
  reference also absorbs day-to-day weather, so it needs an **outlier cap** (required
  hygiene). Independent-reference validation (MODIS / ECOSTRESS) is a **recommended
  confidence check, not a blocker** — the method is a standard correction for a documented
  ~1–5 K per-scene error and is unlikely to be challenged for this product's use.
- 5-year is the default (WRI-aligned); 3-year is the lighter fallback. The main cost is
  global-scale reads (5 years of Landsat per tile) — a Coiled / Earth Search job, not local.

## References
- [`docs/findings-destriping-and-multiyear.md`](../findings-destriping-and-multiyear.md)
  (narrative: measurements, plots, full alternative sweep)
- `scripts/season_aware_p95_test.py` (de-striping prototype)
- [ADR-004](004-geozarr-multiscale-overviews.md) (output layout this composite writes into)
