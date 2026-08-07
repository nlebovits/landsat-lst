# ADR-006: Leave ASTER GED coverage gaps empty

**Status:** Accepted
**Date:** 2026-08-06
**Authors:** @nlebovits

## Context

Landsat Collection 2 Level-2 Surface Temperature is derived using emissivity from the
ASTER Global Emissivity Dataset (GED), built from clear-sky ASTER scenes acquired
2000-2008. Where ASTER never caught clear sky in that window, GED has no emissivity, and
USGS produces no Surface Temperature. The affected pixels are permanently absent from
the upstream product in every year, 1982 to present.

These are not observation gaps that more scenes would close. Our multi-year pooling
(ADR-005) fills per-month coverage holes precisely because those come from cloud cover
on individual overpasses. An emissivity gap survives every window length, because the
missing input is a static auxiliary dataset rather than a measurement.

`scripts/aster_gap_urban_analysis.py` measures the footprint against GHS-SMOD; the
narrative and numbers live in
[`docs/findings-aster-ged-gaps.md`](../findings-aster-ged-gaps.md). Across the global
urban domain, 2.66% of land has no emissivity and a further 10.23% rests on one or two
observations.

The gaps follow persistent cloud, so they concentrate in the wet tropics: 12.07% of
urban Southeast Asia and 11.62% of urban Amazonia, against 0.00% for the Sahara and
Sahel and 0.30% for Australia. Deserts are the best-covered places in the product,
because ASTER could see the ground on almost every pass.

Alternative emissivity sources exist. ASTER GED v4.1 and the UW-Madison CAMEL combined
emissivity database both offer broader coverage, and either could in principle be used
to recompute Surface Temperature where GED is absent.

## Decision

**Leave the gaps empty. Do not substitute an alternative emissivity source.**

The pipeline reports gaps rather than filling them:

1. `qa_count == 0` over land marks affected pixels. The band already ships as a 12-band
   monthly climatology COG, so the signal costs nothing extra to publish.
2. The land mask (`masks.py`) distinguishes ocean from gap, so an empty pixel inside the
   land mask is unambiguous.
3. `README.md` documents the limitation, and the findings doc quantifies it.

## Consequences

**What this buys.** Every LST value in the product comes from one processing chain with
one emissivity source. A user comparing two cities compares like with like. Filling gaps
would make the emissivity source vary by pixel, in a pattern determined by 2000-2008
ASTER cloud cover, which correlates with climate and therefore with the very signal the
product measures. That bias would be invisible in the output and impossible to express
in `qa_count`.

**What it costs.** Users in affected regions get holes rather than estimates, and the
burden falls unevenly. A city in Southeast Asia or Amazonia loses roughly one pixel in
eight to gaps where a Saharan city loses none, and no amount of reprocessing on our side
changes that.

**What would reopen this.** A gap-filled product is a legitimate future variant, but as
a separate, clearly labelled layer with its own provenance band, not as silent
substitution inside this one. Recomputing Surface Temperature from an alternative
emissivity source also means reimplementing the USGS single-channel algorithm, which is
a far larger scope than compositing.
