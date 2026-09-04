# ASTER GED Coverage Gaps: How Much Urban Land Has No Surface Temperature

**Date:** 2026-08-07
**Status:** Complete
**Analysis:** `scripts/aster_gap_urban_analysis.py`

---

## Summary

Landsat Collection 2 Level-2 Surface Temperature needs an emissivity value for
every pixel, and it takes that value from the ASTER Global Emissivity Dataset.
ASTER GED was built from clear-sky ASTER scenes acquired between 2000 and 2008.
Where ASTER never caught clear sky in those nine years, GED holds no emissivity
and USGS produces no Surface Temperature. The affected pixels are missing in
every year of the archive and no amount of reprocessing on our side recovers
them.

Measured against GHS-SMOD R2023A, **2.66% of the world's urban land has no
emissivity at all** (80,397 km² of 3,027,063 km²), and a further **10.23% rests
on one or two observations** (309,655 km²).

Where that urban land sits matters more than the worldwide average. Gaps track
persistent cloud, so they concentrate in the wet tropics and are close to absent
in deserts: **12.07% of urban Southeast Asia and 11.62% of urban Amazonia
against 0.00% of the urban Sahara and Sahel**. Averaged over every city on
Earth, 2.66% badly understates the problem for the cities that have it.

## Problem

The gaps are invisible in the product. An affected pixel looks exactly like a
pixel that happened to be cloudy: empty, with `qa_count == 0`. Nothing
distinguishes "no observation this month" from "no emissivity, ever."

That distinction drives what a user should do next. Cloud gaps close as the
compositing window grows, which is why multi-year pooling exists
([ADR-005](adr/005-multiyear-monthly-qa-and-destriping.md)). Emissivity gaps
survive every window length, because the missing input is a static auxiliary
dataset rather than a measurement. A user who widens their window to chase an
emissivity gap is spending compute on nothing.

### Where the gaps are

![ASTER GED emissivity coverage: blue where data exists, white where it does not](images/aster-ged-coverage-usgs.jpg)

*ASTER GED coverage. Blue is available data, white is none. Figure by USGS,
public domain, from [Landsat Collection 2 Surface Temperature data gaps due to
missing ASTER GED](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature-data-gaps-due-missing-aster-ged).*

This is a global land view and a binary one, so it is context rather than
evidence for any number here; every figure in this document is urban land and
comes from the per-pixel `NumObs` layer instead. It does corroborate the pattern
independently. The Sahara, Arabian Peninsula, and Australia are solid blue,
while white streaks cut through Amazonia, Indonesia, and Siberia along ASTER
orbit tracks, and Antarctica and Greenland are absent entirely.

## Method

**Gap definition.** ASTER GED ships an observation-count layer, so the
definition is direct rather than a fill-value heuristic. Each AG1km v003 granule
is a 1°×1° HDF5 tile of 100×100 pixels; `/Observations/NumObs` gives the number
of clear-sky scenes behind each pixel.

| Tier | Condition | Meaning |
|---|---|---|
| Gap | `NumObs == 0` | No emissivity. Surface Temperature never exists. |
| Low confidence | `NumObs` 1–2 | Emissivity rests on one or two scenes. |
| Normal | `NumObs >= 3` | |

**Land and water.** `/Land Water Map/LWmap` codes land as 0 and water as 1,
with −9999 as fill. This is the reverse of the convention the analysis first
assumed, and getting it backwards inverts the entire result silently: it
measures gaps over ocean and discards the land. The codes were confirmed
against the granules rather than taken on trust. Delhi (28N 77E), entirely
inland, is 0 across all 10,000 pixels. In coastal Tokyo (35N 139E) the 1-valued
pixels average 0.7 m elevation and NDVI 1.0, against 242 m and 44 for the
0-valued ones.

**Urban reference.** GHS-SMOD R2023A at 1 km, epoch 2020, on the Mollweide
grid (ESRI:54009). The urban domain is classes 21 (suburban or peri-urban),
22 (semi-dense urban cluster), 23 (dense urban cluster), and 30 (urban centre).

**Alignment.** The tier mosaic is warped into the SMOD grid by nearest
neighbour, never the reverse. Mollweide is equal-area, so at 1 km one pixel is
one km² and a pixel count converts to area without cosine weighting. SMOD itself
is never resampled, so its class codes stay exact.

**Scope.** Every figure here is urban. The denominator is always GHS-SMOD
classes 21, 22, 23, and 30, and rural and water classes are dropped from the
outputs rather than reported partially. Only 1° cells containing urban land were
fetched, 8,770 granules of the 24,873 in the collection, so every urban pixel
sits inside a fetched cell and the urban denominators are complete.

## Results

### By settlement class

| Settlement class | Land km² | Gap km² | Gap % | Low confidence % |
|---|---:|---:|---:|---:|
| Suburban or peri-urban | 1,834,726 | 43,885 | 2.39 | 10.32 |
| Semi-dense urban cluster | 299,799 | 10,206 | 3.40 | 10.79 |
| Dense urban cluster | 294,122 | 7,761 | 2.64 | 9.82 |
| Urban centre | 598,416 | 18,545 | 3.10 | 9.87 |
| **All urban** | **3,027,063** | **80,397** | **2.66** | **10.23** |

Gap share is flat across the settlement hierarchy. Dense cities are no better
served than their suburbs, which follows from the cause: cloud climatology
during 2000–2008 has nothing to do with how built-up a pixel became by 2020.

### By region

| Region | Urban land km² | Gap km² | Gap % |
|---|---:|---:|---:|
| Southeast Asia | 177,495 | 21,415 | 12.07 |
| Amazonia | 17,673 | 2,054 | 11.62 |
| Southern Africa | 35,682 | 2,983 | 8.36 |
| Central Africa | 68,034 | 4,465 | 6.56 |
| East Africa | 161,548 | 7,442 | 4.61 |
| Europe | 243,604 | 6,825 | 2.80 |
| East Asia | 705,225 | 16,526 | 2.34 |
| Andes and Southern Cone | 21,719 | 317 | 1.46 |
| North America | 188,086 | 2,227 | 1.18 |
| West Africa | 70,957 | 310 | 0.44 |
| Middle East | 76,151 | 247 | 0.32 |
| Australia | 9,514 | 29 | 0.30 |
| South Asia | 798,990 | 2,230 | 0.28 |
| Sahara and Sahel | 50,024 | 0 | 0.00 |

**The gaps follow cloud, not aridity.** Deserts come out best: the Sahara has no
urban gap at all, and Australia has 0.30%. Both are places ASTER could see the
ground almost whenever it passed. The worst-affected regions are the
persistently cloudy tropics. Any description of these gaps as a desert or
southern-Africa phenomenon is wrong, and inverts the physical mechanism.

## Limitations

**4% of urban land is unevaluated.** GHS-SMOD marks 3,152,163 km² as urban;
125,100 km² of it (4.0%) falls where ASTER GED maps water. These are coastal
cities, reclaimed land, and large river deltas where the two datasets disagree
about the coastline. Those pixels have no emissivity either, so the true urban
gap is somewhat larger than 2.66%; they are excluded rather than counted because
their cause is a land-mask mismatch and not ASTER cloud cover.

**The tier mosaic predicts gaps; it does not measure them.** It says where
emissivity is absent, which is a necessary condition for missing Surface
Temperature rather than a direct observation of it.

**Epoch mismatch is deliberate.** SMOD describes settlement in 2020 and ASTER
describes observations from 2000–2008. That is the correct pairing for the
question asked, which is how much of today's urban land is affected.

## Reproducing

Requires the `analysis` extra and a NASA Earthdata Login. Downloads about 4 GB
of granules on a cold cache and caches everything under `data/`.

```bash
uv run python -c \
  "import earthaccess; earthaccess.login(persist=True)"

uv run --extra analysis python \
  scripts/aster_gap_urban_analysis.py
```

Outputs land in `results/aster-gaps/`: `gap_by_class.csv`, `summary.json`,
`fig_gap_by_class.png`, `report.md`, and the full tier mosaic on the SMOD grid
as `gap_tiers_smod_grid.tif`.

### The published-tile cross-tab

The output mask adopted in #116 rests on a per-pixel pass over the published
S30W065 tile, cross-tabbing every output pixel by the observation count of the
GED cell it falls in. That pass is `landsat-lst ged-analyze`, and its record is
`results/decision/ged_gap_s30w065.json`. Reading is windowed, so it costs one
pass over the COG (~35 s, ~1.2 GB resident) and no cloud compute.

```bash
uv run --extra analysis landsat-lst ged-analyze \
  --raster https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/nlebovits/landsat-lst/lst-p95-2021-2025/S30W065/lst_p95_2021-2025_S30W065.tif \
  --tile S30W065 \
  --ged-dir data/aster_ged \
  --out results/decision/ged_gap_s30w065.json
```

The table is a **spatial association** between output pixels and ASTER
observation counts. It does not trace which observations USGS used to retrieve
any pixel's emissivity, so on its own it cannot show that interpolated
emissivity *caused* a hot retrieval.

### Whether a GED source can cover production at all

The mask is on by default for all 700 land tiles, so the archive backing it has
to be checked against what they need. That is local arithmetic -- the tile list,
the global grid, the buffer, the naming grammar -- and it is
`landsat-lst ged-coverage`, recorded in `results/decision/ged_coverage.json`.

```bash
uv run landsat-lst ged-coverage \
  --ged-dir data/aster_ged \
  --fetch-domain data/ghsl/cells_1deg_21_22_23_30.npy \
  --out results/decision/ged_coverage.json
```

The local archive does **not** cover production: 19,300 granules are needed and
8,444 of them are held, leaving 615 of 700 tiles missing a granule that falls
inside the tile rather than its margin. That follows directly from the fetch
described above, which requested urban cells only. No artifact is packaged into
the wheel until this reports complete, because an artifact built from a partial
archive reads exactly like one built over a region with no gaps.

## References

- [ADR-006: Leave ASTER GED coverage gaps empty](adr/006-no-aster-gap-filling.md)
- [USGS: Landsat Collection 2 Surface Temperature data gaps due to missing ASTER GED](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature-data-gaps-due-missing-aster-ged)
- [ASTER GED product page (LP DAAC)](https://lpdaac.usgs.gov/products/ag1kmv003/)
- [Landsat Collection 2 Level-2 Science Products](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products)
- [GHS-SMOD R2023A](https://human-settlement.emergency.copernicus.eu/ghs_smod2023.php)
