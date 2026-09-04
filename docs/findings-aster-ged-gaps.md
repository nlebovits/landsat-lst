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
pass over the COG (~35 s to a few minutes over https, ~1.2 GB resident) and no
cloud compute. The committed record renders its own tables with no raster and
no granules:

```bash
uv run landsat-lst ged-analyze \
  --from-record results/decision/ged_gap_s30w065.json
```

Recomputing it needs the granule archive and the published COG:

```bash
uv run --extra analysis landsat-lst ged-analyze \
  --raster https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/nlebovits/landsat-lst/lst-p95-2021-2025/S30W065/lst_p95_2021-2025_S30W065.tif \
  --tile S30W065 \
  --ged-dir data/aster_ged \
  --out results/decision/ged_gap_s30w065.json
```

Inputs, as the record states them: the published `lst_p95_2021-2025_S30W065.tif`
(18,000 x 18,000 `uint16`, EPSG:4326, nodata 0, scale 0.01, offset -50, 4,403
scenes per its own tag), whose affine matches the global grid exactly; the 49
AG100 v003 granules the tile's padded window touches, mapped by
`ged.cell_indices_for_geobox`, the same function the production mask uses; and
a hot-tail threshold of 70 C, which is DN 12,000.

The tile holds 324,000,000 pixels: 323,916,905 valid (99.974%), 83,095 missing
(0.0256%), and 2,793 at or above 70 C (0.000862% of valid). Every rate below is
relative to those tile-wide bases.

| NumObs | pixels | valid | missing | >= 70 C | hot / valid | hot enrichment | share of hot tail | share of missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 784,080 | 701,279 | 82,801 | 2,231 | 0.318% | **369x** | 79.9% | 99.6% |
| 1 | 4,446,576 | 4,446,468 | 108 | 20 | 0.00045% | 0.52x | 0.7% | 0.13% |
| 2 | 17,160,336 | 17,160,230 | 106 | 320 | 0.00186% | 2.16x | 11.5% | 0.13% |
| 3 | 33,749,136 | 33,749,071 | 65 | 222 | 0.00066% | 0.76x | 7.9% | 0.08% |
| >= 4 | 267,859,872 | 267,859,857 | 15 | **0** | 0 | 0 | 0 | 0.02% |

Two things the table carries that a percentage alone would not. The tail is
not monotone in observation count: tiers 1 and 3 are *depleted* relative to the
tile, and only tiers 0 and 2 are enriched, so "one or two observations" is not
itself a hot-risk class. And the `NumObs >= 4` tier, 82.7% of the tile, holds
no pixel at or above 70 C at all.

The registration of output pixels to GED cells was checked by data rather than
assumed. Shifting the cell assignment by up to two cells in each axis and
re-counting the share of missing pixels that land on `NumObs == 0` peaks at
99.65% at zero shift, against 57.7% and 58.8% one cell either way. The
`Geolocation/Latitude` arrays in the granules are a linspace at 1/99 degree
spacing while the product is documented at 0.01 degree; the scan resolves that
ambiguity in favour of equal 0.01 degree cells.

#### The mask rules, on the same tile

| rule | valid removed | valid % | hot removed | hot % | hot left | missing annotated | missing % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `NumObs == 0` | 701,279 | 0.2165 | 2,231 | 79.88 | 562 | 82,801 | 99.65 |
| `NumObs == 0` + 1-cell buffer | 2,799,286 | 0.8642 | 2,582 | 92.45 | 211 | 83,018 | 99.91 |
| **the same, AND `>= 70 C`** (shipped) | **2,582** | **0.0008** | **2,582** | **92.45** | **211** | **0** | **0** |
| `NumObs <= 2` | 22,307,977 | 6.887 | 2,571 | 92.05 | 222 | 83,015 | 99.90 |
| `NumObs <= 2` + 1-cell buffer | 36,913,817 | 11.396 | 2,793 | 100.0 | 0 | 83,095 | 100.0 |

Reading the third row against the second shows an identical hot tail reached
across a difference of 2,796,704 ordinary pixels.

#### Why the mask is a conjunction

#116 shipped the second row: every pixel of the gap region, whatever its value.
Weighing the hot column against the 0.8642% beside it made the second number
look like the price of the first. Ordinary surface temperature accounts for
2,796,704 of those pixels. At 1,083 lost pixels per artifact, removing them put
visible holes in the product.

Gap geometry and gap damage are different sets. Where `NumObs == 0`, USGS
interpolated the emissivity, which describes the auxiliary input rather than
the retrieval that consumed it. Most pixels in those cells carry an ordinary
temperature. Where the retrieval did fail outright, the tile already holds no
value: before any mask runs, 99.9% of its missing pixels sit inside this
geometry. Annotating them again is work the mask never had to do.

Production therefore intersects the two conditions. Geometry locates the
interpolated emissivity, while the threshold
(`settings.ged_gap_hot_threshold_c`, 70 C) selects the pixels that
interpolation plausibly broke. Everything outside their intersection stays in
the composite.

The 70 C threshold is empirical and local. It marks where this tile's artifact
population separates from the rest of the distribution: 2,793 pixels reaching
77.87 C, enriched 369 times on `NumObs == 0` cells, with none of it on a cell
carrying more than three observations. Because the threshold never acts alone,
it carries no claim about the maximum physical land surface temperature. A
pixel above 70 C outside a gap survives. Keeping it is what stops the rule
from hardening into a global hot clamp, which none of this evidence supports.
`lst_valid_max` at 80 C remains the only unconditional ceiling.

One tile calibrated that number, which bounds what it can support. Check the
tail against 70 C on the next tile carrying a substantial gap population.

Because the conjunction bounds what the buffer can reach, the buffer stays at
1. It takes 351 more artifact pixels than the bare gap cells while touching no
ordinary ones. Its old cost of 2,098,007 additional valid pixels was entirely
collateral.

Extending the geometry to `NumObs <= 2` with the buffer removes the whole tail
at a cost of 11.4% of the tile, applied unconditionally. The 211 survivors sit
on cells carrying one to three observations, and chasing them is not worth that
price.

**0.8642%, not 0.863%.** Earlier prose quoted 0.863%. Both figures are the
same calculation over different edge handling: the 2026-08-23 pass clipped its
dilation at the tile edge, and production pads the cell window by one cell so a
gap cell just outside the tile buffers into it. S30W065's margin ring carries 6
gap cells, 2 of which buffer in, and 2 cells x 36 x 36 pixels is exactly the
2,592-pixel difference. The padded figure is the one production ships.

#### Independent verification

The cross-tab was re-derived by a second implementation that imports nothing
from `landsat_lst` and reads the granules and the published COGs directly. Its
records are tracked under `results/ged-audit/`: `crosstab_primary.json`,
`crosstab_with_qa.json`, `rule_numobs_le0.json`, `rule_numobs_le2.json`,
`registration.json`, and `inputs_manifest.json` (with a sha256 per granule and
the content length of each COG). Every tier count, every unbuffered rule, and
the registration surface agree exactly. The buffered rules differ by the
2,592-pixel clipped-versus-padded edge described above, and by 31,104 pixels
(24 cells) for the `NumObs <= 2` variant, for the same reason.

The audit also cross-tabbed `qa_count` against the LST. In every tier
`obs_any == valid` and `obs_but_fill == 0`: the published rasters cannot
separate "no clear observation" from "clear observation, no emissivity". So an
observation-level claim about *why* a pixel is missing or hot is not checkable
against anything on disk, and none is made here.

#### What this does and does not show

The durable claim is narrow. `NumObs == 0` identifies cells where ASTER GED has
no emissivity observation, USGS's surface temperature has no emissivity support
there, and the LST output stays nodata in those cells rather than carrying a
value with no physical basis. That follows from the product's definition, not
from this tile.

The 70-78 C fringe is *strongly associated* with those cells: 369 times the
tile's base rate, none of it above three observations, and a registration scan
that peaks sharply at identity. The association is spatial. Nothing here traces
which ASTER observations USGS used to retrieve any pixel's emissivity, so
whether interpolated emissivity *causes* the hot retrievals is inference from
the product documentation and the pattern, not a demonstrated mechanism. The
record says so in its own `association_only` field.

### Whether a GED source can cover production, and what "complete" means

The mask is on by default for all 700 land tiles, so the archive behind it has
to be checked against what they need. That is local arithmetic (the tile list,
the global grid, the buffer, the naming grammar) in `landsat-lst ged-coverage`,
recorded in `results/decision/ged_coverage.json`.

The 700 tiles with a one-cell buffer expect 19,300 granules. On 2026-09-04 the
local archive, fetched for the urban study above, held 8,444 of them. But
"expected and not on disk" is not by itself a defect: AG100 has no granule over
open ocean, and no offline listing can tell an ocean cell from a granule that
was never downloaded. So completeness is judged against the collection's own
inventory, persisted from one CMR query by `scripts/fetch_ged_granules.py` to
`results/decision/ged_upstream_inventory.json` (`AG1km` v003, 24,873 granules,
queried 2026-09-04T08:22:16Z). Against it:

| | granules |
|---|---:|
| expected by 700 tiles, 1-cell buffer | 19,300 |
| present in the collection | 16,926 |
| absent from the collection | 2,374 |
| of those, inside a tile core rather than its margin | 1,480 |

Of the 1,480 core cells the collection lacks, 1,396 hold no land at all and 67
hold under 10% land (Natural Earth 10m). The 17 with more are Pacific island
groups (New Caledonia, Fiji, Vanuatu, the Solomons) and Kerguelen. The land
inside all 1,480 sums to 5.53 square degrees. Where the collection has no
granule there is no emissivity, so USGS produces no surface temperature there
either; the mask contributes no gap cells over those cells and applies no
buffer to their neighbours. That is a measured, unmasked residue, not a rule
change: a cell with no `NumObs` is not a cell with `NumObs == 0`.

`AG1km.v003.-34.-066`, cited in earlier prose as a cell the collection lacks,
is in the inventory. It was a fetch gap and is now held.

**Complete** therefore means every expected granule the collection holds is
consumed. A granule the collection has but the build lacks is refused at load
with `MissingGranuleError` naming it, exactly as before. A granule the
collection lacks is recorded in the artifact and served with a structured
warning.

```bash
uv run --extra analysis python scripts/fetch_ged_granules.py      # inventory + shortfall
uv run landsat-lst ged-coverage \
  --upstream-inventory results/decision/ged_upstream_inventory.json \
  --fetch-domain data/ghsl/cells_1deg_21_22_23_30.npy \
  --out results/decision/ged_coverage.json                        # COMPLETE, exit 0
```

The shortfall was 8,482 granules, 3.9 GB, fetched in 459 s at 32 threads.

### How the mask reaches a production worker

`settings.ged_gap_mask` defaults on, so every fleet tile reaches for a mask,
and a Coiled VM has no repository checkout, no `data/`, and no granule
archive. Until #118 no source could reach one: `data/ged_gap_mask.npz` was
gitignored, sat outside the wheel's package root, and nothing uploaded it, so
every production tile would have raised `FileNotFoundError` at the mask step.

The artifact now ships inside the wheel at
`src/landsat_lst/data/ged_gap_mask.npz`, tracked in git and listed under
`[tool.hatch.build.targets.wheel] artifacts`. `ged._resolve_source` tries, in
order: the configured `settings.ged_artifact` if that file exists (an explicit
override), the packaged artifact via `importlib.resources`, then the granule
directory, and otherwise raises naming all three. A worker with no source fails
the tile rather than shipping it unmasked.

The packaged artifact's identity:

| | |
|---|---|
| format version | 3 |
| granules consumed | 17,253 (16,926 expected + 327 outside the tile list) |
| expected granules absent upstream | 2,374 |
| gap cells (`NumObs == 0`, unbuffered) | 18,526,168 |
| inventory | `AG1km` v003, 24,873 granules, 2026-09-04T08:22:16Z |
| content sha256 | `62e9ca8f22e3bd0810f1a0034197ca327c20ef0ffef94a20a75d7ac291ac058f` |
| size | 6.70 MB |

The content hash is over the canonically ordered gap cells, every consumed
granule's name and sha256, the absent-upstream list, and the inventory
identity, never over the `.npz` bytes (zip embeds timestamps). It is pinned in
`ged.GED_ARTIFACT_CONTENT_SHA256` and verified on every load, so a rebuilt
artifact that was not deliberately re-pinned is refused, and a partial build
cannot pass as this one. Rebuilding on the same archive reproduces the digest;
`scripts/build_ged_gap_mask.py --require-complete` exits non-zero on any
fetchable shortfall.

```bash
uv run --extra analysis python scripts/build_ged_gap_mask.py \
  --upstream-inventory results/decision/ged_upstream_inventory.json \
  --require-complete \
  --out src/landsat_lst/data/ged_gap_mask.npz
```

Verified on 2026-09-04: `tests/integration/test_ged_artifact_delivery.py`
builds the wheel, installs it into a clean virtualenv, and resolves the mask
from a foreign working directory with no `data/`; the packaged artifact and the
granule path produce bit-identical masks on the full S30W065 geobox (2,882,304
pixels masked by each); and the re-derived S30W065 record differs from the one
committed before the fetch in exactly one field, `ged.absent_cells` (101 to 0),
with every count and every rule unchanged.

## References

- [ADR-006: Leave ASTER GED coverage gaps empty](adr/006-no-aster-gap-filling.md)
- [USGS: Landsat Collection 2 Surface Temperature data gaps due to missing ASTER GED](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature-data-gaps-due-missing-aster-ged)
- [ASTER GED product page (LP DAAC)](https://lpdaac.usgs.gov/products/ag1kmv003/)
- [Landsat Collection 2 Level-2 Science Products](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products)
- [GHS-SMOD R2023A](https://human-settlement.emergency.copernicus.eu/ghs_smod2023.php)
