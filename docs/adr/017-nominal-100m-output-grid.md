# ADR-017: A nominal ~100 m delivered grid, aggregated before the percentile

**Status:** Accepted
**Date:** 2026-09-01
**Authors:** @nlebovits

Supersedes the output-resolution half of [ADR-008](008-global-mosaic-topology.md). The
topology argument there — one global array, an integer pixel density, tiles cut from the
grid rather than anchored to their own bounds — is unchanged and now applies to two grids
instead of one.

## Context

USGS acquires Landsat 8/9 TIRS thermal radiance at **100 m** and resamples it, with cubic
convolution, onto the delivered 30 m Collection 2 grid. The Level-2 surface-temperature
retrieval also leans on ASTER GED emissivity at roughly 1 km, itself interpolated.
Delivery spacing and resolving power are different things, and publishing the delivered
30 m cells as independent thermal observations overstates what the product knows.

The current pipeline runs its most expensive operations across those 30 m cells. Issue
#120 put the question directly: is the 30 m detail scientifically defensible, and is it
worth the compute and publication cost?

Two failure modes had to be avoided while answering it.

**Downsampling a finished P95 is a different statistic.** The mean of nine percentiles is
not the percentile of the pooled observations, and it saves none of the percentile work
— it computes the expensive thing first and then throws most of it away.

**Coarsening is not de-striping.** Reducing 3×3 blocks makes a WRS-aligned seam less
sharp without making it less wrong. Issue #119 remains the release blocker on its own
terms, and nothing here may be presented as addressing it.

## Decision

**V1 publishes a single nominal ~100 m product**, on the existing global EPSG:4326 tiling
at an exact 3× coarser grid. There is no parallel 30 m rendering.

- `1/1200` degree spacing, **6,000 × 6,000** pixels per five-degree tile;
- the same tile bounds, names, adjacency, and CRS as before;
- exactly nine times fewer output pixels than the `1/3600`, 18,000² grid;
- the global delivered array is 432,000 × 144,000.

This is a **nominal ~100 m geographic grid, not a claim of 100 × 100 m cells**. Physical
east-west extent varies with latitude: about 93 m at the equator and about 46 m at 60°.
That limitation is stated wherever resolution is described, in the catalog and in the
methodology, and not only here.

A true global equal-area 100 m redesign is out of scope for V1. It would be
scientifically tidier and would require a CRS change, a new grid index, re-mapped land
and GED masks, rewritten COG transforms, and a rebuilt catalog. The 3× grid is the
version that ships.

### Three grids, named apart

The one change most likely to be got wrong is conflating grids that used to be one, so
they are separate settings, separate geobox constructors, and separately documented.

| Grid | Setting | Spacing | What lives on it |
|---|---|---|---|
| **Source** | `pixels_per_degree` (3600) | 1/3600° | Scene loads, solar-day fusion, QA, scaling, the clamp |
| **Offset** | `source / destripe_offset_resolution_factor` (2) | 1/1800° | Per-scene offset estimation, and nothing else |
| **Delivered** | `output_pixels_per_degree` (1200) | 1/1200° | The correction, the P95, `qa_count`, both masks, both COGs |

**The offset grid did not move, and must not.** Its accuracy bound was calibrated against
the source grid ([findings](../findings-offset-subsampling.md)), and
`destripe_min_scene_pixels` counts its pixels. So `compute_annual_composite` estimates on
the source or offset side and applies on the delivered side. Deferring the application
costs nothing: subtracting one scalar per scene commutes exactly with an area-weighted
mean over that scene's cells.

### Processing order

Aggregation sits after masking and before the correction and the percentile.

```
1. load and fuse overlapping granules into one solar-day observation   SOURCE grid
   (odc-stac groupby="solar_day", deterministic overlap priority)
2. QA mask, DN=0 fill test, Collection 2 scaling, plausibility clamp   SOURCE grid
3. aggregate aligned 3x3 blocks: area-weighted mean over valid cells,  -> DELIVERED
   at least 5 of 9 valid, else nodata for that observation
4. apply the per-scene offset, and reject the scenes that fail the cap DELIVERED
5. pooled 2021-2025 P95 over the surviving delivered observations      DELIVERED
6. monthly qa_count over that same population                          DELIVERED
7. output-only land and ASTER GED policy; no temperature is invented   DELIVERED
```

Computing the P95 at 30 m and downsampling afterwards is explicitly non-compliant.

### The valid-area rule

Each delivered cell is the **area-weighted mean of the valid source cells** in its aligned
3×3 block, and needs **at least 5 of 9** valid. Below that the cell is nodata for that
solar-day observation.

An invalid cell contributes nothing to the numerator *and* nothing to the denominator, so
a fill value or a zero can never enter the mean. That is the whole of the rule, and it is
easy to get wrong: `fillna(0).mean()` would report 16.7 °C for a block holding five clear
cells at 30 °C.

Weights come from the cells' real areas. On a sphere a cell spanning constant `dlon`
between two latitudes has area proportional to `sin(lat_top) - sin(lat_bottom)`, so within
one block the three *rows* differ and the three columns do not. The variation is small:
relative spread across a block is `tan(lat) * (factor - 1) * dlat`, which at 60° — the
edge of the published band, and the worst case — is 1.7e-5. An unweighted mean would agree
to well inside float32. The weights are computed anyway, because a reducer whose
correctness rests on the error being small is a reducer nobody can check, and because the
same code has to stay right if the factor or the grid moves.

`5/9` is the pre-registered default, with `1/9` and `9/9` as the sensitivity arms
(`landsat-lst sensitivity`, bounds in `landsat_lst.sensitivity`). **The threshold is not
tuned after looking.** An unstable result is a finding to report, not a menu.

### `qa_count` counts delivered observations

Month M is the number of **nominal ~100 m solar-day observations** in month M that met the
valid-area rule, pooled across the window. It is never a sum of 30 m counts, and it never
exceeds the number of solar days in that month. The percentile and the counts read the
same NaN pattern, so the two describe one population by construction.

## Consequences

**The read does not fall.** Every delivered cell is reduced from nine source cells, and
those nine are still fetched and decoded. `budgets._native_bytes` and
`projection.native_pass_gb` stay on the source shape deliberately: a composite budget
scaled by the output grid would give the stage a ninth of the time its reads need.

**What falls is everything downstream.** Uncompressed output goes from 4.54 GB to 0.50 GB
per tile, an exact nine. Measured locally on synthetic geometry
(`results/decision/aggregation_cost_local.json`), peak RSS falls with scene depth — 0.90×
at 24 scenes, 0.79× at 60, 0.67× at 120 — because the scene-dependent term is the
single-time-chunk rechunk, which is nine times smaller on the delivered grid, while the
flat terms are not. At 2,930 production scenes that term dominates.

**The graph gets bigger.** Three coarsen reductions over the full stack add tasks: 2.53×
at 24 scenes, 3.96× at 120. ADR-013's single native pass survives — 1.00 passes in every
arm — because within one `dask.compute` each source key is produced once whatever is
downstream.

**`R_COMPOSITE_MB_S` is unchanged and unverified.** It is a decode rate measured before
this change. Whether reducing the percentile's working set moves it is an open question
that an acceptance run answers; `landsat_lst.projection` does not presume it, and its
projections must not be scaled by the pixel-count ratio.

**Row bands are cut over delivered rows.** `shards.band_edges` sees 6,000 rows, and a
composite shard reads source rows `[3*start, 3*stop)`. Band boundaries are multiples of
the COG block size and three times a multiple of 512 is still a whole number of source
cells, so no aligned block is ever split across two bands. The delivered remainder is
different and the rule is not: 6,000 = 512 × 11 + 368.

**The plan digest covers the aggregation contract.** `output_pixels_per_degree`,
`min_valid_source_cells`, and `aggregate.AGGREGATION_VERSION` all enter it, so a shard
running a different contract refuses the plan rather than contributing a band that merges
cleanly and means something else.

**Native chunks round up to a whole number of delivered cells** (512 → 513). A straddling
chunk makes `coarsen` rechunk the stack first, and unevenly. Correctness is unaffected;
this is about not paying a shuffle for a reduction that has none.

**Delivered coordinates come from the geobox, not from averaged source labels.** The mean
of three source centres lands on the delivered centre only up to float64 round-off, which
is enough to look right and not enough to *align* — xarray joins a mask to a stack on
exact index equality. The grid definition is authoritative, for the same reason
`pixels_per_degree` is an integer.

**Overviews still belong to the global array.** A delivered five-degree tile is 6,000 px,
and 6,000 = 2⁴ · 3 · 5³. It divides by 4 and by 16 but not by 64, exactly as 18,000 did.
ADR-008's conclusion is unchanged.

**Accuracy is inherited.** Absolute temperature accuracy comes from USGS Collection 2.
This project validates its own masking, aggregation, solar-day fusion, percentile,
encoding, and publication; it does not independently validate the retrieval, and
aggregation reconstructs neither native TIRS nor ASTER measurements.

## Alternatives considered

**A true metric 100 m grid.** Scientifically cleaner and correct about physical cell size.
Rejected for V1 on scope: it changes the CRS, the tile index, both masks, every COG
transform, and the catalog. Revisit for V2.

**Downsampling a finished 30 m P95.** Rejected on both grounds that matter: it is a
different statistic, and it does none of the intended work.

**Publishing 30 m as a companion rendering.** Rejected. Two products invite the reading
that the finer one carries independent thermal information, which is the claim this ADR
exists to avoid making.

**Keeping 30 m.** Rejected. The detail is not independently supported below 100 m, and it
costs nine times the output for it.
