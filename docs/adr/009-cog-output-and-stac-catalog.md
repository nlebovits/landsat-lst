# ADR-009: Publish per-tile COGs in a Portolan STAC catalog, and retire the Zarr output

**Status:** Accepted
**Date:** 2026-08-12
**Authors:** @nlebovits
**Supersedes:** ADR-003 (direct Zarr writes), ADR-004 (GeoZarr multiscale overviews), and the
mosaic decision of ADR-008. ADR-008's shared-global-grid decision is retained.

## Context

[ADR-001 §11](001-architecture-decisions.md) chose Cloud-Optimized GeoTIFF.
[ADR-003](003-direct-zarr-architecture.md) reversed that choice for one reason: GDAL requires a
COG block size that is a multiple of 16, VirtualZarr requires array dimensions divisible by the
chunk size, and no chunk size satisfied both against the tile shape of the day.

That conflict belonged to VirtualZarr, and ADR-003 removed VirtualZarr in the same document. With
no virtual byte-range layer, a COG's block size answers to GDAL alone, and 512 satisfies GDAL. The
constraint that killed the COG left with the thing that created it, and nothing put the decision
back on the table until now.

[ADR-004](004-geozarr-multiscale-overviews.md) then built a GeoZarr multiscale pyramid per tile on
Icechunk, and [ADR-008](008-global-mosaic-topology.md) replaced 700 per-tile pyramids with one
global sparse mosaic. ADR-008 rested on two findings. The `multiscales` convention maps one level
to one asset, so per-tile pyramids cannot form a single global layer. And nothing in the
consumption stack mosaics, because `xpublish-tiles` takes one Xarray dataset, `icechunk-js` reads
one store, and Earthmover names on-the-fly mosaicking as a capability they do not plan to support.

Both findings are correct, and both are statements about Zarr readers. Issue #42, which asks how
the data reaches a consumer, stayed open through all of it. Every candidate answer needed
infrastructure the project does not have: a tiling proxy, a JavaScript store reader, or an
xpublish deployment to maintain. The product is meant for municipal staff opening a layer in QGIS,
and no path from a global Icechunk mosaic to that person is short.

## Decision

**Publish two Cloud-Optimized GeoTIFFs per tile on the shared global grid, cataloged as a single
Portolan STAC collection. Retire the Zarr and Icechunk output entirely.**

### Products

Each tile produces one item with two assets, both cut from the same window of the global grid.

| Asset | Type | Bands | Nodata | Encoding |
|---|---|--:|---|---|
| `lst_p95` | uint16 COG | 1 | 0 | GDAL band scale 0.01, offset -50.0, so viewers decode DN to Celsius |
| `qa_count` | uint8 COG | 12 | none | one band per calendar month, band descriptions January through December |

The uint16 encoding contract from [ADR-001 §12](001-architecture-decisions.md) is unchanged and
now lives in `landsat_lst.encoding`, so the COG writer and the tests read the same constants.
`qa_count` sets no nodata because 0 is a measurement. It means no valid observation survived
de-striping in that month, which is exactly the value a user diagnosing a gap needs to see.

### The shared global grid is retained

`tiling.global_geobox` and `tiling.geobox_for_bbox` are untouched. Every tile is a window sliced
out of the single 1,296,000 × 432,000 grid at exactly 1/3600 degrees, loaded by passing `geobox=`
to `stac_load` rather than `bbox` plus `resolution`. ADR-008 established this to stop each tile
anchoring to its own bbox, which overshot the eastern edge by 0.484 px and misregistered
neighbours by about 0.14 px.

That clause is not incidental to this ADR. It is the precondition for everything below. A
collection of co-gridded COGs mosaics. A collection of independently anchored ones does not.

### Mosaicking moves to the client, because raster clients already do it

ADR-008's objection does not carry over. Every general-purpose raster client mosaics a STAC
collection of co-gridded COGs as a matter of course:

- TiTiler serves tiles from a STAC search or a mosaic definition, reading only the overview level
  and the byte ranges a tile needs.
- GDAL builds a VRT over any list of rasters sharing a CRS and a grid, and every GDAL-backed tool
  reads that VRT as one raster. QGIS is a GDAL-backed tool.
- `odc-stac` loads a whole item collection into one array, which is the same call the pipeline
  already makes against Landsat scenes.

The convention problem disappears with it. A COG carries its overviews inside the file, so no
metadata document has to describe a global pyramid or declare a scale relationship that crosses
tiles. `multiscales` mapping one level to one asset stops mattering once no one is writing
`multiscales`.

### Overview resampling differs per product, for the reason ADR-004 gave

ADR-004 built overviews with a fill-masked mean, excluding the `-9999`/NaN fill before averaging,
because fill decodes to -50 °C and would drag overview temperatures down. That reasoning survives,
and applying it honestly produces two different rules rather than one:

- **`lst_p95` excludes nodata from the average.** DN 0 is fill. Averaging it in would report a
  coarse pixel as tens of degrees colder than any observation contributing to it.
- **`qa_count` averages its zeros deliberately.** A zero is a real count, not a missing value. A
  coarse QA pixel should report the mean observation count over the area it covers, and dropping
  the zeros would inflate that number precisely where coverage is worst.

The arithmetic is identical in both cases. The difference is what zero means, and that is a
property of the product rather than of the resampling.

### The catalog is Portolan-compliant

One collection, `lst-p95-2021-2025`, holds one item per land tile, roughly 700 of them, with the
two assets above. The window label keys the collection, so a future window becomes a sibling
collection rather than a mutation of this one.

- **Providers** use the official STAC shape, with explicit roles rather than a free-text string.
- **License** is CC0-1.0. The product exists to be used without anyone asking permission.
- **Hrefs are relative.** A self-contained catalog survives being moved, mirrored, or cloned,
  which a catalog full of absolute S3 URLs does not.
- **`items.parquet`** mirrors the item metadata as stac-geoparquet, so a consumer can filter 700
  items with one query instead of fetching 700 JSON documents.
- **A thumbnail** ships with the collection, because a catalog entry with no picture gets skipped.
- **No `styles/` directory.** Raster styling is incubating in the Portolan spec, and the STAC
  render extension's colorize-from-source path already covers this case. The LST COG carries its
  scale, offset, and band statistics, so a client has everything it needs to build a ramp from the
  file. A style document would restate that and then drift from it.

## Consequences

**Atomicity is gone, and the completion check absorbs it.** Icechunk committed native data and
every overview in one transaction, so a reader never saw a half-built pyramid. Two S3 objects
cannot land together. Completion therefore means both assets are present, and `cog_exists` checks
both. A tile whose LST asset uploaded and whose QA asset did not reads as incomplete and gets
reprocessed, which is the outcome worth having.

**Resumability gets better, not worse.** ADR-008 recorded that the mosaic broke `zarr_exists`,
because every tile's region exists from the moment the global array is initialized, and completion
had to be tracked in a side log. Per-tile COGs restore the direct answer. Existence of the two
objects is existence of the tile, with no external bookkeeping to keep in sync.

**Versioning and time travel go away.** Republishing a tile overwrites it. That was a real
Icechunk benefit and it is being spent deliberately, because the audit trail served the producer
and the mosaicability serves the consumer.

**Overview alignment across tiles holds to 16x and then drifts.** A 5-degree tile is 18,000 px,
and 18,000 = 2⁴ · 1125. Decimation by 2, 4, 8, and 16 divides the tile exactly and lands on the
global grid, so those levels mosaic seamlessly. Beyond 16x the block no longer divides, and one
tile's coarse pixel stops tiling the global grid exactly. At that point a coarse pixel is about a
kilometre across, and the offset is a fraction of it, so it falls below one screen pixel at any
zoom where such a level is selected. This is the residue of ADR-008's divisibility argument, and
it is the only part of that argument this decision does not resolve outright.

**Code goes away.** `zarr_writer.py` and `IcechunkStorage` are deleted, along with the `icechunk`
and `zarr` write-path dependencies. `landsat_lst.cog` becomes the only writer.

**Published Phase 0 stores are superseded.** They predate ADR-007's de-striping and ADR-008's
grid, so they were already due for reprocessing, and nothing is lost by not migrating them.

**Issue #42 closes.** ADR-008 left one question open at its end, asking whether a derived
plain-Zarr or COG export was needed for portolan. The answer is that the COG is not a derived
export at all. It is the primary and only output, and there is nothing left for it to be derived
from.

## References
- Issue #42 (consumption stack, closed by this decision)
- Issue #41 (grid and topology, whose grid half this decision keeps)
- [ADR-001 §11](001-architecture-decisions.md), the original COG decision, restored here
- [ADR-003](003-direct-zarr-architecture.md) and [ADR-004](004-geozarr-multiscale-overviews.md),
  superseded
- [ADR-008](008-global-mosaic-topology.md), whose grid decision is retained and whose mosaic
  decision is superseded
- [Portolan specification](https://github.com/portolan-sdi/portolan-spec) (`structure.md`,
  `formats/raster.md`, `best-practices.md`)
- [STAC render extension](https://github.com/stac-extensions/render)
