# ADR-008: Build the pyramid on one global sparse mosaic, not on 700 per-tile pyramids

**Status:** Partly superseded by [ADR-009](009-cog-output-and-stac-catalog.md). The shared-grid
decision stands.
**Date:** 2026-08-07
**Authors:** @nlebovits

> **Partly superseded (2026-08-12) by [ADR-009](009-cog-output-and-stac-catalog.md).** The mosaic
> decision below is superseded: the output is per-tile COGs cataloged as one STAC collection, and
> no global array is written. **The shared-global-grid clause is retained in full.** Tiles are
> still windows cut from one 1,296,000 × 432,000 grid at exactly 1/3600 degrees, loaded through
> `tiling.geobox_for_bbox`, and that is what makes the COG collection mosaicable at all. The
> "nothing in the consumption stack mosaics" finding was true of Zarr readers and does not carry
> to raster clients, which mosaic a collection of co-gridded COGs routinely. See ADR-009.

## Context

[ADR-004](004-geozarr-multiscale-overviews.md) settled the format: a GeoZarr multiscale
pyramid on Icechunk, written per 5-degree tile. It deferred the topology of the global web
layer to issue #41, which framed the choice as per-tile pyramids assembled by a tiling proxy
(Option A) against a single global sparse mosaic coarsened as one array (Option B), and
listed seam severity as something to confirm by eye.

The three references that issue cites answer it without a visual check. Two of the reasons
have nothing to do with seams.

### The multiscales convention cannot express per-tile pyramids as one layer

[zarr-conventions/multiscales](https://github.com/zarr-conventions/multiscales), UUID
`d35379db-88df-4056-af3a-620245f8e347`, is the convention `zarr_writer.py` already writes.
Each entry in its `layout` array points at exactly one `asset`: one level, one group or
array. The specification defines no tile grid, no mosaic construct, and no per-tile offset.
Absolute positioning lives in `spatial:transform` per layout entry, and `transform` inside a
layout entry carries only the relative scale between levels.

700 per-tile pyramids are therefore 700 separate multiscale datasets. They are not a global
multiscale layer, and no amount of metadata makes them one.

### Nothing in the consumption stack mosaics

`xpublish-tiles` takes a single Xarray dataset. Earthmover names on-the-fly mosaicking of
many scenes as a TiTiler capability they do not plan to support. `EarthyScience/icechunk-js`
is a read-only zarrita reader at v0.6.0; it reads one store. The proxy that Option A needs,
one that selects the right tile and then the right overview within it, does not exist in
either tool the issue names.

### The reference implementation derives the tiling from the array

In `earth-mover/icechunk-multiscales-demo`, `generate_overview_tiles` (`src/lib.py:784`)
partitions the **output overview grid** into `chunk_size` tiles, and `OverviewTile.source_slice`
(`src/lib.py:309`) computes the read window as `tile_slice * coarsen_factor`. Every read
window is an exact multiple of the coarsen factor by construction, so an internal boundary
cannot fall mid-block. Trimming happens once, at the outer edge of the global array, and the
code warns when it does (`src/lib.py:400`).

Their tiles are a work partition of an array that already exists. Ours are the unit the data
is produced in, with any global view derived from them afterward. That reversal is the seam
mechanism, and it operates at two levels.

**Native grids do not align.** `settings.resolution = 0.00027778` truncates 1/3600, and
`stac_load(bbox=..., resolution=...)` re-anchors each tile to its own bbox. Measured on
`results/phase0/N40W075`, `longitude[-1] = -69.99986555`: the tile overshoots its own `-70`
boundary by 0.484 px. Adjacent tile grids sit about 0.14 px apart, and tile shapes wobble
between `(18000, 18001)` and `(18001, 18001)`. These are sub-pixel seams at full resolution,
before any overview exists.

**Overviews trim inconsistently.** `coarsen(boundary="trim")` drops a different remainder at
each level. On the smoke store, 721 px becomes 180, 45, and 11, which is three levels
covering three different footprints while the `multiscales` layout declares a pure scale
relationship between them.

**The coarsest level fails even on an exact grid.** At exactly 1/3600, a 5-degree tile is
18000 px, and 18000 factors as 2⁴·3²·5³. The default `pyramid_factors = [4, 16, 64]` divides
cleanly at 4 and 16 and fails at 64, giving 281.25.

That last number is the whole argument in miniature. The global array at exactly 1/3600 is
1,296,000 × 432,000, which divides by 4, by 16, and by 64, landing on 20,250 × 6,750 at the
coarsest level. The divisibility problem belongs to the tile, not to the pyramid.

## Decision

> Superseded by [ADR-009](009-cog-output-and-stac-catalog.md), except for the shared-grid
> requirement two paragraphs below, which ADR-009 depends on and keeps.

**Write one global sparse mosaic per window. Tiles become work partitions, not storage units.**

Each level is a single array spanning the full global extent, with only land chunks
materialized. Ocean costs nothing, because Zarr writes no chunk that was never touched.
Per-tile groups are retired.

This also requires that tiles sit on one shared grid, which they currently do not.
`settings.resolution` becomes exactly 1/3600, guarded so that the global grid and the tile
both come out integral, and scenes load through an explicit `geobox=` sliced out of a single
global `GeoBox` rather than through `bbox` plus `resolution`. Without that, the mosaic would
inherit the same sub-pixel misregistration the per-tile layout has now.

Three of issue #41's open questions close as consequences rather than as separate decisions.

**Pyramid factors stay `[4, 16, 64]`.** Tile-divisibility stops mattering once blocks are cut
from the global array, so there is no need to trade the web-standard powers of two for a
tile-divisible set such as `[4, 16, 80]`. Earthmover's guidance holds independently: a 2×
pyramid costs about +33% and its finest level alone is roughly 75% of overview storage, a 4×
pyramid about +6.7%, an 8× about +1.6%. The existing sparse 4× default is already their
recommended starting point for continental-scale viewing.

**EPSG:4326 stays native.** Reprojecting the stored product to Web Mercator buys little,
because in Web Mercator `x` depends only on longitude and `y` only on latitude. The 4326→3857
transform preserves rectilinearity and costs `nlon + nlat` rather than `nlon · nlat`, which
is a fast path xpublish-tiles implements directly.

**Seam severity needs no visual survey.** The misalignment is arithmetic, and it is measured
above. A visual check across a former tile boundary remains worth doing once the mosaic
exists, as confirmation rather than as evidence.

## Consequences

The write path changes shape. Tile jobs move from creating a group to writing a region of an
array that a separate initialization step created, which is the two-stage pattern the demo
uses. Parallel writers move from one commit per tile with `ConflictError` retry to
`Session.fork` and `Session.merge` with batched commits; both are available in icechunk 2.0.4.

Resumability loses its current answer. `StorageBackend.zarr_exists` asks whether a tile group
exists, and under a mosaic every tile's region exists from initialization onward. Completion
has to be tracked explicitly, as the demo does with its `--skip-existing` log.

`build_overviews` keeps deriving every level from native rather than from the level above.
That stays correct and stays conservative: mean is composable, but only across exact block
counts, and deriving from native avoids depending on that.

The grid change alters the output. Tiles become uniformly 18000 px instead of wobbling
between 18000 and 18001, and pixel centers shift by up to about 0.14 px. Phase 0 tiles
predate both this and ADR-007's de-striping, so they were already due for reprocessing.

Issue #42 narrows. Its Option A, a client reading per-tile stores directly, is eliminated by
the same finding that eliminates Option A here: neither `icechunk-js` nor `xpublish-tiles`
mosaics across stores. What remains open there is whether the mosaic is served by a proxy or
read directly, and whether a derived plain-Zarr or COG export is needed for portolan.

> **Answered by [ADR-009](009-cog-output-and-stac-catalog.md) (2026-08-12).** The COG is not a
> derived export. It is the primary and only output, published as a Portolan STAC collection, and
> no proxy stands between the data and the reader. Issue #42 closes with it.

## References
- Issue #41 (this decision); issue #42 (consumption stack, still open)
- [ADR-004](004-geozarr-multiscale-overviews.md), which deferred this question
- [zarr-conventions/multiscales](https://github.com/zarr-conventions/multiscales)
- `earth-mover/icechunk-multiscales-demo`, `src/lib.py`
- Earthmover, ["Multiscale overviews in Arraylake"](https://www.earthmover.io/blog/multiscales-in-al/) (2026-06-16)
- Earthmover, ["Dynamic map tile rendering with xpublish-tiles"](https://www.earthmover.io/blog/dynamic-map-tile-rendering-icechunk-zarr-data-xpublish-tiles)
