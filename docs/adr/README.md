# Architecture Decision Records

Records of significant architecture decisions for the Landsat LST pipeline. Each ADR
captures the context, the decision, and its consequences at a point in time. Superseded
ADRs are kept for history and link to their replacement.

| ADR | Title | Status |
|-----|-------|--------|
| [001](001-architecture-decisions.md) | Pipeline architecture decisions (data source, CRS, tiling, temporal scope, QA, encoding) | Accepted (§11 output format restored to COG by [009](009-cog-output-and-stac-catalog.md)) |
| [002](002-virtualzarr-icechunk-integration.md) | VirtualZarr + Icechunk integration | ⚠️ Superseded by [003](003-direct-zarr-architecture.md) |
| [003](003-direct-zarr-architecture.md) | Direct Zarr writes + Icechunk versioning | ⚠️ Superseded by [009](009-cog-output-and-stac-catalog.md) |
| [004](004-geozarr-multiscale-overviews.md) | GeoZarr multiscale overviews on Icechunk | ⚠️ Superseded by [009](009-cog-output-and-stac-catalog.md) |
| [005](005-multiyear-monthly-qa-and-destriping.md) | Multi-year composites, monthly QA climatology, and de-striping | Accepted (de-striping settled in [007](007-scene-normalization.md)) |
| [006](006-no-aster-gap-filling.md) | Leave ASTER GED coverage gaps empty | Accepted |
| [007](007-scene-normalization.md) | Per-scene normalization against a monthly climatology; discard uncorrectable scenes | Accepted |
| [008](008-global-mosaic-topology.md) | Global sparse mosaic for the pyramid, and one shared tile grid | Shared-grid decision accepted; mosaic decision ⚠️ superseded by [009](009-cog-output-and-stac-catalog.md) |
| [009](009-cog-output-and-stac-catalog.md) | Per-tile COGs in a Portolan STAC catalog, retiring the Zarr output | Accepted |
| [010](010-coiled-batch-for-distributed-runs.md) | Coiled Batch for distributed runs, replacing Coiled Functions | Accepted |
| [011](011-static-planning-and-synthetic-benchmarks.md) | Static graph planning, synthetic-geometry benchmarks, and per-key profiling | Accepted |
| [012](012-cached-scene-offsets.md) | Input-keyed scene-offset cache and a standalone offset phase | Accepted |
| [013](013-single-native-pass.md) | One pass over the native stack per tile: shared rechunk, fused export writes, coverage from the raster | Accepted |
| [014](014-run-self-explanation.md) | One state object per attempt, priced runs, and a watch UI that keeps its history | Accepted |

## Adding an ADR

1. Copy the format of an existing ADR (`# ADR-NNN: Title`, then `**Status:** / **Date:** / **Authors:**`, then `## Context` / `## Decision` / `## Consequences`).
2. Name the file `NNN-kebab-case-title.md` with the next number.
3. Add a row to the table above. If it supersedes an earlier ADR, mark that one
   `⚠️ Superseded by [NNN](...)` and link back.
