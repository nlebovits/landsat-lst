# Architecture Decision Records

Records of significant architecture decisions for the Landsat LST pipeline. Each ADR
captures the context, the decision, and its consequences at a point in time. Superseded
ADRs are kept for history and link to their replacement.

| ADR | Title | Status |
|-----|-------|--------|
| [001](001-architecture-decisions.md) | Pipeline architecture decisions (data source, CRS, tiling, temporal scope, QA, encoding) | Accepted |
| [002](002-virtualzarr-icechunk-integration.md) | VirtualZarr + Icechunk integration | ⚠️ Superseded by [003](003-direct-zarr-architecture.md) |
| [003](003-direct-zarr-architecture.md) | Direct Zarr writes + Icechunk versioning | Accepted |
| [004](004-geozarr-multiscale-overviews.md) | GeoZarr multiscale overviews on Icechunk | Accepted |
| [005](005-multiyear-monthly-qa-and-destriping.md) | Multi-year composites, monthly QA climatology, and de-striping | Accepted (de-striping settled in [007](007-scene-normalization.md)) |
| [006](006-no-aster-gap-filling.md) | Leave ASTER GED coverage gaps empty | Accepted |
| [007](007-scene-normalization.md) | Per-scene normalization against a monthly climatology; discard uncorrectable scenes | Accepted |

## Adding an ADR

1. Copy the format of an existing ADR (`# ADR-NNN: Title`, then `**Status:** / **Date:** / **Authors:**`, then `## Context` / `## Decision` / `## Consequences`).
2. Name the file `NNN-kebab-case-title.md` with the next number.
3. Add a row to the table above. If it supersedes an earlier ADR, mark that one
   `⚠️ Superseded by [NNN](...)` and link back.
