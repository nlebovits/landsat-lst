# ADR-004: GeoZarr Multiscale Overviews on Icechunk

**Status:** Accepted
**Date:** 2026-06-19
**Authors:** @nlebovits

## Context

[ADR-003](003-direct-zarr-architecture.md) established direct Zarr writes versioned
with Icechunk. The output carried GDAL-style CRS metadata (`_CRS` WKT, GeoTransform)
but **not** the GeoZarr conventions, and had **no overviews**.

For a global 30 m product (a tile is 18001 × 18001; the global grid is ~1.3M × 0.67M
px/variable), this matters for visualization. As the GeoZarr/Earthmover communities
have noted, a GeoZarr **without** a multiscales pyramid renders in a browser like a raw
GeoTIFF — the client must pull full-resolution chunks to draw anything. With
multiscales it performs like a COG with overviews. Web viewers (xpublish-tiles,
icechunk-js, deck.gl-zarr) also need the standardized `proj`/`spatial` conventions to
auto-interpret CRS and pixel geolocation.

Zarr, Icechunk, and GeoZarr are different layers, not alternatives: Zarr is the format,
Icechunk is a transactional storage *engine* on top of Zarr, and GeoZarr is a set of
metadata conventions orthogonal to the engine. We keep Icechunk (versioning + atomic
multi-array commits) and add GeoZarr metadata + overviews on top.

## Decision

`write_zarr` emits a **GeoZarr multiscale pyramid** following three conventions:
`proj` (CRS), `spatial` (affine transform), and `multiscales` (the pyramid).

1. **Layout — native moves to a level subgroup.** Native resolution is written to level
   group `0`; overviews are sibling groups `1`, `2`, `3`, …. The parent tile group
   (`{year}/{tile}`) holds no arrays — only the GeoZarr metadata. **This is a breaking
   change to the read path:** consumers must open `{year}/{tile}/0` for native data.

2. **Conventions.** The parent group carries `multiscales.layout` plus `proj:code`,
   `spatial:dimensions`, `spatial:transform`, `spatial:shape`. Each level group also
   carries `proj`/`spatial` (computed from its own coarsened coordinates) plus the
   existing GDAL `_CRS`/scale-offset attrs (kept for GDAL readers).

3. **Pyramid spacing — sparse 4x default.** `settings.pyramid_factors = [4, 16, 64]`
   (≈ +6.7% storage), tuned for mostly zoomed-out global viewing. A full 2x pyramid
   (`[2, 4, 8, 16, 32, 64]`, ≈ +33%) is available via config for smoother near-native
   zoom.

4. **Fill-masked coarsening.** Overviews are built with `xarray.coarsen().mean()`, each
   level derived from native (exact block mean, not weighted mean-of-means). The
   `-9999`/NaN fill is **excluded** before averaging — otherwise fill (DN=0 decodes to
   −50 °C) would drag overview temperatures down. `lst_p95` and `qa_count` both use mean.

5. **Compression.** All arrays are Blosc(zstd, clevel=5) compressed (previously
   uncompressed). Configurable via `settings.compression_codec`/`compression_level`.

6. **Atomicity.** For Icechunk, native + every overview + the parent metadata write into
   one session, so the caller's single `session.commit()` makes the whole pyramid atomic
   — readers never see a half-built pyramid (important for updates/backfills).

## Data Organization

```
icechunk://source-coop-radiant-earth/landsat-lst/
├── (Icechunk metadata: branches, commits, snapshots)
└── 2024/
    └── N40W075/                  # tile group: GeoZarr multiscales + proj/spatial attrs
        ├── 0/                    # native ~30 m   (scale [1, 1])
        │   ├── lst_p95/
        │   └── qa_count/
        ├── 1/                    # 4x overview    (scale [4, 4])
        ├── 2/                    # 16x overview   (scale [16, 16])
        └── 3/                    # 64x overview   (scale [64, 64])
```

## Consequences

### Positive
- Web-optimized: viewers select the overview matching the zoom level instead of reading
  full resolution at global scale.
- Standards-compliant GeoZarr (`proj`/`spatial`/`multiscales`) → auto-interpreted by
  xpublish-tiles / icechunk-js / deck.gl-zarr.
- Compression reduces storage and egress.
- Atomic pyramid commits keep all zoom levels consistent across updates.

### Negative
- **Breaking layout change:** native data is now at `{year}/{tile}/0`, not the tile
  group root. Existing Phase 0 repos predate this. Read paths/tests were updated; any
  external consumer must follow.
- ~6.7% storage overhead for the sparse pyramid (more for a full 2x pyramid).

## Open Questions (deferred)

- **Per-tile pyramids vs a global sparse mosaic** for the web layer — see issue #41.
  This ADR's per-tile pyramids validate the mechanics; the global topology is undecided.
- **Consumption stack** (icechunk-js client vs xpublish-tiles proxy vs a derived
  plain-GeoZarr/COG export) — see issue #42.

## References
- PR #40 (implementation + smoke tests)
- [ADR-003](003-direct-zarr-architecture.md) (direct Zarr + Icechunk; this builds on it)
- GeoZarr conventions: `zarr-conventions/{proj,spatial,multiscales}`
- Earthmover, "Multiscale overviews in Arraylake" (2026-06-16);
  `earth-mover/icechunk-multiscales-demo`
