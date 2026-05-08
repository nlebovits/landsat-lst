# ADR-003: Architecture Pivot to Direct Zarr Writes

**Status:** Accepted
**Date:** 2026-05-08
**Authors:** @nlebovits
**Supersedes:** ADR-002 (VirtualZarr + Icechunk Integration)

## Context

ADR-001 and ADR-002 established an architecture where:
1. LST composites are stored as **Cloud-Optimized GeoTIFFs (COGs)** on Source Cooperative
2. **VirtualZarr + Icechunk** provides a virtual Zarr layer with byte-range references to COG internal tiles
3. Users access data via `xr.open_zarr("icechunk://...")` which fetches byte ranges from COGs

This architecture aimed to provide both standalone COG access (for GIS users) and efficient Zarr/xarray access (for Python analysts) without duplicating data.

---

## Problem Discovered

During implementation (PR #14), we discovered a fundamental constraint conflict:

| Requirement | Constraint |
|-------------|------------|
| GDAL/rasterio COG creation | Block size must be **multiple of 16** |
| VirtualZarr concatenation | Array dimensions must be **evenly divisible by chunk size** |

Our 5° tiles at ~30m resolution produce **~18,500 pixels** per side. The prime factorization is:
```
18,500 = 2² × 5³ × 37
```

No multiple of 16 divides 18,500 evenly:
- 512 ÷ 16 = 32 ✓ (GDAL OK) but 18,500 ÷ 512 = 36.13 ✗ (VirtualZarr fails)
- 500 ÷ 16 = 31.25 ✗ (GDAL fails)

**Result:** The COG + VirtualZarr architecture cannot work with our tile dimensions without compromising either GDAL compatibility or Zarr concatenation.

---

## Validation: Direct Zarr Spike (Issue #15)

Before committing to a pivot, we validated that direct Zarr writes could serve our use cases:

### Test 1: Sample Zarr Creation
- Created synthetic 2-tile dataset with 500×500 chunks
- Uploaded to Source Coop: `s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst-test/sample.zarr/`
- Verified uint16 storage, CRS metadata, CF-compatible attributes

### Test 2: Python/xarray Access
```python
import xarray as xr
import fsspec

mapper = fsspec.get_mapper(
    "s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst-test/sample.zarr/N40W075",
    anon=True
)
ds = xr.open_zarr(mapper)
# Works correctly
```

### Test 3: QGIS Plugin Validation
Built minimal QGIS plugin that:
1. Reads spatial subset from remote Zarr via rioxarray
2. Converts to temporary GeoTIFF
3. Loads as QGIS raster layer

**Result:** Works on QGIS 3.28 LTS (older version without native GDAL Zarr driver).

### Findings Documented
See [findings-direct-zarr-spike.md](../findings-direct-zarr-spike.md) for detailed learnings:
- CF attribute names (`scale_factor`, `add_offset`) are auto-consumed by xarray — use prefixed names
- GDAL Zarr driver looks for `_CRS` attribute with WKT string
- Zarr v3 is now default with xarray 2024.x + zarr 3.x
- 500×500 chunks work (no GDAL multiple-of-16 constraint for Zarr)

---

## Decision

**Pivot to direct Zarr writes with Icechunk versioning, abandoning COG + VirtualZarr virtual references.**

Key distinction:
- **Removed:** COGs, VirtualZarr virtual byte-range references
- **Kept:** Icechunk for versioned storage (Git-like commits, time-travel)

### New Architecture

```
Landsat STAC → Process → Zarr writes → Icechunk Repository
                                              │
                                              ├── Python: xr.open_zarr(session.store)
                                              ├── QGIS: rioxarray → temp GeoTIFF → layer
                                              └── Versioning: time-travel, audit trail
```

### Storage Format

| Property | Value |
|----------|-------|
| Format | Zarr v3 |
| Chunk size | 500 × 500 |
| Compression | Blosc (zstd) |
| Data type | uint16 |
| Fill value | 0 |

### Data Organization

Data is stored in a single Icechunk repository with groups per tile-year:

```
icechunk://source-coop-radiant-earth/landsat-lst/
├── (Icechunk metadata: branches, commits, snapshots)
├── 2023/
│   ├── N40W075/           # Group per tile-year
│   │   ├── lst_p50/
│   │   ├── lst_p95/
│   │   └── qa_count/
│   ├── N40W070/
│   └── ...
└── 2024/
    └── ...
```

Each tile write creates a commit: `"Add N40W075 for 2023"`

**Rationale for single repository with tile groups:**
- Git-like versioning across entire dataset (time-travel, audit trail)
- Per-tile groups enable independent processing
- Conflict retry handles concurrent distributed writes
- Avoids VirtualZarr concatenation constraints entirely

---

## Consequences

### Positive
- **Simpler architecture** — no COGs, no VirtualZarr virtual references
- **Versioned storage** — Icechunk provides Git-like commits and time-travel
- **Flexible chunk size** — 500×500 aligns perfectly with tile dimensions (18,500 ÷ 500 = 37)
- **Direct writes** — no COG intermediate step
- **Lower storage** — single format instead of COG + virtual refs
- **Easier debugging** — standard Zarr tooling, no byte-range indirection

### Negative
- **No standalone COG access** — GIS users must use QGIS plugin or convert
- **Zarr ecosystem maturity** — fewer tools support Zarr natively vs COG
- **Migration effort** — existing COG-focused code needs revision

### Superseded Documents
- **ADR-002** — VirtualZarr + Icechunk Integration (fully superseded)
- **ADR-001 §11** — Output Format section (COG specifics superseded)
- **plan-virtualzarr-implementation.md** — Implementation plan (superseded)
- **docs/plans/010-virtualzarr-icechunk-implementation.md** — Issue #10 plan (superseded)

### What Remains Valid
- **ADR-001 §1-10, §12-16** — Data source, CRS, tiling, QA, encoding decisions unchanged
- **findings-phase0.md** — STAC, odc.stac, temperature conversion findings still apply
- **Per-tile processing model** — Job structure, retry/resume logic unchanged

---

## Migration Path

### Phase 1: Documentation (this PR)
- Create ADR-003 (this document)
- Mark ADR-002 as superseded (VirtualZarr virtual references only)
- Update ADR-001 §11 with supersession notice
- Update README with new architecture

### Phase 2: Implementation (this PR)
- Create `zarr_writer.py` for direct Zarr writes
- Add `IcechunkStorage` class to `storage.py`
- Update `job.py` with Icechunk commit and conflict retry
- Remove COG dependencies (`rio-cogeo`, `virtualizarr`, `virtual-tiff`)
- Keep `icechunk` dependency for versioned storage
- Update integration tests

### Phase 3: Cleanup (this PR)
- Remove `cog.py` and `virtual.py` modules
- Remove spike scripts (keep for reference)
- Archive old implementation plans

---

## References

- [Issue #15: Validate Zarr consumption path](https://github.com/nlebovits/landsat-lst/issues/15) — validation task
- [PR #14: WIP VirtualZarr + Icechunk](https://github.com/nlebovits/landsat-lst/pull/14) — blocked implementation that prompted pivot
- [findings-direct-zarr-spike.md](../findings-direct-zarr-spike.md) — technical findings
- Sample Zarr: `s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst-test/sample.zarr/`
