# ADR-001: Landsat LST Pipeline Architecture Decisions

**Status:** Accepted
**Date:** 2025-05-07
**Authors:** @nlebovits

## Context

This pipeline produces global annual Land Surface Temperature (LST) composites from Landsat Collection 2 Level-2 data. The output is intended for municipal decision-makers analyzing urban heat, not for research scientists who would composite raw data themselves.

Output will be hosted on Source Cooperative as Zarr stores with STAC catalog. See [ADR-003](adr/003-direct-zarr-architecture.md) for the architecture pivot from COG+Icechunk to direct Zarr.

---

## Decisions

### 1. Data Source

**Decision:** Microsoft Planetary Computer STAC API

- **Endpoint:** `https://planetarycomputer.microsoft.com/api/stac/v1`
- **Collection:** `landsat-c2-l2`

**Rationale:** Planetary Computer provides **free egress** for Landsat data, eliminating data transfer costs entirely. Earth Search uses USGS requester-pays S3 buckets (~$0.09/GB egress), which would cost ~$630 for the full 7TB archive.

**Alternatives considered:**
- Earth Search — requester-pays S3, significant egress costs
- USGS direct — less cloud-optimized

**Note:** Phase 0 testing (2026-05) validated this decision. Token signing is handled via `planetary_computer.sign_inplace()`.

---

### 2. Landsat Missions

**Decision:** Landsat 8 and Landsat 9 only

**Rationale:** Consistent sensor characteristics (TIRS). Landsat 7 has scan line corrector failure artifacts. Historical backfill (L5/L7) is on the roadmap but not in initial scope.

---

### 3. Temporal Scope

**Decision:** Calendar year composites (January 1 – December 31)

- **Initial years:** 2021, 2022, 2023, 2024, 2025
- **Backfill:** Pre-2021 is on roadmap

**Rationale:** Calendar years align with administrative reporting cycles for municipal users.

---

### 4. Coordinate Reference System

**Decision:** EPSG:4326 (WGS84 geographic)

**Rationale:**
- STAC ecosystem expects 4326 for spatial queries
- Web map display (3857) handled by tile servers (TiTiler) on-the-fly
- Municipal users downloading COGs expect 4326
- Area-aware analysis is out of scope for target users; researchers can reproject

**Alternatives considered:**
- EPSG:6933 (equal-area) — better for area calculations but unfamiliar to users, adds complexity
- Native UTM — preserves resolution but creates ~120 zones globally

---

### 5. Tiling Scheme

**Decision:** 5° × 5° fixed grid

| Property | Value |
|----------|-------|
| Tile size | 5° latitude × 5° longitude |
| Naming | Northwest corner, e.g., `N40W075` |
| Pixels at 30m | ~18,500 × 18,500 (varies with latitude) |
| Total tiles | 1,728 (24 lat bands × 72 lon columns) |
| Land tiles | 700 (59.5% reduction from land mask) |

**Rationale:**
- Simple, predictable grid
- Good parallelism for Coiled (~800 independent tasks)
- Manageable file sizes (~500-800MB compressed per tile)
- Easy to skip ocean tiles

**Alternatives considered:**
- 10° tiles — too large (5GB+), fewer parallelism opportunities
- 1° tiles — too many (~20,000), overhead dominates
- WRS-2 path/row — complex overlaps, ~57,000 combinations

---

### 6. Spatial Filtering

**Decision:** Land surface only, ±60° latitude

| Filter | Implementation |
|--------|----------------|
| Land mask | Natural Earth 110m land polygons (hardcoded tile list) |
| Latitude bounds | 60°N – 60°S (tile NW corners from N60 to S55) |
| Water bodies | Excluded via land mask (oceans, major lakes) |

**Implementation (2026-05-08):**

The land mask is implemented as a hardcoded `frozenset` of 700 tile names in `tiling.py`, generated from Natural Earth 110m land polygons. This approach was chosen for:

1. **Zero dependencies** — no geodatasets package or runtime downloads
2. **Zero I/O** — instant frozenset lookup vs shapefile parsing
3. **Reproducibility** — deterministic tile list every run
4. **Sufficient precision** — 110m resolution is adequate for 5° (~550km) tiles

**Rationale for ±60° bounds:**
- 60°N includes all populated northern areas (Russia, Canada, Scandinavia)
- 55°S includes Tierra del Fuego (southernmost populated land)
- Excludes Antarctica (34+ tiles below -65°S with no inhabitants)
- Arctic cities (65-70°N) excluded as edge cases for urban heat analysis

**Result:** 700 land tiles vs 1,728 total = 59.5% reduction in processing.

**Alternatives considered:**
- Runtime Natural Earth lookup — adds I/O and potential network dependency
- GHS-POP population threshold — adds complexity, marginal benefit
- geodatasets package — adds dependency for one-time data generation
- 10m Natural Earth — overkill for 5° tile filtering

---

### 7. QA Filtering

**Decision:** Bitwise pixel-level masking from `qa_pixel` band

```python
def qa_mask(qa: xr.DataArray) -> xr.DataArray:
    """Returns True for GOOD (usable) pixels."""
    cloud = (qa >> 3) & 1        # bit 3: cloud
    shadow = (qa >> 4) & 1       # bit 4: cloud shadow
    snow = (qa >> 5) & 1         # bit 5: snow/ice
    return (cloud == 0) & (shadow == 0) & (snow == 0)
```

**Rationale:**
- Standard approach for Landsat science products
- Pixel-level masking preserves maximum data
- Simple binary flags (present/absent) rather than confidence levels

**Note:** Scene-level cloud cover filter (`eo:cloud_cover < 20`) applied at STAC query time to reduce data volume before pixel-level masking.

---

### 8. Day/Night Passes

**Decision:** Daytime passes only

**Rationale:**
- Urban heat island analysis focuses on daytime heating
- Consistent solar illumination conditions
- Simpler interpretation for municipal users

---

### 9. Temperature Units

**Decision:** Celsius

**Conversion:**
```python
# Landsat C2 L2 ST band scaling
lst_kelvin = lwir11 * 0.00341802 + 149.0
lst_celsius = lst_kelvin - 273.15
```

**Rationale:** Intuitive for municipal decision-makers worldwide (except US, but they'll manage).

---

### 10. Output Bands

**Decision:** Three bands per tile COG

| Band | Name | Description |
|------|------|-------------|
| 1 | `lst_p50` | Median (50th percentile) LST in °C |
| 2 | `lst_p95` | 95th percentile LST in °C |
| 3 | `qa_count` | Count of valid observations |

**Rationale:**
- p50 (median) — robust central tendency, less affected by outliers than mean
- p95 — captures extreme heat events without being max (which catches errors)
- qa_count — data quality indicator, shows observation density

---

### 11. Output Format

> **⚠️ SUPERSEDED (2026-05-08):** This section's COG decision has been superseded by [ADR-003](003-direct-zarr-architecture.md), which pivots to direct Zarr writes. The historical context below is preserved for reference.

**Decision:** ~~Cloud-Optimized GeoTIFF (COG)~~ → **Zarr v3** (see ADR-003)

| Property | Original (COG) | Current (Zarr) |
|----------|----------------|----------------|
| Format | COG | Zarr v3 |
| Compression | DEFLATE | Blosc (zstd) |
| Chunk size | 512×512 | 500×500 |
| Data type | uint16 | uint16 |
| Nodata/Fill | 0 | 0 |

**Why the pivot:** GDAL requires COG block sizes to be multiples of 16, but VirtualZarr concatenation requires array dimensions evenly divisible by chunk size. Our ~18,500 pixel tiles have no chunk size satisfying both constraints. Direct Zarr writes avoid this conflict entirely.

**Chunk size rationale (500×500):**
- 18,500 ÷ 500 = 37 exactly (no partial edge chunks)
- No GDAL multiple-of-16 constraint for Zarr
- Aligns with industry practice (Earthmover, Dynamical.org use non-power-of-2 chunks)

<details>
<summary>Historical COG rationale (superseded)</summary>

GDAL/rasterio requires COG block sizes to be **multiples of 16**. Our 5° tiles at ~30m resolution produce ~18,500 pixels per side. No multiple of 16 divides 18,500 evenly (18,500 = 2² × 5³ × 37):
- 512 ÷ 16 = 32 ✓ (GDAL-compatible)
- 500 ÷ 16 = 31.25 ✗ (fails GDAL constraint)
- 18,500 ÷ 512 = 36.13 (partial edge chunks)

Non-power-of-2 chunk sizes are standard in geospatial (when GDAL constraint is met):
- Earthmover serverless-datacube: 1200×1200 default
- Dynamical.org reformatters: 50×50, 121×121, varies by dataset
- USGS Landsat COGs: 256×256
- Microsoft Planetary Computer ERA5: 150×150

</details>

**References:**
- [ADR-003: Direct Zarr Architecture](003-direct-zarr-architecture.md) — current architecture
- [findings-direct-zarr-spike.md](../findings-direct-zarr-spike.md) — validation findings

---

### 12. Data Encoding (uint16 Packing)

**Decision:** Pack LST values as uint16 with linear scale/offset

| Property | Value |
|----------|-------|
| Scale | 0.01 |
| Offset | -50.0 |
| Nodata (packed) | 0 |
| Nodata (internal) | -9999.0 |
| Valid DN range | 1–65535 |
| Temperature range | -49.99°C to +605.35°C |

**Decode formula:**
```python
celsius = dn * 0.01 + (-50.0)
# or equivalently: celsius = (dn - 5000) * 0.01
```

**Rationale:**
- **50% storage reduction** — uint16 (2 bytes) vs float32 (4 bytes)
- **0.01°C precision** — exceeds measurement accuracy of Landsat TIRS (~0.5°C)
- **-50°C floor** — covers coldest realistic urban temperatures (pipeline excludes polar regions via ±60° latitude filter)
- **DN=0 as nodata** — follows CF Conventions `_FillValue` pattern for packed data

**Nodata handling:**
- Internal processing uses `-9999.0` (float, clearly invalid temperature)
- Encoding step maps `-9999.0 → 0` (uint16 nodata)
- This separation keeps config.py unchanged and isolates packing logic to `cog.py`

**Metadata storage:**

| Location | Purpose | Implementation |
|----------|---------|----------------|
| TIFF tags | Self-describing COG | `LST_SCALE`, `LST_OFFSET`, `LST_UNITS` per band |
| STAC properties | Catalog-level discovery | `lst:scale`, `lst:offset`, `lst:units` (see #2) |
| Icechunk attrs | Zarr/xarray access | `scale_factor`, `add_offset` per variable (see #10) |

**Rationale for TIFF tags as primary:**
- **Self-describing** — metadata travels with the file
- **Universal access** — GDAL, rasterio, QGIS all read TIFF tags
- **No external dependencies** — users don't need STAC catalog or Icechunk

**Decode helper approach:**
- **README snippet only** — users downloading COGs won't have `landsat_lst` installed
- **No package function** — avoids forcing dependency installation for simple decode
- **COG tags are primary** — snippet is backup documentation

**Alternatives considered:**
- float32 output — simpler but doubles storage cost
- int16 with different offset — narrower range, no benefit
- Package decode function — adds friction for GIS users who just want the formula

---

### 13. STAC Structure

**Decision:** One collection per year, tiles as items

```
landsat-lst-annual/
├── 2021/           # Collection: landsat-lst-2021
│   ├── N40W075/    # Item
│   ├── N40W070/    # Item
│   └── ...
├── 2022/           # Collection: landsat-lst-2022
└── ...
```

**Rationale:**
- Aligns with temporal query patterns (users query by year)
- Items are individual tiles (spatial units)
- Clean separation for incremental updates

---

### 14. Processing Infrastructure

**Decision:** Coiled on AWS

- **Cluster:** Dask distributed via Coiled
- **Region:** us-west-2 (co-located with Earth Search data)
- **Worker spec:** TBD based on profiling

**Rationale:** Coiled handles cluster provisioning, scales to global processing, data locality with S3.

---

### 15. Final Destination

**Decision:** Source Cooperative

> **Updated (2026-05-08):** Per [ADR-003](003-direct-zarr-architecture.md), output format changed from COG to Zarr.

- **Zarr stores** on Source Cooperative (one per tile)
- STAC catalog published (pointing to Zarr stores)
- Direct xarray access via `xr.open_zarr()`

**Rationale:** Source Cooperative is the appropriate home for open geospatial data products.

---

### 16. Retry/Resume with Idempotent Writes

> **Updated (2026-05-08):** Per [ADR-003](003-direct-zarr-architecture.md), Icechunk commits replaced with direct Zarr writes. The core retry/resume pattern remains.

**Decision:** Idempotent per-tile Zarr writes with existence checks

**Context:** Global processing = 10,000+ tile-year jobs. Failures are guaranteed (network errors, OOM, spot preemption). Without checkpointing, partial progress is lost and re-runs redo completed work.

**Implementation (two layers):**

| Layer | Purpose | Implementation |
|-------|---------|----------------|
| Idempotent check | Skip completed work | Check if `.zarr/` exists on S3 |
| Worker retry | Transient failure recovery | Coiled `@function(retries=3)` decorator |

**Per-tile write pattern:**
```python
def process_tile(job: TileYearJob, storage: ZarrStorage) -> str | None:
    # Layer 1: Idempotent check
    if zarr_exists(job):
        return None  # Already done

    # Process tile...
    composite = create_composite(job)

    # Layer 2: Write Zarr store
    output_path = storage.get_zarr_path(job)
    composite.to_zarr(output_path, mode="w")
    return output_path
```

**Zarr existence check:**

| Environment | Check method |
|-------------|--------------|
| Local (testing) | `Path(output_dir / "tile.zarr").exists()` |
| S3 (production) | `s3.head_object(Bucket=bucket, Key="tile.zarr/.zmetadata")` |

Factory pattern in `storage.py` abstracts this.

**CLI support:**
- `--force` flag bypasses existence check for reprocessing
- Progress logging via structlog with tile/year context

**Parallel write safety:**
- Each tile writes to its own independent Zarr store
- No conflicts possible (unlike Icechunk shared repository)
- Simpler than previous Icechunk conflict/retry pattern

<details>
<summary>Historical Icechunk pattern (superseded)</summary>

The original design used Icechunk per-tile commits with conflict retry:
- Workers opened Icechunk sessions, committed independently
- `ConflictError` caught and retried with fresh session
- This complexity is eliminated with direct Zarr writes

</details>
- Coiled checkpointing — doesn't integrate with Icechunk transactional model

---

## Consequences

### Positive
- Simple, well-understood architecture
- Good parallelism for global processing
- Output format familiar to municipal GIS users
- Cloud-native from source to destination

### Negative
- 4326 has area distortion (acceptable for target users)
- 5° tiles may be suboptimal for some regions (acceptable tradeoff)
- Calendar year composites may miss seasonal patterns (out of scope)

### Risks
- Earth Search availability/pricing changes
- Coiled cost scaling for global processing
- Source Cooperative storage limits

---

## References

- [Landsat Collection 2 Level-2 Science Products](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products)
- [Element 84 Earth Search](https://earth-search.aws.element84.com/v1)
- [COG Specification](https://www.cogeo.org/)
- [STAC Specification](https://stacspec.org/)
