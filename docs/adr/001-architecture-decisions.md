# ADR-001: Landsat LST Pipeline Architecture Decisions

**Status:** Accepted
**Date:** 2025-05-07
**Authors:** @nlebovits

## Context

This pipeline produces global annual Land Surface Temperature (LST) composites from Landsat Collection 2 Level-2 data. The output is intended for municipal decision-makers analyzing urban heat, not for research scientists who would composite raw data themselves.

Output will be hosted on Source Cooperative as STAC-compliant COGs, with a virtual Zarr layer via IceChunk.

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

**Decision:** Cloud-Optimized GeoTIFF (COG)

| Property | Value |
|----------|-------|
| Format | COG |
| Compression | DEFLATE |
| Tiling | 512×512 internal tiles |
| Overviews | Yes (nearest neighbor for qa_count, average for LST) |
| Data type | uint16 (all bands) |
| Nodata | 0 (packed output) |

**Rationale:** COG is the standard for cloud-native geospatial. DEFLATE balances compression ratio and read speed.

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

- COGs stored on Source Cooperative
- STAC catalog published
- Virtual Zarr via IceChunk for analysis-ready access

**Rationale:** Source Cooperative is the appropriate home for open geospatial data products.

---

### 16. Retry/Resume with Per-Tile Commits

**Decision:** Per-tile Icechunk commits with idempotent COG checks and retry-on-conflict

**Context:** Global processing = 10,000+ tile-year jobs. Failures are guaranteed (network errors, OOM, spot preemption). Without checkpointing, partial progress is lost and re-runs redo completed work.

**Implementation (three layers):**

| Layer | Purpose | Implementation |
|-------|---------|----------------|
| Idempotent COG check | Skip completed work | `s3.head_object()` before processing |
| Per-tile commits | Durable partial progress | Icechunk commit after each tile |
| Worker retry | Transient failure recovery | Coiled `@function(retries=3)` decorator |

**Per-tile commit pattern:**
```python
def process_tile(job: TileYearJob, storage: IcechunkStorage) -> str | None:
    # Layer 1: Idempotent check
    if cog_exists(job):
        return None  # Already done

    # Process tile...

    # Layer 2: Per-tile commit with conflict retry
    while True:
        try:
            repo = Repository.open(storage)
            session = repo.writable_session("main")
            vds.virtualize.to_icechunk(session.store)
            return session.commit(f"Add {job.tile_id} {job.year}")
        except ConflictError:
            continue  # Retry with fresh session
```

**Rationale for per-tile commits (vs ADR-002 session merge):**

ADR-002 Section 4 describes session pickling + `merge_sessions()` for batch commits. We chose per-tile commits instead:

| Factor | Per-tile commits | Session merge |
|--------|-----------------|---------------|
| Durability | High — survives worker death | Low — all-or-nothing |
| Progress recovery | Resume from any point | Restart entire batch |
| Overhead | ~1 commit per tile | 1 commit per batch |
| Conflict handling | Simple retry loop | Complex merge resolution |
| Failure cost | ~minutes | ~hours |

For 10,000+ jobs with guaranteed failures, per-tile durability outweighs commit overhead. The ~1 second commit overhead is negligible vs ~5 minute tile processing time.

**Icechunk conflict resolution:**

When two workers commit concurrently, Icechunk raises `ConflictError`. Our strategy:
- **Uncooperative distributed writes** — each worker opens own session, commits independently
- **Retry on conflict** — catch `ConflictError`, reopen session, retry commit
- **No rebase** — workers process different tiles, so conflicts resolve on retry

This follows the [Icechunk parallel writes documentation](https://github.com/earth-mover/icechunk/blob/main/docs/docs/icechunk-python/parallel.md).

**COG existence check location:**

| Environment | Check method |
|-------------|--------------|
| Local (testing) | `Path(output_dir / filename).exists()` |
| S3 (production) | `s3.head_object(Bucket=bucket, Key=key)` |

Factory pattern in `storage.py` abstracts this.

**CLI support:**
- `--force` flag bypasses COG existence check for reprocessing
- Progress logging via structlog with tile/year context

**Alternatives considered:**
- Session merge (ADR-002 pattern) — too fragile for 10k jobs, partial progress lost on any failure
- External state tracking (SQLite/JSON) — unnecessary given COG existence is authoritative
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
