# Findings: Direct Zarr Write Spike (2026-05-08)

**Context:** Validated direct Zarr writes (bypassing COGs) for QGIS plugin consumption.
**Issue:** #15 (closed — validation successful)
**Outcome:** ✅ **Architecture pivot approved** — see [ADR-003](adr/003-direct-zarr-architecture.md)
**Sample data:** `s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst-test/sample.zarr/`

---

## 1. CF Encoding Gotcha

**Problem:** Standard CF attributes `scale_factor` and `add_offset` are auto-consumed by xarray on read, converting uint16 → float64.

```python
# What we wrote
ds["lst_p50"].attrs = {"scale_factor": 0.01, "add_offset": -50.0}
ds.to_zarr(path)

# What xarray returns on read
ds = xr.open_zarr(path)
ds["lst_p50"].dtype  # float64, not uint16
ds["lst_p50"].attrs  # scale_factor/add_offset are GONE
```

**Solution:** Rename to non-CF attribute names (`lst_scale_factor`, `lst_add_offset`) to preserve them for GDAL/QGIS while storing raw uint16.

**Implication for pipeline:** If we want QGIS to see the encoding attrs, we cannot use standard CF names. Document this in consumer-facing docs.

---

## 2. GDAL CRS Compatibility

**Problem:** GDAL/QGIS needs CRS info to georeference Zarr data. rioxarray uses `spatial_ref` coordinate, but that requires rioxarray on the reading side.

**Solution:** Use `_CRS` attribute with WKT string — this is what GDAL's Zarr driver looks for.

```python
from pyproj import CRS

ds.attrs["_CRS"] = CRS.from_epsg(4326).to_wkt()
ds.attrs["crs"] = "EPSG:4326"  # human-readable backup
```

**Verified:** The sample Zarr includes `_CRS` and opens correctly via GDAL tooling.

---

## 3. Zarr v3 is Now Default

**Observation:** xarray 2024.x + zarr 3.x writes Zarr v3 format by default.

```
zarr_format: 3
```

**Implications:**
- Older zarr clients (v2) cannot read the data
- Some ecosystem tools may not support v3 yet
- Consolidated metadata warning appears (v3 spec doesn't standardize it)

**Decision needed:** Do we pin to Zarr v2 for broader compatibility, or proceed with v3?

---

## 4. Tile-as-Group Structure

**Approach tested:** Each tile is a separate Zarr group, not concatenated:

```
sample.zarr/
├── N40W075/
│   ├── zarr.json
│   ├── lst_p50/
│   ├── lst_p95/
│   └── qa_count/
└── N45W075/
    └── ...
```

**Pros:**
- No heterogeneous dimension handling (tiles can vary in size)
- Simpler writes (no concat coordination)
- Natural partitioning for distributed processing

**Cons:**
- Client must know tile names or enumerate groups
- No single `xr.open_zarr()` for entire dataset

**For QGIS plugin:** This is fine — plugin will request specific AOI, we map to tiles.

---

## 5. Source Coop Access Pattern

**Anonymous S3 access works:**

```python
import fsspec
import xarray as xr

url = "s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst-test/sample.zarr/N40W075"
store = fsspec.get_mapper(url, anon=True)
ds = xr.open_zarr(store)
```

**Public HTTPS also works:** `https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/...`

---

## 6. uint16 Storage Verified

**Confirmed:** Despite xarray reading as float32/64, the underlying Zarr stores uint16:

```json
// zarr.json
{
  "data_type": "uint16",
  "shape": [500, 500],
  "chunk_grid": {"configuration": {"chunk_shape": [500, 500]}}
}
```

**Storage efficiency:** 500×500 uint16 = 500KB per variable per tile (before compression).

---

## 7. Chunk Size Freedom (No GDAL Constraint)

**Key realization:** The 512×512 constraint in ADR-001 §11 applies to **COG internal tiles** because GDAL/rasterio requires blocksize to be a multiple of 16.

**For direct Zarr writes:** No such constraint exists. We can use 500×500 chunks (or any size).

```python
# COG path (constrained):
blocksize = 512  # Must be multiple of 16

# Direct Zarr path (unconstrained):
chunks = (500, 500)  # Any size works
```

**Implication:** If we pivot to direct Zarr, we regain the 500×500 chunk size that was originally preferred for even grid division.

---

## 8. QGIS Plugin Validation ✅ PRODUCTION READY

**Status:** Complete — validated for production (2026-05-08)

Built minimal QGIS plugin that demonstrates the data path:

```
Remote Zarr on Source Coop
    │
    ▼ fsspec.get_mapper(url, anon=True)
    │
    ▼ xr.open_zarr(mapper)
    │
    ▼ .rio.clip_box(minx, miny, maxx, maxy)  # spatial subset
    │
    ▼ .rio.to_raster(temp_path)  # write temp GeoTIFF
    │
    ▼ QgsRasterLayer(temp_path)  # load into QGIS
```

**Tested on:**
- QGIS 3.28 LTS (no native GDAL Zarr driver)
- rioxarray backend for spatial operations

**Result:** Works correctly. Users can query arbitrary AOIs, the plugin fetches only the required chunks from Source Coop, converts to temp GeoTIFF, and displays in QGIS.

**Key insight:** The rioxarray → GeoTIFF path is actually preferable for QGIS users because:
- Handles CRS/transform automatically
- Produces standard GeoTIFF that all QGIS versions understand
- Only downloads the spatial subset, not the entire tile

---

## Outcome: Architecture Pivot Approved

Based on the successful validation:

1. ✅ **QGIS plugin works** — rioxarray → temp GeoTIFF → QGIS layer path validated
2. ✅ **Direct Zarr writes adopted** — see [ADR-003](adr/003-direct-zarr-architecture.md)
3. ⏸️ **Zarr v2 vs v3** — proceeding with v3 (xarray default); will revisit if ecosystem issues emerge

---

## Code Reference

- Sample Zarr script: `scripts/create_sample_zarr.py`
- Constants: `src/landsat_lst/cog.py` (LST_SCALE, LST_OFFSET)
- Tile parsing: `src/landsat_lst/tiling.py` (parse_tile_name)
- Architecture decision: [ADR-003](adr/003-direct-zarr-architecture.md)
