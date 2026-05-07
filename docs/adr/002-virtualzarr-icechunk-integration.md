# ADR-002: VirtualZarr and Icechunk Integration for Virtual Datacube

**Status:** Accepted  
**Date:** 2026-05-07  
**Authors:** @nlebovits

## Context

ADR-001 established that our pipeline produces COGs with STAC metadata for Source Cooperative. This ADR addresses the additional requirement: providing a **virtual Zarr datacube** that allows xarray/Zarr access across all tiles without duplicating data.

The goal is to enable analysis workflows like:

```python
import xarray as xr
ds = xr.open_zarr("icechunk://source.coop/radiant-earth/landsat-lst")
ds.lst_p50.sel(time="2023", latitude=slice(45, 40), longitude=slice(-75, -70)).mean()
```

This requires storing **virtual references** (byte-range pointers) to the COG data, not copying the pixels.

---

## Decisions

### 1. Virtual Reference Technology

**Decision:** VirtualZarr + Icechunk

| Component | Role |
|-----------|------|
| **VirtualZarr** | Creates byte-range references from COG internal tiles |
| **Icechunk** | Stores references with transactional commits, provides Zarr-compatible store |

**Rationale:**
- VirtualZarr is the successor to kerchunk, maintained by Zarr developers
- Icechunk provides ACID transactions for distributed writes
- Both are designed to work together (same maintainer: Earthmover)
- Native Zarr v3 support

**Alternatives considered:**
- Kerchunk JSON/Parquet — no transactions, harder to update atomically
- Zarr with actual data — duplicates COG data, doubles storage
- GDAL VRT — not Zarr-compatible, limited to GDAL ecosystem

**References:**
- VirtualZarr docs: https://github.com/zarr-developers/virtualizarr
- Icechunk docs: https://icechunk.io/
- Earthmover blog on architecture: https://www.earthmover.io/blog/serverless-datacube-pipeline

---

### 2. COG to Virtual Reference Mapping

**Decision:** VirtualZarr's `open_virtual_dataset` with `filetype='tiff'`

```python
from virtualizarr import open_virtual_dataset

vds = open_virtual_dataset(
    "s3://source-coop/landsat-lst/2023/N40W075.tif",
    filetype="tiff"
)
```

**Key behavior:** VirtualZarr does **not** rechunk data. It creates byte-range references to the COG's **existing internal tiles**. The COG's internal tile size (512×512 per ADR-001) becomes the effective Zarr chunk size.

**Chunk manifest structure:**
```python
{
    "0.0": {"path": "s3://bucket/tile.tif", "offset": 1000, "length": 52428},
    "0.1": {"path": "s3://bucket/tile.tif", "offset": 53428, "length": 52428},
    # ... one entry per COG internal tile
}
```

**Rationale:**
- No data transformation or copying
- Byte-range requests are efficient for cloud storage
- COG internal tiling is already optimized for spatial access

**References:**
- VirtualZarr TIFF support: https://github.com/zarr-developers/virtualizarr/blob/develop/docs/faq.md
- Kerchunk tiff_to_zarr (underlying implementation): https://fsspec.github.io/kerchunk/

---

### 3. Writing Virtual References to Icechunk

**Decision:** Use `vds.virtualize.to_icechunk(session.store)`

```python
import icechunk

storage = icechunk.s3_storage(
    bucket="source-coop-radiant-earth",
    prefix="landsat-lst/icechunk",
    region="us-west-2",
)
repo = icechunk.Repository.open_or_create(storage)
session = repo.writable_session("main")

vds.virtualize.to_icechunk(session.store)
session.commit("Added tile N40W075 for 2023")
```

**Rationale:**
- Direct integration between VirtualZarr and Icechunk
- Transactional commits ensure readers never see partial state
- Branch-based workflow enables atomic updates

**References:**
- VirtualZarr to_icechunk: https://github.com/zarr-developers/virtualizarr/blob/develop/docs/usage.md
- Icechunk virtual references: https://github.com/earth-mover/icechunk/blob/main/docs/docs/icechunk-python/virtual.md

---

### 4. Distributed Writes with Session Merging

**Decision:** Pickle sessions to workers, merge on completion

```python
from concurrent.futures import ProcessPoolExecutor
from icechunk.distributed import merge_sessions

session = repo.writable_session("main")

with ProcessPoolExecutor() as executor:
    with session.allow_pickling():
        futures = [
            executor.submit(process_tile, tile_id=t, session=session)
            for t in tiles
        ]
        worker_sessions = [f.result() for f in futures]

session = merge_sessions(session, *worker_sessions)
session.commit("Processed all tiles for 2023")
```

**Worker function returns session:**
```python
def process_tile(tile_id: str, session: Session) -> Session:
    # ... generate COG ...
    vds = open_virtual_dataset(cog_path, filetype="tiff")
    vds.virtualize.to_icechunk(session.store)
    return session
```

**Rationale:**
- Enables parallel processing across Coiled workers
- Each worker writes to its own session copy
- `merge_sessions` consolidates all changes atomically
- Pattern proven in production at Earthmover (serverless-datacube-demo)

**References:**
- Icechunk distributed writes: https://github.com/earth-mover/icechunk/blob/main/docs/docs/icechunk-python/parallel.md
- Serverless datacube demo: https://github.com/earth-mover/serverless-datacube-demo/blob/main/src/storage.py

---

### 5. Icechunk Store Location

**Decision:** Co-located with COGs on Source Cooperative

```
s3://source-coop-radiant-earth/
├── landsat-lst/
│   ├── 2023/
│   │   ├── N40W075.tif          # COG
│   │   ├── N40W070.tif          # COG
│   │   └── ...
│   ├── icechunk/                # Icechunk store (virtual refs)
│   │   └── ... (internal structure)
│   └── catalog/                 # STAC catalog
│       └── ...
```

**Rationale:**
- Single bucket simplifies access control
- Icechunk references use relative paths where possible
- Source Cooperative as single distribution point

---

### 6. Combining Multiple Tiles into Single Virtual Dataset

**Decision:** Use xarray's `combine_nested` before writing to Icechunk

```python
virtual_datasets = [
    open_virtual_dataset(f"s3://bucket/{tile}.tif", filetype="tiff")
    for tile in tile_list
]

combined = xr.combine_nested(
    virtual_datasets,
    concat_dim=["latitude", "longitude"],
    combine_attrs="override"
)

combined.virtualize.to_icechunk(session.store)
```

**Rationale:**
- Creates single logical datacube from all tiles
- Spatial dimensions concatenated appropriately
- Time dimension added via coordinate assignment

**References:**
- VirtualZarr combining datasets: https://github.com/zarr-developers/virtualizarr/blob/develop/docs/index.md

---

### 7. Handling Annual Composites (Time Dimension)

**Decision:** Each year processed separately, combined with `concat_dim=['time']`

```python
# Process each year
for year in [2021, 2022, 2023, 2024]:
    year_vds = process_year(year)  # Returns combined virtual dataset for year
    year_vds = year_vds.expand_dims(time=[pd.Timestamp(f"{year}-01-01")])
    year_vds.virtualize.to_icechunk(session.store, append_dim="time")

session.commit(f"Added years 2021-2024")
```

**Rationale:**
- Years can be processed incrementally
- `append_dim` allows extending existing store
- Time coordinate enables temporal slicing

---

## Consequences

### Positive
- Zero data duplication (COGs are the source of truth)
- Zarr/xarray access to entire datacube
- Atomic updates via Icechunk transactions
- Scales to global coverage with parallel processing
- COGs remain independently accessible via STAC

### Negative
- Additional complexity vs COG-only distribution
- Icechunk is newer technology (v2.0 as of 2026)
- Readers need Icechunk-aware Zarr (zarr-python 3.x)

### Risks
- VirtualZarr TIFF reader edge cases with complex COG structures
- Icechunk storage costs for reference metadata
- Version compatibility between VirtualZarr/Icechunk/Zarr

---

## Validation

Before full implementation, validate with single-tile proof of concept:

1. Generate one COG for Pergamino test region
2. Create VirtualZarr reference
3. Write to local Icechunk store
4. Open with `xr.open_zarr()` and verify data access
5. Test distributed write with 2 workers

---

## References

### Primary Sources
- [VirtualZarr Documentation](https://github.com/zarr-developers/virtualizarr) — virtual dataset creation
- [Icechunk Documentation](https://icechunk.io/) — transactional Zarr store
- [Earthmover Serverless Datacube Blog](https://www.earthmover.io/blog/serverless-datacube-pipeline) — architecture patterns
- [Serverless Datacube Demo](https://github.com/earth-mover/serverless-datacube-demo) — reference implementation

### Supporting Sources
- [Kerchunk TIFF Support](https://fsspec.github.io/kerchunk/) — underlying TIFF→reference implementation
- [Zarr v3 Specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) — storage format
- [Dynamical.org Reformatters](https://github.com/dynamical-org/reformatters) — production Icechunk patterns

### Landsat Data Sources
- [Earth Search Landsat C2 L2 Collection](https://earth-search.aws.element84.com/v1/collections/landsat-c2-l2) — STAC collection
- [USGS Landsat C2 L2 Products](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) — band specifications
