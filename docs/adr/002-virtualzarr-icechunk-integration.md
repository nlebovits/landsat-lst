# ADR-002: VirtualZarr and Icechunk Integration for Virtual Datacube

**Status:** ⚠️ **SUPERSEDED** by [ADR-003](003-direct-zarr-architecture.md)
**Date:** 2026-05-07
**Updated:** 2026-05-08 (API corrections from spike testing)
**Superseded:** 2026-05-08
**Authors:** @nlebovits

> ## ⚠️ This ADR is Superseded
>
> **This architecture was not implemented.** During implementation (PR #14), we discovered that GDAL's COG blocksize constraint (multiples of 16) conflicts with VirtualZarr's concatenation requirement (dimensions divisible by chunk size). No chunk size satisfies both constraints for our ~18,500 pixel tiles.
>
> **See [ADR-003](003-direct-zarr-architecture.md)** for the replacement architecture: direct Zarr writes without COG intermediate.
>
> The content below is preserved for historical reference.

---

## Context (Historical)

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

**Decision:** VirtualiZarr with `VirtualTIFF(ifd=0)` parser from virtual-tiff package

```python
from virtualizarr import open_virtual_dataset
from virtual_tiff import VirtualTIFF
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import from_url

# Create registry for S3 access
registry = ObjectStoreRegistry({
    "s3://source-coop-radiant-earth/": from_url(
        "s3://source-coop-radiant-earth/",
        region="us-west-2",
    )
})

# Open COG as virtual dataset
# IMPORTANT: ifd=0 is required for COGs with overviews to select full resolution
vds = open_virtual_dataset(
    "s3://source-coop-radiant-earth/landsat-lst/2023/N40W075.tif",
    parser=VirtualTIFF(ifd=0),  # IFD 0 = full resolution
    registry=registry,
)
```

**Key behavior:** VirtualZarr does **not** rechunk data. It creates byte-range references to the COG's **existing internal tiles**. The COG's internal tile size (512×512 per ADR-001) becomes the effective Zarr chunk size.

**IFD Selection:** COGs with overviews have multiple IFDs (Image File Directories). `ifd=0` selects the full-resolution image. Without this parameter, VirtualZarr attempts to read all IFDs, causing dimension conflicts.

**Chunk manifest structure:**
```python
{
    "0.0.0": {"path": "s3://bucket/tile.tif", "offset": 1000, "length": 52428},
    "0.0.1": {"path": "s3://bucket/tile.tif", "offset": 53428, "length": 52428},
    # ... one entry per COG internal tile
}
```

**Rationale:**
- No data transformation or copying
- Byte-range requests are efficient for cloud storage
- COG internal tiling is already optimized for spatial access

**References:**
- virtual-tiff package: https://github.com/virtual-zarr/virtual-tiff
- VirtualZarr usage: https://virtualizarr.readthedocs.io/en/latest/usage.html

---

### 3. Writing Virtual References to Icechunk

**Decision:** Use `vds.vz.to_icechunk(session.store)` with VirtualChunkContainer configuration

```python
import icechunk as ic

# Configure storage for Icechunk metadata
storage = ic.s3_storage(
    bucket="source-coop-radiant-earth",
    prefix="landsat-lst/icechunk",
    region="us-west-2",
)

# Configure VirtualChunkContainer for COG byte-range access
config = ic.config.RepositoryConfig.default()
config.set_virtual_chunk_container(
    ic.virtual.VirtualChunkContainer(
        "s3://source-coop-radiant-earth/",
        ic.storage.s3_store(region="us-west-2"),
    )
)

# Credentials for virtual chunk container access
credentials = ic.credentials.containers_credentials({
    "s3://source-coop-radiant-earth/": ic.credentials.s3_credentials(
        anonymous=True  # Source Cooperative is public
    )
})

# Create repository with virtual chunk authorization
repo = ic.Repository.create(
    storage,
    config=config,
    authorize_virtual_chunk_access=credentials,
)

session = repo.writable_session("main")

# Write virtual refs using .vz accessor (current API)
vds.vz.to_icechunk(session.store)
session.commit("Added tile N40W075 for 2023")
```

**Rationale:**
- Direct integration between VirtualZarr and Icechunk
- Transactional commits ensure readers never see partial state
- Branch-based workflow enables atomic updates
- VirtualChunkContainer enables byte-range fetches from COGs during reads

**References:**
- VirtualZarr to_icechunk: https://virtualizarr.readthedocs.io/en/latest/usage.html
- Icechunk virtual references: https://icechunk.io/en/stable/virtual/

---

### 4. Spatial Tile Concatenation Strategy

**Decision:** Concatenate all spatial tiles BEFORE writing to Icechunk

**Important:** VirtualZarr/Icechunk require spatial tiles to be combined into a single dataset before committing. Per-tile spatial commits are **not supported** because each `to_icechunk()` call writes a complete dataset to the root group.

```python
import xarray as xr
from virtualizarr import open_virtual_dataset
from virtual_tiff import VirtualTIFF

# Step 1: Open all tiles as virtual datasets
vds_list = []
for cog_path in tile_paths:
    vds = open_virtual_dataset(
        f"s3://bucket/{cog_path}",
        parser=VirtualTIFF(ifd=0),
        registry=registry,
    )
    vds_list.append(vds)

# Step 2: Concatenate along spatial dimension
# Note: combine_by_coords may not work if dimension coords aren't preserved
# Use xr.concat with explicit dimension instead
combined = xr.concat(vds_list, dim="y", combine_attrs="override")

# Step 3: Write combined dataset to Icechunk
session = repo.writable_session("main")
combined.vz.to_icechunk(session.store)
session.commit("Added all tiles for 2023")
```

**Constraints:**
- Array size along concat axis must be evenly divisible by chunk size (VirtualZarr limitation)
- VirtualZarr cannot `expand_dims` on arrays with transpose codecs (affects adding time dimension after parsing)

**Rationale:**
- Icechunk stores a single dataset at the root group
- Spatial tiles are different coordinates of the same datacube
- Concatenation preserves the chunk manifest (no data copied)

**Note on ADR-001 §16 per-tile commits:** Those are for retry/resume within a year's batch processing (idempotent COG checks + conflict retry), not for incremental spatial additions to Icechunk.

**References:**
- Icechunk virtual references: https://icechunk.io/en/stable/virtual/
- VirtualZarr concatenation: https://virtualizarr.readthedocs.io/en/latest/usage.html

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

**Decision:** Use `xr.concat` with explicit dimension (not `combine_by_coords`)

```python
from virtualizarr import open_virtual_dataset
from virtual_tiff import VirtualTIFF

virtual_datasets = [
    open_virtual_dataset(
        f"s3://bucket/{tile}.tif",
        parser=VirtualTIFF(ifd=0),
        registry=registry,
    )
    for tile in tile_list
]

# Use xr.concat with explicit dimension
# combine_by_coords doesn't work because VirtualTIFF doesn't preserve dimension coordinates
combined = xr.concat(
    virtual_datasets,
    dim="y",  # or appropriate spatial dimension
    combine_attrs="override",
)

combined.vz.to_icechunk(session.store)
```

**Important constraint:** Array sizes along the concatenation axis must be evenly divisible by the chunk size. This is a VirtualZarr limitation for maintaining regular chunk grids.

**Rationale:**
- Creates single logical datacube from all tiles
- Spatial dimensions concatenated appropriately
- Chunk manifests are combined without copying data

**References:**
- VirtualZarr combining datasets: https://virtualizarr.readthedocs.io/en/latest/usage.html

---

### 7. Handling Annual Composites (Time Dimension)

**Decision:** Add time coordinate BEFORE VirtualZarr parsing, then use `append_dim`

**Important:** VirtualZarr cannot use `expand_dims` on arrays with transpose codecs (which COGs have). The time dimension must be added during concatenation, not after.

```python
import pandas as pd

# Option A: Concatenate all years at once (simpler)
year_datasets = []
for year in [2021, 2022, 2023, 2024]:
    year_vds = process_year_tiles(year)  # Spatially concatenated tiles
    year_vds = year_vds.assign_coords(time=pd.Timestamp(f"{year}-01-01"))
    year_datasets.append(year_vds)

# Concatenate along time dimension
combined = xr.concat(year_datasets, dim="time", combine_attrs="override")
combined.vz.to_icechunk(session.store)
session.commit("Added years 2021-2024")

# Option B: Incremental year-by-year (for append_dim to work)
# First year: create the dataset
year_2021 = process_year_tiles(2021)
year_2021 = year_2021.assign_coords(time=pd.Timestamp("2021-01-01"))
# Add time dimension during concatenation with itself
year_2021 = xr.concat([year_2021], dim="time")

session = repo.writable_session("main")
year_2021.vz.to_icechunk(session.store)
session.commit("Added 2021")

# Subsequent years: append
for year in [2022, 2023, 2024]:
    year_vds = process_year_tiles(year)
    year_vds = year_vds.assign_coords(time=pd.Timestamp(f"{year}-01-01"))
    year_vds = xr.concat([year_vds], dim="time")

    session = repo.writable_session("main")
    year_vds.vz.to_icechunk(session.store, append_dim="time")
    session.commit(f"Added {year}")
```

**Rationale:**
- Years can be processed incrementally using `append_dim`
- Time coordinate enables temporal slicing
- Workaround avoids VirtualZarr's expand_dims limitation

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
- **VirtualTIFF requires ifd=0** for COGs with overviews — without it, dimension conflicts occur
- **expand_dims fails** on arrays with transpose codecs — time dimension must be added during concatenation
- **Array sizes must be divisible by chunk size** for concatenation to work
- Icechunk storage costs for reference metadata
- Version compatibility between VirtualZarr/Icechunk/Zarr/virtual-tiff

---

## Validation

**Status:** Completed 2026-05-08

Spike test (`scripts/spike_virtualzarr_icechunk.py`) validated:

1. ✅ VirtualTIFF(ifd=0) parser for COGs with overviews
2. ✅ Spatial tile concatenation with `xr.concat(dim="y")`
3. ✅ Write to local Icechunk store with VirtualChunkContainer
4. ✅ Read via `xr.open_zarr()` with `authorize_virtual_chunk_access`
5. ⚠️ Temporal append works, but `expand_dims` fails on transpose-coded arrays

Key findings:
- `combine_by_coords` doesn't work (VirtualTIFF doesn't preserve dimension coordinates)
- Array size must be divisible by chunk size for concatenation
- Per-tile spatial commits not supported — must concatenate first

---

## References

### Primary Sources
- [VirtualiZarr Documentation](https://virtualizarr.readthedocs.io/) — virtual dataset creation
- [virtual-tiff Package](https://github.com/virtual-zarr/virtual-tiff) — VirtualTIFF parser for COGs
- [Icechunk Documentation](https://icechunk.io/) — transactional Zarr store
- [Earthmover Serverless Datacube Blog](https://www.earthmover.io/blog/serverless-datacube-pipeline) — architecture patterns

### Supporting Sources
- [obspec-utils Registry](https://github.com/zarr-developers/obspec-utils) — ObjectStoreRegistry for cloud access
- [Zarr v3 Specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html) — storage format
- [Dynamical.org Reformatters](https://github.com/dynamical-org/reformatters) — production Icechunk patterns

### Landsat Data Sources
- [Earth Search Landsat C2 L2 Collection](https://earth-search.aws.element84.com/v1/collections/landsat-c2-l2) — STAC collection
- [USGS Landsat C2 L2 Products](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) — band specifications
