# Implementation Plan: Issue #10 — VirtualZarr + Icechunk Integration

**Issue:** https://github.com/nlebovits/landsat-lst/issues/10
**Status:** ⚠️ **SUPERSEDED** (2026-05-08)
**Date:** 2026-05-08
**Prerequisites:** Issues #5 (uint16 encoding) and #6 (retry/resume) merged

> ## ⚠️ This Plan is Superseded
>
> **This implementation approach was abandoned.** During implementation (PR #14), we discovered that GDAL's COG blocksize constraint conflicts with VirtualZarr's concatenation requirements.
>
> **See [ADR-003](../adr/003-direct-zarr-architecture.md)** for the replacement architecture: direct Zarr writes.
>
> The content below is preserved for historical reference.

---

## Overview (Historical)

Implement virtual Zarr datacube access over COGs using VirtualZarr and Icechunk, enabling:

```python
import xarray as xr

ds = xr.open_zarr("icechunk://source.coop/radiant-earth/landsat-lst")
ds.lst_p50.sel(latitude=slice(45, 40), longitude=slice(-75, -70)).mean()
```

---

## Pre-Implementation Research Summary

### Spike Validation (2026-05-08)

Spike script (`scripts/spike_virtualzarr_icechunk.py`) validated:

| Component | Status | Finding |
|-----------|--------|---------|
| VirtualTIFF parser | ✅ | `VirtualTIFF(ifd=0)` required for COGs with overviews |
| Spatial concatenation | ✅ | `xr.concat(dim="y")` — tiles must be combined BEFORE Icechunk commit |
| Icechunk virtual refs | ✅ | `authorize_virtual_chunk_access` required; `.vz.to_icechunk()` is current API |
| xarray access | ✅ | Byte-range reads from COGs work correctly |
| Temporal append | ⚠️ | `expand_dims` fails on transpose-coded arrays — add time coord during concatenation |

### Industry Research

Chunk size practices from comparable projects:

| Project | Chunk Size | Power of 2? |
|---------|------------|-------------|
| Earthmover serverless-datacube | 1200×1200 | No |
| Dynamical.org reformatters | 50×50 to 121×121 | No |
| USGS Landsat COGs | 256×256 | Yes |
| MS Planetary Computer ERA5 | 150×150 | No |

**Decision:** Use 500×500 chunks (18,500 ÷ 500 = 37 exact). Updated in ADR-001 §11.

**References:**
- https://github.com/earth-mover/serverless-datacube-demo
- https://github.com/dynamical-org/reformatters
- USGS LSDS-1388 Landsat COG Data Format Control Book

---

## Architecture

### Two-Phase Approach

```
Phase 1: landsat-lst process --year 2023
         └── Generates COGs with idempotency checks (existing job.py)

Phase 2: landsat-lst virtualize --year 2023
         └── Creates virtual datacube from COGs (new virtual.py)
```

**Rationale:** Separation of concerns. COG generation is compute-intensive; virtualization is metadata-only (~seconds). Allows rebuilding virtual layer without reprocessing COGs.

### Data Flow

```
COGs on S3                    VirtualZarr                 Icechunk
─────────────────────────────────────────────────────────────────────
N40W075.tif  ─┐
N41W075.tif  ─┼─► VirtualTIFF(ifd=0) ─► xr.concat() ─► .vz.to_icechunk()
N42W075.tif  ─┘
     │                              │                         │
     │                              │                         ▼
     │                              │              icechunk/ (refs only)
     │                              │                         │
     └──────────────────────────────┴─────────────────────────┘
                     Byte-range reads during xr.open_zarr()
```

---

## Implementation Tasks

### Task 1: Update COG chunk size (cog.py)

Change default blocksize from 512 to 500:

```python
# cog.py
def write_cog(
    composite: xr.Dataset,
    output_path: Path | str,
    *,
    blocksize: int = 500,  # Changed from 512 for VirtualZarr alignment
    ...
)
```

**Files:** `src/landsat_lst/cog.py`
**Tests:** Update `tests/test_cog_write.py` if blocksize is tested

---

### Task 2: Create virtual.py module

New module with core virtualization functions:

```python
# src/landsat_lst/virtual.py

from virtualizarr import open_virtual_dataset
from virtual_tiff import VirtualTIFF
from obspec_utils.registry import ObjectStoreRegistry
import icechunk as ic
import xarray as xr


def create_registry(storage: StorageBackend) -> ObjectStoreRegistry:
    """Create ObjectStoreRegistry for COG access."""
    ...


def open_tile_virtual(
    cog_url: str,
    tile_name: str,
    registry: ObjectStoreRegistry,
) -> xr.Dataset:
    """Open a single COG as virtual dataset with proper coords and names."""
    vds = open_virtual_dataset(
        cog_url,
        parser=VirtualTIFF(ifd=0),  # Required for COGs with overviews
        registry=registry,
    )

    # Rename bands from numeric to meaningful names
    vds = vds.rename({"0": "lst_p50", "1": "lst_p95", "2": "qa_count"})

    # Assign coordinates based on tile name
    vds = assign_tile_coords(vds, tile_name)

    return vds


def assign_tile_coords(vds: xr.Dataset, tile_name: str) -> xr.Dataset:
    """Add lat/lon coordinates based on tile name (e.g., N40W075)."""
    from landsat_lst.tiling import parse_tile_name

    lat_start, lon_start = parse_tile_name(tile_name)
    n_lat, n_lon = vds.sizes["y"], vds.sizes["x"]

    # 5° tiles, north-to-south latitude, west-to-east longitude
    lat = np.linspace(lat_start + 5, lat_start, n_lat)
    lon = np.linspace(lon_start, lon_start + 5, n_lon)

    return vds.assign_coords(latitude=("y", lat), longitude=("x", lon))


def create_icechunk_repo(
    storage: StorageBackend,
    container_prefix: str,
) -> ic.Repository:
    """Create Icechunk repository with VirtualChunkContainer."""
    icechunk_storage = storage.icechunk_storage()

    config = ic.config.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        ic.virtual.VirtualChunkContainer(
            container_prefix,
            storage.virtual_chunk_store(),  # New method needed on StorageBackend
        )
    )

    credentials = ic.credentials.containers_credentials(
        {
            container_prefix: storage.virtual_chunk_credentials(),  # New method
        }
    )

    return ic.Repository.create(
        icechunk_storage,
        config=config,
        authorize_virtual_chunk_access=credentials,
    )


def create_virtual_datacube(
    tile_paths: list[str],
    tile_names: list[str],
    year: int,
    storage: StorageBackend,
) -> str:
    """Create virtual datacube from COGs for a single year."""
    registry = create_registry(storage)

    # Open all tiles as virtual datasets
    vds_list = [
        open_tile_virtual(path, name, registry)
        for path, name in zip(tile_paths, tile_names, strict=True)
    ]

    # Concatenate spatially
    combined = xr.concat(vds_list, dim="y", combine_attrs="override")

    # Add time coordinate
    combined = combined.assign_coords(time=pd.Timestamp(f"{year}-01-01"))
    combined = xr.concat([combined], dim="time")  # Wrap in time dim

    # Write to Icechunk
    container_prefix = storage.cog_container_prefix()
    repo = create_icechunk_repo(storage, container_prefix)

    session = repo.writable_session("main")
    combined.vz.to_icechunk(session.store)
    snapshot_id = session.commit(f"Add {year} virtual datacube ({len(tile_paths)} tiles)")

    return snapshot_id
```

**Files:** `src/landsat_lst/virtual.py` (new)

---

### Task 3: Extend StorageBackend for virtual chunks

Add methods to `StorageBackend` ABC and implementations:

```python
# storage.py additions


class StorageBackend(ABC):
    @abstractmethod
    def virtual_chunk_store(self) -> ic.storage.Store:
        """Return Icechunk store for virtual chunk access."""

    @abstractmethod
    def virtual_chunk_credentials(self) -> ic.credentials.AnyCredential | None:
        """Return credentials for virtual chunk container."""

    @abstractmethod
    def cog_container_prefix(self) -> str:
        """Return URL prefix for COG container (e.g., s3://bucket/)."""


class LocalStorage(StorageBackend):
    def virtual_chunk_store(self) -> ic.storage.Store:
        return ic.storage.local_filesystem_store(str(self.output_dir))

    def virtual_chunk_credentials(self) -> None:
        return None  # Local files don't need credentials

    def cog_container_prefix(self) -> str:
        return f"file://{self.output_dir}/"


class S3Storage(StorageBackend):
    def virtual_chunk_store(self) -> ic.storage.Store:
        return ic.storage.s3_store(region=self.region)

    def virtual_chunk_credentials(self) -> ic.credentials.AnyCredential:
        return ic.credentials.s3_credentials(anonymous=True)  # Source Coop is public

    def cog_container_prefix(self) -> str:
        return f"s3://{self.bucket}/"
```

**Files:** `src/landsat_lst/storage.py`

---

### Task 4: Add CLI command

```python
# cli.py addition


@app.command()
def virtualize(
    year: int = typer.Argument(..., help="Year to virtualize"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing"),
) -> None:
    """Create virtual datacube from existing COGs for a year."""
    from landsat_lst.virtual import create_virtual_datacube
    from landsat_lst.storage import get_storage
    from landsat_lst.tiling import get_land_tiles

    storage = get_storage()
    tiles = get_land_tiles()

    # Filter to tiles that have COGs
    tile_paths = []
    tile_names = []
    for tile in tiles:
        if storage.cog_exists(year, tile):
            tile_paths.append(storage.cog_path(year, tile))
            tile_names.append(tile)

    if not tile_paths:
        console.print(f"[red]No COGs found for {year}[/red]")
        raise typer.Exit(1)

    console.print(f"Virtualizing {len(tile_paths)} tiles for {year}...")

    snapshot_id = create_virtual_datacube(tile_paths, tile_names, year, storage)

    console.print(f"[green]✓ Created virtual datacube: {snapshot_id[:12]}...[/green]")
```

**Files:** `src/landsat_lst/cli.py`

---

### Task 5: Add integration test

```python
# tests/integration/test_virtual_datacube.py


def test_virtual_datacube_creation(tmp_path):
    """Test end-to-end virtual datacube creation and access."""
    # Create test COGs
    storage = LocalStorage(tmp_path)
    # ... create 2 test COGs ...

    # Create virtual datacube
    snapshot_id = create_virtual_datacube(
        tile_paths=[...],
        tile_names=["N40W075", "N41W075"],
        year=2023,
        storage=storage,
    )

    # Verify access
    credentials = ic.credentials.containers_credentials({f"file://{tmp_path}/": None})
    repo = ic.Repository.open(
        storage.icechunk_storage(),
        authorize_virtual_chunk_access=credentials,
    )

    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)

    assert "lst_p50" in ds.data_vars
    assert "latitude" in ds.coords
    assert ds.sizes["time"] == 1
```

**Files:** `tests/integration/test_virtual_datacube.py` (new)

---

### Task 6: Add LST metadata attributes

Per issue #10 requirements:

```python
# In virtual.py create_virtual_datacube()

# Add dataset-level attributes
combined.attrs["lst_scale"] = 0.01
combined.attrs["lst_offset"] = -50.0

# Add variable-level attributes
combined["lst_p50"].attrs["scale_factor"] = 0.01
combined["lst_p50"].attrs["add_offset"] = -50.0
combined["lst_p50"].attrs["units"] = "celsius"
combined["lst_p50"].attrs["long_name"] = "Land Surface Temperature (median)"

combined["lst_p95"].attrs["scale_factor"] = 0.01
combined["lst_p95"].attrs["add_offset"] = -50.0
combined["lst_p95"].attrs["units"] = "celsius"
combined["lst_p95"].attrs["long_name"] = "Land Surface Temperature (95th percentile)"

combined["qa_count"].attrs["units"] = "count"
combined["qa_count"].attrs["long_name"] = "Valid observation count"
```

---

### Task 7: Update README

Add usage documentation:

```markdown
## Virtual Datacube Access

After processing, create the virtual datacube layer:

\`\`\`bash
landsat-lst virtualize 2023
\`\`\`

Access via xarray:

\`\`\`python
import xarray as xr

ds = xr.open_zarr("icechunk://source.coop/radiant-earth/landsat-lst")

# Spatial query
subset = ds.lst_p50.sel(
    latitude=slice(45, 40),
    longitude=slice(-75, -70)
)

# Decode uint16 to Celsius
lst_celsius = subset * 0.01 + (-50.0)
\`\`\`
```

**Files:** `README.md`

---

## Task Checklist

| # | Task | Files | Est. |
|---|------|-------|------|
| 1 | Update COG chunk size to 500 | `cog.py` | 5 min |
| 2 | Create `virtual.py` module | `virtual.py` (new) | 1 hr |
| 3 | Extend `StorageBackend` | `storage.py` | 30 min |
| 4 | Add `virtualize` CLI command | `cli.py` | 30 min |
| 5 | Add integration test | `test_virtual_datacube.py` (new) | 45 min |
| 6 | Add LST metadata attributes | `virtual.py` | 15 min |
| 7 | Update README | `README.md` | 15 min |

**Total estimated time:** ~3.5 hours

---

## Dependencies

```toml
# Already in pyproject.toml
"icechunk>=1.1.21",
"virtualizarr>=2.5.1",

# Verify these are present (should be via virtualizarr)
# virtual-tiff
# obspec-utils
# obstore
```

---

## Testing Strategy

1. **Unit tests:** Mock VirtualZarr/Icechunk for `virtual.py` functions
2. **Integration test:** End-to-end with local storage (in `tests/integration/`)
3. **Manual validation:** Run spike script to verify before/after

---

## Rollout

1. Create feature branch `feature/virtualzarr-icechunk`
2. Implement tasks 1-6
3. Run integration tests locally
4. Update README (task 7)
5. Create PR with spike script output as evidence
6. Merge after review

---

## References

- **ADR-001 §11:** Chunk size decision (500×500)
- **ADR-002:** VirtualZarr + Icechunk architecture (updated 2026-05-08)
- **Spike script:** `scripts/spike_virtualzarr_icechunk.py`
- **Issue #10:** https://github.com/nlebovits/landsat-lst/issues/10
