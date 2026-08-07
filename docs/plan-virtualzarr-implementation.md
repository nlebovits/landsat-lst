# Implementation Plan: VirtualZarr + Icechunk Integration

**Date:** 2026-05-07
**Status:** ⚠️ **SUPERSEDED** (2026-05-08)
**Related ADRs:** [ADR-001](adr/001-architecture-decisions.md), [ADR-002](adr/002-virtualzarr-icechunk-integration.md)
**Findings:** [Phase 0 Findings](findings-phase0.md)

> ## ⚠️ This Plan is Superseded
>
> **This implementation approach was abandoned** after discovering that GDAL's COG blocksize constraint (multiples of 16) conflicts with VirtualZarr's concatenation requirement. The architecture has pivoted to direct Zarr writes.
>
> **See:**
> - [ADR-003: Direct Zarr Architecture](adr/003-direct-zarr-architecture.md) — replacement architecture
> - [findings-direct-zarr-spike.md](findings-direct-zarr-spike.md) — validation findings
>
> The content below is preserved for historical reference only.

---

## Overview (Historical)

This plan describes the implementation of the VirtualZarr + Icechunk virtual datacube layer on top of the COG + STAC output defined in ADR-001.

**End state:** Users can access the full Landsat LST datacube via:
```python
import xarray as xr

ds = xr.open_zarr("icechunk://source.coop/radiant-earth/landsat-lst")
```

---

## Scaling Estimates

| Scope | Size (uint16) | Notes |
|-------|---------------|-------|
| 0.25° test tile | ~2.4 MB | 901×901 px, 3 bands uint16, DEFLATE |
| 5° production tile | ~950 MB | 18,000×18,000 px |
| Global land (1 year) | ~550 GB | ~560 land tiles × 950 MB |
| Full archive (2013-2025) | ~7 TB | 13 years × 550 GB |

**50% storage savings** from using uint16 instead of float32 (see [Data Type decision](findings-phase0.md#data-type-uint16-with-scaleoffset)).

COGs stored on Source Coop; Icechunk stores lightweight references (~MBs).

---

## Phase 0: Proof of Concept ✅ COMPLETE

**Goal:** Validate the VirtualZarr → Icechunk pipeline with a single tile before building full infrastructure.

**Status:** Completed via TDD. See [findings-phase0.md](findings-phase0.md) for details.

### Validated Components

| Component | Status | Key Finding |
|-----------|--------|-------------|
| STAC query | ✅ | Use Planetary Computer (free egress) |
| odc.stac loading | ✅ | Dimensions are `latitude`/`longitude`, not `x`/`y` |
| QA masking | ✅ | Bits 3/4/5 for cloud/shadow/snow |
| Temperature conversion | ✅ | `DN * 0.00341802 + 149.0 - 273.15` |
| COG creation | ✅ | DEFLATE, 512×512, predictor=2, `overview_level=0` |
| VirtualZarr 2.x | ✅ | Requires `ObjectStoreRegistry` + `VirtualTIFF` parser |
| Icechunk 2.x | ✅ | Requires `VirtualChunkContainer` for external data |
| Roundtrip read | ✅ | Structure validated; zstd codec issue on Python 3.14 |

### Optimal Settings (from parameter sweeps)

```python
# STAC query
query = {"eo:cloud_cover": {"lt": 20}}

# Data loading
stac_load(
    items,
    bands=["lwir11", "qa_pixel"],
    chunks={"time": 10, "latitude": 512, "longitude": 512},
    resampling={"lwir11": "bilinear", "qa_pixel": "nearest"},
)

# Composite statistics
lst_median = lst_celsius.median(dim="time", skipna=True)  # NOT .quantile(0.5)

# COG output - DEFLATE for Source Coop compatibility
dst_profile = cog_profiles.get("deflate")
dst_profile["blockxsize"] = 512
dst_profile["blockysize"] = 512
cog_translate(src, dst, dst_profile, use_cog_driver=True, overview_level=0)
```

### Tests Created

- `tests/integration/test_phase0_assumptions.py` — 35 API validation tests
- `tests/integration/test_parameter_sweeps.py` — 25 optimization tests
- `tests/integration/test_full_tile.py` — 10-step end-to-end pipeline test

---

## Phase 1: Core Infrastructure (3-5 days)

**Goal:** Build production-ready modules for storage, job processing, and virtual reference generation.

### 1.1 Storage Module: `src/landsat_lst/storage.py`

```python
"""Storage abstraction for COGs and Icechunk virtual store."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import icechunk
from icechunk import (
    ObjectStoreConfig,
    Repository,
    RepositoryConfig,
    Session,
    VirtualChunkContainer,
)
from icechunk.distributed import merge_sessions


@dataclass
class StorageConfig:
    """Configuration for storage backends."""

    cog_bucket: str
    cog_prefix: str
    icechunk_bucket: str
    icechunk_prefix: str
    region: str = "us-west-2"


class IcechunkStorage:
    """Manages Icechunk repository for virtual references."""

    def __init__(self, config: StorageConfig):
        self.config = config
        self._session: Session | None = None
        self._repo: Repository | None = None

    def _get_storage(self) -> icechunk.Storage:
        return icechunk.s3_storage(
            bucket=self.config.icechunk_bucket,
            prefix=self.config.icechunk_prefix,
            region=self.config.region,
        )

    def _get_virtual_chunk_container(self) -> VirtualChunkContainer:
        """Configure access to COGs on S3 for virtual chunk resolution."""
        s3_config = ObjectStoreConfig.S3(
            bucket=self.config.cog_bucket,
            region=self.config.region,
        )
        return VirtualChunkContainer(
            f"s3://{self.config.cog_bucket}/{self.config.cog_prefix}/",
            s3_config,
            name="cogs",
        )

    @contextmanager
    def writable_session(self, branch: str = "main") -> Iterator[Session]:
        """Get a writable session for the Icechunk store."""
        storage = self._get_storage()

        # Configure virtual chunk container for COG access
        config = RepositoryConfig.default()
        config.set_virtual_chunk_container(self._get_virtual_chunk_container())

        self._repo = Repository.open_or_create(storage, config=config)
        session = self._repo.writable_session(branch)
        self._session = session
        try:
            with session.allow_pickling():
                yield session
        finally:
            self._session = None

    def commit(self, message: str, worker_sessions: list[Session] | None = None) -> str:
        """Commit changes, merging worker sessions if provided."""
        if self._session is None:
            raise RuntimeError("No active session")

        if worker_sessions:
            valid_sessions = [s for s in worker_sessions if s is not None]
            self._session = merge_sessions(self._session, *valid_sessions)

        return self._session.commit(message)

    def cog_path(self, tile_id: str, year: int) -> str:
        """Generate S3 path for a COG."""
        return f"s3://{self.config.cog_bucket}/{self.config.cog_prefix}/{year}/{tile_id}.tif"


class LocalStorage:
    """Local filesystem storage for development."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.cog_dir = base_path / "cogs"
        self.icechunk_dir = base_path / "icechunk"
        self._session: Session | None = None
        self._repo: Repository | None = None

    def _get_virtual_chunk_container(self) -> VirtualChunkContainer:
        """Configure access to local COGs for virtual chunk resolution."""
        local_config = ObjectStoreConfig.LocalFileSystem(str(self.cog_dir))
        return VirtualChunkContainer(
            f"file://{self.cog_dir}/",
            local_config,
            name="local",
        )

    @contextmanager
    def writable_session(self, branch: str = "main") -> Iterator[Session]:
        self.cog_dir.mkdir(parents=True, exist_ok=True)
        self.icechunk_dir.mkdir(parents=True, exist_ok=True)

        storage = icechunk.local_filesystem_storage(str(self.icechunk_dir))

        # Configure virtual chunk container
        config = RepositoryConfig.default()
        config.set_virtual_chunk_container(self._get_virtual_chunk_container())

        self._repo = Repository.open_or_create(storage, config=config)
        session = self._repo.writable_session(branch)
        self._session = session
        try:
            with session.allow_pickling():
                yield session
        finally:
            self._session = None

    def open_readonly(self, branch: str = "main"):
        """Open repository for reading with virtual chunk authorization."""
        storage = icechunk.local_filesystem_storage(str(self.icechunk_dir))
        config = RepositoryConfig.default()
        config.set_virtual_chunk_container(self._get_virtual_chunk_container())

        return Repository.open(
            storage,
            config=config,
            authorize_virtual_chunk_access={f"file://{self.cog_dir}/": None},
        )
```

### 1.2 Job Module: `src/landsat_lst/job.py`

```python
"""Job definitions for tile processing."""

from dataclasses import dataclass

import pandas as pd
import planetary_computer
import pystac_client
import xarray as xr
from icechunk import Session
from odc.stac import configure_rio, stac_load

from landsat_lst.models import TileId
from landsat_lst.cog import write_cog
from landsat_lst.virtual import create_virtual_reference


@dataclass(frozen=True)
class TileYearJob:
    """Processing job for a single tile and year."""

    tile_id: TileId
    year: int

    def process(self, session: Session, cog_output_path: str) -> Session:
        """
        Process tile: STAC query → composite → COG → virtual reference.

        Returns the session with virtual references written.
        """
        # 1. Query STAC (Planetary Computer for free egress)
        configure_rio(cloud_defaults=True)
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=self.tile_id.bbox,
            datetime=f"{self.year}-01-01/{self.year}-12-31",
            query={
                "eo:cloud_cover": {"lt": 20},
                "platform": {"in": ["landsat-8", "landsat-9"]},
            },
        )
        items = list(search.items())

        if len(items) == 0:
            return session  # No data for this tile/year

        # 2. Load scenes with optimized settings
        ds = stac_load(
            items,
            bands=["lwir11", "qa_pixel"],
            crs="EPSG:4326",
            resolution=0.00027778,  # ~30m
            chunks={"time": 10, "latitude": 512, "longitude": 512},
            resampling={"lwir11": "bilinear", "qa_pixel": "nearest"},
            bbox=self.tile_id.bbox,
            groupby="solar_day",
        )

        # 3. Apply QA mask
        qa = ds["qa_pixel"]
        cloud = (qa >> 3) & 1
        shadow = (qa >> 4) & 1
        snow = (qa >> 5) & 1
        mask = (cloud == 0) & (shadow == 0) & (snow == 0)

        # 4. Convert to Celsius
        lst_masked = ds["lwir11"].where(mask)
        lst_kelvin = lst_masked * 0.00341802 + 149.0
        lst_celsius = lst_kelvin - 273.15

        # 5. Compute composite (median is 200x faster than quantile)
        lst_p50 = lst_celsius.median(dim="time", skipna=True)
        lst_p95 = lst_celsius.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")

        valid_count = (~lst_celsius.isnull()).sum(dim="time").astype("int16")

        nodata = -9999.0
        composite = xr.Dataset(
            {
                "lst_p50": lst_p50.where(valid_count > 0, nodata).astype("float32"),
                "lst_p95": lst_p95.where(valid_count > 0, nodata).astype("float32"),
                "qa_count": valid_count,
            }
        )
        composite = composite.compute()

        # 6. Write COG
        cog_path = write_cog(composite, cog_output_path, self.tile_id, self.year)

        # 7. Create virtual reference and write to Icechunk
        vds = create_virtual_reference(cog_path, str(self.tile_id), self.year)
        vds.virtualize.to_icechunk(session.store)

        return session


def generate_jobs(
    tile_ids: list[TileId],
    years: list[int],
) -> list[TileYearJob]:
    """Generate all jobs for given tiles and years."""
    return [TileYearJob(tile_id=t, year=y) for t in tile_ids for y in years]
```

### 1.3 COG Writer: `src/landsat_lst/cog.py`

```python
"""COG writing utilities with Source Coop-compatible settings.

Data type: uint16 with scale/offset encoding for 50% storage reduction.
This follows standard practice for analysis-ready LST products:
- Landsat C2 L2 ST: uint16, scale=0.00341802, offset=149.0 (Kelvin)
- MODIS MOD11: uint16, scale=0.02, offset=0 (Kelvin)

References:
- USGS: https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products
- MODIS: https://lpdaac.usgs.gov/documents/118/MOD11_User_Guide_V6.pdf
"""

from pathlib import Path
import tempfile

import numpy as np
import rasterio
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import xarray as xr

from landsat_lst.models import TileId

# Encoding constants for Celsius → uint16
# Range: -50°C to +105.535°C with 0.01°C precision
# This exceeds source precision (~0.1K) so no information is lost
LST_SCALE = 0.01
LST_OFFSET = -50.0
LST_NODATA_UINT16 = 0  # DN=0 reserved for nodata
LST_NODATA_CELSIUS = -9999.0


def encode_celsius_to_uint16(celsius: np.ndarray) -> np.ndarray:
    """Encode Celsius temperatures to uint16 with scale/offset.

    Formula: DN = (celsius - offset) / scale
    Decode:  celsius = DN * scale + offset

    DN=0 is reserved for nodata. Valid range is DN=1 to DN=65535,
    representing -49.99°C to +105.535°C.
    """
    valid = celsius != LST_NODATA_CELSIUS
    dn = np.zeros_like(celsius, dtype=np.uint16)
    dn[valid] = np.round((celsius[valid] - LST_OFFSET) / LST_SCALE).clip(1, 65535).astype(np.uint16)
    return dn


def write_cog(
    ds: xr.Dataset,
    output_path: str,
    tile_id: TileId,
    year: int,
) -> str:
    """
    Write xarray Dataset to Cloud-Optimized GeoTIFF.

    Data encoding:
    - lst_p50, lst_p95: uint16, scale=0.01, offset=-50.0 (Celsius)
    - qa_count: uint16 (raw count, no scaling)

    Uses DEFLATE compression for universal compatibility (per portolan ADR-0019).
    Disables overviews for VirtualZarr compatibility.

    Returns the path to the written COG.
    """
    # Encode temperature bands to uint16
    lst_p50_uint16 = encode_celsius_to_uint16(ds["lst_p50"].values)
    lst_p95_uint16 = encode_celsius_to_uint16(ds["lst_p95"].values)
    qa_count_uint16 = ds["qa_count"].values.astype(np.uint16)

    # Stack into multiband array
    data = np.stack([lst_p50_uint16, lst_p95_uint16, qa_count_uint16], axis=0)

    # Get spatial info
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    transform = rasterio.transform.from_bounds(
        lon.min(), lat.min(), lon.max(), lat.max(), data.shape[2], data.shape[1]
    )

    # Write temp GeoTIFF
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": data.shape[2],
        "height": data.shape[1],
        "count": 3,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": 0,  # DN=0 is nodata for all bands
    }

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = ("lst_p50", "lst_p95", "qa_count")

    # Note: Scale/offset stored in STAC metadata and Icechunk attrs,
    # not in COG (cog_translate doesn't preserve per-band tags)

    # Translate to COG with Source Coop-compatible settings
    # DEFLATE for universal compatibility (not zstd)
    # No overviews for VirtualZarr compatibility
    dst_profile = cog_profiles.get("deflate")
    dst_profile["blockxsize"] = 512
    dst_profile["blockysize"] = 512
    dst_profile["predictor"] = 2  # Horizontal differencing

    cog_translate(
        tmp_path,
        output_path,
        dst_profile,
        use_cog_driver=True,
        overview_level=0,  # No overviews - required for VirtualZarr
        quiet=True,
    )

    Path(tmp_path).unlink()  # Cleanup temp file

    return output_path
```

### 1.4 Virtual Reference Module: `src/landsat_lst/virtual.py`

```python
"""VirtualZarr 2.x integration for creating virtual datacube."""

import pandas as pd
import xarray as xr
from icechunk import Session
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore, S3Store
from virtual_tiff import VirtualTIFF
from virtualizarr import open_virtual_dataset


def create_virtual_reference(
    cog_path: str,
    tile_id: str,
    year: int,
) -> xr.Dataset:
    """
    Create virtual dataset from COG with proper coordinates.

    Uses VirtualZarr 2.x API with explicit registry and parser.
    """
    # Configure object store registry based on path scheme
    registry = ObjectStoreRegistry()

    if cog_path.startswith("s3://"):
        registry.register("s3://", S3Store())
        uri = cog_path
    else:
        registry.register("file://", LocalStore())
        uri = f"file://{cog_path}" if not cog_path.startswith("file://") else cog_path

    # Open virtual dataset with VirtualTIFF parser
    vds = open_virtual_dataset(uri, registry=registry, parser=VirtualTIFF())

    # Add time coordinate (mid-year for annual composite)
    vds = vds.expand_dims(time=[pd.Timestamp(f"{year}-07-01")])

    # Add tile metadata
    vds.attrs["tile_id"] = tile_id
    vds.attrs["year"] = year

    return vds


def combine_tile_references(
    virtual_datasets: list[xr.Dataset],
    concat_dim: str = "time",
) -> xr.Dataset:
    """
    Combine multiple virtual datasets into single datacube.
    """
    return xr.combine_nested(
        virtual_datasets,
        concat_dim=[concat_dim],
        combine_attrs="override",
    )


def write_to_icechunk(
    vds: xr.Dataset,
    session: Session,
    append_dim: str | None = None,
) -> None:
    """
    Write virtual dataset to Icechunk store.
    """
    if append_dim:
        vds.virtualize.to_icechunk(session.store, append_dim=append_dim)
    else:
        vds.virtualize.to_icechunk(session.store)
```

---

## Phase 2: Parallel Processing Integration (2-3 days)

**Goal:** Wire up Coiled for distributed tile processing.

### 2.1 Coiled Runner: `src/landsat_lst/runner.py`

```python
"""Coiled-based parallel execution."""

import coiled
from icechunk import Session
from tqdm import tqdm

from landsat_lst.job import TileYearJob
from landsat_lst.storage import IcechunkStorage, StorageConfig


@coiled.function(
    region="us-west-2",
    cpu=4,
    memory="16GB",
    environ={"ZARR_V3_EXPERIMENTAL_API": "1"},
)
def process_tile_job(
    job: TileYearJob,
    session: Session,
    cog_bucket: str,
) -> Session:
    """Process a single tile job on Coiled worker."""
    cog_path = f"s3://{cog_bucket}/landsat-lst/{job.year}/{job.tile_id}.tif"
    return job.process(session, cog_path)


def run_parallel(
    jobs: list[TileYearJob],
    storage: IcechunkStorage,
    max_workers: int = 100,
) -> None:
    """Run all jobs in parallel on Coiled."""

    with storage.writable_session() as session:
        # Map jobs to workers
        results = list(
            tqdm(
                process_tile_job.map(
                    jobs,
                    session=session,
                    cog_bucket=storage.config.cog_bucket,
                    retries=3,
                ),
                total=len(jobs),
                desc="Processing tiles",
            )
        )

        # Filter successful results
        worker_sessions = [r for r in results if r is not None]

        # Commit all changes
        storage.commit(
            f"Processed {len(worker_sessions)} tiles",
            worker_sessions=worker_sessions,
        )
```

### 2.2 CLI Integration: Update `src/landsat_lst/cli.py`

```python
@click.command()
@click.option("--year", type=int, required=True)
@click.option("--parallel/--no-parallel", default=True)
@click.option("--max-workers", type=int, default=100)
def process_year(year: int, parallel: bool, max_workers: int):
    """Process all tiles for a given year."""
    from landsat_lst.tiling import generate_land_tiles
    from landsat_lst.job import generate_jobs
    from landsat_lst.runner import run_parallel
    from landsat_lst.storage import IcechunkStorage, StorageConfig

    # Generate tile list
    tiles = generate_land_tiles()
    jobs = generate_jobs(tiles, [year])

    click.echo(f"Processing {len(jobs)} tiles for {year}")

    # Configure storage
    config = StorageConfig(
        cog_bucket="source-coop-radiant-earth",
        cog_prefix="landsat-lst",
        icechunk_bucket="source-coop-radiant-earth",
        icechunk_prefix="landsat-lst/icechunk",
    )
    storage = IcechunkStorage(config)

    if parallel:
        run_parallel(jobs, storage, max_workers=max_workers)
    else:
        # Sequential for debugging
        with storage.writable_session() as session:
            for job in tqdm(jobs):
                cog_path = storage.cog_path(str(job.tile_id), job.year)
                job.process(session, cog_path)
            storage.commit(f"Processed {len(jobs)} tiles for {year}")
```

---

## Phase 3: Testing & Validation (2 days)

### 3.1 Unit Tests

```python
# tests/unit/test_virtual.py
def test_create_virtual_reference(tmp_path):
    """Virtual reference creation from COG."""
    cog_path = create_test_cog(tmp_path / "test.tif")

    vds = create_virtual_reference(str(cog_path), "N40W075", 2023)

    assert "time" in vds.dims
    assert vds.attrs["tile_id"] == "N40W075"


# tests/unit/test_storage.py
def test_local_storage_roundtrip(tmp_path):
    """Write and read back from local Icechunk."""
    storage = LocalStorage(tmp_path)

    with storage.writable_session() as session:
        vds = create_test_virtual_dataset()
        vds.virtualize.to_icechunk(session.store)
        storage.commit("Test commit")

    # Read back with authorization
    repo = storage.open_readonly()
    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)

    assert len(ds.data_vars) > 0
```

### 3.2 Integration Tests

Already created in Phase 0:
- `tests/integration/test_phase0_assumptions.py` — 35 tests
- `tests/integration/test_parameter_sweeps.py` — 25 tests
- `tests/integration/test_full_tile.py` — 10 tests

Additional tests needed:
```python
# tests/integration/test_multi_tile.py
@pytest.mark.integration
def test_multi_tile_sequential(tmp_path):
    """Process multiple tiles sequentially."""
    tiles = [TileId(lat=-34, lon=-61), TileId(lat=-34, lon=-60)]
    jobs = generate_jobs(tiles, [2024])

    storage = LocalStorage(tmp_path)
    with storage.writable_session() as session:
        for job in jobs:
            cog_path = str(tmp_path / "cogs" / f"{job.tile_id}_{job.year}.tif")
            job.process(session, cog_path)
        storage.commit("Multi-tile test")

    repo = storage.open_readonly()
    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
    assert "time" in ds.dims
```

---

## Phase 4: Production Deployment (1-2 days)

### 4.1 Environment Configuration

```bash
# .env.production
ICECHUNK_BUCKET=source-coop-radiant-earth
ICECHUNK_PREFIX=landsat-lst/icechunk
COG_BUCKET=source-coop-radiant-earth
COG_PREFIX=landsat-lst
AWS_REGION=us-west-2
```

### 4.2 Initial Backfill

```bash
# Process years sequentially to manage costs
# Estimated: ~1.1 TB per year, ~560 tiles
for year in 2021 2022 2023 2024; do
    uv run landsat-lst process-year --year $year --max-workers 200
done
```

### 4.3 Monitoring

- Coiled dashboard for worker status
- CloudWatch for S3 access patterns
- Icechunk commit history for audit trail

---

## Timeline Summary

| Phase | Duration | Status | Deliverable |
|-------|----------|--------|-------------|
| 0: POC | 1-2 days | ✅ Complete | Validated APIs, optimal settings identified |
| 1: Core | 3-5 days | Pending | storage.py, job.py, cog.py, virtual.py |
| 2: Parallel | 2-3 days | Pending | Coiled integration, CLI commands |
| 3: Testing | 2 days | Partial | Unit + integration test suite |
| 4: Deploy | 1-2 days | Pending | Production config, initial backfill |

**Remaining: 8-12 days**

---

## Dependencies

### Packages (validated versions)
```toml
[project.dependencies]
icechunk = ">=1.1.21"
virtualizarr = ">=2.5.1"
virtual-tiff = ">=0.5.0"
odc-stac = ">=0.3.10"
odc-geo = ">=0.5.1"
rio-cogeo = ">=7.0.2"
planetary-computer = ">=1.0.0"
coiled = ">=1.0.0"
```

### External Services
- AWS S3 (Source Cooperative bucket)
- Coiled account with AWS credentials
- **Planetary Computer STAC API** (public, free egress)

### Python Version
Pin to **Python 3.12** until zstd codec compatibility with 3.14 is resolved.

---

## Open Items

1. **Source Cooperative bucket setup** — need bucket name and write credentials
2. **Coiled cluster sizing** — profile with subset before full run
3. **STAC catalog generation** — parallel to Icechunk or derive from it?
4. **Retry/resume strategy** — how to handle partial failures mid-run
5. **Distributed write validation** — test `merge_sessions` with Coiled workers

---

## Key Learnings from Phase 0

See [findings-phase0.md](findings-phase0.md) for full details.

1. **Use Planetary Computer** over Earth Search (free egress vs $0.09/GB)
2. **VirtualZarr 2.x API** requires explicit `ObjectStoreRegistry` + `VirtualTIFF` parser
3. **Icechunk 2.x** requires `VirtualChunkContainer` configuration for external data
4. **COG overviews break VirtualZarr** — use `overview_level=0`
5. **DEFLATE over ZSTD** for Source Coop compatibility (~same size, universal support)
6. **Use `.median()` not `.quantile(0.5)`** — 200× faster for same result
7. **Use `bilinear` resampling** for thermal data — 8× faster than nearest
8. **Use uint16 with scale/offset** — 50% storage reduction, standard practice for LST products
