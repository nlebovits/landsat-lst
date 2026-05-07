# Implementation Plan: VirtualZarr + Icechunk Integration

**Date:** 2026-05-07  
**Status:** Draft  
**Related ADRs:** [ADR-001](adr/001-architecture-decisions.md), [ADR-002](adr/002-virtualzarr-icechunk-integration.md)

---

## Overview

This plan describes the implementation of the VirtualZarr + Icechunk virtual datacube layer on top of the COG + STAC output defined in ADR-001.

**End state:** Users can access the full Landsat LST datacube via:
```python
import xarray as xr
ds = xr.open_zarr("icechunk://source.coop/radiant-earth/landsat-lst")
```

---

## Phase 0: Proof of Concept (1-2 days)

**Goal:** Validate the VirtualZarr → Icechunk pipeline with a single tile before building full infrastructure.

### Tasks

#### 0.1 Add Dependencies
```bash
uv add icechunk virtualizarr odc-stac odc-geo
```

Dependencies and their roles:
| Package | Version | Purpose |
|---------|---------|---------|
| `icechunk` | `>=2.0` | Transactional Zarr store for virtual references |
| `virtualizarr` | `>=1.0` | Create virtual datasets from COGs |
| `odc-stac` | `>=0.3` | Load Landsat from STAC into xarray |
| `odc-geo` | `>=0.4` | GeoBox creation for spatial alignment |

#### 0.2 Create POC Notebook: `notebooks/poc-virtualzarr.ipynb`

```python
# Cell 1: Query STAC for Pergamino test region
import pystac_client
import odc.stac

catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
items = list(catalog.search(
    collections=["landsat-c2-l2"],
    bbox=[-60.6, -34.0, -60.4, -33.8],  # Pergamino subset
    datetime="2023-01-01/2023-12-31",
    query={"eo:cloud_cover": {"lt": 20}},
).items())

print(f"Found {len(items)} scenes")

# Cell 2: Load and composite
from odc.geo.geobox import GeoBox

geobox = GeoBox.from_bbox([-60.6, -34.0, -60.4, -33.8], crs="EPSG:4326", resolution=0.00027778)

ds = odc.stac.load(
    items,
    bands=["lwir11", "qa_pixel"],
    geobox=geobox,
    chunks={"time": 1, "x": 512, "y": 512},
    resampling={"lwir11": "bilinear", "qa_pixel": "nearest"},
)

# Apply QA mask and compute median
qa = ds.qa_pixel
cloud = (qa >> 3) & 1
shadow = (qa >> 4) & 1
mask = (cloud == 0) & (shadow == 0)

lst_kelvin = ds.lwir11.where(mask) * 0.00341802 + 149.0
lst_celsius = lst_kelvin - 273.15
lst_median = lst_celsius.median(dim="time")

# Cell 3: Write COG
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

# ... (write to temp file, translate to COG)

# Cell 4: Create VirtualZarr reference
from virtualizarr import open_virtual_dataset

vds = open_virtual_dataset("pergamino_2023.tif", filetype="tiff")
print(vds)

# Cell 5: Write to local Icechunk
import icechunk

storage = icechunk.local_filesystem_storage("./local_icechunk")
repo = icechunk.Repository.create(storage)
session = repo.writable_session("main")

vds.virtualize.to_icechunk(session.store)
session.commit("Added Pergamino 2023 test tile")

# Cell 6: Verify read-back
import xarray as xr
ds_read = xr.open_zarr(repo.readonly_session("main").store)
print(ds_read)
ds_read.lst_median.plot()
```

#### 0.3 Validate Distributed Write Pattern

```python
# Test session pickling with 2 local workers
from concurrent.futures import ProcessPoolExecutor
from icechunk.distributed import merge_sessions

def worker_task(tile_path: str, session) -> Session:
    vds = open_virtual_dataset(tile_path, filetype="tiff")
    vds.virtualize.to_icechunk(session.store)
    return session

session = repo.writable_session("main")
with ProcessPoolExecutor(max_workers=2) as executor:
    with session.allow_pickling():
        futures = [
            executor.submit(worker_task, f"tile_{i}.tif", session)
            for i in range(2)
        ]
        worker_sessions = [f.result() for f in futures]

session = merge_sessions(session, *worker_sessions)
session.commit("Test distributed write")
```

### Success Criteria
- [ ] VirtualZarr opens Landsat COG without error
- [ ] References written to Icechunk successfully
- [ ] `xr.open_zarr()` reads data correctly
- [ ] Distributed write with session merge works

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
from icechunk import Session
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
    
    def _get_storage(self) -> icechunk.Storage:
        return icechunk.s3_storage(
            bucket=self.config.icechunk_bucket,
            prefix=self.config.icechunk_prefix,
            region=self.config.region,
        )
    
    @contextmanager
    def writable_session(self, branch: str = "main") -> Iterator[Session]:
        """Get a writable session for the Icechunk store."""
        storage = self._get_storage()
        repo = icechunk.Repository.open_or_create(storage)
        session = repo.writable_session(branch)
        self._session = session
        try:
            with session.allow_pickling():
                yield session
        finally:
            self._session = None
    
    def commit(
        self, 
        message: str, 
        worker_sessions: list[Session] | None = None
    ) -> str:
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
    
    @contextmanager
    def writable_session(self, branch: str = "main") -> Iterator[Session]:
        storage = icechunk.local_filesystem_storage(str(self.icechunk_dir))
        repo = icechunk.Repository.open_or_create(storage)
        session = repo.writable_session(branch)
        self._session = session
        try:
            with session.allow_pickling():
                yield session
        finally:
            self._session = None
    
    # ... similar methods
```

### 1.2 Job Module: `src/landsat_lst/job.py`

```python
"""Job definitions for tile processing."""

from dataclasses import dataclass
from pathlib import Path

import xarray as xr
from icechunk import Session
from virtualizarr import open_virtual_dataset

from landsat_lst.models import TileId
from landsat_lst.pipeline import query_stac, load_scenes, compute_annual_composite
from landsat_lst.cog import write_cog


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
        # 1. Query STAC for scenes
        scenes = query_stac(
            bbox=self.tile_id.bbox,
            year=self.year,
            max_cloud_cover=20,
        )
        
        if len(scenes) == 0:
            return session  # No data for this tile/year
        
        # 2. Load and composite
        ds = load_scenes(scenes, geobox=self.tile_id.geobox)
        composite = compute_annual_composite(ds)
        
        # 3. Write COG
        cog_path = write_cog(
            composite,
            output_path=cog_output_path,
            tile_id=self.tile_id,
            year=self.year,
        )
        
        # 4. Create virtual reference and write to Icechunk
        vds = open_virtual_dataset(cog_path, filetype="tiff")
        vds = vds.expand_dims(time=[pd.Timestamp(f"{self.year}-07-01")])
        vds.virtualize.to_icechunk(session.store)
        
        return session


def generate_jobs(
    tile_ids: list[TileId],
    years: list[int],
) -> list[TileYearJob]:
    """Generate all jobs for given tiles and years."""
    return [
        TileYearJob(tile_id=t, year=y)
        for t in tile_ids
        for y in years
    ]
```

### 1.3 COG Writer: `src/landsat_lst/cog.py`

```python
"""COG writing utilities."""

from pathlib import Path
import tempfile

import numpy as np
import rasterio
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import xarray as xr

from landsat_lst.models import TileId


def write_cog(
    ds: xr.Dataset,
    output_path: str,
    tile_id: TileId,
    year: int,
    blocksize: int = 512,
) -> str:
    """
    Write xarray Dataset to Cloud-Optimized GeoTIFF.
    
    Returns the path to the written COG.
    """
    # Prepare data arrays
    lst_p50 = ds["lst_p50"].values
    lst_p95 = ds["lst_p95"].values  
    qa_count = ds["qa_count"].values
    
    # Stack into multiband array
    data = np.stack([lst_p50, lst_p95, qa_count], axis=0)
    
    # Get transform from dataset
    transform = ds.rio.transform()
    crs = ds.rio.crs
    
    # Write temp GeoTIFF
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": data.shape[2],
        "height": data.shape[1],
        "count": 3,
        "crs": crs,
        "transform": transform,
        "nodata": -9999,
    }
    
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = ("lst_p50", "lst_p95", "qa_count")
    
    # Translate to COG
    cog_profile = cog_profiles.get("deflate")
    cog_translate(
        tmp_path,
        output_path,
        cog_profile,
        blocksize=blocksize,
        overview_resampling="average",
        use_cog_driver=True,
    )
    
    Path(tmp_path).unlink()  # Cleanup temp file
    
    return output_path
```

### 1.4 Virtual Reference Module: `src/landsat_lst/virtual.py`

```python
"""VirtualZarr integration for creating virtual datacube."""

import pandas as pd
import xarray as xr
from virtualizarr import open_virtual_dataset
from icechunk import Session


def create_virtual_reference(
    cog_path: str,
    tile_id: str,
    year: int,
) -> xr.Dataset:
    """
    Create virtual dataset from COG with proper coordinates.
    """
    vds = open_virtual_dataset(cog_path, filetype="tiff")
    
    # Add time coordinate
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
        results = list(tqdm(
            process_tile_job.map(
                jobs,
                session=session,
                cog_bucket=storage.config.cog_bucket,
                retries=3,
            ),
            total=len(jobs),
            desc="Processing tiles",
        ))
        
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
    # Create test COG
    cog_path = create_test_cog(tmp_path / "test.tif")
    
    vds = create_virtual_reference(cog_path, "N40W075", 2023)
    
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
    
    # Read back
    repo = icechunk.Repository.open(
        icechunk.local_filesystem_storage(str(tmp_path / "icechunk"))
    )
    ds = xr.open_zarr(repo.readonly_session("main").store)
    
    assert "lst_p50" in ds.data_vars
```

### 3.2 Integration Tests

```python
# tests/integration/test_full_pipeline.py
@pytest.mark.integration
def test_single_tile_pipeline(pergamino_bbox):
    """Full pipeline test with single Pergamino tile."""
    job = TileYearJob(
        tile_id=TileId.from_bbox(pergamino_bbox),
        year=2023,
    )
    
    with LocalStorage(tmp_path).writable_session() as session:
        result_session = job.process(session, tmp_path / "test.tif")
        assert result_session is not None
    
    # Verify COG exists and is valid
    assert (tmp_path / "test.tif").exists()
    
    # Verify virtual reference works
    ds = xr.open_zarr(session.store)
    assert ds.lst_p50.shape[0] > 0
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

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 0: POC | 1-2 days | Validated VirtualZarr→Icechunk flow |
| 1: Core | 3-5 days | storage.py, job.py, cog.py, virtual.py |
| 2: Parallel | 2-3 days | Coiled integration, CLI commands |
| 3: Testing | 2 days | Unit + integration test suite |
| 4: Deploy | 1-2 days | Production config, initial backfill |

**Total: 9-14 days**

---

## Dependencies

### New Packages
```toml
[project.dependencies]
icechunk = ">=2.0"
virtualizarr = ">=1.0"
odc-stac = ">=0.3"
odc-geo = ">=0.4"
rio-cogeo = ">=5.0"
coiled = ">=1.0"
```

### External Services
- AWS S3 (Source Cooperative bucket)
- Coiled account with AWS credentials
- Earth Search STAC API (public, no auth)

---

## Open Items

1. **Source Cooperative bucket setup** — need bucket name and write credentials
2. **Coiled cluster sizing** — profile with subset before full run
3. **STAC catalog generation** — parallel to Icechunk or derive from it?
4. **Retry/resume strategy** — how to handle partial failures mid-run
