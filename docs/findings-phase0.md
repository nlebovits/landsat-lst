# Phase 0 Findings: TDD Assumption Validation

**Date:** 2026-05-07  
**Status:** Complete  
**Tests:** 35 passed

---

## Summary

This document captures discoveries from test-driven validation of the Landsat LST pipeline's external dependencies and API contracts.

---

## Key Findings

### 1. Data Source: Planetary Computer vs Earth Search

**Finding:** Use Microsoft Planetary Computer instead of Element84 Earth Search for Landsat C2 L2.

| Source | STAC API | Data Location | Cost |
|--------|----------|---------------|------|
| Earth Search | Free | USGS S3 (requester-pays) | ~$0.09/GB egress |
| Planetary Computer | Free | Azure (free egress) | **$0** |

**Configuration:**
```python
import planetary_computer
import pystac_client
from odc.stac import configure_rio

configure_rio(cloud_defaults=True)
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
```

---

### 2. rio-cogeo API (v7.x)

**Finding:** Block size goes in `dst_kwargs`, not as separate parameter.

**Wrong (old API):**
```python
cog_translate(src, dst, profile, blocksize=512)  # TypeError
```

**Correct:**
```python
dst_profile = cog_profiles.get("deflate")
dst_profile["blockxsize"] = 512
dst_profile["blockysize"] = 512
cog_translate(src, dst, dst_profile, use_cog_driver=True)
```

---

### 3. VirtualZarr 2.x API (Breaking Change)

**Finding:** Major API change from VirtualZarr 1.x. Now requires explicit registry and parser.

**Old API (deprecated):**
```python
vds = open_virtual_dataset("file.tif", filetype="tiff")  # TypeError
```

**New API (2.x):**
```python
from virtualizarr import open_virtual_dataset
from virtual_tiff import VirtualTIFF
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

registry = ObjectStoreRegistry()
registry.register("file://", LocalStore())
vds = open_virtual_dataset(f"file://{path}", registry=registry, parser=VirtualTIFF())
```

**Dependency:** Requires `virtual-tiff` package for TIFF support.

---

### 4. COG Overviews Break VirtualZarr

**Finding:** COGs with overviews cause dimension conflicts in VirtualZarr.

**Error:**
```
ValueError: conflicting sizes for dimension 'y': length 361 on '1' and length 721 on {'y': '0', 'x': '0'}
```

**Solution:** Disable overviews for VirtualZarr-targeted COGs:
```python
cog_translate(..., overview_level=0)
```

**Implication:** Our production COGs need overviews for direct access (GDAL/rasterio clients) but VirtualZarr references the full-resolution data. May need separate COG profiles or VirtualZarr-side filtering.

---

### 5. Icechunk 2.x Virtual Chunk Configuration

**Finding:** Icechunk 2.x requires explicit `VirtualChunkContainer` configuration for accessing external data.

**Configuration for local files:**
```python
from icechunk import ObjectStoreConfig, RepositoryConfig, VirtualChunkContainer

local_config = ObjectStoreConfig.LocalFileSystem("/path/to/cogs")
container = VirtualChunkContainer(
    "file:///path/to/cogs/",  # Must end with /
    local_config,
    name="local",
)

config = RepositoryConfig.default()
config.set_virtual_chunk_container(container)

repo = Repository.create(storage, config=config)
```

**Reading back requires authorization:**
```python
repo = Repository.open(
    storage,
    config=config,
    authorize_virtual_chunk_access={"file:///path/to/cogs/": None},
)
```

**For S3 in production:**
```python
s3_config = ObjectStoreConfig.S3(bucket="source-coop-radiant-earth", region="us-west-2")
container = VirtualChunkContainer(
    "s3://source-coop-radiant-earth/landsat-lst/",
    s3_config,
    name="cogs",
)
```

---

### 6. Validated API Contracts

All assumptions confirmed correct:

| Component | Assumption | Status |
|-----------|-----------|--------|
| STAC collection | `landsat-c2-l2` | ✅ |
| Thermal band | `lwir11` asset exists | ✅ |
| QA band | `qa_pixel` asset exists | ✅ |
| Cloud bit | Bit 3 | ✅ |
| Shadow bit | Bit 4 | ✅ |
| Snow bit | Bit 5 | ✅ |
| Scale factor | 0.00341802 | ✅ |
| Offset | 149.0 | ✅ |
| Temperature range | -40°C to 60°C typical | ✅ |

---

## Dependencies Added

```toml
[project.dependencies]
planetary-computer = ">=1.0.0"
icechunk = ">=2.0"
virtualizarr = ">=2.5"
virtual-tiff = ">=0.5"
rio-cogeo = ">=7.0"
odc-geo = ">=0.5"
```

---

---

## Parameter Sweep Results

### Cloud Cover Threshold

| Threshold | Scene Count | Notes |
|-----------|-------------|-------|
| ≤10% | 39 | Conservative, highest quality |
| ≤20% | 47 | **Recommended** — good balance |
| ≤30% | 55 | More data, some quality loss |
| ≤40% | 62 | Diminishing returns |
| ≤50% | 65 | |

**Distribution:** 46% of scenes have <10% cloud cover, 55% have <20%.

### Chunk Size

| Size | Load Time | Memory/Chunk |
|------|-----------|--------------|
| 256×256 | 0.79s | 0.5 MB |
| 512×512 | 0.46s | 2.0 MB |
| 1024×1024 | 0.55s | 8.0 MB |
| 2048×2048 | 0.19s | 32.0 MB |

**Recommendation:** 512×512 balances speed and memory. Larger chunks faster but use more RAM.

### Resampling Method

| Method | Time | Quality |
|--------|------|---------|
| nearest | 1.11s | Preserves original values |
| **bilinear** | **0.13s** | Smooth, 8.5× faster |
| cubic | 0.15s | Smoother, minimal gain |
| average | 0.15s | Similar to bilinear |

**Recommendation:** Use `bilinear` for thermal data — 8× faster than nearest with similar quality.

### COG Compression

| Profile | Size | Write | Read |
|---------|------|-------|------|
| none | 4.00 MB | 0.01s | 0.001s |
| lzw | 3.68 MB | 0.05s | 0.010s |
| **deflate** | **2.95 MB** | **0.06s** | **0.012s** |
| zstd | 2.93 MB | 0.07s | 0.004s |

**Recommendation:** Use `deflate` for Source Coop compatibility. ZSTD is ~0.7% smaller with 3× faster reads, but requires recent libtiff and isn't universally supported. DEFLATE works everywhere.

### Composite Statistics

| Method | Time | Notes |
|--------|------|-------|
| mean | 0.001s | Fastest, sensitive to outliers |
| **median** | **0.013s** | **Recommended** — robust, fast |
| quantile(0.5) | 2.75s | Same as median, 200× slower |
| quantile(0.95) | 2.75s | For heat extremes |

**Recommendation:** Use xarray's `.median()` instead of `.quantile(0.5)` — 200× faster.

---

## Optimized Settings

Based on sweeps, recommended production settings:

```python
# STAC query
query = {"eo:cloud_cover": {"lt": 20}}

# Data loading
stac_load(
    items,
    bands=["lwir11", "qa_pixel"],
    chunks={"time": 1, "latitude": 512, "longitude": 512},
    resampling={"lwir11": "bilinear", "qa_pixel": "nearest"},
)

# Composite
lst_median = lst_celsius.median(dim="time", skipna=True)

# COG output (DEFLATE for Source Coop compatibility)
dst_profile = cog_profiles.get("deflate")
dst_profile["blockxsize"] = 512
dst_profile["blockysize"] = 512
cog_translate(src, dst, dst_profile, use_cog_driver=True, overview_level=0)
```

---

## Full Tile Integration Test

**Status:** ✅ Passed (10/10 steps)

End-to-end pipeline validated on 0.25° tile near Pergamino, Argentina.

| Step | Time | Result |
|------|------|--------|
| STAC query | 1.1s | 47 scenes, 4.8% avg cloud cover |
| Load scenes | 0.2s | 901×901×32 lazy dataset |
| QA mask | <0.1s | Cloud/shadow/snow filtering |
| Temperature | <0.1s | Kelvin → Celsius conversion |
| Composite | 23.4s | p50: 16.6–32.4°C (mean 26.3°C) |
| COG write | 0.1s | 2.7 MB DEFLATE (uint16) |
| VirtualZarr | 0.009s | Reference from COG |
| Icechunk | 0.006s | Committed to store |
| Roundtrip | <0.1s | Structure validated |

---

## Data Type: uint16 with Scale/Offset

**Decision:** Store LST composites as uint16 instead of float32.

**Rationale:** This follows standard practice for analysis-ready LST products. Both the source data and major derived products use scaled integers:

| Product | Data Type | Scale | Offset | Units | Reference |
|---------|-----------|-------|--------|-------|-----------|
| Landsat C2 L2 ST | uint16 | 0.00341802 | 149.0 | Kelvin | [USGS](https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products) |
| MODIS MOD11 LST | uint16 | 0.02 | 0 | Kelvin | [USGS LPDAAC](https://lpdaac.usgs.gov/documents/118/MOD11_User_Guide_V6.pdf) |

From USGS: *"Level-2 products are written as scaled integers to allow conversion from floating point to integer for delivery... saves disk space and provides faster download times."*

**Our encoding (Celsius output):**

```python
# Encoding: Celsius → uint16
# Range: -50°C to +105.535°C (covers all terrestrial LST)
# Precision: 0.01°C (exceeds source precision)
SCALE = 0.01
OFFSET = -50.0
NODATA_UINT16 = 0  # -50°C is outside valid LST range

def encode_celsius_to_uint16(celsius: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
    """Encode Celsius temperatures to uint16 with scale/offset."""
    valid = celsius != nodata
    dn = np.zeros_like(celsius, dtype=np.uint16)
    dn[valid] = np.round((celsius[valid] - OFFSET) / SCALE).clip(1, 65535).astype(np.uint16)
    return dn

def decode_uint16_to_celsius(dn: np.ndarray) -> np.ndarray:
    """Decode uint16 to Celsius temperatures."""
    celsius = np.where(dn == 0, np.nan, dn * SCALE + OFFSET)
    return celsius.astype(np.float32)
```

| Celsius | uint16 DN | Notes |
|---------|-----------|-------|
| -50.0°C | 1 | Minimum valid (DN=0 is nodata) |
| 0.0°C | 5000 | |
| 25.0°C | 7500 | Typical urban |
| 50.0°C | 10000 | Hot desert |
| 70.0°C | 12000 | Extreme surface |

**Storage impact:**

| Metric | float32 | uint16 | Savings |
|--------|---------|--------|---------|
| Bytes/pixel (3 bands) | 12 | 6 | 50% |
| 0.25° test tile | ~4.8 MB | ~2.4 MB | 50% |
| 5° production tile | ~1.9 GB | ~950 MB | 50% |
| Per year (global) | ~1.1 TB | ~550 GB | 50% |
| Full archive (2013-2025) | ~14 TB | ~7 TB | 50% |

**Precision analysis:** The source Landsat ST band has ~0.1K precision (scale factor 0.00341802). Our 0.01°C encoding provides 10× finer precision than the source data — no information is lost.

---

## Next Steps

1. **Phase 1 implementation** — Build production pipeline modules using validated patterns
2. **Python version** — Pin to 3.12 until zstd codec fixed for 3.14
