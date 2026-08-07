# Landsat LST

Global Land Surface Temperature composites — annual or multi-year — from Landsat Collection 2 Level-2 data.

## Overview

This pipeline produces annual or multi-year LST composites for municipal decision-makers analyzing urban heat. A composite pools *every* scene in its window (one year, e.g. `2024`, or a range, e.g. `2020-2024`) into a single percentile — multi-year windows fill cloud/orbit gaps and suppress scene-footprint striping. Output includes:

- **lst_p95**: 95th percentile LST (°C), pooled across all scenes in the window
- **qa_count**: 12-month climatology of valid-observation counts (`(month, latitude, longitude)`, one band per calendar month; month M = valid observations in calendar month M pooled across the window)

Data is tiled on a 5° global grid and stored as **GeoZarr multiscale pyramids**
(versioned with Icechunk), published with a STAC catalog. Each tile is a pyramid:
native resolution in level group `0` and coarsened overviews in `1`/`2`/`3` — see
[Data Access](#data-access) and [ADR-004](docs/adr/004-geozarr-multiscale-overviews.md).

## Architecture

Design decisions are recorded as ADRs in [`docs/adr/`](docs/adr/README.md).

## Data Encoding

LST bands are stored as **uint16** to reduce file size by 50%. To convert back to Celsius:

```python
import xarray as xr
import fsspec

# Open native resolution (multiscale level "0") from Source Coop
mapper = fsspec.get_mapper(
    "s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst/2023/N40W075.zarr",
    anon=True
)
ds = xr.open_zarr(mapper, group="0")

# Read scale/offset from Zarr attributes
scale = ds["lst_p95"].attrs["lst_scale_factor"]    # 0.01
offset = ds["lst_p95"].attrs["lst_add_offset"]     # -50.0

# Decode to Celsius
celsius = ds["lst_p95"] * scale + offset
```

**Quick decode (if you know the constants):**
```python
celsius = ds["lst_p95"] * 0.01 + (-50.0)  # fill_value=0 is nodata
```

| Variable | Name | Scale | Offset | Units |
|----------|------|-------|--------|-------|
| lst_p95 | 95th percentile LST | 0.01 | -50.0 | celsius |
| qa_count | Valid observations per calendar month (uint8, 12 bands: `month` × lat × lon) | — | — | count |

## Installation

```bash
uv sync
```

## Configuration

Settings live in [`landsat_lst.config`](src/landsat_lst/config.py) and can be
overridden via environment variables (prefix `LST_`) or a `.env` file. Notable
data-quality and performance settings:

| Setting | Env var | Default | Purpose |
|---------|---------|---------|---------|
| `lst_valid_min` | `LST_LST_VALID_MIN` | `-50.0` | Drop physically implausible cold LST (e.g. ~-124 °C DN=0 / resampling artifacts) |
| `lst_valid_max` | `LST_LST_VALID_MAX` | `80.0` | Drop high-DN saturation/fill artifacts without clipping real extreme heat |
| `load_chunk_size` | `LST_LOAD_CHUNK_SIZE` | `512` | odc-stac spatial (lat/lon) chunk; smaller (e.g. 256) cuts peak memory for the P95 quantile on multi-year / large-tile runs |
| `max_cloud_cover` | `LST_MAX_CLOUD_COVER` | `100` | Scene-level cloud filter; 100 disables it and relies on pixel-level QA |

## Usage

```bash
# Process a single tile for one year
landsat-lst process --year 2023 --tile N40W075

# Process all tiles for a year
landsat-lst process --year 2023

# List available tiles
landsat-lst list-tiles
```

#### Multi-year composites

A `ProcessingJob` accepts an optional `end_year` to pool every scene across a
multi-year window into a single P95 (percentiles are computed on the pooled
scenes, never averaged across per-year P95s). The window label
(`2024` or `2020-2024`) keys the output storage group and COG filenames:

```python
from landsat_lst.models import ProcessingJob
from landsat_lst.pipeline import process_tile
from landsat_lst.tiling import parse_tile_name

# Single year (backward compatible)
job = ProcessingJob(tile=parse_tile_name("N40W075"), year=2024)

# Five-year window 2020-2024 -> group/COGs keyed "2020-2024"
job = ProcessingJob(tile=parse_tile_name("N40W075"), year=2020, end_year=2024)
composite = process_tile(job)
```

### Distributed Processing (Coiled)

For production-scale processing, the pipeline runs on [Coiled](https://coiled.io) with AWS:

```bash
# Ensure AWS SSO session is active
aws sso login --profile radiant-earth

# Run E2E test for a single tile
uv run python scripts/e2e_coiled_s3.py --tile N40W075 --year 2024

# Dry run (show config without processing)
uv run python scripts/e2e_coiled_s3.py --dry-run
```

Output is written to Source Cooperative S3.

## Data Access

Each tile is stored as an independent GeoZarr multiscale pyramid on Source Cooperative.
The tile store is a group of resolution levels — open level `0` for native resolution:

```python
import xarray as xr
import fsspec

# Access a specific tile at native resolution (level "0")
url = "s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst/2023/N40W075.zarr"
mapper = fsspec.get_mapper(url, anon=True)
ds = xr.open_zarr(mapper, group="0")

# Spatial subset (only fetches required chunks)
subset = ds.lst_p95.sel(
    latitude=slice(42, 40),
    longitude=slice(-74, -75)
)

# Decode uint16 to Celsius
lst_celsius = subset * 0.01 + (-50.0)
```

#### Multiscale overviews (GeoZarr)

Each tile group follows the GeoZarr `multiscales` convention: native resolution is
level `0`, with coarsened overviews in `1` (4x), `2` (16x), `3` (64x) for fast
zoomed-out / web rendering. The tile group's attributes describe the pyramid:

```python
import zarr

root = zarr.open_group(mapper, mode="r")          # the tile group
print(root.attrs["multiscales"]["layout"])         # level layout + scale factors
print(root.attrs["proj:code"], root.attrs["spatial:transform"])  # GeoZarr proj/spatial

# Open a coarse overview (16x) instead of full resolution
overview = xr.open_zarr(mapper, group="2")
```

### QGIS Access

For QGIS users without native Zarr support, use rioxarray to convert to GeoTIFF:

```python
import rioxarray
import fsspec

url = "s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst/2023/N40W075.zarr"
ds = xr.open_zarr(fsspec.get_mapper(url, anon=True), group="0")
ds["lst_p95"].rio.to_raster("lst_subset.tif")
# Open lst_subset.tif in QGIS
```

See [docs/adr/003-direct-zarr-architecture.md](docs/adr/003-direct-zarr-architecture.md) for architecture details.

### COG export

QGIS-ready Cloud-Optimized GeoTIFFs are produced by the reusable
[`landsat_lst.cog`](src/landsat_lst/cog.py) module (`cog_export`), which derives
two COGs from a native-resolution composite level:

- a **single-band LST COG** with GDAL scale/offset embedded, so viewers auto-decode
  DN to Celsius (`degC = DN * 0.01 - 50`); and
- a **12-band monthly QA COG** (one `qa_count` band per calendar month, no nodata so
  a value of 0 = "no valid observations that month" stays visible for gap diagnosis).

```python
import xarray as xr
from landsat_lst.cog import cog_export

native = xr.open_zarr(store, group="2020-2024/0")  # decoded native level
cog_export(native, "lst.tif", "qa_monthly.tif")
```

## Architecture

See [docs/adr/001-architecture-decisions.md](docs/adr/001-architecture-decisions.md) for detailed design decisions.

Key choices:
- **Data source**: Earth Search (Landsat C2 L2)
- **Output format**: Zarr v3 with 500×500 chunks
- **CRS**: EPSG:4326
- **Tiling**: 5° × 5° grid
- **Temporal**: Calendar-year or multi-year window composites (pooled P95)
- **Spatial**: Land only, ±60° latitude

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run linting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run ty check src/

# Run tests
uv run pytest

# Install pre-commit hooks
uv run prek install
```

### Optional extras

`uv sync --all-extras` installs everything; individual extras can be installed on
their own:

- `analysis` — `matplotlib` for figure generation in analysis/findings writeups, plus
  `h5py` and `earthaccess` for the ASTER GED gap analysis.
- `frisky` — experimental Rust reimplementation of the Dask scheduler
  ([getfrisky.dev](https://getfrisky.dev)). Kept behind a fallback: the multi-year
  decision driver uses it when installed but reverts to plain Dask otherwise
  (it crashed gathering large results, so **production uses Dask**).

### Scripts

Notable scripts in [`scripts/`](scripts/):

- `pergamino_multiyear_decision.py` — multi-window (1/3/5-year) decision driver that
  writes GeoZarr pyramids + COGs and emits a `report.md` of gap/striping and measured
  compression metrics. Runs locally against Planetary Computer; `--no-frisky` forces
  plain Dask.
- `season_aware_p95_test.py` — prototype of season-aware de-striping (per-scene bias
  removal relative to a per-pixel monthly climatology). Not part of the production
  pipeline — see [docs/findings-destriping-and-multiyear.md](docs/findings-destriping-and-multiyear.md)
  and [docs/adr/005-multiyear-monthly-qa-and-destriping.md](docs/adr/005-multiyear-monthly-qa-and-destriping.md).
- `aster_gap_urban_analysis.py` — measures ASTER GED coverage gaps against GHS-SMOD to
  quantify how much urban land has no Surface Temperature. Needs the `analysis` extra
  and a NASA Earthdata login; see [Known Limitations](#known-limitations).

## License

MIT
