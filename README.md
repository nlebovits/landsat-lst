# Landsat LST

Global annual Land Surface Temperature composites from Landsat Collection 2 Level-2 data.

## Overview

This pipeline produces annual LST composites for municipal decision-makers analyzing urban heat. Output includes:

- **lst_p95**: 95th percentile LST (°C)
- **qa_count**: Number of valid observations

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
| qa_count | Observation count | — | — | count |

## Installation

```bash
uv sync
```

## Usage

```bash
# Process a single tile for one year
landsat-lst process --year 2023 --tile N40W075

# Process all tiles for a year
landsat-lst process --year 2023

# List available tiles
landsat-lst list-tiles
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

## Architecture

See [docs/adr/001-architecture-decisions.md](docs/adr/001-architecture-decisions.md) for detailed design decisions.

Key choices:
- **Data source**: Earth Search (Landsat C2 L2)
- **Output format**: Zarr v3 with 500×500 chunks
- **CRS**: EPSG:4326
- **Tiling**: 5° × 5° grid
- **Temporal**: Calendar year composites
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

## License

MIT
