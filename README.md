# Landsat LST

Global annual Land Surface Temperature composites from Landsat Collection 2 Level-2 data.

## Overview

This pipeline produces annual LST composites for municipal decision-makers analyzing urban heat. Output includes:

- **lst_p50**: Median LST (°C)
- **lst_p95**: 95th percentile LST (°C)
- **qa_count**: Number of valid observations

Data is tiled on a 5° global grid, stored as Zarr v3 stores, and published with a STAC catalog.

## Data Encoding

LST bands are stored as **uint16** to reduce file size by 50%. To convert back to Celsius:

```python
import xarray as xr
import fsspec

# Open a tile from Source Coop
mapper = fsspec.get_mapper(
    "s3://us-west-2.opendata.source.coop/radiant-earth/landsat-lst/2023/N40W075.zarr",
    anon=True
)
ds = xr.open_zarr(mapper)

# Read scale/offset from Zarr attributes
scale = ds["lst_p50"].attrs["lst_scale_factor"]    # 0.01
offset = ds["lst_p50"].attrs["lst_add_offset"]     # -50.0

# Decode to Celsius
celsius = ds["lst_p50"] * scale + offset
```

**Quick decode (if you know the constants):**
```python
celsius = ds["lst_p50"] * 0.01 + (-50.0)  # fill_value=0 is nodata
```

| Variable | Name | Scale | Offset | Units |
|----------|------|-------|--------|-------|
| lst_p50 | Median LST | 0.01 | -50.0 | celsius |
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

## Data Access

Each tile is stored as an independent Zarr store on Source Cooperative:

```python
import xarray as xr
import fsspec

# Access a specific tile
url = "s3://us-west-2.opendata.source.coop/radiant-earth/landsat-lst/2023/N40W075.zarr"
mapper = fsspec.get_mapper(url, anon=True)
ds = xr.open_zarr(mapper)

# Spatial subset (only fetches required chunks)
subset = ds.lst_p50.sel(
    latitude=slice(42, 40),
    longitude=slice(-74, -75)
)

# Decode uint16 to Celsius
lst_celsius = subset * 0.01 + (-50.0)
```

### QGIS Access

For QGIS users without native Zarr support, use rioxarray to convert to GeoTIFF:

```python
import rioxarray
import fsspec

url = "s3://us-west-2.opendata.source.coop/radiant-earth/landsat-lst/2023/N40W075.zarr"
ds = xr.open_zarr(fsspec.get_mapper(url, anon=True))
ds["lst_p50"].rio.to_raster("lst_subset.tif")
# Open lst_subset.tif in QGIS
```

See [docs/adr/003-direct-zarr-architecture.md](docs/adr/003-direct-zarr-architecture.md) for architecture details.

## Architecture

See [docs/adr/001-architecture-decisions.md](docs/adr/001-architecture-decisions.md) for detailed design decisions.

Key choices:
- **Data source**: Microsoft Planetary Computer (Landsat C2 L2)
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
