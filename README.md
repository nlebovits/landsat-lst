# Landsat LST

Global annual Land Surface Temperature composites from Landsat Collection 2 Level-2 data.

## Overview

This pipeline produces annual LST composites for municipal decision-makers analyzing urban heat. Output includes:

- **lst_p50**: Median LST (°C)
- **lst_p95**: 95th percentile LST (°C)
- **qa_count**: Number of valid observations

Data is tiled on a 5° global grid, stored as Cloud-Optimized GeoTIFFs (COGs), and published as a STAC catalog.

## Data Encoding

LST bands are stored as **uint16** to reduce file size by 50%. To convert back to Celsius:

```python
import rasterio

with rasterio.open("N40W075_2023.tif") as src:
    # Read scale/offset from TIFF tags
    tags = src.tags(1)  # Band 1 (lst_p50)
    scale = float(tags["LST_SCALE"])    # 0.01
    offset = float(tags["LST_OFFSET"])  # -50.0

    # Read data and decode
    dn = src.read(1)
    nodata_mask = dn == 0
    celsius = dn * scale + offset
    celsius[nodata_mask] = float("nan")
```

**Quick decode (if you know the constants):**
```python
celsius = dn * 0.01 + (-50.0)  # DN=0 is nodata
```

| Band | Name | Scale | Offset | Units |
|------|------|-------|--------|-------|
| 1 | lst_p50 | 0.01 | -50.0 | celsius |
| 2 | lst_p95 | 0.01 | -50.0 | celsius |
| 3 | qa_count | — | — | count |

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

## Architecture

See [docs/adr/001-architecture-decisions.md](docs/adr/001-architecture-decisions.md) for detailed design decisions.

Key choices:
- **Data source**: Element 84 Earth Search (Landsat C2 L2)
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
