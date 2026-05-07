# Landsat LST

Global annual Land Surface Temperature composites from Landsat Collection 2 Level-2 data.

## Overview

This pipeline produces annual LST composites for municipal decision-makers analyzing urban heat. Output includes:

- **lst_p50**: Median LST (°C)
- **lst_p95**: 95th percentile LST (°C)
- **qa_count**: Number of valid observations

Data is tiled on a 5° global grid, stored as Cloud-Optimized GeoTIFFs (COGs), and published as a STAC catalog.

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
