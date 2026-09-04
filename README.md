# Landsat LST

Build annual or multi-year land surface temperature (LST) composites from
Landsat Collection 2 Level-2 data.

## Overview

This pipeline produces LST composites for municipal decision-makers who analyze
urban heat. Each composite pools *every* scene in a one-year or multi-year window
into a single percentile. Longer windows fill cloud and orbit gaps. **Production
uses the five-year window from 2021 through 2025 by default.**

Each tile contains two assets:

- **lst_p95**: 95th percentile LST (°C), pooled across all scenes in the window
- **qa_count**: 12-month climatology of valid-observation counts (`(month, latitude, longitude)`, one band per calendar month; month M = valid observations in calendar month M pooled across the window)

Before compositing, the pipeline estimates one scene-wide offset against a
per-pixel monthly climatology and shifts the scene by that amount. This
normalization removes seams at satellite footprint boundaries. The pipeline
discards a scene when its correction would be implausibly large, and `qa_count`
counts only surviving observations. [`docs/methodology.md`](docs/methodology.md)
explains the method. [ADR-007](docs/adr/007-scene-normalization.md) records the
measurements behind it.

The pipeline cuts each tile from a shared 5° global grid and publishes two
Cloud-Optimized GeoTIFFs (COGs) per tile. One
[Portolan](https://github.com/portolan-sdi/portolan-spec) SpatioTemporal Asset
Catalog (STAC) collection catalogs the results. Because every tile uses the same
grid, QGIS, GDAL, TiTiler, and odc-stac can mosaic the collection without
reprojection. See [Data access](#data-access) for examples and
[ADR-009](docs/adr/009-cog-output-and-stac-catalog.md) for the design.

## Data encoding

The writer stores LST as **uint16** to halve the file size and embeds the scale
and offset in the GDAL band metadata. QGIS, `gdalinfo`, and rioxarray read this
decoding rule from the file. Most readers then apply it automatically:

```python
import rioxarray

da = rioxarray.open_rasterio("lst_p95.tif", masked=True)

# The decoding rule travels with the file.
scale, offset = da.rio.scales[0], da.rio.offsets[0]  # 0.01, -50.0
celsius = da * scale + offset
```

**Quick decode (if you know the constants):**
```python
celsius = dn * 0.01 + (-50.0)  # DN 0 is nodata
```

| Asset | Name | Dtype | Scale | Offset | Nodata | Units |
|-------|------|-------|-------|--------|--------|-------|
| lst_p95 | 95th percentile LST | uint16, 1 band | 0.01 | -50.0 | 0 | celsius |
| qa_count | Valid observations per calendar month | uint8, 12 bands (Jan..Dec) | — | — | none | count |

`qa_count` has no nodata value by design. A value of 0 means that no valid
observation survived de-striping for that month. Keeping zero visible lets you
distinguish this gap from masked data.

## Installation

```bash
uv sync
```

## Configuration

[`landsat_lst.config`](src/landsat_lst/config.py) defines the settings. Override
them with `LST_`-prefixed environment variables or a `.env` file. These settings
control data quality and resource use:

| Setting | Env var | Default | Purpose |
|---------|---------|---------|---------|
| `lst_valid_min` | `LST_LST_VALID_MIN` | `-50.0` | Drop implausibly cold LST, such as the ~-124 °C produced by DN=0 or resampling artifacts |
| `lst_valid_max` | `LST_LST_VALID_MAX` | `80.0` | Drop high-DN saturation and fill artifacts without clipping real extreme heat |
| `load_chunk_size` | `LST_LOAD_CHUNK_SIZE` | `512` | Set the odc-stac spatial chunk. A smaller chunk, such as 256, reduces peak memory during the P95 quantile for multi-year or large-tile runs |
| `max_cloud_cover` | `LST_MAX_CLOUD_COVER` | `100` | Keep scenes where `eo:cloud_cover` is below this value. The strict comparison at 100 already drops the 154 of 2,912 N40W075 scenes reported as 100% cloudy. Use 101 for a true no-op. Read the [findings](docs/findings-cloud-cover-filter.md) before lowering it |
| `destripe` | `LST_DESTRIPE` | `True` | Normalize each scene against a monthly climatology before compositing. Disable this setting only when benchmarking raw composites |
| `destripe_max_offset_c` | `LST_DESTRIPE_MAX_OFFSET_C` | `15.0` | Discard a scene when its offset exceeds this value. [ADR-007](docs/adr/007-scene-normalization.md) calibrates the default at Pergamino; use `scripts/calibrate_destripe_cap.py` for other climates |
| `destripe_min_scene_pixels` | `LST_DESTRIPE_MIN_SCENE_PIXELS` | `500` | Set the sparse floor for native-resolution offset estimates. On a coarse grid, use `destripe_min_offset_samples` instead; do not convert between the two floors |
| `destripe_min_offset_samples` | `LST_DESTRIPE_MIN_OFFSET_SAMPLES` | `200` | Set the sparse floor for coarse-grid offset estimates, measured in pixels on that grid |
| `destripe_offset_resolution_factor` | `LST_DESTRIPE_OFFSET_RESOLUTION_FACTOR` | `2` | Estimate offsets from source COG overviews at `resolution × factor`. Factor 2 cuts the offset pass from 20.2 GB to 5.1 GB and is the largest validated value. Validation rejected factor 4 for [#81](https://github.com/nlebovits/landsat-lst/issues/81); see the [findings](docs/findings-offset-subsampling.md) |

## Usage

```bash
# Process all land tiles for the production window (2021-2025)
landsat-lst process

# Process a single tile for the production window
landsat-lst process --tile N40W075

# Explicit window
landsat-lst process --year 2021 --end-year 2025 --tile N40W075

# Single year
landsat-lst process --year 2023 --tile N40W075

# List available tiles
landsat-lst list-tiles
```

### Multi-year composites

Set `ProcessingJob.end_year` to pool every scene in a multi-year window into one
P95. The pipeline computes the percentile over the pooled scenes rather than
averaging per-year P95 values. A label such as `2024` or `2020-2024` identifies
the STAC collection that receives the tile COGs:

```python
from landsat_lst.models import ProcessingJob
from landsat_lst.pipeline import process_tile
from landsat_lst.tiling import parse_tile_name

# Single year (backward compatible)
job = ProcessingJob(tile=parse_tile_name("N40W075"), year=2024)

# Five-year window 2020-2024 -> collection "lst-p95-2020-2024"
job = ProcessingJob(tile=parse_tile_name("N40W075"), year=2020, end_year=2024)
composite = process_tile(job)
```

### Distributed processing with Coiled Batch

Production runs use [Coiled Batch](https://docs.coiled.io/user_guide/batch.html).
Each task processes one tile in a plain process on its own virtual machine; the
tasks do not share a scheduler. [ADR-010](docs/adr/010-coiled-batch-for-distributed-runs.md)
explains this design.

Before submitting, inspect the configuration locally. Array shape and chunking
determine the task count and memory floor, which the planner reports in seconds.
[ADR-011](docs/adr/011-static-planning-and-synthetic-benchmarks.md) explains why
these checks happen before a cloud run.

```bash
# Build both of a tile's Dask graphs against synthetic data
landsat-lst plan -t N40W075

# Chunk size crossed with thread count, cheapest floor first
landsat-lst plan -t N40W075 --sweep
```

The planner reports a memory floor, not a forecast. Reject a configuration when
the floor does not fit, but do not assume that a fitting floor guarantees enough
memory. To measure the gap, sweep scene count against production chunking with
`scripts/synthetic_scaling.py`.

```bash
# Ensure AWS SSO session is active
aws sso login --profile radiant-earth

# Submit every land tile for the production window; returns immediately
landsat-lst process --distributed

# Or a few tiles, waiting for them to finish
landsat-lst process --distributed --wait -t N40W075 -t S05W060

# Dry run (show the job list without submitting)
landsat-lst process --distributed --dry-run
```

Submission prints a run ID before handing the work to Coiled. The run continues
after you close the shell.

```bash
# Live view: phase, elapsed time, and heartbeat age for every tile
landsat-lst watch <run-id>

# Afterwards: the durable record of what the run produced
landsat-lst reconcile <run-id>
```

Each tile publishes a heartbeat to `_runs/{run_id}/{tile}.progress.json` every
minute and at each phase change. When the tile exits, it uploads stdout and stderr
to `_runs/{run_id}/{tile}.log`. The `watch` command renders these heartbeats in one
table. A wedged tile develops a stale heartbeat within two minutes, while a failed
tile leaves its traceback in the bucket. You can run `watch` and `reconcile` from
any machine; neither command depends on the submitting shell.

The cluster dashboard cannot report tile progress because a batch task never
registers with the Dask scheduler represented by its panels (issue #68).

The manifest stores per-tile status, duration, scene count, and peak memory under
`settings.manifest_dir`. An S3 listing determines completion, which limits a
resumed run to tiles that are missing an asset.

### Verifying published tiles

Object presence does not prove public readability. Verify each published tile:

```bash
landsat-lst verify -t N40W075 --urls
```

The command opens each COG without credentials through the public read host. It
prints the dtype, shape, nodata value, scale, offset, and overview levels. A tile
that requires credentials fails verification, and any failure produces a nonzero
exit status.

## Data access

Each window produces one Portolan STAC collection. The collection contains one
item per tile and two COG assets per item. Source Cooperative uses this layout:

```
nlebovits/landsat-lst/
├── catalog.json
└── lst-p95-2021-2025/
    ├── collection.json
    ├── items.parquet          # stac-geoparquet mirror of every item
    ├── thumbnail.png
    └── N40W075/
        ├── N40W075.json       # STAC item
        ├── lst_p95.tif        # uint16, 1 band
        └── qa_count.tif       # uint8, 12 bands (Jan..Dec)
```

Every asset is a window on the same global grid at exactly 1/3600°. Tiles
therefore align pixel for pixel, allowing a reader to combine any number of them
as one raster.

### Python

```python
import rioxarray

# Once published:
url = "https://data.source.coop/nlebovits/landsat-lst/lst-p95-2021-2025/N40W075/lst_p95.tif"

# Range requests: only the bytes for the requested window are fetched.
da = rioxarray.open_rasterio(url, masked=True)
subset = da.rio.clip_box(minx=-75, miny=40, maxx=-74, maxy=41)
celsius = subset * da.rio.scales[0] + da.rio.offsets[0]
```

To read many tiles at once, load the STAC collection instead of individual files:

```python
import odc.stac
import pystac_client

catalog = pystac_client.Client.open("https://data.source.coop/nlebovits/landsat-lst")
items = list(catalog.get_collection("lst-p95-2021-2025").get_items())
mosaic = odc.stac.load(items, bands=["lst_p95"], chunks={})
```

### Open a tile in QGIS

QGIS can open an asset without a plugin or local download. Choose **Layer > Add
Layer > Add Raster Layer**, select **Protocol: HTTP(S)** as the source type, and
paste the asset URL. QGIS reads the overview for the current zoom and applies the
embedded scale and offset. The layer then displays temperatures in Celsius.

### Inspect a tile with gdalinfo

```bash
gdalinfo /vsicurl/https://data.source.coop/nlebovits/landsat-lst/\
lst-p95-2021-2025/N40W075/lst_p95.tif
```

The report includes the block size, overview levels, band scale and offset, and
embedded statistics. Because the writer stores statistics in the COG rather than
an `.aux.xml` sidecar, readers receive them over HTTPS.

### Producing COGs from a composite

[`landsat_lst.cog`](src/landsat_lst/cog.py) writes the same asset pair as the
pipeline. Pass it a composite that follows the uint16 contract defined in
[`landsat_lst.encoding`](src/landsat_lst/encoding.py):

```python
import xarray as xr
from landsat_lst.cog import cog_export
from landsat_lst.encoding import encode_lst_uint16
from landsat_lst.models import ProcessingJob
from landsat_lst.pipeline import process_tile
from landsat_lst.tiling import parse_tile_name

job = ProcessingJob(tile=parse_tile_name("N40W075"), year=2021, end_year=2025)
composite = process_tile(job).compute()

native = xr.Dataset(
    {
        "lst_p95": encode_lst_uint16(composite["lst_p95"]),
        "qa_count": composite["qa_count"],
    }
)
cog_export(native, "lst_p95.tif", "qa_count.tif")
```

### Publishing the catalog

```bash
landsat-lst catalog build \
  --source s3://bucket/prefix \
  --out ./catalog
landsat-lst catalog validate ./catalog
landsat-lst catalog publish ./catalog \
  --remote s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst/ \
  --dry-run
```

`publish` assigns each object the media type declared by its extension. It always
resends JSON and Markdown but skips an asset when its remote size matches. After
publishing the tree, add
`--live --live-base-url https://data.source.coop/nlebovits/landsat-lst/` to
`validate`. This probes the host for range support, CORS, and Content-Length.

The dataset is not yet published. Follow
[`docs/runbook-publication.md`](docs/runbook-publication.md) for the operational
sequence and the two unresolved decisions.

## Architecture

The project records its design decisions in
[`docs/adr/`](docs/adr/README.md). Start with
[ADR-001](docs/adr/001-architecture-decisions.md) for the architectural context.

- **Data source:** Earth Search with Landsat Collection 2 Level-2 data
- **Output:** COGs with 512 × 512 blocks in a Portolan STAC catalog ([ADR-009](docs/adr/009-cog-output-and-stac-catalog.md))
- **Coordinate reference system:** EPSG:4326
- **Grid:** 5° × 5° tiles on one global 1/3600° grid ([ADR-008](docs/adr/008-global-mosaic-topology.md))
- **Time:** Multi-year pooled-P95 windows, with 2021–2025 as the production default
- **Coverage:** Land between 60° S and 60° N

## Known limitations

### Permanent gaps from ASTER emissivity

Landsat Collection 2 Level-2 Surface Temperature requires an emissivity value for
each pixel. It draws those values from the ASTER Global Emissivity Dataset, which
uses clear-sky ASTER scenes acquired from 2000 through 2008. Some locations had
no clear-sky ASTER observation during that period. Without an emissivity value,
USGS cannot produce Surface Temperature, leaving those pixels empty throughout
the Landsat archive.

The pipeline leaves these gaps empty because processing cannot recover the
missing input. A wider window closes cloud and orbit gaps but cannot replace a
missing value in the static emissivity dataset.

![ASTER GED emissivity coverage: blue where data exists, white where it does not](docs/images/aster-ged-coverage-usgs.jpg)

*Blue is available data, white is none. Figure by USGS, public domain, from
[Landsat Collection 2 Surface Temperature data gaps due to missing ASTER
GED](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature-data-gaps-due-missing-aster-ged).
This view is global land; the numbers below are urban land only.*

Against GHS-SMOD R2023A, the analysis measured no emissivity over **2.66% of the
world's urban land**: 80,397 km² of 3,027,063 km². Another **10.23% rests on one
or two observations**. These figures cover urban land only. The global average
hides regional variation because persistent cloud determines where gaps occur:

| Region | Urban gap % |
|---|---:|
| Southeast Asia | 12.07 |
| Amazonia | 11.62 |
| Southern Africa | 8.36 |
| Europe | 2.80 |
| North America | 1.18 |
| Australia | 0.30 |
| Sahara and Sahel | 0.00 |

For this product, deserts have the highest coverage and the wet tropics have the
lowest.

To detect an affected pixel, check for `qa_count == 0` in all 12 months within
the land mask. Because `process_tile` also sets `qa_count` to zero over water,
running the test without the land mask would conflate gaps with ocean:

```python
from landsat_lst.masks import get_land_mask_for_bbox, load_land_polygons

land = get_land_mask_for_bbox(
    tile.bbox,
    settings.resolution,
    load_land_polygons(),
    target_shape=shape,
)
gap = (composite["qa_count"].sum("month").to_numpy() == 0) & land
```

[The ASTER GED gap findings](docs/findings-aster-ged-gaps.md) contain the complete
results and method. [ADR-006](docs/adr/006-no-aster-gap-filling.md) explains why
the product leaves gaps empty instead of filling them from another emissivity
source.

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

# Run every blocking commit-stage check
uv run prek run --all-files --show-diff-on-failure
```

Before you commit documentation, run the advisory voice checks on the files you
changed. `prek` hides output from a successful hook unless you pass `--verbose`,
so keep that option on the Vale audit:

```bash
uv run prek --verbose run vale-audit --files README.md --stage manual
uv run prek run proselint --files README.md --stage manual
```

Replace `README.md` with your changed Markdown files, or use `--all-files` for a
repository-wide audit. The repository has known findings in historical documents.
The audits do not block CI, although proselint exits nonzero when it reports an
advisory finding. The [prose guide](docs/PROSE.md) explains the repository voice,
readability scores, suppressions, and rule ownership.

### Optional extras

`uv sync --all-extras` installs every optional dependency. You can also select
an individual extra:

- `analysis` — `matplotlib` for figure generation in analysis/findings writeups, plus
  `h5py` and `earthaccess` for the ASTER GED gap analysis.
- `frisky` — experimental Rust reimplementation of the Dask scheduler
  ([getfrisky.dev](https://getfrisky.dev)). The multi-year decision driver uses
  Frisky when available and falls back to Dask otherwise. Because Frisky crashed
  while gathering large results, **production uses Dask**.

### Scripts

[`scripts/`](scripts/) contains these development and analysis tools:

- `pergamino_multiyear_decision.py` compares one-, three-, and five-year
  windows against Planetary Computer data. It writes a COG pair for each window
  and a `report.md` with gap, striping, and measured compression metrics. Pass
  `--no-frisky` to force Dask.
- `smoke_small_tile_cog.py` runs the pipeline on a ~0.2° slice of Pergamino. Use
  it to check the path from STAC query through COG export without recomputing a
  5° tile.
- `season_aware_p95_test.py` preserves the original season-aware de-striping
  prototype as a standalone driver. The production method now lives in
  `landsat_lst.normalization`. See
  [docs/findings-destriping-and-multiyear.md](docs/findings-destriping-and-multiyear.md)
  and [ADR-007](docs/adr/007-scene-normalization.md).
- `calibrate_destripe_cap.py` sweeps `destripe_max_offset_c` candidates over one
  load and reports the discarded-scene fraction. It produced the 15 °C default;
  rerun it for climates unlike mid-latitude cropland.
- `validate_offset_subsampling.py` compares coarse-overview offset estimates
  with full-resolution estimates for each scene. It produced the shipped
  `destripe_offset_resolution_factor`. Rerun it before raising the factor because
  a value that passed in August failed on the current grid.
- `analyze_cloud_cover_filter.py` weighs the scenes skipped by a candidate
  `max_cloud_cover` against the valid observations lost. It reads validation
  output without loading source imagery.
- `measure_climatology_thinning.py` measures how a smaller scene set thins the
  monthly climatology and changes offsets for the remaining scenes.
- `compare_destripe_composites.py` builds raw, natively de-striped, and
  coarse-offset P95 composites from one load. It reports their differences and
  writes COGs for QGIS when you pass `--cogs`.
- `aster_gap_urban_analysis.py` measures ASTER GED gaps against GHS-SMOD to find
  the share of urban land without Surface Temperature. It requires the
  `analysis` extra and a NASA Earthdata login; see [Known limitations](#known-limitations).

## License

MIT
