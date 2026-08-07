# landsat-lst Project Instructions

## CRITICAL: STAC Endpoint Rules

**This is enforced by a hook — violations will be blocked.**

| Environment | STAC Endpoint | Why |
|-------------|---------------|-----|
| **Local/dev** | Planetary Computer | Free, no egress costs |
| **AWS/Coiled** | Earth Search | Same-region S3, no egress |

### Local Testing

```python
from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings

# Override for local testing
settings.stac_url = STAC_PLANETARY_COMPUTER
```

Or set environment variable:
```bash
export LST_STAC_URL="https://planetarycomputer.microsoft.com/api/stac/v1"
```

### Production (Coiled/AWS)

The default `STAC_EARTH_SEARCH` is correct for production. No changes needed.

**NEVER use Earth Search locally** — you will pay egress costs for no benefit.

**NEVER use Planetary Computer on AWS** — cross-cloud egress is expensive.

---

## Production Pipeline

**Always use `process_tile()` or `process_tile_job()` for production data.**

These functions apply the full pipeline including:
- STAC query with proper endpoint signing
- Scene loading with improved QA masking (dilated-cloud + cirrus + cloud/shadow/snow)
- LST computation with fill-value handling and a physical-plausibility clamp
- **Land masking** (Natural Earth 10m) to exclude ocean pixels

### Single- and multi-year windows

**The production window is 2021–2025** (`job.DEFAULT_WINDOW`). `generate_jobs()`
with no arguments emits one five-year job per land tile.

`ProcessingJob` takes an optional `end_year`. When set, `datetime_range` spans
`[year, end_year]` and `compute_annual_composite()` pools **every** scene in the
window into one P95 (never average per-year P95s — percentile-of-percentiles is
wrong). `window_label` (`"2024"` or `"2021-2025"`) keys the output storage group
and COG filenames. `end_year=None` is a single-year composite (backward compatible).

### Correct Usage

```python
# Option 1: CLI. No --year runs the 2021-2025 production window.
landsat-lst process --tile N40W075
landsat-lst process --year 2021 --end-year 2025 --tile N40W075
landsat-lst process --year 2024 --tile N40W075   # single year

# Option 2: Python API (single or multi-year)
from landsat_lst.pipeline import process_tile
from landsat_lst.models import ProcessingJob
from landsat_lst.tiling import parse_tile_name

job = ProcessingJob(tile=parse_tile_name("N40W075"), year=2024)
composite = process_tile(job)  # ✅ Includes land mask

# Multi-year window 2021-2025 (pooled P95)
job = ProcessingJob(tile=parse_tile_name("N40W075"), year=2021, end_year=2025)
composite = process_tile(job)  # ✅ Includes land mask

# Option 3: Distributed (Coiled)
from landsat_lst.job import run_distributed, generate_jobs
jobs = generate_jobs()              # 700 tiles x 2021-2025
results = run_distributed(jobs)     # ✅ Includes land mask
```

### DO NOT use `compute_annual_composite()` directly

This low-level function applies land masking only when you hand it one
(`compute_annual_composite(data, land_mask=...)`). `process_tile` always does.
Calling it bare gives you an unmasked composite whose de-striping offsets were
estimated over ocean as well as land. Only use it for benchmarking or when you
explicitly don't want ocean masking. (Despite the name, it also handles
multi-year windows — it pools whatever scenes it is given.)

---

## Data Quality

LST values should be in reasonable ranges:
- Valid range: -20°C to 60°C for most land surfaces
- Summer urban areas: 20°C to 50°C typical
- P95 (hot season): expect 30°C to 50°C for urban tiles

If you see values like -124°C, the data didn't load correctly (DN=0 → invalid conversion).

### Quality controls in the pipeline

- **QA masking** (`create_qa_mask`, `qa.py`): masks dilated cloud (bit 1) and
  cirrus (bit 2) **in addition to** cloud/shadow/snow. The extra bits catch
  cloud-edge and thin-cloud/haze contamination that would otherwise survive as
  per-scene warm/cool residuals and drive scene-footprint striping.
- **Physical-plausibility clamp** (`convert_to_celsius`, `qa.py`): drops LST
  outside `[settings.lst_valid_min, settings.lst_valid_max]` (default −50 / 80 °C).
  This removes the ~−124 °C artifacts produced when reprojection interpolates near
  the DN=0 fill (not caught by the exact `!= 0` test), plus high-DN saturation junk.
- **`qa_count` is a 12-month climatology** (`(month, latitude, longitude)`, `uint8`),
  not a single annual count: month M = valid observations in calendar month M pooled
  across the window. It exports as a 12-band monthly COG (`cog.py`). It counts only
  observations that survive de-striping, so it reports the evidence behind each P95
  value rather than raw data availability.

### De-striping (season-aware normalization) — IN PRODUCTION

`normalization.seasonal_debias` runs inside `compute_annual_composite` whenever
`settings.destripe` is on (the default). It shifts each scene by one scene-wide
offset estimated against a per-pixel **monthly** climatology, so the seasonal
cycle survives. An annual reference was tried and cooled the composite from
40.6 °C to 29.8 °C — do not reintroduce it.

**Discard, never clamp.** Scenes whose offset exceeds
`settings.destripe_max_offset_c`, or that have fewer than
`settings.destripe_min_scene_pixels` valid land pixels, are removed from the
stack entirely. Bounding an offset would leave most of the error in place while
presenting the scene as corrected. The guiding principle for this dataset is to
prefer honest omission over questionable correction.

Consequences worth remembering:

- `qa_count` is computed from the **surviving** stack, so counts describe the
  evidence behind each P95 value, not raw data availability.
- Offsets are estimated over **land only**. `process_tile` builds the land mask
  before compositing and passes it to `compute_annual_composite(land_mask=...)`.
- Offsets are estimated from a **coarse read** (`destripe_offset_resolution_factor`,
  default 2) served from the source COGs' overviews. Do **not** "optimize" this
  by striding or `.coarsen()`-ing the loaded array: that cuts compute but not
  I/O, because dask materializes each chunk before discarding it. Only a coarser
  `resolution=` on `stac_load` reduces bytes fetched. Offset error grows linearly
  in the factor, so raising it requires re-running
  `scripts/validate_offset_subsampling.py`; the ceiling is ~2× regardless, since
  the P95 still needs a native pass.
- `destripe_min_scene_pixels` applies on the **native** path;
  `destripe_min_offset_samples` applies on a **coarse** path. A coarse valid-pixel
  count cannot be scaled back to a native one — averaging spreads data across
  nodata, so coarse loading over-reports coverage (1 valid native pixel read as
  13 at factor 8).
- `destripe_max_offset_c` (15.0 °C) is **calibrated**, not guessed — Pergamino
  2021-2025, 390 solar-day scenes. The offset distribution is a tight core
  (82.7% within ±15 °C, std 5.71) plus a one-sided cold tail from undetected
  cloud, so 15 °C sits near 2.6 core-σ and rejects 21.8% (63 cold scenes vs 1
  warm). Do **not** reason about this cap from the full-sample std (17.01) —
  it is entirely tail-driven and implies a much harsher cut than reality.
  Re-run `scripts/calibrate_destripe_cap.py` on a humid tropical tile before
  the global build; the calibration AOI is mid-latitude cropland.

See [docs/adr/007-scene-normalization.md](docs/adr/007-scene-normalization.md),
[docs/methodology.md](docs/methodology.md), and
[docs/findings-destriping-and-multiyear.md](docs/findings-destriping-and-multiyear.md).

---

## Testing

- Unit tests: `uv run pytest tests/unit/`
- Integration tests: `uv run pytest tests/integration/`
- Full tile test (slow): `uv run pytest -m tile`
