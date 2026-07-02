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

`ProcessingJob` takes an optional `end_year`. When set, `datetime_range` spans
`[year, end_year]` and `compute_annual_composite()` pools **every** scene in the
window into one P95 (never average per-year P95s — percentile-of-percentiles is
wrong). `window_label` (`"2024"` or `"2020-2024"`) keys the output storage group
and COG filenames. `end_year=None` is a single-year composite (backward compatible).

### Correct Usage

```python
# Option 1: CLI (single year; recommended for batch processing)
landsat-lst process --year 2024 --tile N40W075

# Option 2: Python API (single or multi-year)
from landsat_lst.pipeline import process_tile
from landsat_lst.models import ProcessingJob
from landsat_lst.tiling import parse_tile_name

job = ProcessingJob(tile=parse_tile_name("N40W075"), year=2024)
composite = process_tile(job)  # ✅ Includes land mask

# Multi-year window 2020-2024 (pooled P95)
job = ProcessingJob(tile=parse_tile_name("N40W075"), year=2020, end_year=2024)
composite = process_tile(job)  # ✅ Includes land mask

# Option 3: Distributed (Coiled)
from landsat_lst.job import run_distributed, generate_jobs
jobs = generate_jobs([2024])
results = run_distributed(jobs)  # ✅ Includes land mask
```

### DO NOT use `compute_annual_composite()` directly

This low-level function does NOT apply land masking. Only use it for
benchmarking or when you explicitly don't want ocean masking. (Despite the
name, it also handles multi-year windows — it pools whatever scenes it is given.)

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
  across the window. It exports as a 12-band monthly COG (`cog.py`).

### De-striping (season-aware normalization) — PROTOTYPE, NOT IN PRODUCTION

Season-aware per-scene bias normalization is the **chosen** de-striping approach,
but it currently lives **only** in the prototype `scripts/season_aware_p95_test.py`
and is **NOT wired into the production pipeline** (`process_tile` / `compute_annual_composite`).
Production instead relies on stricter QA masking, the plausibility clamp, and
multi-year pooling to suppress striping. Do not describe season-aware normalization
as a production feature. See
[docs/findings-destriping-and-multiyear.md](docs/findings-destriping-and-multiyear.md)
and [docs/adr/005-multiyear-monthly-qa-and-destriping.md](docs/adr/005-multiyear-monthly-qa-and-destriping.md).

---

## Testing

- Unit tests: `uv run pytest tests/unit/`
- Integration tests: `uv run pytest tests/integration/`
- Full tile test (slow): `uv run pytest -m tile`
