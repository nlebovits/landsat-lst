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
- Scene loading with QA masking
- LST computation with fill value handling
- **Land masking** (Natural Earth 10m) to exclude ocean pixels

### Correct Usage

```python
# Option 1: CLI (recommended for batch processing)
landsat-lst process --year 2024 --tile N40W075

# Option 2: Python API
from landsat_lst.pipeline import process_tile
from landsat_lst.models import ProcessingJob
from landsat_lst.tiling import parse_tile_name

job = ProcessingJob(tile=parse_tile_name("N40W075"), year=2024)
composite = process_tile(job)  # ✅ Includes land mask

# Option 3: Distributed (Coiled)
from landsat_lst.job import run_distributed, generate_jobs
jobs = generate_jobs([2024])
results = run_distributed(jobs)  # ✅ Includes land mask
```

### DO NOT use `compute_annual_composite()` directly

This low-level function does NOT apply land masking. Only use it for
benchmarking or when you explicitly don't want ocean masking.

---

## Data Quality

LST values should be in reasonable ranges:
- Valid range: -20°C to 60°C for most land surfaces
- Summer urban areas: 20°C to 50°C typical
- P95 (hot season): expect 30°C to 50°C for urban tiles

If you see values like -124°C, the data didn't load correctly (DN=0 → invalid conversion).

---

## Testing

- Unit tests: `uv run pytest tests/unit/`
- Integration tests: `uv run pytest tests/integration/`
- Full tile test (slow): `uv run pytest -m tile`
