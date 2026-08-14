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

# Option 3: Distributed (Coiled Batch, one VM per tile)
from landsat_lst.batch import submit_batch, reconcile_run
from landsat_lst.job import generate_jobs

jobs = generate_jobs()                    # 700 tiles x 2021-2025
submission = submit_batch(jobs)           # returns immediately; run outlives this process
results = reconcile_run(submission.run_id)  # after the run: builds the manifest
```

Or from the CLI:

```bash
landsat-lst process --distributed     # submits, prints a run id, returns
landsat-lst reconcile <run-id>        # builds the manifest afterwards
```

### DO NOT use `compute_annual_composite()` directly

This low-level function applies land masking only when you hand it one
(`compute_annual_composite(data, land_mask=...)`). `process_tile` always does.
Calling it bare gives you an unmasked composite whose de-striping offsets were
estimated over ocean as well as land. Only use it for benchmarking or when you
explicitly don't want ocean masking. (Despite the name, it also handles
multi-year windows — it pools whatever scenes it is given.)

---

## Distributed execution — Coiled Batch, never Coiled Functions

Each tile runs as a plain process on its own VM. Do not put `process_tile_job` back inside a
dask cluster's worker: a multi-hour tile graph inside another cluster killed three validation
runs on 2026-08-12, first by escaping to the shared scheduler, then by crushing it with three
tiles at once, then by starving the worker heartbeat until Coiled tore the VM down mid-tile.
See [ADR-010](docs/adr/010-coiled-batch-for-distributed-runs.md).

Two phases that never share a process, plus a live view that needs neither:

```bash
landsat-lst process --distributed   # submits, prints a run id, returns
landsat-lst watch <run-id>          # live: phase and heartbeat age per tile
landsat-lst reconcile <run-id>      # builds the manifest from S3 afterwards
```

Rules worth keeping:

- **Completion is bytes in the bucket, not a task exit code.** A task can exit non-zero after its
  COGs landed. `storage.list_completed()` decides status; exit codes only explain a tile that has
  no output.
- **The submitting shell is disposable.** Nothing may require it to stay open. `submit_batch`
  writes `{run_id}.submission.json` before returning, and that file is all `reconcile_run` needs.
- **Each VM reports for itself** to `_runs/{run_id}/{tile}.json` (duration, scene count, peak RSS,
  error). A missing record is ordinary, not an error: preempted and timed-out VMs never write one.
- **A running tile is only visible through what it publishes.** The cluster dashboard describes a
  dask scheduler that a batch task never registers with, task stdout never reaches `coiled logs`,
  and the exit code Coiled records is the tee wrapper's. So each tile beats to
  `{tile}.progress.json` every 60s (`progress.TileHeartbeat`, rendered by `landsat-lst watch`) and
  uploads its own stdout and stderr to `{tile}.log` on exit either way. Do not reason about a live
  run from the dashboard, and do not trust an exit code. See issue #68.
- **Instrumentation never fails a tile.** Every heartbeat and log write is best-effort: a failure
  is logged and swallowed. Losing observability costs less than losing a two-hour composite.
- **VMs carry 64 GiB.** A heavy tile OOMed at 28.77 GiB on a 32 GiB `r6i.xlarge`.
- **Cost caps are `coiled_max_workers` (concurrent VMs) and `coiled_job_timeout`** (per-task
  wall clock), not a fixed cluster size.

---

## Price a configuration before you run it

Never submit a run to learn a number that follows from array shape and chunking. Task count
and the memory floor are both knowable on a laptop in seconds. Ten validation attempts on
2026-08-13 produced zero completed tiles, at twenty minutes a turn, because every lever was
tested serially in the cloud. See [ADR-011](docs/adr/011-static-planning-and-synthetic-benchmarks.md)
and issue #76.

```bash
landsat-lst plan -t N40W075                      # both graphs, 2021-2025 defaults
landsat-lst plan -t N40W075 --scenes 300 --threads 4
landsat-lst plan -t N40W075 --sweep --fast       # chunk x threads, cheapest first
landsat-lst plan -t N40W075 --json | jq          # stdout is pure JSON
```

Rules worth keeping:

- **Task counts come from the fused graph, never the raw one.** `dask.optimize` runs before
  counting, because that is the graph the scheduler runs and the one `GraphProgress` counts
  against. Raw held 905,923 tasks for the 300-scene N40W075 offset pass where the run
  reported 598,604; fused gives 613,240. Fusion is not a constant (1.48x offsets at 300,
  1.59x at 1,000, 2.71x composite), so a raw count cannot be scaled into a real one. `--fast`
  skips fusion and says so in the output. Never quote an unfused count as a task count.

- **Reported memory is a floor, not a forecast.** Three terms: concurrent per-block time
  stacks (`threads * chunk**2 * scenes * 4`), the resident monthly climatology
  (`12 * height * width * 4`), and a process baseline. A configuration that cannot fit the
  floor is disqualified for free. One that fits may still OOM — the 300-scene N40W075 sample
  peaked at 78.6 GB against a floor of a few GB.
- **Benchmark memory with `scripts/synthetic_scaling.py`, never against a small AOI.**
  Below roughly one degree the whole stack fits in RAM and dask never streams; a five degree
  tile streams from its first block. `scripts/measure_memory_scaling.py` is deprecated for
  exactly this reason, and the new script refuses to extrapolate when peak RSS did not move.
- **Keep `pipeline.TIME_CHUNK` and `profiling.synthetic_dataset` in step.** A synthetic stack
  chunked differently from a real load builds a different graph, which would make planning
  against it worthless.
- **`settings.profile_dask` answers *which* tasks.** A heartbeat fraction of `4182/18600` reads
  the same whether the hour is in `median-aggregate` or a rechunk shuffle. Turn it on for a
  sampled run, not a 700-tile build. `settings.profile_dask_cache` is gated separately: it
  retains one record per task, and these graphs hold hundreds of thousands.

---

## The offset pass is cached — do not pay it twice

`scene_offsets` is 27 of the ~35 minutes in a 300-scene tile (598,604 tasks) and its whole
output is ~600 float64 values. It is cached at
`_offsets/{tile}/{window}/f{factor}/v{version}-{digest}.json`. See
[ADR-012](docs/adr/012-cached-scene-offsets.md) and issue #77.

```bash
landsat-lst offsets   -t N40W075   # estimate, persist, report the rejection fraction
landsat-lst composite -t N40W075   # one tile to COGs, reading whatever is cached
landsat-lst process --distributed  # unchanged fleet driver; forwards the cache flag
```

Rules worth keeping:

- **The cache is keyed, not versioned.** The digest covers the sorted scene ids,
  `destripe_offset_resolution_factor`, and the clamp bounds; `offsets.ALGORITHM_VERSION`
  covers code changes a hash cannot see. **Bump it** when you change `offset_graph`, the QA
  bits in `create_qa_mask`, or the DN-to-Celsius conversion. Nothing else detects that.
- **Only the estimate is cached, never the rejection.** `max_offset_c` and the sparse floors
  are applied to whatever the cache returns, so a cap sweep pays the estimator once. Do not
  "optimize" this by caching the debiased stack.
- **A cache failure never fails a tile.** Same rule as the heartbeat: log and swallow, then
  recompute. Losing 27 minutes beats losing the run.
- **`--no-offset-cache` and `--force` are different.** The first disables both halves and
  leaves the record alone (validating the estimator). The second skips the read and still
  writes (rebuilding an estimate whose inputs did not change). `--force` on `composite` and
  `process` is unrelated: it is about the output COGs.
- **A sampled window cannot check a rejection fraction.** 300 scenes over five years leaves
  each month ~25 scenes for its climatology instead of 244, and the noisy reference inflates
  offsets: 69% rejected on the sample against 21.8% at Pergamino.
- **`rejection_floor` and `scene_keep_mask` are shared** between `seasonal_debias` and
  `landsat-lst offsets` on purpose. A second copy of the rule would drift.

Phases are split finer than the work is, so a silence is attributable
(`stac_query`, `loading`, `land_mask`, `destriping`, `composite_graph`, `exporting`,
`uploading`). `graph_state` in the heartbeat says whether a dask graph is
running at all, which a `None` task count could not. Wrap anything that can exceed ~10s in
`progress.timed_section`. A caller that builds pipeline graphs without being a tile — only
`landsat-lst plan` — wraps them in `progress.silence_sections`, or `plan --json` stops
being parseable.

---

## One native pass per tile — keep it that way

A tile used to read the full native stack three times: an eager coverage reduction, the
LST write, and the QA write. Measured on a synthetic tile, that was exactly 3.0x one pass.
It is 1.0x now. See [ADR-013](docs/adr/013-single-native-pass.md) and issue #80.

Two things hold it there, and either one alone gives back a pass:

- **`_composite_graph` rechunks time to a single chunk before building either output.**
  `quantile` needs the whole time series per pixel and would insert that rechunk anyway;
  `groupby("time.month").sum()` would not, and two differently chunked consumers means
  every source block is materialized twice. Worse, the fused write then has no
  block-by-block order that satisfies both, so the scheduler fans out and holds the stack:
  **10.88 GB** peak against 1.30 GB, on a 4096² x 120 synthetic tile. With the shared
  rechunk it is 1.60 GB. The rechunk adds nothing to the memory floor, because the P95
  already forced it.
- **`cog_export` writes both intermediates in one `dask.compute`.** `rio.to_raster(compute=False)`
  returns a deferred store per product; handing both to one compute retires the shared
  source blocks once. Calling `export_lst_cog` and `export_qa_cog` in sequence costs two
  passes, which is what `tests/integration/test_cog.py` pins from both sides.

The coverage diagnostic survived the deletion. `valid_coverage_obs_per_pixel` reports the
same four numbers, now accumulated as a histogram of per-pixel month sums during the
windowed statistics walk the exporter already runs over the written QA raster. That walk
also went from one pass per band to one pass total. Do not restore an eager `.values` on
`qa_count` to get a number the COG tags already carry — but note that QA
`STATISTICS_VALID_PERCENT` is **always 100** by construction (`nodata=None`, because 0
observations is data), so it is the LST band's `VALID_PERCENT` and the coverage line, not
the QA one, that catch a run that filled wholesale.

---

## Output grid — one shared grid, always

`settings.pixels_per_degree` (3600) is the grid definition; `settings.resolution` is a
**derived property** (`1/3600`), not a settable field. Do not reintroduce a resolution float:
the old `0.00027778` truncated 1/3600 and left every tile anchored to its own bbox, overshooting
its eastern edge by 0.484 px and misregistering ~0.14 px against its neighbour.

Load scenes through `tiling.geobox_for_bbox(bbox, factor)` and pass `geobox=` to `stac_load`.
Never pass `crs` + `resolution` + `bbox` — odc-stac anchors the grid to the bbox, which is
exactly the bug above.

Numbers worth remembering: global grid 1,296,000 × 432,000; a 5° tile is 18,000² = 2⁴·3²·5³.
That divides by 4 and 16 but **not** by 64, which is why overviews belong to the global array
and not to a tile. See [ADR-008](docs/adr/008-global-mosaic-topology.md).

Beware `int()` on a resolution-derived span: `int(5 / (1/3600))` is 17999, not 18000. Use
`round()`.

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
