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
landsat-lst process --distributed        # submits, prints a run id, returns
landsat-lst watch <run-id>               # live: phase, task rate, ETA, RSS trend, cost
landsat-lst watch <run-id> --detail      # adds per-tile panels: sparklines, phase bars, headroom
landsat-lst explain <run-id> [tile]      # every attempt: state, timings, profile, log tail
landsat-lst reconcile <run-id>           # manifest, attempt series, and cost, from S3
```

Rules worth keeping:

- **Completion is bytes in the bucket, not a task exit code.** A task can exit non-zero after its
  COGs landed. `storage.list_completed()` decides status; exit codes only explain a tile that has
  no output.
- **The submitting shell is disposable.** Nothing may require it to stay open. `submit_batch`
  writes `{run_id}.submission.json` before returning, and that file is all `reconcile_run` needs.
- **Each VM reports for itself, once per attempt**, to `_runs/{run_id}/{tile}.{attempt}.json`: one
  merged state object carrying both the live phase and the outcome. A copy of the final state goes
  to the unsuffixed `{tile}.json`, and its presence is how every reader tells a settled tile from a
  running one. `status` is `null` until a tile settles, so use `phase` for liveness. A missing
  object is ordinary, not an error: preempted and timed-out VMs never write one.
- **Every artifact is keyed by attempt, and the number is resolved once per process.** Three Coiled
  retries used to write the same three keys, so the manifest reported a 10-second failure against a
  33-minute run and the attempt that reached `land_mask` was erased. `runs.resolve_attempt` counts
  state objects, logs, *and* profiles, because a VM preempted before it published state still
  leaves a log. Asking twice in one process would number the log higher than the state object, since
  the log uploads last. `runs.py` owns the key grammar; do not re-derive suffixes anywhere else.
- **An escaping exception writes the attempt object but not the pointer.** A transient failure
  re-raises so Coiled retries it, and no reader should see a failed final answer mid-retry.
- **A running tile is only visible through what it publishes.** The cluster dashboard describes a
  dask scheduler that a batch task never registers with, task stdout never reaches `coiled logs`,
  and the exit code Coiled records is the tee wrapper's. So each tile republishes its state every
  60s (`progress.TileHeartbeat`, rendered by `landsat-lst watch`) and uploads its own stdout and
  stderr to `{tile}.{attempt}.log` on exit either way. Do not reason about a live run from the
  dashboard, and do not trust an exit code. See issue #68 and ADR-014.
- **Instrumentation never fails a tile.** Every heartbeat and log write is best-effort: a failure
  is logged and swallowed. Losing observability costs less than losing a two-hour composite.
- **VMs carry 64 GiB.** A heavy tile OOMed at 28.77 GiB on a 32 GiB `r6i.xlarge`.
- **Cost caps are `coiled_max_workers` (concurrent VMs) and `coiled_job_timeout`** (per-task
  wall clock), not a fixed cluster size.
- **Cost is an estimate and a range, never a scalar.** `pricing.json` holds committed list prices
  with an `as_of` date. Spot spans 0.30-0.75 of on-demand, sampled at 0.35x, 0.44x, and 0.71x on
  one day, so a single discount factor would be wrong by more than 2x for one of the two configured
  VM types. `spot_with_fallback` with no measured lifecycle spans 0.30 to 1.00. The instance type
  is read from EC2 IMDS on the VM, not assumed from `coiled_vm_types[0]`, because the fallback type
  costs 1.52x the primary for the same 64 GiB. EC2 bills per second with a 60-second minimum.
- **The STAC client retries 429 and 5xx on every verb.** `pystac_client` mounts an adapter from a
  plain int, which leaves `status_forcelist` empty and excludes POST, so a 500 on a search was
  never retried and one blip killed a five-hour tile at second 10. Retry in the client, never by
  raising `coiled_retries`: a VM restart destroys the tile's progress and an HTTP retry keeps it.
- **Nothing may render tracebacks with frame locals.** `logging_config.configure_logging` installs
  `structlog.dev.plain_traceback`, because the default `ConsoleRenderer` uses rich with
  `show_locals=True` and one `logger.exception` rendered 3.8 MB of deserialized STAC collection,
  evicting a tile's whole phase history from its log. Raising `task_log_max_bytes` is not the fix.

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
- **Benchmark memory with `landsat-lst benchmark`, never against a small AOI.**
  Below roughly one degree the whole stack fits in RAM and dask never streams; a five degree
  tile streams from its first block. `scripts/measure_memory_scaling.py` is deprecated for
  exactly this reason, and the sweep refuses to extrapolate when peak RSS did not move.
  `scripts/synthetic_scaling.py` is now a thin wrapper over the same code, which lives in
  `landsat_lst.benchmarks` because Coiled ships the installed package, not `scripts/`.
- **Keep `pipeline.TIME_CHUNK` and `profiling.synthetic_dataset` in step.** A synthetic stack
  chunked differently from a real load builds a different graph, which would make planning
  against it worthless. `tests/benchmark/conftest.py` fails loudly if `TIME_CHUNK` moves.
- **`settings.profile_dask` answers *which* tasks.** A heartbeat fraction of `4182/18600` reads
  the same whether the hour is in `median-aggregate` or a rechunk shuffle. It is **on by
  default for any run passing `--max-scenes`**, because a sample exists to be measured; a
  700-tile build passes no `--max-scenes` and is untouched. An explicit `LST_PROFILE_DASK`
  wins either way, read from the environment rather than from `settings.model_fields_set`,
  which pydantic mutates on plain attribute assignment. `settings.profile_dask_cache` is gated
  separately: it retains one record per task, and these graphs hold hundreds of thousands.

---

## Three benchmark tiers, and never confuse them

Ten validation attempts on 2026-08-13 produced zero completed tiles because every lever was
tested serially in the cloud. The failure was not spending money. It was spending time on the
wrong tier. See [findings](docs/findings-memory-model.md) and issue #94.

| Tier | Command | Cost | Answers |
|---|---|---|---|
| Instant, local | `landsat-lst plan` | seconds | task counts, the memory floor |
| CI regression | `pytest tests/benchmark` | ~30s | did a change move the number |
| Cloud, sampled | `landsat-lst benchmark --distributed` | ~20 min, under a dollar | peak RSS at production geometry |
| Cloud, full window | `landsat-lst process --distributed` | hours | the product |

Rules worth keeping:

- **`plan` reports a floor; only a VM reports a peak.** The floor landed at ~17 GB for the run
  that OOMed at 46.5 GB. The gap is the `groupby` shuffle, the anomaly broadcast, and the
  spatial median, none of which the three-term model covers. Do not quote a floor as a forecast.
- **Execution goes to the VM; the laptop gets graph inspection.** `landsat-lst benchmark`
  refuses more than 200 scenes locally, `plan` carries `--max-tasks`, and an unbounded local
  build has taken a 64 GB desktop down. Building a graph allocates Python objects whether or
  not you compute it.
- **Every measurement runs in a fresh subprocess.** `getrusage` reports a high-water mark for
  the life of a process, so a second configuration in the first one's interpreter inherits its
  peak and draws a flat curve whatever the truth is.
- **`tests/benchmark/` asserts on bands, never on values.** A benchmark that fails on 3% drift
  gets disabled within a month. The bands are sized against a known regression: deleting the
  shared rechunk in `_composite_graph` moves the composite from 828 tasks to 1,326 (1.60x) and
  peak RSS from 308 MB to 842 MB (2.73x), so the composite band is 1.4x where the offset band
  is 2.0x.
- **The read tally does not catch the `_composite_graph` regression.** It stays at 1.0 pass
  either way: both consumers descend from the same source keys, and within one `dask.compute`
  each key is produced once whatever is downstream. Pass count is the right guard for
  `cog_export`, where the products are separate computes, and the wrong one here. An earlier
  draft asserted it, passed with the fix deleted, and would have shipped the regression.
- **The trend file is the point, not the pass/fail.** A single green build cannot show a number
  drifting toward a cliff over five PRs. Nightly uploads `results/benchmark/trend.jsonl`.
- **A sweep publishes under `_benchmarks/`, never `_runs/`.** `runs.py` classifies everything
  under the run prefix as a tile attempt, and a sweep is not a tile. The corollary is that
  `landsat-lst watch` and `explain` cannot see it, so the sweep publishes its own progress:
  the whole result object is republished when a scene count **starts** (`in_flight`) and again
  when it **lands** (`completed`, `status`), and stdout goes to `_benchmarks/{run_id}/sweep.log`
  on exit either way. `--distributed` follows by default; `--follow <run-id>` re-attaches and
  `--no-follow` opts out. Publishing only at the end would be 25 minutes of silence and nothing
  at all on a crash, which is the mistake ADR-014 already paid for once. Announcing only on
  completion is the same mistake at smaller scale, because the top point runs twelve minutes.

---

## A cached fixture for accuracy work, and what it cannot answer

Comparing two offset estimators means running both over the same scenes. Without a fixture that
is a STAC query and hundreds of gigabytes of coarse reads per iteration, for an answer that is
600 floats.

```bash
landsat-lst fixture --tile N40W075 --factor 8    # ~6.1 GB, fetched once
landsat-lst fixture --list
landsat-lst fixture --factor 2 --dry-run         # prints the arithmetic, fetches nothing
```

- **Check the size before you fetch, because the command does.** A five-degree tile at
  production's offset factor 2 is a 9,000 squared grid, and 300 scenes of two `uint16` bands
  over it is **97 GB**. Factor 4 is 24.3, factor 8 is 6.1, factor 16 is 1.5. The guard refuses
  above 8 GB and names both ways out; it runs before the STAC query, not after the download.
- **The fixture answers a relative question, not an absolute one.** Offset error grows linearly
  in the factor, but both estimators read the same pixels, so a comparison is exact at any
  factor. Do not quote a fixture offset as a production offset.
- **It cannot answer the memory question.** Below the streaming regime the stack fits in RAM and
  dask never streams, which is the behaviour under test there. Use `landsat-lst benchmark`.
- **Stored uncompressed, deliberately.** Plain `.npy` per band means `load_fixture` memory-maps
  it and hands dask a lazy array at production chunking, so the graph built on a fixture is the
  graph a real tile builds. Compressing it would force a materialized stack and make every
  downstream measurement describe a pipeline that does not stream.

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
- **Time stamps are load-bearing, and serialized at nanoseconds.** `debias_with_offsets`
  joins offsets to a stack **by coordinate value**, so a stamp that does not round-trip
  exactly is a stamp the join cannot find. `_times_iso` wrote seconds; real Landsat
  solar-day stamps carry sub-second components, so a rebuilt axis was a *different* axis
  and every composite shard of S30W065 died with `lst carries a time step the offsets do
  not`. Positional alignment used to hide it, and every synthetic fixture used whole
  seconds. **Any fixture whose offsets round-trip through JSON must use sub-second
  times.** Records written at second precision are still read, via an unambiguous
  truncated match (duplicates in the truncated axis are a miss, never a guess); the values
  never changed, so `ALGORITHM_VERSION` is unchanged.
- **Legacy *plans* needed the same treatment, and the record fix alone did not
  reach them.** `plan.scene_times` written before 2026-08-22 are second-precision;
  `shard_tasks._time_coord` rebuilds the offset axis from them, so a resumed run
  hands the join a truncated axis and fails identically. `load_context` recovers the
  fraction from `items.json` (`upgrade_legacy_scene_times`). Note `items.json` holds
  one entry per **scene** and the axis one per **solar-day group**, so several items
  inside one second is ordinary: the group's stamp is the *earliest* of them, because
  odc-stac sorts each group by `nominal_datetime` and takes `group[0]`. Two entries in
  the *stored axis* truncating to one second is the ambiguity that has no answer, and
  that is a hard error. The plan digest covers scene ids and settings, never the
  stamps, so the upgrade cannot move it.
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
- **Factor 4 was tried for #81 and rejected.** It would have cut the offset pass
  from 613,240 tasks to 155,239, and it fails the pre-registered accuracy bound:
  max |Δ| 0.546 °C against a limit of 0.5. It *passed* at 0.431 in August, so the
  gate must be **re-run, never cited** — the shipped grid moved under ADR-008
  (3600×3601 → 3600×3600) and `scene_offsets` fused its two reductions since.
  That same re-run has the factor-1 reference reproducing the committed cap
  calibration to 0.0701 °C rather than 0.0005, which is why the script now prints
  `NO -- not comparable to the shipped cap`. Harmless against a 15 °C cap, but do
  not silence it. See [findings](docs/findings-offset-subsampling.md).
- **A scene-level cloud filter is not a free cost lever, and `max_cloud_cover=100`
  is not a no-op.** The query is `eo:cloud_cover <`, so the default already drops
  every scene reported at exactly 100% cloud (154 of 2,912 for N40W075); 101 is the
  true no-op. Lowering it thins the monthly climatology, which shifts the offsets of
  the scenes that *stay* by up to 3.0 °C at a 90% threshold and 4.4 °C at 80 — six
  times the bound that disqualified offset factor 4, for a fifth of the saving. The
  keep-set is stable (zero decision flips); it is the correction that moves. Issue
  #34's "redundant" verdict predates de-striping and could not see this. See
  [findings](docs/findings-cloud-cover-filter.md) and #81.
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

### The offset pass runs as bounded work units, not one graph

`offset_graph` holds two medians reducing along **orthogonal axes** — a per-pixel median
over time, a per-scene median over space. No chunking satisfies both, so one graph
serving both materializes the stack: construction alone exceeded 26 GB above 2,000 scenes
and execution held a scene-independent ~21 GB plateau. The month-loop reformulation
(PR #99) removed the shuffle but not the two rechunks, because they are the estimator's
shape rather than its spelling.

`scene_offsets` therefore dispatches on `settings.destripe_bounded_units` (default on) to
`offsets_as_units`: `climatology_by_blocks` shards over space, `offsets_by_scene` shards
over scene, and neither builds a graph spanning the window. See
[ADR-015](docs/adr/015-bounded-work-unit-offsets.md).

Rules worth keeping:

- **Bit-exact, so `offsets.ALGORITHM_VERSION` is not bumped.** `max |Δ| = 0` against the
  graph form on 300 real scenes, at two block sizes, from two median kernels, with
  identical NaN patterns and valid counts. The version invalidates caches when *values*
  change; these do not. `offset_graph` stays as the equivalence oracle and is pinned in
  `tests/unit`, `tests/integration`, and `tests/benchmark`.
- **The I/O block and the compute panel are different parameters.** Reads want few and
  large; the kernel wants a working set in cache. Measured on 2250² at 300 scenes: panel
  256 runs in 65.8 s against 116–122 s at 512/1024/2048, which are flat within 5% of each
  other. A cache cliff, not a trend. Every panel size gives an identical checksum.
- **Both phases align to source chunks, and this is correctness-adjacent, not tuning.**
  `_scene_batches` groups **whole** `TIME_CHUNK` chunks; a batch of 8 against
  `TIME_CHUNK = 10` never aligns, every boundary chunk materializes twice, and the pass
  pays ~25% of an extra full read. `_io_block_edge` refuses to go below the spatial chunk
  edge for the same reason. Measured 2.51 passes before the fix, 2.00 after, and pinned by
  `test_unit_form_reads_the_stack_exactly_twice`.
- **Two passes is the accepted cost.** Sharding in orthogonal axes means the phases cannot
  share a traversal. The test pins exactly two so it cannot quietly become three.
- **Phase A memory climbs to a bound; that is not a leak.** `ref` is
  `12 × 9000² × 4 B = 3.89 GB` of lazily-allocated pages touched block by block, so RSS
  rises ~12 MB per block and converges. Phase B is scene-count *independent*: the batch is
  one `TIME_CHUNK` at any window depth.
- **Progress is units, not a task fraction.** `blocks_done/blocks_total` and
  `scenes_done/scenes_total` reach the heartbeat; there is no `GraphProgress` here because
  there is no single graph. A block index localizes a stall where a fraction cannot.
- **Graph construction is not a cost.** 617 slices against a 379,728-chunk array cost 13.5 s
  total, 0.024% of a tile. Do not optimize it.
- **`notnull` and `isfinite` diverge on ±inf.** The graph form counts `n_valid` with the
  first, the unit form with the second. `convert_to_celsius`'s clamp makes this unreachable
  today. It is latent, not fixed.
- **There is no intra-tile checkpoint** *inside one process*. Sharding is the checkpoint
  that was never built: see below.

---

## One tile across many VMs — S3 barriers, never a Coiled DAG

A tile does not fit in an hour on one VM and never will: `projection.tile_projection`
prices a 2,930-scene tile at ~950 GB read twice plus a 3.8 TB native pass, from *measured*
rates. So a tile is cut into stages, each stage into shards, and the stages are sequenced
by a **local driver polling S3**. See [ADR-016](docs/adr/016-sharded-tile-execution.md).

```bash
landsat-lst shard process --tile N40W075     # drives the whole tile; prints a run id
landsat-lst shard resume <run-id> N40W075    # continues a killed driver, from the bucket
landsat-lst shard composite --run-id <id> --tile N40W075 --index 3   # what a VM runs
```

Rules worth keeping:

- **Two fleets per tile, not five.** Offsets-side shards computed ~6 min each while their
  stages held fleets ~30: boots and queueing dominated. `resolve`, `climatology`, and
  `offsets` are now sub-phases of one fused task (`run_offsets_stage`) — shard 0 resolves
  (only if no plan exists), everyone waits for the plan, reduces its blocks, waits at an
  **in-process** phase-A barrier, then estimates. The work-unit bodies are unchanged and
  reused; only the wrappers moved.
- **The fused fleet's width is fixed before the plan exists**, because shard 0 writes the
  plan. `shards.offsets_fleet_units` decides it and `--units` carries it to the planner.
  Never re-derive it on the VM: a disagreement means the stage waits forever for a partial
  nobody owns.
- **The composite fleet starts from inside the offsets barrier** once phase B is producing
  (`shard_composite_overlap`, default: first partial). Evidence, not a timer. It goes
  through `ensure_started`, so the later composite barrier *adopts* it. A composite shard
  therefore **waits** for the merged offset record rather than refusing — refusing would
  burn the boot the overlap saves.
- **The export is claimed by the composite worker that writes the last band**, not
  submitted as a fleet. The claim key is not a lock and does not need to be: the export is
  idempotent at the canonical COG keys, so a lost race is waste, not corruption. The driver
  submits the old export stage only if the COGs are still absent
  `shard_export_claim_fallback_s` after every band exists (the claiming VM was preempted).
- **A shard is complete when its artifact is listed.** Never an exit code, never a state
  object — the same rule tile completion already follows, one level down. The key is a
  pure function of the shard index (`shards.py` owns the grammar), which is what makes a
  resubmission safe: the driver cannot tell a slow shard from a dead one, so every shard
  checks its own output first and exits if it is there.
- **`coiled.batch_run` has no dependency mechanism.** One array per stage, and the
  ordering between them is the poll loop. Do not reach for a cluster: ADR-010 records
  three runs killed by exactly that in one day.
- **The driver holds no state a crash could lose.** Its shell must stay open while the
  tile runs, because it is the thing sequencing stages; it must never be the thing
  remembering them. `resume_tile` reconstructs the position from one listing.
- **The driver requires `LST_STORAGE_BACKEND=s3`, and refuses rather than overriding.**
  Coiled VMs always write S3 (`_worker_environ`), so a driver on the default local backend
  polls a directory nothing will ever write to. `S30W065` on 2026-08-21: `plan.json` was on
  S3 in 3.5 minutes and the resolve barrier never closed. A barrier that cannot see its
  artifacts fails as a *hang*, the most expensive shape a failure takes.
- **A stage already in flight is adopted, never restarted.** Shards publish nothing until
  they finish, so artifacts cannot tell "still booting" from "not started". The driver
  writes `state/{stage}.submission.{round}.json` **before** it submits; a record younger
  than `shard_barrier_timeout_s` means watch, do not submit. Resuming into a live stage
  used to collide: `Unable to add batch jobs to existing cluster '...-climato'`. Cluster
  names now carry the round (`stage_cluster_name`, run id hashed so truncation cannot eat
  the marker).
- **Nothing submits before the credit quota is checked.** `quota.preflight_credits` is
  state zero, and it refuses rather than guessing. A workspace at its quota gets its
  healthy fleet killed mid-stage (2026-08-22, 400 credits) and its cluster creates
  rejected with an *empty* `ServerError`. The estimate comes from the budget model;
  `--ack-quota` is the escape when no balance can be read.
- **Deadlines are derived, never typed.** `landsat_lst.budgets` computes each stage's
  deadline from bytes over measured rates, per shard, times `shard_budget_safety`.
  `shard_barrier_timeout_s` is now an explicit override defaulting to `None`. The
  *widest* shard sets the budget (a barrier waits for the slowest), and the composite
  budget includes an `offsets_tail` phase because that fleet boots during phase B.
- **Every round gets a deadline computed when that round opens.** A barrier used to
  measure from the first submission, so round 2 opened at T+46min against a deadline that
  expired at T+45 and failed having watched for nothing. Pinned by a test asserting two
  rounds cost two budgets of wall clock.
- **Control-plane errors are classified.** Terminal (quota/credits/billing/auth) fails the
  tile now with the reason surfaced; everything else — **including an error with no
  message** — is transient and retried with backoff. An empty `ServerError` killed the
  driver once; guessing "terminal" for the unknown case would bring that back.
- **A cluster reported dead ends its barrier early.** The probe can only end a barrier
  *sooner*, never declare success, and a dead report is re-checked against the bucket
  first (a fleet whose last task uploaded and then stopped is a finished stage).
- **The driver takes an injectable `Clock`.** `tests/unit/test_driver_state_machine.py`
  runs 45 scenarios in under a second. Both defects above are time arithmetic; time
  arithmetic that cannot be tested is time arithmetic nobody checks.
- **Failure is bounded.** On barrier expiry the driver resubmits *only the missing
  indexes*, at most `shard_barrier_rounds` submissions per stage **counted across
  drivers**, then fails naming the keys. Per-driver counting would hand every resume a
  fresh budget. A fleet that resent the whole stage would also finish, which is why the
  test asserts on which indexes the second call carried.
- **Row bands only, never column bands.** `odc-stac` derives its `solar_day` shift from
  the geobox centroid longitude, so two column bands can group the same items onto
  different time axes and the tile-wide offsets would stop lining up.
- **The merge runs in the driver and writes the ordinary ADR-012 record** at the canonical
  `_offsets/` key. That is the seam: every band reads the same estimate back, and because
  only the estimate is cached, rejection is applied tile-wide and identically.
- **Shard objects live under `_shards/`, never `_runs/`.** `runs.classify` reads every key
  under the run prefix as a tile attempt, and seven shards share one tile name.
  `TileHeartbeat(key=...)` mirrors `capture_task_log(key=...)` for this.
- **`shard_composite_chunk` (1024) is applied by every shard process *and by the
  planner*.** The plan digest covers `load_chunk_size`; setting it only in the composite
  shard would make that shard refuse a plan its own planner had cut.
- **Fleet widths come from `projection.tile_projection` when the `shard_*_vms` settings
  are 0**, clamped to the work available. A shard with no block is a VM that boots to bill
  a minute.

---

## Testing

- Unit tests: `uv run pytest tests/unit/`
- Integration tests: `uv run pytest tests/integration/`
- Full tile test (slow): `uv run pytest -m tile`
