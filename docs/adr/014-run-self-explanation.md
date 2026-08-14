# ADR-014: One state object per attempt, and a priced run

**Status:** Accepted
**Date:** 2026-08-14
**Authors:** @nlebovits

## Context

A tile runs for hours on a VM nobody can reach. Everything knowable about it is
what it publishes to `_runs/{run_id}/`. ADR-010 established that channel and
issue #68 filled it out. What neither did was join, trend, price, or retain any
of it, so on 2026-08-14 two runs of N40W075 each failed to explain themselves in
a different way.

### Run `2021-2025-20260814T092642Z`, failed at second 10

Earth Search returned HTTP 500 to the STAC query and the tile exited immediately,
out of a five-hour budget.

`pystac_client` 0.9.0 was already mounting a retry adapter. It built one from a
plain integer, and `requests` turns that into `Retry.from_int(5)`, whose defaults
leave `status_forcelist` empty, the backoff at zero, and `allowed_methods`
restricted to idempotent verbs. So a 500 was never retried, a STAC search is a
POST and was not retried either, and nothing waited between attempts. The number
was right and every other term was wrong.

The manifest then reported `duration_s` 10.375 against a run spanning 09:26:42Z
to 09:59:35Z, a 33-minute wall clock. `settings.coiled_retries` is 3, and all
three attempts wrote the same three keys. An earlier attempt had reached
`land_mask`, further than that tile had ever gone, and its successor erased it.

The uploaded log hit exactly `task_log_max_bytes` and reported `3829375 earlier
bytes dropped`. The source was our own `logger.exception("tile_failed", ...)`:
nothing in this repository configured structlog, so it fell back to
`ConsoleRenderer`, whose default exception formatter is
`RichTracebackFormatter(show_locals=True, max_frames=100)` whenever rich is
importable. One traceback rendered the deserialized Landsat collection out of a
frame local and evicted the run's phase history.

Diagnosing all of this took four manual steps: read the manifest, list the run
prefix, download the log, slice its head and tail. Every input was already
published.

### Run `2021-2025-20260814T102049Z`, running

| Elapsed | Phase | Graph | RSS |
|---|---|---|---|
| 3m00s | destriping | starting | 6.0 GB |
| 11m01s | destriping | 1% | 31.7 GB |
| 14m01s | destriping | 1% | 35.1 GB |

Everything an operator needs is in that table and nothing in `watch` computed
it. The graph percentage was integer-rounded, so 1.0% and 1.9% rendered
identically and three minutes at `1%` could not be told from a wedged graph. RSS
was a point sample, so the climb from 6.0 GB was invisible. Task rate, ETA, and
cost were absent.

## Decision

### One state object per tile per attempt

`{tile}.json` and `{tile}.progress.json` were the same object described twice,
differing in about 80% of their fields. They merge into one object at
`_runs/{run_id}/{tile}.{attempt}.json`, rewritten every beat and once more when
the tile settles.

`status` stays `None` until a tile settles. `phase` carries liveness and
`TERMINAL_PHASES` decides what settled means, so a mid-run reconcile has no
verdict to misread and `watch` has no second liveness signal to disagree with.

A copy of the final state goes to the unsuffixed `{tile}.json`. Its body is a
superset of the old run record at the old key, so an external reader keeps
working with no attempt logic, and its presence is how every reader tells a
settled tile from a running one. That holds for a run written before this ADR
too, because the old record was also written exactly once, at the end.

**The pointer is written at the terminal boundary, never from the beat loop.**
Per tile per run the count is unchanged: roughly 45 state writes plus one
pointer, against 45 heartbeats plus one record before. The `about $0.42` in
`config.py` still holds. A pointer refreshed every minute would have doubled it
to buy a key that is already published under its own name.

An attempt that ends with an exception escaping writes its own object and **not**
the pointer. A transient failure re-raises so Coiled retries it, and no reader
should see a failed final answer while a retry is in flight.

### `runs.py` owns the key grammar

`watch`, `reconcile`, and now `explain` all read this layout, and each used to
re-derive it with its own suffix tests. One of them was wrong:
`{tile}.{label}.profile.json` also ends in `.json`, so a profiled tile appeared
in `watch` as a phantom tile named `N40W075.destripe_offsets`, rendered as a
finished row, and was subtracted from the pending count. `runs.classify` tests
`.profile.json` before `.json` and fixes it in every reader at once.

The attempt number is discovered by listing the tile's own artifact prefix and
taking one more than the highest. Coiled exposes no retry counter —
`COILED_ARRAY_TASK_ID` is the array index and is identical on every retry — so
the bucket is the only record. **It is resolved once per process and threaded
down.** Asking twice would give the log a higher number than the state object,
because the log uploads last and would see the state object already written.

The count spans state objects, logs, and profiles rather than state alone. A VM
preempted before it published any state still leaves a log, and that log is the
only evidence the attempt happened.

`LocalStorage.list_prefix` had to be fixed first. It treated its argument as a
directory while `S3Storage` did true key-prefix matching, so a tile-scoped
listing returned nothing locally and the right answer in production. Since
`LocalStorage(tmp_path)` is this repository's only fake backend, every test of
attempt discovery would have passed against a function that finds nothing.

### Cost is a range, from a committed table

`pricing.json` follows `calibration.json`: committed data, an `as_of` date, and
a provenance label beside every figure. `Provenance` gains `PUBLISHED`, because
a vendor list price is not measured, not derived, and calling it assumed
understates it.

**Spot is a band, not a factor.** Three quotes sampled on one day, in one region
and one instance generation, spanned 0.35x to 0.71x of on-demand. A single
discount would be wrong by more than 2x for one of the two configured VM types.
An unreported lifecycle spans 0.30 to 1.00 of on-demand, because
`spot_with_fallback` guarantees the tile ran on one of the two and nothing
published says which.

That is also why the instance type is read from EC2 IMDS on the VM rather than
assumed from `settings.coiled_vm_types[0]`. The fallback type prices at 1.52x
the primary for the same 64 GiB, and measuring the lifecycle narrows a 3.3x
interval to a band or a point. The probe is bounded at 0.3s, cached for the
process, and falls back to the configured assumption labelled as one.

EC2 bills per second with a 60-second minimum, so the 10.375-second failure
above still cost a full minute. The estimate omits Coiled's fee, S3 requests,
EBS, transfer, and the boot and teardown minutes Coiled bills that a tile's own
clock never sees. It says so wherever it prints.

### The planner is checked against the run

`plan_memory_record` computes the memory floor from `tile_geobox` and
`predict_peak` with no graph construction, measured at 1.24 ms, so `submit_batch`
stores it for free. `reconcile` reports the **memory ratio** rather than a
verdict: the floor is a floor, and the 300-scene sample peaking at 78.6 GB
against a floor of a few GB was itself the useful signal. Scene count is the
honest cross-check, because `plan --scenes` is an assumption and the run reports
what STAC returned.

Task counts are deliberately not stored. Counting them means building the fused
graph, which costs minutes per tile, and the diff needs only the floor and the
scene count.

### `watch` keeps its history, and `explain` replaces the bucket dig

Polled heartbeats are retained in memory for the session. No new writes, no new
storage cost. There is no history to backfill from storage — each tile has one
object, overwritten in place — so the UI marks a late attach rather than drawing
a curve that begins mid-story. Phase history needs no retention at all, because
`phase_seconds` is already cumulative in every beat.

A rate never survives a graph boundary. `set_phase` clears the counts and a tile
runs several graphs, so `graph_seq` is published and increments on each
idle-to-running edge, making the epoch exact instead of inferred. The displayed
rate comes from the newest pair so a decay toward zero is visible immediately;
the ETA divides by a five-sample window so one slow poll does not throw the
finish time. `-` means not measurable and `0/s` means measured zero.

`landsat-lst explain <run-id> [tile]` prints each attempt's state, phase
timings, dask profile, and log tail from the run prefix alone.

### The Coiled dashboard is demoted

A batch task never registers with the dask scheduler, so that page describes a
scheduler the run never joined, and `coiled logs` never receives task stdout.
Submit output now promotes `landsat-lst watch`. The cluster id stays, because it
is the right handle for billing and for Coiled support.

## Consequences

- **A retry costs a key, not the evidence.** A tile that succeeded on attempt 3
  after two infrastructure failures reads differently from one that succeeded on
  attempt 1, and the attempt that reached `land_mask` would have survived.
- **An HTTP blip costs seconds, not a VM.** Retrying in the STAC client rather
  than through `coiled_retries` keeps the tile's progress. A real outage still
  exhausts the budget in about 30 seconds rather than holding a VM for
  `coiled_job_timeout`.
- **Tracebacks stay legible.** `task_log_max_bytes` is unchanged; raising it
  would have preserved more of one traceback instead of the phase history that
  makes a run readable.
- **Bumping `progress.SCHEMA_VERSION`** is required when the published object's
  shape changes. A body with no `schema` key is version 1, the split pair.
- **A run in flight during a deploy writes both shapes.** Coiled installs the
  package at task start, so a retry scheduled after this merges writes
  `{tile}.2.json` beside an old `{tile}.progress.json`. Readers handle the
  mixture; an operator on an unpulled checkout would see `N40W075.2` as a
  phantom tile. No data is lost either way, because old artifacts are never
  deleted.
- **`pricing.json` goes stale.** It carries its date and the CLI prints it. One
  edit to the file moves every command that quotes it.

## References

- Issue [#92](https://github.com/nlebovits/landsat-lst/issues/92), superseding
  #84 through #91.
- [ADR-010](010-coiled-batch-for-distributed-runs.md) for why a batch task is
  invisible to the dashboard.
- [ADR-011](011-static-planning-and-synthetic-benchmarks.md) for the planner
  this now checks.
