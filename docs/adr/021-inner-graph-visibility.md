# ADR-021: The inner graph of a shard is observable, on an in-process scheduler

Status: accepted for issue #155 Parts A (inner visibility), B (the futures
executor), and C (its limits and outer view). Demonstration 2 on the cloud and
the full-tile acceptance are the remaining gates before the futures executor
becomes the default.

## Context

A composite shard runs a dask graph of tens of thousands of tasks inside one
process on one VM. Until this decision the graph was opaque from outside the
process. The threaded scheduler fires `dask.callbacks` hooks, and
`landsat_lst.exectrace` used them to count active tasks by class, but nothing
answered the questions an operator asks first: which task is running, what did
it depend on, and how long did the main thread spend building the graph before
anything was dispatched. Issue #154 was found with a native stack capture, and
the read-phase mutex (`composite-read-wait-is-single-mutex`) with `strace`.
Weeks went to forensic sessions for answers a scheduler holds.

ADR-010 kept the shard's graph away from any distributed scheduler because a
multi-hour tile graph inside a shared cluster killed three runs in one day. The
Sep 7 probes reopened that: a bounded shard ran as one opaque future on a Dask
worker (1,366.6 s, 19.32 GB, stable), while the attempt to submit the inner
graph to the shared scheduler failed before scheduling with
`TypeError: Could not serialize object of type _HLGExprSequence`.

## What was measured before deciding

All on frisky 0.7.2 and dask 2026.7.1, 2026-09-08, locally.

1. **The serialization failure is a `threading.Lock`.** `write_intermediates_bounded`
   handed `dask_array.store` a plain lock; the store expression for the two
   products then fails `pickle`, and distributed reports only the wrapper type.
   With `SerializableLock` the same expression pickles. The odc-stac load graph
   and the composite graph pickle. That blocker is closed.
2. **A Frisky client nested inside a Frisky worker task kills the scheduler.**
   The inner tasks execute across both workers, then the scheduler closes with
   `scheduler recv ended: io error: early eof`, on unix-socket and TCP
   transport. `frisky.dask.get` also refuses `workers=`, so an inner graph on
   the shared scheduler could not be pinned to the VM that holds the local
   GeoTIFF target. Reproduction: `scripts/experimental/frisky_nested_client_repro.py`,
   kept live by `tests/integration/test_frisky_nested_client.py` under
   `xfail(strict=True)`.
3. **An in-process Frisky scheduler inside the worker task works.** Every inner
   task is a span, with GIL, deserialize, store, and receive spans beside it,
   and `frisky observe` reads the dumped spans offline.
4. **`frisky.get_spans()` drains the process buffer.** In a process that is
   itself a Frisky worker, the worker plugin and an in-process trace take each
   other's spans, and the outer scheduler received nothing. An inner cluster
   whose one worker is a subprocess keeps its spans in its own process, and the
   inner scheduler's REST endpoint serves them without draining, incrementally
   by `start_ns`.
5. **Frisky dispatches while the client is still submitting**, so
   "first dispatch minus submitted" can be negative. The span clock is the
   epoch clock (offset under 1 µs).
6. **Execution spans can land after `gather` returns.** Read too early, forty
   executed keys of one group were filed as not executed and credited to the
   next group.

## Decision

**Every shard's graphs run on a scheduler that reports, and the trace persists
what it reports while the shard runs.** `landsat_lst.innertrace` provides:

- `inner_scheduler(mode)`. `"threads"` is dask's threaded scheduler, the
  production scheduler before this change, and the default
  (`settings.inner_scheduler`). `"frisky"` starts a `frisky.LocalCluster` with
  one subprocess worker on a TCP transport and a random loopback dashboard
  port, and binds dask's `scheduler` to a callable that materializes the
  expression once, captures the resulting dict, and hands **that same dict**
  to Frisky's translate and submit. There is no second optimization pass. The
  capture cost is counted inside the group's construction wall, never
  subtracted.
- `InnerTrace`. Coarse sections (`phase:*`, `composite_graph`, `encode`,
  `header_write`, one `group` per longitude group, `upload`, `finalize`); per
  group an `inner-graph` file (keys, class, dependency edges, roots, outputs)
  written before the first dispatch and an `inner-exec` file (per key: start,
  end, worker or thread, GIL and deserialize time, scheduler state, and, for a
  key that did not execute, a reason drawn only from scheduler evidence:
  `blocked_on_deps`, `released`, `erred`, `ready_not_started`,
  `completed_without_exec_span`, or `unknown`); span chunks every
  `inner_trace_flush_s` and at every section close; an overwritten
  `inner-progress.json`; a final object on exit either way. Sections are
  forwarded to an outer scheduler, when one exists, as `shard.*` spans keyed by
  the outer task key.
- The construction record per group: `compute_entry` (the observer's
  `compute_start` event, the moment `dask.compute` is called), `pre_scheduler`
  (dask's optimize and materialize, which Frisky never sees and which is where
  #154 lives), `materialize`, `capture_overhead`, `translate`, `submit`,
  `first_dispatch`, and the headline `construction_s = first_dispatch -
  compute_entry`. Under threads the same total comes from the Callback
  boundaries.
- `landsat-lst shard explain <run> <tile> --index N [--group G --task KEY]`
  reads the two files and renders a task's dependency tree with each node's
  execution time, the dependents, and the delay before the task began, from
  `dask.compute` entry, decomposed.

**The Batch path is unchanged.** `run_shard` takes the scheduler from the
outer binding the futures wrapper sets, and from `settings.inner_scheduler`
(`threads`) otherwise. Nothing on the Batch command line sets a binding. A unit
test blocks the `frisky` import and runs a composite shard to prove it.

**Mode I replaces the threaded scheduler on the futures path.** That is a real
execution change, so the demonstrations compare Mode I and Mode T on the same
input: pixels identical, no spill spans, peak memory within 10%, and the
visibility artifacts side by side.

**The inner graph on the shared scheduler (Mode S) is blocked upstream**, by
findings 2 and the absence of worker pinning, and stays tracked rather than
built. The lock fix removed the only serialization obstacle.

## Consequences

- An operator can pick one inner task of a running or finished shard and see
  its dependencies, when and where it ran, and every part of the delay before
  it started, from the persisted objects alone. Demonstration 1
  (`scripts/inner_visibility_demo.py`, pinned by
  `tests/integration/test_inner_visibility.py`) does exactly that, locally,
  with no credentials.
- The exec-trace's `rio_read` hook and Callback recorder see nothing under
  Mode I, because reads and tasks run in the worker subprocess. The trace's
  spans replace the task records; per-read timing is an open item for Mode I.
- The shard's memory is the process tree's. `host_rates` reports the parent,
  the worker child, and their sum; the heartbeat's `rss_mb` still describes the
  parent alone.
- Graph capture writes one gzip per group before dispatch. On a production
  band that is a few thousand nodes and about a megabyte; the serialization
  runs on a helper thread and its cost is recorded as `capture_overhead_s`.
- The nested-client crash and the missing worker pinning are Frisky defects to
  report upstream. A release that fixes them turns the strict xfail red, which
  is the cue to reconsider Mode S.

## Alternatives considered

- **Inner graph on the shared scheduler through a nested client.** Kills the
  scheduler (finding 2). Rejected until Frisky changes.
- **In-process inner worker.** Works alone, but shares the drained span buffer
  with an outer worker plugin (finding 4). Rejected for the futures path.
- **Reading construction from Frisky's `client.*` spans.** They live in the
  parent buffer the outer plugin drains. Replaced by local stamps at the
  scheduler `get` seam, which mirror `frisky.dask.get` stage by stage.
- **Byte identity between schedulers.** Two threaded runs already differ in
  tile write order. Pixel identity is the comparison.

## References

- Issue #155, issue #154, ADR-010, ADR-016.
- `docs/findings-composite-exec-trace.md` for the read-rate ceiling the trace
  now shows per task.
- Memory: `frisky-072-facts-for-inner-visibility`.

## Part B and C: the futures executor, its limits, and its outer view

Added 2026-09-08, after Demonstration 1 and before Demonstration 2.

### The scheduler owns the stages

`landsat_lst.futures_driver` submits one future per shard on a cluster session
and lets the scheduler own the edges, the retries, and the task state: a plan
future when no plan exists, an offsets future per unit that depends on it, a
merge future that depends on every offsets future, a composite future per
requested band that depends on the merge, and an export future that depends on
every composite future, submitted only when the run finalizes. Every future
runs `futures_tasks.run_shard_task`, which is the same shard the Batch path
runs, under an outer binding that names the outer key, the trace id, the
Frisky sink, and the inner scheduler.

The driver never imports coiled, frisky, or distributed; a test parses its
source to prove it. It takes an `Executor` and a `RunObserver`, both
duck-typed, and the whole state machine runs against `tests/unit/futures_fixtures.py`
in milliseconds: dependency wiring, scheduler-owned retries, a shard that
published and then died (its dependents are resubmitted with the dead edge
removed, because the artifact is in the bucket), a terminal error that releases
every pending future, a resume that recomputes nothing, reconciliation per
stage, and the deadline at every blocking point.

Two things the scheduler cannot see are named by the driver. The fused
offsets shards barrier in-process, so their width must not exceed the worker
cap and the driver refuses when it does; an offsets index still pending while
its peers run past `futures_offsets_stall_s` is reported as `stalled`. And a
shard's durable completion is its artifact, so an error is classified only
after the bucket is checked.

### Three limits, in decreasing strength

- **Workers.** `futures_max_workers` (16) binds every stage. Composite bands
  queue behind it in waves. The count is fixed and printed before the cluster
  exists, with the VM type, its vCPUs, and the deadline, as the maximum spend
  the operator authorizes.
- **Deadline.** One `Deadline` created before the cluster, derived from the
  budget model over boot, the offsets stage, and the composite waves the band
  count needs, and checked while waiting for workers, while waiting for the
  plan, and at every completion. Expiry releases every unfinished future and
  shuts the cluster down. `futures_run_timeout_s` overrides it outright.
- **Credits.** `--credit-cap` is required. The estimate times the safety
  factor must fit under it before anything boots; afterwards a balance poll
  stops the run when the drawdown exceeds it. That stop is best-effort and
  labelled so in the code and the summary: Coiled usage lags billing, and
  other jobs move the same balance.

Cleanup is confirmed, never assumed: after `cluster.shutdown()` the control
plane is polled until the cluster reports stopped or `futures_cleanup_timeout_s`
passes, and `state/cleanup.json` records what was seen. `landsat-lst shard stop
<run-id> <tile>` shuts a dead driver's cluster by its derived name.

### The outer view, persisted while the run runs

`landsat_lst.futures_observer` joins what the driver submitted to what the
scheduler reports, every `futures_observer_poll_s`, into
`state/orchestration.json` (overwritten) and `orchestration.events.{seq}.jsonl`
(transitions, worker joins and losses), dumps Frisky's span buffer to
`_shards/timings/{run_id}/frisky-spans.{seq}.json` deduplicated at the cursor,
and writes `frisky-verification.json`: whether the scheduler listed every
submitted key, whether a running shard's story shows a placement, whether the
`shard.*` spans arrive keyed by the outer key, one `worker.exec.call` span per
attempt, the buffer below saturation, and spans persisted after the cluster.
Each check records pass, fail, or not applicable with the observed values.

The observability gate reads the bucket at the end: every composite shard's
inner final object with no error and a sink that built, both trace files per
group, no failed verification check, span chunks persisted. `summary.status`
is `accepted` only when the tile completed and the gate passed. A run with
correct pixels and missing visibility reports `completed_unobserved` and exits
non-zero, with its outputs left published; `futures_require_observability`
downgrades that to a warning for diagnosis runs.

### What stays Batch

`shard_executor` defaults to `batch`. The Batch driver, its barriers, rounds,
adoption, cluster probes, and submission records are untouched. Under futures
those are superseded by the scheduler and are not deleted until the futures
path passes full-tile acceptance.
