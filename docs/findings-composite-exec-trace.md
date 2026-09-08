# Composite execution trace: S30W065 band 16

Issue #139. Two production traces of the same shard: r2 at 16 dask threads
(cluster 2001031, sampler cadence gate failed, host series descriptive only) and
r3 at 32 threads (cluster 2001315, every gate passed). Both replay the retained
production plan and item list for band 16, rows 8704:9216, on one `m6i.4xlarge`.

## Result first

The composite shard is bound by a fixed aggregate read rate of about 10 MB/s
that does not respond to thread count. Doubling the dask thread pool from 16 to
32 doubled the number of reads in flight and doubled the duration of each read,
and left network throughput, CPU, and wall clock unchanged.

| | r2, 16 threads | r3, 32 threads |
|---|---:|---:|
| `exporting` phase | 1,553.6 s | 1,504.7 s |
| active read tasks, mean | 15.0 | 30.5 |
| network receive, mean | 9.50 MB/s | 9.60 MB/s |
| network receive, integral | 14,499 MB | 14,449 MB |
| CPU busy cores, mean | 1.58 | 1.41 |
| `rio_read` calls | 48,358 | 48,358 |
| sampled `rio_read` p50 | 0.379 s | 0.764 s |
| sampled `rio_read` p95 | 0.859 s | 1.436 s |
| peak RSS | 38,881 MB | 42,652 MB |
| Coiled credits | 7.3734 | 7.2134 |

The read phase, which is the first 1,440 s of both runs, looks like this in the
r3 timeline (means per 180 s window, RSS is the window maximum):

| elapsed s | active read | rechunk | compute | store | CPU cores | RX MB/s | peak RSS MB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–180 | 31.3 | 0.0 | 0.3 | 0.0 | 0.97 | 8.90 | 5,946 |
| 180–360 | 31.7 | 0.1 | 0.2 | 0.0 | 1.14 | 10.48 | 9,834 |
| 360–540 | 31.9 | 0.1 | 0.1 | 0.0 | 1.15 | 10.52 | 14,485 |
| 540–720 | 30.6 | 0.3 | 1.0 | 0.2 | 1.38 | 9.93 | 25,706 |
| 720–900 | 31.6 | 0.1 | 0.4 | 0.0 | 1.16 | 10.63 | 20,102 |
| 900–1,080 | 31.2 | 0.1 | 0.6 | 0.0 | 1.23 | 11.02 | 27,697 |
| 1,080–1,260 | 31.0 | 0.1 | 0.8 | 0.1 | 1.29 | 8.87 | 34,960 |
| 1,260–1,440 | 31.6 | 0.1 | 0.4 | 0.0 | 1.16 | 8.64 | 29,006 |
| 1,440–1,504 | 11.9 | 2.9 | 13.1 | 1.4 | 6.31 | 3.76 | 42,652 |

Read-class tasks account for 45,902 of 47,925 task-seconds in r3. Compute is
1,526 task-seconds, rechunk 342, store 155.

## Why does the shard read at about 10 to 11 MB/s?

**Answered at the level the trace can see.** Throughput is not limited by read
concurrency and not by CPU. Every worker thread was inside a read for 93% of
the interval in both runs, CPU held near one busy core, and the receive rate
held near 10 MB/s whether 15 or 30 reads were in flight. Per-read latency
scaled with concurrency: p50 0.38 s at 16 threads, 0.76 s at 32. The reads
queue behind a shared resource that delivers about 10 MB/s and about 32 to 34
completed reads per second regardless of how many are waiting.

The trace cannot name that resource. Candidates it does not measure are the
object store's per-connection service rate, GDAL's curl connection handling,
and a serialized section in the Python or GDAL read path. The one measured
hint is that CPU sat at 1.15 cores through the read phase in both runs, which
is what a single serialized thread of work would show. That is consistent with
serialization and is not proof of it.

## Why does exporting scale with the number of distinct items?

**Answered as a count, not as a mechanism.** Band 16 touches 2,010 distinct
items and issues 48,358 `rio_read` calls, 24.1 per item, plus 4,022 `open`
tasks, 2 per item. At the measured 32 reads per second that is 1,510 s, which
is the whole exporting phase. The per-item cost is the number of reads per
item multiplied by the fixed aggregate read rate.

The open is not the expensive part. With the split hook, each sampled read
spends p50 0.078 s in `rasterio.open` and setup and p50 0.647 s in the read or
warp itself, 282 of 2,004 sampled seconds against 1,701. A distinct item costs
its reads, not its opens.

## Why does RSS grow through exporting?

**Answered as timing, not ownership.** RSS climbs roughly linearly while reads
are active and compute is near zero: 5.9 GB at 180 s, 14.5 GB at 540 s, 27.7 GB
at 1,080 s. In that regime completed read tasks are retiring and no reduction
is running, so what accumulates is loaded source data waiting for the
per-pixel quantile, which needs every scene of a spatial chunk before it can
start. Two mid-phase steps of 2 to 3 GB (646 s and 1,079 s) coincide with short
compute waves where reads drop to zero and 26 compute tasks run.

The peak comes at the end. At 1,472 s RSS reached 42.65 GB with zero reads
active, 25 compute tasks, 4 store tasks, and 9 to 13 CPU cores busy: the final
reduction and store wave over all held chunks. The 32-thread run peaked 3.8 GB
higher than the 16-thread run at the same point. The trace shows the classes
active at every step and does not show which objects hold the memory.

## Method, deviations, and cost

The r3 observation ran revision `2dc820c93593be506b46aa82177caeac914a8b8e` with
`LST_EXEC_TRACE=1`, `LST_DASK_MAX_THREADS=32`, `LST_GED_GAP_MASK=false` (the
retained workload predates the mask), and `LST_COILED_RETRIES=0`. The host
sampler ran as a spawned child process after the r2 in-thread sampler held a
mean cadence of 1.61 s and 120 gaps above 2 s while waiting for the GIL; r3
held 1,505 samples at a mean of 1.000 s and a maximum gap of 1.003 s. The read
hook sampled every 20th call, 2,417 of 48,358, and recorded the open/read
split for all of them. The first upload began 0.480 s after compute returned.

The local gate ran on Planetary Computer with 40 scenes over a 512 x 2,048
strip and validated recorder mechanics only.

Coiled billed 7.2134 credits for cluster 2001315, on-demand despite the spot
policy, 27 min 52 s of lifetime. Cumulative issue cost across the failed first
observation (0.8934), r2 (7.3734), and r3 is 15.4802 credits, under the
20-credit cap; the derived EC2 list-price total is $0.75 and is not an invoice.

No retry, optimization, or follow-up experiment was performed.

## Reading the inner-trace artifacts (added 2026-09-08, issue #155)

The exec trace above counts active tasks by class; it cannot name a task or
say what it waited for. The inner trace (`landsat_lst.innertrace`, ADR-021)
writes, beside the three exec-trace artifacts and under the same stem
`_shards/timings/{run_id}/{stage}.{tile}.{index:04d}`:

| Key suffix | Written | Holds |
|---|---|---|
| `.inner-graph.a{attempt:02d}.g{group:02d}.json.gz` | before the group's first dispatch | the dict the scheduler received: `nodes[{key, prefix, class, deps, n_dependents}]`, `roots`, `outputs`, `capture_s`, `materialize_s` |
| `.inner-exec.a{attempt:02d}.g{group:02d}.json.gz` | after the group's compute | per key `start_ns`, `end_ns`, `worker`/`thread`, `gil_ns`, `deserialize_ns`, `received_ns`, `state`, `not_executed_reason`; the group's `construction` record |
| `.inner-spans.a{attempt:02d}.{seq:04d}.json.gz` | every flush and every section close | raw Frisky spans since the previous chunk |
| `.inner-progress.json` | every flush, overwritten | the open section, `tasks_started`, `active_by_class`, `host_rates` (parent, worker child, tree) |
| `.inner.a{attempt:02d}.final.json` | on exit, either way | every section with its duration, group construction records, chunk and file keys, `closed_by`, `error` |

Rules for a reader:

- Join exec rows to graph nodes by `key`; the spellings match because both use
  Frisky's `str(key)`.
- `construction_s` is `first_dispatch_ns - compute_entry_ns`. Its parts are
  `pre_scheduler_s` (dask's optimize and materialize, invisible to Frisky),
  `materialize_s`, `capture_overhead_s`, `translate_s`, `submit_s`, and
  `submitted_to_first_dispatch_s`, which can be negative because Frisky starts
  dispatching before submission finishes. Sum the parts only up to
  `graph_captured_ns`; after that they overlap.
- Union intervals, never sum them, when attributing wall time to a class: the
  same second can hold fifteen reads and one reduction.
- A row with `start_ns: null` did not execute in this group as far as the
  trace saw. Read `not_executed_reason` and `state`; `unknown` means no
  scheduler evidence was fetched (the story limit was reached), not that the
  key was ready.
- `landsat-lst shard explain --index N --group G --task KEY` renders one key's
  dependency tree, execution times, and the decomposed delay before it started
  from these files alone.
