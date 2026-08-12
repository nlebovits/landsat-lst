# ADR-010: Run distributed tiles on Coiled Batch, not Coiled Functions

**Status:** Accepted
**Date:** 2026-08-12
**Authors:** @nlebovits

## Context

The distributed path submitted each tile as a task to a Coiled Functions cluster. A task ran
`process_tile_job`, which builds a dask graph over a few hundred Landsat scenes and computes it.
So a multi-hour dask computation ran inside another dask cluster's worker, and the driver held a
client connection to that cluster for the length of the run.

Three validation runs on 2026-08-12 failed, each one layer deeper than the last.

**Run `2021-2025-20260812T142408Z`** (1 tile, r6i.xlarge). The tile graph escaped to the cluster's
shared scheduler, because inside a Coiled task the cluster's own client is ambient and an
unqualified `compute()` submits there. The single worker died at 28.77 GiB, `killed by signal 9`.

**Run `2021-2025-20260812T150618Z`** (3 tiles, r6i.2xlarge). Three tile graphs arrived at the
shared scheduler at once and crushed it. All three tiles failed inside five minutes with
`scheduler-connection-lost`.

**Run `2021-2025-20260812T152052Z`** (3 tiles, after [#65](https://github.com/nlebovits/landsat-lst/pull/65)
pinned `dask.config.set(scheduler="threads")`). The graph now computed locally on the worker, which
is correct, and the long GIL-holding compute then starved the worker's heartbeat loop. Coiled
declared the worker dead at the 5-minute TTL and tore it down mid-tile
(`cannot schedule new futures after shutdown`). The client lost the cluster with
`FatalCommClosedError` and Coiled marked the cluster `error`.

Each fix exposed the next layer of the same mismatch. The remaining moves were to disable Coiled's
health systems one at a time (worker TTL, heartbeats, memory monitor) so Functions would tolerate a
workload it was not shaped for. The client side was no better: a laptop had to hold a connection
for the whole run, and a dropped connection killed the driver before it could write the manifest.

The workload is 700 independent, idempotent, S3-writing jobs. None of them talk to each other.
There is nothing for a scheduler to coordinate.

## Decision

**Submit tiles to Coiled Batch, one plain process per VM, and reconcile the run afterwards from S3.**

### Execution

One task per tile, one VM per task, capped at `settings.coiled_max_workers` concurrent VMs. The
task command is a literal shell line:

```
bash -c 'python -m landsat_lst.cli process --run-id RUN --year 2021 --end-year 2025 --tile "$COILED_BATCH_TASK_INPUT"'
```

Coiled sets `COILED_BATCH_TASK_INPUT` per task from the tile list. Jobs are grouped by window
before submission so the window is a literal and only the tile varies. There is no shared
scheduler, no heartbeat competing with the compute, and no client connection to lose.

`process_tile_job` is unchanged and remains the unit of work whether the machine is a laptop or a
batch VM. #65's threaded-scheduler pin stays correct for any in-worker compute, and on a batch VM
the threaded scheduler is simply dask's default.

### Two phases, two processes

`submit_batch` filters completed tiles with one storage listing per window, submits the task array,
writes a submission record to `settings.manifest_dir`, and returns. Killing the shell after that
point does nothing to the run.

`reconcile_run` reads the submission record back and builds the manifest. It runs at any time,
including days later, and produces the same verdict every time.

### Completion is measured in bytes, not exit codes

A tile counts as complete when both COGs are listed in S3. A task can exit non-zero after its
assets landed, and could exit zero having produced nothing. The bucket is the only claim worth
trusting.

Exit codes and task states still earn their place: they explain a tile that has no output, which
is exactly the case the manifest most needs to describe.

### Per-tile run records

A batch run has no live driver holding results, so each VM reports for itself. `process_tile_job`
writes `_runs/{run_id}/{tile}.json` with duration, scene count, peak RSS, and any error. That
prefix is a sibling of the collection directories and invisible to the catalog, which only reads
`lst-p95-*`.

Reconciliation layers three sources, most trustworthy first: the COG listing decides status, the
run record supplies the costing metrics, and Coiled's task state explains a tile that has neither.
A missing record is ordinary — a preempted or timed-out VM never wrote one — and Coiled being
unreachable degrades reconciliation rather than breaking it.

### Instance size

Default VM types move to 64 GiB (`r6i.2xlarge`, `m6i.4xlarge`). The first failure was a genuine
out-of-memory at 28.77 GiB on a 32 GiB machine, and no scheduling change makes a heavy tile fit.

## Consequences

The manifest is no longer written while the run is going. Closing a laptop now costs a manifest
that has not been generated yet, rather than one that was lost, and `landsat-lst reconcile RUN_ID`
generates it whenever convenient. `--wait` still does both in one command for a supervised run.

Cost control shifts from `n_workers` to `max_workers`, which caps concurrent VMs rather than
sizing a fixed cluster, plus `job_timeout` as a per-task wall-clock ceiling. A stuck tile stops
billing on its own.

`run_distributed` is gone, along with the Coiled Functions path. The public surface is
`submit_batch`, `reconcile_run`, and `wait_for_batch` in `landsat_lst.batch`.

Live progress moves entirely to the Coiled dashboard and `coiled batch status`. There is no
per-tile progress bar on the submitting machine, because there is no longer a process there to
draw one.

Per-task logs live in Coiled rather than in the driver's output, so the manifest records
`cluster_id` and `job_id` to make `coiled batch logs <cluster-id>` reachable after the fact.

Refs [#66](https://github.com/nlebovits/landsat-lst/issues/66), [#31](https://github.com/nlebovits/landsat-lst/issues/31).
