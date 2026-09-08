# Runbook: one tile as futures, and how to see what it is doing

For issue #155 and ADR-021. The Batch path (`landsat-lst shard process` with
no `--executor`) is unchanged; everything here is behind `--executor futures`.

## Before a paid run

State the workload, the limits, and the cleanup, then read them back from the
launch output before the cluster boots. The launch prints them and stops for
nothing, so read the estimate line and the limits line before you accept the
spend.

```bash
aws sso login --profile <profile>        # the identity gate refuses an expired session
uv run python scripts/inner_visibility_demo.py   # Demonstration 1: local, free, must pass
```

## Launch

A bounded run, one band and no export, is the shape of Demonstration 2. It
runs from a retained plan, cloned under a new run id: a shard that finds its
slabs exits, so the retained run's own band 16 would skip. The clone's
composite prefix is empty, and the merged offsets record is keyed by the
plan's scene set, so it is found in the cache (`_offsets/S30W065/2021-2025/f2/`)
and the driver submits no offsets stage and no merge.

```bash
landsat-lst shard clone-plan shard-S30W065-2021-2025-20260903T220000Z-v1exact1024-r5 \
  shard-S30W065-2021-2025-20260908T140000Z-futures-demo2 -t S30W065

landsat-lst shard process -t S30W065 \
  --run-id shard-S30W065-2021-2025-20260908T140000Z-futures-demo2 \
  --executor futures --scheduler frisky --inner-scheduler frisky \
  --n-workers 1 --bands 16 --no-finalize \
  --credit-cap 15
```

`--inner-scheduler frisky` is the default and is passed to every shard through
the outer binding; the worker environment's `LST_INNER_SCHEDULER` stays
`threads` and only governs a shard with no binding (the Batch path). The
launch line prints `outer: frisky  inner: frisky`; the shard's heartbeat
`inner.scheduler` and its `inner.final.json` say `frisky` after the fact.

A full tile:

```bash
landsat-lst shard process -t S30W065 \
  --executor futures --n-workers 16 --credit-cap 250
```

What the command does, in order:

1. Gates: identity (STS), write access (a probe object under `_preflight/`),
   the credit estimate against `--credit-cap`, then the workspace balance.
2. Prints the limits: workers, VM type and vCPUs, the run deadline, and the
   product of the three as the maximum vCPU-hours this run can bill. The worker
   count never exceeds `LST_FUTURES_MAX_WORKERS` (16) on any stage.
3. Boots one Coiled cluster, waits for the workers under the deadline, and
   hijacks it with Frisky (`--scheduler dask` keeps plain distributed for a
   comparison run).
4. Submits the futures: plan, offsets per unit, merge, composite per band,
   export when finalizing. The scheduler owns the edges and the retries.
5. Polls the scheduler every 30 s into the bucket (below) and dumps Frisky's
   spans so the evidence outlives the cluster.
6. On any exit: shuts the cluster down, confirms it stopped through the
   control plane, writes `state/cleanup.json`, and prints the verdict.

The shell must stay open. If it dies, the futures it held are released;
running shards finish and publish; resume with the printed line.

## Watch, during the run

Everything is in the bucket. With `AWS_PROFILE` set:

```bash
# every shard's state, worker, error; every worker's memory and CPU
aws s3 cp s3://<bucket>/<prefix>/_shards/<run-id>/S30W065/state/orchestration.json - | jq .counts

# the offsets stall report and transitions, one chunk per poll that changed
aws s3 ls s3://<bucket>/<prefix>/_shards/<run-id>/S30W065/state/ | grep orchestration.events

# one shard's live inner view: open section, tasks started, active by class, host rates
aws s3 cp s3://<bucket>/<prefix>/_shards/timings/<run-id>/composite.S30W065.0016.inner-progress.json - | jq .

# one shard's heartbeat, with the inner block the trace attaches
aws s3 cp s3://<bucket>/<prefix>/_shards/<run-id>/S30W065/state/composite.0016.1.json - | jq .inner
```

Reading the inner progress object:

- `section.name == "group"` with `tasks_started_this_group == 0` and
  `compute_entered` true: the worker is building a graph and has dispatched
  nothing. `first_dispatched` turns true when the first inner task starts.
- `active_by_class.read > 0` with a low `host_rates.net_recv_mb_s`: inner
  tasks are executing and the reads are slow. The read ceiling is about
  10 MB/s per VM (docs/findings-composite-exec-trace.md).
- `spill_count > 0`: the inner worker spilled, which the threaded scheduler
  never did. That fails the bounded demonstration.

The Frisky dashboard URL is printed at launch and works while the cluster is
up. `frisky observe overview <url>` and `frisky observe task <url> <key>`
read the outer futures live.

## Explain, during or after

```bash
landsat-lst shard explain <run-id> S30W065 --index 16                      # every group
landsat-lst shard explain <run-id> S30W065 --index 16 --group 3 --task nanquantile
```

For one inner task: its class and prefix, when and where it ran, GIL and
deserialize time, its queue wait on the worker, the delay before it began
decomposed from `dask.compute` entry, its dependency tree with each node's
execution time, and its dependents. A key that did not execute shows the
scheduler's reason or `unknown`, never an inferred "ready".

Offline Frisky, after the cluster is gone:

```bash
aws s3 cp s3://<bucket>/<prefix>/_shards/timings/<run-id>/frisky-overview.json - | jq .
# or rebuild from the chunks
aws s3 cp --recursive s3://<bucket>/<prefix>/_shards/timings/<run-id>/ ./spans --exclude '*' --include 'frisky-spans.*.json'
jq -s '{spans: map(.spans) | add}' spans/frisky-spans.*.json > all.json
frisky observe overview all.json
```

## Verdict

`futures.summary.json` under `_shards/<run-id>/S30W065/state/` carries
`status`:

- `accepted`: the tile (or the requested bands) completed and the
  observability gate passed.
- `completed_unobserved`: pixels landed, visibility did not. The command exits
  non-zero and names the failures; the outputs stay published. Fix the
  visibility before the next paid run.
- `incomplete` or `failed`: `missing` names what is not in the bucket per
  stage; `errors` names each failed future with its classification; `stopped_by`
  names the deadline, the credit stop, or the terminal error.

`frisky-verification.json` under `_shards/timings/<run-id>/` lists each check
with pass, fail, or not applicable and the observed values. `cleanup.json`
under `state/` says whether the cluster's stop was confirmed.

## Resume and stop

```bash
landsat-lst shard resume <run-id> S30W065 --executor futures --credit-cap 15
landsat-lst shard stop <run-id> S30W065
```

Resume opens a fresh cluster and resubmits every stage. Published shards
return in seconds with a new attempt number and count as skipped. Stop shuts
the run's cluster down by its derived name and confirms it.

## Limits, stated once

- The worker cap and the deadline are guaranteed. The credit cap is a
  refusal at the estimate and a best-effort stop from a balance poll that
  lags billing.
- The offsets stage barriers in-process, so its width must fit under the
  worker cap and its shards must all run at once. A stalled index is reported
  in `orchestration.json`; the scheduler cannot see that barrier.
- Under the Frisky inner scheduler the exec-trace's per-read sampler sees
  nothing; reads are task durations. Host rates are per VM.
- Frisky spans live in a ring buffer per process; the chunks in the bucket
  are the durable copy, and the verification file reports saturation.
- The inner graph on the shared scheduler is blocked by a Frisky defect
  (`tests/integration/test_frisky_nested_client.py`).
