# Issue #125 production calibration

One staged offsets stage for `S30W065`, window 2021-2025, 15 shards on
`r6i.2xlarge` spot in `us-west-2`. Run `shard-S30W065-2021-2025-20260904T021035Z-125cal`,
cluster 2003236, job 384818.

## What it ran against

The plan and item set were copied verbatim from
`shard-S30W065-2021-2025-20260903T231831Z-group6-r1`, which is the same tile,
window, and 1,031-scene set as the timing baseline and whose plan digest
`4230fc98933de63e` is what current code computes. The August anchor run's own
plan is refused by the digest guard, because the digest covers
`algorithm_version`, which the exact transform moved to 2.

Values are compared against `_offsets/S30W065/2021-2025/f2/v2-eac1fa94a9fc760f.json`,
the direct-path record for exactly that scene set under the same estimator
version. Timings are compared against the retained anchor run
`shard-S30W065-2021-2025-20260823T102135Z`.

## Byte model, measured

| Quantity | Value |
|---|---|
| Scenes (solar-day groups) | 1,031 |
| Coarse grid | 9,000 x 9,000, factor 2 |
| One source pass | 334.0 GB |
| Staged, whole tile | 167.0 GB |
| Staged per shard | 11.1 GB |
| Staged objects | 8,424 |
| Object size | 20,971,648 B = 21.0 MB |
| Stage prefix | `.../stage/f1-v2-eac1fa94a9fc760f/` |

The object is one phase-A block of 1,024 squared by one `TIME_CHUNK` of 10
scenes as `uint16`, which is the size the throughput probe measured. The prefix
carries the stage format version, the estimator version, and the digest of the
offsets record it feeds.

## Confounds, stated before the numbers

The phase-A comparison is **not** clean and no #125 claim rests on it. The
reused plan carries `block_weights`, so this run deals phase-A blocks by scene
footprint weight (#134, merged 2026-09-03) where the August baseline dealt them
by count. It also runs the exact transform, which costs more CPU per pixel.
Phase A therefore mixes three changes and cannot attribute its result.

The phase-B comparison is clean. #134 changes only how phase-A blocks are
dealt, not how scene batches are split. The exact transform makes the baseline
comparison *conservative*: a direct-path phase B today would warp on every read,
where the August number did not, and the staged path does not warp at all.

## Result: the run was truncated, and the reference is mislabelled

The cluster stopped at 02:33:07 UTC. Six shards finished and published
partials. Eight are in `error` with no exit code and no uploaded log, all
stopped at that same instant, which is a fleet-level teardown rather than eight
independent failures. One more exited 137. Coiled reports the cluster as
stopped. The credit balance reads `remaining=None, has_quota=True`, 664.2
credits spent in 30 days, period reset not observable, which is the documented
signature of a workspace quota kill. The credits preflight was acknowledged
with `LST_ACK_QUOTA=1` because the balance could not be read, so nothing
refused the launch.

Separately, and independently of this work, the only offsets record for this
scene set under the current estimator version is **mislabelled**:

| Path | Body says |
|---|---|
| `_offsets/S30W065/2021-2025/f2/v1-a662c8e0b67b4254.json` | `algorithm_version` 1, digest `a662c8e0b67b4254`, written 2026-08-22 |
| `_offsets/S30W065/2021-2025/f2/v2-eac1fa94a9fc760f.json` | `algorithm_version` 1, digest `a662c8e0b67b4254`, written 2026-08-22 |

The two files are byte-identical, sha256 `601cd5caa9e95c2a...`. The v1 record
was copied to the v2 key. Any consumer reading the v2 key receives an
approximate-transform estimate believing it is an exact-transform one.

## Correctness: not demonstrated in production, and not disproven

Against that mislabelled reference the six partials differ, by at most 0.0222 C
on offsets, with `n_valid` moved. That is the difference `offsets.py` documents
between the approximate and exact transformers, where the coarse stack moves on
6.0% of pixels and `n_valid` moves with it. It is what a correct exact-transform
run should look like beside a v1 record. It is not evidence about staging, in
either direction, because the comparison has no valid reference.

Production bit-identity therefore remains **untested**. It is established
locally, exhaustively, on the same code and the same inputs, in
`local-gates.md` and `tests/unit/test_staging.py`.

## Measured

Six shards completed phase B. Phase A completed on 17 attempts.

| Phase, seconds | Staged min/med/max | Baseline min/med/max |
|---|---|---|
| `loading` | 53.3 / 55.4 / 108.6 | 66.2 / 134.0 / 199.3 |
| `land_mask` | 12.4 / 13.4 / 24.7 | 12.3 / 24.2 / 38.5 |
| `destripe_climatology` (A) | 322.0 / 511.5 / 694.5 | 420.2 / 704.7 / 989.5 |
| `destripe_climatology_merge` | 14.4 / 16.2 / 23.4 | 12.5 / 14.3 / 17.4 |
| `destripe_offsets` (B) | **42.1 / 46.8 / 52.2** | **488.1 / 612.2 / 686.9** |
| `peak_rss_mb` | 904 / 23,581 / 35,913 | 900 / 55,737 / 59,318 |

Phase B is 13.1x faster at the median and 13.2x at the max. The projection from
the throughput probe was about 101 s of staging work; the measured phase-B read
is 42 to 52 s, and the phase-A write is not separately instrumented.

Peak RSS more than halved, from 55.7 GB median to 23.6 GB on 64 GB VMs. That is
a side effect of reading each block in bounded scene groups rather than one
whole-block compute, not a designed outcome, and it is not confounded by #134.

## Cleanup

`sweep_coarse_stage` removed all 8,424 staged objects; the prefix lists empty.
The prefix from an earlier launch that failed its plan-digest guard was also
removed, 122 objects.

## Cost

About 16 credits observed on the first page of billing events for cluster
2003236, over 15 instance events, for a run whose longest task was 13m 50s.
