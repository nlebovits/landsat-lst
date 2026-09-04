# Issue #125 production validation, two runs

Two 15-shard offsets stages for `S30W065`, window 2021-2025, on the production
VM configuration (`r6i.2xlarge` / `m6i.4xlarge` spot, `us-west-2`), both with
`coiled_retries=0`, both over the same frozen plan and item set copied from
`shard-S30W065-2021-2025-20260903T231831Z-group6-r1`.

| Run | Cluster | Staging | Run id |
|---|---|---|---|
| Baseline | 2004094 | off | `shard-S30W065-2021-2025-20260904T061039Z-v2base` |
| Calibration | 2004187 | on | `shard-S30W065-2021-2025-20260904T064340Z-staged` |

Both completed 15 of 15 shards on the first attempt. Neither retried.

## The reference had to be rebuilt first

`_offsets/S30W065/2021-2025/f2/v2-eac1fa94a9fc760f.json` was invalid. Its body
declared `algorithm_version` 1 and digest `a662c8e0b67b4254`, and it was
byte-identical to the v1 record (sha256 `601cd5caa9e95c2a...`). The v1 record
had been copied to the v2 key, so any consumer reading that key received an
approximate-transform estimate believing it was exact.

It was copied to `_offsets/_quarantine/` with the reason recorded, then deleted.
The v1 record was left untouched. The baseline run then wrote a genuine record
at that key: `algorithm_version` 2, digest `eac1fa94a9fc760f`, 1,031 scenes,
written 2026-09-04T06:39:37Z.

## Correctness: exact

The staged run's 15 partials were merged and compared against that record.

| Check | Result |
|---|---|
| Offsets bit-identical | **yes**, max absolute delta 0.0 |
| `n_valid` identical | **yes** |
| Time axis identical | **yes**, at nanosecond precision |
| Scenes covered | 1,031 of 1,031, from 15 of 15 partials |
| `algorithm_version` | 2 = 2 |
| `digest` | `eac1fa94a9fc760f` = `eac1fa94a9fc760f` |
| `offset_resolution_factor` | 2 = 2 |
| tile, window, scenes | all match |

## Timings, seconds, min / median / max over 15 shards

| Phase | Baseline, unstaged | Staged |
|---|---|---|
| `loading` | 107.8 / 109.0 / 113.5 | 108.1 / 111.0 / 115.8 |
| `land_mask` | 24.3 / 25.2 / 29.2 | 24.6 / 26.1 / 28.7 |
| `destripe_climatology` (A) | 332.9 / 493.5 / 677.4 | 330.8 / 516.0 / 632.9 |
| `destripe_climatology_merge` | 15.6 / 19.8 / 26.0 | 18.2 / 23.7 / 30.5 |
| `destripe_offsets` (B) | 293.7 / 408.7 / 417.7 | **47.4 / 53.6 / 61.0** |
| unit wall clock | 1,139.6 / 1,259.0 / 1,270.9 | **850.8 / 862.6 / 884.4** |
| `peak_rss_mb` | 40,553 / 53,816 / 55,922 | 28,319 / 35,409 / 36,064 |

Phase B is 7.6x faster at the median and 6.8x at the max. The offsets stage is
gated by its slowest shard, and that falls from 1,270.9 s to 884.4 s, a saving
of 386.5 s or 30.4%.

Phase A is a wash: 4.6% slower at the median, where the staging write lands, and
6.6% faster at the max. Peak RSS falls 34% at the median, from 53.8 GB to
35.4 GB on 64 GB VMs, because each block is now read in bounded scene groups
rather than one whole-block compute.

The earlier comparison against the August anchor run overstated the gain at
13.1x. That run used the approximate transform and the unweighted block split,
so its phase B of 488 to 687 s is not what this code does unstaged. The honest
figure is 7.6x, measured here against the same code, plan, and configuration.

## Staging volume and cleanup

8,424 objects of 21.0 MB, one phase-A block of 1,024 squared by one
`TIME_CHUNK` of 10 scenes as `uint16`. 167.0 GB by the byte model
(9,000 x 9,000 x 1,031 x 2 B), half of one 334.0 GB source pass.

`sweep_coarse_stage` removed all 8,424 and the prefix lists empty. A second
sweep removed 0, so it is idempotent.

## Cost

| Run | Credits | Instance events |
|---|---|---|
| Baseline, unstaged | 44.6 | 16 |
| Calibration, staged | 31.4 | 16 |

The staged run cost 29.6% less than the unstaged baseline, because it finished
sooner on the same fleet. Two-run total 76.0 credits.

## Tile-level effect

The offsets stage falls by 386.5 s on its critical path. Nothing else in the
tile changed, and the composite stage was not run here, so a tile-level figure
is a projection rather than a measurement: against a tile critical path of
roughly 3,400 s it is about 11%.
