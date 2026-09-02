# Reproducing the composite performance experiment (candidate 1)

Protocol: `docs/perf/composite-experiment-contract.md`, sections 3, 4, 5 and 6.
Harness: `scripts/perf/composite_experiment.py`. No network, no cloud, no
credentials. Every configuration runs in a fresh subprocess.

## 0. Fixtures

`results/fixtures/{N40W075,S30W065}_2021-2025_n300_f8` must exist. They are
gitignored (~6.08 GB each) and are read-only inputs here; build them with
`landsat-lst fixture --tile <tile> --factor 8` or symlink an existing copy:

```bash
mkdir -p results
ln -sfn /path/to/checkout/results/fixtures results/fixtures
```

## 1. Pin the offsets (once per fixture)

The estimator is out of scope, and its own chunk sensitivity would confound the
composite measurement, so the offset vector is estimated once and read back on
every subsequent run.

```bash
uv run --frozen python scripts/perf/composite_experiment.py \
  pin-offsets --fixture S30W065_2021-2025_n300_f8
uv run --frozen python scripts/perf/composite_experiment.py \
  pin-offsets --fixture N40W075_2021-2025_n300_f8
```

Writes `results/perf/offsets-<tile>.json`, whose sha256 is recorded in every
result row.

## 2. Measure one arm

```bash
uv run --frozen python scripts/perf/composite_experiment.py sweep \
  --fixture S30W065_2021-2025_n300_f8 \
  --fixture N40W075_2021-2025_n300_f8 \
  --chunk 512 --threads 4 --arm baseline --reps 3
```

The first run of a tile writes `results/perf/truth-<tile>.npz` (gitignored,
~90 MB); every later run of that tile compares against it and reports
`equal_lst`, `equal_qa`, `equal_encoded` and `bit_identical`.

Thread sweep (recorded, not an acceptance point):

```bash
... sweep --fixture S30W065_2021-2025_n300_f8 --threads 1,2 --arm baseline --reps 3
```

8 threads is refused: the measured multiplier on this machine projects 24.2 GB
against the contract's 20 GB cap.

## 3. Measure both arms, interleaved

Another agent held two full cores of this laptop throughout, so the two arms are
alternated rather than run in blocks; any drift then hits both equally.

```bash
bash scripts/perf/interleaved.sh
```

## Output

`results/perf/composite-experiment.jsonl`, one object per run. `arm` and
`pipeline_sha256` identify which side of the change produced it.

## Recovering the wall-clock number on a quiet machine

The acceptance run was taken at a load average near 13 of 16 cores, with seven
or eight other agents' jobs on the same laptop. Peak RSS, the native-pass tally,
the fused task count and the checksums are unaffected by that; the wall-clock
rows are not. To recover only that number later:

```bash
REPS=3 bash scripts/perf/rerun_wall.sh
```

It needs nothing else — the offsets are already pinned under `results/perf/` and
the truth arrays rebuild on the first run of each tile. A row whose min-max
spread exceeds 1.30x is invalid whatever the machine looked like.
