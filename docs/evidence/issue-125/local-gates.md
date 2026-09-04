# Issue #125 local gates

Run on 2026-09-04 against `feat/125-stage-coarse-observations`, before any cloud
work. Every gate is a comparison against the two-source-pass path on the same
inputs, in the same process where that is safe and in a fresh process where a
high-water mark would otherwise leak between arms.

## Correctness

| Gate | Result |
|---|---|
| Offsets bit-identical to the direct path | max abs delta 0.0 over 4 seeds |
| `n_valid` identical | exact, every seed |
| Climatology `ref` identical | exact, every seed |
| Reconstructed phase-B float32 input identical | exact, every batch |
| Staged objects are `uint16` | every object, every seed |
| Land-free blocks never staged | staged count equals land blocks times batches |
| Missing staged land block | raises, never read as no-observation |
| Stale digest or algorithm version | different prefix, reads nothing |
| Stage swept after use | swept equals written, zero left |

Provenance is unchanged by construction. `OffsetKey` and the merged record are
untouched, so an identical `offset` and `n_valid` mean an identical record; the
stage inherits the key's terms rather than introducing any of its own.

## Peak RSS

Fresh process per measurement, since `getrusage` reports a high-water mark for
the life of a process and a second arm in the first one's interpreter inherits
its peak.

| Scenes, edge | Direct MB | Staged MB | Delta |
|---|---|---|---|
| 120, 512 | 658.9 | 906.3 | +247.4 |
| 240, 768 | 1,792.6 | 2,064.9 | +272.3 |
| 360, 1,024 | 4,463.3 | 4,463.5 | +0.2 |

The staged arm carries a fixed overhead near 250 MB, which is the bounded DN
read buffer plus the conversion's temporaries. It does not scale with the
problem, and by the largest point measured the working set already dominates it.
Production phase A holds a 12.3 GB float32 block and phase B a 3.24 GB batch, so
the overhead is under 2% of either. There is no proportional regression.

Phase B wall time fell in every arm (0.24 against 0.39 s, 1.17 against 1.76 s,
3.27 against 4.47 s). That is a local-disk analogue and not the number the cloud
calibration produces. It is reported here only because it points the same way.

## Suites

- `tests/unit`: 1,637 passed, run credential-less under an empty `HOME` and
  empty `AWS_*` and `COILED_TOKEN`, per the rule in `CLAUDE.md`.
- `tests/integration`: 101 passed.
- `ty check src/`: clean. `ruff check` and `ruff format --check`: clean.
- New: `tests/unit/test_staging.py` (16), `tests/unit/test_dn_stack.py` (12).
