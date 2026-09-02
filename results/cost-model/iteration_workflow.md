# Cached-data iteration workflow (issue #108)

Status: design document. Submits no production tile and runs nothing in the
cloud.

## The distinction this document exists to hold

Issue #108 asks for two things that are easy to conflate and are almost
unrelated:

- **Fleet throughput** — how much it costs and how long it takes to build 700
  tiles. Consolidation addresses this by amortizing boot across tiles.
- **Single-tile iteration latency** — how long a scientist waits to learn
  whether a change to the estimator, the QA bits, or the export was an
  improvement.

Consolidation does **nothing** for the second. It makes a VM serve many tiles;
it does not make one tile's work smaller. The issue says so directly:
consolidation "must not be sold as sufficient by itself to make a single
full-tile rerun fast." If quality work still waits on a production tile after
#108 ships, the loop is unchanged and the second objective is unmet.

The answer is not a faster tile. It is **not running the tile**, because for
most questions the tile is not the cheapest artifact that can answer them.

## What is already cached, and what each one buys

| Artifact | Key / path | Size | Replaces | Provenance |
|---|---|---|---|---|
| Offset record (ADR-012) | `_offsets/{tile}/{window}/f{factor}/v{version}-{digest}.json` | ~600 floats | the entire offsets stage | measured: 27 of ~35 min on a 300-scene tile |
| Shard plan | `_shards/{run}/{tile}/plan.json` | KB | STAC query + two graph builds | measured: 3.5 min on S30W065, boot included |
| Fixture stack | `landsat-lst fixture --tile T --factor 8` | 6.1 GB | STAC query + hundreds of GB of coarse reads | measured: sizes in CLAUDE.md |
| Band slabs | `_shards/{run}/{tile}/bands/` | GB | the composite pass | inferred from `reexport_qa_count.py` |
| GED gap mask | `data/ged_gap_mask.npz` | 2.8 MB | ASTER granule reads | measured |
| Probe rates | `results/probe/*.json` | KB | I/O ladder runs | user-reported 2026-08-21/22: absent from every commit, see Verification below |

Two of these carry a caveat sharp enough to change conclusions, and both are
already written down in CLAUDE.md:

- **The fixture answers a relative question, not an absolute one.** Offset
  error grows linearly in the coarsening factor. Two estimators read the same
  pixels, so a *comparison* is exact at any factor; a fixture offset quoted as
  a production offset is wrong.
- **A sampled window cannot check a rejection fraction.** 300 scenes over five
  years leaves each month ~25 scenes for its climatology instead of 244, and
  the noisy reference inflates offsets — 69% rejected on the sample against
  21.8% at Pergamino. Any question whose answer is a rejection rate needs the
  full window.

## The load-bearing asymmetry: only the estimate is cached, never the rejection

`max_offset_c` and the sparse floors are applied to whatever the cache
returns. So a **cap sweep pays the estimator once** and every subsequent cap
is arithmetic on ~600 floats. That is the single largest iteration lever in
the codebase, and it already exists.

It has an exact complement that must be stated in the same breath, because it
is where the lever stops working:

**`offsets.ALGORITHM_VERSION` must be bumped whenever `offset_graph`, the QA
bits in `create_qa_mask`, or the DN-to-Celsius conversion change — and bumping
it invalidates every cached record.** The changes that most need fast
iteration are precisely the changes the offset cache cannot serve. Anyone
planning an estimator investigation around "the offsets are cached" has
planned around the wrong artifact; they want the fixture.

## Routing a question to its cheapest artifact

| Question | Cheapest artifact that answers it | Order of magnitude | Cloud? |
|---|---|---|---|
| Task counts, memory floor | `landsat-lst plan` | seconds | no |
| Does estimator A beat B? | fixture, factor 8 | minutes | no |
| Does a QA-bit change move offsets? | fixture, factor 8 | minutes | no |
| Does a cap change the keep-set? | cached offset record | seconds | no |
| Did a change move a benchmark number? | `pytest tests/benchmark` | ~30 s | no |
| Does a COG header/pyramid change work? | band slabs, `reexport_qa_count.py` pattern | minutes | read-only |
| Is the GED/land mask right? | band slabs (output-side, LST only) | minutes | read-only |
| Peak RSS at production geometry | `landsat-lst benchmark --distributed` | ~20 min, <$1 | yes |
| What is the actual rejection fraction? | full-window offsets stage, one tile | hours | yes |
| The product | `landsat-lst process --distributed` | hours | yes |

The rule the table encodes: **a question goes to the cloud only when its answer
depends on production geometry or the full window.** Everything above that line
is a laptop question, and the two rows that most often get sent to the cloud by
mistake — estimator comparison and cap selection — are the two cheapest rows in
the table.

Two guardrails from CLAUDE.md apply to the local rows and are not optional:
`landsat-lst benchmark` refuses more than 200 scenes locally, and an unbounded
local graph build has taken a 64 GB desktop down. Building a graph allocates
Python objects whether or not you compute it.

## Gaps — what is not cached and would have to be built

These are named as gaps, not proposed as work, because each is a real cost and
some may not be worth paying:

1. **Band slabs have no documented lifecycle.** `reexport_qa_count.py` proves
   they can be re-read, but nothing states whether they survive a run's
   completion, how they are keyed for reuse across runs, or when they are
   collected. An iteration path that depends on them is depending on a
   side effect. This is the highest-value gap: it is the difference between
   re-exporting in minutes and recomputing a composite in hours.
2. **No fixture at the composite stage.** The fixture caches the *input*
   stack. There is no cached post-destriping stack, so a change downstream of
   de-striping and upstream of export has no cheap artifact and falls all the
   way through to a production tile.
3. **The acceptance run has no findings record.** Its measurements survive
   only as prose in docstrings, a config description, an ADR, and one test's
   class constants. The `$7.28` figure appears nowhere in the repository. A
   calibration run that is not written down cannot be reconciled against later,
   which is exactly what issue #108's cloud gate requires.

## Latency targets

Proposed as targets to measure against, **not** as claims. None of these has
been measured on this repository, and a target quoted as a result is the
failure mode this document is trying to avoid.

| Loop | Target | Basis |
|---|---|---|
| Cap / rejection sweep | < 1 min | arithmetic on ~600 cached floats |
| Estimator A/B on a fixture | < 10 min | 6.1 GB memory-mapped at production chunking |
| Re-export from band slabs | < 15 min | three passes over output rasters at `R_EXPORT_MB_S` = 100 MB/s |
| Full-window single-tile rerun | unchanged by #108 | consolidation amortizes across tiles, not within one |

The last row is the important one. It is a target *not* to move, stated so that
consolidation is not later credited with an improvement it cannot produce.

## Verification

The iteration path must be exercised by something that runs in CI, or it will
rot exactly as `results/probe/` did — referenced by code and docs, present on
one laptop, absent from the repository. The minimum is a test that a cached
offset record is read back and applied, and that a cap change alters the
keep-set without recomputing the estimate. That test is cheap, it pins the
lever this document is built on, and it does not exist today.
