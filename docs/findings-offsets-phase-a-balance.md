# Findings: the offsets stage's in-process waits, and a scene-weighted block split (#133)

Issue #133 asks whether the 20% of the offsets stage's unit wall spent waiting
inside the process has a static cause, and whether a different deal of the
phase-A blocks would remove part of it. This document answers steps 1 and 2 of
the ticket from retained artifacts alone, records the model's number, and
records the decision not to run step 3.

Provenance tags: **[M]** measured, **[D]** derived from a measurement, **[A]**
assumed. Every number carries its source.

## Verdict

Phase-A wall time per shard tracks the number of STAC item footprints crossing
the blocks the shard owns, at r = 0.907 over the 15 round-1 shards of the
S30W065 acceptance run **[M]**. The current split deals blocks by count when
every block has land, so it handed one shard 4,191 footprint intersections and
another 1,798 **[D]**, and every shard then waited for the heaviest at the
in-process barrier. A contiguous split that deals blocks on that weight is
modelled to end phase A 153 s sooner, with the fit's residual noise included,
which is 8.0% of the 1,922 s unit wall and about 5.1 of the 268 credits the run
billed **[D]**. The ticket's discriminator run was not made, on the operator's
call. The change ships as a rule change to the split with no performance claim,
and the next production tile's `phase_seconds` is the measurement.

## 1. What the retained artifacts can and cannot say

The ticket's step 1 asked for per-block phase-A durations. They do not exist.
Each shard republishes one state object every 60 s and the object is
overwritten, so only the final `blocks_done` survives; the shard log carries one
line per phase and none per block. The join is therefore per shard: 15 points,
one per offsets VM, against the sum over its blocks.

Inputs, all under
`s3://us-west-2.opendata.source.coop/nlebovits/landsat-lst/_shards/shard-S30W065-2021-2025-20260823T102135Z/S30W065/`:

- `state/offsets.{0000..0014}.1.json`: `phase_seconds` per shard **[M]**.
- `plan.json`: 81 blocks of 1,024 coarse pixels, all with land, 15 phase-A shards.
- `items.json`: 4,403 STAC items with footprints, 1,031 solar-day steps.

The shard-to-block assignment is reproduced with the production
`shards.balance_by_land`, and the per-block weight with the production
`shards.block_scene_weights`, by `scripts/analyze_offsets_phase_a.py`. Its
output on the anchor run is committed at `results/issue-133/analyze.out`.

## 2. Phase-A time against what each shard reads

| Shard | Blocks | Footprint intersections | Phase A (s) **[M]** | Barrier wait (s) **[M]** |
|---|---|---|---|---|
| 0 | 6 | 2,338 | 519.6 | 484.8 |
| 1 | 5 | 2,410 | 648.6 | 343.8 |
| 2 | 6 | 4,003 | 966.5 | 30.5 |
| 3 | 5 | 3,015 | 779.1 | 212.6 |
| 4 | 5 | 2,964 | 609.8 | 383.9 |
| 5 | 6 | 3,217 | 859.2 | 131.4 |
| 6 | 5 | 3,170 | 704.7 | 293.0 |
| 7 | 6 | 4,191 | 989.5 | 0.1 |
| 8 | 5 | 2,201 | 555.7 | 434.9 |
| 9 | 5 | 1,860 | 420.2 | 575.8 |
| 10 | 6 | 3,954 | 974.8 | 20.4 |
| 11 | 5 | 3,128 | 722.3 | 283.0 |
| 12 | 6 | 3,681 | 946.0 | 50.6 |
| 13 | 5 | 2,258 | 600.3 | 404.2 |
| 14 | 5 | 1,798 | 670.4 | 323.7 |

Per block, intersections run from 198 to 821 (mean 546). Four candidate
predictors of phase-A seconds, per shard **[D]**:

| Predictor | r | r² | Residual sd |
|---|---|---|---|
| Footprint intersections | 0.907 | 0.822 | 80 s |
| Intersecting footprint area | 0.866 | 0.750 | 94 s |
| Block count | 0.674 | 0.454 | 139 s |
| Distinct solar days | 0.489 | 0.239 | 164 s |

The fit on intersections is `t = 111 s + 0.2105 s per item`. The intercept is
the per-shard fixed cost; the slope is the per-item cost of a coarse read. The
solar-day count is a poor predictor because several path-row items share a
day and each is its own read. This agrees with the composite-shard finding
(`docs/findings-composite-shard-bottleneck.md`): the cost is per distinct
source file.

## 3. What a weighted split would have done

`shards.balance_by_weight` finds the contiguous split whose heaviest group has
the least total intersection count. On the anchor plan it deals 7, 5, 5, 5, 5,
6, 5, 4, 5, 8, 4, 4, 5, 5, 8 blocks to the 15 shards and narrows intersections
per shard from 1,798-4,191 to 2,691-3,217 **[D]**. Three numbers from the fit
**[D]**:

| Model | Phase-A max | Below the measured 990 s | Share of unit wall |
|---|---|---|---|
| Point estimate on the weighted split | 788 s | 201 s | 10.5% |
| Expected max over 15 shards, residual noise included | 884 s | 153 s | 8.0% |
| Ideal non-contiguous bound (every shard at the mean) | 731 s | 258 s | 13.4% |

The middle row is the honest one. The fit leaves 80 s of residual per shard,
and a maximum over 15 draws absorbs about 1.7 of them, so the point estimate
overstates the effect. The same simulation puts the current split's expected
max at 1,038 s with a 10-90% band of 965-1,115 s, which contains the measured
990 s, so the model is consistent with the run it was fit on.

In credits **[D]**: 153 s over 15 VMs at 8 vCPU is 5.1 credits per tile, 1.9%
of the 268.11 credits S30W065 billed. Over 700 tiles that is about 3,600
credits, or about $100 at the anchor's $7.28 per tile. The range on that figure
is wide: 80-200 s per tile from the 10-90% bands above, and the fit is from one
tile.

## 4. Why step 3 was not run

The ticket's gates pass, which authorizes one capped offsets-stage run on
S30W065 as the discriminator: 15 VMs for about 33 minutes, about 66 credits,
$4-5 at the spot prices the anchor sampled. The operator chose not to run it,
on 2026-09-03, for four reasons:

1. **The split cannot change the result or the total work.** Every block is
   reduced once by exactly one shard whichever way the blocks are dealt, and
   the blocks compute the same values on any shard. The only quantity that can
   move is which shard finishes last.
2. **It cannot make the stage slower unless the proxy is anti-correlated with
   the cost**, and on the one tile measured it explains 82% of the spread. A
   land-free block keeps a weight of zero, so a coastal tile is not handed a
   worse deal than the land split gave it.
3. **The dollar ceiling is about $90 over the build**, below the price of the
   discriminator times the number of tiles it would need to be repeated on to
   generalize beyond S30W065.
4. **The next production tile is a better discriminator than a rerun.** Every
   offsets shard publishes `phase_seconds["destripe_climatology"]`, so the
   spread under the weighted split is read for free, on a tile the model was
   not fit on, with `landsat-lst explain <run-id> <tile>`.

The consequence for the pull request is that it makes **no performance claim**.
It changes the rule the split follows and stores the weight the rule uses.
This document is where the reasoning lives.

## 5. What the change is, and what it is not

- The planner counts footprints per block once (`shards.block_scene_weights`)
  and stores `block_weights` in the plan. Nothing recomputes them on a VM.
- `shards.climatology_groups` is the only function that turns a plan into
  groups. The planner, every shard (`shard_tasks.climatology_group`), and the
  budget (`budgets._widest_block_share`) go through it.
- A plan without `block_weights`, which is every plan written before this
  change, splits on the land flag exactly as before. The digest ignores the
  weights: they decide who reduces a block, never what it reduces to.
- The budget keeps counting the widest group's *blocks*, not its weight, and
  retains the legacy land split's widest-block share as a floor. A weighted
  split can hand a thinly covered region more blocks (8 of 81 above), while a
  different layout can make every weighted group narrower than the legacy
  groups; neither case is allowed to shorten the old deadline.
- Not changed: the estimator, the block grammar, the artifact keys, the
  serial resolve (shard 0 still writes the plan while 14 VMs poll, 136 s each),
  and phase B.

## 6. The follow-up measurement

After the first production tile runs under the weighted split, read the 15
offsets state objects and compare:

- `phase_seconds["destripe_climatology"]` max minus min across shards, against
  569 s on the anchor **[M]**;
- `phase_seconds["shard_barrier_wait"]` summed across shards, against 3,973 s.

`scripts/analyze_offsets_phase_a.py` runs unchanged on the new run's
directory and reports `stored_weights=True`. If the spread does not narrow,
the proxy failed on that tile and the split should be re-examined, not tuned.

## 7. Where the evidence lives

- `results/issue-133/analyze.out`: the script's output on the anchor run.
- `scripts/analyze_offsets_phase_a.py`: the analysis, over the production
  split functions.
- Anchor run state objects, plan, and items: the S3 prefix in section 1.
- Anchor billing: 268.11 credits, $7.28 (`docs/adr/018-fleet-consolidation.md`,
  `results/cost-model/`).
