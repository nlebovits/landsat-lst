"""#133: does phase-A wall time track scene overlap, and what would a weighted split do?

Reads the retained artifacts of one sharded run and answers from them alone:

1. Per-shard ``destripe_climatology`` seconds (from ``state/offsets.*.1.json``)
   against the number of STAC item footprints crossing the blocks that shard
   owned (from ``plan.json`` and ``items.json``). Per *shard*, not per block:
   the heartbeat overwrites its state object, so no per-block series survives.
2. The phase-A max a scene-weighted contiguous split would have produced, from
   the fitted line, both as a point estimate and as an expected max over the
   fleet with the fit's residual noise included.

The split it models is the production one: ``shards.block_scene_weights`` and
``shards.balance_by_weight`` are the functions the planner and the shards run,
so the model and the fleet cannot disagree about which shard owns which block.

Usage::

    python scripts/analyze_offsets_phase_a.py <dir>

where ``<dir>`` holds ``plan.json``, ``items.json``, and ``state/`` copied from
``_shards/{run_id}/{tile}/`` on the bucket. Nothing is fetched and nothing is
launched. The tile bbox is read from the plan's tile name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import shape

from landsat_lst import shards
from landsat_lst.tiling import geobox_for_bbox, parse_tile_name

SIMULATIONS = 20_000


def balance_spans(weights: np.ndarray, n: int) -> list[tuple[int, int]]:
    """``(start, stop)`` per group of the production weighted split."""
    spans = shards.block_spans((1, len(weights)), 1)
    groups = shards.balance_by_weight(spans, [float(w) for w in weights], n)
    out = []
    start = 0
    for group in groups:
        out.append((start, start + len(group)))
        start += len(group)
    return out


def main(root: Path) -> None:
    plan = shards.TilePlan.from_dict(json.loads((root / "plan.json").read_text()))
    items = json.loads((root / "items.json").read_text())
    state = {}
    for path in (root / "state").glob("offsets.*.1.json"):
        state[int(path.name.split(".")[1])] = json.loads(path.read_text())
    if len(state) != plan.ref_shards:
        msg = f"{len(state)} round-1 offsets state objects for {plan.ref_shards} shards"
        raise SystemExit(msg)

    tile = parse_tile_name(plan.tile)
    affine = geobox_for_bbox(tile.bbox, plan.offset_factor).affine
    footprints = [shape(it["geometry"]) if it.get("geometry") else None for it in items]
    weights = np.array(
        shards.block_scene_weights(plan.blocks, plan.block_has_land, footprints, affine),
        dtype=float,
    )

    groups = shards.climatology_groups(plan)
    starts = np.cumsum([0] + [len(g) for g in groups])
    phase_a = np.array(
        [state[k]["phase_seconds"]["destripe_climatology"] for k in range(len(groups))]
    )
    waits = np.array([state[k]["phase_seconds"]["shard_barrier_wait"] for k in range(len(groups))])
    per_shard = np.array([weights[starts[k] : starts[k + 1]].sum() for k in range(len(groups))])
    unit_wall = np.mean([sum(state[k]["phase_seconds"].values()) for k in state])

    print(
        f"tile={plan.tile} blocks={len(plan.blocks)} land={sum(plan.block_has_land)} "
        f"items={len(items)} shards={len(groups)} stored_weights={plan.block_weights is not None}"
    )
    print(
        f"per-block items: min={weights.min():.0f} mean={weights.mean():.0f} max={weights.max():.0f}"
    )
    print(f"{'shard':>5} {'blocks':>6} {'items':>7} {'phaseA':>7} {'wait':>6}")
    for k, group in enumerate(groups):
        print(f"{k:>5} {len(group):>6} {per_shard[k]:>7.0f} {phase_a[k]:>7.1f} {waits[k]:>6.1f}")

    r = np.corrcoef(per_shard, phase_a)[0, 1]
    slope, intercept = np.polyfit(per_shard, phase_a, 1)
    resid_sd = (phase_a - (intercept + slope * per_shard)).std(ddof=2)
    print(
        f"\nphase A vs items per shard: r={r:+.3f} r2={r * r:.3f} "
        f"fit t = {intercept:.1f} + {slope:.4f} s/item, resid sd={resid_sd:.1f}s"
    )
    if r < 0.5:
        print("STOP: phase-A time does not track scene overlap; no static balance exists")
        return

    spans = balance_spans(weights, len(groups))
    balanced = np.array([weights[s:e].sum() for s, e in spans])
    point = intercept + slope * balanced
    ideal = intercept + slope * weights.sum() / len(groups)
    print(
        f"\nmeasured phase A: max={phase_a.max():.0f} mean={phase_a.mean():.0f}; unit wall mean={unit_wall:.0f}s"
    )
    print(f"weighted split blocks per shard: {[e - s for s, e in spans]}")
    print(
        f"weighted split items per shard: {balanced.min():.0f}-{balanced.max():.0f} "
        f"(current {per_shard.min():.0f}-{per_shard.max():.0f})"
    )
    print(
        f"point estimate: max={point.max():.0f}s, {phase_a.max() - point.max():.0f}s "
        f"({(phase_a.max() - point.max()) / unit_wall * 100:.1f}% of unit wall) below measured"
    )
    print(
        f"ideal non-contiguous bound: max={ideal:.0f}s, {phase_a.max() - ideal:.0f}s below measured"
    )

    rng = np.random.default_rng(0)

    def expected_max(x: np.ndarray) -> tuple[float, np.ndarray]:
        sims = intercept + slope * x[None, :] + rng.normal(0, resid_sd, (SIMULATIONS, len(x)))
        m = sims.max(axis=1)
        return float(m.mean()), np.percentile(m, [10, 90])

    cur, cur_p = expected_max(per_shard)
    bal, bal_p = expected_max(balanced)
    print(
        f"\nnoise-aware E[max] over {len(groups)} shards (resid sd {resid_sd:.0f}s): "
        f"current={cur:.0f}s [{cur_p[0]:.0f},{cur_p[1]:.0f}] "
        f"weighted={bal:.0f}s [{bal_p[0]:.0f},{bal_p[1]:.0f}]"
    )
    print(
        f"expected difference {cur - bal:.0f}s = {(cur - bal) / unit_wall * 100:.1f}% of unit wall"
    )
    print(f"model check: measured max {phase_a.max():.0f}s vs modelled current E[max] {cur:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
