"""Project the 700-tile build's cost and wall time from measured rates.

Walks every production land tile (``generate_jobs``), computes its land
fraction from the Natural Earth polygons the pipeline already uses,
assigns a scene count by interpolating measured STAC counts over latitude
(see ``_SCENES_BY_ABS_LAT``), and runs each tile through
``landsat_lst.projection.tile_projection``.

Outputs totals against the two acceptance bounds: <= 60 min per tile
(feasible with the projected fleet by construction; the check here is the
fleet size and VM-hour total) and total projected spend comfortably below
$5k. Provenance: rates are [M] from the ladder probe; land fractions are
[M] from geometry; the scene curve is [M] at three latitudes and
interpolated between them, honest to roughly 15%.

    python scripts/estimate_fleet_cost.py
    python scripts/estimate_fleet_cost.py --json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

# Scene counts measured 2026-08-21 via Planetary Computer STAC (landsat-c2-l2,
# L8+L9, 2021-2025, eo:cloud_cover < 100), one 5-degree tile per latitude band.
# A 1/cos(lat) path-overlap model was tried first and is WRONG in both
# directions: the equator measured 3,899 (the model said ~2,340) and 57.5N
# measured 3,802 (the model said 4,300). The curve is V-shaped with its
# minimum at mid-latitudes, so this interpolates |lat| through the three
# measured points and clamps outside them. Three points is a thin basis --
# the summary is honest to ~15%, not better.
_SCENES_BY_ABS_LAT = [(2.5, 3899), (37.5, 2758), (57.5, 3802)]


def scenes_for(lat_center: float) -> int:
    lat = abs(lat_center)
    pts = _SCENES_BY_ABS_LAT
    if lat <= pts[0][0]:
        return pts[0][1]
    for (la, sa), (lb, sb) in itertools.pairwise(pts):
        if la <= lat <= lb:
            t = (lat - la) / (lb - la)
            return round(sa + t * (sb - sa))
    return pts[-1][1]


def land_fractions(tiles) -> dict[str, float]:
    from shapely.geometry import box  # noqa: PLC0415
    from shapely.ops import unary_union  # noqa: PLC0415

    from landsat_lst.masks import load_land_polygons  # noqa: PLC0415

    # Cache next to the script: the NE download is flaky (SSL EOFs) and this
    # script re-runs during planning.
    cache = HERE.parent / "results" / "probe"
    cache.mkdir(parents=True, exist_ok=True)
    land = load_land_polygons(cache_dir=cache)
    sindex = land.sindex
    out: dict[str, float] = {}
    for tile in tiles:
        w, s, e, n = tile.bbox
        cell = box(w, s, e, n)
        cand = land.iloc[list(sindex.query(cell))]
        if cand.empty:
            out[tile.name] = 0.0
            continue
        inter = unary_union(list(cand.intersection(cell).values))
        out[tile.name] = min(1.0, inter.area / cell.area)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from landsat_lst.job import generate_jobs  # noqa: PLC0415
    from landsat_lst.projection import (  # noqa: PLC0415
        SPOT_FACTOR_RANGE,
        VM_HOURLY_ON_DEMAND,
        tile_projection,
    )

    jobs = generate_jobs()
    tiles = [j.tile for j in jobs]
    fractions = land_fractions(tiles)

    rows = []
    for tile in tiles:
        _w, s, _e, n = tile.bbox
        lat_c = (s + n) / 2
        p = tile_projection(scenes=scenes_for(lat_c), land_fraction=fractions[tile.name])
        rows.append(
            {
                "tile": tile.name,
                "lat_center": lat_c,
                "scenes": p.scenes,
                "land_fraction": p.land_fraction,
                "vm_hours": p.vm_hours_per_tile,
                "minutes_1vm": p.minutes_per_tile_1vm,
                "n_vms_offsets": p.n_vms_offsets,
                "n_vms_composite": p.n_vms_composite,
                "cost_on_demand": p.cost_on_demand_usd,
            }
        )

    total_vm_h = sum(r["vm_hours"] for r in rows)
    total_od = total_vm_h * VM_HOURLY_ON_DEMAND
    spot_lo = total_od * SPOT_FACTOR_RANGE[0]
    spot_hi = total_od * SPOT_FACTOR_RANGE[1]
    worst = max(rows, key=lambda r: r["vm_hours"])
    summary = {
        "tiles": len(rows),
        "total_vm_hours": round(total_vm_h, 0),
        "cost_on_demand_usd": round(total_od, 0),
        "cost_spot_usd_range": [round(spot_lo, 0), round(spot_hi, 0)],
        "mean_vm_hours_per_tile": round(total_vm_h / len(rows), 2),
        "max_fleet_offsets": round(max(r["n_vms_offsets"] for r in rows), 0),
        "max_fleet_composite": round(max(r["n_vms_composite"] for r in rows), 0),
        "worst_tile": worst["tile"],
        "worst_tile_vm_hours": worst["vm_hours"],
        "under_5k_on_demand": total_od < 5000,
        "under_5k_spot_high_end": spot_hi < 5000,
    }

    if args.json:
        print(json.dumps({"summary": summary, "tiles": rows}, indent=2))
        return 0

    print(json.dumps(summary, indent=2))
    rows.sort(key=lambda r: r["vm_hours"], reverse=True)
    print(
        f"\n{'tile':>8} {'lat':>6} {'scenes':>7} {'land':>5} {'VM-h':>6} {'1-VM min':>9} {'$OD':>7}"
    )
    for r in rows[:10]:
        print(
            f"{r['tile']:>8} {r['lat_center']:>6.1f} {r['scenes']:>7} "
            f"{r['land_fraction']:>5.2f} {r['vm_hours']:>6.1f} "
            f"{r['minutes_1vm']:>9.0f} {r['cost_on_demand']:>7.2f}"
        )
    print("  (10 most expensive tiles shown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
