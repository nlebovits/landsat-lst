"""Read the #125 calibration off the bucket and decide it against the baseline.

Values are compared against the direct-path v2 offsets record for the same
tile, window, factor, and scene set. Timings are compared against the retained
anchor run's ``phase_seconds``. Then the stage is swept and the sweep verified.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

BASELINE = "shard-S30W065-2021-2025-20260823T102135Z"
TILE = "S30W065"
PHASES = (
    "loading",
    "land_mask",
    "shard_plan_wait",
    "destripe_climatology",
    "destripe_stage_write",
    "shard_barrier_wait",
    "destripe_climatology_merge",
    "destripe_offsets",
    "uploading",
)


def stats(vals):
    return {
        "n": len(vals),
        "min": round(min(vals), 1),
        "median": round(statistics.median(vals), 1),
        "max": round(max(vals), 1),
    }


def phase_rows(storage, root):
    rows = []
    for key in sorted(storage.list_prefix(f"{root}/state/")):
        if re.search(r"offsets\.\d+\.\d+\.json$", key):
            rows.append(json.loads(storage.read_text(key)))
    return rows


def table(rows):
    out = {}
    for phase in PHASES:
        vals = [r["phase_seconds"][phase] for r in rows if phase in (r.get("phase_seconds") or {})]
        if vals:
            out[phase] = stats(vals)
    return out


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--sweep", action="store_true", help="delete the stage after reading it")
    args = ap.parse_args()

    from landsat_lst import shard_tasks, shards  # noqa: PLC0415
    from landsat_lst.offsets import merge_scene_partials  # noqa: PLC0415
    from landsat_lst.storage import get_storage  # noqa: PLC0415

    # The planner and every shard call this, so the plan digest -- which covers
    # load_chunk_size -- is computed from the same number everywhere. A reader
    # that skips it computes a different digest and is refused its own plan.
    shard_tasks.apply_shard_settings()

    storage = get_storage()
    root = shards.shard_root(args.run_id, TILE)
    plan = shards.TilePlan.from_dict(json.loads(storage.read_text(f"{root}/plan.json")))
    time_coord = shard_tasks._time_coord(plan)

    # --- values
    partials, spans = [], []
    for key in sorted(storage.list_prefix(f"{root}/offsets/scene/")):
        partials.append(json.loads(storage.read_text(key)))
        m = re.search(r"scene/s(\d+)-(\d+)\.json$", key)
        if m:
            spans.append((int(m.group(1)), int(m.group(2))))
    offset, n_valid = merge_scene_partials(partials, time_coord)

    ref_key = shard_tasks._offset_key(plan).storage_key
    ref_body = storage.read_text(ref_key)
    verdict = {"reference_key": ref_key, "reference_present": ref_body is not None}
    if ref_body is not None:
        ref = json.loads(ref_body)
        a = np.asarray([np.nan if v is None else v for v in ref["offset"]], dtype=np.float64)
        b = np.asarray(offset.values, dtype=np.float64)
        verdict["offsets_identical"] = bool(np.array_equal(a, b, equal_nan=True))
        verdict["max_abs_delta"] = float(np.nanmax(np.abs(a - b))) if a.shape == b.shape else None
        verdict["n_valid_identical"] = bool(
            np.array_equal(np.asarray(ref["n_valid"]), np.asarray(n_valid.values))
        )
        verdict["scenes"] = int(offset.sizes["time"])
        verdict["partial_spans"] = spans
        verdict["scenes_covered"] = sum(hi - lo for lo, hi in spans)
        # provenance: the record must describe the same estimator and inputs
        expected = shard_tasks._offset_key(plan)
        verdict["provenance"] = {
            "algorithm_version": (ref.get("algorithm_version"), expected.algorithm_version),
            "digest": (ref.get("digest"), expected.digest),
            "offset_resolution_factor": (ref.get("offset_resolution_factor"), plan.offset_factor),
            "tile": (ref.get("tile"), plan.tile),
            "window": (ref.get("window"), plan.window),
            "scenes": (ref.get("scenes"), int(offset.sizes["time"])),
        }
        verdict["provenance_matches"] = all(a == b for a, b in verdict["provenance"].values())
        # Records written before 2026-08-22 carry second-precision stamps, and
        # the reader accepts them through an unambiguous truncated match. The
        # comparison follows the same rule rather than demanding nanoseconds.
        from landsat_lst.offsets import _times_iso, _truncation_of  # noqa: PLC0415

        stored = ref.get("times") or []
        verdict["times_identical"] = stored == _times_iso(time_coord) or _truncation_of(
            stored, time_coord
        )
        verdict["times_precision"] = "ns" if stored == _times_iso(time_coord) else "s (legacy)"

    # --- timings
    now, before = phase_rows(storage, root), phase_rows(storage, shards.shard_root(BASELINE, TILE))
    result = {
        "run_id": args.run_id,
        "partials": len(partials),
        "correctness": verdict,
        "timings_staged": table(now),
        "timings_baseline": table(before),
        "shards_staged": len(now),
        "shards_baseline": len(before),
    }
    staged_now = storage.list_prefix(f"{root}/stage/")
    result["staged_objects_before_sweep"] = len(staged_now)
    peaks = [r.get("peak_rss_mb") for r in now if r.get("peak_rss_mb")]
    base_peaks = [r.get("peak_rss_mb") for r in before if r.get("peak_rss_mb")]
    if peaks:
        result["peak_rss_mb_staged"] = stats(peaks)
    if base_peaks:
        result["peak_rss_mb_baseline"] = stats(base_peaks)

    if args.sweep:
        result["swept"] = shard_tasks.sweep_coarse_stage(args.run_id, TILE, storage=storage)
        result["staged_objects_after_sweep"] = len(storage.list_prefix(f"{root}/stage/"))

    out = ROOT / "results" / "probe" / "125-calibration-analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
