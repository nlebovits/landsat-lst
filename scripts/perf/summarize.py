"""Render the experiment's rows as the contract's acceptance table.

Reads ``results/perf/composite-experiment.jsonl`` and reports, per fixture and
thread count, the median of the reps together with the four acceptance checks of
the contract's section 6. Writes ``results/perf/summary.json`` alongside.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

PERF_DIR = Path("results/perf")

#: Contract section 6: a candidate clears the wall-clock arm at or above this.
WALL_THRESHOLD = 1.50
#: ...or the memory arm at or above this.
MEMORY_THRESHOLD = 1.35
#: ...and must in every case stay inside the composite task band.
TASK_BAND = 1.4
#: A configuration whose reps spread wider than this is invalid (stop 7).
SPREAD_LIMIT = 1.3


def _load() -> list[dict]:
    path = PERF_DIR / "composite-experiment.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _check_arm_integrity(rows: list[dict]) -> None:
    """Every row of one arm must have been produced by one pipeline.py.

    The arm under test is a working-tree edit, and the runner swaps that file
    between arms. Anything else touching it mid-run -- a ``git stash``, an
    editor, a second script -- would silently label a baseline execution as a
    treatment one. Two distinct hashes inside an arm, or one hash shared across
    both, means exactly that happened and the rows are not evidence.
    """
    by_arm: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_arm[row["arm"]].add(row["provenance"]["pipeline_sha256"])
    for arm, shas in by_arm.items():
        if len(shas) != 1:
            msg = f"arm {arm!r} spans {len(shas)} pipeline.py hashes: {sorted(shas)}"
            raise SystemExit(msg)
    if len(by_arm) > 1 and len({next(iter(s)) for s in by_arm.values()}) != len(by_arm):
        msg = f"two arms share one pipeline.py hash: {by_arm}"
        raise SystemExit(msg)


def main() -> int:
    rows = _load()
    _check_arm_integrity(rows)
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["fixture"], row["threads"])].append(row)

    summary: dict[str, dict] = {}
    for (arm, fixture, threads), rows in sorted(groups.items()):
        walls = [r["wall_s"] for r in rows]
        peaks = [r["peak_rss_mb"] for r in rows]
        summary[f"{arm}|{fixture}|t{threads}"] = {
            "arm": arm,
            "fixture": fixture,
            "threads": threads,
            "reps": len(rows),
            "wall_s_median": median(walls),
            "wall_s_min": min(walls),
            "wall_s_max": max(walls),
            "wall_spread": max(walls) / min(walls),
            "peak_rss_mb_median": median(peaks),
            "graph_build_s_median": median(r["graph_build_s"] for r in rows),
            "cores_busy_median": median(r["cores_busy"] for r in rows),
            "native_passes": rows[0]["native_passes"],
            "composite_tasks": rows[0]["composite_tasks"],
            "floor_mb": rows[0]["floor_mb"],
            "scenes_kept": rows[0]["scenes_kept"],
            "lst_sha256": rows[0]["lst_sha256"],
            "qa_sha256": rows[0]["qa_sha256"],
            "encoded_sha256": rows[0]["encoded_sha256"],
            "bit_identical": all(r.get("bit_identical", True) for r in rows),
        }

    verdicts = {}
    for key, treat in summary.items():
        if treat["arm"] != "treatment":
            continue
        base_key = key.replace("treatment|", "baseline|", 1)
        base = summary.get(base_key)
        if base is None:
            continue
        speedup = base["wall_s_median"] / treat["wall_s_median"]
        mem_ratio = base["peak_rss_mb_median"] / treat["peak_rss_mb_median"]
        task_ratio = treat["composite_tasks"] / base["composite_tasks"]
        verdicts[key] = {
            "speedup": speedup,
            "memory_reduction": mem_ratio,
            "task_ratio": task_ratio,
            "bit_identical": treat["bit_identical"]
            and treat["lst_sha256"] == base["lst_sha256"]
            and treat["qa_sha256"] == base["qa_sha256"]
            and treat["encoded_sha256"] == base["encoded_sha256"],
            "native_passes_held": abs(treat["native_passes"] - 1.0) < 1e-9,
            "tasks_in_band": task_ratio <= TASK_BAND,
            "clears_wall": speedup >= WALL_THRESHOLD,
            "clears_memory": mem_ratio >= MEMORY_THRESHOLD,
            "spread_ok": treat["wall_spread"] <= SPREAD_LIMIT
            and base["wall_spread"] <= SPREAD_LIMIT,
        }
        verdicts[key]["accepted"] = (
            verdicts[key]["bit_identical"]
            and verdicts[key]["native_passes_held"]
            and verdicts[key]["tasks_in_band"]
            and (verdicts[key]["clears_wall"] or verdicts[key]["clears_memory"])
        )

    out = {"runs": summary, "verdicts": verdicts}
    (PERF_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")

    print(
        f"{'arm':<10}{'fixture':<12}{'thr':>4}{'reps':>5}{'wall_s':>10}{'spread':>8}"
        f"{'peak_MB':>10}{'tasks':>8}{'passes':>8}  bit"
    )
    for r in summary.values():
        print(
            f"{r['arm']:<10}{r['fixture'].split('_')[0]:<12}{r['threads']:>4}{r['reps']:>5}"
            f"{r['wall_s_median']:>10.2f}{r['wall_spread']:>8.2f}"
            f"{r['peak_rss_mb_median']:>10.0f}{r['composite_tasks']:>8}"
            f"{r['native_passes']:>8.2f}  {r['bit_identical']}"
        )
    print()
    for key, v in verdicts.items():
        print(
            f"{key}: speedup {v['speedup']:.2f}x  mem {v['memory_reduction']:.2f}x  "
            f"tasks {v['task_ratio']:.2f}x  bit_identical {v['bit_identical']}  "
            f"passes_held {v['native_passes_held']}  ACCEPTED {v['accepted']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
