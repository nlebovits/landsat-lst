"""Compare the u16-gate shard's band 27 slabs against the retained production slabs.

Output equivalence for the ADR-019 cloud discriminator (issue #136): the
encoded ``lst_p95`` may differ by at most one DN with identical fill masks,
and ``qa_count`` must be equal. Writes the contract's equivalence report with
``max_abs_diff`` and ``passed`` (``passed == max_abs_diff <= tolerance``).

Usage::

    uv run python scripts/experimental/u16_gate_equivalence.py \\
        --baseline results/issue-136/cloud/baseline \\
        --candidate results/issue-136/cloud/candidate \\
        --out results/issue-136/cloud/equivalence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

TOLERANCE_DN = 1


def _read(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(), {
            "dtype": src.dtypes[0],
            "shape": [src.count, src.height, src.width],
            "nodata": src.nodata,
            "transform": list(src.transform)[:6],
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lst_b, meta_lb = _read(args.baseline / "lst_p95_band027.tif")
    lst_c, meta_lc = _read(args.candidate / "lst_p95_band027.tif")
    qa_b, meta_qb = _read(args.baseline / "qa_count_band027.tif")
    qa_c, meta_qc = _read(args.candidate / "qa_count_band027.tif")

    geometry_equal = (
        meta_lb["shape"] == meta_lc["shape"]
        and meta_qb["shape"] == meta_qc["shape"]
        and meta_lb["transform"] == meta_lc["transform"]
        and meta_lb["dtype"] == meta_lc["dtype"] == "uint16"
        and meta_qb["dtype"] == meta_qc["dtype"] == "uint8"
    )
    a = lst_b[0].astype(np.int64)
    b = lst_c[0].astype(np.int64)
    fill_a, fill_b = a == 0, b == 0
    fill_disagreements = int((fill_a != fill_b).sum())
    both = ~fill_a & ~fill_b
    d = np.abs(a - b)[both]
    max_abs = int(d.max()) if d.size else 0
    qa_equal = bool(np.array_equal(qa_b, qa_c))

    passed = bool(
        geometry_equal and fill_disagreements == 0 and max_abs <= TOLERANCE_DN and qa_equal
    )
    report = {
        "method": "pixel-wise integer difference of encoded uint16 lst_p95 band027.tif; qa_count array_equal",
        "tolerance": TOLERANCE_DN,
        "max_abs_diff": max_abs,
        "passed": passed,
        "geometry_equal": geometry_equal,
        "lst_p95": {
            "pixels_compared": int(both.sum()),
            "fill_pixels_baseline": int(fill_a.sum()),
            "fill_pixels_candidate": int(fill_b.sum()),
            "fill_disagreements": fill_disagreements,
            "identical": int((d == 0).sum()),
            "one_dn": int((d == 1).sum()),
            "more_than_one_dn": int((d > 1).sum()),
            "flip_fraction": float((d >= 1).mean()) if d.size else 0.0,
            "mean_signed_dn": float((b - a)[both].mean()) if both.any() else 0.0,
        },
        "qa_count": {
            "equal": qa_equal,
            "differing_cells": int((qa_b != qa_c).sum()),
            "shape": meta_qb["shape"],
        },
        "baseline_meta": {"lst_p95": meta_lb, "qa_count": meta_qb},
        "candidate_meta": {"lst_p95": meta_lc, "qa_count": meta_qc},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
