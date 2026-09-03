"""Write and validate the decision manifest for the ADR-019 cloud discriminator.

Reads the shard's state object (peak_rss_mb, phase_seconds), the equivalence
report, the contract, and the observed cost, and writes the result manifest
the contract names. Then validates it with the repository's own validator, so
the decision the file records is the one the arithmetic supports.

Usage::

    uv run python scripts/experimental/u16_gate_result.py \\
        --contract results/issue-136/cloud/contract.json \\
        --state results/issue-136/cloud/candidate/composite.0027.1.json \\
        --equivalence results/issue-136/cloud/equivalence.json \\
        --cost-usd 0.42 --credits 3.1 \\
        --limitation "..." --out results/issue-136/cloud/result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from landsat_lst.evidence_contract import load_contract, validate_result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--equivalence", type=Path, required=True)
    ap.add_argument("--cost-usd", type=float, required=True)
    ap.add_argument("--credits", type=float, required=True)
    ap.add_argument("--limitation", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    contract = load_contract(args.contract)
    state = json.loads(args.state.read_text())
    equivalence = json.loads(args.equivalence.read_text())
    base_dir = args.contract.parent

    baseline_value = float(contract["baseline"]["value"])
    treatment_value = float(state["peak_rss_mb"])
    effect = (baseline_value - treatment_value) / baseline_value
    worthwhile = effect >= contract["minimum_effect"]["fraction"]
    passed = bool(equivalence["passed"])

    def rel(path: Path) -> str:
        return str(path.resolve().relative_to(base_dir.resolve()))

    result = {
        "schema_version": 1,
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        "environment": "production",
        "input_identity": contract["inputs"]["real_data"]["identity"],
        "baseline_revision": contract["inputs"]["baseline_revision"],
        "treatment_revision": contract["inputs"]["treatment_revision"],
        "baseline_observations": [
            {
                "metric": "peak_rss_mb",
                "value": baseline_value,
                "unit": "MB",
                "artifact": contract["baseline"]["artifact"],
            }
        ],
        "treatment_observations": [
            {
                "metric": "peak_rss_mb",
                "value": treatment_value,
                "unit": "MB",
                "artifact": rel(args.state),
            }
        ],
        "profiling_artifact": rel(args.state),
        "observed_cloud_cost_usd": args.cost_usd,
        "observed_coiled_credits": args.credits,
        "observed_effect_fraction": effect,
        "minimum_effect_met": worthwhile,
        "output_equivalence_passed": passed,
        "decision": "proceed" if worthwhile and passed else "stop",
        "treatment_detail": {
            "instance_type": state.get("instance_type"),
            "instance_lifecycle": state.get("instance_lifecycle"),
            "elapsed_s": state.get("elapsed_s"),
            "phase_seconds": state.get("phase_seconds"),
            "scenes_found": state.get("scenes_found"),
            "scenes_kept": state.get("scenes_kept"),
            "peak_rss_gib": treatment_value / 1024,
            "gate_28_gib_mb": 28 * 1024,
            "under_28_gib": treatment_value <= 28 * 1024,
        },
        "equivalence_summary": {
            "max_abs_diff_dn": equivalence["max_abs_diff"],
            "lst_p95": equivalence["lst_p95"],
            "qa_count_equal": equivalence["qa_count"]["equal"],
        },
        "limitations": args.limitation,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    errors = validate_result(result, contract, contract_path=args.contract)
    print(json.dumps(result, indent=2))
    if errors:
        print("INVALID: " + "; ".join(errors), file=sys.stderr)
        return 1
    print("result manifest valid; decision =", result["decision"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
