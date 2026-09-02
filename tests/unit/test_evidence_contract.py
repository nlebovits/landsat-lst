from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from landsat_lst.evidence_contract import ContractError, load_contract, validate_contract

ROOT = Path(__file__).resolve().parents[2]


def valid_contract() -> dict:
    return {
        "schema_version": 1,
        "claim": "Treatment reduces production shard wall time.",
        "inputs": {
            "workload": "S30W065 band 1",
            "baseline_revision": "a" * 40,
            "treatment_revision": "b" * 40,
        },
        "baseline": {
            "metric": "wall_s",
            "value": 840,
            "unit": "seconds",
            "classification": "measured",
            "artifact": "results/baseline.json",
        },
        "target_metric": "wall_s",
        "minimum_effect": {"direction": "decrease", "fraction": 0.1},
        "production_discriminator": "One sequential production-shard A/B.",
        "stop_rule": "Reject below ten percent.",
        "max_cloud_cost_usd": 1,
        "max_coiled_credits": 20,
        "code_identity_required": True,
        "output_equivalence_required": True,
    }


def test_template_is_deliberately_not_runnable() -> None:
    template = json.loads(
        (ROOT / "docs/templates/performance-experiment-contract.json").read_text()
    )
    assert validate_contract(template)


def test_contract_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(valid_contract()))
    assert load_contract(path)["baseline"]["value"] == 840


def test_invalid_contract_reports_defect(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text("{}")
    with pytest.raises(ContractError, match="schema_version"):
        load_contract(path)


def run_hook(name: str, command: str, env: dict[str, str] | None = None) -> dict:
    payload = {"tool_name": "Bash", "tool_input": {"command": command}}
    clean_env = dict(os.environ)
    clean_env.pop("LST_EVIDENCE_CONTRACT", None)
    if env:
        clean_env.update(env)
    result = subprocess.run(
        [sys.executable, str(ROOT / ".claude/hooks" / name)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
        env=clean_env,
    )
    return json.loads(result.stdout)


def test_cloud_guard_blocks_without_contract() -> None:
    result = run_hook("performance-evidence-guard.py", "python scripts/probe_x.py --launch")
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_cloud_guard_accepts_valid_contract(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(valid_contract()))
    assert (
        run_hook(
            "performance-evidence-guard.py",
            "python scripts/probe_x.py --launch",
            {"LST_EVIDENCE_CONTRACT": str(path)},
        )
        == {}
    )


def test_cloud_guard_allows_read_only_status() -> None:
    assert run_hook("performance-evidence-guard.py", "coiled batch status 123") == {}


def test_stac_guard_uses_current_hook_schema() -> None:
    result = run_hook("stac-endpoint-guard.py", "curl https://earth-search.aws.element84.com/v1")
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_policy_wiring_check() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_evidence_policy.py")], cwd=ROOT, check=False
    )
    assert result.returncode == 0
