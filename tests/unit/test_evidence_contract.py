from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from landsat_lst.evidence_contract import (
    ContractError,
    load_contract,
    load_result,
    sha256_file,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMMAND = "python scripts/probe_x.py --launch"
CURRENT_REVISION = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
).strip()


def valid_contract(tmp_path: Path, command: str = DEFAULT_COMMAND) -> dict:
    artifact = tmp_path / "baseline.json"
    artifact.write_text('{"wall_s": 840}\n')
    equivalence = tmp_path / "equivalence.json"
    equivalence.write_text('{"passed": true}\n')
    return {
        "schema_version": 1,
        "claim": "Treatment reduces production shard wall time.",
        "inputs": {
            "workload": "S30W065 band 1",
            "launch_command": command,
            "baseline_revision": "c84448bbac2c95af408ba523521b712b43ba58e8",
            "treatment_revision": CURRENT_REVISION,
            "real_data": {
                "kind": "real",
                "identity": "S30W065/2021-2025/composite-band-1/scenes-sha256",
                "production_relationship": "Exact production shard and immutable scene set.",
                "known_differences": ["Single shard rather than the fleet."],
            },
        },
        "baseline": {
            "metric": "wall_s",
            "value": 840,
            "unit": "seconds",
            "classification": "measured",
            "artifact": str(artifact),
        },
        "target_metric": "wall_s",
        "measurement_plan": {
            "target_environment": "production",
            "phases": ["remote read", "reduction", "compression and store"],
            "metrics": ["wall_s", "cpu_s", "peak_rss_mb", "bytes_read"],
            "raw_artifacts": ["heartbeat JSON", "Coiled CPU and network series"],
            "profiling": {
                "method": "Dask task-prefix and host-resource profile",
                "artifact": "profile.json",
                "observer_effect_control": "Identical profiler configuration in both arms.",
            },
            "baseline_repetitions": 1,
            "treatment_repetitions": 1,
            "aggregation": "single",
        },
        "minimum_effect": {"direction": "decrease", "fraction": 0.1},
        "production_discriminator": "One sequential production-shard A/B.",
        "stop_rule": "Reject below ten percent.",
        "result_artifact": "result.json",
        "max_cloud_cost_usd": 1,
        "max_coiled_credits": 20,
        "code_identity_required": True,
        "output_equivalence_required": True,
        "output_equivalence": {
            "method": "SHA-256",
            "acceptance_criterion": "both output checksums match",
            "result_artifact": str(equivalence),
        },
    }


def valid_result(tmp_path: Path, contract_path: Path) -> dict:
    baseline_raw = tmp_path / "baseline-observation.json"
    treatment_raw = tmp_path / "treatment-observation.json"
    profile = tmp_path / "profile.json"
    baseline_raw.write_text('{"wall_s": 840}\n')
    treatment_raw.write_text('{"wall_s": 756}\n')
    profile.write_text('{"tasks": {"open_rasterio": 200}}\n')
    return {
        "schema_version": 1,
        "contract_sha256": sha256_file(contract_path),
        "environment": "production",
        "input_identity": "S30W065/2021-2025/composite-band-1/scenes-sha256",
        "baseline_revision": "c84448bbac2c95af408ba523521b712b43ba58e8",
        "treatment_revision": CURRENT_REVISION,
        "baseline_observations": [
            {
                "metric": "wall_s",
                "value": 840,
                "unit": "seconds",
                "artifact": baseline_raw.name,
            }
        ],
        "treatment_observations": [
            {
                "metric": "wall_s",
                "value": 756,
                "unit": "seconds",
                "artifact": treatment_raw.name,
            }
        ],
        "profiling_artifact": profile.name,
        "observed_effect_fraction": 0.1,
        "minimum_effect_met": True,
        "output_equivalence_passed": True,
        "decision": "proceed",
        "limitations": ["One production shard; fleet-level variance remains unknown."],
    }


def test_template_is_deliberately_not_runnable() -> None:
    template = json.loads(
        (ROOT / "docs/templates/performance-experiment-contract.json").read_text()
    )
    assert validate_contract(template)


def test_contract_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(valid_contract(tmp_path)))
    assert load_contract(path)["baseline"]["value"] == 840


def test_invalid_contract_reports_defect(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text("{}")
    with pytest.raises(ContractError, match="schema_version"):
        load_contract(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda contract: contract["inputs"].update(baseline_revision="a" * 39 + "z"),
            "40-character",
        ),
        (lambda contract: contract["baseline"].update(artifact="missing.json"), "existing file"),
        (lambda contract: contract.update(max_cloud_cost_usd=1e9), "at most 100"),
        (lambda contract: contract.update(max_coiled_credits=1e9), "at most 400"),
        (
            lambda contract: contract["output_equivalence"].update(method=""),
            "output_equivalence.method",
        ),
    ],
)
def test_contract_rejects_non_binding_fields(tmp_path: Path, mutation, message: str) -> None:
    contract = valid_contract(tmp_path)
    mutation(contract)
    assert any(message in error for error in validate_contract(contract, base_dir=tmp_path))


def test_contract_is_bound_to_exact_command_and_tile(tmp_path: Path) -> None:
    command = "landsat-lst shard process --tile S30W065"
    contract = valid_contract(tmp_path, command)
    assert not validate_contract(contract, base_dir=tmp_path, launch_command=command)
    assert any(
        "does not match" in error
        for error in validate_contract(
            contract, base_dir=tmp_path, launch_command="landsat-lst shard resume run N40W075"
        )
    )
    contract["inputs"]["launch_command"] = "landsat-lst shard process --tile N40W075"
    assert any(
        "workload must name" in error
        for error in validate_contract(
            contract,
            base_dir=tmp_path,
            launch_command="landsat-lst shard process --tile N40W075",
        )
    )


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


@pytest.mark.parametrize(
    "command",
    [
        "landsat-lst shard process --tile N40W075",
        "landsat-lst shard resume run-1 N40W075",
        "coiled batch run --region us-west-2 python x.py",
        "python -m landsat_lst.cli process --tile N40W075 --distributed",
        "python scripts/probe_x.py --launch",
        "uv run -- landsat-lst shard process --tile N40W075",
        "env landsat-lst shard process --tile N40W075",
        "uv run --python 3.12 landsat-lst shard process --tile N40W075",
        "env -u UNUSED landsat-lst shard process --tile N40W075",
        "./.venv/bin/landsat-lst shard process --tile N40W075",
        "command landsat-lst shard process --tile N40W075",
        "(landsat-lst shard process --tile N40W075)",
        'bash -lc "landsat-lst shard process --tile N40W075"',
    ],
)
def test_cloud_guard_blocks_real_launch_forms_without_contract(command: str) -> None:
    result = run_hook("performance-evidence-guard.py", command)
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_cloud_guard_accepts_only_the_registered_launch(tmp_path: Path) -> None:
    command = "landsat-lst shard process --tile S30W065"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(valid_contract(tmp_path, command)))
    assert (
        run_hook("performance-evidence-guard.py", command, {"LST_EVIDENCE_CONTRACT": str(path)})
        == {}
    )
    result = run_hook(
        "performance-evidence-guard.py",
        "landsat-lst shard process --tile N40W075",
        {"LST_EVIDENCE_CONTRACT": str(path)},
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_cloud_guard_rejects_contract_for_another_revision(tmp_path: Path) -> None:
    command = "landsat-lst shard process --tile S30W065"
    payload = valid_contract(tmp_path, command)
    payload["inputs"]["treatment_revision"] = "e" * 40
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload))

    result = run_hook(
        "performance-evidence-guard.py", command, {"LST_EVIDENCE_CONTRACT": str(path)}
    )
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "treatment_revision" in result["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.parametrize(
    "command",
    [
        "coiled batch status 123",
        'grep -rn "coiled.batch_run(" src/',
        'rg "landsat-lst shard process" docs/',
        "command -v landsat-lst",
    ],
)
def test_cloud_guard_allows_read_only_commands(command: str) -> None:
    assert run_hook("performance-evidence-guard.py", command) == {}


@pytest.mark.parametrize("name", ["performance-evidence-guard.py", "stac-endpoint-guard.py"])
def test_guards_fail_closed_on_malformed_input(name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / ".claude/hooks" / name)],
        input="not json",
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    payload = json.loads(result.stdout)
    assert payload["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_stac_guard_uses_current_hook_schema() -> None:
    result = run_hook("stac-endpoint-guard.py", "curl https://earth-search.aws.element84.com/v1")
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_policy_wiring_check() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_evidence_policy.py")], cwd=ROOT, check=False
    )
    assert result.returncode == 0


def test_result_recomputes_the_decision_from_retained_observations(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(valid_contract(tmp_path)))
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(valid_result(tmp_path, contract_path)))

    assert load_result(result_path, contract_path)["decision"] == "proceed"

    payload = json.loads(result_path.read_text())
    payload["observed_effect_fraction"] = 0.5
    result_path.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="does not match retained observations"):
        load_result(result_path, contract_path)


def test_result_cannot_override_a_failed_stop_rule(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(valid_contract(tmp_path)))
    payload = valid_result(tmp_path, contract_path)
    payload["treatment_observations"][0]["value"] = 820
    payload["observed_effect_fraction"] = (840 - 820) / 840
    payload["minimum_effect_met"] = False
    payload["decision"] = "proceed"
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(payload))

    with pytest.raises(ContractError, match="decision must be stop"):
        load_result(result_path, contract_path)


def test_contract_rejects_synthetic_gate_data(tmp_path: Path) -> None:
    payload = valid_contract(tmp_path)
    payload["inputs"]["real_data"]["kind"] = "synthetic"
    assert any("synthetic data cannot gate" in error for error in validate_contract(payload))
