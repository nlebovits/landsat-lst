from __future__ import annotations

import json
import os
import subprocess
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
BASELINE_REVISION = "c84448bbac2c95af408ba523521b712b43ba58e8"
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "HOME": "/nonexistent",
}


def git_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A two-commit checkout to launch from: (root, baseline sha, head sha).

    The guard binds a contract to the checkout the command runs *from*, so
    the tests give it one of their own rather than depending on the state
    of the developer's working tree.
    """
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, env={**os.environ, **_GIT_ENV}
        ).strip()

    git("init", "-q")
    (root / "src" / "module.py").write_text("VERSION = 1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "baseline")
    baseline = git("rev-parse", "HEAD")
    (root / "src" / "module.py").write_text("VERSION = 2\n")
    git("commit", "-q", "-am", "treatment")
    return root, baseline, git("rev-parse", "HEAD")


def valid_contract(
    tmp_path: Path,
    command: str = DEFAULT_COMMAND,
    *,
    baseline_revision: str = BASELINE_REVISION,
    treatment_revision: str = CURRENT_REVISION,
) -> dict:
    artifact = tmp_path / "baseline.json"
    artifact.write_text('{"wall_s": 840}\n')
    equivalence = tmp_path / "equivalence.json"
    equivalence.write_text('{"passed": true, "max_abs_diff": 0}\n')
    return {
        "schema_version": 1,
        "claim": "Treatment reduces production shard wall time.",
        "inputs": {
            "workload": "S30W065 band 1",
            "launch_command": command,
            "baseline_revision": baseline_revision,
            "treatment_revision": treatment_revision,
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
            "tolerance": 0,
            "result_artifact": str(equivalence),
        },
    }


def valid_result(tmp_path: Path, contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text())
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
        "baseline_revision": contract["inputs"]["baseline_revision"],
        "treatment_revision": contract["inputs"]["treatment_revision"],
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
        "observed_cloud_cost_usd": 0.5,
        "observed_coiled_credits": 4,
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


def test_binding_requires_both_revisions_to_be_commits(tmp_path: Path) -> None:
    from landsat_lst.evidence_contract import bind_contract_to_repository

    root, _baseline, head = git_repo(tmp_path)
    payload = valid_contract(tmp_path, baseline_revision="f" * 40, treatment_revision=head)
    with pytest.raises(ContractError, match=r"baseline_revision .* is not a commit"):
        bind_contract_to_repository(payload, root)


def test_launch_root_refuses_a_directory_outside_any_checkout(tmp_path: Path) -> None:
    from landsat_lst.evidence_contract import launch_root

    outside = tmp_path / "wheel-install"
    outside.mkdir()
    with pytest.raises(ContractError, match="not inside a git checkout"):
        launch_root(outside)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["minimum_effect"].__setitem__("fraction", 1e-12), "at least 0.05"),
        (lambda c: c["minimum_effect"].__setitem__("fraction", 0.04), "at least 0.05"),
        (
            lambda c: c["inputs"].__setitem__(
                "treatment_revision", c["inputs"]["baseline_revision"]
            ),
            "must differ",
        ),
        (lambda c: c["inputs"].__setitem__("baseline_revision", "0" * 40), "null revision"),
        (lambda c: c["output_equivalence"].pop("tolerance"), "output_equivalence.tolerance"),
        (
            lambda c: c["output_equivalence"].__setitem__("tolerance", -1),
            "output_equivalence.tolerance",
        ),
    ],
)
def test_contract_rejects_unfalsifiable_designs(tmp_path: Path, mutation, message: str) -> None:
    payload = valid_contract(tmp_path)
    mutation(payload)
    assert any(message in error for error in validate_contract(payload)), validate_contract(payload)


def test_contract_and_result_refuse_non_finite_json(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(valid_contract(tmp_path)))
    result_path = tmp_path / "result.json"
    text = json.dumps(valid_result(tmp_path, contract_path)).replace(
        '"value": 756', '"value": Infinity'
    )
    assert "Infinity" in text
    result_path.write_text(text)
    with pytest.raises(ContractError, match="Infinity"):
        load_result(result_path, contract_path)


def test_result_must_carry_exactly_the_registered_observations(tmp_path: Path) -> None:
    """Nine runs against one registered repetition let a median hide four disasters."""
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(valid_contract(tmp_path)))
    payload = valid_result(tmp_path, contract_path)
    extra = dict(payload["treatment_observations"][0])
    payload["treatment_observations"].append(extra)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="exactly 1 observation"):
        load_result(result_path, contract_path)


def test_result_aggregates_the_registered_repetitions(tmp_path: Path) -> None:
    """The mean and median paths, which no fixture exercised before."""
    for aggregation, values, expected_effect in (
        ("median", [700, 756, 9000], (840 - 756) / 840),
        ("mean", [740, 760, 768], (840 - 756) / 840),
    ):
        contract = valid_contract(tmp_path)
        contract["measurement_plan"]["aggregation"] = aggregation
        contract["measurement_plan"]["treatment_repetitions"] = 3
        contract_path = tmp_path / f"contract-{aggregation}.json"
        contract_path.write_text(json.dumps(contract))
        payload = valid_result(tmp_path, contract_path)
        template = payload["treatment_observations"][0]
        payload["treatment_observations"] = [{**template, "value": v} for v in values]
        payload["observed_effect_fraction"] = expected_effect
        result_path = tmp_path / f"result-{aggregation}.json"
        result_path.write_text(json.dumps(payload))
        assert load_result(result_path, contract_path)["decision"] == "proceed"

        payload["observed_effect_fraction"] = expected_effect + 0.01
        result_path.write_text(json.dumps(payload))
        with pytest.raises(ContractError, match="does not match retained observations"):
            load_result(result_path, contract_path)


def test_result_is_checked_against_its_own_cost_caps(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(valid_contract(tmp_path)))
    payload = valid_result(tmp_path, contract_path)
    payload["observed_coiled_credits"] = 21
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="exceeds the pre-registered max_coiled_credits"):
        load_result(result_path, contract_path)

    del payload["observed_coiled_credits"]
    result_path.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="observed_coiled_credits must be"):
        load_result(result_path, contract_path)


def test_equivalence_verdict_is_recomputed_from_max_abs_diff(tmp_path: Path) -> None:
    from landsat_lst.evidence_contract import equivalence_passed

    assert equivalence_passed({"passed": True, "max_abs_diff": 0}, 0) is True
    assert equivalence_passed({"passed": False, "max_abs_diff": 0.5}, 0.1) is False
    with pytest.raises(ContractError, match="contradicts"):
        equivalence_passed({"passed": True, "max_abs_diff": 9999}, 0)
    with pytest.raises(ContractError, match="max_abs_diff"):
        equivalence_passed({"passed": True}, 0)
