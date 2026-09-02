"""Validate pre-registered performance contracts and measured decisions."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

CLASSIFICATIONS = {"measured", "derived", "assumed", "user_reported", "unknown"}
REQUIRED_TEXT = (
    "claim",
    "target_metric",
    "production_discriminator",
    "stop_rule",
    "result_artifact",
)
MAX_CLOUD_COST_USD = 100.0
MAX_COILED_CREDITS = 400.0
TARGET_ENVIRONMENTS = {"production", "production-representative"}
_FULL_REVISION = re.compile(r"[0-9a-f]{40}")
_TILE = re.compile(r"(?<![A-Z0-9])[NS]\d{2}[EW]\d{3}(?![A-Z0-9])", re.IGNORECASE)
_CONTRACT_ASSIGNMENT = re.compile(r"\bLST_EVIDENCE_CONTRACT=(?:'[^']+'|\"[^\"]+\"|[^\s]+)\s*")


class ContractError(ValueError):
    """A performance contract or result is absent, malformed, or unsupported."""


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _required_strings(data: dict[str, Any], names: tuple[str, ...], prefix: str = "") -> list[str]:
    return [f"{prefix}{name} must be non-empty" for name in names if not _nonempty(data.get(name))]


def _resolve(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def sha256_file(path: Path) -> str:
    """Return the digest used to bind result manifests to contracts and artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_inputs(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["inputs must be an object"]
    errors = _required_strings(
        value,
        ("workload", "launch_command", "baseline_revision", "treatment_revision"),
        "inputs.",
    )
    for name in ("baseline_revision", "treatment_revision"):
        revision = value.get(name)
        if _nonempty(revision) and _FULL_REVISION.fullmatch(revision) is None:
            errors.append(f"inputs.{name} must be a full 40-character lowercase hex revision")

    real_data = value.get("real_data")
    if not isinstance(real_data, dict):
        errors.append("inputs.real_data must be an object")
        return errors
    errors.extend(
        _required_strings(
            real_data,
            ("identity", "production_relationship"),
            "inputs.real_data.",
        )
    )
    if real_data.get("kind") != "real":
        errors.append(
            "inputs.real_data.kind must be real; synthetic data cannot gate implementation"
        )
    differences = real_data.get("known_differences")
    if not isinstance(differences, list) or not all(
        isinstance(item, str) and item.strip() for item in differences
    ):
        errors.append("inputs.real_data.known_differences must be a list of non-empty strings")
    return errors


def _validate_baseline(value: Any, base_dir: Path) -> list[str]:
    if not isinstance(value, dict):
        return ["baseline must be an object"]
    errors = _required_strings(value, ("metric", "unit", "artifact"), "baseline.")
    if value.get("classification") != "measured":
        errors.append("baseline.classification must be measured")
    number = value.get("value")
    if not isinstance(number, int | float) or isinstance(number, bool) or number <= 0:
        errors.append("baseline.value must be a positive value")
    artifact = value.get("artifact")
    if _nonempty(artifact) and not _resolve(base_dir, artifact).is_file():
        errors.append(f"baseline.artifact must be an existing file: {_resolve(base_dir, artifact)}")
    return errors


def _validate_effect(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["minimum_effect must be an object"]
    errors = []
    if value.get("direction") not in {"increase", "decrease"}:
        errors.append("minimum_effect.direction must be increase or decrease")
    fraction = value.get("fraction")
    if not isinstance(fraction, int | float) or isinstance(fraction, bool):
        errors.append("minimum_effect.fraction must be numeric")
    elif not 0 < fraction < 1:
        errors.append("minimum_effect.fraction must be between 0 and 1")
    return errors


def _validate_limits(data: dict[str, Any]) -> list[str]:
    errors = []
    limits = {
        "max_cloud_cost_usd": MAX_CLOUD_COST_USD,
        "max_coiled_credits": MAX_COILED_CREDITS,
    }
    for field, ceiling in limits.items():
        value = data.get(field)
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            errors.append(f"{field} must be positive")
        elif value > ceiling:
            errors.append(f"{field} must be at most {ceiling:g}")
    for field in ("code_identity_required", "output_equivalence_required"):
        if data.get(field) is not True:
            errors.append(f"{field} must be true")
    return errors


def _validate_output_equivalence(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["output_equivalence must be an object"]
    return _required_strings(
        value,
        ("method", "acceptance_criterion", "result_artifact"),
        "output_equivalence.",
    )


def _string_list(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        return [f"{field} must be {'a' if allow_empty else 'a non-empty'} list of strings"]
    if not all(isinstance(item, str) and item.strip() for item in value):
        return [f"{field} must contain only non-empty strings"]
    return []


def _validate_measurement_plan(value: Any, target_metric: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["measurement_plan must be an object"]
    errors = _required_strings(
        value,
        ("target_environment", "aggregation"),
        "measurement_plan.",
    )
    if value.get("target_environment") not in TARGET_ENVIRONMENTS:
        errors.append(
            "measurement_plan.target_environment must be production or production-representative"
        )
    for field in ("phases", "metrics", "raw_artifacts"):
        errors.extend(_string_list(value.get(field), f"measurement_plan.{field}"))
    metrics = value.get("metrics")
    if isinstance(metrics, list) and _nonempty(target_metric) and target_metric not in metrics:
        errors.append("measurement_plan.metrics must include target_metric")
    aggregation = value.get("aggregation")
    if aggregation not in {"single", "mean", "median"}:
        errors.append("measurement_plan.aggregation must be single, mean, or median")
    for field in ("baseline_repetitions", "treatment_repetitions"):
        repetitions = value.get(field)
        if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
            errors.append(f"measurement_plan.{field} must be a positive integer")
    if aggregation == "single" and any(
        value.get(field) != 1 for field in ("baseline_repetitions", "treatment_repetitions")
    ):
        errors.append("measurement_plan.single aggregation requires one repetition per arm")

    profiling = value.get("profiling")
    if not isinstance(profiling, dict):
        errors.append("measurement_plan.profiling must be an object")
    else:
        errors.extend(
            _required_strings(
                profiling,
                ("method", "artifact", "observer_effect_control"),
                "measurement_plan.profiling.",
            )
        )
    return errors


def repository_identity(root: Path) -> dict[str, Any]:
    """Return the exact tracked checkout identity used to submit worker code."""

    def git(*args: str) -> str:
        result = subprocess.run(  # nosec B603 B607
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    try:
        return {
            "revision": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"cannot resolve launch code identity in {root}: {exc}") from exc


def bind_contract_to_repository(data: dict[str, Any], root: Path) -> dict[str, Any]:
    """Require the treatment revision to identify the tracked launch checkout."""
    identity = repository_identity(root)
    if identity["dirty"]:
        raise ContractError(
            "launch checkout has tracked changes; commit them before the experiment"
        )
    treatment = data["inputs"]["treatment_revision"]
    if treatment != identity["revision"]:
        raise ContractError(
            f"inputs.treatment_revision does not match launch checkout {identity['revision']}"
        )
    return identity


def normalize_launch_command(command: str) -> str:
    """Canonical command form stored in a contract (excluding its own path)."""
    return " ".join(_CONTRACT_ASSIGNMENT.sub("", command).split())


def _validate_launch(data: dict[str, Any], launch_command: str) -> list[str]:
    inputs = data.get("inputs")
    if not isinstance(inputs, dict):
        return []
    registered = inputs.get("launch_command")
    if _nonempty(registered) and normalize_launch_command(registered) != normalize_launch_command(
        launch_command
    ):
        return ["inputs.launch_command does not match the command being launched"]
    workload = inputs.get("workload")
    command_tiles = {tile.upper() for tile in _TILE.findall(launch_command)}
    workload_tiles = {tile.upper() for tile in _TILE.findall(workload or "")}
    missing = sorted(command_tiles - workload_tiles)
    if missing:
        return [f"inputs.workload must name command tile(s): {', '.join(missing)}"]
    return []


def validate_contract(
    data: Any, *, base_dir: Path | None = None, launch_command: str | None = None
) -> list[str]:
    """Return every actionable contract defect without stopping at the first."""
    if not isinstance(data, dict):
        return ["contract must be a JSON object"]
    errors = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    errors.extend(_required_strings(data, REQUIRED_TEXT))
    errors.extend(_validate_inputs(data.get("inputs")))
    errors.extend(_validate_baseline(data.get("baseline"), base_dir or Path.cwd()))
    errors.extend(_validate_effect(data.get("minimum_effect")))
    errors.extend(_validate_limits(data))
    errors.extend(_validate_output_equivalence(data.get("output_equivalence")))
    errors.extend(
        _validate_measurement_plan(data.get("measurement_plan"), data.get("target_metric"))
    )
    placeholders = (
        "exact tile",
        "full git SHA",
        "smallest target-environment",
        "exact cloud-launch",
        "immutable real input",
        "how this input represents production",
        "named phase",
        "raw machine-readable artifact",
        "profiling or phase-timing method",
        "how profiler overhead",
        "post-run decision",
    )
    if any(placeholder in json.dumps(data) for placeholder in placeholders):
        errors.append("replace every template placeholder before running")
    if launch_command is not None:
        errors.extend(_validate_launch(data, launch_command))
    return errors


def load_contract(path: str | Path, *, launch_command: str | None = None) -> dict[str, Any]:
    """Load and validate one contract, raising a concise ContractError."""
    source = Path(path)
    try:
        data = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read contract {source}: {exc}") from exc
    errors = validate_contract(data, base_dir=source.parent, launch_command=launch_command)
    if errors:
        raise ContractError("invalid performance contract: " + "; ".join(errors))
    return data


def _observations(
    value: Any,
    *,
    name: str,
    base_dir: Path,
    metric: str,
    unit: str,
    minimum: int,
) -> tuple[list[float], list[str]]:
    errors: list[str] = []
    numbers: list[float] = []
    if not isinstance(value, list) or len(value) < minimum:
        return numbers, [f"{name} must contain at least {minimum} observation(s)"]
    for index, observation in enumerate(value):
        prefix = f"{name}[{index}]"
        if not isinstance(observation, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if observation.get("metric") != metric:
            errors.append(f"{prefix}.metric must equal contract target_metric")
        if observation.get("unit") != unit:
            errors.append(f"{prefix}.unit must equal contract baseline.unit")
        number = observation.get("value")
        if not isinstance(number, int | float) or isinstance(number, bool) or number <= 0:
            errors.append(f"{prefix}.value must be positive")
        else:
            numbers.append(float(number))
        artifact = observation.get("artifact")
        if not _nonempty(artifact):
            errors.append(f"{prefix}.artifact must be non-empty")
        elif not _resolve(base_dir, artifact).is_file():
            errors.append(f"{prefix}.artifact must be an existing file")
    return numbers, errors


def _aggregate(values: list[float], method: str) -> float:
    if method == "single":
        return values[0]
    if method == "mean":
        return statistics.fmean(values)
    return float(statistics.median(values))


def _validate_result_identity(
    data: dict[str, Any], contract: dict[str, Any], contract_path: Path
) -> list[str]:
    errors: list[str] = []
    plan = contract["measurement_plan"]
    inputs = contract["inputs"]
    expected = (
        ("schema_version", 1, "result.schema_version must be 1"),
        (
            "contract_sha256",
            sha256_file(contract_path),
            "result.contract_sha256 does not match the contract file",
        ),
        (
            "environment",
            plan["target_environment"],
            "result.environment does not match measurement_plan.target_environment",
        ),
        (
            "input_identity",
            inputs["real_data"]["identity"],
            "result.input_identity does not match inputs.real_data.identity",
        ),
        (
            "baseline_revision",
            inputs["baseline_revision"],
            "result.baseline_revision does not match contract",
        ),
        (
            "treatment_revision",
            inputs["treatment_revision"],
            "result.treatment_revision does not match contract",
        ),
    )
    for field, value, message in expected:
        if data.get(field) != value:
            errors.append(message)
    return errors


def _validate_result_support(
    data: dict[str, Any], plan: dict[str, Any], base_dir: Path
) -> tuple[bool | None, list[str]]:
    errors: list[str] = []
    profile = data.get("profiling_artifact")
    expected_profile = plan["profiling"]["artifact"]
    if not isinstance(profile, str) or not profile.strip():
        errors.append("result.profiling_artifact must be non-empty")
    elif profile != expected_profile:
        errors.append(
            "result.profiling_artifact does not match measurement_plan.profiling.artifact"
        )
    elif not _resolve(base_dir, profile).is_file():
        errors.append("result.profiling_artifact must be an existing file")
    errors.extend(_string_list(data.get("limitations"), "result.limitations"))
    equivalence_passed = data.get("output_equivalence_passed")
    if not isinstance(equivalence_passed, bool):
        errors.append("result.output_equivalence_passed must be boolean")
        return None, errors
    return equivalence_passed, errors


def _effect(baseline_value: float, treatment_value: float, direction: str) -> float:
    if direction == "decrease":
        return (baseline_value - treatment_value) / baseline_value
    return (treatment_value - baseline_value) / baseline_value


def _validate_result_decision(
    data: dict[str, Any],
    contract: dict[str, Any],
    baseline: list[float],
    treatment: list[float],
    equivalence_passed: bool | None,
) -> list[str]:
    if not baseline or not treatment:
        return []
    errors: list[str] = []
    plan = contract["measurement_plan"]
    baseline_value = _aggregate(baseline, plan["aggregation"])
    treatment_value = _aggregate(treatment, plan["aggregation"])
    if not math.isclose(
        baseline_value,
        float(contract["baseline"]["value"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        errors.append("aggregated result baseline does not match contract baseline.value")
    effect = _effect(baseline_value, treatment_value, contract["minimum_effect"]["direction"])
    recorded = data.get("observed_effect_fraction")
    if not isinstance(recorded, int | float) or isinstance(recorded, bool):
        errors.append("result.observed_effect_fraction must be numeric")
    elif not math.isclose(float(recorded), effect, rel_tol=1e-9, abs_tol=1e-9):
        errors.append("result.observed_effect_fraction does not match retained observations")
    worthwhile = effect >= contract["minimum_effect"]["fraction"]
    if data.get("minimum_effect_met") is not worthwhile:
        errors.append("result.minimum_effect_met does not match the recomputed effect")
    expected_decision = "proceed" if worthwhile and equivalence_passed is True else "stop"
    if data.get("decision") != expected_decision:
        errors.append(f"result.decision must be {expected_decision}")
    return errors


def validate_result(
    data: Any,
    contract: dict[str, Any],
    *,
    contract_path: Path,
    base_dir: Path | None = None,
) -> list[str]:
    """Validate a measured decision and recompute its performance conclusion."""
    if not isinstance(data, dict):
        return ["result must be a JSON object"]
    errors = _validate_result_identity(data, contract, contract_path)
    plan = contract["measurement_plan"]
    observation_dir = base_dir or contract_path.parent
    baseline, baseline_errors = _observations(
        data.get("baseline_observations"),
        name="result.baseline_observations",
        base_dir=observation_dir,
        metric=contract["target_metric"],
        unit=contract["baseline"]["unit"],
        minimum=plan["baseline_repetitions"],
    )
    treatment, treatment_errors = _observations(
        data.get("treatment_observations"),
        name="result.treatment_observations",
        base_dir=observation_dir,
        metric=contract["target_metric"],
        unit=contract["baseline"]["unit"],
        minimum=plan["treatment_repetitions"],
    )
    errors.extend(baseline_errors)
    errors.extend(treatment_errors)
    equivalence_passed, support_errors = _validate_result_support(data, plan, observation_dir)
    errors.extend(support_errors)
    errors.extend(
        _validate_result_decision(data, contract, baseline, treatment, equivalence_passed)
    )
    return errors


def load_result(path: str | Path, contract_path: str | Path) -> dict[str, Any]:
    """Load a post-run decision and reject unsupported or arithmetically false claims."""
    source = Path(path)
    contract_source = Path(contract_path)
    contract = load_contract(contract_source)
    try:
        data = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read result {source}: {exc}") from exc
    errors = validate_result(
        data,
        contract,
        contract_path=contract_source,
        base_dir=source.parent,
    )
    if errors:
        raise ContractError("invalid performance result: " + "; ".join(errors))
    return data
