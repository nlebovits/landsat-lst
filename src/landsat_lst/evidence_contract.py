"""Validate the pre-registered contract required before cloud experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CLASSIFICATIONS = {"measured", "derived", "assumed", "user_reported", "unknown"}
REQUIRED_TEXT = ("claim", "target_metric", "production_discriminator", "stop_rule")
MAX_CLOUD_COST_USD = 100.0
MAX_COILED_CREDITS = 400.0
_FULL_REVISION = re.compile(r"[0-9a-f]{40}")
_TILE = re.compile(r"(?<![A-Z0-9])[NS]\d{2}[EW]\d{3}(?![A-Z0-9])", re.IGNORECASE)
_CONTRACT_ASSIGNMENT = re.compile(r"\bLST_EVIDENCE_CONTRACT=(?:'[^']+'|\"[^\"]+\"|[^\s]+)\s*")


class ContractError(ValueError):
    """A performance contract is absent, malformed, or still a template."""


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _required_strings(data: dict[str, Any], names: tuple[str, ...], prefix: str = "") -> list[str]:
    return [f"{prefix}{name} must be non-empty" for name in names if not _nonempty(data.get(name))]


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
    return errors


def _validate_baseline(value: Any, base_dir: Path) -> list[str]:
    if not isinstance(value, dict):
        return ["baseline must be an object"]
    errors = _required_strings(value, ("metric", "unit", "artifact"), "baseline.")
    if value.get("classification") not in CLASSIFICATIONS:
        errors.append("baseline.classification is invalid")
    number = value.get("value")
    if not isinstance(number, int | float) or isinstance(number, bool) or number <= 0:
        errors.append("baseline.value must be a positive value")
    artifact = value.get("artifact")
    if _nonempty(artifact):
        path = Path(artifact)
        if not path.is_absolute():
            path = base_dir / path
        if not path.is_file():
            errors.append(f"baseline.artifact must be an existing file: {path}")
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
    placeholders = (
        "exact tile",
        "full git SHA",
        "smallest target-environment",
        "exact cloud-launch",
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
