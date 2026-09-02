"""Validate the pre-registered contract required before cloud experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CLASSIFICATIONS = {"measured", "derived", "assumed", "user_reported", "unknown"}
REQUIRED_TEXT = ("claim", "target_metric", "production_discriminator", "stop_rule")


class ContractError(ValueError):
    """A performance contract is absent, malformed, or still a template."""


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _required_strings(data: dict[str, Any], names: tuple[str, ...], prefix: str = "") -> list[str]:
    return [f"{prefix}{name} must be non-empty" for name in names if not _nonempty(data.get(name))]


def _validate_inputs(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["inputs must be an object"]
    return _required_strings(
        value, ("workload", "baseline_revision", "treatment_revision"), "inputs."
    )


def _validate_baseline(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["baseline must be an object"]
    errors = _required_strings(value, ("metric", "unit", "artifact"), "baseline.")
    if value.get("classification") not in CLASSIFICATIONS:
        errors.append("baseline.classification is invalid")
    number = value.get("value")
    if not isinstance(number, int | float) or isinstance(number, bool) or number <= 0:
        errors.append("baseline.value must be a positive value")
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
    for field in ("max_cloud_cost_usd", "max_coiled_credits"):
        value = data.get(field)
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            errors.append(f"{field} must be positive")
    for field in ("code_identity_required", "output_equivalence_required"):
        if data.get(field) is not True:
            errors.append(f"{field} must be true")
    return errors


def validate_contract(data: Any) -> list[str]:
    """Return every actionable contract defect without stopping at the first."""
    if not isinstance(data, dict):
        return ["contract must be a JSON object"]
    errors = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    errors.extend(_required_strings(data, REQUIRED_TEXT))
    errors.extend(_validate_inputs(data.get("inputs")))
    errors.extend(_validate_baseline(data.get("baseline")))
    errors.extend(_validate_effect(data.get("minimum_effect")))
    errors.extend(_validate_limits(data))
    placeholders = ("exact tile", "full git SHA", "smallest target-environment")
    if any(placeholder in json.dumps(data) for placeholder in placeholders):
        errors.append("replace every template placeholder before running")
    return errors


def load_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate one contract, raising a concise ContractError."""
    source = Path(path)
    try:
        data = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read contract {source}: {exc}") from exc
    errors = validate_contract(data)
    if errors:
        raise ContractError("invalid performance contract: " + "; ".join(errors))
    return data
