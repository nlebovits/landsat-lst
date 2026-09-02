#!/usr/bin/env python3
"""Enforce the performance-evidence declaration on every human pull request."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from landsat_lst.evidence import validate_evidence_bundle
from landsat_lst.evidence_contract import ContractError, load_contract, load_result

ROOT = Path(__file__).resolve().parents[1]
_DECLARATION = re.compile(
    r"<!--\s*performance-evidence\s+"
    r"stage:\s*(?P<stage>[^\n]+)\s+"
    r"contract:\s*(?P<contract>[^\n]+)\s+"
    r"evidence:\s*(?P<evidence>[^\n]+)\s*-->",
    re.IGNORECASE,
)
_PERFORMANCE_CLAIM = re.compile(
    r"\b(?:optimi[sz](?:e|ed|es|ing|ation)|speed-?up|faster|benchmark(?:ing)?|"
    r"profil(?:e|ed|es|ing)|bottleneck|throughput|"
    r"(?:reduce|reduced|reduces|improve|improved|improves)\s+"
    r"(?:runtime|latency|memory|cost|I/O|io))\b",
    re.IGNORECASE,
)
_STAGES = {"none", "governance", "instrumentation", "measurement", "optimization"}


def _repo_path(value: str, *, root: Path, field: str) -> tuple[Path | None, list[str]]:
    if value.lower() == "n/a":
        return None, [f"{field} must name a committed repository-relative JSON file"]
    candidate = Path(value)
    if candidate.is_absolute():
        return None, [f"{field} must be repository-relative"]
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        return None, [f"{field} does not resolve to a committed file: {value}"]
    return resolved, []


def validate_pr(  # noqa: PLR0911, PLR0912, PLR0915 - explicit stage gates
    payload: dict[str, Any], *, root: Path = ROOT
) -> list[str]:
    """Return actionable PR declaration and evidence failures."""
    pull = payload.get("pull_request")
    if not isinstance(pull, dict):
        return []
    user = pull.get("user")
    login = user.get("login", "") if isinstance(user, dict) else ""
    if login == "dependabot[bot]":
        return []

    title = pull.get("title") if isinstance(pull.get("title"), str) else ""
    body = pull.get("body") if isinstance(pull.get("body"), str) else ""
    matches = list(_DECLARATION.finditer(body))
    if len(matches) != 1:
        return ["PR body must contain exactly one performance-evidence declaration"]

    declaration = matches[0].groupdict()
    stage = declaration["stage"].strip().lower()
    contract_name = declaration["contract"].strip()
    evidence_name = declaration["evidence"].strip()
    if stage not in _STAGES:
        return [f"performance-evidence stage must be one of: {', '.join(sorted(_STAGES))}"]

    visible_body = _DECLARATION.sub("", body)
    if stage == "none" and _PERFORMANCE_CLAIM.search(f"{title}\n{visible_body}"):
        return ["performance language requires a non-none performance-evidence stage"]
    if stage in {"none", "governance"}:
        errors = []
        if contract_name.lower() != "n/a":
            errors.append(f"{stage} PR must declare contract: n/a")
        if evidence_name.lower() != "n/a":
            errors.append(f"{stage} PR must declare evidence: n/a")
        return errors

    contract_path, errors = _repo_path(contract_name, root=root, field="contract")
    if contract_path is not None:
        try:
            contract = load_contract(contract_path)
        except ContractError as exc:
            errors.append(str(exc))
            contract = None
    else:
        contract = None

    if stage == "instrumentation":
        if evidence_name.lower() != "n/a":
            errors.append("instrumentation PR must declare evidence: n/a")
        return errors

    evidence_path, evidence_errors = _repo_path(evidence_name, root=root, field="evidence")
    errors.extend(evidence_errors)
    result = None
    if contract_path is not None and contract is not None:
        result_value = contract["result_artifact"]
        result_path = Path(result_value)
        if not result_path.is_absolute():
            result_path = contract_path.parent / result_path
        try:
            result = load_result(result_path, contract_path)
        except ContractError as exc:
            errors.append(str(exc))
    if evidence_path is not None:
        try:
            bundle = validate_evidence_bundle(
                evidence_path,
                require_proceed=stage == "optimization",
            )
        except ContractError as exc:
            errors.append(str(exc))
        else:
            if contract is not None and bundle.get("contract") != contract:
                errors.append("evidence bundle contract does not match the declared contract")
            if result is not None and bundle.get("decision") != result:
                errors.append("evidence bundle decision does not match the declared result")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path)
    args = parser.parse_args()
    event_name = os.environ.get("GITHUB_EVENT_NAME")
    source = args.event or (
        Path(os.environ["GITHUB_EVENT_PATH"])
        if event_name == "pull_request" and os.environ.get("GITHUB_EVENT_PATH")
        else None
    )
    if source is None:
        print("Performance PR declaration check skipped outside pull_request.")
        return 0
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read pull-request event: {exc}")
        return 1
    failures = validate_pr(payload)
    for failure in failures:
        print(f"ERROR: {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
