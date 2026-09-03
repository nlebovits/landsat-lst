#!/usr/bin/env python3
"""Enforce the performance-evidence declaration on every human pull request.

Runs from the *base* branch's checkout under ``pull_request_target`` (see
``.github/workflows/evidence-gate.yml``), with the pull request's own tree
checked out beside it and passed as ``--root``. The gate and its validators
therefore come from ``main``; a pull request cannot edit the gate that judges
it in the same diff.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
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
#: Words that make a performance claim on their own, and verb-object pairs
#: that make one across a clause. The pairs allow up to forty characters
#: between verb and object so "reduces the runtime by 40%" is a claim. A
#: sentence about memory in this repository is a claim about the product.
_PERFORMANCE_CLAIM = re.compile(
    r"\b(?:"
    r"optimi[sz](?:e|ed|es|ing|ation|ations)|"
    r"speed-?ups?|faster|slower|quicker|accelerat(?:e|ed|es|ing|ion)|"
    r"benchmark(?:s|ed|ing)?|profil(?:e|ed|es|ing|er)|bottlenecks?|"
    r"throughput|latency|runtime|wall[- ]?(?:clock|time)|"
    r"perf|performance|efficien(?:t|cy)|"
    r"peak\s+(?:rss|memory)|rss|oom|out[- ]of[- ]memory|"
    r"task\s+counts?|fewer\s+tasks|"
    r"credits?|cheaper|cost(?:s|ly|lier)?|"
    r"vectori[sz](?:e|ed|es|ing)|paralleli[sz](?:e|ed|es|ing)|concurren(?:t|cy)|"
    r"scal(?:es|ed|ing|ability)|"
    r"(?:reduc|improv|cut|halv|lower|shrink|trim)(?:e|ed|es|ing)?\b.{0,40}?\b"
    r"(?:runtime|latency|memory|cost|time|i/o|io|reads?|bytes)"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)
#: Characters that render as Latin letters but are not. Mapped before the
#: scan so a Cyrillic small o inside "optimization" cannot dodge the regex.
_CONFUSABLES = str.maketrans(
    {
        # Cyrillic
        "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
        "\u0445": "x", "\u0443": "y", "\u0456": "i", "\u0458": "j", "\u0455": "s",
        "\u04bb": "h", "\u0501": "d",
        "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M",
        "\u041d": "H", "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T",
        "\u0425": "X", "\u0405": "S", "\u0406": "I", "\u0408": "J",
        # Greek
        "\u03bf": "o", "\u03b1": "a", "\u03b5": "e", "\u03b9": "i", "\u03ba": "k",
        "\u03bd": "v", "\u03c1": "p", "\u03c4": "t", "\u03c5": "u",
        "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z", "\u0397": "H",
        "\u0399": "I", "\u039a": "K", "\u039c": "M", "\u039d": "N", "\u039f": "O",
        "\u03a1": "P", "\u03a4": "T", "\u03a5": "Y", "\u03a7": "X",
        # Latin script g, two Armenian letters
        "\u0261": "g", "\u0578": "n", "\u057d": "u",
    }
)  # fmt: skip
#: Soft hyphen, zero-width space/joiners, directional marks, word joiner, BOM.
_INVISIBLE = re.compile("[\u00ad\u200b\u200c\u200d\u200e\u200f\u2060\ufeff]")
_STAGES = {"none", "governance", "instrumentation", "measurement", "optimization"}
_EXEMPT_LOGINS = {"dependabot[bot]"}
_EVENTS = {"pull_request", "pull_request_target"}


def normalize_prose(text: str) -> str:
    """Fold homoglyphs and invisible characters so the claim scan sees plain ASCII."""
    folded = unicodedata.normalize("NFKC", text).translate(_CONFUSABLES)
    return _INVISIBLE.sub("", folded)


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


def _describe(exc: BaseException) -> str:
    if isinstance(exc, ContractError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def validate_pr(  # noqa: PLR0911, PLR0912, PLR0915 - explicit stage gates
    payload: dict[str, Any], *, root: Path = ROOT
) -> list[str]:
    """Return actionable PR declaration and evidence failures."""
    pull = payload.get("pull_request")
    if not isinstance(pull, dict):
        return []
    user = pull.get("user")
    login = user.get("login", "") if isinstance(user, dict) else ""
    if login in _EXEMPT_LOGINS:
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
    prose = normalize_prose(f"{title}\n{visible_body}")
    if stage == "none" and (claim := _PERFORMANCE_CLAIM.search(prose)):
        return [
            "performance language requires a non-none performance-evidence stage "
            f"(matched {claim.group(0)!r})"
        ]
    if stage in {"none", "governance"}:
        errors = []
        if contract_name.lower() != "n/a":
            errors.append(f"{stage} PR must declare contract: n/a")
        if evidence_name.lower() != "n/a":
            errors.append(f"{stage} PR must declare evidence: n/a")
        return errors

    contract_path, errors = _repo_path(contract_name, root=root, field="contract")
    contract = None
    if contract_path is not None:
        try:
            contract = load_contract(contract_path)
        except Exception as exc:
            errors.append(f"contract is invalid: {_describe(exc)}")

    if stage == "instrumentation":
        if evidence_name.lower() != "n/a":
            errors.append("instrumentation PR must declare evidence: n/a")
        return errors

    evidence_path, evidence_errors = _repo_path(evidence_name, root=root, field="evidence")
    errors.extend(evidence_errors)
    result = None
    if contract_path is not None and contract is not None:
        result_path = Path(contract["result_artifact"])
        if not result_path.is_absolute():
            result_path = contract_path.parent / result_path
        try:
            result = load_result(result_path, contract_path)
        except Exception as exc:
            errors.append(f"result is invalid: {_describe(exc)}")
    if evidence_path is not None:
        try:
            bundle = validate_evidence_bundle(
                evidence_path,
                require_proceed=stage == "optimization",
            )
        except Exception as exc:
            errors.append(f"evidence bundle is invalid: {_describe(exc)}")
        else:
            if contract is not None and bundle.get("contract") != contract:
                errors.append("evidence bundle contract does not match the declared contract")
            if result is not None and bundle.get("decision") != result:
                errors.append("evidence bundle decision does not match the declared result")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path, help="pull-request event JSON (default: CI's)")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="checkout of the pull request's tree, where contract and evidence paths resolve",
    )
    args = parser.parse_args(argv)
    event_name = os.environ.get("GITHUB_EVENT_NAME")
    source = args.event or (
        Path(os.environ["GITHUB_EVENT_PATH"])
        if event_name in _EVENTS and os.environ.get("GITHUB_EVENT_PATH")
        else None
    )
    if source is None:
        print("Performance PR declaration check skipped outside a pull-request event.")
        return 0
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read pull-request event: {exc}")
        return 1
    try:
        failures = validate_pr(payload, root=args.root)
    except Exception as exc:
        failures = [f"declaration check crashed: {_describe(exc)}"]
    for failure in failures:
        print(f"ERROR: {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
