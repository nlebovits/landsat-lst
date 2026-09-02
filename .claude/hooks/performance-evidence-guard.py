#!/usr/bin/env python3
"""Deny cloud experiments without a valid pre-registered evidence contract."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from landsat_lst.evidence_contract import load_contract  # noqa: E402

LAUNCH_PATTERNS = (
    re.compile(r"\bcoiled\.(?:batch_run|Cluster)\s*\("),
    re.compile(r"\blandsat-lst\s+(?:process|benchmark|shard\s+process)\b.*\s--distributed\b"),
    re.compile(r"\b(?:python\s+)?scripts/probe_[^\s]+\.py\b.*\s--launch\b"),
)


def deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def contract_path(command: str) -> str | None:
    inline = re.search(r"\bLST_EVIDENCE_CONTRACT=(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))", command)
    if inline:
        return next(value for value in inline.groups() if value is not None)
    return os.environ.get("LST_EVIDENCE_CONTRACT")


def main() -> None:
    payload = json.load(sys.stdin)
    if payload.get("tool_name") != "Bash":
        print("{}")
        return
    command = payload.get("tool_input", {}).get("command", "")
    if not any(pattern.search(command) for pattern in LAUNCH_PATTERNS):
        print("{}")
        return
    contract = contract_path(command)
    if not contract:
        print(
            json.dumps(
                deny(
                    "Cloud experiment blocked: set LST_EVIDENCE_CONTRACT to a valid pre-registered JSON contract. The contract does not replace explicit operator approval."
                )
            )
        )
        return
    try:
        path = Path(contract)
        load_contract(path if path.is_absolute() else ROOT / path)
    except (OSError, ValueError) as exc:
        print(json.dumps(deny(f"Cloud experiment blocked: invalid evidence contract: {exc}")))
        return
    print("{}")


if __name__ == "__main__":
    main()
