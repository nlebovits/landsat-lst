#!/usr/bin/env python3
"""Deny cloud experiments without a valid, launch-bound evidence contract."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Match command segments at their executable, not arbitrary source text inside
# arguments. The old ``coiled.batch_run(`` search denied ``grep`` while the
# production shard CLI passed straight through.
_SEGMENT = r"(?:^|[\n;&|]+)\s*"
_PREFIX = (
    r"(?:(?:[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|\"[^\"]*\"|[^\s]+))\s+)*(?:(?:uv|poetry)\s+run\s+)?"
)
_PYTHON = r"python(?:3(?:\.\d+)?)?"
LAUNCH_PATTERNS = (
    re.compile(_SEGMENT + _PREFIX + r"landsat-lst\s+process\b[^\n;&|]*\s--distributed\b"),
    re.compile(_SEGMENT + _PREFIX + r"landsat-lst\s+benchmark\b"),
    re.compile(_SEGMENT + _PREFIX + r"landsat-lst\s+shard\s+(?:process|resume)\b"),
    re.compile(_SEGMENT + _PREFIX + r"coiled\s+batch\s+run\b"),
    re.compile(
        _SEGMENT
        + _PREFIX
        + _PYTHON
        + r"\s+-m\s+landsat_lst\.cli\s+(?:process\b[^\n;&|]*\s--distributed\b|benchmark\b|shard\s+(?:process|resume)\b)"
    ),
    re.compile(
        _SEGMENT + _PREFIX + _PYTHON + r"\s+scripts/probe_[^\s]+\.py\b[^\n;&|]*\s--launch\b"
    ),
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
        sys.path.insert(0, str(ROOT / "src"))
        from landsat_lst.evidence_contract import load_contract  # noqa: PLC0415

        path = Path(contract)
        load_contract(path if path.is_absolute() else ROOT / path, launch_command=command)
    except (OSError, ValueError) as exc:
        print(json.dumps(deny(f"Cloud experiment blocked: invalid evidence contract: {exc}")))
        return
    print("{}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Hook failures are advisory in Claude Code. Emit a deny rather than a
        # traceback so malformed input or an import regression cannot fail open.
        print(json.dumps(deny(f"Cloud experiment blocked: evidence guard failed: {exc}")))
