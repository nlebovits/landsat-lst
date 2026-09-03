#!/usr/bin/env python3
"""Enforce the cost-safe STAC endpoint for local and AWS commands."""

from __future__ import annotations

import json
import os
import sys

EARTH_SEARCH_PATTERNS = (
    "earth-search.aws.element84.com",
    "STAC_EARTH_SEARCH",
    "stac_url=STAC_EARTH_SEARCH",
    'stac_url="https://earth-search',
    "stac_url='https://earth-search",
)
PC_PATTERNS = (
    "planetarycomputer.microsoft.com",
    "STAC_PLANETARY_COMPUTER",
    "planetary_computer",
    "import planetary_computer",
)


def is_on_aws() -> bool:
    indicators = (
        "AWS_EXECUTION_ENV",
        "ECS_CONTAINER_METADATA_URI",
        "AWS_BATCH_JOB_ID",
        "COILED_CLUSTER_NAME",
        "COILED_SOFTWARE_NAME",
    )
    return any(os.environ.get(name) for name in indicators)


def deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def check_command(command: str) -> dict | None:
    patterns = PC_PATTERNS if is_on_aws() else EARTH_SEARCH_PATTERNS
    for pattern in patterns:
        if pattern not in command:
            continue
        if is_on_aws():
            return deny(
                f"BLOCKED: Planetary Computer on AWS/Coiled. Use Earth Search. Found: {pattern}"
            )
        return deny(
            f"BLOCKED: Earth Search locally. Use Planetary Computer; Earth Search is for AWS/Coiled. Found: {pattern}"
        )
    return None


def main() -> None:
    payload = json.load(sys.stdin)
    if payload.get("tool_name") != "Bash":
        print("{}")
        return
    result = check_command(payload.get("tool_input", {}).get("command", ""))
    print(json.dumps(result) if result else "{}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps(deny(f"BLOCKED: STAC endpoint guard failed: {exc}")))
