"""Tests for the STAC endpoint guard hook.

The guard denies an Earth Search call from a local session, where the endpoint
costs egress and buys nothing. See the STAC endpoint rules in ``CLAUDE.md``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOOK = "stac-endpoint-guard.py"


def run_hook(name: str, command: str, env: dict[str, str] | None = None, cwd: Path = ROOT) -> dict:
    payload = {"tool_name": "Bash", "tool_input": {"command": command}, "cwd": str(cwd)}
    clean_env = dict(os.environ)
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


def test_guard_fails_closed_on_malformed_input() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / ".claude/hooks" / HOOK)],
        input="not json",
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    payload = json.loads(result.stdout)
    assert payload["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_guard_uses_current_hook_schema() -> None:
    result = run_hook(HOOK, "curl https://earth-search.aws.element84.com/v1")
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
