#!/usr/bin/env python3
"""Fail CI when the always-on performance-evidence safeguards drift."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    failures: list[str] = []
    required = {
        "CLAUDE.md": "docs/performance-evidence-policy.md",
        "docs/performance-evidence-policy.md": "## Required order",
    }
    for name, marker in required.items():
        path = ROOT / name
        if not path.is_file() or marker not in path.read_text():
            failures.append(f"{name}: missing {marker!r}")
    settings = json.loads((ROOT / ".claude/settings.json").read_text())
    commands = [
        hook["command"]
        for groups in settings.get("hooks", {}).values()
        for group in groups
        for hook in group.get("hooks", [])
        if hook.get("type") == "command"
    ]
    for name in ("performance-evidence-context.py", "performance-evidence-guard.py"):
        if not any(name in command for command in commands):
            failures.append(f".claude/settings.json: {name} is not wired")
    if not (ROOT / "docs/templates/performance-experiment-contract.json").is_file():
        failures.append("missing experiment-contract template")
    for failure in failures:
        print(f"ERROR: {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
