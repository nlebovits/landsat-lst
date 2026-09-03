#!/usr/bin/env python3
"""Fail CI when mandatory performance-evidence safeguards drift.

Each check asserts something that has to be true for the gate to *run*, not
merely for a phrase to be present: the gate script is executed by the
pull_request_target workflow as an uncommented run step, the hooks are wired
by their real paths, the instruction surfaces point at each other.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE_WORKFLOW = ".github/workflows/evidence-gate.yml"
GATE_SCRIPT = "scripts/check_performance_pr.py"


def _run_steps(workflow_text: str) -> list[str]:
    """Every uncommented ``run:`` line of a workflow, whitespace-collapsed."""
    steps = []
    for raw in workflow_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("run:"):
            steps.append(" ".join(line[len("run:") :].split()))
    return steps


def main() -> int:  # noqa: PLR0912 - one branch per safeguard
    failures: list[str] = []
    required = {
        "AGENTS.override.md": "validated evidence bundle records",
        "CLAUDE.md": "## Mandatory performance-evidence gate",
        "docs/performance-evidence-policy.md": "## Required lifecycle",
        ".github/pull_request_template.md": "<!-- performance-evidence",
        GATE_SCRIPT: 'require_proceed=stage == "optimization"',
    }
    for name, marker in required.items():
        path = ROOT / name
        if not path.is_file() or marker not in path.read_text():
            failures.append(f"{name}: missing {marker!r}")

    # Codex reads AGENTS.md, which claude-mem regenerates; the tracked rules
    # live in AGENTS.override.md and CLAUDE.md must send every reader there.
    claude_md = (ROOT / "CLAUDE.md").read_text() if (ROOT / "CLAUDE.md").is_file() else ""
    if "AGENTS.override.md" not in claude_md:
        failures.append("CLAUDE.md: does not point Codex readers at AGENTS.override.md")

    settings = json.loads((ROOT / ".claude/settings.json").read_text())
    commands = [
        hook["command"]
        for groups in settings.get("hooks", {}).values()
        for group in groups
        for hook in group.get("hooks", [])
        if hook.get("type") == "command"
    ]
    for name in (
        "stac-endpoint-guard.py",
        "performance-evidence-context.py",
        "performance-evidence-guard.py",
    ):
        wired = [command for command in commands if name in command]
        if not wired:
            failures.append(f".claude/settings.json: {name} is not wired")
        elif not all("$CLAUDE_PROJECT_DIR" in command for command in wired):
            failures.append(
                f".claude/settings.json: {name} must be wired through $CLAUDE_PROJECT_DIR, "
                "or a session started from a subdirectory or worktree runs nothing"
            )
        if not (ROOT / ".claude/hooks" / name).is_file():
            failures.append(f".claude/hooks/{name}: missing")

    for name in (
        "docs/templates/performance-experiment-contract.json",
        "docs/templates/performance-result.json",
    ):
        if not (ROOT / name).is_file():
            failures.append(f"missing required template: {name}")

    gate = ROOT / GATE_WORKFLOW
    if not gate.is_file():
        failures.append(f"{GATE_WORKFLOW}: missing")
    else:
        text = gate.read_text()
        steps = _run_steps(text)
        if not any(
            re.fullmatch(rf"uv run python {re.escape(GATE_SCRIPT)} --root \S+", s) for s in steps
        ):
            failures.append(
                f"{GATE_WORKFLOW}: no uncommented run step executes {GATE_SCRIPT} --root <pr tree>"
            )
        if "pull_request_target" not in text:
            failures.append(f"{GATE_WORKFLOW}: must run under pull_request_target")
        if not re.search(r"types:\s*\[[^\]]*\bedited\b", text):
            failures.append(f"{GATE_WORKFLOW}: must re-run on the edited event")
        if "continue-on-error" in text:
            failures.append(f"{GATE_WORKFLOW}: continue-on-error would let a failed gate pass")

    for failure in failures:
        print(f"ERROR: {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
