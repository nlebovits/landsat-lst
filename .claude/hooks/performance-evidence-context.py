#!/usr/bin/env python3
"""Put the repository's performance rules in front of every Claude session."""

from __future__ import annotations

import json
from pathlib import Path

#: Resolved from this file, not the working directory: a session started from
#: a subdirectory or a worktree must still see the policy.
ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "docs/performance-evidence-policy.md"


def main() -> None:
    if not POLICY.is_file():
        raise FileNotFoundError(f"performance policy is missing: {POLICY}")
    summary = (
        "PERFORMANCE AND COST WORK, THE ORDER IS FIXED. This repository lost two "
        "months to optimizations that were plausible, locally fast, and null in "
        "production (#126: 4.4x local, 1.1% in production, I/O-bound at 10% CPU), "
        "with a day of architecture and review before the 28-minute, $0.36 test that "
        "ended it. Before any plan, estimate, analysis, subagent, or line of code "
        "touching runtime, memory, I/O, concurrency, scaling, or cost: (1) say it in "
        "the conversation in one paragraph: the measured number in this repository "
        "your idea moves, the buried idea you are not repeating, what would make it "
        "worthless; if you cannot, say so and stop. (2) name the bare-minimum Coiled "
        "smoke test that would kill it (one real shard or landsat-lst benchmark "
        "--distributed; under a dollar, under 30 minutes), the number it produces, "
        "and the value that means stop; then wait for the operator to say go. "
        "(3) run that one test and pull the actual numbers with landsat-lst evidence "
        "collect --cluster-id (Coiled lifecycle, billing, CPU and memory series, "
        "logs). (4) only then plan, architect, review, or fan out agents. Every cost "
        "or time figure carries its billed anchor and a range (S30W065: $7.28 per "
        f"tile billed, 268 credits where the model said 75). Lifecycle: {POLICY}. "
        "The section of the same name at the top of CLAUDE.md is the rule."
    )
    print(
        json.dumps(
            {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": summary}}
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # A SessionStart hook cannot deny anything, so the loudest available
        # failure is context that says the policy could not be loaded.
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": (
                            "MANDATORY PERFORMANCE POLICY could not be loaded: "
                            f"{exc}. Treat the repository as gated and read "
                            "docs/performance-evidence-policy.md before any performance work."
                        ),
                    }
                }
            )
        )
