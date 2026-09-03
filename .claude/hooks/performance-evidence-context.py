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
        f"MANDATORY PERFORMANCE POLICY: read {POLICY}. "
        "A plausible mechanism, synthetic or local benchmark, model, test, estimate, "
        "or user report permits only a pre-registered bounded discriminator and its "
        "smallest experimental treatment. Do not productionize, harden, or claim an "
        "optimization until a validated representative real-data evidence bundle "
        "records decision=proceed. Never invent or relabel measurements, omit "
        "contrary runs, or extrapolate local results. A valid LST_EVIDENCE_CONTRACT "
        "constrains cloud experiments but never authorizes one."
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
