#!/usr/bin/env python3
"""Put the repository's performance rules into every Claude session."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    if not Path("docs/performance-evidence-policy.md").is_file():
        print("{}")
        return
    summary = (
        "MANDATORY PERFORMANCE POLICY: read docs/performance-evidence-policy.md. "
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
    main()
