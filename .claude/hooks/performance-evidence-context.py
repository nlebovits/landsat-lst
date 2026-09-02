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
        "MANDATORY PERFORMANCE POLICY: read docs/performance-evidence-policy.md "
        "before performance or cost work. Validate the premise and retained "
        "production baseline first; pre-register the cheapest production "
        "discriminator and stop rule; do not extrapolate local results. A valid "
        "LST_EVIDENCE_CONTRACT constrains cloud experiments but never authorizes one."
    )
    print(
        json.dumps(
            {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": summary}}
        )
    )


if __name__ == "__main__":
    main()
