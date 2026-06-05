#!/usr/bin/env python3
"""
PreToolUse hook: Enforce correct STAC endpoint for environment.

- LOCAL testing: Planetary Computer only (free, no egress costs)
- AWS/Coiled: Earth Search only (same-region, no egress costs)

This prevents accidental use of Earth Search locally (expensive egress)
or Planetary Computer on AWS (cross-cloud egress).
"""

import json
import os
import sys

# Patterns that indicate Earth Search usage
EARTH_SEARCH_PATTERNS = [
    "earth-search.aws.element84.com",
    "STAC_EARTH_SEARCH",
    "stac_url=STAC_EARTH_SEARCH",
    'stac_url="https://earth-search',
    "stac_url='https://earth-search",
]

# Patterns that indicate Planetary Computer usage
PC_PATTERNS = [
    "planetarycomputer.microsoft.com",
    "STAC_PLANETARY_COMPUTER",
    "planetary_computer",
    "import planetary_computer",
]


def is_on_aws():
    """Check if running on AWS (Coiled, EC2, Lambda, etc.)."""
    # Check for AWS environment indicators
    aws_indicators = [
        "AWS_EXECUTION_ENV",  # Lambda
        "ECS_CONTAINER_METADATA_URI",  # ECS/Fargate
        "AWS_BATCH_JOB_ID",  # Batch
        "COILED_CLUSTER_NAME",  # Coiled
        "COILED_SOFTWARE_NAME",  # Coiled
    ]
    return any(os.environ.get(var) for var in aws_indicators)


def check_command(command: str) -> dict | None:
    """Check a bash command for STAC endpoint violations."""
    on_aws = is_on_aws()

    if on_aws:
        # On AWS: block Planetary Computer
        for pattern in PC_PATTERNS:
            if pattern in command:
                return {
                    "decision": "block",
                    "reason": (
                        f"BLOCKED: Using Planetary Computer on AWS/Coiled.\n"
                        f"On AWS, use Earth Search (same-region S3, no egress).\n"
                        f"Found: {pattern}"
                    ),
                }
    else:
        # Local: block Earth Search
        for pattern in EARTH_SEARCH_PATTERNS:
            if pattern in command:
                return {
                    "decision": "block",
                    "reason": (
                        f"BLOCKED: Using Earth Search locally.\n"
                        f"For local testing, use Planetary Computer (free).\n"
                        f"Earth Search is ONLY for AWS/Coiled (same-region S3).\n\n"
                        f"Fix: Set LST_STAC_URL or use settings override:\n"
                        f"  from landsat_lst.config import STAC_PLANETARY_COMPUTER\n"
                        f"  settings.stac_url = STAC_PLANETARY_COMPUTER\n\n"
                        f"Found pattern: {pattern}"
                    ),
                }

    return None  # Allow


def main():
    input_data = json.load(sys.stdin)

    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Only check Bash commands
    if tool_name != "Bash":
        print(json.dumps({"decision": "allow"}))
        return

    command = tool_input.get("command", "")

    result = check_command(command)
    if result:
        print(json.dumps(result))
    else:
        print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()
