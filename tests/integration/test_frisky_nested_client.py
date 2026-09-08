"""The upstream blocker behind ADR-021's inner-scheduler decision, kept live.

A Frisky client nested inside a Frisky worker task kills the shared scheduler
on frisky 0.7.2 (``scripts/experimental/frisky_nested_client_repro.py``). While
that holds, a shard's inner graph runs on an in-process scheduler. This test
asserts the failure *still happens*; ``xfail(strict=True)`` turns a fixed
Frisky into a test failure, which is the reminder to revisit Mode S.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("frisky")

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "experimental" / "frisky_nested_client_repro.py"


@pytest.mark.xfail(
    strict=True, reason="frisky 0.7.2: nested client kills the scheduler; see ADR-021"
)
def test_a_nested_client_survives_the_shared_scheduler():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--transport", "tcp"],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
    )
    body, _status = result.stdout.rstrip().rsplit("\n", 1)
    payload = json.loads(body)
    assert not payload["reproduced"], payload["errors"]
    assert len(payload["results"]) == 2
