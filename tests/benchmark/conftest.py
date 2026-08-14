"""Fixtures for the regression-guard tier.

These are guards, not measurements. A CI runner is smaller than the dev box and
much smaller than a production VM, so it cannot reproduce a production peak and
does not try. It runs reduced geometry and asserts on the **shape** of the
answer. ``scripts/synthetic_scaling.py`` supplies the production numbers; this
tier catches the change that would have moved them.

Every band here is wide on purpose. A benchmark that fails on a 3% drift gets
disabled within a month; one that fails when peak RSS doubles is the one that
would have caught the regression that shipped without comment.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from landsat_lst.benchmarks import CI_GEOMETRY, Geometry, Measurement, measure
from landsat_lst.pipeline import TIME_CHUNK

#: Where the trend lands. One JSON object per measurement, appended. A single
#: pass or fail cannot show a number drifting toward a cliff over five PRs, so
#: the band assertion and the record are separate jobs: the band fails a build,
#: the record is what a human reads afterwards.
TREND_PATH = Path(os.environ.get("LST_BENCHMARK_TREND", "results/benchmark/trend.jsonl"))

#: The chunk edge these fixtures pin. Held apart from ``settings.load_chunk_size``
#: so that changing production's default does not silently re-point the
#: benchmarks at a different graph, which is the one way this tier could report
#: a green build while measuring something else.
PINNED_CHUNK = CI_GEOMETRY.chunk


@pytest.fixture(scope="session", autouse=True)
def pinned_time_chunk() -> None:
    """Fail loudly if the time chunk moves out from under the pinned numbers.

    ``synthetic_dataset`` reproduces ``pipeline.TIME_CHUNK`` exactly, and a
    synthetic stack chunked differently from a real load builds a different
    graph. Every task count recorded in this directory was measured at 10.
    """
    assert TIME_CHUNK == 10, (
        f"pipeline.TIME_CHUNK is {TIME_CHUNK}, not the 10 these benchmarks were "
        "pinned at. Re-measure the bands and update them deliberately."
    )


def _git_sha() -> str:
    """The commit under measurement, so a trend line can be attributed."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return out.stdout.strip() or "unknown"


@pytest.fixture(scope="session")
def record_measurement():
    """Append a measurement to the trend file. Never fails a test.

    Recording is instrumentation, and instrumentation never fails the thing it
    instruments. A trend file that cannot be written costs a line of history; a
    benchmark that dies writing one costs the signal it was there to give.
    """
    sha = _git_sha()

    def _record(name: str, m: Measurement) -> None:
        try:
            TREND_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "name": name,
                "commit": sha,
                "recorded_at": datetime.now(UTC).isoformat(),
                **m.as_dict(),
            }
            with TREND_PATH.open("a") as fh:
                fh.write(json.dumps(payload) + "\n")
        except OSError as e:  # pragma: no cover - depends on the filesystem
            print(f"benchmark trend not recorded: {e}")

    return _record


def _measured(geometry: Geometry) -> Measurement:
    """Run one configuration, or skip the test that asked for it.

    A child that dies leaves its stderr on the skip reason rather than on an
    assertion, because a machine too small to run the configuration is not a
    regression in the code under test.
    """
    m = measure(geometry)
    if not m.ok:
        pytest.skip(f"benchmark child failed for {geometry.label}: {m.error}")
    return m


@pytest.fixture(scope="session")
def pinned(record_measurement) -> Measurement:
    """One run of the pinned configuration, shared by every test that reads it.

    Session-scoped because the measurement costs a subprocess and several
    seconds, and every assertion in this directory is about the same run.
    """
    m = _measured(CI_GEOMETRY)
    record_measurement("pinned", m)
    return m


@pytest.fixture(scope="session")
def measure_one(record_measurement):
    """Measure an arbitrary configuration and record it under a name."""

    def _run(name: str, geometry: Geometry) -> Measurement:
        m = _measured(geometry)
        record_measurement(name, m)
        return m

    return _run
