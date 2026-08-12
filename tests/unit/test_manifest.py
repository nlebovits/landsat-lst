"""Unit tests for per-run manifest writing."""

import json
from datetime import UTC, datetime

import pytest

from landsat_lst.job import JobResult
from landsat_lst.manifest import write_run_manifest
from landsat_lst.models import ProcessingJob, TileId


@pytest.fixture
def results():
    completed = JobResult(
        job=ProcessingJob(tile=TileId(lat=40, lon=-75), year=2021, end_year=2025),
        status="completed",
        lst_key="lst-p95-2021-2025/N40W075/lst_p95_2021-2025_N40W075.tif",
        qa_key="lst-p95-2021-2025/N40W075/qa_count_2021-2025_N40W075.tif",
        duration_s=812.3,
        scene_count=412,
        peak_rss_mb=24100.5,
    )
    skipped = JobResult(
        job=ProcessingJob(tile=TileId(lat=-5, lon=-60), year=2021, end_year=2025),
        status="skipped",
    )
    failed = JobResult(
        job=ProcessingJob(tile=TileId(lat=60, lon=-150), year=2021, end_year=2025),
        status="failed",
        error="task failed after 3 retries (see Coiled cluster logs)",
    )
    return [completed, skipped, failed]


def test_writes_manifest_with_counts_and_tiles(results, tmp_path):
    path = write_run_manifest(
        results,
        run_id="2021-2025-20260812T140000Z",
        window="2021-2025",
        started_at=datetime(2026, 8, 12, 14, 0, tzinfo=UTC),
        retries=3,
        out_dir=tmp_path,
    )

    assert path == tmp_path / "2021-2025-20260812T140000Z.json"
    payload = json.loads(path.read_text())

    assert payload["run_id"] == "2021-2025-20260812T140000Z"
    assert payload["window"] == "2021-2025"
    assert payload["counts"] == {"total": 3, "completed": 1, "skipped": 1, "failed": 1}
    assert payload["config"]["retries"] == 3
    assert payload["config"]["region"] == "us-west-2"

    by_tile = {t["tile"]: t for t in payload["tiles"]}
    assert by_tile["N40W075"]["duration_s"] == 812.3
    assert by_tile["N40W075"]["scene_count"] == 412
    assert by_tile["S05W060"]["status"] == "skipped"
    assert "retries" in by_tile["N60W150"]["error"]


def test_creates_missing_directory(results, tmp_path):
    out_dir = tmp_path / "nested" / "runs"

    path = write_run_manifest(
        results,
        run_id="r1",
        window="2021-2025",
        started_at=datetime.now(tz=UTC),
        retries=1,
        out_dir=out_dir,
    )

    assert path.exists()


def test_roundtrips_valid_json(results, tmp_path):
    path = write_run_manifest(
        results,
        run_id="r2",
        window="2021-2025",
        started_at=datetime.now(tz=UTC),
        retries=1,
        out_dir=tmp_path,
    )

    payload = json.loads(path.read_text())
    assert len(payload["tiles"]) == 3
    assert payload["started_at"] < payload["finished_at"]
