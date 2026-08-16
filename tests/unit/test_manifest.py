"""Unit tests for per-run manifest writing."""

import json
from datetime import UTC, datetime

import pytest

from landsat_lst import pricing
from landsat_lst.job import JobResult
from landsat_lst.manifest import write_run_manifest
from landsat_lst.models import ProcessingJob, TileId


def _write(results, tmp_path, **extra):
    """Write one manifest under a fixed run id and read it back."""
    path = write_run_manifest(
        results,
        run_id="r",
        window="2021-2025",
        started_at=datetime(2026, 8, 12, 14, 0, tzinfo=UTC),
        retries=3,
        out_dir=tmp_path,
        **extra,
    )
    return json.loads(path.read_text())


def _attempt(number: int, **fields) -> dict:
    row = {
        "attempt": number,
        "phase": "composite_graph",
        "status": None,
        "duration_s": 800.0,
        "peak_rss_mb": 41200.0,
        "error": None,
        "log_key": f"_runs/r/N40W075.{number}.log",
    }
    row.update(fields)
    return row


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
    assert payload["config"]["max_workers"] == 4
    assert payload["config"]["job_timeout"] == "24 hours"

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


def test_records_the_cluster_that_ran_it(results, tmp_path):
    """Task logs live in Coiled; the manifest has to say where to find them."""
    path = write_run_manifest(
        results,
        run_id="r3",
        window="2021-2025",
        started_at=datetime.now(tz=UTC),
        retries=3,
        cluster_id=4242,
        job_id=77,
        out_dir=tmp_path,
    )

    payload = json.loads(path.read_text())
    assert payload["cluster_id"] == 4242
    assert payload["job_id"] == 77


def test_cluster_is_null_when_nothing_was_submitted(results, tmp_path):
    path = write_run_manifest(
        results,
        run_id="r4",
        window="2021-2025",
        started_at=datetime.now(tz=UTC),
        retries=3,
        out_dir=tmp_path,
    )

    payload = json.loads(path.read_text())
    assert payload["cluster_id"] is None


class TestAttempts:
    def test_series_and_summary(self, results, tmp_path):
        payload = _write(
            results,
            tmp_path,
            attempts={"N40W075": [_attempt(1, status="failed"), _attempt(2, status="completed")]},
        )

        by_tile = {t["tile"]: t for t in payload["tiles"]}
        assert payload["attempts"] == {"tiles_retried": 1, "total": 2, "max": 2}
        assert by_tile["N40W075"]["attempt"] == 2
        assert [row["attempt"] for row in by_tile["N40W075"]["attempts"]] == [1, 2]

    def test_tile_with_no_series_reports_none(self, results, tmp_path):
        """A skipped tile never ran, and a pre-attempt run numbered nothing."""
        payload = _write(results, tmp_path)

        by_tile = {t["tile"]: t for t in payload["tiles"]}
        assert by_tile["S05W060"]["attempts"] == []
        assert by_tile["S05W060"]["attempt"] == 0
        assert payload["attempts"] == {"tiles_retried": 0, "total": 0, "max": 0}


class TestCost:
    def test_totals_the_run_and_projects_a_fleet(self, results, tmp_path):
        estimate = pricing.tile_cost(
            duration_s=3600.0, instance_type="m6i.4xlarge", lifecycle="on-demand"
        )

        payload = _write(results, tmp_path, costs={"N40W075": estimate})

        by_tile = {t["tile"]: t for t in payload["tiles"]}
        assert by_tile["N40W075"]["cost"]["low"] == 0.768
        assert by_tile["N40W075"]["cost"]["provenance"] == "derived"
        assert by_tile["S05W060"]["cost"] is None
        assert payload["cost"]["priced_tiles"] == 1
        assert payload["cost"]["total"]["low"] == 0.768
        assert payload["cost"]["fleet"]["tiles"] == pricing.FLEET_TILES
        assert payload["cost"]["fleet"]["mean_usd_per_tile"]["low"] == 0.768
        assert payload["cost"]["disclaimer"] == pricing.DISCLAIMER

    def test_nothing_priced_projects_nothing(self, results, tmp_path):
        payload = _write(results, tmp_path)

        assert payload["cost"]["total"] is None
        assert payload["cost"]["fleet"] is None
        assert payload["cost"]["priced_tiles"] == 0


class TestPlan:
    def test_block_is_written_when_one_was_supplied(self, results, tmp_path):
        plan = {"planned": {"scenes": 2930}, "memory": {"floor_gib": 51.4, "ratio": 1.53}}

        payload = _write(results, tmp_path, plan=plan)

        assert payload["plan"] == plan

    def test_block_is_absent_when_the_run_stored_no_plan(self, results, tmp_path):
        payload = _write(results, tmp_path)

        assert "plan" not in payload
