"""Unit tests for the process CLI command (no processing, no cluster)."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from landsat_lst.batch import BatchSubmission
from landsat_lst.cli import main
from landsat_lst.job import JobResult
from landsat_lst.models import ProcessingJob
from landsat_lst.tiling import parse_tile_name


def _submission(**overrides) -> BatchSubmission:
    defaults = {
        "run_id": "2021-2025-20260812T140000Z",
        "window": "2021-2025",
        "cluster_id": 4242,
        "job_id": 77,
        "submitted_at": "2026-08-12T14:00:00+00:00",
        "submitted_tiles": ["N40W075"],
        "year": 2021,
        "end_year": 2025,
    }
    return BatchSubmission(**{**defaults, **overrides})


def _result(tile: str, status: str, error: str | None = None) -> JobResult:
    job = ProcessingJob(tile=parse_tile_name(tile), year=2021, end_year=2025)
    return JobResult(job=job, status=status, error=error)


@pytest.fixture
def runner():
    return CliRunner()


class TestProcessDryRun:
    def test_repeated_tiles(self, runner):
        result = runner.invoke(main, ["process", "--dry-run", "-t", "N40W075", "-t", "S05W060"])

        assert result.exit_code == 0
        assert "N40W075" in result.output
        assert "S05W060" in result.output

    def test_limit_slices_job_list(self, runner):
        result = runner.invoke(
            main,
            ["process", "--dry-run", "--limit", "1", "-t", "N40W075", "-t", "S05W060"],
        )

        assert result.exit_code == 0
        assert "Would process locally: N40W075" in result.output
        assert "Would process locally: S05W060" not in result.output

    def test_distributed_dry_run_does_not_dispatch(self, runner):
        with patch("landsat_lst.batch.submit_batch") as mock_submit:
            result = runner.invoke(main, ["process", "--dry-run", "--distributed", "-t", "N40W075"])

        assert result.exit_code == 0
        assert "on Coiled" in result.output
        mock_submit.assert_not_called()


class TestProcessDistributed:
    def test_submits_and_returns_without_waiting(self, runner):
        with (
            patch("landsat_lst.batch.submit_batch", return_value=_submission()) as mock_submit,
            patch("landsat_lst.batch.reconcile_run") as mock_reconcile,
            patch("landsat_lst.batch.wait_for_batch") as mock_wait,
        ):
            result = runner.invoke(main, ["process", "--distributed", "-t", "N40W075"])

        assert result.exit_code == 0
        _, kwargs = mock_submit.call_args
        assert kwargs["force"] is False
        assert "4242" in result.output
        assert "landsat-lst reconcile 2021-2025-20260812T140000Z" in result.output
        # The whole point of the pivot: no held connection, no live manifest.
        mock_wait.assert_not_called()
        mock_reconcile.assert_not_called()

    def test_wait_reconciles_and_reports(self, runner):
        with (
            patch("landsat_lst.batch.submit_batch", return_value=_submission()),
            patch("landsat_lst.batch.wait_for_batch", return_value="done (success)") as mock_wait,
            patch(
                "landsat_lst.batch.reconcile_run",
                return_value=[_result("N40W075", "completed")],
            ) as mock_reconcile,
        ):
            result = runner.invoke(main, ["process", "--distributed", "--wait", "-t", "N40W075"])

        assert result.exit_code == 0
        mock_wait.assert_called_once()
        mock_reconcile.assert_called_once()
        assert "Completed: 1" in result.output
        assert "Manifest:" in result.output

    def test_finished_window_reports_without_a_cluster(self, runner):
        submission = _submission(cluster_id=None, job_id=None, submitted_tiles=[])
        with (
            patch("landsat_lst.batch.submit_batch", return_value=submission),
            patch(
                "landsat_lst.batch.reconcile_run",
                return_value=[_result("N40W075", "skipped")],
            ),
        ):
            result = runner.invoke(main, ["process", "--distributed", "-t", "N40W075"])

        assert result.exit_code == 0
        assert "already complete" in result.output
        assert "Skipped: 1" in result.output


class TestReconcileCommand:
    def test_reports_failed_tiles(self, runner):
        results = [
            _result("N40W075", "completed"),
            _result("S05W060", "failed", "task exited 137 (state error)"),
        ]
        with patch("landsat_lst.batch.reconcile_run", return_value=results) as mock_reconcile:
            result = runner.invoke(main, ["reconcile", "run-1"])

        assert result.exit_code == 0
        mock_reconcile.assert_called_once_with("run-1")
        assert "Completed: 1" in result.output
        assert "Failed: 1" in result.output
        assert "task exited 137" in result.output
        assert "Manifest:" in result.output

    def test_unknown_run_id_fails_loudly(self, runner):
        with patch("landsat_lst.batch.reconcile_run", side_effect=FileNotFoundError("no record")):
            result = runner.invoke(main, ["reconcile", "nope"])

        assert result.exit_code != 0


class TestProcessLocal:
    """The local path is also what one batch VM runs."""

    def test_run_id_reaches_the_job(self, runner):
        with patch("landsat_lst.job.process_tile_job") as mock_job:
            mock_job.return_value = _result("N40W075", "completed")
            result = runner.invoke(main, ["process", "-t", "N40W075", "--run-id", "run-1"])

        assert result.exit_code == 0
        assert mock_job.call_args.kwargs["run_id"] == "run-1"

    def test_failed_tile_exits_non_zero(self, runner):
        """Coiled reads the exit code; a silent zero would mask a dead tile."""
        with patch("landsat_lst.job.process_tile_job") as mock_job:
            mock_job.return_value = _result("N40W075", "failed", "No scenes found")
            result = runner.invoke(main, ["process", "-t", "N40W075", "--run-id", "run-1"])

        assert result.exit_code == 1
        assert "No scenes found" in result.output
