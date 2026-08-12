"""Unit tests for the process CLI command (no processing, no cluster)."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from landsat_lst.cli import main
from landsat_lst.job import JobResult


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
        with patch("landsat_lst.job.run_distributed") as mock_run:
            result = runner.invoke(main, ["process", "--dry-run", "--distributed", "-t", "N40W075"])

        assert result.exit_code == 0
        assert "on Coiled" in result.output
        mock_run.assert_not_called()


class TestProcessDistributed:
    def test_dispatches_and_reports_manifest(self, runner):
        job_result = None

        def fake_run(jobs, *, force, run_id):
            nonlocal job_result
            job_result = JobResult(job=jobs[0], status="completed")
            return [job_result]

        with patch("landsat_lst.job.run_distributed", side_effect=fake_run) as mock_run:
            result = runner.invoke(main, ["process", "--distributed", "-t", "N40W075"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["force"] is False
        assert kwargs["run_id"].startswith("2021-2025-")
        assert "Manifest:" in result.output
        assert "Completed: 1" in result.output

    def test_failed_tiles_listed(self, runner):
        def fake_run(jobs, *, force, run_id):
            return [JobResult(job=jobs[0], status="failed", error="task failed after 3 retries")]

        with patch("landsat_lst.job.run_distributed", side_effect=fake_run):
            result = runner.invoke(main, ["process", "--distributed", "-t", "N40W075"])

        assert result.exit_code == 0
        assert "Failed: 1" in result.output
        assert "task failed after 3 retries" in result.output
