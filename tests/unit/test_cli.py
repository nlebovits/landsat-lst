"""Unit tests for the process CLI command (no processing, no cluster)."""

from datetime import UTC, datetime
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


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _result(tile: str, status: str, error: str | None = None) -> JobResult:
    job = ProcessingJob(tile=parse_tile_name(tile), year=2021, end_year=2025)
    return JobResult(job=job, status=status, error=error)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def scratch_storage(tmp_path, monkeypatch):
    """Keep whatever a ``--run-id`` invocation uploads out of the repository.

    A batch task uploads its own log, so any test that passes ``--run-id`` now
    writes to the configured backend. Pointing that at ``tmp_path`` is the
    difference between a test and a side effect.
    """
    from landsat_lst.config import settings

    monkeypatch.setattr(settings, "output_dir", tmp_path / "cogs")
    monkeypatch.setattr(settings, "manifest_dir", tmp_path / "runs")


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

    def test_batch_task_uploads_its_own_log(self, runner, tmp_path):
        """Coiled keeps task stdout on the VM, so the tile has to publish it."""
        from landsat_lst.storage import LocalStorage

        with patch("landsat_lst.job.process_tile_job") as mock_job:
            mock_job.return_value = _result("N40W075", "completed")
            result = runner.invoke(main, ["process", "-t", "N40W075", "--run-id", "run-1"])

        storage = LocalStorage(output_dir=tmp_path / "cogs")
        assert result.exit_code == 0
        assert storage.read_text(storage.log_key("run-1", "N40W075")) is not None

    def test_a_failed_tile_still_leaves_its_log(self, runner, tmp_path):
        from landsat_lst.storage import LocalStorage

        with patch("landsat_lst.job.process_tile_job") as mock_job:
            mock_job.return_value = _result("N40W075", "failed", "No scenes found")
            result = runner.invoke(main, ["process", "-t", "N40W075", "--run-id", "run-1"])

        storage = LocalStorage(output_dir=tmp_path / "cogs")
        assert result.exit_code == 1
        assert storage.read_text(storage.log_key("run-1", "N40W075")) is not None

    def test_a_local_run_uploads_nothing(self, runner, tmp_path):
        """Output is already in front of somebody; do not pay S3 to repeat it."""
        with patch("landsat_lst.job.process_tile_job") as mock_job:
            mock_job.return_value = _result("N40W075", "completed")
            runner.invoke(main, ["process", "-t", "N40W075"])

        assert not (tmp_path / "cogs" / "_runs").exists()

    def test_a_multi_tile_sweep_uploads_nothing(self, runner, tmp_path):
        """The capture is per task, and a task is exactly one tile."""
        with patch("landsat_lst.job.process_tile_job") as mock_job:
            mock_job.return_value = _result("N40W075", "completed")
            runner.invoke(main, ["process", "-t", "N40W075", "-t", "S05W060", "--run-id", "run-1"])

        assert not (tmp_path / "cogs" / "_runs").exists()


class TestWatchCommand:
    def test_follows_the_run(self, runner):
        from landsat_lst.watch import RunSnapshot

        snapshot = RunSnapshot(run_id="run-1", taken_at=_now())
        with patch("landsat_lst.watch.watch_run", return_value=snapshot) as mock_watch:
            result = runner.invoke(main, ["watch", "run-1"])

        assert result.exit_code == 0
        assert mock_watch.call_args.args == ("run-1",)
        assert mock_watch.call_args.kwargs["once"] is False

    def test_forwards_its_options(self, runner):
        from landsat_lst.watch import RunSnapshot

        snapshot = RunSnapshot(run_id="run-1", taken_at=_now())
        with patch("landsat_lst.watch.watch_run", return_value=snapshot) as mock_watch:
            result = runner.invoke(main, ["watch", "run-1", "--once", "--interval", "5", "--all"])

        assert result.exit_code == 0
        kwargs = mock_watch.call_args.kwargs
        assert kwargs["once"] is True
        assert kwargs["interval_s"] == 5
        assert kwargs["show_all"] is True

    def test_points_at_reconcile_once_the_run_is_over(self, runner):
        from landsat_lst.watch import RunSnapshot, TileStatus

        snapshot = RunSnapshot(
            run_id="run-1",
            taken_at=_now(),
            tiles=[TileStatus(tile="N40W075", category="done", phase="done")],
            submitted=1,
        )
        with patch("landsat_lst.watch.watch_run", return_value=snapshot):
            result = runner.invoke(main, ["watch", "run-1"])

        assert "landsat-lst reconcile run-1" in result.output

    def test_ctrl_c_leaves_the_run_alone(self, runner):
        """Watching is a bystander's act; stopping it must not read as damage."""
        with patch("landsat_lst.watch.watch_run", side_effect=KeyboardInterrupt):
            result = runner.invoke(main, ["watch", "run-1"])

        assert result.exit_code == 0
        assert "run is untouched" in result.output
