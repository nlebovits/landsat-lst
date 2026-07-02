"""Unit tests for job orchestration module."""

from unittest.mock import MagicMock, patch

import pytest

from landsat_lst.job import (
    JobResult,
    _write_to_icechunk_with_retry,
    generate_jobs,
    process_tile_job,
)
from landsat_lst.models import ProcessingJob, TileId
from landsat_lst.storage import IcechunkStorage


@pytest.fixture
def sample_job():
    """Create a sample processing job."""
    return ProcessingJob(tile=TileId(lat=40, lon=-75), year=2023)


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = MagicMock()
    storage.zarr_exists.return_value = False
    storage.zarr_path.return_value = "/tmp/test/2023/N40W075.zarr"
    return storage


@pytest.fixture
def mock_icechunk_storage():
    """Create a mock IcechunkStorage backend."""
    storage = MagicMock(spec=IcechunkStorage)
    storage.zarr_exists.return_value = False
    storage.zarr_path.return_value = "2023/N40W075"
    session = MagicMock()
    session.commit.return_value = "snapshot_abc123"
    storage.writable_session.return_value = session
    return storage


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_completed_result(self, sample_job):
        result = JobResult(
            job=sample_job,
            status="completed",
            zarr_path="/tmp/test.zarr",
        )
        assert result.status == "completed"
        assert result.zarr_path == "/tmp/test.zarr"
        assert result.error is None

    def test_completed_with_commit_id(self, sample_job):
        result = JobResult(
            job=sample_job,
            status="completed",
            zarr_path="2023/N40W075",
            commit_id="abc123def456",
        )
        assert result.status == "completed"
        assert result.commit_id == "abc123def456"

    def test_skipped_result(self, sample_job):
        result = JobResult(job=sample_job, status="skipped")
        assert result.status == "skipped"
        assert result.zarr_path is None

    def test_failed_result(self, sample_job):
        result = JobResult(job=sample_job, status="failed", error="Test error")
        assert result.status == "failed"
        assert result.error == "Test error"


class TestIdempotentCheck:
    """Tests for idempotent Zarr existence check."""

    def test_skips_existing_zarr(self, sample_job, mock_storage):
        """Should skip processing when Zarr already exists."""
        mock_storage.zarr_exists.return_value = True

        result = process_tile_job(sample_job, force=False, storage=mock_storage)

        assert result.status == "skipped"
        mock_storage.zarr_exists.assert_called_once_with("2023", "N40W075")

    def test_force_reprocesses_existing_zarr(self, sample_job, mock_storage):
        """Should reprocess when force=True even if Zarr exists."""
        mock_storage.zarr_exists.return_value = True

        with (
            patch("landsat_lst.job.process_tile") as mock_process,
            patch("landsat_lst.job.write_zarr") as mock_write,
        ):
            mock_process.return_value = MagicMock()
            mock_write.return_value = "/tmp/test.zarr"

            result = process_tile_job(sample_job, force=True, storage=mock_storage)

        assert result.status == "completed"
        mock_storage.zarr_exists.assert_not_called()

    def test_processes_new_tile(self, sample_job, mock_storage):
        """Should process tile when Zarr doesn't exist."""
        mock_storage.zarr_exists.return_value = False

        with (
            patch("landsat_lst.job.process_tile") as mock_process,
            patch("landsat_lst.job.write_zarr") as mock_write,
        ):
            mock_process.return_value = MagicMock()
            mock_write.return_value = "/tmp/test.zarr"

            result = process_tile_job(sample_job, force=False, storage=mock_storage)

        assert result.status == "completed"
        mock_process.assert_called_once()


class TestIcechunkConflictRetry:
    """Tests for Icechunk conflict retry logic."""

    def test_successful_commit_first_try(self, sample_job, mock_icechunk_storage):
        """Should commit successfully on first try."""
        composite = MagicMock()
        logger = MagicMock()

        with patch("landsat_lst.job.write_zarr"):
            group_path, commit_id = _write_to_icechunk_with_retry(
                composite, mock_icechunk_storage, sample_job, logger
            )

        assert group_path == "2023/N40W075"
        assert commit_id == "snapshot_abc123"
        mock_icechunk_storage.writable_session.assert_called_once()

    def test_retries_on_conflict(self, sample_job, tmp_path):
        """Should retry on ConflictError."""
        import icechunk as ic

        storage = IcechunkStorage.from_local(tmp_path / "icechunk")
        composite = MagicMock()
        logger = MagicMock()

        # ConflictError requires (expected_parent, actual_parent)
        conflict_error = ic.ConflictError("expected_snap", "actual_snap")

        # Mock writable_session to return sessions that fail then succeed
        session1 = MagicMock()
        session1.commit.side_effect = conflict_error
        session2 = MagicMock()
        session2.commit.return_value = "snapshot_success"

        with (
            patch.object(storage, "writable_session", side_effect=[session1, session2]),
            patch("landsat_lst.job.write_zarr"),
        ):
            _group_path, commit_id = _write_to_icechunk_with_retry(
                composite, storage, sample_job, logger
            )

        assert commit_id == "snapshot_success"
        assert logger.warning.call_count == 1  # One conflict warning

    def test_max_retries_exceeded(self, sample_job, tmp_path):
        """Should raise after max retries."""
        import icechunk as ic

        storage = IcechunkStorage.from_local(tmp_path / "icechunk")
        composite = MagicMock()
        logger = MagicMock()

        # ConflictError requires (expected_parent, actual_parent)
        conflict_error = ic.ConflictError("expected_snap", "actual_snap")

        # Mock to always conflict
        session = MagicMock()
        session.commit.side_effect = conflict_error

        with (
            patch.object(storage, "writable_session", return_value=session),
            patch("landsat_lst.job.write_zarr"),
            patch("landsat_lst.job.settings") as mock_settings,
        ):
            mock_settings.icechunk_max_retries = 3

            with pytest.raises(ic.ConflictError):
                _write_to_icechunk_with_retry(composite, storage, sample_job, logger)

        assert session.commit.call_count == 3


class TestIcechunkIntegration:
    """Tests for IcechunkStorage in process_tile_job."""

    def test_uses_icechunk_path(self, sample_job, mock_icechunk_storage):
        """Should use Icechunk write path when storage is IcechunkStorage."""
        with (
            patch("landsat_lst.job.process_tile") as mock_process,
            patch("landsat_lst.job._write_to_icechunk_with_retry") as mock_write,
            patch("landsat_lst.job.isinstance", return_value=True),
        ):
            mock_process.return_value = MagicMock()
            mock_write.return_value = ("2023/N40W075", "snapshot123")

            # Use real isinstance check by actually using IcechunkStorage mock
            result = process_tile_job(sample_job, storage=mock_icechunk_storage)

        assert result.status == "completed"
        assert result.commit_id == "snapshot123"


class TestGenerateJobs:
    """Tests for job generation."""

    def test_generates_jobs_for_year(self):
        """Should generate jobs for all land tiles."""
        jobs = generate_jobs([2023])

        assert len(jobs) == 700  # LAND_TILES count
        assert all(j.year == 2023 for j in jobs)

    def test_generates_jobs_for_multiple_years(self):
        """Should generate jobs for multiple years."""
        jobs = generate_jobs([2023, 2024])

        assert len(jobs) == 1400  # 700 tiles x 2 years
        years = {j.year for j in jobs}
        assert years == {2023, 2024}

    def test_jobs_are_sorted(self):
        """Job tiles should be sorted alphabetically."""
        jobs = generate_jobs([2023])

        tile_names = [j.tile.name for j in jobs]
        assert tile_names == sorted(tile_names)


class TestProcessTileJobFailure:
    """Tests for failure handling in process_tile_job."""

    def test_returns_failed_on_exception(self, sample_job, mock_storage):
        """Should return failed status when pipeline raises."""
        mock_storage.zarr_exists.return_value = False

        with patch("landsat_lst.job.process_tile") as mock_process:
            mock_process.side_effect = ValueError("No scenes found")

            result = process_tile_job(sample_job, storage=mock_storage)

        assert result.status == "failed"
        assert "No scenes found" in result.error
