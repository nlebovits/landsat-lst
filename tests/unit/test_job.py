"""Unit tests for job orchestration module."""

from unittest.mock import MagicMock, patch

import pytest

from landsat_lst.job import (
    JobResult,
    _commit_to_icechunk,
    generate_jobs,
    process_tile_job,
)
from landsat_lst.models import ProcessingJob, TileId


@pytest.fixture
def sample_job():
    """Create a sample processing job."""
    return ProcessingJob(tile=TileId(lat=40, lon=-75), year=2023)


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = MagicMock()
    storage.cog_exists.return_value = False
    storage.cog_path.return_value = "/tmp/test/2023/N40W075.tif"
    storage.icechunk_storage.return_value = MagicMock()
    return storage


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_completed_result(self, sample_job):
        result = JobResult(
            job=sample_job,
            status="completed",
            commit_id="abc123",
            cog_path="/tmp/test.tif",
        )
        assert result.status == "completed"
        assert result.commit_id == "abc123"
        assert result.error is None

    def test_skipped_result(self, sample_job):
        result = JobResult(job=sample_job, status="skipped")
        assert result.status == "skipped"
        assert result.commit_id is None

    def test_failed_result(self, sample_job):
        result = JobResult(job=sample_job, status="failed", error="Test error")
        assert result.status == "failed"
        assert result.error == "Test error"


class TestIdempotentCheck:
    """Tests for idempotent COG existence check."""

    def test_skips_existing_cog(self, sample_job, mock_storage):
        """Should skip processing when COG already exists."""
        mock_storage.cog_exists.return_value = True

        result = process_tile_job(sample_job, force=False, storage=mock_storage)

        assert result.status == "skipped"
        mock_storage.cog_exists.assert_called_once_with(2023, "N40W075")

    def test_force_reprocesses_existing_cog(self, sample_job, mock_storage):
        """Should reprocess when force=True even if COG exists."""
        mock_storage.cog_exists.return_value = True

        with (
            patch("landsat_lst.job.process_tile") as mock_process,
            patch("landsat_lst.job.write_cog"),
            patch("landsat_lst.job._commit_to_icechunk") as mock_commit,
        ):
            mock_process.return_value = MagicMock()
            mock_commit.return_value = "commit123"

            result = process_tile_job(sample_job, force=True, storage=mock_storage)

        assert result.status == "completed"
        mock_storage.cog_exists.assert_not_called()

    def test_processes_new_tile(self, sample_job, mock_storage):
        """Should process tile when COG doesn't exist."""
        mock_storage.cog_exists.return_value = False

        with (
            patch("landsat_lst.job.process_tile") as mock_process,
            patch("landsat_lst.job.write_cog"),
            patch("landsat_lst.job._commit_to_icechunk") as mock_commit,
        ):
            mock_process.return_value = MagicMock()
            mock_commit.return_value = "commit123"

            result = process_tile_job(sample_job, force=False, storage=mock_storage)

        assert result.status == "completed"
        mock_process.assert_called_once()


class TestIcechunkCommit:
    """Tests for Icechunk commit with conflict retry."""

    def test_successful_commit(self, sample_job, mock_storage):
        """Should commit successfully on first try."""
        mock_repo = MagicMock()
        mock_session = MagicMock()
        mock_session.commit.return_value = "commit_id_123"
        mock_repo.writable_session.return_value = mock_session

        with patch("landsat_lst.job.icechunk") as mock_ic:
            mock_ic.Repository.open_or_create.return_value = mock_repo
            with patch("landsat_lst.job.open_virtual_dataset") as mock_vds:
                mock_vds.return_value = MagicMock()

                commit_id = _commit_to_icechunk("/tmp/test.tif", mock_storage, sample_job)

        assert commit_id == "commit_id_123"

    def test_retry_on_conflict(self, sample_job, mock_storage):
        """Should retry on ConflictError."""
        mock_repo = MagicMock()
        mock_session = MagicMock()

        # Create a custom exception class for testing
        class MockConflictError(Exception):
            pass

        # Fail first two times, succeed third
        mock_session.commit.side_effect = [
            MockConflictError("conflict"),
            MockConflictError("conflict"),
            "commit_id_123",
        ]
        mock_repo.writable_session.return_value = mock_session

        with patch("landsat_lst.job.icechunk") as mock_ic:
            mock_ic.Repository.open_or_create.return_value = mock_repo
            mock_ic.ConflictError = MockConflictError
            with patch("landsat_lst.job.open_virtual_dataset") as mock_vds:
                mock_vds.return_value = MagicMock()

                commit_id = _commit_to_icechunk("/tmp/test.tif", mock_storage, sample_job)

        assert commit_id == "commit_id_123"
        assert mock_session.commit.call_count == 3

    def test_max_retries_exceeded(self, sample_job, mock_storage):
        """Should raise after max retries."""
        mock_repo = MagicMock()
        mock_session = MagicMock()

        # Create a custom exception class for testing
        class MockConflictError(Exception):
            pass

        mock_session.commit.side_effect = MockConflictError("conflict")
        mock_repo.writable_session.return_value = mock_session

        with patch("landsat_lst.job.icechunk") as mock_ic:
            mock_ic.Repository.open_or_create.return_value = mock_repo
            mock_ic.ConflictError = MockConflictError
            with patch("landsat_lst.job.open_virtual_dataset") as mock_vds:
                mock_vds.return_value = MagicMock()

                with pytest.raises(MockConflictError):
                    _commit_to_icechunk("/tmp/test.tif", mock_storage, sample_job, max_retries=3)


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
        mock_storage.cog_exists.return_value = False

        with patch("landsat_lst.job.process_tile") as mock_process:
            mock_process.side_effect = ValueError("No scenes found")

            result = process_tile_job(sample_job, storage=mock_storage)

        assert result.status == "failed"
        assert "No scenes found" in result.error
