"""Unit tests for job orchestration module."""

from unittest.mock import MagicMock, patch

import pytest

from landsat_lst.job import (
    DEFAULT_WINDOW,
    JobResult,
    generate_jobs,
    process_tile_job,
)
from landsat_lst.models import ProcessingJob, TileId
from landsat_lst.storage import LocalStorage


@pytest.fixture
def sample_job():
    """Create a sample processing job."""
    return ProcessingJob(tile=TileId(lat=40, lon=-75), year=2023)


@pytest.fixture
def mock_storage(tmp_path):
    """Mock backend: real key layout, mocked existence checks and uploads."""
    storage = MagicMock()
    storage.cog_exists.return_value = False
    storage.cog_key.side_effect = LocalStorage(output_dir=tmp_path).cog_key
    return storage


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_completed_result(self, sample_job):
        result = JobResult(
            job=sample_job,
            status="completed",
            lst_key="2023/N40W075/lst_p95_2023_N40W075.tif",
            qa_key="2023/N40W075/qa_count_2023_N40W075.tif",
        )
        assert result.status == "completed"
        assert result.lst_key.endswith("lst_p95_2023_N40W075.tif")
        assert result.qa_key.endswith("qa_count_2023_N40W075.tif")
        assert result.error is None

    def test_skipped_result(self, sample_job):
        result = JobResult(job=sample_job, status="skipped")
        assert result.status == "skipped"
        assert result.lst_key is None
        assert result.qa_key is None

    def test_failed_result(self, sample_job):
        result = JobResult(job=sample_job, status="failed", error="Test error")
        assert result.status == "failed"
        assert result.error == "Test error"


class TestIdempotentCheck:
    """Tests for the idempotent two-asset existence check."""

    def test_skips_existing_cogs(self, sample_job, mock_storage):
        """Should skip processing when both COGs already exist."""
        mock_storage.cog_exists.return_value = True

        result = process_tile_job(sample_job, force=False, storage=mock_storage)

        assert result.status == "skipped"
        mock_storage.cog_exists.assert_called_once_with("2023", "N40W075")

    def test_force_reprocesses_existing_cogs(self, sample_job, mock_storage):
        """Should reprocess when force=True even if the COGs exist."""
        mock_storage.cog_exists.return_value = True

        with (
            patch("landsat_lst.job.process_tile") as mock_process,
            patch("landsat_lst.job._encode_native") as mock_encode,
            patch("landsat_lst.job.cog_export") as mock_export,
        ):
            mock_process.return_value = MagicMock()
            mock_encode.return_value = MagicMock()
            mock_export.return_value = (MagicMock(), MagicMock())

            result = process_tile_job(sample_job, force=True, storage=mock_storage)

        assert result.status == "completed"
        mock_storage.cog_exists.assert_not_called()

    def test_processes_new_tile(self, sample_job, mock_storage):
        """Should process the tile when the COGs don't exist."""
        mock_storage.cog_exists.return_value = False

        with (
            patch("landsat_lst.job.process_tile") as mock_process,
            patch("landsat_lst.job._encode_native") as mock_encode,
            patch("landsat_lst.job.cog_export") as mock_export,
        ):
            mock_process.return_value = MagicMock()
            mock_encode.return_value = MagicMock()
            mock_export.return_value = (MagicMock(), MagicMock())

            result = process_tile_job(sample_job, force=False, storage=mock_storage)

        assert result.status == "completed"
        mock_process.assert_called_once()


class TestCogWrite:
    """Tests for the export/upload half of process_tile_job."""

    def _run(self, job, storage):
        with (
            patch("landsat_lst.job.process_tile") as mock_process,
            patch("landsat_lst.job._encode_native") as mock_encode,
            patch("landsat_lst.job.cog_export") as mock_export,
        ):
            mock_process.return_value = MagicMock()
            mock_encode.return_value = MagicMock()
            mock_export.return_value = (MagicMock(), MagicMock())
            result = process_tile_job(job, storage=storage)
        return result, mock_export

    def test_uploads_both_assets(self, sample_job, mock_storage):
        result, _ = self._run(sample_job, mock_storage)

        assert result.status == "completed"
        assert result.lst_key == "lst-p95-2023/N40W075/lst_p95_2023_N40W075.tif"
        assert result.qa_key == "lst-p95-2023/N40W075/qa_count_2023_N40W075.tif"
        uploaded_keys = {call.args[1] for call in mock_storage.upload.call_args_list}
        assert uploaded_keys == {result.lst_key, result.qa_key}

    def test_scratch_dir_is_removed(self, sample_job, mock_storage):
        """The temp export dir must not survive the job -- workers reuse disks."""
        _result, mock_export = self._run(sample_job, mock_storage)

        lst_local = mock_export.call_args.args[1]
        assert lst_local.name == "lst_p95_2023_N40W075.tif"
        assert not lst_local.parent.exists()

    def test_scratch_dir_removed_when_export_fails(self, sample_job, mock_storage):
        """A failed export must not leak its scratch dir either."""
        with (
            patch("landsat_lst.job.process_tile") as mock_process,
            patch("landsat_lst.job._encode_native") as mock_encode,
            patch("landsat_lst.job.cog_export") as mock_export,
        ):
            mock_process.return_value = MagicMock()
            mock_encode.return_value = MagicMock()
            mock_export.side_effect = RuntimeError("gdal exploded")

            result = process_tile_job(sample_job, storage=mock_storage)

        assert result.status == "failed"
        assert not mock_export.call_args.args[1].parent.exists()
        mock_storage.upload.assert_not_called()


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

    def test_defaults_to_the_production_window(self):
        """With no years, emit one multi-year job per tile (issue #46)."""
        jobs = generate_jobs()

        assert len(jobs) == 700
        assert DEFAULT_WINDOW == (2021, 2025)
        assert all(j.year == 2021 and j.end_year == 2025 for j in jobs)
        assert all(j.window_label == "2021-2025" for j in jobs)

    def test_custom_window(self):
        """An explicit window produces one job per tile, not one per year."""
        jobs = generate_jobs(window=(2022, 2024))

        assert len(jobs) == 700
        assert all(j.window_label == "2022-2024" for j in jobs)

    def test_years_override_window(self):
        """Passing years keeps the single-year behavior for backfill."""
        jobs = generate_jobs([2023], window=(2021, 2025))

        assert len(jobs) == 700
        assert all(j.end_year is None for j in jobs)
        assert all(j.window_label == "2023" for j in jobs)


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
