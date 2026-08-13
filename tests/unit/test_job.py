"""Unit tests for job orchestration module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from landsat_lst.job import (
    DEFAULT_WINDOW,
    JobResult,
    _is_transient,
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

    def test_transient_error_reraises(self, sample_job, mock_storage):
        """Transient failures must escape so Coiled task retries engage."""
        mock_storage.cog_exists.return_value = False

        with patch("landsat_lst.job.process_tile") as mock_process:
            mock_process.side_effect = TimeoutError("read timed out")

            with pytest.raises(TimeoutError):
                process_tile_job(sample_job, storage=mock_storage)

    def test_failed_result_records_duration(self, sample_job, mock_storage):
        mock_storage.cog_exists.return_value = False

        with patch("landsat_lst.job.process_tile") as mock_process:
            mock_process.side_effect = ValueError("boom")
            result = process_tile_job(sample_job, storage=mock_storage)

        assert result.status == "failed"
        assert result.duration_s is not None


class TestIsTransient:
    """Tests for the transient/deterministic failure split."""

    @staticmethod
    def _client_error(status: int, code: str):
        from botocore.exceptions import ClientError

        return ClientError(
            {
                "Error": {"Code": code, "Message": "x"},
                "ResponseMetadata": {"HTTPStatusCode": status},
            },
            "GetObject",
        )

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (ConnectionError("reset"), True),
            (TimeoutError("timeout"), True),
            (ValueError("no scenes"), False),
            (KeyError("band"), False),
        ],
    )
    def test_builtin_exceptions(self, exc, expected):
        assert _is_transient(exc) is expected

    def test_rasterio_io_error(self):
        from rasterio.errors import RasterioIOError

        assert _is_transient(RasterioIOError("curl error")) is True

    @pytest.mark.parametrize(
        ("status", "code", "expected"),
        [
            (500, "InternalError", True),
            (503, "SlowDown", True),
            (400, "Throttling", True),
            (404, "NoSuchKey", False),
            (403, "AccessDenied", False),
        ],
    )
    def test_client_errors(self, status, code, expected):
        assert _is_transient(self._client_error(status, code)) is expected


class TestRunRecord:
    """A batch VM reports for itself; these records are all reconciliation gets."""

    def _run(self, job, storage, run_id):
        with (
            patch("landsat_lst.job.process_tile") as mock_process,
            patch("landsat_lst.job._encode_native") as mock_encode,
            patch("landsat_lst.job.cog_export") as mock_export,
        ):
            composite = MagicMock()
            composite.attrs = {"scene_count": 412}
            mock_process.return_value = composite
            mock_encode.return_value = MagicMock()
            mock_export.return_value = (MagicMock(), MagicMock())
            return process_tile_job(job, storage=storage, run_id=run_id)

    def test_no_record_without_a_run_id(self, sample_job, mock_storage):
        self._run(sample_job, mock_storage, None)

        mock_storage.write_text.assert_not_called()

    def test_completed_record_carries_costing_metrics(self, sample_job, mock_storage):
        result = self._run(sample_job, mock_storage, "run-1")

        key, text = mock_storage.write_text.call_args.args
        assert key == mock_storage.run_record_key("run-1", "N40W075")
        record = json.loads(text)
        assert record["status"] == "completed"
        assert record["scene_count"] == 412
        assert record["duration_s"] == result.duration_s
        assert record["peak_rss_mb"] is not None

    def test_failed_record_carries_the_error(self, sample_job, mock_storage):
        with patch("landsat_lst.job.process_tile") as mock_process:
            mock_process.side_effect = ValueError("No scenes found")
            process_tile_job(sample_job, storage=mock_storage, run_id="run-1")

        record = json.loads(mock_storage.write_text.call_args.args[1])
        assert record["status"] == "failed"
        assert "No scenes found" in record["error"]

    def test_skipped_tile_still_reports(self, sample_job, mock_storage):
        """A resumed tile must appear in the manifest, not vanish from it."""
        mock_storage.cog_exists.return_value = True

        process_tile_job(sample_job, storage=mock_storage, run_id="run-1")

        record = json.loads(mock_storage.write_text.call_args.args[1])
        assert record["status"] == "skipped"

    def test_transient_failure_writes_no_record(self, sample_job, mock_storage):
        """The retry on a fresh VM writes the record; a doomed attempt must not.

        It does still beat, and its last beat says why it died: that is what a
        watcher sees while the retry is being scheduled.
        """
        with (
            patch("landsat_lst.job.process_tile") as mock_process,
            pytest.raises(TimeoutError),
        ):
            mock_process.side_effect = TimeoutError("read timed out")
            process_tile_job(sample_job, storage=mock_storage, run_id="run-1")

        written = [call.args[0] for call in mock_storage.write_text.call_args_list]
        assert mock_storage.run_record_key("run-1", "N40W075") not in written
        assert mock_storage.progress_key("run-1", "N40W075") in written

    def test_unwritable_record_does_not_fail_an_uploaded_tile(self, sample_job, mock_storage):
        mock_storage.write_text.side_effect = OSError("bucket on fire")

        result = self._run(sample_job, mock_storage, "run-1")

        assert result.status == "completed"

    def test_record_roundtrips(self, sample_job, mock_storage):
        result = self._run(sample_job, mock_storage, "run-1")

        restored = JobResult.from_record(json.loads(mock_storage.write_text.call_args.args[1]))

        assert restored.job == result.job
        assert restored.status == result.status
        assert restored.scene_count == result.scene_count
        assert restored.lst_key == result.lst_key


class _NoUploadStorage(LocalStorage):
    """Real storage for the small objects, no-op for the assets.

    These tests read heartbeats back out of storage, which a mock cannot give
    them, but ``cog_export`` is patched out so there is no asset on disk to
    upload.
    """

    def upload(self, local, key) -> None:
        pass


class TestHeartbeat:
    """The tile is the only thing that knows it is alive, so it has to say so."""

    def _storage(self, tmp_path) -> LocalStorage:
        return _NoUploadStorage(output_dir=tmp_path / "cogs")

    def _phases(self, storage: LocalStorage, tile: str = "N40W075") -> list[str]:
        raw = storage.read_text(storage.progress_key("run-1", tile))
        return [] if raw is None else [json.loads(raw)["phase"]]

    def _run(self, job, storage, *, run_id="run-1", pipeline=None):
        with (
            patch("landsat_lst.job.process_tile") as mock_process,
            patch("landsat_lst.job._encode_native") as mock_encode,
            patch("landsat_lst.job.cog_export") as mock_export,
        ):
            mock_process.side_effect = pipeline or (lambda _job: MagicMock())
            mock_encode.return_value = MagicMock()
            mock_export.return_value = (MagicMock(), MagicMock())
            return process_tile_job(job, storage=storage, run_id=run_id)

    def test_phases_reported_from_the_pipeline_reach_storage(self, sample_job, tmp_path):
        """The pipeline reports through a context variable, not an argument."""
        from landsat_lst.pipeline import report_phase

        storage = self._storage(tmp_path)
        seen = []

        def pipeline(job):
            report_phase("stac_query")
            seen.append(json.loads(storage.read_text(storage.progress_key("run-1", "N40W075"))))
            return MagicMock()

        self._run(sample_job, storage, pipeline=pipeline)

        assert seen[0]["phase"] == "stac_query"
        assert seen[0]["tile"] == "N40W075"

    def test_a_completed_tile_ends_on_done(self, sample_job, tmp_path):
        storage = self._storage(tmp_path)

        self._run(sample_job, storage)

        assert self._phases(storage) == ["done"]

    def test_export_and_upload_are_visible(self, sample_job, tmp_path):
        """Two of the longest phases; a flat dashboard here is what started #68."""
        storage = self._storage(tmp_path)
        phases = []

        original = storage.write_text

        def spy(key, text, **kwargs):
            if key.endswith(".progress.json"):
                phases.append(json.loads(text)["phase"])
            original(key, text, **kwargs)

        storage.write_text = spy
        self._run(sample_job, storage)

        assert "exporting" in phases
        assert "uploading" in phases

    def test_a_failed_tile_ends_on_failed_with_its_error(self, sample_job, tmp_path):
        storage = self._storage(tmp_path)

        def pipeline(job):
            msg = "No scenes found for the window"
            raise ValueError(msg)

        result = self._run(sample_job, storage, pipeline=pipeline)

        published = json.loads(storage.read_text(storage.progress_key("run-1", "N40W075")))
        assert result.status == "failed"
        assert published["phase"] == "failed"
        assert published["error"] == "No scenes found for the window"

    def test_a_tile_with_no_run_id_beats_to_nobody(self, sample_job, tmp_path):
        storage = self._storage(tmp_path)

        self._run(sample_job, storage, run_id=None)

        assert not (storage.output_dir / "_runs").exists()

    def test_a_skipped_tile_writes_only_its_record(self, sample_job, tmp_path):
        storage = self._storage(tmp_path)
        for product in ("lst_p95", "qa_count"):
            path = storage.output_dir / storage.cog_key("2023", "N40W075", product)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"tif")

        result = process_tile_job(sample_job, storage=storage, run_id="run-1")

        assert result.status == "skipped"
        assert storage.read_text(storage.run_record_key("run-1", "N40W075")) is not None
        assert storage.read_text(storage.progress_key("run-1", "N40W075")) is None


class TestThreadCap:
    """Peak memory during de-striping is threads * chunk**2 * scenes * 4 bytes.

    Capping threads cuts that term linearly without multiplying per-chunk
    overhead the way halving the chunk size does.
    """

    def test_no_cap_leaves_dask_alone(self, sample_job, mock_storage, monkeypatch):
        import dask

        from landsat_lst.config import settings

        monkeypatch.setattr(settings, "dask_max_threads", None)
        seen = {}

        def fake_process(job):
            seen["num_workers"] = dask.config.get("num_workers", None)
            return MagicMock()

        with (
            patch("landsat_lst.job.process_tile", side_effect=fake_process),
            patch("landsat_lst.job._write_cogs", return_value=("a", "b")),
        ):
            process_tile_job(sample_job, storage=mock_storage)

        assert seen["num_workers"] is None

    def test_cap_applies_for_the_whole_tile(self, sample_job, mock_storage, monkeypatch):
        import dask

        from landsat_lst.config import settings

        monkeypatch.setattr(settings, "dask_max_threads", 3)
        seen = {}

        def fake_process(job):
            seen["num_workers"] = dask.config.get("num_workers", None)
            seen["scheduler"] = dask.config.get("scheduler", None)
            return MagicMock()

        with (
            patch("landsat_lst.job.process_tile", side_effect=fake_process),
            patch("landsat_lst.job._write_cogs", return_value=("a", "b")),
        ):
            process_tile_job(sample_job, storage=mock_storage)

        assert seen["num_workers"] == 3
        assert seen["scheduler"] == "threads"

    def test_cap_is_scoped_to_the_tile(self, sample_job, mock_storage, monkeypatch):
        """A cap must not leak into whatever the process does next."""
        import dask

        from landsat_lst.config import settings

        monkeypatch.setattr(settings, "dask_max_threads", 2)
        before = dask.config.get("num_workers", None)

        with (
            patch("landsat_lst.job.process_tile", return_value=MagicMock()),
            patch("landsat_lst.job._write_cogs", return_value=("a", "b")),
        ):
            process_tile_job(sample_job, storage=mock_storage)

        assert dask.config.get("num_workers", None) == before
