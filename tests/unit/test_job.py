"""Unit tests for job orchestration module."""

from unittest.mock import MagicMock, patch

import pytest

from landsat_lst.job import (
    DEFAULT_WINDOW,
    JobResult,
    _is_transient,
    generate_jobs,
    process_tile_job,
    run_distributed,
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


class _FakeCoiledFunction:
    """Stand-in for the object coiled.function() wraps a callable into."""

    def __init__(self, fn, drop_tiles=()):
        self.fn = fn
        self.drop_tiles = set(drop_tiles)

    def map(self, jobs, forces, retries=None, errors="raise"):
        return [
            self.fn(job, force)
            for job, force in zip(jobs, forces, strict=True)
            if job.tile.name not in self.drop_tiles
        ]


@pytest.fixture
def fake_coiled(monkeypatch):
    """Install a coiled stub; returns dict capturing decorator kwargs."""
    import sys
    import types

    captured: dict = {}

    def function(**kwargs):
        captured.update(kwargs)

        def deco(fn):
            return _FakeCoiledFunction(fn, drop_tiles=captured.pop("_drop_tiles", ()))

        return deco

    fake = types.ModuleType("coiled")
    fake.function = function
    monkeypatch.setitem(sys.modules, "coiled", fake)
    # Static creds so _worker_environ never reaches for boto3/SSO in tests.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    return captured


@pytest.fixture
def manifest_dir(tmp_path, monkeypatch):
    from landsat_lst.config import settings

    monkeypatch.setattr(settings, "manifest_dir", tmp_path / "runs")
    return tmp_path / "runs"


def _jobs(*tiles: str) -> list[ProcessingJob]:
    from landsat_lst.tiling import parse_tile_name

    return [ProcessingJob(tile=parse_tile_name(t), year=2021, end_year=2025) for t in tiles]


class TestRunDistributed:
    """Tests for the hardened Coiled driver (all mocked, no cluster)."""

    def _stub_worker(self, monkeypatch):
        def fake_process(job, force=False):
            return JobResult(job=job, status="completed", duration_s=1.0)

        monkeypatch.setattr("landsat_lst.job.process_tile_job", fake_process)

    def test_resume_filters_completed_tiles(self, fake_coiled, manifest_dir, monkeypatch):
        self._stub_worker(monkeypatch)
        storage = MagicMock()
        storage.list_completed.return_value = {"N40W075"}

        jobs = _jobs("N40W075", "S05W060", "N60W150")
        results = run_distributed(jobs, storage=storage)

        storage.list_completed.assert_called_once_with("2021-2025")
        by_status = {r.job.tile.name: r.status for r in results}
        assert by_status == {
            "N40W075": "skipped",
            "S05W060": "completed",
            "N60W150": "completed",
        }

    def test_force_bypasses_resume(self, fake_coiled, manifest_dir, monkeypatch):
        self._stub_worker(monkeypatch)
        storage = MagicMock()

        results = run_distributed(_jobs("N40W075"), force=True, storage=storage)

        storage.list_completed.assert_not_called()
        assert [r.status for r in results] == ["completed"]

    def test_missing_results_marked_failed(self, fake_coiled, manifest_dir, monkeypatch):
        """A task dropped by errors='skip' is reconciled as failed."""
        self._stub_worker(monkeypatch)
        storage = MagicMock()
        storage.list_completed.return_value = set()
        fake_coiled["_drop_tiles"] = {"S05W060"}

        results = run_distributed(_jobs("N40W075", "S05W060"), storage=storage)

        by_status = {r.job.tile.name: r.status for r in results}
        assert by_status["N40W075"] == "completed"
        assert by_status["S05W060"] == "failed"
        failed = next(r for r in results if r.status == "failed")
        assert "retries" in failed.error

    def test_coiled_kwargs_pinned(self, fake_coiled, manifest_dir, monkeypatch):
        """Region, scaling, spot policy, and environ must never be defaults."""
        self._stub_worker(monkeypatch)
        storage = MagicMock()
        storage.list_completed.return_value = set()

        run_distributed(_jobs("N40W075"), storage=storage, run_id="test-run")

        assert fake_coiled["region"] == "us-west-2"
        assert fake_coiled["n_workers"] == 4
        assert fake_coiled["spot_policy"] == "spot_with_fallback"
        assert fake_coiled["vm_type"] == ["r6i.xlarge", "m6i.2xlarge"]
        assert fake_coiled["name"] == "lst-test-run"
        assert fake_coiled["tags"] == {"project": "landsat-lst", "run_id": "test-run"}
        environ = fake_coiled["environ"]
        assert environ["LST_STORAGE_BACKEND"] == "s3"
        assert environ["AWS_REQUEST_PAYER"] == "requester"
        assert environ["AWS_ACCESS_KEY_ID"] == "test-key"

    def test_stac_url_not_forwarded(self, fake_coiled, manifest_dir, monkeypatch):
        """A local Planetary Computer override must not leak onto AWS workers."""
        self._stub_worker(monkeypatch)
        monkeypatch.setenv("LST_STAC_URL", "https://planetarycomputer.example")
        monkeypatch.setenv("LST_S3_BUCKET", "custom-bucket")
        storage = MagicMock()
        storage.list_completed.return_value = set()

        run_distributed(_jobs("N40W075"), storage=storage)

        environ = fake_coiled["environ"]
        assert "LST_STAC_URL" not in environ
        assert environ["LST_S3_BUCKET"] == "custom-bucket"

    def test_no_cluster_when_nothing_to_run(self, fake_coiled, manifest_dir, monkeypatch):
        """A fully-completed window must not pay for a cluster at all."""
        self._stub_worker(monkeypatch)
        storage = MagicMock()
        storage.list_completed.return_value = {"N40W075"}

        results = run_distributed(_jobs("N40W075"), storage=storage)

        assert [r.status for r in results] == ["skipped"]
        assert "region" not in fake_coiled  # coiled.function never invoked

    def test_writes_manifest(self, fake_coiled, manifest_dir, monkeypatch):
        import json

        self._stub_worker(monkeypatch)
        storage = MagicMock()
        storage.list_completed.return_value = set()

        run_distributed(_jobs("N40W075"), storage=storage, run_id="manifest-run")

        payload = json.loads((manifest_dir / "manifest-run.json").read_text())
        assert payload["run_id"] == "manifest-run"
        assert payload["counts"]["completed"] == 1

    def test_worker_task_pins_threaded_scheduler(self, fake_coiled, manifest_dir, monkeypatch):
        """The tile graph must compute on the worker's own threads, never on
        the shared cluster scheduler (scheduler-connection-lost regression)."""
        import dask

        captured = {}

        def fake_process(job, force=False):
            captured["scheduler"] = dask.config.get("scheduler", None)
            return JobResult(job=job, status="completed")

        monkeypatch.setattr("landsat_lst.job.process_tile_job", fake_process)
        storage = MagicMock()
        storage.list_completed.return_value = set()

        run_distributed(_jobs("N40W075"), storage=storage)

        assert captured["scheduler"] == "threads"
