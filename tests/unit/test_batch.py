"""Unit tests for the Coiled Batch submission and reconciliation driver.

Every test stubs the ``coiled`` module: nothing here starts a cluster, and the
storage backend is always :class:`LocalStorage` under ``tmp_path``.
"""

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

from landsat_lst import pricing
from landsat_lst.batch import (
    BatchSubmission,
    _task_command,
    load_submission,
    reconcile_run,
    submission_path,
    submit_batch,
    wait_for_batch,
)
from landsat_lst.config import settings
from landsat_lst.models import ProcessingJob
from landsat_lst.storage import PRODUCTS, LocalStorage, collection_prefix
from landsat_lst.tiling import parse_tile_name


def _jobs(*tiles: str, year: int = 2021, end_year: int | None = 2025):
    return [ProcessingJob(tile=parse_tile_name(t), year=year, end_year=end_year) for t in tiles]


def _finish_tile(root, window: str, tile: str) -> None:
    """Write both COGs for one tile, the only proof of completion that counts."""
    for product in PRODUCTS:
        path = root / collection_prefix(window) / tile / f"{product}_{window}_{tile}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tif")


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(output_dir=tmp_path / "cogs")


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Point settings.manifest_dir at a scratch directory."""
    from landsat_lst.config import settings

    path = tmp_path / "runs"
    monkeypatch.setattr(settings, "manifest_dir", path)
    return path


@pytest.fixture
def fake_coiled(monkeypatch):
    """Install a coiled stub. Returns the dict of captured batch_run kwargs."""
    captured: dict = {}
    state: dict = {"jobs": []}

    def batch_run(**kwargs):
        captured.update(kwargs)
        return {"cluster_id": 4242, "cluster_name": kwargs.get("name"), "job_id": 77}

    fake = types.ModuleType("coiled")
    fake.batch_run = batch_run

    def status(cluster):
        state["queried_cluster"] = cluster
        return state["jobs"]

    def wait_for_job_done(job_id, timeout=None):
        state["waited_on"] = (job_id, timeout)
        return "done (success)"

    fake_batch = types.ModuleType("coiled.batch")
    fake_batch.status = status
    fake_batch.wait_for_job_done = wait_for_job_done
    fake.batch = fake_batch

    monkeypatch.setitem(sys.modules, "coiled", fake)
    monkeypatch.setitem(sys.modules, "coiled.batch", fake_batch)
    # Static creds so _worker_environ never reaches for boto3/SSO in tests.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")

    captured["_state"] = state
    return captured


def _set_tasks(fake_coiled, *tasks: dict) -> None:
    """Install the per-task state coiled.batch.status will report."""
    fake_coiled["_state"]["jobs"] = [{"tasks": list(tasks)}]


def _task(index: int, *, state="done", exit_code=0, start=None, stop=None) -> dict:
    return {
        "array_task_id": index,
        "state": state,
        "exit_code": exit_code,
        "start": start,
        "stop": stop,
    }


def _submit_run(storage, *tiles: str):
    """Submit one run under the fixed id the reconcile tests read back."""
    return submit_batch(_jobs(*tiles), storage=storage, run_id="r")


def _write_state(storage, tile: str, attempt: int | None, **fields) -> None:
    """Publish one tile state object, in the shape a VM writes it.

    ``attempt=None`` writes the unsuffixed key, which is both the pointer a
    settled tile copies its final state to and the whole of what a run written
    before attempts were numbered leaves behind.
    """
    body = {
        "schema": 2,
        "run_id": "r",
        "tile": tile,
        "window": "2021-2025",
        "attempt": attempt,
        "year": 2021,
        "end_year": 2025,
        "max_scenes": None,
        "phase": "composite_graph",
        "status": None,
        "elapsed_s": None,
        "duration_s": None,
        "peak_rss_mb": None,
        "scene_count": None,
        "lst_key": None,
        "qa_key": None,
        "error": None,
    }
    body.update(fields)
    storage.write_text(storage.run_record_key("r", tile, attempt), json.dumps(body))


def _manifest(runs_dir) -> dict:
    return json.loads((runs_dir / "r.json").read_text())


class TestTaskCommand:
    """The command string is the entire contract with the VM."""

    def test_multi_year_window(self):
        command = _task_command(run_id="r1", year=2021, end_year=2025, force=False)

        assert command == (
            "#!/bin/bash\n"
            "python -m landsat_lst.cli process --run-id r1 --year 2021 "
            '--end-year 2025 --tile "$COILED_BATCH_TASK_INPUT"\n'
        )

    def test_single_year_omits_end_year(self):
        command = _task_command(run_id="r1", year=2024, end_year=None, force=False)

        assert "--end-year" not in command
        assert "--year 2024" in command

    def test_force_is_forwarded(self):
        command = _task_command(run_id="r1", year=2021, end_year=2025, force=True)

        assert "--force" in command

    def test_task_input_is_not_expanded_locally(self):
        """The tile placeholder must survive as a literal for bash on the VM."""
        command = _task_command(run_id="r1", year=2021, end_year=2025, force=False)

        assert '"$COILED_BATCH_TASK_INPUT"' in command

    def test_is_a_script_so_coiled_ships_it_verbatim(self):
        """A list or a plain string is split and rejoined by Coiled, and the
        quotes around the tile placeholder did not survive that round trip: the
        CLI received --tile with literal quote characters, parse_tile_name
        rejected it, and the task died in 0.6s having written nothing. A "#!"
        command is passed through untouched.
        """
        command = _task_command(run_id="r1", year=2024, end_year=None, force=False)

        assert command.startswith("#!/bin/bash\n")
        assert command.endswith("\n")

    def test_max_scenes_is_forwarded(self):
        command = _task_command(run_id="r1", year=2021, end_year=2025, force=False, max_scenes=300)

        assert "--max-scenes 300" in command

    def test_every_job_field_reaches_the_vm(self):
        """The VM rebuilds its job from these arguments alone.

        A field the command omits does not travel: it silently takes its
        default on the worker. That turned a 300-scene sample into a full
        2,930-scene run which, from the submitting side, still looked like a
        sample. This fails when a field is added to ProcessingJob and not
        forwarded here.
        """
        from landsat_lst.models import ProcessingJob
        from landsat_lst.tiling import parse_tile_name

        job = ProcessingJob(
            tile=parse_tile_name("N40W075"), year=2021, end_year=2025, max_scenes=300
        )
        command = _task_command(
            run_id="r1",
            year=job.year,
            end_year=job.end_year,
            force=True,
            max_scenes=job.max_scenes,
        )

        for name in ProcessingJob.model_fields:
            # The tile is the one field supplied per task, through the input
            # variable rather than baked into the shared command.
            if name == "tile":
                continue
            value = getattr(job, name)
            if value is None:
                continue
            assert str(value) in command, f"{name} never reaches the VM"

    def test_run_id_is_quoted(self):
        """A run id is generated, but the command must not be shell-injectable."""
        command = _task_command(run_id="r1; rm -rf /", year=2021, end_year=None, force=False)

        assert "'r1; rm -rf /'" in command


class TestSubmit:
    def test_pins_every_coiled_knob(self, fake_coiled, runs_dir, storage):
        submit_batch(_jobs("N40W075"), storage=storage, run_id="test-run")

        assert fake_coiled["region"] == "us-west-2"
        assert fake_coiled["vm_type"] == ["r6i.2xlarge", "m6i.4xlarge"]
        assert fake_coiled["spot_policy"] == "spot_with_fallback"
        assert fake_coiled["max_workers"] == 4
        assert fake_coiled["max_retries"] == 3
        assert fake_coiled["job_timeout"] == "24 hours"
        assert fake_coiled["name"] == "lst-test-run"
        assert fake_coiled["tag"] == {"project": "landsat-lst", "run_id": "test-run"}
        assert fake_coiled["forward_aws_credentials"] is False

    def test_task_array_maps_over_tiles(self, fake_coiled, runs_dir, storage):
        submit_batch(_jobs("N40W075", "S05W060"), storage=storage, run_id="r")

        assert fake_coiled["map_over_values"] == ["N40W075", "S05W060"]
        assert fake_coiled["command"].startswith("#!/bin/bash\n")

    def test_forwards_worker_environment(self, fake_coiled, runs_dir, storage, monkeypatch):
        monkeypatch.setenv("LST_S3_BUCKET", "custom-bucket")
        monkeypatch.setenv("LST_STAC_URL", "https://planetarycomputer.example")

        submit_batch(_jobs("N40W075"), storage=storage, run_id="r")

        env = fake_coiled["env"]
        assert env["LST_STORAGE_BACKEND"] == "s3"
        assert env["AWS_REQUEST_PAYER"] == "requester"
        assert env["AWS_ACCESS_KEY_ID"] == "test-key"
        assert env["LST_S3_BUCKET"] == "custom-bucket"
        # A local Planetary Computer override must not leak onto AWS VMs.
        assert "LST_STAC_URL" not in env

    def test_resume_filters_completed_tiles(self, fake_coiled, runs_dir, storage):
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")

        submission = submit_batch(_jobs("N40W075", "S05W060"), storage=storage, run_id="r")

        assert submission.submitted_tiles == ["S05W060"]
        assert submission.skipped_tiles == ["N40W075"]
        assert fake_coiled["map_over_values"] == ["S05W060"]

    def test_force_bypasses_resume(self, fake_coiled, runs_dir, storage):
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")

        submission = submit_batch(_jobs("N40W075"), force=True, storage=storage, run_id="r")

        assert submission.submitted_tiles == ["N40W075"]
        assert "--force" in fake_coiled["command"]

    def test_no_cluster_when_everything_is_done(self, fake_coiled, runs_dir, storage):
        """A finished window must not pay for a cluster at all."""
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")

        submission = submit_batch(_jobs("N40W075"), storage=storage, run_id="r")

        assert submission.cluster_id is None
        assert "region" not in fake_coiled  # batch_run never called

    def test_writes_submission_record(self, fake_coiled, runs_dir, storage):
        submit_batch(_jobs("N40W075"), storage=storage, run_id="r")

        payload = json.loads((runs_dir / "r.submission.json").read_text())
        assert payload["cluster_id"] == 4242
        assert payload["job_id"] == 77
        assert payload["submitted_tiles"] == ["N40W075"]
        assert payload["window"] == "2021-2025"

    def test_submission_record_roundtrips(self, fake_coiled, runs_dir, storage):
        submitted = submit_batch(_jobs("N40W075"), storage=storage, run_id="r")

        assert load_submission("r") == submitted
        assert submission_path("r") == runs_dir / "r.submission.json"

    def test_generated_run_id_carries_the_window(self, fake_coiled, runs_dir, storage):
        submission = submit_batch(_jobs("N40W075"), storage=storage)

        assert submission.run_id.startswith("2021-2025-")

    def test_rejects_mixed_windows(self, fake_coiled, runs_dir, storage):
        jobs = _jobs("N40W075") + _jobs("S05W060", year=2024, end_year=None)

        with pytest.raises(ValueError, match="one window per call"):
            submit_batch(jobs, storage=storage)

    def test_rejects_empty_job_list(self, fake_coiled, runs_dir, storage):
        with pytest.raises(ValueError, match="No jobs"):
            submit_batch([], storage=storage)


class TestReconcile:
    def _submit(self, fake_coiled, storage, *tiles: str):
        return submit_batch(_jobs(*tiles), storage=storage, run_id="r")

    def test_completion_comes_from_the_cog_listing(self, fake_coiled, runs_dir, storage):
        self._submit(fake_coiled, storage, "N40W075", "S05W060")
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        _set_tasks(fake_coiled, _task(0), _task(1))

        results = reconcile_run("r", storage=storage)

        assert {r.job.tile.name: r.status for r in results} == {
            "N40W075": "completed",
            "S05W060": "failed",
        }
        assert fake_coiled["_state"]["queried_cluster"] == 4242

    def test_zero_exit_without_cogs_is_a_failure(self, fake_coiled, runs_dir, storage):
        """A clean exit that produced no bytes is not a completed tile."""
        self._submit(fake_coiled, storage, "N40W075")
        _set_tasks(fake_coiled, _task(0, state="done", exit_code=0))

        (result,) = reconcile_run("r", storage=storage)

        assert result.status == "failed"
        assert "no COGs written" in result.error

    def test_records_supply_the_costing_metrics(self, fake_coiled, runs_dir, storage):
        self._submit(fake_coiled, storage, "N40W075")
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        storage.write_text(
            storage.run_record_key("r", "N40W075"),
            json.dumps(
                {
                    "tile": "N40W075",
                    "year": 2021,
                    "end_year": 2025,
                    "status": "completed",
                    "lst_key": "lst-p95-2021-2025/N40W075/lst_p95_2021-2025_N40W075.tif",
                    "qa_key": "lst-p95-2021-2025/N40W075/qa_count_2021-2025_N40W075.tif",
                    "error": None,
                    "duration_s": 1834.2,
                    "scene_count": 907,
                    "peak_rss_mb": 41200.0,
                }
            ),
        )
        _set_tasks(fake_coiled, _task(0))

        (result,) = reconcile_run("r", storage=storage)

        assert result.status == "completed"
        assert result.duration_s == 1834.2
        assert result.scene_count == 907
        assert result.peak_rss_mb == 41200.0
        assert result.lst_key.endswith("lst_p95_2021-2025_N40W075.tif")

    def test_record_error_explains_a_failure(self, fake_coiled, runs_dir, storage):
        self._submit(fake_coiled, storage, "N40W075")
        storage.write_text(
            storage.run_record_key("r", "N40W075"),
            json.dumps(
                {
                    "tile": "N40W075",
                    "year": 2021,
                    "end_year": 2025,
                    "status": "failed",
                    "error": "No scenes found for the window",
                    "duration_s": 12.0,
                }
            ),
        )
        _set_tasks(fake_coiled, _task(0, state="error", exit_code=1))

        (result,) = reconcile_run("r", storage=storage)

        assert result.status == "failed"
        assert result.error == "No scenes found for the window"
        assert result.duration_s == 12.0

    def test_task_state_explains_a_tile_with_no_record(self, fake_coiled, runs_dir, storage):
        """A VM killed before it could report still has to be explained."""
        self._submit(fake_coiled, storage, "N40W075")
        _set_tasks(fake_coiled, _task(0, state="error", exit_code=137))

        (result,) = reconcile_run("r", storage=storage)

        assert "exited 137" in result.error

    def test_failure_names_the_uploaded_task_log(self, fake_coiled, runs_dir, storage):
        """Coiled reports the tee wrapper's exit code, so the log is the evidence."""
        self._submit(fake_coiled, storage, "N40W075")
        storage.write_text(
            storage.log_key("r", "N40W075"), "Traceback...", content_type="text/plain"
        )
        _set_tasks(fake_coiled, _task(0, state="error", exit_code=1))

        (result,) = reconcile_run("r", storage=storage)

        assert "_runs/r/N40W075.log" in result.error

    def test_failure_without_a_log_does_not_promise_one(self, fake_coiled, runs_dir, storage):
        self._submit(fake_coiled, storage, "N40W075")
        _set_tasks(fake_coiled, _task(0, state="error", exit_code=137))

        (result,) = reconcile_run("r", storage=storage)

        assert "task log" not in result.error

    def test_records_are_read_only_where_they_exist(self, fake_coiled, runs_dir, storage):
        """A 700-tile run must not spend a request discovering each absence."""
        self._submit(fake_coiled, storage, "N40W075", "S05W060")
        storage.write_text(
            storage.run_record_key("r", "N40W075"),
            json.dumps({"tile": "N40W075", "year": 2021, "end_year": 2025, "status": "failed"}),
        )
        reads: list[str] = []
        original = storage.read_text
        storage.read_text = lambda key: (reads.append(key), original(key))[1]
        _set_tasks(fake_coiled, _task(0), _task(1))

        reconcile_run("r", storage=storage)

        assert reads == [storage.run_record_key("r", "N40W075")]

    def test_tile_never_scheduled_is_explained(self, fake_coiled, runs_dir, storage):
        self._submit(fake_coiled, storage, "N40W075")
        _set_tasks(fake_coiled)

        (result,) = reconcile_run("r", storage=storage)

        assert "never have been scheduled" in result.error

    def test_completed_tile_without_a_record_still_carries_keys(
        self, fake_coiled, runs_dir, storage
    ):
        """The manifest is the catalog's shopping list; keys cannot be null."""
        self._submit(fake_coiled, storage, "N40W075")
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        _set_tasks(fake_coiled, _task(0))

        (result,) = reconcile_run("r", storage=storage)

        assert result.lst_key == "lst-p95-2021-2025/N40W075/lst_p95_2021-2025_N40W075.tif"
        assert result.qa_key == "lst-p95-2021-2025/N40W075/qa_count_2021-2025_N40W075.tif"

    def test_duration_falls_back_to_task_timings(self, fake_coiled, runs_dir, storage):
        self._submit(fake_coiled, storage, "N40W075")
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        _set_tasks(
            fake_coiled,
            _task(0, start="2026-08-12T14:00:00+00:00", stop="2026-08-12T14:30:00+00:00"),
        )

        (result,) = reconcile_run("r", storage=storage)

        assert result.duration_s == 1800.0

    def test_one_failed_tile_does_not_abort_the_others(self, fake_coiled, runs_dir, storage):
        self._submit(fake_coiled, storage, "N40W075", "S05W060", "N60W150")
        for tile in ("N40W075", "N60W150"):
            _finish_tile(storage.output_dir, "2021-2025", tile)
        _set_tasks(fake_coiled, _task(0), _task(1, state="error", exit_code=1), _task(2))

        results = reconcile_run("r", storage=storage)

        assert sum(1 for r in results if r.status == "completed") == 2
        assert sum(1 for r in results if r.status == "failed") == 1

    def test_skipped_tiles_stay_in_the_manifest(self, fake_coiled, runs_dir, storage):
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        submit_batch(_jobs("N40W075", "S05W060"), storage=storage, run_id="r")
        _finish_tile(storage.output_dir, "2021-2025", "S05W060")
        _set_tasks(fake_coiled, _task(0))

        results = reconcile_run("r", storage=storage)

        assert {r.job.tile.name: r.status for r in results} == {
            "S05W060": "completed",
            "N40W075": "skipped",
        }

    def test_writes_the_manifest(self, fake_coiled, runs_dir, storage):
        self._submit(fake_coiled, storage, "N40W075")
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        _set_tasks(fake_coiled, _task(0))

        reconcile_run("r", storage=storage)

        payload = json.loads((runs_dir / "r.json").read_text())
        assert payload["run_id"] == "r"
        assert payload["cluster_id"] == 4242
        assert payload["job_id"] == 77
        assert payload["counts"]["completed"] == 1

    def test_is_repeatable(self, fake_coiled, runs_dir, storage):
        """Reconciling twice must not change the verdict."""
        self._submit(fake_coiled, storage, "N40W075")
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        _set_tasks(fake_coiled, _task(0))

        first = reconcile_run("r", storage=storage)
        second = reconcile_run("r", storage=storage)

        assert [r.status for r in first] == [r.status for r in second]

    def test_survives_coiled_being_unreachable(self, fake_coiled, runs_dir, storage, monkeypatch):
        """S3 still knows what finished when the control plane does not answer."""
        self._submit(fake_coiled, storage, "N40W075")
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")

        def explode(cluster):
            msg = f"coiled unreachable for cluster {cluster}"
            raise ConnectionError(msg)

        monkeypatch.setattr(sys.modules["coiled.batch"], "status", explode)

        (result,) = reconcile_run("r", storage=storage)

        assert result.status == "completed"

    def test_unreadable_record_does_not_stop_reconciliation(self, fake_coiled, runs_dir, storage):
        self._submit(fake_coiled, storage, "N40W075")
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        storage.write_text(storage.run_record_key("r", "N40W075"), "{not json")
        _set_tasks(fake_coiled, _task(0))

        (result,) = reconcile_run("r", storage=storage)

        assert result.status == "completed"

    def test_unknown_run_id_is_a_clear_error(self, fake_coiled, runs_dir, storage):
        with pytest.raises(FileNotFoundError, match="No submission record"):
            reconcile_run("never-submitted", storage=storage)


class TestWait:
    def test_returns_final_state(self, fake_coiled, runs_dir, storage):
        submit_batch(_jobs("N40W075"), storage=storage, run_id="r")

        assert wait_for_batch("r", timeout_s=30) == "done (success)"
        assert fake_coiled["_state"]["waited_on"] == (77, 30)

    def test_nothing_to_wait_for(self, fake_coiled, runs_dir, storage):
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        submit_batch(_jobs("N40W075"), storage=storage, run_id="r")

        assert wait_for_batch("r") is None


def test_submission_dashboard_url():
    submission = BatchSubmission(
        run_id="r",
        window="2021-2025",
        cluster_id=4242,
        job_id=77,
        submitted_at="2026-08-12T14:00:00+00:00",
        submitted_tiles=["N40W075"],
        year=2021,
        end_year=2025,
    )

    assert submission.dashboard_url == "https://cloud.coiled.io/clusters/4242"


def test_missing_coiled_is_reported_clearly(runs_dir, storage, monkeypatch):
    monkeypatch.setitem(sys.modules, "coiled", None)

    with pytest.raises(ImportError, match="Coiled is required"):
        submit_batch(_jobs("N40W075"), storage=storage)


class TestAttemptSeries:
    """A retried tile has to explain every VM the run paid for."""

    def test_reports_every_attempt(self, fake_coiled, runs_dir, storage):
        """Attempt 1 failed, attempt 2 died mid-land_mask, attempt 3 wrote the tile.

        The middle attempt never settled, so it has no status and no duration.
        Dropping it would hide the most expensive half hour of the tile, which
        is what the shared-key layout used to do.
        """
        _submit_run(storage, "N40W075")
        _write_state(
            storage,
            "N40W075",
            1,
            phase="loading",
            status="failed",
            duration_s=612.0,
            peak_rss_mb=18000.0,
            error="RasterioIOError on scene 41",
        )
        _write_state(
            storage, "N40W075", 2, phase="land_mask", elapsed_s=1980.0, peak_rss_mb=48000.0
        )
        _write_state(
            storage,
            "N40W075",
            3,
            phase="uploading",
            status="completed",
            duration_s=2100.0,
            peak_rss_mb=41200.0,
            scene_count=907,
        )
        storage.write_text(storage.log_key("r", "N40W075", 2), "Killed", content_type="text/plain")
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        _set_tasks(fake_coiled, _task(0))

        (result,) = reconcile_run("r", storage=storage)

        assert result.status == "completed"
        assert result.duration_s == 2100.0
        tile = _manifest(runs_dir)["tiles"][0]
        assert tile["attempt"] == 3
        assert [row["attempt"] for row in tile["attempts"]] == [1, 2, 3]
        assert tile["attempts"][1] == {
            "attempt": 2,
            "phase": "land_mask",
            "status": None,
            "duration_s": 1980.0,
            "peak_rss_mb": 48000.0,
            "error": None,
            "log_key": "_runs/r/N40W075.2.log",
        }

    def test_summarises_retries_across_the_run(self, fake_coiled, runs_dir, storage):
        _submit_run(storage, "N40W075", "S05W060")
        _write_state(storage, "N40W075", 1, status="failed", duration_s=10.0, error="boom")
        _write_state(storage, "N40W075", 2, status="completed", duration_s=800.0)
        _write_state(storage, "S05W060", 1, status="completed", duration_s=700.0)
        for tile in ("N40W075", "S05W060"):
            _finish_tile(storage.output_dir, "2021-2025", tile)
        _set_tasks(fake_coiled, _task(0), _task(1))

        reconcile_run("r", storage=storage)

        assert _manifest(runs_dir)["attempts"] == {"tiles_retried": 1, "total": 3, "max": 2}

    def test_verdict_comes_from_the_newest_attempt(self, fake_coiled, runs_dir, storage):
        """An earlier attempt's error must not outlive the attempt that worked."""
        _submit_run(storage, "N40W075")
        _write_state(storage, "N40W075", 1, status="failed", duration_s=10.0, error="boom")
        _write_state(storage, "N40W075", 2, status="completed", duration_s=812.3)
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        _set_tasks(fake_coiled, _task(0))

        (result,) = reconcile_run("r", storage=storage)

        assert result.error is None
        assert result.duration_s == 812.3

    def test_healthy_tile_costs_one_read(self, fake_coiled, runs_dir, storage):
        """A settled tile publishes its attempt and a copy at the pointer key.

        Reading both would double the request count of a 700-tile run to buy a
        second copy of the same body.
        """
        _submit_run(storage, "N40W075")
        _write_state(storage, "N40W075", 1, status="completed", duration_s=800.0)
        _write_state(storage, "N40W075", None, status="completed", duration_s=800.0)
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        reads: list[str] = []
        original = storage.read_text
        storage.read_text = lambda key: (reads.append(key), original(key))[1]
        _set_tasks(fake_coiled, _task(0))

        reconcile_run("r", storage=storage)

        assert reads == [storage.run_record_key("r", "N40W075", 1)]

    def test_legacy_run_reconciles_unchanged(self, fake_coiled, runs_dir, storage):
        """A run written before attempts were numbered still reads back whole.

        Its outcome lived at the unsuffixed key and its liveness at a separate
        ``.progress.json``. Neither carries an attempt number, so the series is
        empty rather than invented.
        """
        _submit_run(storage, "N40W075")
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        storage.write_text(
            storage.run_record_key("r", "N40W075"),
            json.dumps(
                {
                    "tile": "N40W075",
                    "year": 2021,
                    "end_year": 2025,
                    "status": "completed",
                    "duration_s": 1834.2,
                    "scene_count": 907,
                    "peak_rss_mb": 41200.0,
                }
            ),
        )
        storage.write_text(
            f"{storage.run_prefix('r')}N40W075.progress.json",
            json.dumps({"tile": "N40W075", "phase": "exporting"}),
        )
        _set_tasks(fake_coiled, _task(0))

        (result,) = reconcile_run("r", storage=storage)

        assert result.status == "completed"
        assert result.duration_s == 1834.2
        tile = _manifest(runs_dir)["tiles"][0]
        assert tile["attempts"] == []
        assert tile["attempt"] == 0


class TestCost:
    """What the run cost, as a range that says how much of it is assumed."""

    def test_prices_a_tile_from_its_own_instance_type(self, fake_coiled, runs_dir, storage):
        """An on-demand VM of a known type is a published rate, so it is a point."""
        _submit_run(storage, "N40W075")
        _write_state(
            storage,
            "N40W075",
            1,
            status="completed",
            duration_s=3600.0,
            instance_type="m6i.4xlarge",
            instance_lifecycle="on-demand",
        )
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        _set_tasks(fake_coiled, _task(0))

        reconcile_run("r", storage=storage)

        payload = _manifest(runs_dir)
        cost = payload["tiles"][0]["cost"]
        assert cost["instance_type"] == "m6i.4xlarge"
        assert cost["lifecycle"] == "on-demand"
        assert cost["low"] == cost["high"] == 0.768
        assert cost["provenance"] == "derived"
        assert payload["cost"]["priced_tiles"] == 1
        assert payload["cost"]["total"]["low"] == 0.768
        assert payload["cost"]["fleet"]["tiles"] == pricing.FLEET_TILES
        assert payload["cost"]["fleet"]["observed_tiles"] == 1
        assert payload["cost"]["disclaimer"] == pricing.DISCLAIMER

    def test_tile_without_an_instance_type_is_assumed(self, fake_coiled, runs_dir, storage):
        """A VM that never said what it was is priced as the type the fleet asks
        for first, on a lifecycle the purchase policy does not pin down. Both
        substitutions widen the band and both are labelled.
        """
        _submit_run(storage, "N40W075")
        _write_state(storage, "N40W075", 1, status="completed", duration_s=1800.0)
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        _set_tasks(fake_coiled, _task(0))

        reconcile_run("r", storage=storage)

        cost = _manifest(runs_dir)["tiles"][0]["cost"]
        assert cost["instance_type"] == settings.coiled_vm_types[0]
        assert cost["lifecycle"] == "unknown"
        assert cost["provenance"] == "assumed"
        assert cost["low"] < cost["high"]

    def test_short_failure_is_billed_a_full_minute(self, fake_coiled, runs_dir, storage):
        """A task that died in ten seconds still launched an instance."""
        _submit_run(storage, "N40W075")
        _write_state(storage, "N40W075", 1, status="failed", duration_s=10.375, error="boom")
        _set_tasks(fake_coiled, _task(0, state="error", exit_code=1))

        reconcile_run("r", storage=storage)

        assert _manifest(runs_dir)["tiles"][0]["cost"]["billed_s"] == 60.0

    def test_unpriceable_run_projects_nothing(self, fake_coiled, runs_dir, storage):
        """Zero dollars for 700 tiles is a claim, and it would be false."""
        _submit_run(storage, "N40W075")
        _set_tasks(fake_coiled, _task(0))

        reconcile_run("r", storage=storage)

        payload = _manifest(runs_dir)
        assert payload["tiles"][0]["cost"] is None
        assert payload["cost"]["priced_tiles"] == 0
        assert payload["cost"]["total"] is None
        assert payload["cost"]["fleet"] is None


class TestPlanComparison:
    """The floor the run was submitted expecting, against what it reached."""

    def test_submission_stores_a_plan(self, fake_coiled, runs_dir, storage, monkeypatch):
        monkeypatch.setattr(settings, "dask_max_threads", 4)

        submission = _submit_run(storage, "N40W075")

        assert submission.plan["threads"] == 4
        assert submission.plan["threads_source"] == "settings.dask_max_threads"
        assert submission.plan["phases"]["composite"]["floor_gib"] > 0
        assert load_submission("r").plan == submission.plan

    def test_plan_says_when_the_thread_count_came_from_this_host(
        self, fake_coiled, runs_dir, storage, monkeypatch
    ):
        """The default reads the submitting laptop's cores, not the VM's."""
        monkeypatch.setattr(settings, "dask_max_threads", None)

        submission = _submit_run(storage, "N40W075")

        assert submission.plan["threads_source"] == "cpu_count of the submitting host"

    def test_plan_failure_does_not_stop_a_submission(
        self, fake_coiled, runs_dir, storage, monkeypatch
    ):
        def explode(**kwargs):
            msg = "no geometry for you"
            raise RuntimeError(msg)

        monkeypatch.setattr("landsat_lst.profiling.plan_memory_record", explode)

        submission = _submit_run(storage, "N40W075")

        assert submission.plan is None
        assert submission.cluster_id == 4242

    def test_manifest_reports_a_ratio_rather_than_a_verdict(self, fake_coiled, runs_dir, storage):
        """The floor is a lower bound, so a run above it is the ordinary case.

        The gap is the finding: a 300-scene N40W075 sample peaked at 78.6 GB
        against a floor of a few GB, which is how the offset pass was found to
        fan out rather than stream.
        """
        _submit_run(storage, "N40W075")
        _write_state(
            storage,
            "N40W075",
            1,
            status="completed",
            duration_s=2100.0,
            peak_rss_mb=51200.0,
            scene_count=907,
        )
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        _set_tasks(fake_coiled, _task(0))

        reconcile_run("r", storage=storage)

        plan = _manifest(runs_dir)["plan"]
        memory = plan["memory"]
        assert set(memory) == {
            "floor_gib",
            "floor_phase",
            "observed_peak_gib",
            "observed_tiles",
            "ratio",
        }
        assert memory["observed_peak_gib"] == 50.0
        assert memory["ratio"] == round(50.0 / memory["floor_gib"], 2)
        assert memory["floor_phase"] in {"destripe_offsets", "composite"}
        assert plan["scenes"]["planned"] == plan["planned"]["scenes"]
        assert plan["scenes"]["observed_max"] == 907

    def test_no_plan_block_when_the_submission_stored_none(self, fake_coiled, runs_dir, storage):
        """An old submission record has no plan, and inventing one is worse."""
        _submit_run(storage, "N40W075")
        path = submission_path("r")
        payload = json.loads(path.read_text())
        del payload["plan"]
        path.write_text(json.dumps(payload))
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")
        _set_tasks(fake_coiled, _task(0))

        reconcile_run("r", storage=storage)

        assert "plan" not in _manifest(runs_dir)


def test_submission_roundtrips_with_and_without_a_plan():
    fields = {
        "run_id": "r",
        "window": "2021-2025",
        "cluster_id": 4242,
        "job_id": 77,
        "submitted_at": "2026-08-12T14:00:00+00:00",
        "submitted_tiles": ["N40W075"],
        "year": 2021,
        "end_year": 2025,
    }
    planned = BatchSubmission(**fields, plan={"scenes": 2930, "phases": {}})
    unplanned = BatchSubmission(**fields)

    assert BatchSubmission.from_dict(planned.to_dict()) == planned
    assert BatchSubmission.from_dict(unplanned.to_dict()) == unplanned

    legacy = unplanned.to_dict()
    del legacy["plan"]
    assert BatchSubmission.from_dict(legacy) == unplanned


def test_credentials_are_resolved_only_when_a_cluster_starts(
    fake_coiled, runs_dir, storage, monkeypatch
):
    """A no-op resume must not demand an SSO login."""
    _finish_tile(storage.output_dir, "2021-2025", "N40W075")
    environ = MagicMock(side_effect=RuntimeError("should not be called"))
    monkeypatch.setattr("landsat_lst.batch._worker_environ", environ)

    submit_batch(_jobs("N40W075"), storage=storage, run_id="r")

    environ.assert_not_called()


class TestShardStageCommand:
    """The command a shard task runs, which is the whole contract with its VM."""

    def test_the_index_is_the_task_input(self):
        from landsat_lst.batch import _shard_task_command

        command = _shard_task_command(stage="composite", run_id="r1", tile="N40W075")

        assert command == (
            "#!/bin/bash\n"
            "python -m landsat_lst.cli shard composite --run-id r1 --tile N40W075 "
            '--index "$COILED_BATCH_TASK_INPUT"\n'
        )

    def test_not_the_array_task_id(self):
        """``COILED_ARRAY_TASK_ID`` is identical on every retry (issue #66).

        Here the index selects which slice of the tile the task owns, so an
        index that meant something else on a retry would recompute the wrong
        slab into the right key.
        """
        from landsat_lst.batch import _shard_task_command

        command = _shard_task_command(stage="offsets", run_id="r1", tile="N40W075")

        assert "COILED_ARRAY_TASK_ID" not in command

    def test_resolve_carries_the_window_because_no_plan_exists_yet(self):
        from landsat_lst.batch import _shard_task_command

        job = _jobs("N40W075")[0]
        command = _shard_task_command(stage="resolve", run_id="r1", tile="N40W075", job=job)

        assert "--year 2021 --end-year 2025" in command

    def test_a_sampled_window_restates_max_scenes(self):
        """A missing --max-scenes resolves a different scene set entirely."""
        from landsat_lst.batch import _shard_task_command
        from landsat_lst.models import ProcessingJob

        job = ProcessingJob(
            tile=parse_tile_name("N40W075"), year=2021, end_year=2025, max_scenes=300
        )
        command = _shard_task_command(stage="resolve", run_id="r1", tile="N40W075", job=job)

        assert "--max-scenes 300" in command


class TestSubmitShardStage:
    """One array per stage, because batch_run has no dependency mechanism."""

    def test_pins_the_knobs_and_maps_over_indexes(self, fake_coiled):
        from landsat_lst.batch import submit_shard_stage

        submission = submit_shard_stage(
            stage="climatology", run_id="r1", tile="N40W075", indexes=[0, 2, 5]
        )

        assert fake_coiled["map_over_values"] == ["0", "2", "5"]
        assert fake_coiled["region"] == settings.coiled_region
        assert fake_coiled["tag"]["stage"] == "climatology"
        assert submission.cluster_id == 4242

    def test_the_fleet_width_is_the_shard_count_not_the_tile_cost_cap(self, fake_coiled):
        """``coiled_max_workers`` bounds a 700-tile fleet, where queueing is the
        point. Here the whole reason to shard is that the pieces run at once.
        """
        from landsat_lst.batch import submit_shard_stage

        submit_shard_stage(stage="offsets", run_id="r1", tile="N40W075", indexes=list(range(9)))

        assert fake_coiled["max_workers"] == 9
        assert settings.coiled_max_workers < 9

    def test_shard_stages_never_fall_back_to_on_demand_silently(self, fake_coiled):
        """The budget holds only at spot prices ($1.9-4.7k spot vs $6.2k
        on-demand at measured counts). ``spot_with_fallback`` would let a
        capacity shortfall convert the build to the on-demand bill with
        nobody deciding it; under ``spot`` the shortfall surfaces as missing
        shards, which the driver's bounded rounds retry and then fail loudly.
        """
        from landsat_lst.batch import submit_shard_stage

        submit_shard_stage(stage="offsets", run_id="r1", tile="N40W075", indexes=[0])

        assert fake_coiled["spot_policy"] == settings.shard_spot_policy
        assert settings.shard_spot_policy == "spot"
        # The tile fleet keeps its own policy; only shard stages tighten.
        assert settings.coiled_spot_policy == "spot_with_fallback"

    def test_the_composite_stage_takes_its_own_vm_and_chunk(self, fake_coiled):
        from landsat_lst.batch import submit_shard_stage

        submit_shard_stage(stage="composite", run_id="r1", tile="N40W075", indexes=[0, 1])

        assert fake_coiled["vm_type"] == [settings.shard_composite_vm_type]
        assert fake_coiled["env"]["LST_LOAD_CHUNK_SIZE"] == str(settings.shard_composite_chunk)

    def test_composite_stage_does_not_force_profiler_on_the_vm(self, fake_coiled, monkeypatch):
        from landsat_lst.batch import submit_shard_stage

        monkeypatch.delenv("LST_PROFILE_DASK", raising=False)
        submit_shard_stage(stage="composite", run_id="r1", tile="N40W075", indexes=[0])

        assert "LST_PROFILE_DASK" not in fake_coiled["env"]

    def test_composite_stage_preserves_explicit_profile_opt_out(self, fake_coiled, monkeypatch):
        from landsat_lst.batch import submit_shard_stage

        monkeypatch.setenv("LST_PROFILE_DASK", "0")
        submit_shard_stage(stage="composite", run_id="r1", tile="N40W075", indexes=[0])

        assert fake_coiled["env"]["LST_PROFILE_DASK"] == "0"

    def test_the_export_stage_asks_for_scratch_disk(self, fake_coiled):
        """It holds every band slab, a full-tile intermediate, and a COG at once."""
        from landsat_lst.batch import submit_shard_stage

        submit_shard_stage(stage="export", run_id="r1", tile="N40W075", indexes=[0])

        assert fake_coiled["disk_size"] == settings.shard_export_disk_gb

    def test_the_offset_stages_keep_the_default_vm_preference(self, fake_coiled):
        from landsat_lst.batch import submit_shard_stage

        submit_shard_stage(stage="climatology", run_id="r1", tile="N40W075", indexes=[0])

        assert fake_coiled["vm_type"] == settings.coiled_vm_types
        assert "disk_size" not in fake_coiled

    def test_the_cluster_name_carries_the_round(self, fake_coiled):
        """Coiled refuses a name that matches a running cluster, and a resumed
        driver resubmitting a stage whose first cluster is still in flight hits
        exactly that.
        """
        from landsat_lst.batch import submit_shard_stage

        submit_shard_stage(
            stage="climatology", run_id="r1", tile="N40W075", indexes=[1], submission_round=2
        )

        assert fake_coiled["name"].endswith("-r2")
        assert fake_coiled["tag"]["round"] == "2"

    def test_two_rounds_of_one_stage_never_share_a_name(self):
        from landsat_lst.batch import stage_cluster_name

        args = ("shard-S30W065-2021-2025-20260821T194111Z", "S30W065", "climatology")

        assert stage_cluster_name(*args, 1) != stage_cluster_name(*args, 2)

    def test_the_name_survives_truncation(self):
        """The observed collision was already cut mid-stage at 60 characters,
        so a round marker appended to that shape would have been eaten. The run
        id is hashed instead -- it already holds the tile and the window.
        """
        from landsat_lst.batch import _CLUSTER_NAME_MAX, stage_cluster_name

        name = stage_cluster_name(
            "shard-S30W065-2021-2025-20260821T194111Z", "S30W065", "climatology", 3
        )

        assert len(name) < _CLUSTER_NAME_MAX
        assert name.endswith("-r3")
        assert "S30W065" in name

    def test_an_unknown_stage_is_refused(self, fake_coiled):
        from landsat_lst.batch import submit_shard_stage

        with pytest.raises(ValueError, match="unknown shard stage"):
            submit_shard_stage(stage="polish", run_id="r1", tile="N40W075", indexes=[0])

    def test_an_empty_submission_is_refused(self, fake_coiled):
        """The driver skips a finished stage rather than submitting nothing."""
        from landsat_lst.batch import submit_shard_stage

        with pytest.raises(ValueError, match="no shards to submit"):
            submit_shard_stage(stage="climatology", run_id="r1", tile="N40W075", indexes=[])


class TestSubmitFleetStage:
    """One array per stage per wave, carrying many tiles. See ADR-018."""

    def test_maps_over_tile_qualified_tokens(self, fake_coiled):
        from landsat_lst.batch import submit_fleet_stage

        submission = submit_fleet_stage(
            stage="composite",
            run_id="r1",
            units=[("N40W075", 0), ("N40W075", 1), ("N35W080", 0)],
            wave=2,
        )

        assert fake_coiled["map_over_values"] == ["N40W075:0", "N40W075:1", "N35W080:0"]
        assert fake_coiled["tag"]["wave"] == "2"
        assert fake_coiled["tag"]["fleet"] == "1"
        assert submission.tiles == ["N40W075", "N35W080"]

    def test_the_cap_bounds_workers_and_never_the_unit_list(self, fake_coiled):
        """Where the boot saving comes from: surplus units queue onto booted VMs.

        Splitting the unit list to fit the cap would instead write a submission
        record for a partial index set and cost the remainder a barrier round.
        """
        from landsat_lst.batch import submit_fleet_stage

        units = [(f"N{30 + i}W075", 0) for i in range(10)]
        submit_fleet_stage(stage="offsets", run_id="r1", units=units, max_workers=3)

        assert fake_coiled["max_workers"] == 3
        assert len(fake_coiled["map_over_values"]) == 10

    def test_a_consolidated_wave_never_falls_back_to_on_demand(self, fake_coiled):
        """The one failure mode that converts a spot build into the on-demand bill.

        The consolidated path must not acquire a fallback the per-tile path does
        not have, which is why neither submitter spells the policy itself.
        """
        from landsat_lst.batch import submit_fleet_stage

        submit_fleet_stage(stage="offsets", run_id="r1", units=[("N40W075", 0)])

        assert fake_coiled["spot_policy"] == settings.shard_spot_policy
        assert settings.shard_spot_policy == "spot"

    def test_the_composite_stage_keeps_its_vm_and_chunk(self, fake_coiled):
        from landsat_lst.batch import submit_fleet_stage

        submit_fleet_stage(stage="composite", run_id="r1", units=[("N40W075", 0)])

        assert fake_coiled["vm_type"] == [settings.shard_composite_vm_type]
        assert fake_coiled["env"]["LST_LOAD_CHUNK_SIZE"] == str(settings.shard_composite_chunk)

    def test_the_export_stage_keeps_its_scratch_disk(self, fake_coiled):
        from landsat_lst.batch import submit_fleet_stage

        submit_fleet_stage(stage="export", run_id="r1", units=[("N40W075", 0)])

        assert fake_coiled["disk_size"] == settings.shard_export_disk_gb

    def test_a_later_wave_cannot_collide_with_a_live_cluster(self, fake_coiled):
        from landsat_lst.batch import _CLUSTER_NAME_MAX, fleet_cluster_name

        first = fleet_cluster_name("fleet-20260901T120000Z", "composite", 1)
        second = fleet_cluster_name("fleet-20260901T120000Z", "composite", 2)

        assert first != second
        assert second.endswith("-w2")
        assert len(second) < _CLUSTER_NAME_MAX

    def test_the_command_carries_no_window_because_the_roster_does(self, fake_coiled):
        """One command serves every tile, so per-tile parameters cannot ride on it."""
        from landsat_lst.batch import _fleet_task_command

        command = _fleet_task_command(stage="offsets", run_id="r1", units=8)

        assert "shard unit" in command
        assert "--units 8" in command
        assert "--year" not in command
        assert "COILED_BATCH_TASK_INPUT" in command

    def test_a_malformed_token_is_refused_rather_than_parsed_loosely(self):
        """A loose parse computes the wrong slab and publishes it under a good key."""
        from landsat_lst.shards import parse_fleet_unit

        for bad in ("N40W075", "N40W075:", ":3", "N40W075:x", ""):
            with pytest.raises(ValueError, match="malformed fleet unit token"):
                parse_fleet_unit(bad)

    def test_an_empty_wave_is_refused(self, fake_coiled):
        from landsat_lst.batch import submit_fleet_stage

        with pytest.raises(ValueError, match="no units to submit"):
            submit_fleet_stage(stage="offsets", run_id="r1", units=[])

    def test_an_unknown_stage_is_refused(self, fake_coiled):
        from landsat_lst.batch import submit_fleet_stage

        with pytest.raises(ValueError, match="unknown shard stage"):
            submit_fleet_stage(stage="polish", run_id="r1", units=[("N40W075", 0)])


class TestWorkerEnvironContract:
    """``LST_EVIDENCE_CONTRACT`` binds the launch checkout on every submission."""

    @staticmethod
    def _repo(tmp_path, monkeypatch):
        from tests.unit.test_evidence_contract import git_repo, valid_contract

        root, baseline, head = git_repo(tmp_path)
        contract = valid_contract(tmp_path, baseline_revision=baseline, treatment_revision=head)
        # Untracked, and outside src/: exactly the kind of file the binding ignores.
        (root / "contract.json").write_text(json.dumps(contract))
        monkeypatch.chdir(root)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
        monkeypatch.setenv("LST_EVIDENCE_CONTRACT", "contract.json")
        return root, head

    def test_an_operators_revision_is_forwarded_when_no_contract_is_set(self, monkeypatch):
        from landsat_lst.job import _worker_environ

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
        monkeypatch.delenv("LST_EVIDENCE_CONTRACT", raising=False)
        monkeypatch.setenv("LST_CODE_REVISION", "a" * 40)

        assert _worker_environ()["LST_CODE_REVISION"] == "a" * 40

    def test_the_contract_binds_head_of_the_launch_checkout(self, tmp_path, monkeypatch):
        """The bound revision wins over an exported one, and it is cwd's HEAD."""
        from landsat_lst.job import _worker_environ

        _root, head = self._repo(tmp_path, monkeypatch)
        monkeypatch.setenv("LST_CODE_REVISION", "a" * 40)

        environ = _worker_environ()
        assert environ["LST_CODE_REVISION"] == head
        assert environ["LST_EVIDENCE_CONTRACT"] == "contract.json"

    def test_a_dirty_launch_checkout_refuses_to_submit(self, tmp_path, monkeypatch):
        from landsat_lst.evidence_contract import ContractError
        from landsat_lst.job import _worker_environ

        root, _head = self._repo(tmp_path, monkeypatch)
        (root / "src" / "module.py").write_text("VERSION = 99\n")

        with pytest.raises(ContractError, match="tracked changes"):
            _worker_environ()

    def test_a_launch_outside_any_checkout_says_so(self, tmp_path, monkeypatch):
        from landsat_lst.evidence_contract import ContractError
        from landsat_lst.job import _worker_environ

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
        monkeypatch.setenv("LST_EVIDENCE_CONTRACT", "contract.json")
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ContractError, match="not inside a git checkout"):
            _worker_environ()
