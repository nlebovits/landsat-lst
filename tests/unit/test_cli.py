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
        assert storage.read_text(storage.log_key("run-1", "N40W075", 1)) is not None

    def test_a_failed_tile_still_leaves_its_log(self, runner, tmp_path):
        from landsat_lst.storage import LocalStorage

        with patch("landsat_lst.job.process_tile_job") as mock_job:
            mock_job.return_value = _result("N40W075", "failed", "No scenes found")
            result = runner.invoke(main, ["process", "-t", "N40W075", "--run-id", "run-1"])

        storage = LocalStorage(output_dir=tmp_path / "cogs")
        assert result.exit_code == 1
        assert storage.read_text(storage.log_key("run-1", "N40W075", 1)) is not None

    def test_an_unusable_tile_argument_still_leaves_its_log(self, tmp_path):
        """The one failure mode that used to leave no evidence at all.

        A tile name the CLI cannot parse raises while the jobs are being built.
        That happened before any capture existed, so the task died in 0.6s on a
        VM whose stdout stays on the VM, having written nothing anywhere.
        """
        from landsat_lst.storage import LocalStorage
        from tests.unit.test_progress import stdio_on_descriptors

        # The capture tees descriptors, so the test needs the arrangement a
        # real run has: sys.stderr over fd 2. CliRunner installs its own
        # redirection inside invoke and would swallow the traceback again, so
        # the command is invoked directly.
        with stdio_on_descriptors(), pytest.raises(ValueError):
            main.main(
                ["process", "-t", '"N40W075"', "--run-id", "run-1"],
                standalone_mode=False,
            )

        storage = LocalStorage(output_dir=tmp_path / "cogs")
        log = storage.read_text(storage.log_key("run-1", "_N40W075_", 1))
        assert log is not None
        assert "Invalid tile name format" in log

    def test_one_process_numbers_every_artifact_alike(self, runner, tmp_path):
        """The log and the state object carry the same attempt number.

        The attempt is resolved once, before the capture opens, and threaded
        down. A second caller asking the bucket would see this process's own
        state object and number itself one higher, and the log uploads last, so
        the split would land on exactly the two artifacts that have to agree.
        """
        from landsat_lst.storage import LocalStorage

        storage = LocalStorage(output_dir=tmp_path / "cogs")
        # An earlier attempt already left a state object, so this one is 2.
        storage.write_text(storage.run_record_key("run-1", "N40W075", 1), "{}")

        with patch("landsat_lst.job.process_tile_job") as mock_job:
            mock_job.return_value = _result("N40W075", "completed")
            result = runner.invoke(main, ["process", "-t", "N40W075", "--run-id", "run-1"])

        assert result.exit_code == 0
        assert mock_job.call_args.kwargs["attempt"] == 2
        assert storage.read_text(storage.log_key("run-1", "N40W075", 2)) is not None
        assert storage.read_text(storage.log_key("run-1", "N40W075", 3)) is None

    def test_the_first_attempt_is_numbered_one(self, runner, tmp_path):
        """An empty run prefix means nothing has been tried yet."""
        from landsat_lst.storage import LocalStorage

        with patch("landsat_lst.job.process_tile_job") as mock_job:
            mock_job.return_value = _result("N40W075", "completed")
            runner.invoke(main, ["process", "-t", "N40W075", "--run-id", "run-1"])

        storage = LocalStorage(output_dir=tmp_path / "cogs")
        assert mock_job.call_args.kwargs["attempt"] == 1
        assert storage.read_text(storage.log_key("run-1", "N40W075", 1)) is not None

    def test_a_local_run_resolves_no_attempt(self, runner, tmp_path):
        """No run id means no artifacts to number, so nothing is listed."""
        with patch("landsat_lst.job.process_tile_job") as mock_job:
            mock_job.return_value = _result("N40W075", "completed")
            runner.invoke(main, ["process", "-t", "N40W075"])

        assert mock_job.call_args.kwargs["attempt"] is None

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


class TestPlan:
    """`plan` prices a tile's graphs statically: no network, no cluster, no pixels.

    Scene counts stay tiny. The command builds real 18,000-squared graphs, and
    the production default of 2,930 scenes would take half a minute per case.
    """

    def test_reports_both_phases_and_their_task_counts(self, runner):
        result = runner.invoke(main, ["plan", "-t", "N40W075", "--scenes", "8"])

        assert result.exit_code == 0, result.output
        assert "destripe_offsets" in result.output
        assert "composite" in result.output
        assert "18000x18000" in result.output
        assert "floor" in result.output

    def test_json_output_round_trips(self, runner):
        import json as json_module

        result = runner.invoke(
            main, ["plan", "-t", "N40W075", "--scenes", "8", "--threads", "4", "--json"]
        )

        assert result.exit_code == 0, result.output
        phases = json_module.loads(result.output)
        assert [p["phase"] for p in phases] == ["destripe_offsets", "composite"]
        assert phases[1]["shape"] == [18000, 18000]
        assert phases[0]["memory"]["threads"] == 4

    def test_sweep_emits_one_row_per_configuration(self, runner):
        import json as json_module

        # --fast skips graph fusion. A sweep ranks configurations by memory
        # floor, which is exact either way, and fusing three chunk sizes takes
        # longer than the pre-push per-test timeout allows.
        result = runner.invoke(
            main, ["plan", "-t", "N40W075", "--scenes", "8", "--sweep", "--fast", "--json"]
        )

        assert result.exit_code == 0, result.output
        rows = json_module.loads(result.output)
        # Three chunk sizes crossed with four thread counts.
        assert len(rows) == 12
        assert [r["floor_gib"] for r in rows] == sorted(r["floor_gib"] for r in rows)
        assert all(r["optimized"] is False for r in rows)

    def test_fast_labels_its_counts_as_unfused(self, runner):
        """A raw count must never read as one a heartbeat would report."""
        result = runner.invoke(main, ["plan", "-t", "N40W075", "--scenes", "8", "--fast"])

        assert result.exit_code == 0, result.output
        assert "unfused" in result.output
        assert "after fusion" not in result.output

    def test_default_reports_fused_counts(self, runner):
        """The headline count is the one the scheduler runs."""
        result = runner.invoke(main, ["plan", "-t", "N40W075", "--scenes", "8", "--json"])

        import json as json_module

        phases = json_module.loads(result.output)
        assert all(p["graph"]["optimized"] for p in phases)
        assert all(p["graph"]["tasks"] <= p["graph"]["raw_tasks"] for p in phases)

    def test_offset_factor_shrinks_the_offset_grid_and_leaves_the_composite(self, runner):
        """Pricing a factor change is the point of the flag.

        The offset pass reads a grid coarsened by the factor, so its task count
        falls as ``factor**2``. The composite always runs at native resolution,
        so it must not move at all -- that invariance is the reason the offset
        saving caps out where it does.
        """
        import json as json_module

        def phases(factor):
            result = runner.invoke(
                main,
                [
                    "plan",
                    "-t",
                    "N40W075",
                    "--scenes",
                    "8",
                    "--offset-factor",
                    str(factor),
                    "--json",
                ],
            )
            assert result.exit_code == 0, result.output
            return json_module.loads(result.output)

        two, four = phases(2), phases(4)

        assert two[0]["shape"] == [9000, 9000]
        assert four[0]["shape"] == [4500, 4500]
        assert four[0]["graph"]["tasks"] < two[0]["graph"]["tasks"]
        assert four[1] == two[1]

    def test_flags_a_tile_that_is_not_land(self, runner):
        """Ocean tiles are never processed, so planning one is likely a typo."""
        result = runner.invoke(main, ["plan", "-t", "S55W180", "--scenes", "4"])

        assert result.exit_code == 0, result.output
        assert "not in the land tiles set" in result.output

    def test_rejects_an_unparseable_tile(self, runner):
        result = runner.invoke(main, ["plan", "-t", "not-a-tile", "--scenes", "4"])

        assert result.exit_code != 0


class TestOffsetsCommand:
    """`landsat-lst offsets`: the expensive phase, run and persisted on its own."""

    def _estimate(self, *, cached=False):
        from landsat_lst.offsets import OffsetKey
        from landsat_lst.pipeline import OffsetEstimate

        return OffsetEstimate(
            key=OffsetKey.build(tile="N40W075", window="2021-2025", factor=2, scene_ids=("a", "b")),
            scenes=300,
            diagnostics={
                "n_scenes": 300.0,
                "n_kept": 234.0,
                "rejected_frac": 0.22,
                "std": 5.71,
                "p1": -14.2,
                "p50": 0.3,
                "p99": 11.8,
            },
            cached=cached,
            duration_s=1612.0,
        )

    def test_reports_the_rejection_fraction_and_the_key(self, runner):
        """The rejection fraction is the number the cap was calibrated against."""
        with patch("landsat_lst.pipeline.compute_tile_offsets") as mock:
            mock.return_value = self._estimate()
            result = runner.invoke(main, ["offsets", "-t", "N40W075"])

        assert result.exit_code == 0, result.output
        assert "234/300" in result.output
        assert "22.0%" in result.output
        assert "_offsets/N40W075/2021-2025/f2/" in result.output

    def test_says_when_the_answer_came_from_cache(self, runner):
        with patch("landsat_lst.pipeline.compute_tile_offsets") as mock:
            mock.return_value = self._estimate(cached=True)
            result = runner.invoke(main, ["offsets", "-t", "N40W075"])

        assert "cached" in result.output

    def test_defaults_to_the_production_window(self, runner):
        with patch("landsat_lst.pipeline.compute_tile_offsets") as mock:
            mock.return_value = self._estimate()
            runner.invoke(main, ["offsets", "-t", "N40W075"])

        assert mock.call_args.args[0].window_label == "2021-2025"

    def test_no_offset_cache_disables_both_halves(self, runner):
        with patch("landsat_lst.pipeline.compute_tile_offsets") as mock:
            mock.return_value = self._estimate()
            runner.invoke(main, ["offsets", "-t", "N40W075", "--no-offset-cache"])

        assert mock.call_args.kwargs["use_offset_cache"] is False
        assert mock.call_args.kwargs["refresh"] is False

    def test_force_refreshes_rather_than_disabling(self, runner):
        """--force rebuilds the estimate and replaces what was stored."""
        with patch("landsat_lst.pipeline.compute_tile_offsets") as mock:
            mock.return_value = self._estimate()
            runner.invoke(main, ["offsets", "-t", "N40W075", "--force"])

        assert mock.call_args.kwargs["use_offset_cache"] is True
        assert mock.call_args.kwargs["refresh"] is True

    def test_requires_a_tile(self, runner):
        """No fleet-wide offset pass: this is a per-tile iteration command."""
        assert runner.invoke(main, ["offsets"]).exit_code != 0


class TestCompositeCommand:
    """`landsat-lst composite`: one tile to COGs, reading whatever is cached."""

    def _result(self, status="completed", **fields):
        job = ProcessingJob(tile=parse_tile_name("N40W075"), year=2021, end_year=2025)
        base = {
            "lst_key": "lst-p95-2021-2025/N40W075/lst_p95_2021-2025_N40W075.tif",
            "qa_key": "lst-p95-2021-2025/N40W075/qa_count_2021-2025_N40W075.tif",
            "duration_s": 900.0,
            "scene_count": 2930,
            "peak_rss_mb": 41000.0,
        }
        return JobResult(job=job, status=status, **{**base, **fields})

    def test_reports_both_asset_keys(self, runner):
        with patch("landsat_lst.job.process_tile_job") as mock:
            mock.return_value = self._result()
            result = runner.invoke(main, ["composite", "-t", "N40W075"])

        assert result.exit_code == 0, result.output
        assert "lst_p95_2021-2025_N40W075.tif" in result.output
        assert "qa_count_2021-2025_N40W075.tif" in result.output

    def test_skips_a_tile_whose_cogs_exist(self, runner):
        """Completion is bytes in the bucket, and this command honours that."""
        with patch("landsat_lst.job.process_tile_job") as mock:
            mock.return_value = JobResult(job=self._result().job, status="skipped")
            result = runner.invoke(main, ["composite", "-t", "N40W075"])

        assert "skipped" in result.output
        assert "--force" in result.output

    def test_forwards_force_and_the_cache_flag(self, runner):
        with patch("landsat_lst.job.process_tile_job") as mock:
            mock.return_value = self._result()
            runner.invoke(main, ["composite", "-t", "N40W075", "--force", "--no-offset-cache"])

        assert mock.call_args.kwargs["force"] is True
        assert mock.call_args.kwargs["use_offset_cache"] is False

    def test_exits_non_zero_on_failure(self, runner):
        with patch("landsat_lst.job.process_tile_job") as mock:
            mock.return_value = self._result(status="failed", error="no scenes")
            result = runner.invoke(main, ["composite", "-t", "N40W075"])

        assert result.exit_code == 1
        assert "no scenes" in result.output


class TestProcessOffsetCacheFlag:
    """The fleet driver forwards the flag both locally and to Coiled."""

    def test_local_run_forwards_it(self, runner):
        with patch("landsat_lst.job.process_tile_job") as mock:
            mock.return_value = JobResult(
                job=ProcessingJob(tile=parse_tile_name("N40W075"), year=2024),
                status="completed",
            )
            runner.invoke(main, ["process", "-t", "N40W075", "-y", "2024", "--no-offset-cache"])

        assert mock.call_args.kwargs["use_offset_cache"] is False

    def test_the_batch_task_command_carries_it(self):
        """A VM rebuilds its own job arguments, so an omitted flag reverts."""
        from landsat_lst.batch import _task_command

        assert "--no-offset-cache" in _task_command(
            run_id="r", year=2021, end_year=2025, force=False, use_offset_cache=False
        )
        assert "--no-offset-cache" not in _task_command(
            run_id="r", year=2021, end_year=2025, force=False
        )


class TestSampledRunsProfileThemselves:
    """``--max-scenes`` is a sample by definition, and a sample exists to be measured.

    The run that prompted issue #94 produced a profile only because
    ``LST_PROFILE_DASK`` happened to be set by hand. See issue #94 item 3.
    """

    @pytest.fixture
    def profile_off(self, monkeypatch):
        from landsat_lst.config import settings

        monkeypatch.delenv("LST_PROFILE_DASK", raising=False)
        monkeypatch.setattr(settings, "profile_dask", False)
        return settings

    def _run(self, runner, *extra):
        with patch("landsat_lst.job.process_tile_job") as mock:
            mock.return_value = JobResult(
                job=ProcessingJob(tile=parse_tile_name("N40W075"), year=2024),
                status="completed",
            )
            runner.invoke(main, ["process", "-t", "N40W075", "-y", "2024", *extra])

    def test_max_scenes_turns_profiling_on(self, runner, profile_off):
        self._run(runner, "--max-scenes", "300")
        assert profile_off.profile_dask is True

    def test_a_full_run_leaves_it_alone(self, runner, profile_off):
        """The docstring's reasoning still holds for a 700-tile build."""
        self._run(runner)
        assert profile_off.profile_dask is False

    def test_an_explicit_setting_wins(self, runner, monkeypatch):
        """Somebody who turned profiling off on purpose does not get it back."""
        from landsat_lst.config import settings

        monkeypatch.setenv("LST_PROFILE_DASK", "0")
        monkeypatch.setattr(settings, "profile_dask", False)

        self._run(runner, "--max-scenes", "300")

        assert settings.profile_dask is False

    def test_cache_profiling_stays_gated_on_its_own(self, runner, profile_off, monkeypatch):
        """CacheProfiler retains a record per task; a sampled graph still has many."""
        monkeypatch.setattr(profile_off, "profile_dask_cache", False)

        self._run(runner, "--max-scenes", "300")

        assert profile_off.profile_dask_cache is False


@pytest.fixture
def s3_backend(monkeypatch):
    """The only backend the driver will submit Coiled work against.

    Also stubs *all three* preflights, which run before the run id is printed
    and must never reach a control plane from a unit test. A CI runner has no
    AWS session, so an unstubbed identity check refuses there and passes only
    on a laptop that happens to be logged in -- which is how this escaped once
    already. The write probe is worse: unstubbed it puts a real object in the
    publication bucket. The refusal paths have their own scenarios in
    tests/unit/test_driver_state_machine.py.
    """
    from landsat_lst import quota
    from landsat_lst.config import settings

    monkeypatch.setattr(settings, "storage_backend", "s3")
    monkeypatch.setattr(quota, "preflight_identity", lambda **_: "arn:aws:sts::0:assumed-role/t")
    monkeypatch.setattr(
        quota, "preflight_write_access", lambda **_: ["arn:aws:sts::0:assumed-role/t"]
    )
    monkeypatch.setattr(
        quota, "read_balance", lambda: quota.CreditBalance(remaining=10_000.0, source="test")
    )


class TestShardGroup:
    """The two commands a person runs, and the five a VM runs."""

    def test_every_stage_is_reachable(self, runner):
        from landsat_lst.shards import STAGES

        result = runner.invoke(main, ["shard", "--help"])

        assert result.exit_code == 0
        for stage in (*STAGES, "process", "resume"):
            assert stage in result.output

    def test_a_local_backend_fails_before_a_run_id_is_printed(self, runner, monkeypatch):
        """The acceptance run printed one and then hung: the driver polled the
        laptop while the VMs wrote S3. A resume hint for a run that never
        started is worse than no output at all.
        """
        from landsat_lst.config import settings

        monkeypatch.setattr(settings, "storage_backend", "local")

        result = runner.invoke(main, ["shard", "process", "--tile", "N40W075"])

        assert result.exit_code != 0
        assert "LST_STORAGE_BACKEND=s3" in result.output
        assert "shard resume" not in result.output

    def test_an_identity_that_cannot_write_fails_before_a_run_id_is_printed(
        self, runner, s3_backend, monkeypatch
    ):
        """Same placement rule as the backend and credit gates.

        A read-only identity clears STS, so the run would boot a fleet, stage
        nothing, and fail on its first artifact -- one wasted boot per worker,
        showing up as shards that never published. Refusing here costs a
        round trip.
        """
        del s3_backend
        from landsat_lst import quota

        def refuse(**_kwargs) -> list[str]:
            raise quota.WriteAccessRefused(
                "PutObject was refused (AccessDenied) for the driver, which runs as "
                "arn:aws:iam::392361759182:user/vercel-data-access from the default "
                "credential chain. The target is s3://a-bucket/a-prefix/.",
                arn="arn:aws:iam::392361759182:user/vercel-data-access",
                bucket="a-bucket",
                key="a-prefix/_preflight/probe.json",
            )

        monkeypatch.setattr(quota, "preflight_write_access", refuse)

        result = runner.invoke(main, ["shard", "process", "--tile", "N40W075"])

        assert result.exit_code != 0
        assert "vercel-data-access" in result.output
        assert "a-bucket" in result.output
        assert "shard resume" not in result.output

    def test_process_prints_the_resume_line_before_it_starts(self, runner, s3_backend):
        """The run id is the only thing a resume needs, so it is printed first.

        A driver killed mid-run is ordinary; a driver whose run id was never
        shown leaves a bucket full of shards nothing can pick up.
        """
        summary = _shard_summary()
        with patch("landsat_lst.shard_driver.drive_tile", return_value=summary) as drive:
            result = runner.invoke(main, ["shard", "process", "--tile", "N40W075"])

        assert result.exit_code == 0
        assert "shard resume" in result.output
        assert drive.call_args.kwargs["run_id"] in result.output

    def test_a_stage_that_never_finished_fails_the_command(self, runner, s3_backend):
        from landsat_lst.shard_driver import ShardStageFailed

        with patch(
            "landsat_lst.shard_driver.drive_tile",
            side_effect=ShardStageFailed("composite", ["_shards/r/N40W075/composite/x.tif"]),
        ):
            result = runner.invoke(main, ["shard", "process", "--tile", "N40W075"])

        assert result.exit_code != 0
        assert "composite" in result.output

    def test_resume_takes_a_run_id_and_a_tile_and_nothing_else(self, runner):
        summary = _shard_summary()
        with patch("landsat_lst.shard_driver.resume_tile", return_value=summary) as resume:
            result = runner.invoke(main, ["shard", "resume", "run-7", "N40W075"])

        assert result.exit_code == 0
        assert resume.call_args.args == ("run-7", "N40W075")

    def test_resume_formats_a_write_refusal_as_a_cli_error(self, runner):
        from landsat_lst import quota

        refusal = quota.WriteAccessRefused(
            "ListObjectsV2 was refused", arn="arn:aws:iam::1:user/x", bucket="b", key="k"
        )
        with patch("landsat_lst.shard_driver.resume_tile", side_effect=refusal):
            result = runner.invoke(main, ["shard", "resume", "run-7", "N40W075"])

        assert result.exit_code != 0
        assert "Error: ListObjectsV2 was refused" in result.output
        assert result.exception is not refusal

    def test_resume_fleet_formats_a_write_refusal_as_a_cli_error(self, runner):
        from landsat_lst import quota

        refusal = quota.WriteAccessRefused(
            "PutObject was refused", arn="arn:aws:iam::1:user/x", bucket="b", key="k"
        )
        with patch("landsat_lst.fleet_driver.resume_fleet", side_effect=refusal):
            result = runner.invoke(main, ["shard", "resume-fleet", "run-7"])

        assert result.exit_code != 0
        assert "Error: PutObject was refused" in result.output
        assert result.exception is not refusal

    def test_a_stage_subcommand_forwards_its_index(self, runner):
        with patch("landsat_lst.shard_tasks.run_shard", return_value=[]) as run_shard:
            result = runner.invoke(
                main,
                ["shard", "composite", "--run-id", "r", "--tile", "N40W075", "--index", "3"],
            )

        assert result.exit_code == 0
        assert run_shard.call_args.args == ("composite", "r", "N40W075", 3)

    def test_resolve_forwards_the_window_it_was_given(self, runner):
        from landsat_lst.shards import TilePlan

        plan = TilePlan(
            tile="N40W075",
            window="2024",
            scene_ids=["a"],
            scene_times=["2024-07-04T00:00:00"],
            offset_factor=2,
            coarse_shape=(8, 8),
            native_shape=(1024, 1024),
            block_edge=4,
            blocks=[(0, 4, 0, 4)],
            block_has_land=[True],
            scene_batches=[(0, 1)],
            bands=[(0, 1024)],
        )
        with patch("landsat_lst.shard_tasks.run_shard", return_value=plan) as run_shard:
            result = runner.invoke(
                main,
                ["shard", "resolve", "--run-id", "r", "--tile", "N40W075", "--year", "2024"],
            )

        assert result.exit_code == 0
        assert run_shard.call_args.kwargs["job"].year == 2024
        assert run_shard.call_args.kwargs["job"].end_year is None


def _shard_summary():
    from landsat_lst.shard_driver import StageOutcome, TileRunSummary

    return TileRunSummary(
        run_id="run-7",
        tile="N40W075",
        window="2021-2025",
        stages=[StageOutcome("resolve", 1, 0, 1, 3.0)],
        completed=True,
    )
