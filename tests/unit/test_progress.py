"""Unit tests for tile heartbeats and task-log capture.

Nothing here runs a pipeline or starts a cluster. Storage is always
:class:`LocalStorage` under ``tmp_path``, and the "tile" is whatever the test
writes to stdout.
"""

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from landsat_lst.job import JobResult
from landsat_lst.models import ProcessingJob
from landsat_lst.progress import (
    GRAPH_IDLE,
    GRAPH_RUNNING,
    SCHEMA_VERSION,
    TERMINAL_PHASES,
    GraphProgress,
    TileHeartbeat,
    active_heartbeat,
    capture_task_log,
    peak_rss_mb,
    report_failed,
    report_phase,
    rss_mb,
    write_final_state,
)
from landsat_lst.storage import LocalStorage
from landsat_lst.tiling import parse_tile_name

RUN, TILE, WINDOW = "run-1", "N40W075", "2021-2025"

#: The job every heartbeat here reports for. A heartbeat takes the job rather
#: than a tile and a window because the object it publishes has to carry enough
#: of the job for a reader to rebuild it, ``max_scenes`` included.
JOB = ProcessingJob(tile=parse_tile_name(TILE), year=2021, end_year=2025)


class RecordingStorage(LocalStorage):
    """LocalStorage that remembers what each write declared itself to be."""

    def __init__(self, output_dir):
        super().__init__(output_dir=output_dir)
        self.writes: list[tuple[str, str]] = []

    def write_text(self, key, text, *, content_type="application/json"):
        self.writes.append((key, content_type))
        super().write_text(key, text, content_type=content_type)


class BrokenStorage(LocalStorage):
    """A backend that cannot write, which must never be fatal to a tile."""

    def write_text(self, key, text, *, content_type="application/json"):
        msg = "bucket on fire"
        raise OSError(msg)


@pytest.fixture
def storage(tmp_path):
    return RecordingStorage(tmp_path / "cogs")


@contextmanager
def stdio_on_descriptors():
    """Restore the arrangement a real run has: ``sys.stdout`` over fd 1.

    pytest replaces ``sys.stdout`` with an object writing straight into its own
    temp file, and re-installs it at the start of each test phase, so a bare
    ``print`` never touches the descriptor this module tees. Without this the
    print-based tests would measure pytest rather than the capture. It has to
    be entered inside the test body for the same reason: a fixture's patch is
    overwritten before the body runs.

    Both streams stay block-buffered, as a run into a pipe or a log file finds
    them, so the tests cover the flush ordering on the way out too.
    """
    out = open(1, "w", closefd=False)  # noqa: SIM115 - closed below
    err = open(2, "w", closefd=False)  # noqa: SIM115 - closed below
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        yield
    finally:
        out.flush()
        err.flush()
        sys.stdout, sys.stderr = saved


def _beat(storage, **overrides) -> TileHeartbeat:
    kwargs = {
        "run_id": RUN,
        "job": JOB,
        "storage": storage,
        "interval_s": 3600,
    }
    return TileHeartbeat(**{**kwargs, **overrides})


def _published(storage, attempt: int = 1) -> dict:
    """One attempt's own state object, which is rewritten on every beat."""
    raw = storage.read_text(storage.run_record_key(RUN, TILE, attempt))
    assert raw is not None, f"no state was published for attempt {attempt}"
    return json.loads(raw)


def _pointer(storage) -> dict | None:
    """The copy of the settled state, written once when a tile stops."""
    raw = storage.read_text(storage.run_record_key(RUN, TILE))
    return None if raw is None else json.loads(raw)


class TestHeartbeat:
    def test_publishes_on_entry(self, storage):
        """A tile that has not reached its first phase is still visibly alive."""
        with _beat(storage):
            payload = _published(storage)

        assert payload["phase"] == "starting"
        assert payload["tile"] == TILE
        assert payload["window"] == WINDOW
        assert payload["run_id"] == RUN
        assert payload["attempt"] == 1
        assert payload["schema"] == SCHEMA_VERSION
        assert payload["elapsed_s"] >= 0
        assert datetime.fromisoformat(payload["updated_at"]).tzinfo is not None

    def test_phase_change_publishes_at_once(self, storage):
        """Waiting a minute to say a two-hour phase started is not observability."""
        with _beat(storage) as beat:
            beat.set_phase("compositing")

            assert _published(storage)["phase"] == "compositing"

    def test_counts_accumulate_across_phases(self, storage):
        with _beat(storage) as beat:
            beat.set_phase("loading", scenes_found=1870)
            beat.set_phase("compositing", scenes_kept=1462)
            payload = _published(storage)

        assert payload["scenes_found"] == 1870
        assert payload["scenes_kept"] == 1462

    def test_none_counts_are_not_published(self, storage):
        """A caller with nothing to add passes the keyword anyway."""
        with _beat(storage) as beat:
            beat.set_phase("compositing", scenes_kept=None)

            assert "scenes_kept" not in _published(storage)

    def test_beats_without_being_asked(self, storage):
        """The point of the thread: a long phase keeps proving it is alive."""
        state_key = storage.run_record_key(RUN, TILE, 1)
        with _beat(storage, interval_s=0.05):
            time.sleep(0.3)
            beats = sum(1 for key, _ in storage.writes if key == state_key)

        assert beats >= 2

    def test_exit_marks_the_tile_done(self, storage):
        with _beat(storage) as beat:
            beat.set_phase("uploading")

        assert _published(storage)["phase"] == "done"

    def test_exit_reports_an_escaping_exception(self, storage):
        with pytest.raises(TimeoutError), _beat(storage):
            msg = "read timed out"
            raise TimeoutError(msg)

        payload = _published(storage)
        assert payload["phase"] == "failed"
        assert payload["error"] == "TimeoutError: read timed out"

    def test_reported_failure_beats_a_reconstructed_one(self, storage):
        """A deterministic failure is returned, not raised, so it self-reports."""
        with _beat(storage) as beat:
            beat.set_failed("No scenes found for the window")

        payload = _published(storage)
        assert payload["phase"] == "failed"
        assert payload["error"] == "No scenes found for the window"

    def test_thread_stops_with_the_context(self, storage):
        beat = _beat(storage, interval_s=0.05)
        with beat:
            pass
        time.sleep(0.2)
        after = len(storage.writes)
        time.sleep(0.2)

        assert len(storage.writes) == after

    def test_a_dead_backend_does_not_kill_the_tile(self, tmp_path):
        """Losing the heartbeat is cheaper than losing two hours of composite."""
        with _beat(BrokenStorage(output_dir=tmp_path)) as beat:
            beat.set_phase("compositing")

    def test_peak_rss_is_reported(self, storage):
        with _beat(storage):
            payload = _published(storage)

        assert payload["peak_rss_mb"] == pytest.approx(peak_rss_mb(), rel=0.5)

    def test_current_rss_is_reported_alongside_the_peak(self, storage):
        """The peak alone draws a staircase that never falls.

        It cannot show a phase releasing memory, and cannot tell "holding 35 GB
        now" from "touched 35 GB an hour ago". Headroom against the VM needs the
        current figure and the alarm needs both, so both are published.
        """
        with _beat(storage):
            payload = _published(storage)

        assert payload["rss_mb"] == pytest.approx(rss_mb(), rel=0.5)
        assert payload["peak_rss_mb"] is not None

    def test_rss_is_none_where_proc_is_unavailable(self, monkeypatch):
        """A machine without ``/proc`` reports no current RSS rather than dying."""

        def no_proc(*_args, **_kwargs):
            msg = "/proc/self/statm"
            raise FileNotFoundError(msg)

        monkeypatch.setattr(Path, "read_text", no_proc)

        assert rss_mb() is None

    def test_status_is_none_until_the_tile_settles(self, storage):
        """A running tile must not publish a verdict.

        A mid-run reconcile reads whatever status it finds, and ``watch`` would
        gain a second liveness signal to disagree with ``phase``. The status
        appears only when the terminal beat publishes the folded-in result.
        """
        with _beat(storage) as beat:
            assert _published(storage)["status"] is None
            beat.set_phase("destriping")
            assert _published(storage)["status"] is None
            beat.set_result(JobResult(job=JOB, status="completed", scene_count=412))

        settled = _published(storage)
        assert settled["status"] == "completed"
        assert settled["scene_count"] == 412

    def test_the_settled_state_is_copied_to_the_pointer(self, storage):
        """A reader that knows only ``{tile}.json`` still gets the whole object.

        The pointer is a copy of the attempt's own final state, so the old run
        record's fields are a subset of it and no reader needs attempt logic.
        """
        with _beat(storage) as beat:
            beat.set_result(JobResult(job=JOB, status="completed", duration_s=12.5))

        pointer, attempt_state = _pointer(storage), _published(storage)
        # Both are rendered from the same live object, so only the clock fields
        # differ between the two writes.
        volatile = {"updated_at", "elapsed_s", "phase_seconds", "rss_mb", "peak_rss_mb"}
        assert {k: v for k, v in pointer.items() if k not in volatile} == {
            k: v for k, v in attempt_state.items() if k not in volatile
        }
        assert pointer["status"] == "completed"
        assert pointer["duration_s"] == 12.5
        assert pointer["phase"] == "done"

    def test_the_pointer_is_written_once_at_the_end(self, storage):
        """Refreshing it every beat would double the run's PUT bill for nothing."""
        pointer_key = storage.run_record_key(RUN, TILE)
        with _beat(storage, interval_s=0.05):
            time.sleep(0.3)
            assert not [key for key, _ in storage.writes if key == pointer_key]

        assert len([key for key, _ in storage.writes if key == pointer_key]) == 1

    def test_an_escaping_exception_leaves_no_pointer(self, storage):
        """An escaping failure means a retry is in flight, so nothing has settled.

        The attempt still publishes its own object, which is the evidence the
        retry would otherwise destroy. Only the settled-state pointer waits.
        """
        with pytest.raises(TimeoutError), _beat(storage):
            msg = "read timed out"
            raise TimeoutError(msg)

        assert _published(storage)["phase"] == "failed"
        assert _pointer(storage) is None

    def test_terminal_phases_are_the_ones_the_context_writes(self, storage):
        """Whatever a watcher treats as terminal has to be what is published."""
        with _beat(storage) as beat:
            beat.set_failed("boom")
        assert _published(storage)["phase"] in TERMINAL_PHASES

        with _beat(storage):
            pass
        assert _published(storage)["phase"] in TERMINAL_PHASES


class TestReporting:
    def test_report_phase_without_a_heartbeat_does_nothing(self):
        """Local runs, tests, and benchmarks call the same pipeline."""
        report_phase("compositing", scenes_found=3)
        report_failed("nobody is listening")

        assert active_heartbeat() is None

    def test_report_phase_reaches_the_running_tile(self, storage):
        with _beat(storage):
            report_phase("destriping", scenes_found=390)

            payload = _published(storage)

        assert payload["phase"] == "destriping"
        assert payload["scenes_found"] == 390

    def test_active_heartbeat_is_cleared_on_exit(self, storage):
        with _beat(storage) as beat:
            assert active_heartbeat() is beat

        assert active_heartbeat() is None


def _log(storage, attempt: int = 1) -> str:
    raw = storage.read_text(storage.log_key(RUN, TILE, attempt))
    assert raw is not None, f"no task log was uploaded for attempt {attempt}"
    return raw


class TestCaptureTaskLog:
    def test_captures_python_output(self, storage):
        with (
            stdio_on_descriptors(),
            capture_task_log(run_id=RUN, tile=TILE, storage=storage) as key,
        ):
            print("composite written")
            print("warning: overview 5 is short", file=sys.stderr)

        assert key == storage.log_key(RUN, TILE, 1)
        assert "composite written" in _log(storage)
        assert "overview 5 is short" in _log(storage)

    def test_captures_writes_that_bypass_python(self, storage):
        """GDAL and rasterio write to the descriptor, not to ``sys.stdout``."""
        with capture_task_log(run_id=RUN, tile=TILE, storage=storage):
            os.write(1, b"ERROR 4: not a supported format\n")
            os.write(2, b"CPLE_OpenFailed\n")

        log = _log(storage)
        assert "ERROR 4: not a supported format" in log
        assert "CPLE_OpenFailed" in log

    def test_captures_child_process_output(self, storage):
        with capture_task_log(run_id=RUN, tile=TILE, storage=storage):
            subprocess.run([sys.executable, "-c", "print('child said so')"], check=True)

        assert "child said so" in _log(storage)

    def test_output_still_reaches_the_console(self, storage, capfd):
        """Capturing must not blind whoever is watching the run live."""
        with stdio_on_descriptors(), capture_task_log(run_id=RUN, tile=TILE, storage=storage):
            print("still visible")

        assert "still visible" in capfd.readouterr().out

    def test_uploads_and_reraises_on_failure(self, storage):
        with (
            pytest.raises(ValueError, match="all scenes rejected"),
            stdio_on_descriptors(),
            capture_task_log(run_id=RUN, tile=TILE, storage=storage),
        ):
            print("de-striping 390 scenes")
            msg = "all scenes rejected"
            raise ValueError(msg)

        log = _log(storage)
        assert "de-striping 390 scenes" in log
        # The traceback is the whole reason the log exists: the interpreter
        # would otherwise print it after the descriptors were restored.
        assert "Traceback (most recent call last)" in log
        assert "ValueError: all scenes rejected" in log

    def test_systemexit_keeps_the_log_without_a_traceback(self, storage):
        """The CLI already said why; a SystemExit traceback is noise."""
        with (
            pytest.raises(SystemExit),
            stdio_on_descriptors(),
            capture_task_log(run_id=RUN, tile=TILE, storage=storage),
        ):
            print("Failed: N40W075 - No scenes found")
            raise SystemExit(1)

        log = _log(storage)
        assert "No scenes found" in log
        assert "Traceback" not in log

    def test_long_logs_upload_as_their_tail(self, storage):
        notice = "[truncated: 3517 earlier bytes dropped]\n"
        with (
            stdio_on_descriptors(),
            capture_task_log(run_id=RUN, tile=TILE, storage=storage, max_bytes=512),
        ):
            print("x" * 4000)
            print("the last thing it ever said")

        log = _log(storage)
        assert log.startswith(notice)
        assert "the last thing it ever said" in log
        assert len(log.encode()) == len(notice) + 512

    def test_uploads_as_plain_text(self, storage):
        with stdio_on_descriptors(), capture_task_log(run_id=RUN, tile=TILE, storage=storage):
            print("hello")

        assert (storage.log_key(RUN, TILE, 1), "text/plain") in storage.writes

    def test_an_unuploadable_log_does_not_fail_the_tile(self, tmp_path):
        with (
            stdio_on_descriptors(),
            capture_task_log(run_id=RUN, tile=TILE, storage=BrokenStorage(output_dir=tmp_path)),
        ):
            print("hello")

    def test_descriptors_are_restored(self, storage, capfd):
        with stdio_on_descriptors():
            with capture_task_log(run_id=RUN, tile=TILE, storage=storage):
                print("inside")
            print("outside")
            os.write(1, b"outside-fd\n")

        out = capfd.readouterr().out
        assert "outside" in out
        assert "outside-fd" in out
        assert "outside" not in _log(storage)

    def test_empty_output_still_uploads(self, storage):
        """An empty log is evidence too: it says the process never spoke."""
        with capture_task_log(run_id=RUN, tile=TILE, storage=storage):
            pass

        assert _log(storage) == ""


def test_state_and_log_share_the_run_prefix(storage):
    """Everything one tile reports lands where reconciliation already looks."""
    prefix = storage.run_prefix(RUN)

    assert storage.run_record_key(RUN, TILE, 1).startswith(prefix)
    assert storage.log_key(RUN, TILE, 1).startswith(prefix)
    assert storage.profile_key(RUN, TILE, "destripe_offsets", 1).startswith(prefix)
    assert storage.run_record_key(RUN, TILE).startswith(prefix)


def test_every_artifact_of_one_attempt_shares_one_tile_prefix(storage):
    """One attempt's objects are selectable without reading the whole run.

    ``resolve_attempt`` lists ``{tile}.`` rather than the run, so the next
    attempt is numbered from a listing that stays small as the run grows.
    """
    prefix = f"{storage.run_prefix(RUN)}{TILE}."

    assert storage.run_record_key(RUN, TILE, 2).startswith(prefix)
    assert storage.log_key(RUN, TILE, 2).startswith(prefix)
    assert storage.profile_key(RUN, TILE, "destripe_offsets", 2).startswith(prefix)


class TestAttemptKeying:
    """A retry must not erase the attempt before it (issue #92).

    Every attempt used to write the same keys, so the last write won. Run
    ``2021-2025-20260814T092642Z`` reported a 10-second failure against a
    33-minute wall clock, and the attempt that reached ``land_mask`` -- further
    than that tile had ever gone -- was unrecoverable.
    """

    def test_two_attempts_leave_two_state_objects(self, storage):
        with _beat(storage, attempt=1) as first:
            first.set_phase("land_mask")
            first.set_result(JobResult(job=JOB, status="failed", error="killed"))
        with _beat(storage, attempt=2) as second:
            second.set_result(JobResult(job=JOB, status="completed", scene_count=412))

        assert _published(storage, 1)["status"] == "failed"
        assert _published(storage, 1)["phase_seconds"].get("land_mask") is not None
        assert _published(storage, 2)["status"] == "completed"

    def test_the_pointer_holds_the_last_attempt_to_settle(self, storage):
        """One answer per tile, and it is the newest one."""
        with _beat(storage, attempt=1) as first:
            first.set_result(JobResult(job=JOB, status="failed", error="killed"))
        with _beat(storage, attempt=2) as second:
            second.set_result(JobResult(job=JOB, status="completed"))

        assert _pointer(storage)["status"] == "completed"
        assert _pointer(storage)["attempt"] == 2

    def test_an_attempt_states_which_attempt_it_is(self, storage):
        """The number is in the body as well as the key, so a read is enough."""
        with _beat(storage, attempt=3):
            pass

        assert _published(storage, 3)["attempt"] == 3


class TestGraphSequence:
    """Which graph a task count belongs to.

    A tile runs two hour-scale graphs. A rate spliced across the boundary would
    be an ETA for a graph that already finished, so each one is numbered.
    """

    def test_each_graph_gets_its_own_number(self, storage):
        beat = _beat(storage)

        assert beat.payload()["graph_seq"] == 0
        beat.set_graph_state(GRAPH_RUNNING)
        assert beat.payload()["graph_seq"] == 1
        beat.set_graph_state(GRAPH_IDLE)
        beat.set_graph_state(GRAPH_RUNNING)
        assert beat.payload()["graph_seq"] == 2

    def test_a_repeated_running_state_does_not_count_twice(self, storage):
        """Counted on the idle-to-running edge, so a re-report cannot inflate it."""
        beat = _beat(storage)

        beat.set_graph_state(GRAPH_RUNNING)
        beat.set_graph_state(GRAPH_RUNNING)

        assert beat.payload()["graph_seq"] == 1

    def test_two_graph_progress_blocks_number_themselves(self, storage):
        """The real path: one number per ``GraphProgress``, not per compute."""
        import dask.array as da

        with _beat(storage) as beat:
            with GraphProgress():
                da.ones((4, 4), chunks=2).sum().compute()
            assert beat.payload()["graph_seq"] == 1
            with GraphProgress():
                da.ones((4, 4), chunks=2).sum().compute()
            assert beat.payload()["graph_seq"] == 2


def test_write_final_state_publishes_without_beating(storage, monkeypatch):
    """A tile that does no work publishes twice and starts no thread.

    Entering a heartbeat to publish a number that never changes would cost a
    resumed 700-tile run several hundred thread churns.
    """

    def never(_self):
        msg = "a settled tile must not start a heartbeat thread"
        raise AssertionError(msg)

    monkeypatch.setattr(TileHeartbeat, "__enter__", never)

    write_final_state(
        run_id=RUN,
        job=JOB,
        storage=storage,
        attempt=1,
        result=JobResult(job=JOB, status="skipped"),
    )

    assert [key for key, _ in storage.writes] == [
        storage.run_record_key(RUN, TILE, 1),
        storage.run_record_key(RUN, TILE),
    ]
    assert _published(storage)["phase"] == "skipped"
    assert _published(storage)["status"] == "skipped"
    assert _pointer(storage)["status"] == "skipped"


def test_utc_is_used_for_published_timestamps(storage):
    with _beat(storage):
        stamped = datetime.fromisoformat(_published(storage)["updated_at"])

    assert abs((datetime.now(tz=UTC) - stamped).total_seconds()) < 60


class TestPhaseTimings:
    """A tile that took three hours should say which phase took them."""

    def test_time_accrues_per_phase(self, storage):
        with TileHeartbeat(run_id=RUN, job=JOB, storage=storage, interval_s=3600) as hb:
            hb.set_phase("loading")
            hb.set_phase("destriping")
            payload = hb.payload()

        assert set(payload["phase_seconds"]) >= {"starting", "loading", "destriping"}
        assert all(v >= 0 for v in payload["phase_seconds"].values())

    def test_current_phase_is_counted_before_it_ends(self, storage):
        """A tile killed mid-phase still reports where its hours went."""
        with TileHeartbeat(run_id=RUN, job=JOB, storage=storage, interval_s=3600) as hb:
            hb.set_phase("destriping")
            time.sleep(0.05)
            payload = hb.payload()

        assert payload["phase_seconds"]["destriping"] > 0


class TestGraphProgress:
    """Sub-phase progress: a one-hour dask graph published only its name."""

    def test_reports_task_counts_into_the_heartbeat(self, storage):
        import dask
        import dask.array as da

        with TileHeartbeat(run_id=RUN, job=JOB, storage=storage, interval_s=3600) as hb:
            seen = []
            original = hb.set_task_progress

            def record(done, total):
                seen.append((done, total))
                original(done, total)

            hb.set_task_progress = record
            with GraphProgress():
                dask.compute(da.ones((40, 40), chunks=(10, 10)).sum())

        counted = [pair for pair in seen if pair[0] is not None]
        assert counted, "no task progress was reported"
        assert all(done <= total for done, total in counted)
        assert counted[-1][1] >= counted[0][1] > 0

    def test_progress_is_cleared_when_the_graph_ends(self, storage):
        """A stale fraction on a later phase would be worse than none."""
        import dask
        import dask.array as da

        with TileHeartbeat(run_id=RUN, job=JOB, storage=storage, interval_s=3600) as hb:
            with GraphProgress():
                dask.compute(da.ones((20, 20), chunks=(10, 10)).sum())
            payload = hb.payload()

        assert payload["tasks_done"] is None
        assert payload["tasks_total"] is None

    def test_is_inert_without_a_heartbeat(self):
        """Local runs and benchmarks wrap their computes unconditionally."""
        import dask
        import dask.array as da

        with GraphProgress():
            result = dask.compute(da.ones((10, 10), chunks=(5, 5)).sum())

        assert result[0] == 100


class TestGraphState:
    """Telling "no graph here" apart from "a graph that has not reported".

    Before this, both published ``tasks_total=None``, so a heartbeat could not
    say whether a silent phase was working or wedged. See issue #77 item 4.
    """

    def _beat(self, tmp_path):
        from landsat_lst.progress import TileHeartbeat
        from landsat_lst.storage import LocalStorage

        return TileHeartbeat(run_id="r", job=JOB, storage=LocalStorage(output_dir=tmp_path))

    def test_defaults_to_idle(self, tmp_path):
        assert self._beat(tmp_path).payload()["graph_state"] == "idle"

    def test_graph_progress_marks_running_then_idle(self, tmp_path):
        import dask.array as da

        from landsat_lst.progress import GraphProgress

        heartbeat = self._beat(tmp_path)
        with heartbeat:
            with GraphProgress():
                assert heartbeat.payload()["graph_state"] == "running"
            assert heartbeat.payload()["graph_state"] == "idle"
            # And a real compute inside one still leaves it idle afterwards.
            with GraphProgress():
                da.ones((4, 4), chunks=2).sum().compute()
            assert heartbeat.payload()["graph_state"] == "idle"


class TestTimedSection:
    """The wrapper that lights up stretches running no dask graph."""

    def test_reports_the_phase(self, tmp_path):
        from landsat_lst.progress import TileHeartbeat, timed_section
        from landsat_lst.storage import LocalStorage

        heartbeat = TileHeartbeat(run_id="r", job=JOB, storage=LocalStorage(output_dir=tmp_path))
        with heartbeat, timed_section("composite_graph"):
            assert heartbeat.payload()["phase"] == "composite_graph"

    def test_records_phase_seconds(self, tmp_path):
        from landsat_lst.progress import TileHeartbeat, timed_section
        from landsat_lst.storage import LocalStorage

        heartbeat = TileHeartbeat(run_id="r", job=JOB, storage=LocalStorage(output_dir=tmp_path))
        with heartbeat:
            with timed_section("land_mask"):
                pass
            with timed_section("composite_graph"):
                pass
            assert "land_mask" in heartbeat.payload()["phase_seconds"]

    def test_silenced_sections_do_not_report(self, tmp_path):
        """`landsat-lst plan` builds the same graphs and is not a tile.

        Its `--json` output must stay parseable, so nothing inside a silenced
        block narrates itself.
        """
        from landsat_lst.progress import TileHeartbeat, silence_sections, timed_section
        from landsat_lst.storage import LocalStorage

        heartbeat = TileHeartbeat(run_id="r", job=JOB, storage=LocalStorage(output_dir=tmp_path))
        with heartbeat, silence_sections(), timed_section("composite_graph"):
            assert heartbeat.payload()["phase"] != "composite_graph"

    def test_the_phase_is_reported_even_when_the_body_raises(self, tmp_path):
        from landsat_lst.progress import TileHeartbeat, timed_section
        from landsat_lst.storage import LocalStorage

        heartbeat = TileHeartbeat(run_id="r", job=JOB, storage=LocalStorage(output_dir=tmp_path))
        with heartbeat:
            with pytest.raises(RuntimeError), timed_section("destriping"):
                raise RuntimeError("boom")
            assert heartbeat.payload()["phase"] == "destriping"
