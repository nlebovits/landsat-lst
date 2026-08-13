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

import pytest

from landsat_lst.progress import (
    TERMINAL_PHASES,
    GraphProgress,
    TileHeartbeat,
    active_heartbeat,
    capture_task_log,
    peak_rss_mb,
    report_failed,
    report_phase,
)
from landsat_lst.storage import LocalStorage

RUN, TILE, WINDOW = "run-1", "N40W075", "2021-2025"


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
        "tile": TILE,
        "window": WINDOW,
        "storage": storage,
        "interval_s": 3600,
    }
    return TileHeartbeat(**{**kwargs, **overrides})


def _published(storage) -> dict:
    raw = storage.read_text(storage.progress_key(RUN, TILE))
    assert raw is not None, "no heartbeat was published"
    return json.loads(raw)


class TestHeartbeat:
    def test_publishes_on_entry(self, storage):
        """A tile that has not reached its first phase is still visibly alive."""
        with _beat(storage):
            payload = _published(storage)

        assert payload["phase"] == "starting"
        assert payload["tile"] == TILE
        assert payload["window"] == WINDOW
        assert payload["run_id"] == RUN
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
        with _beat(storage, interval_s=0.05):
            time.sleep(0.3)
            beats = sum(1 for key, _ in storage.writes if key.endswith(".progress.json"))

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


def _log(storage) -> str:
    raw = storage.read_text(storage.log_key(RUN, TILE))
    assert raw is not None, "no task log was uploaded"
    return raw


class TestCaptureTaskLog:
    def test_captures_python_output(self, storage):
        with (
            stdio_on_descriptors(),
            capture_task_log(run_id=RUN, tile=TILE, storage=storage) as key,
        ):
            print("composite written")
            print("warning: overview 5 is short", file=sys.stderr)

        assert key == storage.log_key(RUN, TILE)
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

        assert (storage.log_key(RUN, TILE), "text/plain") in storage.writes

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


def test_heartbeat_and_log_share_the_run_prefix(storage):
    """Everything one tile reports lands where reconciliation already looks."""
    prefix = storage.run_prefix(RUN)

    assert storage.progress_key(RUN, TILE).startswith(prefix)
    assert storage.log_key(RUN, TILE).startswith(prefix)
    assert storage.run_record_key(RUN, TILE).startswith(prefix)


def test_utc_is_used_for_published_timestamps(storage):
    with _beat(storage):
        stamped = datetime.fromisoformat(_published(storage)["updated_at"])

    assert abs((datetime.now(tz=UTC) - stamped).total_seconds()) < 60


class TestPhaseTimings:
    """A tile that took three hours should say which phase took them."""

    def test_time_accrues_per_phase(self, storage):
        with TileHeartbeat(
            run_id=RUN, tile=TILE, window=WINDOW, storage=storage, interval_s=3600
        ) as hb:
            hb.set_phase("loading")
            hb.set_phase("destriping")
            payload = hb.payload()

        assert set(payload["phase_seconds"]) >= {"starting", "loading", "destriping"}
        assert all(v >= 0 for v in payload["phase_seconds"].values())

    def test_current_phase_is_counted_before_it_ends(self, storage):
        """A tile killed mid-phase still reports where its hours went."""
        with TileHeartbeat(
            run_id=RUN, tile=TILE, window=WINDOW, storage=storage, interval_s=3600
        ) as hb:
            hb.set_phase("destriping")
            time.sleep(0.05)
            payload = hb.payload()

        assert payload["phase_seconds"]["destriping"] > 0


class TestGraphProgress:
    """Sub-phase progress: a one-hour dask graph published only its name."""

    def test_reports_task_counts_into_the_heartbeat(self, storage):
        import dask
        import dask.array as da

        with TileHeartbeat(
            run_id=RUN, tile=TILE, window=WINDOW, storage=storage, interval_s=3600
        ) as hb:
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

        with TileHeartbeat(
            run_id=RUN, tile=TILE, window=WINDOW, storage=storage, interval_s=3600
        ) as hb:
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

        return TileHeartbeat(
            run_id="r",
            tile="N40W075",
            window="2021-2025",
            storage=LocalStorage(output_dir=tmp_path),
        )

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

        heartbeat = TileHeartbeat(
            run_id="r",
            tile="N40W075",
            window="2021-2025",
            storage=LocalStorage(output_dir=tmp_path),
        )
        with heartbeat, timed_section("composite_graph"):
            assert heartbeat.payload()["phase"] == "composite_graph"

    def test_records_phase_seconds(self, tmp_path):
        from landsat_lst.progress import TileHeartbeat, timed_section
        from landsat_lst.storage import LocalStorage

        heartbeat = TileHeartbeat(
            run_id="r",
            tile="N40W075",
            window="2021-2025",
            storage=LocalStorage(output_dir=tmp_path),
        )
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

        heartbeat = TileHeartbeat(
            run_id="r",
            tile="N40W075",
            window="2021-2025",
            storage=LocalStorage(output_dir=tmp_path),
        )
        with heartbeat, silence_sections(), timed_section("composite_graph"):
            assert heartbeat.payload()["phase"] != "composite_graph"

    def test_the_phase_is_reported_even_when_the_body_raises(self, tmp_path):
        from landsat_lst.progress import TileHeartbeat, timed_section
        from landsat_lst.storage import LocalStorage

        heartbeat = TileHeartbeat(
            run_id="r",
            tile="N40W075",
            window="2021-2025",
            storage=LocalStorage(output_dir=tmp_path),
        )
        with heartbeat:
            with pytest.raises(RuntimeError), timed_section("destriping"):
                raise RuntimeError("boom")
            assert heartbeat.payload()["phase"] == "destriping"
