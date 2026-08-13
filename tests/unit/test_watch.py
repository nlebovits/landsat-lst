"""Unit tests for the live view over a batch run's heartbeats.

Every object a run would leave in storage is written by hand here, with its
modification time set explicitly, so "this tile stopped beating four minutes
ago" is a fact of the fixture rather than of the clock.
"""

import json
import os
import time
from io import StringIO

import pytest
from rich.console import Console

from landsat_lst.batch import BatchSubmission, submission_path
from landsat_lst.progress import TERMINAL_PHASES
from landsat_lst.storage import LocalStorage
from landsat_lst.watch import (
    _TERMINAL_CATEGORY,
    RunWatcher,
    format_duration,
    render_snapshot,
    watch_run,
)

RUN = "2021-2025-20260813T090000Z"


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(output_dir=tmp_path / "cogs")


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    """Point the submission lookup at a scratch directory.

    Left empty, it is what a machine that did not launch the run sees, which is
    the default this module tests against.
    """
    from landsat_lst.config import settings

    monkeypatch.setattr(settings, "manifest_dir", tmp_path / "runs")


@pytest.fixture
def submitted(runs_dir):
    """Write a submission record, as the machine that launched the run has."""

    def _submit(*tiles: str) -> None:
        submission = BatchSubmission(
            run_id=RUN,
            window="2021-2025",
            cluster_id=4242,
            job_id=77,
            submitted_at="2026-08-13T09:00:00+00:00",
            submitted_tiles=list(tiles),
            year=2021,
            end_year=2025,
        )
        path = submission_path(RUN)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(submission.to_dict()))

    return _submit


def _age(storage: LocalStorage, key: str, seconds: float) -> None:
    """Backdate an object, which is how a stale heartbeat is made."""
    stamp = time.time() - seconds
    os.utime(storage.output_dir / key, (stamp, stamp))


def _beat(storage: LocalStorage, tile: str, *, phase: str, age_s: float = 0, **extra) -> None:
    key = storage.progress_key(RUN, tile)
    storage.write_text(key, json.dumps({"tile": tile, "phase": phase, **extra}))
    _age(storage, key, age_s)


def _record(storage: LocalStorage, tile: str, *, status: str = "completed") -> None:
    storage.write_text(storage.run_record_key(RUN, tile), json.dumps({"status": status}))


def _log(storage: LocalStorage, tile: str) -> None:
    storage.write_text(storage.log_key(RUN, tile), "Traceback...", content_type="text/plain")


def _by_tile(snapshot):
    return {tile.tile: tile for tile in snapshot.tiles}


class TestClassification:
    def test_a_recent_heartbeat_is_running(self, storage):
        _beat(storage, "N40W075", phase="compositing", age_s=12, scenes_found=1870)

        tile = _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"]

        assert tile.category == "running"
        assert tile.phase == "compositing"
        assert tile.scenes_found == 1870
        assert tile.heartbeat_age_s == pytest.approx(12, abs=5)

    def test_an_old_heartbeat_is_stale(self, storage):
        """The acceptance case: a killed tile has to look different from a busy one."""
        _beat(storage, "N40W075", phase="compositing", age_s=600)

        tile = _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"]

        assert tile.category == "stale"

    def test_staleness_threshold_is_configurable(self, storage):
        _beat(storage, "N40W075", phase="compositing", age_s=90)

        watcher = RunWatcher(RUN, storage=storage, stale_after_s=60)

        assert _by_tile(watcher.poll())["N40W075"].category == "stale"

    def test_age_comes_from_the_store_not_the_payload(self, storage):
        """A VM with a skewed clock must not be able to report itself fresh."""
        _beat(
            storage,
            "N40W075",
            phase="compositing",
            age_s=600,
            updated_at="2099-01-01T00:00:00+00:00",
        )

        assert _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"].category == "stale"

    def test_a_finished_tile_is_done(self, storage):
        _beat(storage, "N40W075", phase="done", age_s=4000)

        assert _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"].category == "done"

    def test_a_failed_tile_carries_its_error_and_log(self, storage):
        _beat(storage, "N40W075", phase="failed", age_s=30, error="No scenes found")
        _log(storage, "N40W075")

        tile = _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"]

        assert tile.category == "failed"
        assert tile.error == "No scenes found"
        assert tile.log_key == storage.log_key(RUN, "N40W075")

    def test_a_record_settles_a_tile_whose_last_beat_never_landed(self, storage):
        """The terminal write is best-effort; the record is the durable one."""
        _beat(storage, "N40W075", phase="uploading", age_s=900)
        _record(storage, "N40W075")

        assert _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"].category == "done"

    def test_a_tile_that_only_left_a_record_still_appears(self, storage):
        """A skipped tile writes a record and never beats at all."""
        _record(storage, "N40W075", status="skipped")

        assert _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"].category == "done"

    def test_submitted_tiles_that_have_not_started_are_pending(self, storage, submitted):
        submitted("N40W075", "S05W060")
        _beat(storage, "N40W075", phase="loading")

        tiles = _by_tile(RunWatcher(RUN, storage=storage).poll())

        assert tiles["N40W075"].category == "running"
        assert tiles["S05W060"].category == "pending"

    def test_an_unreadable_heartbeat_is_not_fatal(self, storage):
        """A body read while it is being replaced is ordinary, not an error."""
        key = storage.progress_key(RUN, "N40W075")
        storage.write_text(key, "{not json")

        tile = _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"]

        assert tile.tile == "N40W075"
        assert tile.phase == "unknown"

    def test_every_terminal_phase_has_a_category(self):
        """A phase the writer can publish must be one the watcher can classify."""
        assert set(_TERMINAL_CATEGORY) == set(TERMINAL_PHASES)


class TestPolling:
    def test_unchanged_heartbeats_are_not_re_read(self, storage):
        """A 700-tile run polled every 30s must not re-download 700 finished tiles."""
        reads: list[str] = []
        original = storage.read_text
        storage.read_text = lambda key: (reads.append(key), original(key))[1]

        _beat(storage, "N40W075", phase="done")
        watcher = RunWatcher(RUN, storage=storage)
        watcher.poll()
        reads.clear()
        watcher.poll()

        assert reads == []

    def test_a_new_beat_is_re_read(self, storage):
        watcher = RunWatcher(RUN, storage=storage)
        _beat(storage, "N40W075", phase="loading")
        watcher.poll()

        _beat(storage, "N40W075", phase="compositing", age_s=-1)

        assert _by_tile(watcher.poll())["N40W075"].phase == "compositing"

    def test_counts_cover_every_category(self, storage, submitted):
        submitted("N40W075", "S05W060", "N60W150", "S30E025")
        _beat(storage, "N40W075", phase="compositing")
        _beat(storage, "S05W060", phase="loading", age_s=600)
        _beat(storage, "N60W150", phase="failed")

        counts = RunWatcher(RUN, storage=storage).poll().counts()

        assert counts == {"running": 1, "stale": 1, "failed": 1, "done": 0, "pending": 1}

    def test_live_tiles_exclude_the_settled_ones(self, storage, submitted):
        submitted("N40W075", "S05W060", "N60W150")
        _beat(storage, "N40W075", phase="compositing")
        _beat(storage, "S05W060", phase="done")

        live = RunWatcher(RUN, storage=storage).poll().live

        assert [tile.tile for tile in live] == ["N40W075"]

    def test_finished_needs_every_submitted_tile_to_have_stopped(self, storage, submitted):
        submitted("N40W075", "S05W060")
        _beat(storage, "N40W075", phase="done")
        watcher = RunWatcher(RUN, storage=storage)

        assert watcher.poll().finished is False

        _beat(storage, "S05W060", phase="failed")

        assert watcher.poll().finished is True

    def test_a_run_this_machine_did_not_submit_never_declares_itself_over(self, storage):
        """With no tile list, quiet storage is indistinguishable from a cold start."""
        snapshot = RunWatcher(RUN, storage=storage).poll()

        assert snapshot.submitted is None
        assert snapshot.finished is False

    def test_an_empty_run_polls_cleanly(self, storage):
        snapshot = RunWatcher(RUN, storage=storage).poll()

        assert snapshot.tiles == []
        assert snapshot.run_id == RUN

    def test_other_runs_are_not_mixed_in(self, storage):
        storage.write_text(storage.progress_key("another-run", "S05W060"), json.dumps({}))
        _beat(storage, "N40W075", phase="loading")

        assert [tile.tile for tile in RunWatcher(RUN, storage=storage).poll().tiles] == ["N40W075"]


class TestRendering:
    def _render(self, snapshot, **kwargs) -> str:
        console = Console(file=StringIO(), width=200, color_system=None)
        console.print(render_snapshot(snapshot, **kwargs))
        return console.file.getvalue()

    def test_shows_the_live_tiles_and_counts_the_rest(self, storage, submitted):
        submitted("N40W075", "S05W060", "N60W150")
        _beat(storage, "N40W075", phase="compositing", scenes_found=1870, scenes_kept=1462)
        _beat(storage, "S05W060", phase="done")

        out = self._render(RunWatcher(RUN, storage=storage).poll())

        assert "N40W075" in out
        assert "compositing" in out
        assert "1462/1870" in out
        assert "S05W060" not in out
        assert "done: 1" in out
        assert "pending: 1" in out

    def test_show_all_gives_every_tile_a_row(self, storage, submitted):
        submitted("N40W075", "S05W060")
        _beat(storage, "N40W075", phase="compositing")
        _beat(storage, "S05W060", phase="done")

        out = self._render(RunWatcher(RUN, storage=storage).poll(), show_all=True)

        assert "S05W060" in out

    def test_a_stale_tile_says_how_long_it_has_been_quiet(self, storage):
        _beat(storage, "N40W075", phase="compositing", age_s=305)

        out = self._render(RunWatcher(RUN, storage=storage).poll())

        assert "5m05s" in out

    def test_a_failed_tile_points_at_its_log(self, storage):
        _beat(storage, "N40W075", phase="failed", error="No scenes found")
        _log(storage, "N40W075")

        out = self._render(RunWatcher(RUN, storage=storage).poll())

        assert "No scenes found" in out
        assert "N40W075.log" in out

    def test_the_title_names_the_run(self, storage):
        _beat(storage, "N40W075", phase="loading")

        assert RUN in self._render(RunWatcher(RUN, storage=storage).poll())


class TestWatchRun:
    def test_once_polls_and_returns(self, storage):
        _beat(storage, "N40W075", phase="compositing")
        console = Console(file=StringIO(), width=200, color_system=None)

        snapshot = watch_run(RUN, storage=storage, once=True, console=console)

        assert [tile.tile for tile in snapshot.tiles] == ["N40W075"]
        assert "N40W075" in console.file.getvalue()

    def test_returns_when_the_run_is_over(self, storage, submitted):
        """A finished run must not hold the terminal open forever."""
        submitted("N40W075")
        _beat(storage, "N40W075", phase="done")
        console = Console(file=StringIO(), width=200, color_system=None)

        snapshot = watch_run(RUN, storage=storage, interval_s=0.01, console=console)

        assert snapshot.finished is True


def test_format_duration():
    assert format_duration(None) == "-"
    assert format_duration(0) == "0s"
    assert format_duration(14.6) == "14s"
    assert format_duration(501) == "8m21s"
    assert format_duration(3720) == "1h02m"
