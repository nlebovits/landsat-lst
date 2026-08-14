"""Unit tests for the live view over a batch run's state objects.

Every object a run would leave in storage is written by hand here, with its
modification time set explicitly, so "this tile stopped beating four minutes
ago" is a fact of the fixture rather than of the clock. Beats carry an explicit
``elapsed_s`` for the same reason: every rate below is arithmetic over the
fixture, with no wall clock in it.
"""

import json
import os
import time
from datetime import UTC, datetime
from io import StringIO

import pytest
from rich.console import Console

from landsat_lst.batch import BatchSubmission, submission_path
from landsat_lst.progress import TERMINAL_PHASES
from landsat_lst.storage import LocalStorage
from landsat_lst.watch import (
    _SAMPLE_LIMIT,
    _TERMINAL_CATEGORY,
    RunWatcher,
    TileSample,
    TileStatus,
    _History,
    _rate,
    render_snapshot,
    watch_run,
)

pytestmark = pytest.mark.unit

RUN = "2021-2025-20260813T090000Z"

#: A VM type ``pricing.json`` carries a rate for in the default region, so a
#: cost cell in these tests is a real multiplication rather than a stub.
PRICED_VM = "r6i.2xlarge"


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


def _beat(
    storage: LocalStorage, tile: str, *, phase: str, age_s: float = 0, attempt: int = 1, **extra
) -> None:
    """Publish one attempt's state object, as a running tile does every minute."""
    key = storage.run_record_key(RUN, tile, attempt)
    storage.write_text(key, json.dumps({"tile": tile, "phase": phase, **extra}))
    _age(storage, key, age_s)


def _legacy_beat(
    storage: LocalStorage, tile: str, *, phase: str, age_s: float = 0, **extra
) -> None:
    """Publish a pre-#92 heartbeat, which lived at its own key."""
    key = f"{storage.run_prefix(RUN)}{tile}.progress.json"
    storage.write_text(key, json.dumps({"tile": tile, "phase": phase, **extra}))
    _age(storage, key, age_s)


def _pointer(storage: LocalStorage, tile: str, **body) -> None:
    """Write the unsuffixed key a tile copies its final state to when it stops."""
    storage.write_text(storage.run_record_key(RUN, tile), json.dumps(body))


def _log(storage: LocalStorage, tile: str, attempt: int | None = None) -> None:
    storage.write_text(
        storage.log_key(RUN, tile, attempt), "Traceback...", content_type="text/plain"
    )


def _profile(storage: LocalStorage, tile: str, label: str, attempt: int = 1) -> None:
    storage.write_text(storage.profile_key(RUN, tile, label, attempt), json.dumps({"wall_s": 12.0}))


def _by_tile(snapshot):
    return {tile.tile: tile for tile in snapshot.tiles}


def _run_beats(storage: LocalStorage, tile: str, beats: list[dict]):
    """Publish beats in order, polling between them as a live watcher does.

    Each beat is backdated one second further ahead than the last, so the store
    timestamps rise and the watcher sees each one as new.
    """
    watcher = RunWatcher(RUN, storage=storage)
    snapshot = watcher.poll()
    for index, beat in enumerate(beats):
        _beat(storage, tile, age_s=len(beats) - index, **beat)
        snapshot = watcher.poll()
    return snapshot


def _graph_beat(**fields) -> dict:
    """One beat of a reporting dask graph, with the defaults a real one has."""
    return {
        "phase": "destriping",
        "graph_state": "running",
        "graph_seq": 1,
        "tasks_total": 10000,
        **fields,
    }


def _render(snapshot, **kwargs) -> str:
    console = Console(file=StringIO(), width=200, color_system=None)
    console.print(render_snapshot(snapshot, **kwargs))
    return console.file.getvalue()


class TestClassification:
    def test_a_recent_heartbeat_is_running(self, storage):
        _beat(storage, "N40W075", phase="compositing", age_s=12, scenes_found=1870)

        tile = _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"]

        assert tile.category == "running"
        assert tile.phase == "compositing"
        assert tile.scenes_found == 1870
        assert tile.heartbeat_age_s == pytest.approx(12, abs=5)

    def test_an_old_heartbeat_is_stale(self, storage):
        """The acceptance case: a killed tile must look different from a busy one."""
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

    def test_a_skipped_tile_counts_as_done(self, storage):
        """A tile whose COGs already existed did no work and is still finished."""
        _beat(storage, "N40W075", phase="skipped", age_s=4000)

        assert _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"].category == "done"

    def test_a_failed_tile_carries_its_error_and_log(self, storage):
        _beat(storage, "N40W075", phase="failed", age_s=30, error="No scenes found")
        _log(storage, "N40W075", 1)

        tile = _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"]

        assert tile.category == "failed"
        assert tile.error == "No scenes found"
        assert tile.log_key == storage.log_key(RUN, "N40W075", 1)

    def test_a_pointer_settles_a_tile_whose_last_beat_never_landed(self, storage):
        """The terminal beat is best-effort; the unsuffixed key is the durable one."""
        _beat(storage, "N40W075", phase="uploading", age_s=900)
        _pointer(storage, "N40W075", phase="uploading")

        assert _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"].category == "done"

    def test_a_tile_that_only_left_a_pointer_still_appears(self, storage):
        """A skipped tile publishes once and never beats at all."""
        _pointer(storage, "N40W075", status="skipped")

        assert _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"].category == "done"

    def test_a_retried_tile_renders_its_newest_attempt(self, storage):
        """Attempt 2 is the tile's current state. Attempt 1 is its history."""
        _beat(storage, "N40W075", phase="failed", age_s=900, attempt=1, error="preempted")
        _beat(storage, "N40W075", phase="destriping", age_s=10, attempt=2)

        tile = _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"]

        assert tile.phase == "destriping"
        assert tile.attempt == 2
        assert tile.category == "running"
        assert tile.error is None

    def test_a_profiled_tile_gains_no_phantom_row(self, storage, submitted):
        """``{tile}.{label}.profile.json`` also ends in ``.json``. See #92."""
        submitted("N40W075", "S05W060")
        _beat(storage, "N40W075", phase="destriping")
        _profile(storage, "N40W075", "destripe_offsets")

        snapshot = RunWatcher(RUN, storage=storage).poll()

        assert sorted(_by_tile(snapshot)) == ["N40W075", "S05W060"]
        assert snapshot.counts()["pending"] == 1

    def test_a_legacy_run_still_renders(self, storage):
        """A run in flight when this scheme shipped keeps its own key shape."""
        _legacy_beat(storage, "N40W075", phase="compositing", age_s=10, elapsed_s=42.0)

        tile = _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"]

        assert tile.category == "running"
        assert tile.phase == "compositing"
        assert tile.elapsed_s == 42.0

    def test_submitted_tiles_that_have_not_started_are_pending(self, storage, submitted):
        submitted("N40W075", "S05W060")
        _beat(storage, "N40W075", phase="loading")

        tiles = _by_tile(RunWatcher(RUN, storage=storage).poll())

        assert tiles["N40W075"].category == "running"
        assert tiles["S05W060"].category == "pending"

    def test_an_unreadable_body_is_not_fatal(self, storage):
        """A body read while it is being replaced is ordinary, not an error."""
        storage.write_text(storage.run_record_key(RUN, "N40W075", 1), "{not json")

        tile = _by_tile(RunWatcher(RUN, storage=storage).poll())["N40W075"]

        assert tile.tile == "N40W075"
        assert tile.phase == "unknown"

    def test_every_terminal_phase_has_a_category(self):
        """A phase the writer can publish must be one the watcher can classify."""
        assert set(_TERMINAL_CATEGORY) == set(TERMINAL_PHASES)


class TestPolling:
    def test_unchanged_bodies_are_not_re_read(self, storage):
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
        storage.write_text(storage.run_record_key("another-run", "S05W060", 1), json.dumps({}))
        _beat(storage, "N40W075", phase="loading")

        assert [tile.tile for tile in RunWatcher(RUN, storage=storage).poll().tiles] == ["N40W075"]


class TestSampling:
    def test_one_sample_per_heartbeat_not_per_poll(self, storage):
        """Polls run at 30s and tiles beat at 60. Sampling per poll double-counts."""
        _beat(storage, "N40W075", phase="destriping", age_s=10, elapsed_s=60.0)
        watcher = RunWatcher(RUN, storage=storage)

        for _ in range(3):
            snapshot = watcher.poll()

        assert _by_tile(snapshot)["N40W075"].trend.samples == 1

    def test_the_first_sample_survives_eviction(self):
        """The anchor of "6.0 to 35.1 GB" must outlive the ring buffer."""
        history = _History()
        for index in range(_SAMPLE_LIMIT + 5):
            history.append(_sample(elapsed_s=float(index), rss_mb=float(index)))

        series = history.series()

        assert history.count == _SAMPLE_LIMIT + 5
        assert len(series) == _SAMPLE_LIMIT + 1
        assert series[0].rss_mb == 0.0

    def test_attaching_mid_run_is_marked(self, storage):
        """A tile overwrites one object, so the climb before attach is gone."""
        snapshot = _run_beats(
            storage, "N40W075", [_graph_beat(elapsed_s=600.0, tasks_done=10, rss_mb=6144.0)]
        )

        assert _by_tile(snapshot)["N40W075"].trend.attached_late is True

    def test_attaching_at_the_start_is_not_marked(self, storage):
        snapshot = _run_beats(
            storage, "N40W075", [_graph_beat(elapsed_s=30.0, tasks_done=10, rss_mb=6144.0)]
        )

        assert _by_tile(snapshot)["N40W075"].trend.attached_late is False


class TestRate:
    def test_one_sample_gives_no_rate(self, storage):
        snapshot = _run_beats(storage, "N40W075", [_graph_beat(elapsed_s=60.0, tasks_done=100)])
        tile = _by_tile(snapshot)["N40W075"]

        assert tile.trend.tasks_per_s is None
        assert tile.trend.eta_s is None

    def test_two_samples_give_a_rate_and_an_eta(self, storage):
        snapshot = _run_beats(
            storage,
            "N40W075",
            [
                _graph_beat(elapsed_s=60.0, tasks_done=100),
                _graph_beat(elapsed_s=120.0, tasks_done=400),
            ],
        )
        tile = _by_tile(snapshot)["N40W075"]

        assert tile.trend.tasks_per_s == pytest.approx(5.0)
        assert tile.trend.eta_s == pytest.approx(9600 / 5.0)

    def test_a_stalled_graph_reports_zero_not_unmeasurable(self, storage):
        """``0/s`` is the alarm. ``-`` would hide it behind a plausible blank."""
        snapshot = _run_beats(
            storage,
            "N40W075",
            [
                _graph_beat(elapsed_s=60.0, tasks_done=400),
                _graph_beat(elapsed_s=120.0, tasks_done=400),
            ],
        )
        tile = _by_tile(snapshot)["N40W075"]

        assert tile.trend.tasks_per_s == 0
        assert tile.trend.eta_s is None
        assert "0/s" in _render(snapshot)
        assert "stalled" in _render(snapshot)

    def test_a_new_graph_resets_the_rate(self, storage):
        """A rate spliced across two graphs is an ETA for one that finished."""
        snapshot = _run_beats(
            storage,
            "N40W075",
            [
                _graph_beat(elapsed_s=60.0, tasks_done=9000, graph_seq=1),
                _graph_beat(elapsed_s=120.0, tasks_done=20, graph_seq=2),
            ],
        )
        tile = _by_tile(snapshot)["N40W075"]

        assert tile.trend.tasks_per_s is None
        assert len(tile.trend.epoch_samples) == 1

    def test_a_phase_change_resets_the_rate_without_graph_seq(self, storage):
        """The fallback for a beat written before ``graph_seq`` existed."""
        snapshot = _run_beats(
            storage,
            "N40W075",
            [
                {
                    "phase": "destriping",
                    "graph_state": "running",
                    "tasks_done": 9000,
                    "tasks_total": 10000,
                    "elapsed_s": 60.0,
                },
                {
                    "phase": "exporting",
                    "graph_state": "running",
                    "tasks_done": 20,
                    "tasks_total": 40000,
                    "elapsed_s": 120.0,
                },
            ],
        )

        assert _by_tile(snapshot)["N40W075"].trend.tasks_per_s is None

    def test_a_decreasing_count_never_yields_a_negative_rate(self):
        older = _sample(elapsed_s=60.0, tasks_done=400)
        newer = _sample(elapsed_s=120.0, tasks_done=100)

        assert _rate(older, newer) == 0.0

    def test_a_repeated_timestamp_yields_no_rate(self):
        """Two beats at the same elapsed second cannot be divided."""
        older = _sample(elapsed_s=60.0, tasks_done=100)
        newer = _sample(elapsed_s=60.0, tasks_done=400)

        assert _rate(older, newer) is None

    def test_the_eta_window_is_wider_than_the_displayed_rate(self, storage):
        """One slow beat moves the rate immediately and the finish time barely."""
        beats = [
            _graph_beat(elapsed_s=60.0, tasks_done=0),
            _graph_beat(elapsed_s=120.0, tasks_done=600),
            _graph_beat(elapsed_s=180.0, tasks_done=1200),
            _graph_beat(elapsed_s=240.0, tasks_done=1200),
        ]
        tile = _by_tile(_run_beats(storage, "N40W075", beats))["N40W075"]

        assert tile.trend.tasks_per_s == 0
        assert tile.trend.tasks_per_s_windowed == pytest.approx(1200 / 180)


class TestGraphFraction:
    """Sub-phase progress is the difference between slow and wedged."""

    def _status(self, **fields):
        base = {"tile": "N40W075", "category": "running", "phase": "destriping"}
        return TileStatus(**{**base, **fields})

    def test_renders_a_percentage_while_a_graph_runs(self):
        assert self._status(tasks_done=4182, tasks_total=18600).graph_fraction == "22.5%"

    def test_a_reporting_graph_shows_a_percentage(self):
        assert self._status(tasks_done=50, tasks_total=200).graph_fraction == "25.0%"

    def test_one_decimal_separates_two_early_readings(self):
        """The acceptance case: 1.0% and 1.9% are twenty minutes apart."""
        early = self._status(tasks_done=100, tasks_total=10000).graph_fraction
        later = self._status(tasks_done=190, tasks_total=10000).graph_fraction

        assert (early, later) == ("1.0%", "1.9%")

    def test_is_empty_between_graphs(self):
        assert self._status(phase="uploading").graph_fraction == ""

    def test_a_phase_with_no_graph_shows_idle(self):
        """Graph construction and the land mask are work, not silence."""
        assert self._status(phase="composite_graph", graph_state="idle").graph_fraction == "idle"

    def test_a_graph_that_has_not_reported_shows_starting(self):
        assert self._status(graph_state="running").graph_fraction == "starting"

    def test_an_older_heartbeat_without_the_field_shows_nothing(self):
        """Forward compatibility: a run in flight during a deploy still renders."""
        assert self._status().graph_fraction == ""


class TestRendering:
    def test_shows_the_live_tiles_and_counts_the_rest(self, storage, submitted):
        submitted("N40W075", "S05W060", "N60W150")
        _beat(storage, "N40W075", phase="compositing", scenes_found=1870, scenes_kept=1462)
        _beat(storage, "S05W060", phase="done")

        out = _render(RunWatcher(RUN, storage=storage).poll())

        assert "N40W075" in out
        assert "compositing" in out
        assert "S05W060" not in out
        assert "done: 1" in out
        assert "pending: 1" in out

    def test_scenes_moved_to_the_detail_view(self, storage):
        """The default table carries the columns that change while a tile runs."""
        _beat(storage, "N40W075", phase="destriping", scenes_found=1870, scenes_kept=1462)
        snapshot = RunWatcher(RUN, storage=storage).poll()

        assert "1462/1870" not in _render(snapshot)
        assert "1462/1870" in _render(snapshot, detail=True)

    def test_show_all_gives_every_tile_a_row(self, storage, submitted):
        submitted("N40W075", "S05W060")
        _beat(storage, "N40W075", phase="compositing")
        _beat(storage, "S05W060", phase="done")

        out = _render(RunWatcher(RUN, storage=storage).poll(), show_all=True)

        assert "S05W060" in out

    def test_a_stale_tile_says_how_long_it_has_been_quiet(self, storage):
        _beat(storage, "N40W075", phase="compositing", age_s=305)

        out = _render(RunWatcher(RUN, storage=storage).poll())

        assert "5m05s" in out

    def test_a_failed_tile_points_at_its_log(self, storage):
        _beat(storage, "N40W075", phase="failed", error="No scenes found")
        _log(storage, "N40W075", 1)

        out = _render(RunWatcher(RUN, storage=storage).poll())

        assert "No scenes found" in out
        assert "N40W075.1.log" in out

    def test_the_title_names_the_run(self, storage):
        _beat(storage, "N40W075", phase="loading")

        assert RUN in _render(RunWatcher(RUN, storage=storage).poll())

    def test_the_default_table_draws_no_sparkline(self, storage):
        """A sparkline in a table cell is a trend nobody can read. It is detail."""
        snapshot = _run_beats(
            storage,
            "N40W075",
            [
                _graph_beat(elapsed_s=60.0, tasks_done=100, rss_mb=6144.0),
                _graph_beat(elapsed_s=120.0, tasks_done=400, rss_mb=35942.0),
            ],
        )

        assert not set("▁▂▃▄▅▆▇█") & set(_render(snapshot))

    def test_the_caption_says_what_the_eta_covers(self, storage):
        _beat(storage, "N40W075", phase="destriping")

        assert "not the rest of the tile" in _render(RunWatcher(RUN, storage=storage).poll())


class TestMemoryCell:
    def test_the_rss_cell_carries_its_delta(self, storage):
        """A peak alone cannot tell a climb from a spike an hour ago."""
        snapshot = _run_beats(
            storage,
            "N40W075",
            [
                _graph_beat(elapsed_s=60.0, tasks_done=100, rss_mb=6144.0),
                _graph_beat(elapsed_s=120.0, tasks_done=400, rss_mb=35942.0),
            ],
        )

        assert "6.0→35.1G" in _render(snapshot)

    def test_current_rss_can_fall_where_the_peak_cannot(self, storage):
        snapshot = _run_beats(
            storage,
            "N40W075",
            [
                _graph_beat(elapsed_s=60.0, tasks_done=100, rss_mb=35942.0, peak_rss_mb=35942.0),
                _graph_beat(elapsed_s=120.0, tasks_done=400, rss_mb=4096.0, peak_rss_mb=35942.0),
            ],
        )
        out = _render(snapshot, detail=True)

        assert "memory 4.0G now" in out
        assert "peak 35.1G" in out

    def test_a_late_attach_shows_a_point_not_a_climb(self, storage):
        snapshot = _run_beats(
            storage, "N40W075", [_graph_beat(elapsed_s=900.0, tasks_done=100, rss_mb=35942.0)]
        )
        out = _render(snapshot, detail=True)

        assert "→" not in out
        assert "series starts at attach" in out


class TestCost:
    def _priced(self, storage, **fields):
        _beat(
            storage,
            "N40W075",
            phase="destriping",
            age_s=10,
            elapsed_s=3600.0,
            instance_type=PRICED_VM,
            **fields,
        )
        return _render(RunWatcher(RUN, storage=storage).poll())

    def test_a_measured_on_demand_lifecycle_prices_to_a_point(self, storage):
        out = self._priced(storage, instance_lifecycle="on-demand", instance_source="imds")

        assert "$0.50" in out
        assert "≤" not in out

    def test_an_unmeasured_lifecycle_is_marked_as_a_ceiling(self, storage):
        """The prefix itself says whether the metadata service answered."""
        assert "≤$0.50" in self._priced(storage)

    def test_an_unpriced_tile_says_so(self, storage):
        _beat(storage, "N40W075", phase="destriping", elapsed_s=3600.0)

        assert "≤" not in _render(RunWatcher(RUN, storage=storage).poll())

    def test_the_projection_waits_for_a_completed_tile(self, storage, submitted):
        submitted("N40W075", "S05W060")
        _beat(storage, "N40W075", phase="destriping", elapsed_s=3600.0, instance_type=PRICED_VM)

        assert "projected run" not in _render(RunWatcher(RUN, storage=storage).poll())

        _beat(
            storage,
            "S05W060",
            phase="done",
            elapsed_s=3600.0,
            instance_type=PRICED_VM,
            instance_lifecycle="on-demand",
        )

        assert "projected run" in _render(RunWatcher(RUN, storage=storage).poll(), show_all=True)


class TestDetail:
    def _detail(self, storage, tiles: int = 1) -> str:
        watcher = RunWatcher(RUN, storage=storage)
        for index in range(tiles):
            _beat(
                storage,
                f"N4{index}W075",
                phase="destriping",
                age_s=10,
                elapsed_s=1800.0,
                instance_type=PRICED_VM,
                instance_lifecycle="spot",
                instance_source="imds",
                rss_mb=35942.0,
                peak_rss_mb=35942.0,
                tasks_done=4182,
                tasks_total=18600,
                graph_state="running",
                graph_seq=2,
                scenes_found=1870,
                scenes_kept=1462,
                phase_seconds={"loading": 240.0, "destriping": 1560.0},
            )
        return _render(watcher.poll(), detail=True)

    def test_renders_a_panel_per_live_tile(self, storage):
        out = self._detail(storage)

        assert f"instance {PRICED_VM}" in out
        assert "[spot: imds]" in out

    def test_draws_the_phase_history(self, storage):
        out = self._detail(storage)

        assert "loading" in out
        assert "destriping" in out
        assert "█" in out

    def test_marks_the_phase_the_tile_is_in(self, storage):
        assert "←" in self._detail(storage)

    def test_reports_headroom_against_the_vm(self, storage):
        out = self._detail(storage)

        assert "of 64G" in out
        assert "headroom 28.9G" in out

    def test_caps_the_panels_and_counts_the_rest(self, storage):
        out = self._detail(storage, tiles=6)

        assert "+2 more live tiles" in out

    def test_no_panels_without_the_flag(self, storage):
        _beat(storage, "N40W075", phase="destriping", instance_type=PRICED_VM)

        assert "instance" not in _render(RunWatcher(RUN, storage=storage).poll())


class TestWatchRun:
    def test_once_polls_and_returns(self, storage):
        _beat(storage, "N40W075", phase="compositing")
        console = Console(file=StringIO(), width=200, color_system=None)

        snapshot = watch_run(RUN, storage=storage, once=True, console=console)

        assert [tile.tile for tile in snapshot.tiles] == ["N40W075"]
        assert "N40W075" in console.file.getvalue()

    def test_returns_when_the_run_is_over(self, storage, submitted):
        """A finished run is not a terminal left open forever."""
        submitted("N40W075")
        _beat(storage, "N40W075", phase="done")
        console = Console(file=StringIO(), width=200, color_system=None)

        snapshot = watch_run(RUN, storage=storage, interval_s=0.01, console=console)

        assert snapshot.finished is True

    def test_the_field_is_read_off_the_state_object(self, tmp_path):
        storage = LocalStorage(output_dir=tmp_path)
        storage.write_text(
            "_runs/r/N40W075.1.json",
            json.dumps(
                {
                    "run_id": "r",
                    "tile": "N40W075",
                    "window": "2021-2025",
                    "phase": "composite_graph",
                    "elapsed_s": 12.0,
                    "graph_state": "idle",
                }
            ),
        )

        snapshot = watch_run("r", once=True, storage=storage, console=Console(quiet=True))

        assert snapshot.tiles[0].graph_fraction == "idle"


def _sample(**fields) -> TileSample:
    """One beat, with only the fields a given test cares about."""
    base = {"updated": datetime(2026, 8, 13, 9, 0, tzinfo=UTC), "phase": "destriping"}
    return TileSample(**{**base, **fields})
