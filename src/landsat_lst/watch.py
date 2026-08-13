"""Read a running batch run's heartbeats and render them as one table.

The whole view is built from objects under ``_runs/{run_id}/``, so it works
from any machine with read access to the bucket, long after the shell that
submitted the run is gone. That is the same contract
:func:`landsat_lst.batch.reconcile_run` keeps, for the same reason: a run
outlives the process that started it.

What this answers is liveness, not outcome. A tile that stops beating is wedged
or gone, and a tile whose phase advances is working. The verdict on what a run
produced comes from ``landsat-lst reconcile``, which reads the COG listing
rather than anything a VM claimed about itself.

Heartbeat age is measured against the store's own timestamps rather than the
``updated_at`` inside each object, so a VM with a skewed clock cannot report
itself fresh. Bodies are cached by last-modified time, so a poll re-reads only
the tiles that actually beat since the previous one; finished tiles are read
once for the life of the watch.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from landsat_lst.config import settings
from landsat_lst.storage import get_storage

if TYPE_CHECKING:
    from landsat_lst.storage import StorageBackend

_PROGRESS_SUFFIX = ".progress.json"
_RECORD_SUFFIX = ".json"
_LOG_SUFFIX = ".log"

#: Reads issued at once when several tiles beat between polls. Bounded because a
#: watcher is a bystander: it must not throttle the run it is watching.
_READ_WORKERS = 16

#: Order the table is grouped in. Stale first: it is the only row that asks the
#: operator to do something.
CATEGORIES = ("stale", "running", "failed", "done", "pending")

_LIVE_CATEGORIES = ("stale", "running", "failed")

#: Where a tile lands once its heartbeat stops for good. Keyed by
#: :data:`landsat_lst.progress.TERMINAL_PHASES`, and tested against it, so a new
#: terminal phase cannot arrive here as "unknown".
_TERMINAL_CATEGORY = {"done": "done", "failed": "failed"}


@dataclass(frozen=True)
class TileStatus:
    """One tile's liveness, as of the last poll."""

    tile: str
    category: str
    phase: str
    elapsed_s: float | None = None
    heartbeat_age_s: float | None = None
    scenes_found: int | None = None
    scenes_kept: int | None = None
    peak_rss_mb: float | None = None
    host: str | None = None
    error: str | None = None
    log_key: str | None = None
    tasks_done: int | None = None
    tasks_total: int | None = None

    @property
    def graph_fraction(self) -> str:
        """How far through the running dask graph, as a percentage.

        Empty between graphs, and empty for a phase that runs none. Dask tasks
        are wildly uneven, so this indicates progress; it does not predict a
        finish time.
        """
        if not self.tasks_total or self.tasks_done is None:
            return ""
        return f"{100 * self.tasks_done / self.tasks_total:.0f}%"

    @property
    def is_live(self) -> bool:
        """Whether this row is one an operator watches rather than counts."""
        return self.category in _LIVE_CATEGORIES


@dataclass
class RunSnapshot:
    """Every tile's liveness at one instant, plus what the run expected."""

    run_id: str
    taken_at: datetime
    tiles: list[TileStatus] = field(default_factory=list)
    #: Tiles the submission record says were sent, when that record is at hand.
    #: ``None`` from a machine that did not submit the run.
    submitted: int | None = None

    def counts(self) -> dict[str, int]:
        """Tile count per category, including the categories with none."""
        tally = dict.fromkeys(CATEGORIES, 0)
        for tile in self.tiles:
            tally[tile.category] = tally.get(tile.category, 0) + 1
        return tally

    @property
    def live(self) -> list[TileStatus]:
        """Tiles worth showing a row for: running, stale, or failed."""
        return [tile for tile in self.tiles if tile.is_live]

    @property
    def finished(self) -> bool:
        """Whether every submitted tile has stopped.

        Only answerable with the submission record in hand. Without it, absence
        of running tiles is indistinguishable from a run whose first VM has not
        booted yet, so a watcher keeps waiting rather than declaring victory.
        """
        if self.submitted is None:
            return False
        tally = self.counts()
        return tally["done"] + tally["failed"] >= self.submitted


def _sort_key(tile: TileStatus) -> tuple[int, str]:
    return (CATEGORIES.index(tile.category), tile.tile)


def _storage_for_run(run_id: str) -> StorageBackend:
    """The backend a run wrote to, from its submission record if there is one.

    Watching from a machine that never submitted the run leaves nothing to read
    it from, so the configured backend stands in.
    """
    try:
        from landsat_lst.batch import load_submission  # noqa: PLC0415

        return load_submission(run_id).storage()
    except Exception:
        return get_storage()


class RunWatcher:
    """Polls one run's heartbeat objects, caching the bodies that did not change."""

    def __init__(
        self,
        run_id: str,
        *,
        storage: StorageBackend | None = None,
        stale_after_s: float | None = None,
    ) -> None:
        self.run_id = run_id
        # A run's own submission record knows where its tiles wrote. Falling
        # back to the configured backend would search the local output dir for
        # a distributed run and report every live tile as pending.
        self.storage = storage or _storage_for_run(run_id)
        self.stale_after_s = (
            settings.watch_stale_after_s if stale_after_s is None else stale_after_s
        )
        self._bodies: dict[str, tuple[datetime, dict]] = {}
        self._submitted_tiles = self._load_submitted_tiles()

    def _load_submitted_tiles(self) -> list[str] | None:
        """The submitted tile list, when this machine happens to have it.

        Watching from elsewhere is supported and expected, so a missing record
        costs only the pending count and the ability to know the run is over.
        """
        from landsat_lst.batch import load_submission  # noqa: PLC0415

        try:
            return load_submission(self.run_id).submitted_tiles
        except (FileNotFoundError, ValueError, KeyError):
            return None

    def _read_progress(self, keys: list[str]) -> dict[str, dict]:
        """Bodies for ``keys``, fetched concurrently. Unreadable keys drop out."""

        def read(key: str) -> tuple[str, dict | None]:
            raw = self.storage.read_text(key)
            if raw is None:
                return key, None
            try:
                return key, json.loads(raw)
            except ValueError:
                # A heartbeat read while it is being replaced is ordinary; the
                # next poll gets a whole one.
                return key, None

        if not keys:
            return {}
        with ThreadPoolExecutor(max_workers=min(_READ_WORKERS, len(keys))) as pool:
            return {key: body for key, body in pool.map(read, keys) if body is not None}

    def poll(self) -> RunSnapshot:
        """List the run prefix, refresh what changed, and classify every tile."""
        now = datetime.now(tz=UTC)
        listing = self.storage.list_prefix(self.storage.run_prefix(self.run_id))

        progress: dict[str, datetime] = {}
        reported: set[str] = set()
        logs: dict[str, str] = {}
        for key, modified in listing.items():
            name = key.rsplit("/", 1)[-1]
            if name.endswith(_PROGRESS_SUFFIX):
                progress[key] = modified
            elif name.endswith(_LOG_SUFFIX):
                logs[name[: -len(_LOG_SUFFIX)]] = key
            elif name.endswith(_RECORD_SUFFIX):
                reported.add(name[: -len(_RECORD_SUFFIX)])

        changed = [
            key
            for key, modified in progress.items()
            if key not in self._bodies or self._bodies[key][0] != modified
        ]
        for key, body in self._read_progress(changed).items():
            self._bodies[key] = (progress[key], body)

        tiles = [
            self._status(key, modified, now=now, reported=reported, logs=logs)
            for key, modified in progress.items()
        ]
        seen = {tile.tile for tile in tiles}
        tiles += [
            TileStatus(tile=tile, category="done", phase="done", log_key=logs.get(tile))
            for tile in sorted(reported - seen)
        ]
        seen |= reported
        tiles += [
            TileStatus(tile=tile, category="pending", phase="-")
            for tile in sorted(set(self._submitted_tiles or []) - seen)
        ]

        return RunSnapshot(
            run_id=self.run_id,
            taken_at=now,
            tiles=sorted(tiles, key=_sort_key),
            submitted=None if self._submitted_tiles is None else len(self._submitted_tiles),
        )

    def _status(
        self,
        key: str,
        modified: datetime,
        *,
        now: datetime,
        reported: set[str],
        logs: dict[str, str],
    ) -> TileStatus:
        """Classify one tile from its heartbeat, its age, and what else it left."""
        cached = self._bodies.get(key)
        body = cached[1] if cached else {}
        name = key.rsplit("/", 1)[-1]
        tile = body.get("tile") or name[: -len(_PROGRESS_SUFFIX)]
        phase = body.get("phase", "unknown")
        age = (now - modified).total_seconds()

        if phase in _TERMINAL_CATEGORY:
            category = _TERMINAL_CATEGORY[phase]
        elif tile in reported:
            # A record means the tile stopped and reported, whatever its last
            # heartbeat managed to say before the process went away.
            category = "done"
        elif age > self.stale_after_s:
            category = "stale"
        else:
            category = "running"

        return TileStatus(
            tile=tile,
            category=category,
            phase=phase,
            elapsed_s=body.get("elapsed_s"),
            heartbeat_age_s=age,
            scenes_found=body.get("scenes_found"),
            scenes_kept=body.get("scenes_kept"),
            peak_rss_mb=body.get("peak_rss_mb"),
            tasks_done=body.get("tasks_done"),
            tasks_total=body.get("tasks_total"),
            host=body.get("host"),
            error=body.get("error"),
            log_key=logs.get(tile),
        )


def format_duration(seconds: float | None) -> str:
    """Human-scale duration: ``14s``, ``8m21s``, ``1h02m``."""
    if seconds is None:
        return "-"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"


_PHASE_STYLE = {"failed": "red", "done": "green", "unknown": "yellow"}
_CATEGORY_STYLE = {"stale": "red", "failed": "red", "running": "cyan", "done": "green"}


def _scenes(tile: TileStatus) -> str:
    """``kept/found``, or as much of it as the tile has reached."""
    found = "-" if tile.scenes_found is None else str(tile.scenes_found)
    return found if tile.scenes_kept is None else f"{tile.scenes_kept}/{found}"


#: Longest error text a row carries. The note is the only column allowed to
#: flex, so letting it run long is what squeezes every other column into
#: ellipses on an 80-column terminal. The full text is in the tile's log and in
#: the manifest.
_NOTE_CHARS = 44


def _note(tile: TileStatus) -> str:
    """The row's rightmost cell: why it failed, or how long it has been quiet."""
    if tile.error:
        error = " ".join(tile.error.split())
        return error if len(error) <= _NOTE_CHARS else error[: _NOTE_CHARS - 1] + "…"
    if tile.category == "stale":
        return f"quiet for {format_duration(tile.heartbeat_age_s)}"
    return ""


def render_snapshot(snapshot: RunSnapshot, *, show_all: bool = False):
    """Build what one poll draws: the table, and where to read about failures.

    Only live tiles get a row by default. A 700-tile run would otherwise print
    684 finished rows around the three that are working.

    Every column but the note is fixed to its content, so a narrow terminal
    costs note text rather than turning tile names into ellipses.
    """
    from rich.console import Group  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    rows = snapshot.tiles if show_all else snapshot.live
    tally = snapshot.counts()
    total = len(snapshot.tiles) if snapshot.submitted is None else snapshot.submitted
    counted = "  ".join(f"{name}: {tally[name]}" for name in CATEGORIES if tally[name])

    table = Table(
        title=f"Run {snapshot.run_id} - {total} tiles",
        caption=f"{counted or 'nothing has reported yet'}    polled {snapshot.taken_at:%H:%M:%S}Z",
        caption_justify="left",
        title_justify="left",
        pad_edge=False,
    )
    table.add_column("Tile", no_wrap=True)
    table.add_column("Phase", no_wrap=True)
    table.add_column("Elapsed", justify="right", no_wrap=True)
    table.add_column("Graph", justify="right", no_wrap=True)
    table.add_column("Beat", justify="right", no_wrap=True)
    table.add_column("Scenes", justify="right", no_wrap=True)
    table.add_column("RSS", justify="right", no_wrap=True)
    table.add_column("Note", overflow="ellipsis")

    for tile in rows:
        phase_style = _PHASE_STYLE.get(tile.phase)
        table.add_row(
            tile.tile,
            f"[{phase_style}]{tile.phase}[/]" if phase_style else tile.phase,
            format_duration(tile.elapsed_s),
            tile.graph_fraction or "-",
            format_duration(tile.heartbeat_age_s),
            _scenes(tile),
            "-" if tile.peak_rss_mb is None else f"{tile.peak_rss_mb / 1024:.1f}G",
            _note(tile),
            style=_CATEGORY_STYLE.get(tile.category),
        )

    # Below the table rather than in a cell: the key is longer than any column
    # can hold, and it is the one thing an operator copies out of this view.
    logs = [
        f"[dim]{tile.tile} log: {tile.log_key}[/dim]"
        for tile in rows
        if tile.category == "failed" and tile.log_key
    ]
    return Group(table, *logs) if logs else table


def watch_run(
    run_id: str,
    *,
    storage: StorageBackend | None = None,
    interval_s: float | None = None,
    once: bool = False,
    show_all: bool = False,
    console=None,
) -> RunSnapshot:
    """Render the run's tiles until it finishes or the operator stops watching.

    Args:
        run_id: Run token from :func:`landsat_lst.batch.submit_batch`.
        storage: Backend holding the run prefix.
        interval_s: Seconds between polls (default
            ``settings.watch_poll_interval_s``).
        once: Poll a single time and return, for a scripted check.
        show_all: Give every tile a row, not only the live ones.
        console: Rich console to draw on.

    Returns:
        The last snapshot taken, so a caller can act on it.
    """
    from rich.console import Console  # noqa: PLC0415
    from rich.live import Live  # noqa: PLC0415

    console = console or Console()
    interval = settings.watch_poll_interval_s if interval_s is None else interval_s
    watcher = RunWatcher(run_id, storage=storage)

    snapshot = watcher.poll()
    if once:
        console.print(render_snapshot(snapshot, show_all=show_all))
        return snapshot

    with Live(render_snapshot(snapshot, show_all=show_all), console=console, screen=False) as live:
        while not snapshot.finished:
            time.sleep(interval)
            snapshot = watcher.poll()
            live.update(render_snapshot(snapshot, show_all=show_all))
    return snapshot
