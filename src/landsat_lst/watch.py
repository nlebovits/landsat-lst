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
the tiles that actually beat since the previous one, and finished tiles are
read once for the life of the watch.

The key grammar lives in :mod:`landsat_lst.runs`, which is the only module that
parses ``_runs`` names. This one used to test suffixes itself and got one
wrong: ``{tile}.{label}.profile.json`` also ends in ``.json``, so a profiled
tile rendered as a phantom finished row named ``N40W075.destripe_offsets`` and
was subtracted from the pending count.

Trends are sampled once per heartbeat rather than once per poll. Polls run
every 30 seconds and tiles beat every 60, so appending per poll would record
each beat twice and make every rate a function of how often the watcher looked.
There is no history to backfill either. A tile keeps one live object and
overwrites it in place, so attaching late recovers state and not the story
before it. :class:`TileTrend` says so rather than drawing a curve that starts
mid-story.
"""

from __future__ import annotations

import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from landsat_lst.config import settings
from landsat_lst.progress import GRAPH_RUNNING
from landsat_lst.render import (
    format_duration,
    format_gib,
    format_money,
    format_money_range,
    format_rate,
    phase_rows,
    provenance_tag,
    sparkline,
    truncate,
)
from landsat_lst.runs import classify
from landsat_lst.storage import get_storage

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from landsat_lst.runs import TileArtifacts
    from landsat_lst.storage import StorageBackend

#: Reads issued at once when several tiles beat between polls. Bounded because a
#: watcher is a bystander: it must not throttle the run it is watching.
_READ_WORKERS = 16

#: Order the table is grouped in. Stale first: it is the only row that asks the
#: operator to do something.
CATEGORIES = ("stale", "running", "failed", "done", "pending")

_LIVE_CATEGORIES = ("stale", "running", "failed")

#: Where a tile lands once its heartbeat stops for good. Keyed by
#: :data:`landsat_lst.progress.TERMINAL_PHASES`, and tested against it, so a new
#: terminal phase cannot arrive here as "unknown". ``skipped`` maps onto ``done``
#: rather than earning a category: a tile whose COGs already existed is finished
#: from the operator's side, and a sixth column would say nothing an operator
#: acts on.
_TERMINAL_CATEGORY = {"done": "done", "failed": "failed", "skipped": "done"}

#: Beats retained per tile, roughly four hours at
#: ``settings.heartbeat_interval_s``. Long enough to show a memory climb across
#: a whole phase, bounded so a 700-tile watch left open overnight cannot grow
#: without limit.
_SAMPLE_LIMIT = 240

#: Beats the ETA divides across. The displayed rate uses the newest pair, so a
#: decay toward zero is visible on the poll it happens. An ETA computed that way
#: would swing on one slow beat, so it averages over a window instead.
_ETA_WINDOW = 5

#: Panels ``--detail`` draws before it stops and counts the rest. Four fit a
#: screen alongside the table.
_DETAIL_PANELS = 4


@dataclass(frozen=True)
class TileSample:
    """One beat a tile published, kept so a later poll can measure a change."""

    updated: datetime
    phase: str
    elapsed_s: float | None = None
    tasks_done: int | None = None
    tasks_total: int | None = None
    graph_state: str | None = None
    graph_seq: int | None = None
    rss_mb: float | None = None


@dataclass(frozen=True)
class TileTrend:
    """What changed across the beats one watcher retained for one tile.

    ``phase_seconds`` is cumulative in the published object and ``peak_rss_mb``
    is a high-water mark, so both survive a late attach intact. The memory curve
    and the task rate cannot: they start where the watcher did.
    :attr:`attached_late` marks that case so the view says "since attach"
    instead of drawing a climb that began off-screen.
    """

    samples: int = 0
    attached_late: bool = False
    first_rss_mb: float | None = None
    last_rss_mb: float | None = None
    rss_series: list[float | None] = field(default_factory=list)
    #: Rate across the newest pair of beats, so a decay shows immediately.
    tasks_per_s: float | None = None
    #: Rate across up to :data:`_ETA_WINDOW` beats, which is what the ETA
    #: divides by. One slow beat must not move a finish time.
    tasks_per_s_windowed: float | None = None
    eta_s: float | None = None
    #: The beats belonging to the graph running now, oldest first.
    epoch_samples: list[TileSample] = field(default_factory=list)


@dataclass
class _History:
    """Every beat one attempt published, with the oldest kept out of the ring.

    The first beat lives outside the buffer so eviction cannot take it. It is
    the anchor of "6.0 to 35.1 GB", which is the whole of the memory alarm, and
    a ring buffer would drop exactly that end of the series first.
    """

    first: TileSample | None = None
    recent: deque[TileSample] = field(default_factory=lambda: deque(maxlen=_SAMPLE_LIMIT))
    count: int = 0

    def append(self, sample: TileSample) -> None:
        if self.first is None:
            self.first = sample
        self.recent.append(sample)
        self.count += 1

    def series(self) -> list[TileSample]:
        """Every retained beat, oldest first, with the anchor put back."""
        kept = list(self.recent)
        if self.first is not None and (not kept or kept[0] is not self.first):
            kept.insert(0, self.first)
        return kept


def _attached_late(first: TileSample) -> bool:
    """Whether the oldest retained beat is later than the tile's own first.

    A tile publishes on entry, so a first retained beat well past one interval
    means the watcher attached to a tile that was already working. Everything
    before it is gone, because a tile overwrites one object rather than
    appending to a log. Two intervals of slack keeps one delayed first write
    from reading as a late attach.
    """
    if first.elapsed_s is None:
        return False
    return first.elapsed_s > 2 * settings.heartbeat_interval_s


def _same_epoch(sample: TileSample, newest: TileSample) -> bool:
    """Whether two beats describe the same dask graph.

    ``graph_seq`` counts idle-to-running edges, so equal numbers are proof. A
    beat written before that field existed carries none, and the fallback tests
    everything else that would change across a graph boundary. A rate spliced
    across one would be an ETA for a graph that already finished, and a tile
    runs several graphs.
    """
    if newest.graph_seq is not None and sample.graph_seq is not None:
        return sample.graph_seq == newest.graph_seq
    return (
        sample.phase == newest.phase
        and sample.tasks_total == newest.tasks_total
        and sample.graph_state == GRAPH_RUNNING
        and sample.tasks_done is not None
    )


def _epoch(series: list[TileSample]) -> list[TileSample]:
    """The tail of a tile's beats that belongs to the graph running now."""
    if not series:
        return []
    epoch = [series[-1]]
    for sample in reversed(series[:-1]):
        if not _same_epoch(sample, series[-1]):
            break
        if sample.tasks_done is None or epoch[0].tasks_done is None:
            break
        if sample.tasks_done > epoch[0].tasks_done:
            # The count fell going forward, so a graph restarted between these
            # two beats even though everything else about them matched.
            break
        epoch.insert(0, sample)
    return epoch


def _seconds_between(older: TileSample, newer: TileSample) -> float | None:
    """Time between two beats, from the tile's own elapsed clock if it has one.

    ``elapsed_s`` is monotonic on the VM, so it is immune to a store timestamp
    that jitters and to a watcher that polled twice in one second. The store
    timestamps stand in for a beat written before the field existed.
    """
    if older.elapsed_s is not None and newer.elapsed_s is not None:
        return newer.elapsed_s - older.elapsed_s
    return (newer.updated - older.updated).total_seconds()


def _rate(older: TileSample, newer: TileSample) -> float | None:
    """Tasks retired per second between two beats of one graph."""
    if older.tasks_done is None or newer.tasks_done is None:
        return None
    seconds = _seconds_between(older, newer)
    if seconds is None or seconds <= 0:
        return None
    # A count that fell is a graph boundary the epoch walk did not catch. Zero
    # is the honest answer and reads as the stall it is, where a negative rate
    # would render as a finish time in the past.
    return max(newer.tasks_done - older.tasks_done, 0) / seconds


def _eta(newest: TileSample, rate: float | None) -> float | None:
    """Seconds left in the running graph, or ``None`` when that is unknowable.

    A measured zero gives ``None`` rather than infinity. The view renders that
    case as ``stalled``, which is the thing worth reading, where a huge number
    would look like a schedule.
    """
    if rate is None or rate <= 0:
        return None
    if newest.tasks_total is None or newest.tasks_done is None:
        return None
    return max(newest.tasks_total - newest.tasks_done, 0) / rate


def _trend(history: _History) -> TileTrend:
    """Summarize one tile's retained beats into the numbers a row needs."""
    series = history.series()
    if not series:
        return TileTrend()
    epoch = _epoch(series)
    paired = len(epoch) >= 2
    latest = _rate(epoch[-2], epoch[-1]) if paired else None
    windowed = _rate(epoch[max(0, len(epoch) - _ETA_WINDOW)], epoch[-1]) if paired else None
    return TileTrend(
        samples=history.count,
        attached_late=_attached_late(series[0]),
        first_rss_mb=series[0].rss_mb,
        last_rss_mb=series[-1].rss_mb,
        rss_series=[sample.rss_mb for sample in series],
        tasks_per_s=latest,
        tasks_per_s_windowed=windowed,
        eta_s=_eta(epoch[-1], windowed),
        epoch_samples=epoch,
    )


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
    graph_state: str | None = None
    #: What the VM said it was, which is what a cost estimate reads.
    instance_type: str | None = None
    instance_lifecycle: str | None = None
    #: ``"imds"`` when the machine answered, ``"settings"`` when it did not.
    instance_source: str | None = None
    #: Resident memory right now. Distinct from ``peak_rss_mb``, which can only
    #: rise and so cannot show a phase releasing memory.
    rss_mb: float | None = None
    phase_seconds: dict[str, float] | None = None
    graph_seq: int | None = None
    attempt: int | None = None
    trend: TileTrend | None = None

    @property
    def graph_fraction(self) -> str:
        """How far through the running dask graph, or why there is no number.

        Three distinct answers, where there used to be one. A percentage means a
        graph is running and has reported. ``"idle"`` means the tile is in a
        phase that runs no graph at all, such as graph construction or the
        land-mask rasterization, which is work rather than silence.
        ``"starting"`` means a graph is running but has not retired a task yet.

        One decimal place, because these graphs hold hundreds of thousands of
        tasks and a whole percent is over six thousand of them. At integer
        resolution a tile can beat for twenty minutes without the number moving.

        Dask tasks are wildly uneven, so the percentage indicates progress. It
        does not predict a finish time.
        """
        if self.tasks_total and self.tasks_done is not None:
            return f"{100 * self.tasks_done / self.tasks_total:.1f}%"
        if self.graph_state == "idle":
            return "idle"
        if self.graph_state == GRAPH_RUNNING:
            return "starting"
        # An older heartbeat, written before this field existed.
        return ""

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
    """Polls one run's state objects, caching the bodies that did not change."""

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
        # Keyed by body key rather than by tile, so a retry starts its own
        # history. Attempt 2 restarts ``elapsed_s`` and ``graph_seq`` at the
        # beginning, and a rate measured across that boundary is fiction.
        self._samples: dict[str, _History] = {}
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
                # A body read while it is being replaced is ordinary; the next
                # poll gets a whole one.
                return key, None

        if not keys:
            return {}
        with ThreadPoolExecutor(max_workers=min(_READ_WORKERS, len(keys))) as pool:
            return {key: body for key, body in pool.map(read, keys) if body is not None}

    def poll(self) -> RunSnapshot:
        """List the run prefix, refresh what changed, and classify every tile."""
        now = datetime.now(tz=UTC)
        listing = self.storage.list_prefix(self.storage.run_prefix(self.run_id))
        found = classify(listing)
        self._refresh(found.values(), listing)

        tiles = [self._status(artifacts, listing, now=now) for artifacts in found.values()]
        tiles += self._pending(set(found))
        return RunSnapshot(
            run_id=self.run_id,
            taken_at=now,
            tiles=sorted(tiles, key=_sort_key),
            submitted=None if self._submitted_tiles is None else len(self._submitted_tiles),
        )

    def _refresh(self, found: Iterable[TileArtifacts], listing: Mapping[str, datetime]) -> None:
        """Re-read the bodies that changed, and sample each new one once.

        A key whose last-modified time is unchanged is served from cache, so a
        700-tile run polled every 30 seconds re-downloads only the tiles that
        beat. That same test decides when a sample is appended, which is what
        keeps the sample rate the heartbeat's rather than the poll's.
        """
        keys = [artifacts.body_key for artifacts in found if artifacts.body_key is not None]
        changed = [key for key in keys if key in listing and self._is_new(key, listing)]
        for key, body in self._read_progress(changed).items():
            self._bodies[key] = (listing[key], body)
            self._sample(key, listing[key], body)

    def _is_new(self, key: str, listing: Mapping[str, datetime]) -> bool:
        cached = self._bodies.get(key)
        return cached is None or cached[0] != listing[key]

    def _sample(self, key: str, modified: datetime, body: dict) -> None:
        """Record one beat against the attempt that published it."""
        self._samples.setdefault(key, _History()).append(
            TileSample(
                updated=modified,
                phase=body.get("phase", "unknown"),
                elapsed_s=body.get("elapsed_s"),
                tasks_done=body.get("tasks_done"),
                tasks_total=body.get("tasks_total"),
                graph_state=body.get("graph_state"),
                graph_seq=body.get("graph_seq"),
                rss_mb=body.get("rss_mb"),
            )
        )

    def _body(self, key: str | None) -> dict:
        cached = self._bodies.get(key) if key else None
        return cached[1] if cached else {}

    def _category(self, phase: str, *, settled: bool, age_s: float | None) -> str:
        """Which group this tile's row belongs in."""
        if phase in _TERMINAL_CATEGORY:
            return _TERMINAL_CATEGORY[phase]
        if settled:
            # The tile published its final state, whatever its last beat
            # managed to say before the process went away.
            return "done"
        if age_s is None or age_s > self.stale_after_s:
            return "stale"
        return "running"

    def _status(
        self,
        artifacts: TileArtifacts,
        listing: Mapping[str, datetime],
        *,
        now: datetime,
    ) -> TileStatus:
        """Classify one tile from its state object and what else it left."""
        key = artifacts.body_key
        body = self._body(key)
        modified = listing.get(key) if key else None
        age = None if modified is None else (now - modified).total_seconds()
        phase = body.get("phase") or ("done" if artifacts.settled else "unknown")
        history = self._samples.get(key) if key else None

        return TileStatus(
            tile=body.get("tile") or artifacts.tile,
            category=self._category(phase, settled=artifacts.settled, age_s=age),
            phase=phase,
            elapsed_s=body.get("elapsed_s"),
            heartbeat_age_s=age,
            scenes_found=body.get("scenes_found"),
            scenes_kept=body.get("scenes_kept"),
            peak_rss_mb=body.get("peak_rss_mb"),
            rss_mb=body.get("rss_mb"),
            tasks_done=body.get("tasks_done"),
            tasks_total=body.get("tasks_total"),
            graph_state=body.get("graph_state"),
            graph_seq=body.get("graph_seq"),
            phase_seconds=body.get("phase_seconds"),
            instance_type=body.get("instance_type"),
            instance_lifecycle=body.get("instance_lifecycle"),
            instance_source=body.get("instance_source"),
            attempt=artifacts.attempt or None,
            trend=None if history is None else _trend(history),
            host=body.get("host"),
            error=body.get("error"),
            log_key=artifacts.log_key,
        )

    def _pending(self, seen: set[str]) -> list[TileStatus]:
        """Submitted tiles that have published nothing at all yet."""
        return [
            TileStatus(tile=tile, category="pending", phase="-")
            for tile in sorted(set(self._submitted_tiles or []) - seen)
        ]


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
        return truncate(" ".join(tile.error.split()), _NOTE_CHARS)
    if tile.category == "stale":
        return f"quiet for {format_duration(tile.heartbeat_age_s)}"
    return ""


def _rate_cell(tile: TileStatus) -> str:
    """Task rate across the newest pair of beats."""
    return "-" if tile.trend is None else format_rate(tile.trend.tasks_per_s)


def _eta_cell(tile: TileStatus) -> str:
    """Time left in the graph running now, or why there is no number.

    ``stalled`` rather than a duration when the rate is a measured zero. A tile
    that has retired no task across its last beats has no finish time, and the
    enormous number that division would produce reads like a schedule.
    """
    trend = tile.trend
    if trend is None:
        return "-"
    if trend.eta_s is not None:
        return format_duration(trend.eta_s)
    if trend.tasks_per_s_windowed == 0:
        return "stalled"
    return "-"


def _current_rss(tile: TileStatus) -> float | None:
    """Memory right now, from the beat if it carries one, else from the trend."""
    if tile.rss_mb is not None:
        return tile.rss_mb
    return tile.trend.last_rss_mb if tile.trend else None


def _rss_cell(tile: TileStatus) -> str:
    """Current memory, carrying where it started when that is known.

    ``6.0→35.1G`` is the alarm the peak alone cannot raise, because a peak only
    rises and so reads the same whether a tile climbed to 35 GB or touched it
    once an hour ago. A watcher that attached mid-run has no start to show and
    prints the point rather than inventing a climb.
    """
    trend = tile.trend
    now_mb = _current_rss(tile)
    if now_mb is None:
        return format_gib(tile.peak_rss_mb)
    if trend is None or trend.attached_late or trend.first_rss_mb is None:
        return format_gib(now_mb)
    start, end = format_gib(trend.first_rss_mb), format_gib(now_mb)
    return end if start == end else f"{start[:-1]}→{end}"


def _tile_cost(tile: TileStatus) -> Any:
    """This tile's spend so far, or ``None`` when it cannot be priced."""
    from landsat_lst.pricing import tile_cost  # noqa: PLC0415

    if tile.elapsed_s is None or not tile.instance_type:
        return None
    return tile_cost(
        duration_s=tile.elapsed_s,
        instance_type=tile.instance_type,
        lifecycle=tile.instance_lifecycle,
    )


def _cost_cell(tile: TileStatus) -> str:
    """Spend so far, never a projection.

    Projecting one tile needs a finish time for the whole tile, and nothing
    published says how many dask graphs are left to run. The ``≤`` prefix marks
    a figure whose lifecycle was never measured, so the prefix itself says
    whether the machine's metadata service answered.
    """
    estimate = _tile_cost(tile)
    if estimate is None:
        return "-"
    if estimate.usd.is_point:
        return format_money(estimate.usd.low)
    return f"≤{format_money(estimate.usd.high)}"


#: Table columns and how each is aligned. Scenes moved to ``--detail``: it does
#: not change once ``destriping`` starts, and the columns that do change are
#: what a live view is for.
_COLUMNS = (
    ("Tile", "left"),
    ("Phase", "left"),
    ("Elapsed", "right"),
    ("Graph", "right"),
    ("Rate", "right"),
    ("ETA", "right"),
    ("Beat", "right"),
    ("RSS", "right"),
    ("Cost", "right"),
)


def _row(tile: TileStatus) -> list[str]:
    """One table row, in :data:`_COLUMNS` order plus the note."""
    phase_style = _PHASE_STYLE.get(tile.phase)
    return [
        tile.tile,
        f"[{phase_style}]{tile.phase}[/]" if phase_style else tile.phase,
        format_duration(tile.elapsed_s),
        tile.graph_fraction or "-",
        _rate_cell(tile),
        _eta_cell(tile),
        format_duration(tile.heartbeat_age_s),
        _rss_cell(tile),
        _cost_cell(tile),
        _note(tile),
    ]


def _table(snapshot: RunSnapshot, rows: list[TileStatus]) -> Any:
    """The table itself, with every column but the note fixed to its content."""
    from rich.table import Table  # noqa: PLC0415

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
    for name, justify in _COLUMNS:
        table.add_column(name, justify=justify, no_wrap=True)
    table.add_column("Note", overflow="ellipsis")

    for tile in rows:
        table.add_row(*_row(tile), style=_CATEGORY_STYLE.get(tile.category))
    return table


def _projection(snapshot: RunSnapshot) -> str | None:
    """What the whole run looks like it will cost, from the tiles that finished.

    Nothing is printed until a tile completes. A mean over tiles still running
    would divide a partial spend across a whole run and read low by however far
    along those tiles happen to be.
    """
    from landsat_lst.pricing import DISCLAIMER, fleet_cost  # noqa: PLC0415

    settled = [tile for tile in snapshot.tiles if tile.category == "done"]
    estimates = [cost for cost in (_tile_cost(tile) for tile in settled) if cost is not None]
    if not estimates:
        return None
    tiles = snapshot.submitted or len(snapshot.tiles)
    cost = fleet_cost(estimates, tiles=tiles)
    if cost is None:
        return None
    return (
        f"[dim]projected run {format_money_range(cost.usd.low, cost.usd.high)}"
        f" over {tiles} tiles, from {cost.observed_tiles} completed. {DISCLAIMER}[/dim]"
    )


def _footnotes(snapshot: RunSnapshot, rows: list[TileStatus]) -> list[str]:
    """Lines under the table: what the ETA covers, the projection, and logs.

    Log keys go below the table rather than in a cell. A key is longer than any
    column can hold, and it is the one thing an operator copies out of this
    view.
    """
    if not rows:
        return []
    notes = ["[dim]ETA covers the dask graph running now, not the rest of the tile.[/dim]"]
    projection = _projection(snapshot)
    if projection:
        notes.append(projection)
    notes += [
        f"[dim]{tile.tile} log: {tile.log_key}[/dim]"
        for tile in rows
        if tile.category == "failed" and tile.log_key
    ]
    return notes


def _detail_instance(tile: TileStatus) -> str:
    """Which machine, how that was learned, and what it has cost so far."""
    if not tile.instance_type:
        return ""
    tag = provenance_tag(tile.instance_lifecycle or "lifecycle unknown", tile.instance_source)
    estimate = _tile_cost(tile)
    if estimate is None:
        return f"instance {tile.instance_type} {tag}"
    spend = format_money_range(estimate.usd.low, estimate.usd.high)
    return f"instance {tile.instance_type} {tag}   spent so far {spend}"


def _detail_graph(tile: TileStatus) -> str:
    """Graph number, counts, rate, and how the rate has been moving."""
    counts = "-" if not tile.tasks_total else f"{tile.tasks_done or 0}/{tile.tasks_total}"
    seq = "-" if tile.graph_seq is None else str(tile.graph_seq)
    line = (
        f"graph {seq}   {counts} tasks {tile.graph_fraction or '-'}"
        f"   {_rate_cell(tile)}   eta {_eta_cell(tile)} (this graph)"
    )
    trend = tile.trend
    if trend is None or len(trend.epoch_samples) < 2:
        return line
    pairs = zip(trend.epoch_samples, trend.epoch_samples[1:], strict=False)
    spark, top = sparkline([_rate(older, newer) for older, newer in pairs])
    return line if not spark else f"{line}\n  rate {spark} top {format_rate(top)}"


def _detail_memory(tile: TileStatus) -> str:
    """Memory now, its peak, and what is left of the VM it runs on."""
    from landsat_lst.pricing import instance_memory_gib  # noqa: PLC0415

    line = f"memory {format_gib(_current_rss(tile))} now   peak {format_gib(tile.peak_rss_mb)}"
    limit = instance_memory_gib(tile.instance_type) if tile.instance_type else None
    if limit is not None and tile.peak_rss_mb is not None:
        line += f" of {limit:.0f}G   headroom {limit - tile.peak_rss_mb / 1024:.1f}G"

    trend = tile.trend
    if trend is None:
        return line
    if trend.attached_late:
        # The climb this tile made before the watcher attached is unrecoverable,
        # because a tile overwrites one object rather than appending to a log.
        return f"{line}\n  [dim]memory series starts at attach[/dim]"
    spark, top = sparkline(trend.rss_series)
    return line if not spark else f"{line}\n  rss  {spark} top {format_gib(top)}"


def _detail_phases(tile: TileStatus) -> list[str]:
    """Where the tile's wall clock has gone, one bar per phase."""
    return [
        f"  {name:<16}{duration:>8}  {drawn}"
        for name, duration, drawn in phase_rows(tile.phase_seconds or {}, current=tile.phase)
    ]


def _panel(tile: TileStatus) -> Any:
    """Everything about one live tile that does not fit a table row."""
    from rich.panel import Panel  # noqa: PLC0415

    lines = [
        _detail_instance(tile),
        _detail_graph(tile),
        _detail_memory(tile),
        *_detail_phases(tile),
        f"scenes {_scenes(tile)}   attempt {tile.attempt or 1}",
    ]
    return Panel(
        "\n".join(line for line in lines if line),
        title=f"{tile.tile}  {tile.phase}",
        title_align="left",
        expand=False,
    )


def _panels(rows: list[TileStatus]) -> list[Any]:
    """One panel per live tile, capped so a busy run stays readable."""
    live = [tile for tile in rows if tile.is_live]
    panels: list[Any] = [_panel(tile) for tile in live[:_DETAIL_PANELS]]
    if len(live) > _DETAIL_PANELS:
        panels.append(f"[dim]+{len(live) - _DETAIL_PANELS} more live tiles[/dim]")
    return panels


def render_snapshot(snapshot: RunSnapshot, *, show_all: bool = False, detail: bool = False) -> Any:
    """Build what one poll draws: the table, its footnotes, and any panels.

    Only live tiles get a row by default. A 700-tile run would otherwise print
    684 finished rows around the three that are working.

    Every column but the note is fixed to its content, so a narrow terminal
    costs note text rather than turning tile names into ellipses.
    """
    from rich.console import Group  # noqa: PLC0415

    rows = snapshot.tiles if show_all else snapshot.live
    parts: list[Any] = [_table(snapshot, rows), *_footnotes(snapshot, rows)]
    if detail:
        parts += _panels(rows)
    return Group(*parts)


def watch_run(
    run_id: str,
    *,
    storage: StorageBackend | None = None,
    interval_s: float | None = None,
    once: bool = False,
    show_all: bool = False,
    detail: bool = False,
    console: Any = None,
) -> RunSnapshot:
    """Render the run's tiles until it finishes or the operator stops watching.

    Args:
        run_id: Run token from :func:`landsat_lst.batch.submit_batch`.
        storage: Backend holding the run prefix.
        interval_s: Seconds between polls (default
            ``settings.watch_poll_interval_s``).
        once: Poll a single time and return, for a scripted check.
        show_all: Give every tile a row, not only the live ones.
        detail: Add a panel per live tile with its memory trend, task rate,
            phase history, and cost.
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
    view = render_snapshot(snapshot, show_all=show_all, detail=detail)
    if once:
        console.print(view)
        return snapshot

    # Overflow stays visible rather than cropped to the terminal. A detail panel
    # is several rows tall, and a panel silently cut in half is worse than one
    # that scrolls.
    with Live(view, console=console, screen=False, vertical_overflow="visible") as live:
        while not snapshot.finished:
            time.sleep(interval)
            snapshot = watcher.poll()
            live.update(render_snapshot(snapshot, show_all=show_all, detail=detail))
    return snapshot
