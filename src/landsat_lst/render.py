"""Strings the run-reporting commands share, with no terminal involved.

``watch`` and ``explain`` describe the same published state object from
opposite ends of a run, one while a tile works and one after it stops. Both had
started to grow their own answer to "how long is that in hours" and "how wide
is that bar", and two copies of a format drift until the two commands disagree
about one tile.

Every function here returns a plain string, so a test asserts on the exact text
rather than on a rich renderable, and no test needs a console. Nothing imports
rich either. The single concession to it is :func:`provenance_tag`, which
escapes its own opening bracket, because rich reads a bare ``[measured]`` as a
style tag and prints nothing at all.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from landsat_lst.progress import PHASES

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

#: Eight block characters, lightest first. A sparkline drawn from these carries
#: one value per column, which is as much resolution as a table cell has.
_BLOCKS = "▁▂▃▄▅▆▇█"

_BAR_FULL = "█"
_BAR_EMPTY = "░"

#: Written beside the phase a tile is in right now, so a phase history reads as
#: a position rather than as a finished accounting.
_CURRENT_MARKER = "← now"

#: Default width of a phase-history bar. Narrow enough that a phase name, its
#: duration, and its bar fit an 80-column terminal together.
_BAR_WIDTH = 20

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_MB_PER_GIB = 1024


def format_duration(seconds: float | None) -> str:
    """Human-scale duration: ``14s``, ``8m21s``, ``1h02m``."""
    if seconds is None:
        return "-"
    total = int(seconds)
    if total < _SECONDS_PER_MINUTE:
        return f"{total}s"
    if total < _SECONDS_PER_HOUR:
        return f"{total // _SECONDS_PER_MINUTE}m{total % _SECONDS_PER_MINUTE:02d}s"
    hours, rest = divmod(total, _SECONDS_PER_HOUR)
    return f"{hours}h{rest // _SECONDS_PER_MINUTE:02d}m"


def format_gib(mb: float | None) -> str:
    """Mebibytes as gibibytes, ``35.1G``, or ``-`` when nothing was measured.

    Memory is reported in MiB because that is what ``/proc`` and ``getrusage``
    hand back, and read in GiB because that is the unit an instance type is
    sold in.
    """
    if mb is None:
        return "-"
    return f"{mb / _MB_PER_GIB:.1f}G"


def format_rate(per_s: float | None) -> str:
    """Tasks per second, where ``-`` and ``0/s`` say different things.

    ``-`` means not measurable, which is what a graph reports until it has
    published twice. ``0/s`` is a measurement, and it is the alarm worth
    raising: a graph that retired no task between two heartbeats is stalled.
    Rendering the first as the second would hide every stall behind a number
    that looks like an answer.
    """
    if per_s is None:
        return "-"
    if per_s == 0:
        return "0/s"
    if per_s >= 100:
        return f"{per_s:.0f}/s"
    if per_s >= 10:
        return f"{per_s:.1f}/s"
    return f"{per_s:.2f}/s"


def format_money(usd: float | None) -> str:
    """One dollar figure, or ``-`` when there is none.

    A positive figure under a cent prints as ``<$0.01`` rather than ``$0.00``,
    because a cost that rounds to nothing is still a cost and zero is a claim.
    """
    if usd is None:
        return "-"
    if 0 < usd < 0.01:
        return "<$0.01"
    return f"${usd:,.2f}"


def format_money_range(low: float | None, high: float | None) -> str:
    """An interval, collapsed to one figure when the two ends agree.

    Costs arrive as ranges. An unreported instance lifecycle spans 0.30 to 1.00
    of the on-demand rate, and printing either end alone would put back a
    precision the input never had.
    """
    if low is None or high is None:
        return "-"
    if low == high:
        return format_money(low)
    return f"{format_money(low)}-{format_money(high)}"


def _decimate(points: Sequence[float], width: int) -> list[float]:
    """Squeeze a series into ``width`` buckets, keeping each bucket's maximum.

    Averaging would be the obvious choice and the wrong one. The thing a memory
    series is watched for is a spike, and a mean over a bucket is exactly what
    erases one.
    """
    if len(points) <= width:
        return list(points)
    size = len(points) / width
    return [max(points[int(i * size) : int((i + 1) * size)]) for i in range(width)]


def sparkline(values: Iterable[float | None], *, width: int = 24) -> tuple[str, float]:
    """A one-line trend, and the value the top block stands for.

    Scaled from zero rather than from the series minimum. A min-max sparkline
    of memory that moved from 34.9 to 35.1 GB draws the same dramatic climb as
    one that moved from 6 to 35, and the second is an alarm while the first is
    noise. Scaling from zero keeps the two apart.

    Fewer than two values give an empty string, because one point is a reading
    rather than a trend. The scale top comes back alongside the glyphs so a
    caller can print what the tallest block means.
    """
    points = [float(value) for value in values if value is not None]
    if len(points) < 2:
        return "", 0.0
    buckets = _decimate(points, width)
    top = max(buckets)
    if top <= 0:
        return _BLOCKS[0] * len(buckets), top
    last = len(_BLOCKS) - 1
    return "".join(_BLOCKS[min(round(value / top * last), last)] for value in buckets), top


def bar(fraction: float | None, *, width: int) -> str:
    """A fixed-width bar for ``fraction`` of one, empty when there is no number.

    Fractions outside zero to one are clamped rather than rejected. A bar is a
    comparison, and a caller with a slightly stale total should get a full bar
    instead of an exception.
    """
    if fraction is None:
        return ""
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return _BAR_FULL * filled + _BAR_EMPTY * (width - filled)


def _phase_order(names: Iterable[str]) -> list[str]:
    """Pipeline order for the phases we know, then anything else by name.

    Terminal phases and any phase added after this code was written land at the
    end rather than being dropped, so a run in flight during a deploy still
    reports all of its time.
    """
    seen = list(names)
    known = [phase for phase in PHASES if phase in seen]
    return known + sorted(name for name in seen if name not in PHASES)


def phase_rows(
    phase_seconds: Mapping[str, float],
    *,
    current: str | None = None,
    width: int = _BAR_WIDTH,
) -> list[tuple[str, str, str]]:
    """Where one tile's wall clock went, as ``(phase, duration, bar)`` triples.

    Triples rather than finished lines, so a caller picks its own column
    widths. Bars are scaled to the longest phase, which is the comparison worth
    making. A tile that spent 27 minutes in ``destriping`` and 4 in
    ``exporting`` should draw one long bar and one short one, not two short
    ones sized against an hour nobody spent.

    ``phase_seconds`` is cumulative in the published object, so this survives a
    watcher that attached late. It is the one part of a tile's history that
    does.
    """
    if not phase_seconds:
        return []
    longest = max(phase_seconds.values())
    rows = []
    for name in _phase_order(phase_seconds):
        seconds = phase_seconds[name]
        fraction = 0.0 if longest <= 0 else seconds / longest
        drawn = bar(fraction, width=width)
        marked = f"{drawn} {_CURRENT_MARKER}" if name == current else drawn
        rows.append((name, format_duration(seconds), marked))
    return rows


def provenance_tag(*labels: str | None) -> str:
    """A bracketed note rich prints rather than parses.

    ``provenance_tag("measured", "imds")`` gives ``\\[measured: imds]``. Rich
    reads a bare ``[measured]`` as a style tag and swallows it, so the opening
    bracket is escaped here rather than at every call site. Empty labels drop
    out, and no labels at all give an empty string rather than empty brackets.
    """
    kept = [str(label) for label in labels if label]
    if not kept:
        return ""
    return "\\[" + ": ".join(kept) + "]"


#: SGR and cursor escape sequences, which a captured log is full of.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    """Drop terminal escape sequences from captured output.

    A task log is a tee of a real terminal, so it carries the colour codes rich
    and structlog wrote. Replaying those into a second terminal renders a
    box-drawn traceback inside whatever frame is already there, and reading it
    back through a pipe shows the escapes as literal text. Either way the
    content is what matters, so the formatting goes.
    """
    return _ANSI_RE.sub("", text)


def truncate(text: str, chars: int) -> str:
    """``text`` cut to ``chars``, with an ellipsis standing in for the rest.

    Used on error text and log tails, where the full version is a key away in
    the bucket and the point of the cell is to say which failure it was.
    """
    if chars <= 0:
        return ""
    if len(text) <= chars:
        return text
    return text[: chars - 1] + "…"
