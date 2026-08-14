"""The key grammar of ``_runs/{run_id}/``, and how to read it back.

A tile publishes four kinds of object into its run prefix: a state object, a
task log, a dask profile dump, and, once it settles, a copy of its final state
at an unsuffixed key. Every attempt gets its own set, because
``settings.coiled_retries`` is 3 and every attempt used to write the same three
keys. Last write won, so a retry destroyed the attempt before it. That is why
run ``2021-2025-20260814T092642Z`` reported a 10-second failure against a
33-minute wall clock, and why the attempt that reached ``land_mask`` -- further
than that tile had ever gone -- is unrecoverable.

``watch``, ``reconcile``, and ``explain`` all have to read this layout, and
each of them used to re-derive it with its own string suffix tests. One of
those derivations was wrong: ``{tile}.{label}.profile.json`` also ends in
``.json``, so a profiled tile appeared in ``watch`` as a phantom tile literally
named ``N40W075.destripe_offsets``. Putting the grammar in one module fixes it
in every reader at once, and gives the rule a place to be tested without any
storage at all.

``batch`` imports ``watch`` already, so this module deliberately imports
neither. It depends on the standard library and on :mod:`landsat_lst.storage`
for typing only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import structlog

from landsat_lst.storage import RUN_RECORD_PREFIX

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: A basename stem ending in ``.{digits}``. The tile name itself never ends in
#: a dot-digit group, because tile names are ``N40W075``.
ATTEMPT_RE = re.compile(r"^(?P<tile>.+)\.(?P<attempt>\d+)$")

_PROFILE_SUFFIX = ".profile.json"
_LEGACY_PROGRESS_SUFFIX = ".progress.json"
_LOG_SUFFIX = ".log"
_STATE_SUFFIX = ".json"

#: The attempt number of an unsuffixed key. Two different objects carry it, and
#: both mean the same thing to a reader: the pointer a settled tile copies its
#: final state to, and the run record a tile written before this scheme left.
#: Both are written only when a tile stops, which is what makes the number a
#: usable "this tile settled" signal in either world.
POINTER_ATTEMPT = 0


def split_attempt(stem: str) -> tuple[str, int]:
    """Split a basename stem into its tile name and attempt number.

    ``"N40W075.2"`` gives ``("N40W075", 2)``. A stem with no attempt group
    gives :data:`POINTER_ATTEMPT`, which sorts below every real attempt, so a
    retried tile is read from its newest attempt and a run written before this
    scheme is still read at all.
    """
    match = ATTEMPT_RE.match(stem)
    if match is None:
        return stem, POINTER_ATTEMPT
    return match.group("tile"), int(match.group("attempt"))


@dataclass(frozen=True)
class TileArtifacts:
    """Everything one tile left behind in one run, grouped by attempt."""

    tile: str
    #: Attempt number to key. Attempt 0 is the pointer or the legacy record.
    states: dict[int, str] = field(default_factory=dict)
    logs: dict[int, str] = field(default_factory=dict)
    #: Attempt number to ``{label: key}``.
    profiles: dict[int, dict[str, str]] = field(default_factory=dict)
    #: A pre-#92 heartbeat, which lived at its own key rather than being the
    #: state object. Read-only; nothing writes one now.
    legacy_progress: str | None = None

    @property
    def attempt(self) -> int:
        """The newest attempt that left a state object, or 0 if none did."""
        return max((n for n in self.states if n > POINTER_ATTEMPT), default=POINTER_ATTEMPT)

    @property
    def state_key(self) -> str | None:
        """The newest attempt's state object, falling back to the pointer."""
        return self.states.get(self.attempt) or self.states.get(POINTER_ATTEMPT)

    @property
    def log_key(self) -> str | None:
        """The newest attempt's log, falling back to an unsuffixed one."""
        return self.logs.get(self.attempt) or self.logs.get(POINTER_ATTEMPT)

    @property
    def profile_keys(self) -> dict[str, str]:
        """The newest attempt's profile dumps, keyed by label."""
        return self.profiles.get(self.attempt) or self.profiles.get(POINTER_ATTEMPT) or {}

    @property
    def body_key(self) -> str | None:
        """The object to read for this tile's current state.

        A run written before the merge split live state and outcome across two
        objects, and the heartbeat is the one with a phase in it. A merged run
        has one object per attempt, and the newest is both.
        """
        return self.legacy_progress or self.state_key

    @property
    def settled(self) -> bool:
        """Whether this tile stopped, whatever its last beat managed to say.

        An unsuffixed state object means the same thing in both layouts. A
        merged tile copies its final state there when it settles, and a tile
        written before this scheme wrote its run record there once, at the end.
        Neither is written by a tile that is still running.
        """
        return POINTER_ATTEMPT in self.states

    @property
    def attempts(self) -> list[int]:
        """Every real attempt that left a state object, oldest first."""
        return sorted(n for n in self.states if n > POINTER_ATTEMPT)

    @property
    def highest_attempt(self) -> int:
        """The largest attempt number that left an artifact of any kind.

        Wider than :attr:`attempt` on purpose. A VM preempted before it
        published any state still leaves a log, and that log is the only
        evidence the attempt happened. Numbering the next attempt from the
        state objects alone would hand it the same key and overwrite exactly
        that evidence.
        """
        numbers = set(self.states) | set(self.logs) | set(self.profiles)
        return max((n for n in numbers if n > POINTER_ATTEMPT), default=POINTER_ATTEMPT)


def _blank(tile: str) -> TileArtifacts:
    return TileArtifacts(tile=tile)


def _add_profile(found: dict[str, TileArtifacts], stem: str, key: str) -> None:
    """File one ``{tile}.{attempt}.{label}.profile.json`` key.

    Tested before the plain ``.json`` case, which is the bug this module
    exists to close. A profile key read as a state object produced a tile named
    ``N40W075.destripe_offsets`` that ``watch`` rendered as a finished row and
    subtracted from its pending count.
    """
    body, _, label = stem.rpartition(".")
    if not body:
        return
    tile, attempt = split_attempt(body)
    entry = found.setdefault(tile, _blank(tile))
    entry.profiles.setdefault(attempt, {})[label] = key


def _classify_key(found: dict[str, TileArtifacts], key: str) -> None:
    """File one key under its tile, by suffix, most specific first."""
    name = key.rsplit("/", 1)[-1]

    if name.endswith(_PROFILE_SUFFIX):
        _add_profile(found, name[: -len(_PROFILE_SUFFIX)], key)
        return

    if name.endswith(_LEGACY_PROGRESS_SUFFIX):
        tile = name[: -len(_LEGACY_PROGRESS_SUFFIX)]
        entry = found.setdefault(tile, _blank(tile))
        # The dicts carry over by reference, which is what keeps keys filed
        # before this one. Only the scalar field needs the rebuild.
        found[tile] = replace(entry, legacy_progress=key)
        return

    if name.endswith(_LOG_SUFFIX):
        tile, attempt = split_attempt(name[: -len(_LOG_SUFFIX)])
        found.setdefault(tile, _blank(tile)).logs[attempt] = key
        return

    if name.endswith(_STATE_SUFFIX):
        tile, attempt = split_attempt(name[: -len(_STATE_SUFFIX)])
        found.setdefault(tile, _blank(tile)).states[attempt] = key


def classify(listing: Mapping[str, datetime]) -> dict[str, TileArtifacts]:
    """Group one run prefix's keys by tile and attempt.

    A pure function over the listing ``watch`` and ``reconcile`` each already
    perform, so "a profile key is not a tile" is a unit test with no storage in
    it. Keys that match none of the four shapes are ignored rather than
    guessed at.
    """
    found: dict[str, TileArtifacts] = {}
    for key in listing:
        _classify_key(found, key)
    return found


def tile_artifact_prefix(run_id: str, tile: str) -> str:
    """Every artifact one tile left in one run.

    The trailing dot is load-bearing. Without it the prefix would also select a
    longer tile name, and ``N40W07`` would collect ``N40W075``'s attempts.
    """
    return f"{RUN_RECORD_PREFIX}/{run_id}/{tile}."


def resolve_attempt(storage: StorageBackend, run_id: str, tile: str) -> int:
    """The number this attempt should key its artifacts under.

    One more than the highest attempt that already left an artifact. Coiled
    exposes no retry counter -- ``COILED_ARRAY_TASK_ID`` is the array index and
    is identical on every retry -- so the bucket is the only record of how many
    times this tile has been tried.

    Retries of one tile are sequential, so the listing is stable. Spot
    preemption can briefly overlap an outgoing process with its replacement,
    and both would then claim the same number and overwrite each other. That is
    exactly the behaviour this scheme replaces and no worse than it, and the
    ``host`` and ``pid`` fields in the published object make the collision
    identifiable.

    A listing that fails returns 1. A tile must never fail because it could not
    work out what to call its log.
    """
    try:
        listed = storage.list_prefix(tile_artifact_prefix(run_id, tile))
    except Exception as e:
        # Instrumentation never fails a tile. See the module docstring.
        log.warning("attempt_listing_failed", run_id=run_id, tile=tile, error=str(e))
        return 1

    artifacts = classify(listed).get(tile)
    if artifacts is None:
        return 1
    return artifacts.highest_attempt + 1
