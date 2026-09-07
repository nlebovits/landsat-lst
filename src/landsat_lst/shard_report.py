"""What a sharded run did, read back out of the bucket.

``landsat-lst reconcile`` answers this for a run submitted by ``batch.py``,
whose records live under ``_runs/``. A sharded run publishes under ``_shards/``
and cannot use it: ``runs.classify`` reads every key under a run prefix as a
tile attempt, and one shard root holds a plan, an item catalogue, seven shards
per stage, and a submission record per round. The postmortem of the three
S30W065 runs on 2026-09-04 was therefore done by hand, out of the state
objects, the Coiled task records, and the billing page.

Three sources, in decreasing order of authority:

- **The artifacts.** Which indexes are done is a listing, exactly as it is for
  the barrier that waited on them, and through the same ``_expected_keys``, so
  a report and a driver cannot disagree about what a stage owes.
- **The state objects.** ``{root}/state/{stage}.{index}.{attempt}.json`` is
  what each shard published about itself: its phase, its phase timings, its
  peak RSS, its instance type. A shard that was killed leaves the last
  heartbeat it managed, which is how ``exporting`` at 1,106 seconds became a
  readable fact rather than a guess.
- **Coiled's task records**, when credentials are present. They alone carry the
  exit code, and exit 137 at 17:36:38 is the difference between a shard that
  failed and a shard that was killed under a healthy run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from landsat_lst import shards
from landsat_lst.shard_driver import _expected_keys, _read_plan
from landsat_lst.storage import get_storage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: ``{stage}.{index:04d}.{attempt}.json``. The four-digit index is what keeps
#: this from matching a submission record, whose round is three digits and is
#: preceded by the literal word ``submission`` (:func:`shards.stage_submission_key`).
_ATTEMPT_RE = re.compile(
    r"^(?P<stage>[a-z_]+)\.(?P<index>\d{4})\.(?P<attempt>\d+)\.(?P<kind>json|log)$"
)

#: ``{stage}.submission.{round:03d}.json``.
_SUBMISSION_RE = re.compile(r"^(?P<stage>[a-z_]+)\.submission\.(?P<round>\d{3})\.json$")

#: The stages a plan can price an index set for. ``export`` is not one: it has
#: no per-index artifact, and its completion is the pair of COGs.
REPORTED_STAGES = ("climatology", "offsets", "composite")


@dataclass(frozen=True)
class ShardAttempt:
    """One shard's own account of one attempt, from its last heartbeat."""

    stage: str
    index: int
    attempt: int
    phase: str | None = None
    status: str | None = None
    elapsed_s: float | None = None
    peak_rss_mb: float | None = None
    instance_type: str | None = None
    phase_seconds: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    updated_at: datetime | None = None
    log_key: str | None = None
    exit_code: int | None = None
    task_state: str | None = None

    @property
    def settled(self) -> bool:
        """Whether the shard published a verdict. ``status`` is null while running."""
        return self.status is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "index": self.index,
            "attempt": self.attempt,
            "phase": self.phase,
            "status": self.status,
            "elapsed_s": self.elapsed_s,
            "peak_rss_mb": self.peak_rss_mb,
            "instance_type": self.instance_type,
            "phase_seconds": self.phase_seconds,
            "error": self.error,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "log_key": self.log_key,
            "exit_code": self.exit_code,
            "task_state": self.task_state,
        }


@dataclass(frozen=True)
class StageReport:
    """One stage of one tile: what it owed, what it has, and what it said."""

    stage: str
    done: list[int] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)
    attempts: dict[int, list[ShardAttempt]] = field(default_factory=dict)
    rounds: int = 0

    @property
    def expected(self) -> int:
        return len(self.done) + len(self.missing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "expected": self.expected,
            "done": self.done,
            "missing": self.missing,
            "rounds": self.rounds,
            "attempts": {
                str(index): [a.to_dict() for a in attempts]
                for index, attempts in sorted(self.attempts.items())
            },
        }


@dataclass(frozen=True)
class TileReport:
    """One tile's stages, and whether its COGs exist."""

    tile: str
    window: str = ""
    cogs_present: bool = False
    plan_present: bool = False
    stages: list[StageReport] = field(default_factory=list)
    staged_objects: int = 0

    @property
    def missing(self) -> int:
        """Shard artifacts this run still owes, across every stage."""
        return sum(len(stage.missing) for stage in self.stages)

    @property
    def completed(self) -> bool:
        """Whether *this run* finished the tile.

        Both halves are load-bearing. The COGs live at a canonical key that is
        a function of window and tile alone, so a tile shipped by an earlier
        run puts them there; asking only that question told the S30W065
        postmortem the killed run was complete. And a run whose shards all
        published has still not delivered a tile until the export lands.
        """
        return self.cogs_present and self.missing == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile": self.tile,
            "window": self.window,
            "completed": self.completed,
            "cogs_present": self.cogs_present,
            "missing": self.missing,
            "plan_present": self.plan_present,
            "staged_objects": self.staged_objects,
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class RunReport:
    run_id: str
    tiles: list[TileReport] = field(default_factory=list)
    coiled_read: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "coiled_read": self.coiled_read,
            "tiles": [tile.to_dict() for tile in self.tiles],
        }


def _state_bodies(
    storage: StorageBackend, listing: Mapping[str, datetime], root: str
) -> tuple[dict[tuple[str, int, int], dict], dict[tuple[str, int, int], str], list[dict]]:
    """Split one tile's ``state/`` keys into attempts, logs, and submissions."""
    prefix = f"{root}/state/"
    attempts: dict[tuple[str, int, int], dict] = {}
    logs: dict[tuple[str, int, int], str] = {}
    submissions: list[dict] = []
    for key in sorted(listing):
        if not key.startswith(prefix):
            continue
        name = key[len(prefix) :]
        submission = _SUBMISSION_RE.match(name)
        if submission is not None:
            raw = storage.read_text(key)
            if raw is None:
                continue
            try:
                submissions.append(json.loads(raw))
            except ValueError:
                log.warning("shard_report_bad_submission", key=key)
            continue
        match = _ATTEMPT_RE.match(name)
        if match is None:
            continue
        ident = (match["stage"], int(match["index"]), int(match["attempt"]))
        if match["kind"] == "log":
            logs[ident] = key
            continue
        raw = storage.read_text(key)
        if raw is None:
            continue
        try:
            body = json.loads(raw)
        except ValueError:
            log.warning("shard_report_bad_state", key=key)
            continue
        body["_updated_at"] = listing.get(key)
        attempts[ident] = body
    return attempts, logs, submissions


def _attempt(
    ident: tuple[str, int, int],
    body: dict,
    log_key: str | None,
    task: Mapping[str, Any] | None,
) -> ShardAttempt:
    stage, index, attempt = ident
    return ShardAttempt(
        stage=stage,
        index=index,
        attempt=attempt,
        phase=body.get("phase"),
        status=body.get("status"),
        elapsed_s=body.get("elapsed_s"),
        peak_rss_mb=body.get("peak_rss_mb"),
        instance_type=body.get("instance_type"),
        phase_seconds=dict(body.get("phase_seconds") or {}),
        error=body.get("error"),
        updated_at=body.get("_updated_at"),
        log_key=log_key,
        exit_code=None if task is None else task.get("exit_code"),
        task_state=None if task is None else task.get("state"),
    )


def coiled_task_states(submissions: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict]:
    """Per ``(stage, index)`` Coiled task records, keyed through the submission.

    Coiled numbers a task by its position in ``map_over_values``, and the
    submission record persists exactly that list, so the mapping is a lookup
    rather than a guess.

    Coiled being unreachable is not fatal to a report: the artifacts and the
    state objects still describe the run. Same contract as
    ``batch._task_states``, one level down.
    """
    states: dict[tuple[str, int], dict] = {}
    for record in submissions:
        cluster_id = record.get("cluster_id")
        indexes = record.get("indexes") or []
        stage = str(record.get("stage", ""))
        if cluster_id is None or not indexes:
            continue
        try:
            import coiled.batch  # noqa: PLC0415

            jobs = coiled.batch.status(cluster_id)
        except Exception as e:
            log.warning("batch_status_unavailable", cluster_id=cluster_id, error=str(e))
            continue
        for job in jobs:
            for task in job.get("tasks", []):
                position = task.get("array_task_id")
                if position is None or position >= len(indexes):
                    continue
                states[stage, int(indexes[position])] = dict(task)
    return states


def _stage_report(
    stage: str,
    *,
    root: str,
    plan: Any,
    listing: Mapping[str, datetime],
    attempts: Mapping[tuple[str, int, int], dict],
    logs: Mapping[tuple[str, int, int], str],
    states: Mapping[tuple[str, int], dict],
    rounds: int,
) -> StageReport | None:
    """One stage of one tile, or ``None`` when the run holds nothing for it."""
    expected = _expected_keys(plan, stage, root) if plan is not None else {}
    done: list[int] = []
    missing: list[int] = []
    for index, keys in sorted(expected.items()):
        (done if all(key in listing for key in keys) else missing).append(index)

    per_index: dict[int, list[ShardAttempt]] = {}
    for ident, body in sorted(attempts.items()):
        if ident[0] != stage:
            continue
        per_index.setdefault(ident[1], []).append(
            _attempt(ident, body, logs.get(ident), states.get((stage, ident[1])))
        )

    if not expected and not per_index:
        return None
    return StageReport(stage=stage, done=done, missing=missing, attempts=per_index, rounds=rounds)


def _rounds(submissions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """The highest submission round each stage reached, across every driver."""
    highest: dict[str, int] = {}
    for record in submissions:
        stage = str(record.get("stage", ""))
        highest[stage] = max(highest.get(stage, 0), int(record.get("round", 0)))
    return highest


def _tile_report(
    run_id: str,
    tile: str,
    *,
    storage: StorageBackend,
    listing: Mapping[str, datetime],
    task_states: Mapping[tuple[str, int], dict] | None,
) -> TileReport:
    root = shards.shard_root(run_id, tile)
    attempts, logs, submissions = _state_bodies(storage, listing, root)
    states = coiled_task_states(submissions) if task_states is None else dict(task_states)
    plan = _read_plan(run_id, tile, root, storage)
    rounds = _rounds(submissions)

    stage_reports = [
        report
        for stage in REPORTED_STAGES
        if (
            report := _stage_report(
                stage,
                root=root,
                plan=plan,
                listing=listing,
                attempts=attempts,
                logs=logs,
                states=states,
                rounds=rounds.get(stage, 0),
            )
        )
        is not None
    ]

    window = plan.window if plan is not None else ""
    stage_prefix = shards.tile_stage_prefix(root)
    return TileReport(
        tile=tile,
        window=window,
        cogs_present=bool(window) and storage.cog_exists(window, tile),
        plan_present=plan is not None,
        stages=stage_reports,
        staged_objects=sum(1 for key in listing if key.startswith(stage_prefix)),
    )


def reconcile_shard_run(
    run_id: str,
    *,
    storage: StorageBackend | None = None,
    tile: str | None = None,
    task_states: Mapping[tuple[str, int], dict] | None = None,
) -> RunReport:
    """Read one sharded run back out of the bucket.

    Safe at any point, including while shards are still running: an index with
    no artifact is reported missing, and ``status`` is null for a shard that
    has not settled. Safe to call more than once; it writes nothing.

    Args:
        run_id: The run to read.
        storage: Where to read it from. Defaults to the configured backend.
        tile: One tile, or every tile the run published under.
        task_states: Injected Coiled records. ``None`` reads them, when
            credentials allow; an empty mapping asks for none.
    """
    storage = storage or get_storage()
    listing = storage.list_prefix(f"{shards.SHARD_PREFIX}/{run_id}/")
    tiles = [tile] if tile else shards.run_tiles(listing, run_id)
    return RunReport(
        run_id=run_id,
        tiles=[
            _tile_report(run_id, name, storage=storage, listing=listing, task_states=task_states)
            for name in tiles
        ],
        coiled_read=task_states is None,
    )
