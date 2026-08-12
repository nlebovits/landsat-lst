"""Distributed tile processing on Coiled Batch.

Each tile runs as a plain process on its own VM. There is no shared dask
scheduler, no heartbeat loop competing with a multi-hour local compute, and no
long-lived client connection from the submitting shell. The workload is a set
of independent, idempotent, S3-writing jobs, which is the shape Batch is built
for. See ADR-010 for the three Coiled Functions failures that motivated the
move.

The run is split into two phases that never share a process:

1. :func:`submit_batch` filters out already-finished tiles with one storage
   listing per window, submits the task array, records the submission under
   ``settings.manifest_dir``, and returns. Killing the shell after this point
   has no effect on the run.
2. :func:`reconcile_run` reads that submission back, takes tile completion from
   the COG listing, enriches it with the per-tile run records the VMs wrote,
   and produces the run manifest.

Completion is decided by the COG listing, not by a task exit code. A task can
exit non-zero after its assets landed (a failed record write, a preempted VM
during teardown), and a task can exit zero having produced nothing if the
pipeline is ever changed to swallow an error. The bytes in the bucket are the
only claim worth trusting.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from landsat_lst.config import settings
from landsat_lst.job import JobResult, _split_completed, _worker_environ
from landsat_lst.storage import get_storage

if TYPE_CHECKING:
    from pathlib import Path

    from landsat_lst.models import ProcessingJob
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: Coiled sets this per task from the values passed to ``map_over_values``.
TASK_INPUT_VAR = "COILED_BATCH_TASK_INPUT"


@dataclass
class BatchSubmission:
    """What one ``submit_batch`` call handed to Coiled, saved to disk.

    Reconciliation runs in a different process, possibly on a different day, so
    everything it needs to interpret the run is written here at submit time:
    the cluster to query, the tiles that were submitted, and the tiles that
    were already finished and never cost anything.
    """

    run_id: str
    window: str
    cluster_id: int | None
    job_id: int | None
    submitted_at: str
    submitted_tiles: list[str]
    #: First year of the window. Required: every tile in the run is rebuilt
    #: from it during reconciliation, when the jobs themselves are long gone.
    year: int
    end_year: int | None = None
    skipped_tiles: list[str] = field(default_factory=list)
    command: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "window": self.window,
            "cluster_id": self.cluster_id,
            "job_id": self.job_id,
            "submitted_at": self.submitted_at,
            "submitted_tiles": self.submitted_tiles,
            "skipped_tiles": self.skipped_tiles,
            "year": self.year,
            "end_year": self.end_year,
            "command": self.command,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> BatchSubmission:
        return cls(**payload)

    @property
    def dashboard_url(self) -> str:
        return f"https://cloud.coiled.io/clusters/{self.cluster_id}"


def submission_path(run_id: str, out_dir: Path | None = None) -> Path:
    """Where one run's submission record lives."""
    return (out_dir or settings.manifest_dir) / f"{run_id}.submission.json"


def load_submission(run_id: str, out_dir: Path | None = None) -> BatchSubmission:
    """Read back the submission record for ``run_id``.

    Raises:
        FileNotFoundError: If the run was never submitted from this machine.
    """
    path = submission_path(run_id, out_dir)
    if not path.is_file():
        msg = f"No submission record for run {run_id!r} at {path}"
        raise FileNotFoundError(msg)
    return BatchSubmission.from_dict(json.loads(path.read_text()))


def _task_command(*, run_id: str, year: int, end_year: int | None, force: bool) -> str:
    """The shell command one VM runs for one tile.

    The window is baked in as a literal and only the tile varies, which is why
    jobs are grouped by window before submission. ``python -m`` rather than the
    ``landsat-lst`` console script: both need the package importable on the VM,
    but the module path does not additionally depend on the entry point being
    installed on ``PATH`` by package sync.
    """
    parts = [
        "python",
        "-m",
        "landsat_lst.cli",
        "process",
        "--run-id",
        run_id,
        "--year",
        str(year),
    ]
    if end_year is not None:
        parts += ["--end-year", str(end_year)]
    if force:
        parts.append("--force")
    quoted = shlex.join(parts)
    # Expanded by bash on the VM, not by the submitting shell.
    return f'{quoted} --tile "${TASK_INPUT_VAR}"'


def _batch_run_kwargs(
    *, run_id: str, tiles: list[str], command: str, environ: dict[str, str]
) -> dict[str, Any]:
    """Every Coiled Batch knob this project pins, in one inspectable dict.

    Nothing here is left to a Coiled default. An unpinned region reads Landsat
    across regions and pays egress on every scene; an unpinned worker ceiling
    would let a 700-tile submission start 700 machines.
    """
    return {
        "command": ["bash", "-c", command],
        "name": f"lst-{run_id}",
        "region": settings.coiled_region,
        "vm_type": settings.coiled_vm_types,
        "spot_policy": settings.coiled_spot_policy,
        "max_workers": settings.coiled_max_workers,
        "max_retries": settings.coiled_retries,
        "job_timeout": settings.coiled_job_timeout,
        "map_over_values": tiles,
        "env": environ,
        "tag": {"project": "landsat-lst", "run_id": run_id},
        # Credentials are forwarded explicitly through env as frozen SSO
        # values; letting Coiled forward its own would race with those.
        "forward_aws_credentials": False,
    }


def submit_batch(
    jobs: list[ProcessingJob],
    *,
    force: bool = False,
    run_id: str | None = None,
    storage: StorageBackend | None = None,
) -> BatchSubmission:
    """Submit tiles to Coiled Batch and return without waiting.

    Already-completed tiles are filtered out with one storage listing per
    window before anything is submitted, so a resumed run pays for exactly the
    tiles that are missing and a fully-finished window never starts a cluster.

    Args:
        jobs: Processing jobs to run. All must share one window; group by
            ``window_label`` and call once per group otherwise.
        force: If True, skip the resume listing and reprocess every tile.
        run_id: Run token; generated from the window and a UTC timestamp when
            omitted.
        storage: Storage backend used for the resume listing (default from
            :func:`landsat_lst.storage.get_storage`).

    Returns:
        The :class:`BatchSubmission`, also written to
        ``settings.manifest_dir / f"{run_id}.submission.json"``. A submission
        with no ``cluster_id`` means every tile was already finished.

    Raises:
        ImportError: If Coiled is not installed.
        ValueError: If ``jobs`` is empty or spans more than one window.
        RuntimeError: If no AWS credentials can be resolved for the VMs.
    """
    try:
        import coiled  # noqa: PLC0415
    except ImportError as e:
        msg = "Coiled is required for distributed execution. Install with: pip install coiled"
        raise ImportError(msg) from e

    if not jobs:
        msg = "No jobs to submit"
        raise ValueError(msg)

    windows = sorted({job.window_label for job in jobs})
    if len(windows) > 1:
        msg = (
            f"submit_batch takes one window per call, got {windows}. "
            "Group jobs by window_label and submit each group."
        )
        raise ValueError(msg)

    window = windows[0]
    started_at = datetime.now(tz=UTC)
    run_id = run_id or f"{window}-{started_at:%Y%m%dT%H%M%SZ}"
    storage = storage or get_storage()

    to_run, skipped = (jobs, []) if force else _split_completed(jobs, storage)
    tiles = [job.tile.name for job in to_run]

    command = _task_command(
        run_id=run_id,
        year=jobs[0].year,
        end_year=jobs[0].end_year,
        force=force,
    )

    log.info(
        "batch_submit_start",
        run_id=run_id,
        window=window,
        job_count=len(jobs),
        to_run=len(tiles),
        already_completed=len(skipped),
        force=force,
        region=settings.coiled_region,
        vm_types=settings.coiled_vm_types,
        max_workers=settings.coiled_max_workers,
    )

    submission = BatchSubmission(
        run_id=run_id,
        window=window,
        cluster_id=None,
        job_id=None,
        submitted_at=started_at.isoformat(),
        submitted_tiles=tiles,
        skipped_tiles=[r.job.tile.name for r in skipped],
        year=jobs[0].year,
        end_year=jobs[0].end_year,
        command=command,
    )

    if tiles:
        # _worker_environ() resolves SSO credentials, so it runs only when a
        # cluster is actually being started.
        result = coiled.batch_run(
            **_batch_run_kwargs(
                run_id=run_id,
                tiles=tiles,
                command=command,
                environ=_worker_environ(),
            )
        )
        submission.cluster_id = result.get("cluster_id")
        submission.job_id = result.get("job_id")
        log.info(
            "batch_submitted",
            run_id=run_id,
            cluster_id=submission.cluster_id,
            job_id=submission.job_id,
            tasks=len(tiles),
        )
    else:
        log.info("batch_submit_skipped", run_id=run_id, reason="all_tiles_complete")

    path = submission_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(submission.to_dict(), indent=2))
    return submission


def _task_states(submission: BatchSubmission) -> dict[str, dict]:
    """Per-tile Coiled task state, keyed by tile name.

    Task IDs are assigned in the order of the values handed to
    ``map_over_values``, which is why the submitted tile order is persisted.
    Coiled being unreachable is not fatal to reconciliation: the S3 listing and
    the run records still describe the run, so the failure is logged and an
    empty mapping returned.
    """
    if submission.cluster_id is None:
        return {}

    try:
        import coiled.batch  # noqa: PLC0415

        jobs = coiled.batch.status(submission.cluster_id)
    except Exception as e:
        log.warning("batch_status_unavailable", run_id=submission.run_id, error=str(e))
        return {}

    states: dict[str, dict] = {}
    for job in jobs:
        for task in job.get("tasks", []):
            index = task.get("array_task_id")
            if index is None or index >= len(submission.submitted_tiles):
                continue
            states[submission.submitted_tiles[index]] = task
    return states


def _task_duration_s(task: dict) -> float | None:
    """Wall-clock seconds Coiled billed for one task, if it started and stopped."""
    start, stop = task.get("start"), task.get("stop")
    if not start or not stop:
        return None
    return (datetime.fromisoformat(stop) - datetime.fromisoformat(start)).total_seconds()


def _failure_reason(task: dict | None) -> str:
    """Why a tile without both COGs has no output.

    Coiled's task state is all that is left when a VM died before its own
    record could be written, which is exactly the case a manifest most needs to
    explain.
    """
    if task is None:
        return "no task state and no run record; tile may never have been scheduled"
    exit_code = task.get("exit_code")
    state = task.get("state", "unknown")
    if exit_code:
        return f"task exited {exit_code} (state {state})"
    return f"task state {state} with no COGs written"


def _resolve_tile(
    tile: str,
    submission: BatchSubmission,
    *,
    completed: set[str],
    records: dict[str, JobResult],
    tasks: dict[str, dict],
    storage: StorageBackend,
) -> JobResult:
    """One tile's outcome, from the COG listing first and the record second."""
    from landsat_lst.models import ProcessingJob  # noqa: PLC0415
    from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415

    record = records.get(tile)
    task = tasks.get(tile)
    job = (
        record.job
        if record
        else ProcessingJob(
            tile=parse_tile_name(tile),
            year=submission.year,
            end_year=submission.end_year,
        )
    )

    if tile in completed:
        # A tile whose VM never reported still has both assets at known keys,
        # and the manifest is the catalog's shopping list, so derive them.
        return JobResult(
            job=job,
            status="completed",
            lst_key=(record.lst_key if record else None)
            or storage.cog_key(submission.window, tile, "lst_p95"),
            qa_key=(record.qa_key if record else None)
            or storage.cog_key(submission.window, tile, "qa_count"),
            duration_s=(record.duration_s if record else None) or _task_duration_s(task or {}),
            scene_count=record.scene_count if record else None,
            peak_rss_mb=record.peak_rss_mb if record else None,
        )

    return JobResult(
        job=job,
        status="failed",
        error=(record.error if record else None) or _failure_reason(task),
        duration_s=(record.duration_s if record else None) or _task_duration_s(task or {}),
        scene_count=record.scene_count if record else None,
        peak_rss_mb=record.peak_rss_mb if record else None,
    )


def _read_records(submission: BatchSubmission, storage: StorageBackend) -> dict[str, JobResult]:
    """Per-tile run records the VMs wrote, keyed by tile name.

    A tile with no record is normal, not an error: its VM was preempted, timed
    out, or was killed before it could report.
    """
    records: dict[str, JobResult] = {}
    for tile in submission.submitted_tiles:
        raw = storage.read_text(storage.run_record_key(submission.run_id, tile))
        if raw is None:
            continue
        try:
            records[tile] = JobResult.from_record(json.loads(raw))
        except (ValueError, KeyError) as e:
            log.warning("run_record_unreadable", tile=tile, error=str(e))
    return records


def reconcile_run(
    run_id: str,
    *,
    storage: StorageBackend | None = None,
    out_dir: Path | None = None,
) -> list[JobResult]:
    """Build the run manifest for a finished (or abandoned) batch run.

    Safe to call at any point, including while tasks are still running: tiles
    without COGs yet are reported as failed, and a later call supersedes the
    manifest. Safe to call more than once.

    Args:
        run_id: Run token from :func:`submit_batch`.
        storage: Storage backend to read completion and records from.
        out_dir: Directory holding the submission record and receiving the
            manifest (default ``settings.manifest_dir``).

    Returns:
        One :class:`~landsat_lst.job.JobResult` per tile in the run, including
        the tiles that were skipped as already complete.

    Raises:
        FileNotFoundError: If no submission record exists for ``run_id``.
    """
    from landsat_lst.manifest import write_run_manifest  # noqa: PLC0415
    from landsat_lst.models import ProcessingJob  # noqa: PLC0415
    from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415

    submission = load_submission(run_id, out_dir)
    storage = storage or get_storage()

    completed = storage.list_completed(submission.window)
    records = _read_records(submission, storage)
    tasks = _task_states(submission)

    results = [
        _resolve_tile(
            tile,
            submission,
            completed=completed,
            records=records,
            tasks=tasks,
            storage=storage,
        )
        for tile in submission.submitted_tiles
    ]
    results += [
        JobResult(
            job=ProcessingJob(
                tile=parse_tile_name(tile),
                year=submission.year,
                end_year=submission.end_year,
            ),
            status="skipped",
        )
        for tile in submission.skipped_tiles
    ]

    manifest_path = write_run_manifest(
        results,
        run_id=run_id,
        window=submission.window,
        started_at=datetime.fromisoformat(submission.submitted_at),
        retries=settings.coiled_retries,
        cluster_id=submission.cluster_id,
        job_id=submission.job_id,
        out_dir=out_dir,
    )

    log.info(
        "batch_reconciled",
        run_id=run_id,
        completed=sum(1 for r in results if r.status == "completed"),
        skipped=sum(1 for r in results if r.status == "skipped"),
        failed=sum(1 for r in results if r.status == "failed"),
        manifest=str(manifest_path),
    )
    return results


def wait_for_batch(run_id: str, *, timeout_s: int | None = None) -> str | None:
    """Block until the run's Coiled job reaches a done state.

    Optional convenience for a supervised validation run. Nothing depends on
    it: the run survives this process ending, and reconciliation reads durable
    state.

    Args:
        run_id: Run token from :func:`submit_batch`.
        timeout_s: Give up waiting after this many seconds.

    Returns:
        The final job state, or ``None`` on timeout or when there was no job to
        wait for.
    """
    from coiled.batch import wait_for_job_done  # noqa: PLC0415

    submission = load_submission(run_id)
    if submission.job_id is None:
        return None
    return wait_for_job_done(submission.job_id, timeout=timeout_s)
