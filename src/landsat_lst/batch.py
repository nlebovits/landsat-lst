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
   the COG listing, enriches it with everything the VMs published, and produces
   the run manifest.

The manifest answers three questions the Coiled dashboard never held. Which
attempts a tile made, read from :mod:`landsat_lst.runs` and including the ones
whose VMs died without settling. What the run cost, priced by
:mod:`landsat_lst.pricing` from billed seconds and the instance each tile
reported. And how the memory floor stored at submit time compares to the peak
the run reached, which is the only way a configuration argument survives the
run that should settle it.

While the run is going, the tiles themselves are the only source of progress:
each publishes a heartbeat every minute under the same ``_runs/{run_id}/``
prefix, and :mod:`landsat_lst.watch` renders them. The cluster dashboard cannot
help, because a batch task never registers with the dask scheduler its panels
describe.

Completion is decided by the COG listing, not by a task exit code. A task can
exit non-zero after its assets landed (a failed record write, a preempted VM
during teardown), and a task can exit zero having produced nothing if the
pipeline is ever changed to swallow an error. The bytes in the bucket are the
only claim worth trusting.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from landsat_lst import pricing, runs
from landsat_lst.config import settings
from landsat_lst.job import JobResult, _split_completed, _worker_environ
from landsat_lst.storage import S3Storage, get_storage

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from landsat_lst.models import ProcessingJob
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: MiB in a GiB. ``peak_rss_mb`` is MiB and a planned floor is GiB.
MIB_PER_GIB = 1024.0

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
    #: Where the run's tiles wrote. A distributed run always writes to S3,
    #: whatever the submitting shell had configured, so watch and reconcile
    #: read this rather than the local default -- otherwise they search an
    #: empty output directory and report a live run as pending.
    storage_backend: str = "s3"
    s3_bucket: str = ""
    s3_prefix: str = ""
    #: The memory floor this run was submitted expecting, from
    #: :func:`landsat_lst.profiling.plan_memory_record`. Reconciliation reads it
    #: back to put the planned figure next to the observed one. ``None`` for a
    #: run submitted before plans were stored, or one whose plan failed to
    #: build.
    plan: dict | None = None

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
            "storage_backend": self.storage_backend,
            "s3_bucket": self.s3_bucket,
            "s3_prefix": self.s3_prefix,
            "plan": self.plan,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> BatchSubmission:
        # Tolerant of records written before the storage fields existed; those
        # runs predate this and fall back to the configured backend.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def storage(self) -> StorageBackend:
        """The backend this run's tiles actually wrote to."""
        if self.storage_backend != "s3":
            return get_storage()
        return S3Storage(bucket=self.s3_bucket or None, prefix=self.s3_prefix or None)

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


def _task_command(
    *,
    run_id: str,
    year: int,
    end_year: int | None,
    force: bool,
    max_scenes: int | None = None,
    use_offset_cache: bool = True,
) -> str:
    """The shell script one VM runs for one tile.

    The window is baked in as a literal and only the tile varies, which is why
    jobs are grouped by window before submission. ``python -m`` rather than the
    ``landsat-lst`` console script: both need the package importable on the VM,
    but the module path does not additionally depend on the entry point being
    installed on ``PATH`` by package sync.

    A ``#!`` script rather than a command string. Coiled splits a plain string
    on whitespace and rejoins a list, and either round trip mangles the quotes
    around the task-input variable; a script is shipped to the VM verbatim.
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
    # Every field the job carries has to be restated here: the VM builds its
    # own job from these arguments, so anything omitted silently reverts to a
    # default. A missing --max-scenes turned a 300-scene sample into a full
    # 2,930-scene run that looked like a sample from the submitting side.
    if max_scenes is not None:
        parts += ["--max-scenes", str(max_scenes)]
    if not use_offset_cache:
        parts.append("--no-offset-cache")
    quoted = shlex.join(parts)
    # Expanded by bash on the VM, not by the submitting shell.
    return f'#!/bin/bash\n{quoted} --tile "${TASK_INPUT_VAR}"\n'


def _batch_run_kwargs(
    *, run_id: str, tiles: list[str], command: str, environ: dict[str, str]
) -> dict[str, Any]:
    """Every Coiled Batch knob this project pins, in one inspectable dict.

    Nothing here is left to a Coiled default. An unpinned region reads Landsat
    across regions and pays egress on every scene; an unpinned worker ceiling
    would let a 700-tile submission start 700 machines.
    """
    return {
        # A script, not ["bash", "-c", command]. Coiled joins a list back into
        # one shell string, and the inner quotes around the task-input variable
        # came out of that round trip as literal characters: the CLI received
        # --tile "N40W075" with the quote marks, parse_tile_name rejected it,
        # and the task died in 0.6s having written nothing. A command that
        # starts with "#!" is passed through verbatim instead. See issue #66.
        "command": command,
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


def _submission_plan(job: ProcessingJob) -> dict[str, Any] | None:
    """The memory floor this run is being submitted expecting.

    One record for the whole run, built from the first job. Every five degree
    land tile has the same geometry, so only the assumed scene count varies
    between them, and the arithmetic builds no graph: 1.24 ms at submit time,
    which is why there is no flag to turn it off.

    ``threads`` is pinned from configuration rather than left to default,
    because the default reads the CPU count of whatever machine called
    :func:`submit_batch`. That machine is a laptop and the run is not on it, so
    a defaulted figure would describe a VM the run never used. When the setting
    is unset the record says where the number came from, so a later comparison
    can discount it rather than trust it.

    Returns:
        The record, or ``None`` if it could not be built. A planning number is
        not worth failing a submission over, and reconciliation already handles
        a submission that stored no plan.
    """
    from landsat_lst.profiling import PRODUCTION_SCENES, plan_memory_record  # noqa: PLC0415

    threads = settings.dask_max_threads
    try:
        record = plan_memory_record(
            tile=job.tile,
            scenes=job.max_scenes or PRODUCTION_SCENES,
            threads=threads,
        )
    except Exception as e:
        log.warning("submission_plan_failed", tile=job.tile.name, error=str(e))
        return None

    record["threads_source"] = (
        "settings.dask_max_threads" if threads else "cpu_count of the submitting host"
    )
    return record


def submit_batch(
    jobs: list[ProcessingJob],
    *,
    force: bool = False,
    run_id: str | None = None,
    storage: StorageBackend | None = None,
    use_offset_cache: bool = True,
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
        use_offset_cache: Forwarded to every task as ``--no-offset-cache`` when
            False. On by default, which is what makes a preempted tile's retry
            skip the offset pass its first attempt already paid for.

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
        max_scenes=jobs[0].max_scenes,
        use_offset_cache=use_offset_cache,
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
        # Recorded from config rather than from the live backend: the tasks
        # always run with LST_STORAGE_BACKEND=s3, whatever this shell is set to.
        s3_bucket=settings.s3_bucket,
        s3_prefix=settings.s3_prefix,
        plan=_submission_plan(jobs[0]),
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


def _failure_reason(task: dict | None, log_key: str | None = None) -> str:
    """Why a tile without both COGs has no output.

    Coiled's task state is all that is left when a VM died before its own
    record could be written, which is exactly the case a manifest most needs to
    explain. Coiled reports the tee wrapper's exit code rather than the CLI's,
    so the uploaded task log is named whenever the tile got far enough to leave
    one: that is where the traceback is.
    """
    if task is None:
        reason = "no task state and no run record; tile may never have been scheduled"
    elif task.get("exit_code"):
        reason = f"task exited {task['exit_code']} (state {task.get('state', 'unknown')})"
    else:
        reason = f"task state {task.get('state', 'unknown')} with no COGs written"
    return f"{reason}; task log at {log_key}" if log_key else reason


@dataclass(frozen=True)
class _TileEvidence:
    """Everything one tile published in one run, before a verdict is taken.

    Held together rather than passed as five parallel dicts, because every one
    of them is keyed by tile name and read at the same moment.
    """

    tile: str
    #: The newest attempt's state, or the unsuffixed object for a run written
    #: before attempts were numbered. ``None`` when the VM published nothing.
    record: JobResult | None = None
    #: Coiled's own task state, which is all that is left when a VM died before
    #: it could publish.
    task: dict | None = None
    log_key: str | None = None
    #: One summary row per numbered attempt, oldest first.
    attempts: list[dict[str, Any]] = field(default_factory=list)
    instance_type: str | None = None
    lifecycle: str | None = None


def _read_body(storage: StorageBackend, key: str) -> dict | None:
    """One published state object, or ``None`` if it cannot be read as JSON."""
    raw = storage.read_text(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError as e:
        log.warning("run_record_unreadable", key=key, error=str(e))
        return None


def _as_result(body: dict | None) -> JobResult | None:
    """A published state object as a job result, or ``None`` if it is not one."""
    if body is None:
        return None
    try:
        return JobResult.from_record(body)
    except (ValueError, KeyError) as e:
        log.warning("run_record_unusable", tile=body.get("tile"), error=str(e))
        return None


def _attempt_row(attempt: int, body: dict, log_key: str | None) -> dict[str, Any]:
    """One line of a tile's attempt series.

    The attempt number comes from the key rather than the body, because the key
    is what kept the two attempts apart in the first place.

    ``duration_s`` is written when a tile settles, so an attempt whose VM was
    preempted mid-phase reports ``elapsed_s`` from its last beat instead. That
    beat is the only evidence the attempt happened, and the phase it reached is
    the reason to keep it: the run that motivated this scheme lost the attempt
    that got furthest.
    """
    return {
        "attempt": attempt,
        "phase": body.get("phase"),
        "status": body.get("status"),
        "duration_s": body.get("duration_s") or body.get("elapsed_s"),
        "peak_rss_mb": body.get("peak_rss_mb"),
        "error": body.get("error"),
        "log_key": log_key,
    }


def _tile_evidence(
    tile: str,
    artifacts: runs.TileArtifacts | None,
    task: dict | None,
    storage: StorageBackend,
) -> _TileEvidence:
    """Read back what one tile published, newest attempt last.

    Every numbered attempt is read, not only the winning one. A tile that
    succeeded on its third try has two earlier attempts that explain what the
    run cost, and a healthy tile has exactly one, so the series costs a retried
    tile one read per retry and costs everything else nothing.

    The unsuffixed key is read only when there is no numbered attempt, which
    means a run written before attempts were numbered. Reading it as well would
    spend a request per tile on a copy of the newest attempt's body.
    """
    if artifacts is None:
        return _TileEvidence(tile=tile, task=task)

    bodies = {
        n: body
        for n in artifacts.attempts
        if (body := _read_body(storage, artifacts.states[n])) is not None
    }
    rows = [_attempt_row(n, body, artifacts.logs.get(n)) for n, body in sorted(bodies.items())]
    winner = bodies[max(bodies)] if bodies else _pointer_body(artifacts, storage)

    return _TileEvidence(
        tile=tile,
        record=_as_result(winner),
        task=task,
        log_key=artifacts.log_key,
        attempts=rows,
        instance_type=(winner or {}).get("instance_type"),
        lifecycle=(winner or {}).get("instance_lifecycle"),
    )


def _pointer_body(artifacts: runs.TileArtifacts, storage: StorageBackend) -> dict | None:
    """The unsuffixed object, which is where a pre-attempt run left everything."""
    key = artifacts.state_key
    return _read_body(storage, key) if key else None


def _read_evidence(
    submission: BatchSubmission,
    storage: StorageBackend,
    listing: Mapping[str, datetime],
) -> list[_TileEvidence]:
    """One evidence record per submitted tile, in submission order.

    A tile that published nothing is normal rather than an error: its VM was
    preempted, timed out, or was killed before it could report. The run prefix
    listing already says which tiles left artifacts, so a 700-tile run reads
    only the objects that exist instead of spending a request per tile finding
    out that they do not.
    """
    artifacts = runs.classify(listing)
    tasks = _task_states(submission)
    return [
        _tile_evidence(tile, artifacts.get(tile), tasks.get(tile), storage)
        for tile in submission.submitted_tiles
    ]


def _observed_metrics(evidence: _TileEvidence) -> dict[str, Any]:
    """Duration, scene count, and peak memory, published state first.

    Coiled's start and stop timestamps are the only duration left for a VM that
    died before publishing, and they cover the whole task rather than the
    pipeline inside it, so they are the fallback and never the first choice.
    """
    task_duration = _task_duration_s(evidence.task or {})
    record = evidence.record
    if record is None:
        return {"duration_s": task_duration, "scene_count": None, "peak_rss_mb": None}
    return {
        "duration_s": record.duration_s or task_duration,
        "scene_count": record.scene_count,
        "peak_rss_mb": record.peak_rss_mb,
    }


def _completed_result(
    job: ProcessingJob,
    evidence: _TileEvidence,
    metrics: dict[str, Any],
    *,
    window: str,
    storage: StorageBackend,
) -> JobResult:
    """A tile with both COGs, whatever its VM managed to say about itself.

    A tile whose VM never reported still has both assets at known keys, and the
    manifest is the catalog's shopping list, so derive them.
    """
    record = evidence.record
    return JobResult(
        job=job,
        status="completed",
        lst_key=(record.lst_key if record else None)
        or storage.cog_key(window, evidence.tile, "lst_p95"),
        qa_key=(record.qa_key if record else None)
        or storage.cog_key(window, evidence.tile, "qa_count"),
        **metrics,
    )


def _resolve_tile(
    evidence: _TileEvidence,
    submission: BatchSubmission,
    *,
    completed: set[str],
    storage: StorageBackend,
) -> JobResult:
    """One tile's outcome, from the COG listing first and the record second."""
    record = evidence.record
    job = record.job if record else _rebuild_job(evidence.tile, submission)
    metrics = _observed_metrics(evidence)

    if evidence.tile in completed:
        return _completed_result(job, evidence, metrics, window=submission.window, storage=storage)

    return JobResult(
        job=job,
        status="failed",
        error=(record.error if record else None)
        or _failure_reason(evidence.task, evidence.log_key),
        **metrics,
    )


def _rebuild_job(tile: str, submission: BatchSubmission) -> ProcessingJob:
    """The job a tile was submitted with, for a tile that published nothing."""
    from landsat_lst.models import ProcessingJob  # noqa: PLC0415
    from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415

    return ProcessingJob(
        tile=parse_tile_name(tile),
        year=submission.year,
        end_year=submission.end_year,
    )


def _tile_costs(
    evidence: list[_TileEvidence], results: dict[str, JobResult]
) -> dict[str, pricing.CostEstimate]:
    """Price every tile whose duration is known, keyed by tile name.

    A tile that never reported a duration is left out rather than priced at
    zero: it ran on a VM that was billed, and calling that nothing would make a
    crashloop look free.

    A tile that reported no instance type is priced as the first VM type the
    fleet asks for, which is the one an unrecorded tile most likely ran on.
    :mod:`landsat_lst.pricing` labels that substitution, and resolves an
    unreported lifecycle against ``settings.coiled_spot_policy`` the same way,
    so the widened band travels with the figure.
    """
    fallback_type = settings.coiled_vm_types[0]
    costs: dict[str, pricing.CostEstimate] = {}
    for item in evidence:
        result = results.get(item.tile)
        duration = result.duration_s if result else None
        if duration is None:
            continue
        estimate = pricing.tile_cost(
            duration_s=duration,
            instance_type=item.instance_type or fallback_type,
            lifecycle=item.lifecycle,
        )
        if estimate is not None:
            costs[item.tile] = estimate
    return costs


def _memory_comparison(plan: dict, peaks: list[float]) -> dict[str, Any]:
    """The planned floor against the worst peak the run actually reached.

    A ratio, not a verdict. The floor counts the block-time stacks one phase
    holds at once plus a resident climatology and a process baseline, and
    nothing else, so a run above it is the ordinary case. The size of the gap
    is the number worth keeping: a 300-scene N40W075 sample peaked at 78.6 GB
    against a floor of a few GB, and that gap is what said the offset pass fans
    out rather than streams.

    The floor compared against is the largest of the two phases, because peak
    RSS is one high-water mark for the whole process and cannot be attributed
    to a phase after the fact.
    """
    floors = {
        name: phase["floor_gib"]
        for name, phase in plan.get("phases", {}).items()
        if phase.get("floor_gib")
    }
    phase, floor = max(floors.items(), key=lambda item: item[1], default=(None, None))
    observed = round(max(peaks) / MIB_PER_GIB, 2) if peaks else None
    return {
        "floor_gib": floor,
        "floor_phase": phase,
        "observed_peak_gib": observed,
        "observed_tiles": len(peaks),
        "ratio": round(observed / floor, 2) if observed and floor else None,
    }


def _plan_comparison(plan: dict | None, results: list[JobResult]) -> dict[str, Any] | None:
    """What the run was submitted expecting, next to what it did.

    Scenes are compared as well as memory because ``--max-scenes`` and the
    production default are both assumptions, while the run reports what STAC
    returned for each tile.

    Returns ``None`` for a submission that stored no plan, which omits the
    block rather than filling it with nulls.
    """
    if not plan:
        return None

    peaks = [r.peak_rss_mb for r in results if r.peak_rss_mb is not None]
    scenes = [r.scene_count for r in results if r.scene_count is not None]
    return {
        "planned": plan,
        "memory": _memory_comparison(plan, peaks),
        "scenes": {
            "planned": plan.get("scenes"),
            "observed_min": min(scenes, default=None),
            "observed_max": max(scenes, default=None),
            "observed_tiles": len(scenes),
        },
    }


def _skipped_results(submission: BatchSubmission) -> list[JobResult]:
    """The tiles that were already finished and never cost this run anything."""
    return [
        JobResult(job=_rebuild_job(tile, submission), status="skipped")
        for tile in submission.skipped_tiles
    ]


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

    submission = load_submission(run_id, out_dir)
    # The run's own record of where it wrote, not this shell's configuration.
    storage = storage or submission.storage()

    completed = storage.list_completed(submission.window)
    # One listing of the run prefix answers every question about what the tiles
    # left behind: which attempts each one made, which of those published a
    # state, and which left a log worth pointing at. runs.classify is the only
    # place those key names are parsed.
    evidence = _read_evidence(submission, storage, storage.list_prefix(storage.run_prefix(run_id)))

    submitted = [
        _resolve_tile(item, submission, completed=completed, storage=storage) for item in evidence
    ]
    results = submitted + _skipped_results(submission)

    manifest_path = write_run_manifest(
        results,
        run_id=run_id,
        window=submission.window,
        started_at=datetime.fromisoformat(submission.submitted_at),
        retries=settings.coiled_retries,
        cluster_id=submission.cluster_id,
        job_id=submission.job_id,
        attempts={item.tile: item.attempts for item in evidence},
        costs=_tile_costs(evidence, {r.job.tile.name: r for r in submitted}),
        plan=_plan_comparison(submission.plan, results),
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
