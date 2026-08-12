"""Job orchestration with idempotent COG writes and retry support.

This module implements the retry/resume strategy from ADR-001 Section 16:
- Idempotent asset check: skip tiles whose COGs are already stored
- Coiled worker retry: recover from transient failures

Usage:
    from landsat_lst.job import process_tile_job, run_batch
    from landsat_lst.models import ProcessingJob, TileId

    job = ProcessingJob(tile=TileId(lat=40, lon=-75), year=2023)
    result = process_tile_job(job, force=False)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from landsat_lst.cog import cog_export
from landsat_lst.config import settings
from landsat_lst.encoding import encode_lst_uint16
from landsat_lst.models import ProcessingJob
from landsat_lst.pipeline import process_tile
from landsat_lst.storage import StorageBackend, get_storage

if TYPE_CHECKING:
    from collections.abc import Iterable

    import xarray as xr

log = structlog.get_logger()


@dataclass
class JobResult:
    """Result of processing a single tile-window job.

    ``duration_s``, ``scene_count``, and ``peak_rss_mb`` feed the per-run
    manifest; a costed validation run reads them to project the price and
    instance size of the global build.
    """

    job: ProcessingJob
    status: str  # "completed", "skipped", "failed"
    lst_key: str | None = None
    qa_key: str | None = None
    error: str | None = None
    duration_s: float | None = None
    scene_count: int | None = None
    peak_rss_mb: float | None = None


def _peak_rss_mb() -> float | None:
    """Peak resident set size of this process in MiB, if measurable."""
    try:
        import resource  # noqa: PLC0415

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except (ImportError, ValueError):  # pragma: no cover - non-POSIX
        return None


_HTTP_SERVER_ERROR = 500
_RETRYABLE_AWS_CODES = frozenset(
    {
        "Throttling",
        "ThrottlingException",
        "SlowDown",
        "RequestTimeout",
        "RequestTimeoutException",
        "InternalError",
        "ServiceUnavailable",
    }
)


def _is_transient(exc: BaseException) -> bool:
    """Whether an exception is worth retrying on another worker.

    Transient failures (network, throttling, object-store hiccups) re-raise out
    of :func:`process_tile_job` so Coiled's task retries engage. Anything else
    is deterministic: retrying it would buy the same failure again at full
    price, so it is returned as a failed :class:`JobResult` instead.
    """
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    from rasterio.errors import RasterioIOError  # noqa: PLC0415

    if isinstance(exc, RasterioIOError):
        return True

    from botocore.exceptions import ClientError  # noqa: PLC0415

    if isinstance(exc, ClientError):
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        code = exc.response.get("Error", {}).get("Code", "")
        return status >= _HTTP_SERVER_ERROR or code in _RETRYABLE_AWS_CODES

    return False


def _worker_environ() -> dict[str, str]:
    """Environment forwarded to every Coiled worker.

    Ships three things the workers cannot function without:

    - AWS credentials, frozen from the local session or ``settings.aws_profile``
      (SSO tokens resolved via boto3, never stale ``~/.aws/credentials`` reads).
    - ``AWS_REQUEST_PAYER=requester`` for the usgs-landsat source bucket.
    - ``LST_STORAGE_BACKEND=s3`` plus any local ``LST_*`` overrides, so workers
      never fall back to :class:`LocalStorage` and silently write COGs to
      ephemeral worker disk.

    ``LST_STAC_URL`` is deliberately not forwarded: a local Planetary Computer
    override must not leak onto AWS workers, where the config default (Earth
    Search) is the same-region endpoint.
    """
    creds = {
        key: val
        for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
        if (val := os.environ.get(key))
    }
    if "AWS_ACCESS_KEY_ID" not in creds:
        import boto3  # noqa: PLC0415

        session = boto3.Session(profile_name=settings.aws_profile)
        aws_creds = session.get_credentials()
        if aws_creds is None:
            msg = f"No AWS credentials found. Run: aws sso login --profile {settings.aws_profile}"
            raise RuntimeError(msg)
        frozen = aws_creds.get_frozen_credentials()
        creds = {
            "AWS_ACCESS_KEY_ID": frozen.access_key,
            "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
        }
        if frozen.token:
            creds["AWS_SESSION_TOKEN"] = frozen.token

    environ = {**creds, "AWS_REQUEST_PAYER": "requester", "LST_STORAGE_BACKEND": "s3"}
    for key, val in os.environ.items():
        if key.startswith("LST_") and key not in ("LST_STORAGE_BACKEND", "LST_STAC_URL"):
            environ[key] = val
    return environ


def _encode_native(composite: xr.Dataset) -> xr.Dataset:
    """Encode a float composite into the DN form :func:`cog_export` expects.

    ``process_tile`` yields ``lst_p95`` as float32 Celsius (nodata ``-9999``);
    the COG writer expects the uint16 DN whose scale/offset it stamps onto the
    band. ``qa_count`` already leaves the pipeline as ``uint8`` per-month counts.
    """
    return composite.assign(lst_p95=encode_lst_uint16(composite["lst_p95"]))


def _write_cogs(
    composite: xr.Dataset,
    storage: StorageBackend,
    job: ProcessingJob,
    logger,
) -> tuple[str, str]:
    """Export both COGs to a scratch dir, upload them, and drop the scratch dir.

    Assets are uploaded QA first and LST last, but that ordering carries no
    meaning: a reader must not treat the presence of ``lst_p95`` as proof the
    tile is complete. Completion is ``storage.cog_exists``, which checks both.

    Returns:
        ``(lst_key, qa_key)`` as stored.
    """
    window, tile = job.window_label, job.tile.name
    lst_key = storage.cog_key(window, tile, "lst_p95")
    qa_key = storage.cog_key(window, tile, "qa_count")

    scratch = Path(tempfile.mkdtemp(prefix="lst_job_"))
    try:
        lst_local = scratch / job.asset_filename("lst_p95")
        qa_local = scratch / job.asset_filename("qa_count")
        logger.info("cog_exporting")
        cog_export(_encode_native(composite), lst_local, qa_local)

        logger.info("cog_uploading", lst_key=lst_key, qa_key=qa_key)
        storage.upload(qa_local, qa_key)
        storage.upload(lst_local, lst_key)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return lst_key, qa_key


def process_tile_job(
    job: ProcessingJob,
    *,
    force: bool = False,
    storage: StorageBackend | None = None,
) -> JobResult:
    """Process a single tile-window job with retry/resume support.

    This is the main entry point for tile processing. It implements:
    1. Idempotent check: skip if both COGs exist (unless force=True)
    2. Pipeline: query STAC, load scenes, compute composite
    3. COG export: uint16 LST DN + 12-band monthly QA, uploaded to storage

    The @coiled.function decorator enables distributed execution.
    Use with retries parameter for transient failure recovery:
        results = process_tile_job.map(jobs, retries=3)

    Args:
        job: Processing job specification (tile + window)
        force: If True, reprocess even if the COGs exist
        storage: Storage backend (defaults to configured backend)

    Returns:
        JobResult with status and asset keys
    """
    storage = storage or get_storage()
    logger = log.bind(tile=job.tile.name, year=job.window_label)

    # Layer 1: Idempotent check
    if not force and storage.cog_exists(job.window_label, job.tile.name):
        logger.info("tile_skipped", reason="cogs_exist")
        return JobResult(job=job, status="skipped")

    start = time.monotonic()
    try:
        # Layer 2: Process tile through pipeline
        logger.info("tile_processing_start")
        composite = process_tile(job)

        # Layer 3: Export COGs and upload them
        lst_key, qa_key = _write_cogs(composite, storage, job, logger)

        duration = time.monotonic() - start
        logger.info("tile_completed", lst_key=lst_key, qa_key=qa_key, duration_s=duration)
        return JobResult(
            job=job,
            status="completed",
            lst_key=lst_key,
            qa_key=qa_key,
            duration_s=duration,
            scene_count=composite.attrs.get("scene_count"),
            peak_rss_mb=_peak_rss_mb(),
        )

    except Exception as e:
        if _is_transient(e):
            # Re-raise so Coiled's task retries reschedule the tile; the
            # idempotency check above makes a retry after partial upload safe.
            logger.warning("tile_transient_failure", error=str(e))
            raise
        logger.exception("tile_failed", error=str(e))
        return JobResult(
            job=job,
            status="failed",
            error=str(e),
            duration_s=time.monotonic() - start,
        )


def run_batch(
    jobs: Iterable[ProcessingJob],
    *,
    force: bool = False,
    storage: StorageBackend | None = None,
) -> list[JobResult]:
    """Run a batch of tile-year jobs sequentially (local execution).

    For distributed execution, use run_distributed() instead.

    Args:
        jobs: Iterable of processing jobs
        force: If True, reprocess even if the COGs exist
        storage: Storage backend (defaults to configured backend)

    Returns:
        List of JobResult for each job
    """
    storage = storage or get_storage()
    results = []

    for job in jobs:
        result = process_tile_job(job, force=force, storage=storage)
        results.append(result)

    return results


def _split_completed(
    jobs: list[ProcessingJob], storage: StorageBackend
) -> tuple[list[ProcessingJob], list[JobResult]]:
    """Split jobs into (to_run, skipped) using one listing per window.

    Filtering on the driver replaces two HEAD requests per tile from inside
    paid worker tasks, and lets a fully-complete batch skip the cluster.
    """
    done: set[tuple[str, str]] = set()
    for label in sorted({job.window_label for job in jobs}):
        done |= {(label, tile) for tile in storage.list_completed(label)}

    to_run: list[ProcessingJob] = []
    skipped: list[JobResult] = []
    for job in jobs:
        if (job.window_label, job.tile.name) in done:
            skipped.append(JobResult(job=job, status="skipped"))
        else:
            to_run.append(job)
    return to_run, skipped


def _submit_to_coiled(
    to_run: list[ProcessingJob], *, force: bool, retries: int, run_id: str
) -> list[JobResult]:
    """Map jobs over a pinned, tagged Coiled cluster and collect results."""
    import coiled  # noqa: PLC0415

    @coiled.function(
        name=f"lst-{run_id}",
        region=settings.coiled_region,
        vm_type=settings.coiled_vm_types,
        spot_policy=settings.coiled_spot_policy,
        n_workers=settings.coiled_n_workers,
        keepalive=settings.coiled_keepalive,
        environ=_worker_environ(),
        tags={"project": "landsat-lst", "run_id": run_id},
    )
    def _distributed_process(job: ProcessingJob, force: bool) -> JobResult:
        # Inside a Coiled task the cluster's own dask client is ambient, so an
        # unqualified compute() would submit the tile's whole scene graph back
        # to the shared scheduler -- three tiles at once crushed it
        # (scheduler-connection-lost, run 2021-2025-20260812T150618Z). Pin the
        # threaded scheduler so each tile computes on its own VM's cores.
        import dask  # noqa: PLC0415

        with dask.config.set(scheduler="threads"):
            return process_tile_job(job, force=force)

    # errors="skip" drops tasks that still fail after Coiled's retries; the
    # caller reconciles those tiles as failed instead of aborting the whole
    # batch on its last error.
    return list(
        _distributed_process.map(
            to_run,
            [force] * len(to_run),
            retries=retries,
            errors="skip",
        )
    )


def _reconcile_dropped(
    to_run: list[ProcessingJob], results: list[JobResult], retries: int
) -> list[JobResult]:
    """Mark tiles absent from the results as failed-after-retries."""
    returned = {(r.job.window_label, r.job.tile.name) for r in results}
    return [
        JobResult(
            job=job,
            status="failed",
            error=f"task failed after {retries} retries (see Coiled cluster logs)",
        )
        for job in to_run
        if (job.window_label, job.tile.name) not in returned
    ]


def run_distributed(
    jobs: list[ProcessingJob],
    *,
    force: bool = False,
    retries: int | None = None,
    run_id: str | None = None,
    storage: StorageBackend | None = None,
) -> list[JobResult]:
    """Run jobs distributed across Coiled workers with automatic retries.

    This is the production entry point for global processing. The cluster is
    pinned to ``settings.coiled_region`` with a fixed worker count, spot
    instances with on-demand fallback, and cost-attribution tags; workers
    receive AWS credentials and ``LST_*`` config through ``environ`` so they
    write to S3 rather than their own ephemeral disk.

    Already-completed tiles are filtered out with one storage listing per
    window before any task is submitted, so a resumed run pays for exactly the
    tiles that are missing. Every run writes a JSON manifest to
    ``settings.manifest_dir / f"{run_id}.json"`` recording per-tile status,
    duration, scene count, and peak memory.

    Args:
        jobs: List of processing jobs
        force: If True, reprocess even if the COGs exist
        retries: Number of retries per job (default from settings)
        run_id: Manifest and cluster name token; generated from the window
            and UTC timestamp when omitted
        storage: Storage backend used for the resume listing (default from
            :func:`get_storage`)

    Returns:
        List of JobResult for each job, including skipped and failed ones

    Raises:
        ImportError: If Coiled is not configured
        RuntimeError: If no AWS credentials can be resolved for the workers
    """
    try:
        import coiled  # noqa: F401, PLC0415
    except ImportError as e:
        msg = "Coiled is required for distributed execution. Install with: pip install coiled"
        raise ImportError(msg) from e

    from landsat_lst.manifest import write_run_manifest  # noqa: PLC0415

    retries = settings.coiled_retries if retries is None else retries
    storage = storage or get_storage()

    started_at = datetime.now(tz=UTC)
    windows = sorted({job.window_label for job in jobs})
    window = windows[0] if len(windows) == 1 else "multi"
    run_id = run_id or f"{window}-{started_at:%Y%m%dT%H%M%SZ}"

    to_run, skipped_results = (jobs, []) if force else _split_completed(jobs, storage)

    log.info(
        "distributed_batch_start",
        run_id=run_id,
        job_count=len(jobs),
        to_run=len(to_run),
        already_completed=len(skipped_results),
        retries=retries,
        force=force,
        region=settings.coiled_region,
        vm_types=settings.coiled_vm_types,
        spot_policy=settings.coiled_spot_policy,
        n_workers=settings.coiled_n_workers,
    )

    results: list[JobResult] = []
    if to_run:
        results = _submit_to_coiled(to_run, force=force, retries=retries, run_id=run_id)

    all_results = skipped_results + results + _reconcile_dropped(to_run, results, retries)
    manifest_path = write_run_manifest(
        all_results,
        run_id=run_id,
        window=window,
        started_at=started_at,
        retries=retries,
    )

    log.info(
        "distributed_batch_complete",
        run_id=run_id,
        completed=sum(1 for r in all_results if r.status == "completed"),
        skipped=sum(1 for r in all_results if r.status == "skipped"),
        failed=sum(1 for r in all_results if r.status == "failed"),
        manifest=str(manifest_path),
    )

    return all_results


DEFAULT_WINDOW = (2021, 2025)
"""Production window: the five most recent complete calendar years.

Five years gives near-complete monthly coverage (the 1-year composite leaves
~17% of pixels with no November observation; 3 years closes that to ~0%) while
staying representative of present-day conditions. Storage is unchanged across
window lengths, since the QA climatology is 12 bands regardless.
"""


def generate_jobs(
    years: Iterable[int] | None = None,
    *,
    window: tuple[int, int] = DEFAULT_WINDOW,
) -> list[ProcessingJob]:
    """Generate processing jobs for every land tile.

    Uses the LAND_TILES set from tiling.py to skip ocean tiles.

    Args:
        years: Single-year windows to emit, one job per (year, tile). Pass this
            only for benchmarking or backfill; production uses ``window``.
        window: ``(start, end)`` inclusive multi-year window, emitting one job
            per tile. Ignored when ``years`` is given.

    Returns:
        List of ProcessingJob covering all land tiles.
    """
    from landsat_lst.tiling import LAND_TILES, parse_tile_name  # noqa: PLC0415

    tiles = [parse_tile_name(name) for name in sorted(LAND_TILES)]

    if years is not None:
        years = list(years)
        jobs = [ProcessingJob(tile=tile, year=year) for year in years for tile in tiles]
        log.info("jobs_generated", count=len(jobs), years=years)
        return jobs

    start, end = window
    jobs = [ProcessingJob(tile=tile, year=start, end_year=end) for tile in tiles]
    log.info("jobs_generated", count=len(jobs), window=f"{start}-{end}")
    return jobs
