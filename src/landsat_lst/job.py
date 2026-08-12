"""Job orchestration with idempotent COG writes and retry support.

This module implements the retry/resume strategy from ADR-001 Section 16:
- Idempotent asset check: skip tiles whose COGs are already stored
- Retry: transient failures escape so the batch task retries on a fresh VM

:func:`process_tile_job` is the whole unit of work one machine performs, whether
that machine is a laptop or a Coiled Batch VM. Submission and reconciliation of
a distributed run live in :mod:`landsat_lst.batch`.

Usage:
    from landsat_lst.job import process_tile_job, run_batch
    from landsat_lst.models import ProcessingJob, TileId

    job = ProcessingJob(tile=TileId(lat=40, lon=-75), year=2023)
    result = process_tile_job(job, force=False)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
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

    def to_record(self) -> dict:
        """Serialize to the JSON object a batch task leaves in storage.

        The job is written as its three constructor fields rather than as a
        ``model_dump``, so a record round-trips without relying on pydantic
        ignoring the computed fields a dump would also emit.
        """
        return {
            "tile": self.job.tile.name,
            "year": self.job.year,
            "end_year": self.job.end_year,
            "status": self.status,
            "lst_key": self.lst_key,
            "qa_key": self.qa_key,
            "error": self.error,
            "duration_s": self.duration_s,
            "scene_count": self.scene_count,
            "peak_rss_mb": self.peak_rss_mb,
        }

    @classmethod
    def from_record(cls, record: dict) -> JobResult:
        """Rebuild a result from :meth:`to_record` output."""
        from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415

        return cls(
            job=ProcessingJob(
                tile=parse_tile_name(record["tile"]),
                year=record["year"],
                end_year=record["end_year"],
            ),
            status=record["status"],
            lst_key=record.get("lst_key"),
            qa_key=record.get("qa_key"),
            error=record.get("error"),
            duration_s=record.get("duration_s"),
            scene_count=record.get("scene_count"),
            peak_rss_mb=record.get("peak_rss_mb"),
        )


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


def _write_run_record(result: JobResult, storage: StorageBackend, run_id: str, logger) -> None:
    """Leave this tile's outcome in storage for later reconciliation.

    A batch run has no live driver holding the results, so each task reports
    for itself. Reconciliation still takes tile completion from the COG
    listing; the record supplies what a listing cannot know -- duration, scene
    count, peak memory, and the error text behind a failure.

    A record that cannot be written must not fail a tile whose COGs are already
    safely uploaded, so the write is logged and swallowed.
    """
    try:
        storage.write_text(
            storage.run_record_key(run_id, result.job.tile.name),
            json.dumps(result.to_record(), indent=2),
        )
    except Exception as e:
        logger.warning("run_record_write_failed", run_id=run_id, error=str(e))


def process_tile_job(
    job: ProcessingJob,
    *,
    force: bool = False,
    storage: StorageBackend | None = None,
    run_id: str | None = None,
) -> JobResult:
    """Process a single tile-window job with retry/resume support.

    This is the main entry point for tile processing, and the unit of work one
    Coiled Batch VM runs. It implements:

    1. Idempotent check: skip if both COGs exist (unless force=True)
    2. Pipeline: query STAC, load scenes, compute composite
    3. COG export: uint16 LST DN + 12-band monthly QA, uploaded to storage
    4. Run record: the outcome written back to storage, when ``run_id`` is set

    Args:
        job: Processing job specification (tile + window)
        force: If True, reprocess even if the COGs exist
        storage: Storage backend (defaults to configured backend)
        run_id: Distributed run this tile belongs to. When set, the outcome is
            written to ``_runs/{run_id}/{tile}.json`` for
            :func:`landsat_lst.batch.reconcile_run` to collect.

    Returns:
        JobResult with status and asset keys
    """
    storage = storage or get_storage()
    logger = log.bind(tile=job.tile.name, year=job.window_label)

    # Layer 1: Idempotent check
    if not force and storage.cog_exists(job.window_label, job.tile.name):
        logger.info("tile_skipped", reason="cogs_exist")
        result = JobResult(job=job, status="skipped")
        if run_id:
            _write_run_record(result, storage, run_id, logger)
        return result

    start = time.monotonic()
    try:
        # Layer 2: Process tile through pipeline
        logger.info("tile_processing_start")
        composite = process_tile(job)

        # Layer 3: Export COGs and upload them
        lst_key, qa_key = _write_cogs(composite, storage, job, logger)

        duration = time.monotonic() - start
        logger.info("tile_completed", lst_key=lst_key, qa_key=qa_key, duration_s=duration)
        result = JobResult(
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
            # Re-raise so the batch task's retries reschedule the tile on a
            # fresh VM; the idempotency check above makes a retry after a
            # partial upload safe.
            logger.warning("tile_transient_failure", error=str(e))
            raise
        logger.exception("tile_failed", error=str(e))
        result = JobResult(
            job=job,
            status="failed",
            error=str(e),
            duration_s=time.monotonic() - start,
            peak_rss_mb=_peak_rss_mb(),
        )

    if run_id:
        _write_run_record(result, storage, run_id, logger)
    return result


def run_batch(
    jobs: Iterable[ProcessingJob],
    *,
    force: bool = False,
    storage: StorageBackend | None = None,
) -> list[JobResult]:
    """Run a batch of tile-year jobs sequentially (local execution).

    For distributed execution, use :func:`landsat_lst.batch.submit_batch` instead.

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
