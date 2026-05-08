"""Job orchestration with idempotent Zarr writes and retry support.

This module implements the retry/resume strategy from ADR-001 Section 16
(updated per ADR-003 for direct Zarr writes):
- Idempotent Zarr check: skip tiles where output already exists
- Coiled worker retry: recover from transient failures

Usage:
    from landsat_lst.job import process_tile_job, run_batch
    from landsat_lst.models import ProcessingJob, TileId

    job = ProcessingJob(tile=TileId(lat=40, lon=-75), year=2023)
    result = process_tile_job(job, force=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from landsat_lst.config import settings
from landsat_lst.models import ProcessingJob
from landsat_lst.pipeline import process_tile
from landsat_lst.storage import StorageBackend, get_storage
from landsat_lst.zarr_writer import write_zarr

if TYPE_CHECKING:
    from collections.abc import Iterable

log = structlog.get_logger()


@dataclass
class JobResult:
    """Result of processing a single tile-year job."""

    job: ProcessingJob
    status: str  # "completed", "skipped", "failed"
    zarr_path: str | None = None
    error: str | None = None


def process_tile_job(
    job: ProcessingJob,
    *,
    force: bool = False,
    storage: StorageBackend | None = None,
) -> JobResult:
    """Process a single tile-year job with retry/resume support.

    This is the main entry point for tile processing. It implements:
    1. Idempotent check: skip if Zarr exists (unless force=True)
    2. Pipeline: query STAC, load scenes, compute composite
    3. Zarr write: uint16 encoding, proper metadata

    The @coiled.function decorator enables distributed execution.
    Use with retries parameter for transient failure recovery:
        results = process_tile_job.map(jobs, retries=3)

    Args:
        job: Processing job specification (tile + year)
        force: If True, reprocess even if Zarr exists
        storage: Storage backend (defaults to configured backend)

    Returns:
        JobResult with status and path info
    """
    storage = storage or get_storage()
    logger = log.bind(tile=job.tile.name, year=job.year)

    # Layer 1: Idempotent check
    if not force and storage.zarr_exists(job.year, job.tile.name):
        logger.info("tile_skipped", reason="zarr_exists")
        return JobResult(job=job, status="skipped")

    try:
        # Layer 2: Process tile through pipeline
        logger.info("tile_processing_start")
        composite = process_tile(job)

        # Write Zarr store
        zarr_path = storage.zarr_path(job.year, job.tile.name)
        logger.info("zarr_writing", path=zarr_path)
        write_zarr(composite, zarr_path)

        logger.info("tile_completed", zarr_path=zarr_path)
        return JobResult(
            job=job,
            status="completed",
            zarr_path=zarr_path,
        )

    except Exception as e:
        logger.exception("tile_failed", error=str(e))
        return JobResult(job=job, status="failed", error=str(e))


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
        force: If True, reprocess even if Zarr stores exist
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


def run_distributed(
    jobs: list[ProcessingJob],
    *,
    force: bool = False,
    retries: int | None = None,
) -> list[JobResult]:
    """Run jobs distributed across Coiled workers with automatic retries.

    This is the production entry point for global processing.
    Uses Coiled's .map() for parallel execution with retry support.

    Requires Coiled authentication. If Coiled is not available,
    raises ImportError with instructions.

    Args:
        jobs: List of processing jobs
        force: If True, reprocess even if Zarr stores exist
        retries: Number of retries per job (default from settings)

    Returns:
        List of JobResult for each job

    Raises:
        ImportError: If Coiled is not configured
    """
    try:
        import coiled  # noqa: PLC0415
    except ImportError as e:
        msg = "Coiled is required for distributed execution. Install with: pip install coiled"
        raise ImportError(msg) from e

    if retries is None:
        retries = settings.coiled_retries

    log.info(
        "distributed_batch_start",
        job_count=len(jobs),
        retries=retries,
        force=force,
    )

    # Create a fresh coiled.function for distributed execution
    @coiled.function()
    def _distributed_process(job: ProcessingJob, force: bool) -> JobResult:
        return process_tile_job(job, force=force)

    results = list(
        _distributed_process.map(
            jobs,
            [force] * len(jobs),
            retries=retries,
        )
    )

    completed = sum(1 for r in results if r.status == "completed")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")

    log.info(
        "distributed_batch_complete",
        completed=completed,
        skipped=skipped,
        failed=failed,
    )

    return results


def generate_jobs(years: Iterable[int]) -> list[ProcessingJob]:
    """Generate all processing jobs for the given years.

    Uses the LAND_TILES set from tiling.py to skip ocean tiles.

    Args:
        years: Years to process

    Returns:
        List of ProcessingJob for all land tiles and years
    """
    from landsat_lst.tiling import LAND_TILES, parse_tile_name  # noqa: PLC0415

    jobs = []
    for year in years:
        for tile_name in sorted(LAND_TILES):
            tile = parse_tile_name(tile_name)
            jobs.append(ProcessingJob(tile=tile, year=year))

    log.info("jobs_generated", count=len(jobs), years=list(years))
    return jobs
