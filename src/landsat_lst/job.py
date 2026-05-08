"""Job orchestration with per-tile commits and retry/resume support.

This module implements the retry/resume strategy from ADR-001 Section 16:
- Idempotent COG check: skip tiles where output already exists
- Per-tile Icechunk commits: durable partial progress
- ConflictError retry: handle concurrent write conflicts
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

import icechunk
import structlog
from obspec_utils.registry import ObjectStoreRegistry
from virtual_tiff import VirtualTIFF
from virtualizarr import open_virtual_dataset

from landsat_lst.cog import write_cog
from landsat_lst.config import settings
from landsat_lst.models import ProcessingJob
from landsat_lst.pipeline import process_tile
from landsat_lst.storage import StorageBackend, get_storage

if TYPE_CHECKING:
    from collections.abc import Iterable

log = structlog.get_logger()


@dataclass
class JobResult:
    """Result of processing a single tile-year job."""

    job: ProcessingJob
    status: str  # "completed", "skipped", "failed"
    commit_id: str | None = None
    cog_path: str | None = None
    error: str | None = None


def _commit_to_icechunk(
    cog_path: str,
    storage: StorageBackend,
    job: ProcessingJob,
    max_retries: int = 10,
) -> str:
    """Write COG virtual references to Icechunk with conflict retry.

    Implements the "uncooperative distributed writes" pattern from
    Icechunk docs: each worker opens own session, commits independently,
    retries on ConflictError.

    Args:
        cog_path: Path to the COG file (local path or s3:// URL)
        storage: Storage backend for Icechunk
        job: Processing job for commit message
        max_retries: Maximum conflict retries before giving up

    Returns:
        Icechunk commit ID

    Raises:
        icechunk.ConflictError: If max retries exceeded
    """
    for attempt in range(max_retries):
        try:
            repo = icechunk.Repository.open_or_create(storage.icechunk_storage())
            session = repo.writable_session("main")

            registry = ObjectStoreRegistry()
            parser = VirtualTIFF(ifd=0)  # ifd=0 for COGs with overviews
            vds = open_virtual_dataset(cog_path, registry=registry, parser=parser)
            vds.virtualize.to_icechunk(session.store)

            commit_id = session.commit(f"Add {job.tile.name} {job.year}")
            log.info(
                "icechunk_commit",
                tile=job.tile.name,
                year=job.year,
                commit_id=commit_id,
                attempt=attempt + 1,
            )
            return commit_id

        except icechunk.ConflictError:
            log.warning(
                "icechunk_conflict",
                tile=job.tile.name,
                year=job.year,
                attempt=attempt + 1,
                max_retries=max_retries,
            )
            if attempt == max_retries - 1:
                raise
            continue

    msg = f"Max retries ({max_retries}) exceeded for {job.tile.name} {job.year}"
    raise icechunk.ConflictError(msg)


def process_tile_job(
    job: ProcessingJob,
    *,
    force: bool = False,
    storage: StorageBackend | None = None,
) -> JobResult:
    """Process a single tile-year job with retry/resume support.

    This is the main entry point for tile processing. It implements:
    1. Idempotent check: skip if COG exists (unless force=True)
    2. Pipeline: query STAC, load scenes, compute composite
    3. COG write: uint16 encoding, TIFF tags
    4. Icechunk commit: virtual references with conflict retry

    The @coiled.function decorator enables distributed execution.
    Use with retries parameter for transient failure recovery:
        results = process_tile_job.map(jobs, retries=3)

    Args:
        job: Processing job specification (tile + year)
        force: If True, reprocess even if COG exists
        storage: Storage backend (defaults to configured backend)

    Returns:
        JobResult with status and commit info
    """
    storage = storage or get_storage()
    logger = log.bind(tile=job.tile.name, year=job.year)

    # Layer 1: Idempotent check
    if not force and storage.cog_exists(job.year, job.tile.name):
        logger.info("tile_skipped", reason="cog_exists")
        return JobResult(job=job, status="skipped")

    try:
        # Layer 2: Process tile through pipeline
        logger.info("tile_processing_start")
        composite = process_tile(job)

        # Write COG
        cog_path = storage.cog_path(job.year, job.tile.name)
        logger.info("cog_writing", path=cog_path)
        write_cog(composite, cog_path)

        # Layer 3: Commit to Icechunk (with conflict retry)
        commit_id = _commit_to_icechunk(cog_path, storage, job)

        logger.info("tile_completed", commit_id=commit_id)
        return JobResult(
            job=job,
            status="completed",
            commit_id=commit_id,
            cog_path=cog_path,
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
        force: If True, reprocess even if COGs exist
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
        force: If True, reprocess even if COGs exist
        retries: Number of retries per job (default from settings)

    Returns:
        List of JobResult for each job

    Raises:
        ImportError: If Coiled is not configured
    """
    try:
        import coiled  # noqa: PLC0415 - lazy import for optional dependency
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
