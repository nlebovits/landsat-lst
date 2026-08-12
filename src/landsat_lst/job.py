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

import shutil
import tempfile
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
    """Result of processing a single tile-window job."""

    job: ProcessingJob
    status: str  # "completed", "skipped", "failed"
    lst_key: str | None = None
    qa_key: str | None = None
    error: str | None = None


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

    try:
        # Layer 2: Process tile through pipeline
        logger.info("tile_processing_start")
        composite = process_tile(job)

        # Layer 3: Export COGs and upload them
        lst_key, qa_key = _write_cogs(composite, storage, job, logger)

        logger.info("tile_completed", lst_key=lst_key, qa_key=qa_key)
        return JobResult(job=job, status="completed", lst_key=lst_key, qa_key=qa_key)

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
        force: If True, reprocess even if the COGs exist
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
