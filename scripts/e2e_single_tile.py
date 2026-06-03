#!/usr/bin/env python3
"""End-to-end test: Single tile with full profiling and Dask LocalCluster.

Usage:
    # Basic run with timing and logging
    uv run python scripts/e2e_single_tile.py

    # With cProfile output
    uv run python scripts/e2e_single_tile.py --cprofile

    # With py-spy (run externally)
    uv run py-spy record -o profile.svg -- python scripts/e2e_single_tile.py

    # With memray (run externally)
    uv run memray run -o memray.bin python scripts/e2e_single_tile.py
    uv run memray flamegraph memray.bin

Configuration via environment:
    LST_TILE=N40W075         # Tile to process (default: N40W075 - Philadelphia)
    LST_YEAR=2024            # Year to process (default: 2024)
    LST_WORKERS=8            # Dask workers (default: 8)
    LST_THREADS=2            # Threads per worker (default: 2)
    LST_MEMORY=6GB           # Memory per worker (default: 6GB)
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
import tracemalloc
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Generator

# Suppress benign warnings
warnings.filterwarnings("ignore", message=".*NotGeoreferencedWarning.*")
warnings.filterwarnings("ignore", message=".*Dataset has no geotransform.*")
warnings.filterwarnings("ignore", message=".*Sending large graph.*")
warnings.filterwarnings("ignore", message=".*LocalFileSystem storage is not safe.*")

# Configure structlog for verbose console output (without format_exc_info to avoid warning)
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(
            colors=True, exception_formatter=structlog.dev.plain_traceback
        ),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_secs: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class ProfileResult:
    """Complete profiling results for a pipeline run."""

    tile: str
    year: int
    stages: list[StageMetrics] = field(default_factory=list)
    total_duration_secs: float = 0.0
    scene_count: int = 0
    output_path: str = ""
    commit_id: str | None = None
    peak_memory_mb: float = 0.0
    success: bool = False
    error: str | None = None

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            f"PROFILE SUMMARY: {self.tile} / {self.year}",
            "=" * 60,
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
        ]
        if self.error:
            lines.append(f"Error: {self.error}")
        lines.extend(
            [
                f"Scenes processed: {self.scene_count}",
                f"Peak memory: {self.peak_memory_mb:.1f} MB",
                f"Output: {self.output_path}",
            ]
        )
        if self.commit_id:
            lines.append(f"Icechunk commit: {self.commit_id[:12]}")
        lines.extend(
            [
                "",
                "Stage Breakdown:",
                "-" * 40,
            ]
        )
        for stage in self.stages:
            pct = (
                (stage.duration_secs / self.total_duration_secs * 100)
                if self.total_duration_secs > 0
                else 0
            )
            lines.append(f"  {stage.name:20s} {stage.duration_secs:7.2f}s ({pct:5.1f}%)")
            for k, v in stage.extra.items():
                lines.append(f"    {k}: {v}")
        lines.extend(
            [
                "-" * 40,
                f"  {'TOTAL':20s} {self.total_duration_secs:7.2f}s",
                "=" * 60,
            ]
        )
        return "\n".join(lines)


@contextmanager
def timed_stage(name: str, result: ProfileResult) -> Generator[StageMetrics, None, None]:
    """Context manager to time a pipeline stage."""
    stage = StageMetrics(name=name)
    stage.start_time = time.perf_counter()
    log.info("stage_start", stage=name)
    try:
        yield stage
    finally:
        stage.end_time = time.perf_counter()
        stage.duration_secs = stage.end_time - stage.start_time
        result.stages.append(stage)
        log.info(
            "stage_complete", stage=name, duration_secs=round(stage.duration_secs, 2), **stage.extra
        )


def setup_dask_cluster(workers: int, threads: int, memory: str):
    """Create and configure a Dask LocalCluster."""
    from dask.distributed import Client, LocalCluster  # noqa: PLC0415

    log.info(
        "dask_cluster_starting", workers=workers, threads_per_worker=threads, memory_limit=memory
    )

    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=threads,
        memory_limit=memory,
        dashboard_address=":8787",
    )
    client = Client(cluster)

    log.info(
        "dask_cluster_ready",
        dashboard=client.dashboard_link,
        scheduler=client.scheduler_info()["address"],
    )

    return cluster, client


def run_pipeline(tile_name: str, year: int, output_dir: Path) -> ProfileResult:
    """Run the full pipeline with profiling instrumentation."""
    from landsat_lst.models import ProcessingJob  # noqa: PLC0415
    from landsat_lst.pipeline import (  # noqa: PLC0415
        compute_annual_composite,
        load_scenes,
        query_stac,
    )
    from landsat_lst.storage import IcechunkStorage  # noqa: PLC0415
    from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415
    from landsat_lst.zarr_writer import write_zarr  # noqa: PLC0415

    result = ProfileResult(tile=tile_name, year=year)
    total_start = time.perf_counter()

    try:
        tile = parse_tile_name(tile_name)
        job = ProcessingJob(tile=tile, year=year)
        log.info("job_created", tile=tile_name, year=year, bbox=tile.bbox)

        # Stage 1: STAC Query
        with timed_stage("stac_query", result) as stage:
            items = query_stac(job)
            result.scene_count = len(items)
            stage.extra["scene_count"] = len(items)
            if not items:
                raise ValueError(f"No scenes found for {tile_name} in {year}")

        # Stage 2: Load Scenes (this triggers Dask task graph construction)
        with timed_stage("load_scenes", result) as stage:
            data = load_scenes(items, job.tile.bbox)
            stage.extra["shape"] = str(dict(data.sizes))
            stage.extra["chunks"] = str(data.chunks)
            stage.extra["nbytes_lazy"] = f"{data.nbytes / 1e9:.2f} GB"

        # Stage 3: Compute Annual Composite (this triggers Dask compute)
        with timed_stage("compute_composite", result) as stage:
            from dask.diagnostics import ProgressBar  # noqa: PLC0415

            composite = compute_annual_composite(data)
            # Force compute with progress bar
            with ProgressBar(dt=10):  # Update every 10 seconds
                composite = composite.compute()
            stage.extra["output_shape"] = str(dict(composite.sizes))
            stage.extra["nbytes"] = f"{composite.nbytes / 1e6:.2f} MB"

        # Stage 4: Write to Icechunk
        with timed_stage("icechunk_write", result) as stage:
            icechunk_path = output_dir / "icechunk"
            storage = IcechunkStorage.from_local(icechunk_path)

            session = storage.writable_session()
            group_path = storage.zarr_path(year, tile_name)

            write_zarr(composite, session, group=group_path)

            commit_msg = f"E2E test: {tile_name} for {year}"
            commit_id = session.commit(commit_msg)

            result.output_path = str(icechunk_path / group_path)
            result.commit_id = commit_id
            stage.extra["commit_id"] = commit_id[:12]

        result.success = True

    except Exception as e:
        log.exception("pipeline_failed", error=str(e))
        result.success = False
        result.error = str(e)

    result.total_duration_secs = time.perf_counter() - total_start
    return result


def verify_output(result: ProfileResult, output_dir: Path) -> None:
    """Verify the written Zarr can be read back correctly."""
    import xarray as xr  # noqa: PLC0415

    from landsat_lst.storage import IcechunkStorage  # noqa: PLC0415

    log.info("verification_start")

    icechunk_path = output_dir / "icechunk"
    storage = IcechunkStorage.from_local(icechunk_path)
    session = storage.readonly_session()

    group_path = f"{result.year}/{result.tile}"
    ds = xr.open_zarr(session.store, group=group_path)

    log.info(
        "verification_complete",
        variables=list(ds.data_vars),
        shape=dict(ds.sizes),
        lst_p95_dtype=str(ds["lst_p95"].dtype),
        lst_p95_range=f"[{int(ds['lst_p95'].min().values)}, {int(ds['lst_p95'].max().values)}]",
        qa_count_max=int(ds["qa_count"].max().values),
    )

    # Check attributes
    assert "lst_scale_factor" in ds["lst_p95"].attrs, "Missing scale_factor attribute"
    assert "_CRS" in ds.attrs, "Missing _CRS attribute"

    log.info("verification_passed", commit_id=result.commit_id[:12] if result.commit_id else None)


def main():
    parser = argparse.ArgumentParser(description="E2E single tile test with profiling")
    parser.add_argument(
        "--no-cprofile", action="store_true", help="Disable cProfile (enabled by default)"
    )
    parser.add_argument("--cprofile-sort", default="cumtime", help="cProfile sort key")
    parser.add_argument("--output-dir", type=Path, default=Path("output/e2e_test"))
    args = parser.parse_args()
    args.cprofile = not args.no_cprofile  # cProfile enabled by default

    # Configuration from environment
    tile_name = os.environ.get("LST_TILE", "N40W075")
    year = int(os.environ.get("LST_YEAR", "2024"))
    workers = int(os.environ.get("LST_WORKERS", "8"))
    threads = int(os.environ.get("LST_THREADS", "2"))
    memory = os.environ.get("LST_MEMORY", "6GB")

    log.info(
        "e2e_test_starting",
        tile=tile_name,
        year=year,
        workers=workers,
        threads=threads,
        memory=memory,
        output_dir=str(args.output_dir),
        timestamp=datetime.now().isoformat(),
    )

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up Dask cluster
    cluster, client = setup_dask_cluster(workers, threads, memory)

    try:
        # Start memory tracking
        tracemalloc.start()

        if args.cprofile:
            profiler = cProfile.Profile()
            profiler.enable()

        # Run the pipeline
        result = run_pipeline(tile_name, year, args.output_dir)

        # Capture peak memory
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result.peak_memory_mb = peak_memory / 1024 / 1024

        if args.cprofile:
            profiler.disable()
            # Save cProfile stats
            stats_path = args.output_dir / "cprofile_stats.prof"
            profiler.dump_stats(str(stats_path))
            log.info("cprofile_saved", path=str(stats_path))

            # Print top functions
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats(args.cprofile_sort)
            ps.print_stats(30)
            print("\n" + "=" * 60)
            print("CPROFILE TOP 30 FUNCTIONS")
            print("=" * 60)
            print(s.getvalue())

        # Print profile summary
        print(result.summary())

        # Verify output if successful
        if result.success:
            verify_output(result, args.output_dir)

        # Exit code
        sys.exit(0 if result.success else 1)

    finally:
        client.close()
        cluster.close()
        log.info("dask_cluster_shutdown")


if __name__ == "__main__":
    main()
