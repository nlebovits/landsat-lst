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
    lst_cog: Path | None = None
    qa_cog: Path | None = None
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
        if self.qa_cog:
            lines.append(f"QA COG: {self.qa_cog}")
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
    import xarray as xr  # noqa: PLC0415

    from landsat_lst.cog import cog_export  # noqa: PLC0415
    from landsat_lst.encoding import encode_lst_uint16  # noqa: PLC0415
    from landsat_lst.models import ProcessingJob  # noqa: PLC0415
    from landsat_lst.pipeline import process_tile  # noqa: PLC0415
    from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415

    result = ProfileResult(tile=tile_name, year=year)
    total_start = time.perf_counter()

    try:
        tile = parse_tile_name(tile_name)
        job = ProcessingJob(tile=tile, year=year)
        log.info("job_created", tile=tile_name, year=year, bbox=tile.bbox)

        # Run full pipeline (STAC query, load, composite, land mask)
        with timed_stage("process_tile", result) as stage:
            from dask.diagnostics import ProgressBar  # noqa: PLC0415

            composite = process_tile(job)
            result.scene_count = composite.attrs.get("scene_count", 0)
            # Force compute with progress bar
            with ProgressBar(dt=10):  # Update every 10 seconds
                composite = composite.compute()
            stage.extra["output_shape"] = str(dict(composite.sizes))
            stage.extra["nbytes"] = f"{composite.nbytes / 1e6:.2f} MB"
            stage.extra["scene_count"] = result.scene_count

        # Encode to the published uint16 contract and write both COGs.
        with timed_stage("cog_write", result) as stage:
            native = xr.Dataset(
                {
                    "lst_p95": encode_lst_uint16(composite["lst_p95"]),
                    "qa_count": composite["qa_count"],
                }
            )
            item_dir = output_dir / f"lst-p95-{job.window_label}" / tile_name
            lst_cog, qa_cog = cog_export(
                native, item_dir / "lst_p95.tif", item_dir / "qa_count.tif"
            )

            result.lst_cog, result.qa_cog = lst_cog, qa_cog
            result.output_path = str(lst_cog)
            stage.extra["lst_mb"] = round(lst_cog.stat().st_size / 1e6, 1)
            stage.extra["qa_mb"] = round(qa_cog.stat().st_size / 1e6, 1)

        result.success = True

    except Exception as e:
        log.exception("pipeline_failed", error=str(e))
        result.success = False
        result.error = str(e)

    result.total_duration_secs = time.perf_counter() - total_start
    return result


def verify_output(result: ProfileResult) -> None:
    """Verify both COGs are valid, carry overviews, and embed their statistics.

    Statistics are read with ``GDAL_PAM_ENABLED=NO`` so a ``.aux.xml`` sidecar cannot
    satisfy the check. A published COG has to answer for itself, because a consumer
    reading it over HTTPS never sees the sidecar.
    """
    import rasterio  # noqa: PLC0415
    from rio_cogeo.cogeo import cog_validate  # noqa: PLC0415

    log.info("verification_start", lst_cog=str(result.lst_cog), qa_cog=str(result.qa_cog))

    assert result.lst_cog is not None, "No LST COG recorded"
    assert result.qa_cog is not None, "No QA COG recorded"

    for path in (result.lst_cog, result.qa_cog):
        is_valid, errors, warnings_ = cog_validate(str(path))
        assert is_valid, f"{path.name} is not a valid COG: {errors}"
        if warnings_:
            log.warning("cog_validate_warnings", cog=path.name, warnings=warnings_)

    # gdalinfo-style structural checks, with PAM off so nothing comes from a sidecar.
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(result.lst_cog) as src:
            assert src.count == 1, f"LST COG has {src.count} bands, expected 1"
            assert src.dtypes[0] == "uint16", f"LST COG dtype is {src.dtypes[0]}"
            assert src.nodata == 0, f"LST COG nodata is {src.nodata}, expected 0"
            assert src.scales[0] != 1.0, "LST COG carries no band scale"
            lst_overviews = src.overviews(1)
            assert lst_overviews, "LST COG has no overviews"
            lst_tags = src.tags(1)
            assert "STATISTICS_MINIMUM" in lst_tags, "LST COG has no embedded band statistics"
            log.info(
                "lst_cog_verified",
                shape=(src.height, src.width),
                overviews=lst_overviews,
                scale=src.scales[0],
                offset=src.offsets[0],
                stats_min=lst_tags.get("STATISTICS_MINIMUM"),
                stats_max=lst_tags.get("STATISTICS_MAXIMUM"),
            )

        with rasterio.open(result.qa_cog) as src:
            assert src.count == 12, f"QA COG has {src.count} bands, expected 12"
            assert src.dtypes[0] == "uint8", f"QA COG dtype is {src.dtypes[0]}"
            assert src.nodata is None, "QA COG must not set nodata: 0 is a real count"
            qa_overviews = src.overviews(1)
            assert qa_overviews, "QA COG has no overviews"
            assert "STATISTICS_MINIMUM" in src.tags(1), "QA COG has no embedded band statistics"
            log.info(
                "qa_cog_verified",
                shape=(src.height, src.width),
                overviews=qa_overviews,
                band_names=[src.descriptions[i] for i in range(src.count)],
            )

    log.info("verification_passed")


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
            verify_output(result)

        # Exit code
        sys.exit(0 if result.success else 1)

    finally:
        client.close()
        cluster.close()
        log.info("dask_cluster_shutdown")


if __name__ == "__main__":
    main()
