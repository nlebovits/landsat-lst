#!/usr/bin/env python3
"""Benchmark: Planetary Computer vs Earth Search STAC endpoints.

Runs the same tile/year through both endpoints and compares performance.
Use this locally first, then on Coiled for production-representative results.

Usage:
    # Local benchmark (quick validation)
    uv run python scripts/benchmark_endpoints.py

    # Custom tile/year
    LST_TILE=N35W120 LST_YEAR=2023 uv run python scripts/benchmark_endpoints.py

    # Run on Coiled (requires coiled login)
    uv run python scripts/benchmark_endpoints.py --coiled

    # More iterations for statistical confidence
    uv run python scripts/benchmark_endpoints.py --iterations 3

Output:
    Prints comparison table and saves results to output/benchmark/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import structlog

warnings.filterwarnings("ignore", message=".*NotGeoreferencedWarning.*")
warnings.filterwarnings("ignore", message=".*Dataset has no geotransform.*")
warnings.filterwarnings("ignore", message=".*Sending large graph.*")

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()

PLANETARY_COMPUTER = "https://planetarycomputer.microsoft.com/api/stac/v1"
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"


@dataclass
class BenchmarkRun:
    """Results from a single benchmark run."""

    endpoint: str
    endpoint_name: str
    tile: str
    year: int
    wall_time_secs: float = 0.0
    stac_query_secs: float = 0.0
    load_scenes_secs: float = 0.0
    compute_secs: float = 0.0
    scene_count: int = 0
    success: bool = False
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    tile: str
    year: int
    runs: list[BenchmarkRun] = field(default_factory=list)

    def summary_table(self) -> str:
        """Generate comparison table."""
        pc_runs = [r for r in self.runs if r.endpoint_name == "planetary_computer" and r.success]
        es_runs = [r for r in self.runs if r.endpoint_name == "earth_search" and r.success]

        def avg(runs: list[BenchmarkRun], attr: str) -> float:
            if not runs:
                return 0.0
            return sum(getattr(r, attr) for r in runs) / len(runs)

        lines = [
            "",
            "=" * 70,
            f"BENCHMARK RESULTS: {self.tile} / {self.year}",
            "=" * 70,
            "",
            f"{'Metric':<25} {'Planetary Computer':>20} {'Earth Search':>20}",
            "-" * 70,
            f"{'Successful runs':<25} {len(pc_runs):>20} {len(es_runs):>20}",
            f"{'Scenes found':<25} {avg(pc_runs, 'scene_count'):>20.0f} {avg(es_runs, 'scene_count'):>20.0f}",
            f"{'STAC query (s)':<25} {avg(pc_runs, 'stac_query_secs'):>20.2f} {avg(es_runs, 'stac_query_secs'):>20.2f}",
            f"{'Load scenes (s)':<25} {avg(pc_runs, 'load_scenes_secs'):>20.2f} {avg(es_runs, 'load_scenes_secs'):>20.2f}",
            f"{'Compute (s)':<25} {avg(pc_runs, 'compute_secs'):>20.2f} {avg(es_runs, 'compute_secs'):>20.2f}",
            f"{'Total wall time (s)':<25} {avg(pc_runs, 'wall_time_secs'):>20.2f} {avg(es_runs, 'wall_time_secs'):>20.2f}",
            "-" * 70,
        ]

        pc_avg = avg(pc_runs, "wall_time_secs")
        es_avg = avg(es_runs, "wall_time_secs")
        if pc_avg > 0 and es_avg > 0:
            speedup = pc_avg / es_avg
            if speedup > 1:
                lines.append(f"Earth Search is {speedup:.2f}x faster")
            else:
                lines.append(f"Planetary Computer is {1 / speedup:.2f}x faster")

        lines.append("=" * 70)
        return "\n".join(lines)


def run_single_benchmark(
    endpoint: str,
    endpoint_name: str,
    tile_name: str,
    year: int,
) -> BenchmarkRun:
    """Run pipeline with a specific STAC endpoint."""
    from landsat_lst.config import settings  # noqa: PLC0415
    from landsat_lst.models import ProcessingJob  # noqa: PLC0415
    from landsat_lst.pipeline import (  # noqa: PLC0415
        compute_annual_composite,
        load_scenes,
        query_stac,
    )
    from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415

    result = BenchmarkRun(
        endpoint=endpoint,
        endpoint_name=endpoint_name,
        tile=tile_name,
        year=year,
    )

    # Override STAC URL for this run
    original_url = settings.stac_url
    settings.stac_url = endpoint

    log.info(
        "benchmark_run_start",
        endpoint=endpoint_name,
        tile=tile_name,
        year=year,
    )

    total_start = time.perf_counter()

    try:
        tile = parse_tile_name(tile_name)
        job = ProcessingJob(tile=tile, year=year)

        # Stage 1: STAC Query
        t0 = time.perf_counter()
        items = query_stac(job)
        result.stac_query_secs = time.perf_counter() - t0
        result.scene_count = len(items)

        if not items:
            raise ValueError(f"No scenes found for {tile_name} in {year}")

        log.info("stac_query_done", scenes=len(items), secs=round(result.stac_query_secs, 2))

        # Stage 2: Load Scenes
        t0 = time.perf_counter()
        data = load_scenes(items, job.tile.bbox)
        result.load_scenes_secs = time.perf_counter() - t0
        log.info("load_scenes_done", secs=round(result.load_scenes_secs, 2))

        # Stage 3: Compute (this is where I/O actually happens with Dask)
        t0 = time.perf_counter()
        composite = compute_annual_composite(data)
        composite = composite.compute()
        result.compute_secs = time.perf_counter() - t0
        log.info("compute_done", secs=round(result.compute_secs, 2))

        result.success = True

    except Exception as e:
        log.exception("benchmark_run_failed", error=str(e))
        result.success = False
        result.error = str(e)

    finally:
        # Restore original URL
        settings.stac_url = original_url

    result.wall_time_secs = time.perf_counter() - total_start
    log.info(
        "benchmark_run_complete",
        endpoint=endpoint_name,
        success=result.success,
        wall_time=round(result.wall_time_secs, 2),
    )

    return result


def setup_dask_cluster(workers: int, threads: int, memory: str):
    """Create local Dask cluster."""
    from dask.distributed import Client, LocalCluster  # noqa: PLC0415

    log.info("dask_cluster_starting", workers=workers, threads=threads, memory=memory)

    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=threads,
        memory_limit=memory,
        dashboard_address=":8787",
    )
    client = Client(cluster)
    log.info("dask_cluster_ready", dashboard=client.dashboard_link)

    return cluster, client


def main():
    parser = argparse.ArgumentParser(description="Benchmark STAC endpoints")
    parser.add_argument("--iterations", type=int, default=1, help="Runs per endpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("output/benchmark"))
    parser.add_argument("--coiled", action="store_true", help="Run on Coiled (not implemented yet)")
    parser.add_argument("--pc-only", action="store_true", help="Only test Planetary Computer")
    parser.add_argument("--es-only", action="store_true", help="Only test Earth Search")
    args = parser.parse_args()

    if args.coiled:
        log.error("Coiled mode not implemented yet - run locally first")
        sys.exit(1)

    # Configuration
    tile_name = os.environ.get("LST_TILE", "N40W075")
    year = int(os.environ.get("LST_YEAR", "2024"))
    workers = int(os.environ.get("LST_WORKERS", "4"))
    threads = int(os.environ.get("LST_THREADS", "2"))
    memory = os.environ.get("LST_MEMORY", "4GB")

    log.info(
        "benchmark_starting",
        tile=tile_name,
        year=year,
        iterations=args.iterations,
        workers=workers,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up Dask
    cluster, client = setup_dask_cluster(workers, threads, memory)

    results = BenchmarkResults(tile=tile_name, year=year)

    endpoints = []
    if not args.es_only:
        endpoints.append((PLANETARY_COMPUTER, "planetary_computer"))
    if not args.pc_only:
        endpoints.append((EARTH_SEARCH, "earth_search"))

    try:
        for iteration in range(args.iterations):
            log.info("iteration_start", iteration=iteration + 1, total=args.iterations)

            for endpoint, name in endpoints:
                run = run_single_benchmark(
                    endpoint=endpoint,
                    endpoint_name=name,
                    tile_name=tile_name,
                    year=year,
                )
                results.runs.append(run)

        # Print results
        print(results.summary_table())

        # Save to JSON
        results_path = args.output_dir / "results.json"
        with results_path.open("w") as f:
            json.dump(
                {
                    "tile": results.tile,
                    "year": results.year,
                    "runs": [asdict(r) for r in results.runs],
                },
                f,
                indent=2,
            )
        log.info("results_saved", path=str(results_path))

        # Exit code: 0 if at least one run per endpoint succeeded
        pc_ok = any(r.success for r in results.runs if r.endpoint_name == "planetary_computer")
        es_ok = any(r.success for r in results.runs if r.endpoint_name == "earth_search")

        if args.pc_only:
            sys.exit(0 if pc_ok else 1)
        elif args.es_only:
            sys.exit(0 if es_ok else 1)
        else:
            sys.exit(0 if (pc_ok and es_ok) else 1)

    finally:
        client.close()
        cluster.close()
        log.info("dask_cluster_shutdown")


if __name__ == "__main__":
    main()
