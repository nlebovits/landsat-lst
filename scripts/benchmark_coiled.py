#!/usr/bin/env python3
"""Benchmark Earth Search vs Planetary Computer on Coiled (parallel, multi-iteration).

Runs N iterations of each endpoint in parallel on Coiled workers in us-west-2.
Reports statistics, failure modes, and estimated egress costs.

Usage:
    # Set AWS credentials for workers
    export AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile radiant-earth)
    export AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile radiant-earth)
    export AWS_SESSION_TOKEN=$(aws configure get aws_session_token --profile radiant-earth)

    # Run benchmark (default 10 iterations)
    uv run python scripts/benchmark_coiled.py

    # Or specify iterations
    uv run python scripts/benchmark_coiled.py --iterations 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

import coiled

from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Callable

STAC_PLANETARY_COMPUTER = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"

BBOX = (-75.2, 39.9, -75.0, 40.1)  # Small area near Philadelphia
DATETIME = "2024-06-01/2024-06-30"  # 1 month


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    endpoint: str
    iteration: int
    success: bool
    query_secs: float | None = None
    compute_secs: float | None = None
    total_secs: float | None = None
    scene_count: int | None = None
    bytes_read: int | None = None
    error: str | None = None
    retries: int = 0


def benchmark_on_worker(
    endpoint: str,
    use_pc_signing: bool,
    iteration: int,
) -> dict:
    """Run benchmark for a single endpoint. Executes on Coiled worker."""
    import time as time_mod  # noqa: PLC0415

    import planetary_computer as pc  # noqa: PLC0415
    import pystac_client  # noqa: PLC0415
    from odc.stac import stac_load  # noqa: PLC0415

    result = {
        "endpoint": "planetary_computer" if use_pc_signing else "earth_search",
        "iteration": iteration,
        "success": False,
        "retries": 0,
    }

    # Configure requester-pays for Earth Search
    if not use_pc_signing:
        os.environ.setdefault("AWS_REQUEST_PAYER", "requester")

    modifier = pc.sign_inplace if use_pc_signing else None

    try:
        # Query STAC
        t0 = time_mod.time()
        catalog = pystac_client.Client.open(endpoint, modifier=modifier)
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=BBOX,
            datetime=DATETIME,
            query={"eo:cloud_cover": {"lt": settings.max_cloud_cover}},
        )
        items = list(search.items())
        query_time = time_mod.time() - t0

        if not items:
            result["error"] = "No scenes found"
            return result

        result["query_secs"] = round(query_time, 3)
        result["scene_count"] = len(items)

        # Load and compute (limit to 3 scenes for speed)
        t0 = time_mod.time()
        data = stac_load(
            items[:3],
            bands=["lwir11"],
            bbox=BBOX,
            crs="EPSG:4326",
            resolution=0.0003,
        )
        computed = data.compute()
        compute_time = time_mod.time() - t0

        result["compute_secs"] = round(compute_time, 3)
        result["total_secs"] = round(query_time + compute_time, 3)

        # Estimate bytes read (lwir11 is uint16 = 2 bytes per pixel)
        n_pixels = 1
        for dim_size in computed.sizes.values():
            n_pixels *= dim_size
        result["bytes_read"] = n_pixels * 2  # uint16

        result["success"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def run_parallel_benchmarks(
    run_fn: Callable,
    iterations: int,
) -> tuple[list[BenchmarkResult], list[BenchmarkResult]]:
    """Run benchmarks for both endpoints in parallel."""
    pc_results: list[BenchmarkResult] = []
    es_results: list[BenchmarkResult] = []

    # Submit all jobs in parallel
    futures = {}
    with ThreadPoolExecutor(max_workers=iterations * 2) as executor:
        # Submit PC benchmarks
        for i in range(iterations):
            future = executor.submit(
                run_fn,
                STAC_PLANETARY_COMPUTER,
                True,  # use_pc_signing
                i,
            )
            futures[future] = ("pc", i)

        # Submit ES benchmarks
        for i in range(iterations):
            future = executor.submit(
                run_fn,
                STAC_EARTH_SEARCH,
                False,  # use_pc_signing
                i,
            )
            futures[future] = ("es", i)

        # Collect results as they complete
        for future in as_completed(futures):
            endpoint_type, iteration = futures[future]
            try:
                raw = future.result()
                result = BenchmarkResult(
                    endpoint=raw.get("endpoint", endpoint_type),
                    iteration=raw.get("iteration", iteration),
                    success=raw.get("success", False),
                    query_secs=raw.get("query_secs"),
                    compute_secs=raw.get("compute_secs"),
                    total_secs=raw.get("total_secs"),
                    scene_count=raw.get("scene_count"),
                    bytes_read=raw.get("bytes_read"),
                    error=raw.get("error"),
                    retries=raw.get("retries", 0),
                )
            except Exception as e:
                result = BenchmarkResult(
                    endpoint=endpoint_type,
                    iteration=iteration,
                    success=False,
                    error=f"Future error: {type(e).__name__}: {e}",
                )

            if endpoint_type == "pc":
                pc_results.append(result)
            else:
                es_results.append(result)

            status = "✓" if result.success else f"✗ {result.error}"
            print(f"  [{endpoint_type.upper()}][{iteration}] {status}")

    return pc_results, es_results


def compute_stats(results: list[BenchmarkResult]) -> dict:
    """Compute statistics from benchmark results."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    stats = {
        "total_runs": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) if results else 0,
        "errors": [r.error for r in failed if r.error],
    }

    if successful:
        times = [r.total_secs for r in successful if r.total_secs is not None]
        query_times = [r.query_secs for r in successful if r.query_secs is not None]
        compute_times = [r.compute_secs for r in successful if r.compute_secs is not None]
        bytes_list = [r.bytes_read for r in successful if r.bytes_read is not None]

        if times:
            stats["total_mean"] = statistics.mean(times)
            stats["total_std"] = statistics.stdev(times) if len(times) > 1 else 0
            stats["total_min"] = min(times)
            stats["total_max"] = max(times)

        if query_times:
            stats["query_mean"] = statistics.mean(query_times)

        if compute_times:
            stats["compute_mean"] = statistics.mean(compute_times)

        if bytes_list:
            stats["bytes_mean"] = statistics.mean(bytes_list)
            stats["bytes_total"] = sum(bytes_list)

    return stats


def print_results(
    pc_stats: dict,
    es_stats: dict,
    iterations: int,
    wall_time: float,
) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS (Coiled us-west-2)")
    print(f"Iterations: {iterations} per endpoint | Wall time: {wall_time:.1f}s")
    print("=" * 70)

    # Success rates
    print(f"\n{'Success Rate':<25} {'Planetary Computer':>20} {'Earth Search':>20}")
    print("-" * 70)
    pc_rate = f"{pc_stats['successful']}/{pc_stats['total_runs']}"
    es_rate = f"{es_stats['successful']}/{es_stats['total_runs']}"
    print(f"{'Runs (success/total)':<25} {pc_rate:>20} {es_rate:>20}")

    # Timing stats
    if "total_mean" in pc_stats and "total_mean" in es_stats:
        print(f"\n{'Timing (seconds)':<25} {'Planetary Computer':>20} {'Earth Search':>20}")
        print("-" * 70)

        pc_mean = f"{pc_stats['total_mean']:.2f} ± {pc_stats['total_std']:.2f}"
        es_mean = f"{es_stats['total_mean']:.2f} ± {es_stats['total_std']:.2f}"
        print(f"{'Total (mean ± std)':<25} {pc_mean:>20} {es_mean:>20}")

        print(
            f"{'Query (mean)':<25} {pc_stats.get('query_mean', 0):.2f}s{' ':>14} {es_stats.get('query_mean', 0):.2f}s"
        )
        print(
            f"{'Compute (mean)':<25} {pc_stats.get('compute_mean', 0):.2f}s{' ':>14} {es_stats.get('compute_mean', 0):.2f}s"
        )
        print(f"{'Min':<25} {pc_stats['total_min']:.2f}s{' ':>14} {es_stats['total_min']:.2f}s")
        print(f"{'Max':<25} {pc_stats['total_max']:.2f}s{' ':>14} {es_stats['total_max']:.2f}s")

        # Speedup
        print("-" * 70)
        if es_stats["total_mean"] < pc_stats["total_mean"]:
            speedup = pc_stats["total_mean"] / es_stats["total_mean"]
            print(f"Earth Search is {speedup:.2f}x faster (mean)")
        else:
            speedup = es_stats["total_mean"] / pc_stats["total_mean"]
            print(f"Planetary Computer is {speedup:.2f}x faster (mean)")

    # Egress cost estimate
    print(f"\n{'Egress Cost Estimate':<25}")
    print("-" * 70)
    if "bytes_total" in es_stats:
        es_gb = es_stats["bytes_total"] / (1024**3)
        # Same-region S3 to EC2 is $0.00/GB
        # Cross-region would be $0.02/GB, cross-cloud (PC) varies
        print(f"Earth Search: {es_gb:.3f} GB transferred (us-west-2 → us-west-2 = $0.00)")
    if "bytes_total" in pc_stats:
        pc_gb = pc_stats["bytes_total"] / (1024**3)
        # PC egress is free (Microsoft covers it)
        print(f"Planetary Computer: {pc_gb:.3f} GB transferred (free egress via PC)")
    print("Note: Both endpoints have $0 egress for this use case.")

    # Errors
    if pc_stats["errors"] or es_stats["errors"]:
        print(f"\n{'Errors':<25}")
        print("-" * 70)
        for err in pc_stats["errors"]:
            print(f"  [PC] {err}")
        for err in es_stats["errors"]:
            print(f"  [ES] {err}")

    print("=" * 70)


def estimate_cost(iterations: int) -> None:
    """Print cost estimate without running benchmark."""
    # Based on previous benchmark: PC ~6s, ES ~1.4s per run
    # Coiled pricing: ~$0.05/worker-hour for t3.medium (default)
    workers = iterations * 2
    pc_time_per_run = 6.0  # seconds (from previous benchmark)
    es_time_per_run = 1.4  # seconds

    # Total worker-seconds (runs are parallel, but each needs a worker)
    # In practice, max concurrent = workers, each runs once
    pc_worker_hours = (iterations * pc_time_per_run) / 3600
    es_worker_hours = (iterations * es_time_per_run) / 3600

    # Cluster spinup overhead (~30s per unique worker)
    spinup_overhead_hours = (workers * 30) / 3600

    total_worker_hours = pc_worker_hours + es_worker_hours + spinup_overhead_hours

    # Coiled cost estimate (t3.medium ~$0.05/hr, but billed per second)
    cost_per_hour = 0.05
    estimated_cost = total_worker_hours * cost_per_hour

    # Wall time estimate (parallel execution)
    # Spinup happens once, then all workers run in parallel
    # Max runtime = max(PC, ES) since they run concurrently
    wall_time_estimate = 30 + max(pc_time_per_run, es_time_per_run)  # spinup + longest run

    print("=" * 60)
    print("DRY RUN - Cost Estimate")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Iterations per endpoint: {iterations}")
    print(f"  Total benchmark runs: {iterations * 2}")
    print(f"  Max concurrent workers: {workers}")
    print("\nTime estimates (based on previous benchmark):")
    print(f"  PC mean time: {pc_time_per_run:.1f}s/run")
    print(f"  ES mean time: {es_time_per_run:.1f}s/run")
    print("  Cluster spinup: ~30s")
    print(f"  Estimated wall time: ~{wall_time_estimate:.0f}s ({wall_time_estimate / 60:.1f} min)")
    print(f"\nCost estimate (Coiled t3.medium @ ${cost_per_hour}/worker-hr):")
    print(f"  PC worker-hours: {pc_worker_hours:.4f}")
    print(f"  ES worker-hours: {es_worker_hours:.4f}")
    print(f"  Spinup overhead: {spinup_overhead_hours:.4f}")
    print(f"  Total worker-hours: {total_worker_hours:.4f}")
    print(f"  Estimated cost: ${estimated_cost:.4f}")
    print("\nEgress cost: $0.00 (same-region S3 for ES, free via PC)")
    print(f"\nTotal estimated cost: ${estimated_cost:.4f}")
    print("=" * 60)
    print("\nTo run the benchmark, remove --dry-run flag.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark STAC endpoints on Coiled")
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=10,
        help="Number of iterations per endpoint (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate cost without running benchmark",
    )
    args = parser.parse_args()

    if args.dry_run:
        estimate_cost(args.iterations)
        return

    # Get AWS credentials from environment
    aws_env = {}
    for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]:
        val = os.environ.get(key)
        if val:
            aws_env[key] = val

    if "AWS_ACCESS_KEY_ID" not in aws_env:
        print("WARNING: AWS credentials not found in environment.")
        print("Earth Search may fail. Run these first:")
        print(
            "  export AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile radiant-earth)"
        )
        print(
            "  export AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile radiant-earth)"
        )
        print(
            "  export AWS_SESSION_TOKEN=$(aws configure get aws_session_token --profile radiant-earth)"
        )
        print()

    print(f"Starting benchmark: {args.iterations} iterations per endpoint")
    print(
        f"AWS credentials: {'with session token' if 'AWS_SESSION_TOKEN' in aws_env else 'long-term' if aws_env else 'NONE'}"
    )
    print("Spinning up Coiled workers in us-west-2...\n")

    # Create Coiled function
    @coiled.function(
        region="us-west-2",
        environ=aws_env,
        keepalive="5 minutes",
    )
    def run_benchmark(endpoint: str, use_pc_signing: bool, iteration: int) -> dict:
        return benchmark_on_worker(endpoint, use_pc_signing, iteration)

    # Run benchmarks
    t0 = time.time()
    pc_results, es_results = run_parallel_benchmarks(run_benchmark, args.iterations)
    wall_time = time.time() - t0

    # Compute and print statistics
    pc_stats = compute_stats(pc_results)
    es_stats = compute_stats(es_results)
    print_results(pc_stats, es_stats, args.iterations, wall_time)


if __name__ == "__main__":
    main()
