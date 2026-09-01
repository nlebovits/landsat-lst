#!/usr/bin/env python3
"""Parameter sweep validation for landsat-lst pipeline.

Validates the P50 quantile fix and benchmarks chunk/worker configurations.

Usage:
    uv run python scripts/sweep_validation.py

Uses a tiny bbox (0.5 x 0.5 degrees) and 2-week date window for fast iteration.
Total runtime target: ~10 minutes for 6 configurations.
"""

from __future__ import annotations

import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
import structlog
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from odc.stac import stac_load

from landsat_lst.cog import cog_export
from landsat_lst.config import settings
from landsat_lst.encoding import encode_lst_uint16
from landsat_lst.qa import apply_qa_mask, convert_to_celsius

# Suppress benign warnings
warnings.filterwarnings("ignore", message=".*NotGeoreferencedWarning.*")
warnings.filterwarnings("ignore", message=".*Dataset has no geotransform.*")
warnings.filterwarnings("ignore", message=".*Sending large graph.*")
warnings.filterwarnings("ignore", message=".*LocalFileSystem storage is not safe.*")

# Configure structlog
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

# Sweep parameters
SWEEP_BBOX = (-75.25, 39.75, -74.75, 40.25)  # 0.5 x 0.5 degree Philadelphia sub-tile
SWEEP_DATE_START = "2024-06-01"
SWEEP_DATE_END = "2024-06-30"
# Scene-level cloud cover threshold uses settings.max_cloud_cover (default 100)

CHUNK_SIZES = [256, 512, 1024]
WORKER_COUNTS = [4, 8]
THREADS_PER_WORKER = 2
MEMORY_PER_WORKER = "4GB"


@dataclass
class SweepResult:
    """Results from a single sweep configuration."""

    chunk_size: int
    workers: int
    wall_time_secs: float
    scene_count: int
    p50_min: float
    p50_max: float
    p50_mean: float
    p95_min: float
    p95_max: float
    p95_mean: float
    peak_memory_mb: float
    p50_valid: bool  # True if values in realistic LST range
    success: bool
    error: str | None = None


def query_sweep_scenes() -> list:
    """Query STAC for scenes in the sweep bbox and date range."""
    catalog = pystac_client.Client.open(
        settings.stac_url,
        modifier=pc.sign_inplace,
    )

    search = catalog.search(
        collections=[settings.collection],
        bbox=SWEEP_BBOX,
        datetime=f"{SWEEP_DATE_START}/{SWEEP_DATE_END}",
        query={
            "eo:cloud_cover": {"lt": settings.max_cloud_cover},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
    )

    return list(search.items())


def load_scenes_with_chunks(items: list, chunk_size: int) -> xr.Dataset:
    """Load scenes with configurable chunk size."""
    return stac_load(
        items,
        bands=["lwir11", "qa_pixel"],
        crs=settings.crs,
        resolution=settings.source_resolution,
        chunks={"time": 10, "latitude": chunk_size, "longitude": chunk_size},
        groupby="solar_day",
        bbox=SWEEP_BBOX,
    )


def compute_composite(data: xr.Dataset) -> xr.Dataset:
    """Compute LST composite with P50 and P95."""
    masked = apply_qa_mask(data)
    lst = convert_to_celsius(masked["lwir11"])

    valid_mask = ~np.isnan(lst)
    qa_count = valid_mask.sum(dim="time").astype(np.int16)

    # The fix: quantile(0.5) instead of median()
    lst_p50 = lst.quantile(0.5, dim="time", skipna=True).drop_vars("quantile")
    lst_p95 = lst.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")

    lst_p50 = lst_p50.where(qa_count > 0, settings.nodata)
    lst_p95 = lst_p95.where(qa_count > 0, settings.nodata)

    return xr.Dataset(
        {
            "lst_p50": lst_p50.astype(np.float32),
            "lst_p95": lst_p95.astype(np.float32),
            "qa_count": qa_count,
        }
    )


def validate_p50(ds: xr.Dataset) -> tuple[bool, dict]:
    """Check if P50 values are realistic LST temperatures.

    The bug produced uniform value of 1 (encoded uint16).
    Valid: diverse values in physically plausible range (-20 to 80 C).
    """
    p50 = ds["lst_p50"].values
    valid_pixels = p50[~np.isnan(p50) & (p50 != settings.nodata)]

    if len(valid_pixels) == 0:
        return False, {"reason": "no valid pixels"}

    p50_min = float(np.min(valid_pixels))
    p50_max = float(np.max(valid_pixels))
    p50_mean = float(np.mean(valid_pixels))
    p50_std = float(np.std(valid_pixels))

    # Check for the bug: all values are uniform (std near zero)
    # The median() bug produced all 1s with zero variance
    if p50_std < 0.1:
        return False, {
            "reason": "uniform values (bug detected)",
            "min": p50_min,
            "max": p50_max,
            "std": p50_std,
        }

    # Check physically plausible range (LST can be extreme on hot surfaces)
    # -20 to 80 C covers water bodies, rooftops, and extreme cases
    physically_plausible = (p50_min >= -20.0) and (p50_max <= 80.0)

    # Also check that mean is in realistic range for mid-latitude summer
    mean_reasonable = 10.0 <= p50_mean <= 50.0

    valid = physically_plausible and mean_reasonable

    return valid, {
        "min": p50_min,
        "max": p50_max,
        "mean": p50_mean,
        "std": p50_std,
        "physically_plausible": physically_plausible,
        "mean_reasonable": mean_reasonable,
    }


def run_single_config(
    items: list,
    chunk_size: int,
    workers: int,
    output_dir: Path,
    config_idx: int,
) -> SweepResult:
    """Run pipeline with a single configuration."""
    log.info(
        "config_start",
        config=f"{config_idx}/6",
        chunk_size=chunk_size,
        workers=workers,
    )

    start_time = time.perf_counter()
    tracemalloc.start()

    cluster = None
    client = None

    try:
        # Set up Dask cluster
        cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=THREADS_PER_WORKER,
            memory_limit=MEMORY_PER_WORKER,
            dashboard_address=None,  # Disable dashboard for sweep
        )
        client = Client(cluster)

        # Load and compute
        data = load_scenes_with_chunks(items, chunk_size)
        composite = compute_composite(data)

        with ProgressBar(dt=5):
            composite = composite.compute()

        # Extract stats
        p50 = composite["lst_p50"].values
        p95 = composite["lst_p95"].values
        valid_p50 = p50[~np.isnan(p50) & (p50 != settings.nodata)]
        valid_p95 = p95[~np.isnan(p95) & (p95 != settings.nodata)]

        # Validate P50
        p50_valid, _ = validate_p50(composite)

        # Write the COG pair (separate dir per config) so the benchmark carries the
        # same write payload production does.
        config_output = output_dir / f"chunk{chunk_size}_workers{workers}"
        config_output.mkdir(parents=True, exist_ok=True)

        native = xr.Dataset(
            {
                "lst_p95": encode_lst_uint16(composite["lst_p95"]),
                "qa_count": composite["qa_count"].astype(np.uint8),
            }
        )
        cog_export(native, config_output / "lst_p95.tif", config_output / "qa_count.tif")

        # Get memory
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        wall_time = time.perf_counter() - start_time

        result = SweepResult(
            chunk_size=chunk_size,
            workers=workers,
            wall_time_secs=wall_time,
            scene_count=len(items),
            p50_min=float(np.min(valid_p50)) if len(valid_p50) > 0 else np.nan,
            p50_max=float(np.max(valid_p50)) if len(valid_p50) > 0 else np.nan,
            p50_mean=float(np.mean(valid_p50)) if len(valid_p50) > 0 else np.nan,
            p95_min=float(np.min(valid_p95)) if len(valid_p95) > 0 else np.nan,
            p95_max=float(np.max(valid_p95)) if len(valid_p95) > 0 else np.nan,
            p95_mean=float(np.mean(valid_p95)) if len(valid_p95) > 0 else np.nan,
            peak_memory_mb=peak_memory / 1024 / 1024,
            p50_valid=p50_valid,
            success=True,
        )

        log.info(
            "config_complete",
            chunk_size=chunk_size,
            workers=workers,
            wall_time=f"{wall_time:.1f}s",
            p50_valid=p50_valid,
            p50_range=f"[{result.p50_min:.1f}, {result.p50_max:.1f}]",
        )

        return result

    except Exception as e:
        tracemalloc.stop()
        wall_time = time.perf_counter() - start_time

        log.exception("config_failed", chunk_size=chunk_size, workers=workers)

        return SweepResult(
            chunk_size=chunk_size,
            workers=workers,
            wall_time_secs=wall_time,
            scene_count=len(items),
            p50_min=np.nan,
            p50_max=np.nan,
            p50_mean=np.nan,
            p95_min=np.nan,
            p95_max=np.nan,
            p95_mean=np.nan,
            peak_memory_mb=0,
            p50_valid=False,
            success=False,
            error=str(e),
        )

    finally:
        if client:
            client.close()
        if cluster:
            cluster.close()


def print_results_table(results: list[SweepResult]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 90)
    print(
        f"{'Chunks':>8} {'Workers':>8} {'Time (s)':>10} {'P50 Range':>18} "
        f"{'P95 Range':>18} {'Mem (MB)':>10} {'Valid':>8}"
    )
    print("-" * 90)

    for r in sorted(results, key=lambda x: x.wall_time_secs):
        p50_range = (
            f"[{r.p50_min:.1f}, {r.p50_max:.1f}]"
            if r.success and not np.isnan(r.p50_min)
            else "N/A"
        )
        p95_range = (
            f"[{r.p95_min:.1f}, {r.p95_max:.1f}]"
            if r.success and not np.isnan(r.p95_min)
            else "N/A"
        )
        valid_str = "✓" if r.p50_valid else "✗"
        status = valid_str if r.success else "FAIL"

        print(
            f"{r.chunk_size:>8} {r.workers:>8} {r.wall_time_secs:>10.1f} "
            f"{p50_range:>18} {p95_range:>18} {r.peak_memory_mb:>10.0f} {status:>8}"
        )

    print("=" * 90)

    # Summary
    successful = [r for r in results if r.success and r.p50_valid]
    if successful:
        fastest = min(successful, key=lambda x: x.wall_time_secs)
        print(f"\n✓ P50 fix validated: {len(successful)}/{len(results)} configs passed")
        print(
            f"✓ Fastest valid config: chunk_size={fastest.chunk_size}, "
            f"workers={fastest.workers} ({fastest.wall_time_secs:.1f}s)"
        )
        print(f"  P50 range: [{fastest.p50_min:.1f}, {fastest.p50_max:.1f}]°C")
        print(f"  P95 range: [{fastest.p95_min:.1f}, {fastest.p95_max:.1f}]°C")
    else:
        print("\n✗ No configurations passed validation!")
        failed = [r for r in results if not r.success]
        invalid = [r for r in results if r.success and not r.p50_valid]
        if failed:
            print(f"  {len(failed)} configs failed with errors")
        if invalid:
            print(f"  {len(invalid)} configs had invalid P50 values")

    print()


def main():
    """Run the parameter sweep."""
    output_dir = Path("output/sweep_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("LANDSAT-LST PARAMETER SWEEP VALIDATION")
    print(f"{'=' * 60}")
    print(f"Bbox: {SWEEP_BBOX} (0.5 x 0.5 degree Philadelphia)")
    print(f"Date range: {SWEEP_DATE_START} to {SWEEP_DATE_END}")
    print(f"Max cloud cover: {settings.max_cloud_cover}%")
    print(f"Chunk sizes: {CHUNK_SIZES}")
    print(f"Worker counts: {WORKER_COUNTS}")
    print(f"Configurations: {len(CHUNK_SIZES) * len(WORKER_COUNTS)}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'=' * 60}\n")

    # Query scenes once
    log.info("querying_stac")
    items = query_sweep_scenes()
    log.info("stac_query_complete", scene_count=len(items))

    if not items:
        print("ERROR: No scenes found in date range!")
        sys.exit(1)

    # Run all configurations
    results: list[SweepResult] = []
    config_idx = 0

    for chunk_size in CHUNK_SIZES:
        for workers in WORKER_COUNTS:
            config_idx += 1
            result = run_single_config(
                items=items,
                chunk_size=chunk_size,
                workers=workers,
                output_dir=output_dir,
                config_idx=config_idx,
            )
            results.append(result)

    # Print results
    print_results_table(results)

    # Exit code: 0 if any config validated, 1 otherwise
    any_valid = any(r.success and r.p50_valid for r in results)
    sys.exit(0 if any_valid else 1)


if __name__ == "__main__":
    main()
