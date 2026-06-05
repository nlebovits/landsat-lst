#!/usr/bin/env python3
"""End-to-end test: Full tile on Coiled writing to Source Cooperative S3.

This is the production integration test. It:
1. Runs a full 5° tile on Coiled workers (us-west-2)
2. Writes output to Source Cooperative S3 bucket
3. Verifies the Zarr store is readable
4. Provides URLs for QGIS and browser verification

Usage:
    # Ensure SSO session is active
    aws sso login --profile radiant-earth

    # Run E2E test (default: N40W075, 2024)
    uv run python scripts/e2e_coiled_s3.py

    # Custom tile/year
    uv run python scripts/e2e_coiled_s3.py --tile N35W120 --year 2023

    # Dry run (show what would be processed)
    uv run python scripts/e2e_coiled_s3.py --dry-run

Credentials are automatically retrieved from the radiant-earth SSO profile.
"""

from __future__ import annotations

import argparse
import os
import time

import coiled
import structlog

# Configure logging
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

# S3 bucket configuration (Source Cooperative)
S3_BUCKET = "us-west-2.opendata.source.coop"
S3_PREFIX = "nlebovits/landsat-lst"
S3_REGION = "us-west-2"


def process_tile_on_worker(tile_name: str, year: int) -> dict:
    """Process a single tile on a Coiled worker. Returns result dict."""
    import os as worker_os  # noqa: PLC0415
    import time as worker_time  # noqa: PLC0415

    import structlog as worker_structlog  # noqa: PLC0415

    # Configure worker logging
    worker_structlog.configure(
        processors=[
            worker_structlog.stdlib.add_log_level,
            worker_structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            worker_structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=worker_structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=worker_structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    worker_log = worker_structlog.get_logger()

    result = {
        "tile": tile_name,
        "year": year,
        "success": False,
        "stages": {},
    }

    try:
        # Set environment for S3 storage
        worker_os.environ["LST_STORAGE_BACKEND"] = "s3"
        worker_os.environ["LST_S3_BUCKET"] = "us-west-2.opendata.source.coop"
        worker_os.environ["LST_S3_PREFIX"] = "nlebovits/landsat-lst"
        worker_os.environ["LST_S3_REGION"] = "us-west-2"
        # Earth Search requester-pays
        worker_os.environ.setdefault("AWS_REQUEST_PAYER", "requester")

        from landsat_lst.models import ProcessingJob  # noqa: PLC0415
        from landsat_lst.pipeline import process_tile  # noqa: PLC0415
        from landsat_lst.storage import S3Storage  # noqa: PLC0415
        from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415
        from landsat_lst.zarr_writer import write_zarr  # noqa: PLC0415

        tile = parse_tile_name(tile_name)
        job = ProcessingJob(tile=tile, year=year)
        worker_log.info("job_created", tile=tile_name, year=year, bbox=tile.bbox)

        # Run full pipeline (includes STAC query, load, composite, AND land mask)
        t0 = worker_time.time()
        composite = process_tile(job)
        result["stages"]["process_tile"] = round(worker_time.time() - t0, 2)
        result["scene_count"] = composite.attrs.get("scene_count", 0)
        result["output_shape"] = dict(composite.sizes)
        worker_log.info("process_tile_complete", shape=dict(composite.sizes))

        # Write to S3
        t0 = worker_time.time()
        storage = S3Storage()
        zarr_path = storage.zarr_path(year, tile_name)

        # Pass credentials explicitly for Dask workers
        s3_opts = {
            "key": worker_os.environ.get("AWS_ACCESS_KEY_ID"),
            "secret": worker_os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "token": worker_os.environ.get("AWS_SESSION_TOKEN"),
        }
        write_zarr(composite, zarr_path, storage_options=s3_opts)
        result["stages"]["compute_and_write"] = round(worker_time.time() - t0, 2)
        result["zarr_path"] = zarr_path
        worker_log.info("s3_write_complete", path=zarr_path)

        result["success"] = True
        result["total_secs"] = sum(result["stages"].values())

    except Exception as e:
        import traceback  # noqa: PLC0415

        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        worker_log.exception("pipeline_failed", error=str(e))

    return result


def verify_s3_output(zarr_path: str) -> dict:
    """Verify the Zarr store exists and is readable on S3."""
    import xarray as xr  # noqa: PLC0415

    log.info("verification_start", path=zarr_path)

    try:
        # Open the Zarr store from S3
        ds = xr.open_zarr(zarr_path)

        verification = {
            "success": True,
            "variables": list(ds.data_vars),
            "shape": dict(ds.sizes),
            "lst_p95_dtype": str(ds["lst_p95"].dtype),
            "has_scale_factor": "lst_scale_factor" in ds["lst_p95"].attrs,
            "has_crs": "_CRS" in ds.attrs,
        }

        # Check data ranges
        lst_p95 = ds["lst_p95"].values
        verification["lst_p95_min"] = int(lst_p95[~(lst_p95 == -9999)].min())
        verification["lst_p95_max"] = int(lst_p95[~(lst_p95 == -9999)].max())
        verification["qa_count_max"] = int(ds["qa_count"].max().values)

        log.info("verification_passed", **verification)
        return verification

    except Exception as e:
        log.exception("verification_failed", error=str(e))
        return {"success": False, "error": str(e)}


def print_access_urls(zarr_path: str, tile_name: str, year: int) -> None:
    """Print URLs for accessing the output in QGIS and browser."""
    print("\n" + "=" * 70)
    print("ACCESS URLS")
    print("=" * 70)

    # S3 path
    print("\nS3 Zarr Path:")
    print(f"  {zarr_path}")

    # HTTPS URL for browser/QGIS
    https_url = zarr_path.replace("s3://", "https://s3.us-west-2.amazonaws.com/")
    print("\nHTTPS URL (browser):")
    print(f"  {https_url}")

    # Source Cooperative catalog URL (if available)
    print("\nSource Cooperative:")
    print("  https://source.coop/nlebovits/landsat-lst")

    # QGIS instructions
    print("\nQGIS Instructions:")
    print("  1. Install 'qgis-stac-plugin' if not installed")
    print("  2. Add Source Cooperative STAC catalog or use direct S3 path")
    print(f"  3. Navigate to: {year}/{tile_name}.zarr")
    print("  4. Load lst_p95 layer")

    # Python access
    print("\nPython Access:")
    print("  import xarray as xr")
    print(f'  ds = xr.open_zarr("{zarr_path}")')
    print('  ds["lst_p95"].plot()')

    print("=" * 70)


def get_aws_credentials(profile: str = "radiant-earth") -> dict | None:
    """Get AWS credentials from environment or SSO profile.

    First checks environment variables, then falls back to boto3 SSO credentials.
    This avoids the pitfall of `aws configure get` returning stale credentials
    from ~/.aws/credentials instead of fresh SSO tokens.
    """
    # Check environment first (for CI/CD or explicit exports)
    aws_env = {}
    for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"]:
        val = os.environ.get(key)
        if val:
            aws_env[key] = val

    if "AWS_ACCESS_KEY_ID" in aws_env:
        log.info("credentials_from_env")
        return aws_env

    # Fall back to boto3 SSO credentials
    try:
        import boto3  # noqa: PLC0415

        session = boto3.Session(profile_name=profile)
        creds = session.get_credentials()
        if creds is None:
            raise ValueError("No credentials found")

        frozen = creds.get_frozen_credentials()
        aws_env = {
            "AWS_ACCESS_KEY_ID": frozen.access_key,
            "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
        }
        if frozen.token:
            aws_env["AWS_SESSION_TOKEN"] = frozen.token

        log.info("credentials_from_sso", profile=profile)
        return aws_env

    except Exception as e:
        print(f"ERROR: Could not get AWS credentials: {e}")
        print(f"Run: aws sso login --profile {profile}")
        return None


def print_dry_run(tile: str, year: int, zarr_path: str, aws_env: dict) -> None:
    """Print dry run information."""
    print("=" * 70)
    print("DRY RUN - E2E Test Configuration")
    print("=" * 70)
    print(f"\nTile: {tile}")
    print(f"Year: {year}")
    print(f"Output: {zarr_path}")
    print("Coiled region: us-west-2")
    creds = "with session token" if "AWS_SESSION_TOKEN" in aws_env else "long-term"
    print(f"AWS credentials: {creds}")
    print("\nThis will:")
    print("  1. Spin up 2 Coiled workers in us-west-2")
    print("  2. Query Earth Search STAC for Landsat scenes")
    print("  3. Load and compute annual LST composite")
    print("  4. Write Zarr store to Source Cooperative S3")
    print("  5. Verify the output is readable")
    print("\nEstimated time: 10-15 minutes")
    print("Estimated cost: ~$0.40-0.60 (Coiled compute, 2 workers)")
    print("=" * 70)


def print_results(result: dict, wall_time: float, skip_verify: bool) -> None:
    """Print E2E test results."""
    print("\n" + "=" * 70)
    print("E2E TEST RESULTS")
    print("=" * 70)
    print(f"\nStatus: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Tile: {result['tile']}")
    print(f"Year: {result['year']}")
    print(f"Wall time: {wall_time:.1f}s (includes Coiled spinup)")

    if result["success"]:
        print(f"\nScenes processed: {result['scene_count']}")
        print(f"Output shape: {result['output_shape']}")
        print(f"Zarr path: {result['zarr_path']}")

        print("\nStage Breakdown:")
        print("-" * 40)
        for stage, duration in result["stages"].items():
            print(f"  {stage:25s} {duration:7.2f}s")
        print("-" * 40)
        print(f"  {'Pipeline total':25s} {result['total_secs']:7.2f}s")

        if not skip_verify:
            print("\n" + "-" * 40)
            verification = verify_s3_output(result["zarr_path"])
            if verification["success"]:
                print(f"Variables: {verification['variables']}")
                print(f"Shape: {verification['shape']}")
                lo, hi = verification["lst_p95_min"], verification["lst_p95_max"]
                print(f"LST P95 range: [{lo}, {hi}] (raw uint16)")
                print(f"QA count max: {verification['qa_count_max']}")

        print_access_urls(result["zarr_path"], result["tile"], result["year"])
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        if "traceback" in result:
            print(f"\nTraceback:\n{result['traceback']}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="E2E test: Full tile on Coiled → S3")
    parser.add_argument("--tile", "-t", default="N40W075", help="Tile (default: N40W075)")
    parser.add_argument("--year", "-y", type=int, default=2024, help="Year (default: 2024)")
    parser.add_argument("--dry-run", action="store_true", help="Show config without running")
    parser.add_argument("--skip-verify", action="store_true", help="Skip S3 verification")
    args = parser.parse_args()

    aws_env = get_aws_credentials()
    if aws_env is None:
        return

    # Add requester-pays config for usgs-landsat bucket access
    aws_env["AWS_REQUEST_PAYER"] = "requester"

    zarr_path = f"s3://{S3_BUCKET}/{S3_PREFIX}/{args.year}/{args.tile}.zarr"

    if args.dry_run:
        print_dry_run(args.tile, args.year, zarr_path, aws_env)
        return

    log.info("e2e_test_starting", tile=args.tile, year=args.year, output=zarr_path)

    # Use unique name to avoid reusing clusters with stale credentials
    cluster_name = f"e2e-{args.tile}-{int(time.time())}"

    @coiled.function(
        region="us-west-2",
        environ=aws_env,
        n_workers=2,  # Fixed 2 workers: I/O bound, more workers don't help
        keepalive="30 minutes",  # Longer keepalive for batch processing
        name=cluster_name,
    )
    def run_on_coiled(tile_name: str, year: int) -> dict:
        return process_tile_on_worker(tile_name, year)

    print(f"\nProcessing {args.tile} for {args.year} on Coiled...")
    t0 = time.time()
    result = run_on_coiled(args.tile, args.year)
    wall_time = time.time() - t0

    print_results(result, wall_time, args.skip_verify)


if __name__ == "__main__":
    main()
