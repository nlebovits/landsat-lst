#!/usr/bin/env python3
"""End-to-end test: Full tile on Coiled writing COGs to Source Cooperative S3.

This is the production integration test. It:
1. Runs a full 5 degree tile on Coiled workers (us-west-2)
2. Exports the LST and monthly-QA COGs and uploads both to Source Cooperative S3
3. Verifies both objects exist and that the LST COG opens over HTTPS
4. Prints the public URLs for QGIS and browser verification

Usage:
    # Ensure SSO session is active
    aws sso login --profile radiant-earth

    # Run E2E test (default: N40W075, 2021-2025)
    uv run python scripts/e2e_coiled_s3.py

    # Custom tile/window
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

# Public read host for the same objects, which is what a QGIS or gdalinfo user gets.
PUBLIC_HOST = "https://data.source.coop"


def item_keys(window_label: str, tile_name: str) -> tuple[str, str]:
    """S3 keys for a tile's two COG assets, in Portolan item layout.

    One collection per window, one item directory per tile, two assets per item
    (see ADR-009).
    """
    item_dir = f"{S3_PREFIX}/lst-p95-{window_label}/{tile_name}"
    return f"{item_dir}/lst_p95.tif", f"{item_dir}/qa_count.tif"


def process_tile_on_worker(tile_name: str, year: int, end_year: int | None) -> dict:
    """Process a single tile on a Coiled worker and upload its COGs. Returns a result dict."""
    import os as worker_os  # noqa: PLC0415
    import tempfile as worker_tempfile  # noqa: PLC0415
    import time as worker_time  # noqa: PLC0415
    from pathlib import Path as WorkerPath  # noqa: PLC0415

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
        "end_year": end_year,
        "success": False,
        "stages": {},
    }

    try:
        # Earth Search requester-pays
        worker_os.environ.setdefault("AWS_REQUEST_PAYER", "requester")

        import boto3  # noqa: PLC0415
        import xarray as xr  # noqa: PLC0415

        from landsat_lst.cog import cog_export  # noqa: PLC0415
        from landsat_lst.encoding import encode_lst_uint16  # noqa: PLC0415
        from landsat_lst.models import ProcessingJob  # noqa: PLC0415
        from landsat_lst.pipeline import process_tile  # noqa: PLC0415
        from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415

        tile = parse_tile_name(tile_name)
        job = ProcessingJob(tile=tile, year=year, end_year=end_year)
        worker_log.info("job_created", tile=tile_name, window=job.window_label, bbox=tile.bbox)

        # Run full pipeline (includes STAC query, load, composite, AND land mask)
        t0 = worker_time.time()
        composite = process_tile(job).compute()
        result["stages"]["process_tile"] = round(worker_time.time() - t0, 2)
        result["scene_count"] = composite.attrs.get("scene_count", 0)
        result["output_shape"] = dict(composite.sizes)
        worker_log.info("process_tile_complete", shape=dict(composite.sizes))

        # Encode to the published uint16 contract and write both COGs locally.
        t0 = worker_time.time()
        native = xr.Dataset(
            {
                "lst_p95": encode_lst_uint16(composite["lst_p95"]),
                "qa_count": composite["qa_count"],
            }
        )
        scratch = WorkerPath(worker_tempfile.mkdtemp(prefix="lst_e2e_"))
        lst_cog, qa_cog = cog_export(native, scratch / "lst_p95.tif", scratch / "qa_count.tif")
        result["stages"]["cog_export"] = round(worker_time.time() - t0, 2)
        worker_log.info(
            "cog_export_complete",
            lst_mb=round(lst_cog.stat().st_size / 1e6, 1),
            qa_mb=round(qa_cog.stat().st_size / 1e6, 1),
        )

        # Upload both assets. A tile is complete only when both objects land.
        t0 = worker_time.time()
        lst_key, qa_key = item_keys(job.window_label, tile_name)
        s3 = boto3.client("s3", region_name=S3_REGION)
        for path, key in ((lst_cog, lst_key), (qa_cog, qa_key)):
            s3.upload_file(str(path), S3_BUCKET, key)
            worker_log.info("s3_upload_complete", key=key)
        result["stages"]["s3_upload"] = round(worker_time.time() - t0, 2)
        result["lst_key"] = lst_key
        result["qa_key"] = qa_key
        result["window_label"] = job.window_label

        result["success"] = True
        result["total_secs"] = sum(result["stages"].values())

    except Exception as e:
        import traceback  # noqa: PLC0415

        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        worker_log.exception("pipeline_failed", error=str(e))

    return result


def verify_s3_output(lst_key: str, qa_key: str) -> dict:
    """Verify both COG objects exist on S3 and that the LST COG opens over HTTPS."""
    import boto3  # noqa: PLC0415
    import rasterio  # noqa: PLC0415

    log.info("verification_start", lst_key=lst_key, qa_key=qa_key)

    try:
        s3 = boto3.client("s3", region_name=S3_REGION)
        sizes = {}
        for key in (lst_key, qa_key):
            head = s3.head_object(Bucket=S3_BUCKET, Key=key)
            sizes[key.rsplit("/", 1)[-1]] = head["ContentLength"]

        # Open the LST COG the way a public consumer would, over HTTPS.
        lst_url = f"{PUBLIC_HOST}/{lst_key}"
        with rasterio.open(lst_url) as src:
            verification = {
                "success": True,
                "asset_bytes": sizes,
                "dtype": src.dtypes[0],
                "shape": (src.height, src.width),
                "nodata": src.nodata,
                "scales": src.scales,
                "offsets": src.offsets,
                "overviews": src.overviews(1),
            }

        log.info("verification_passed", **verification)
        return verification

    except Exception as e:
        log.exception("verification_failed", error=str(e))
        return {"success": False, "error": str(e)}


def print_access_urls(result: dict) -> None:
    """Print URLs for accessing the output in QGIS, gdalinfo, and Python."""
    lst_url = f"{PUBLIC_HOST}/{result['lst_key']}"
    qa_url = f"{PUBLIC_HOST}/{result['qa_key']}"

    print("\n" + "=" * 70)
    print("ACCESS URLS")
    print("=" * 70)

    print("\nCOG URLs (public read):")
    print(f"  {lst_url}")
    print(f"  {qa_url}")

    print("\nS3 paths:")
    print(f"  s3://{S3_BUCKET}/{result['lst_key']}")
    print(f"  s3://{S3_BUCKET}/{result['qa_key']}")

    print("\nSource Cooperative:")
    print("  https://source.coop/nlebovits/landsat-lst")

    print("\nQGIS:")
    print("  Layer > Add Layer > Add Raster Layer > Protocol: HTTP(S), and paste:")
    print(f"  {lst_url}")
    print("  Values decode to Celsius automatically from the embedded scale/offset.")

    print("\ngdalinfo:")
    print(f"  gdalinfo /vsicurl/{lst_url}")

    print("\nPython:")
    print("  import rioxarray")
    print(f'  da = rioxarray.open_rasterio("{lst_url}", masked=True)')
    print("  celsius = da * da.rio.scales[0] + da.rio.offsets[0]")

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


def print_dry_run(tile: str, window_label: str, aws_env: dict) -> None:
    """Print dry run information."""
    lst_key, qa_key = item_keys(window_label, tile)
    print("=" * 70)
    print("DRY RUN - E2E Test Configuration")
    print("=" * 70)
    print(f"\nTile: {tile}")
    print(f"Window: {window_label}")
    print(f"Output: s3://{S3_BUCKET}/{lst_key}")
    print(f"        s3://{S3_BUCKET}/{qa_key}")
    print("Coiled region: us-west-2")
    creds = "with session token" if "AWS_SESSION_TOKEN" in aws_env else "long-term"
    print(f"AWS credentials: {creds}")
    print("\nThis will:")
    print("  1. Spin up 2 Coiled workers in us-west-2")
    print("  2. Query Earth Search STAC for Landsat scenes")
    print("  3. Load and compute the pooled P95 composite")
    print("  4. Export the LST and monthly-QA COGs and upload both")
    print("  5. Verify both objects exist and the LST COG opens over HTTPS")
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
    print(f"Window: {result.get('window_label', result['year'])}")
    print(f"Wall time: {wall_time:.1f}s (includes Coiled spinup)")

    if result["success"]:
        print(f"\nScenes processed: {result['scene_count']}")
        print(f"Output shape: {result['output_shape']}")
        print(f"LST COG: s3://{S3_BUCKET}/{result['lst_key']}")
        print(f"QA COG:  s3://{S3_BUCKET}/{result['qa_key']}")

        print("\nStage Breakdown:")
        print("-" * 40)
        for stage, duration in result["stages"].items():
            print(f"  {stage:25s} {duration:7.2f}s")
        print("-" * 40)
        print(f"  {'Pipeline total':25s} {result['total_secs']:7.2f}s")

        if not skip_verify:
            print("\n" + "-" * 40)
            verification = verify_s3_output(result["lst_key"], result["qa_key"])
            if verification["success"]:
                print(f"Asset sizes (bytes): {verification['asset_bytes']}")
                print(f"LST dtype: {verification['dtype']} shape: {verification['shape']}")
                print(f"Nodata: {verification['nodata']}")
                print(f"Scale/offset: {verification['scales']} {verification['offsets']}")
                print(f"Overview levels: {verification['overviews']}")

        print_access_urls(result)
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        if "traceback" in result:
            print(f"\nTraceback:\n{result['traceback']}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="E2E test: Full tile on Coiled -> S3 COGs")
    parser.add_argument("--tile", "-t", default="N40W075", help="Tile (default: N40W075)")
    parser.add_argument("--year", "-y", type=int, default=2021, help="Start year (default: 2021)")
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year, omit for a single year (default: 2025)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show config without running")
    parser.add_argument("--skip-verify", action="store_true", help="Skip S3 verification")
    args = parser.parse_args()

    aws_env = get_aws_credentials()
    if aws_env is None:
        return

    # Add requester-pays config for usgs-landsat bucket access
    aws_env["AWS_REQUEST_PAYER"] = "requester"

    end_year = args.end_year if args.end_year and args.end_year != args.year else None
    window_label = f"{args.year}-{end_year}" if end_year else str(args.year)

    if args.dry_run:
        print_dry_run(args.tile, window_label, aws_env)
        return

    log.info("e2e_test_starting", tile=args.tile, window=window_label)

    # Use unique name to avoid reusing clusters with stale credentials
    cluster_name = f"e2e-{args.tile}-{int(time.time())}"

    @coiled.function(
        region="us-west-2",
        environ=aws_env,
        n_workers=2,  # Fixed 2 workers: I/O bound, more workers don't help
        keepalive="30 minutes",  # Longer keepalive for batch processing
        name=cluster_name,
    )
    def run_on_coiled(tile_name: str, year: int, end_year: int | None) -> dict:
        return process_tile_on_worker(tile_name, year, end_year)

    print(f"\nProcessing {args.tile} for {window_label} on Coiled...")
    t0 = time.time()
    result = run_on_coiled(args.tile, args.year, end_year)
    wall_time = time.time() - t0

    print_results(result, wall_time, args.skip_verify)


if __name__ == "__main__":
    main()
