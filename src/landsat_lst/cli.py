"""Command-line interface for Landsat LST pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from rashid import Report

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """Landsat Land Surface Temperature annual composites."""


def _build_jobs(year: int | None, end_year: int | None, tiles: tuple[str, ...]) -> list:
    """Resolve CLI options into the job list to run.

    Omitting ``--year`` selects the production window. Passing ``--year`` alone
    keeps the single-year behavior that predates multi-year windows.
    """
    from landsat_lst.job import DEFAULT_WINDOW, generate_jobs
    from landsat_lst.models import ProcessingJob
    from landsat_lst.tiling import LAND_TILES, parse_tile_name

    if year is None:
        year, end_year = DEFAULT_WINDOW

    if tiles:
        for tile in tiles:
            if tile not in LAND_TILES:
                console.print(f"[red]Warning: {tile} is not in the land tiles set[/red]")
        return [
            ProcessingJob(tile=parse_tile_name(tile), year=year, end_year=end_year)
            for tile in tiles
        ]

    if end_year:
        return generate_jobs(window=(year, end_year))
    return generate_jobs([year])


@main.command()
@click.option("--year", "-y", type=int, help="Start year. Omit to use the default window.")
@click.option("--end-year", type=int, help="End year (inclusive) for a multi-year window")
@click.option(
    "--tile",
    "-t",
    "tiles",
    type=str,
    multiple=True,
    help="Tile to process (e.g., N40W075); repeatable",
)
@click.option("--dry-run", is_flag=True, help="Show what would be processed without running")
@click.option("--force", "-f", is_flag=True, help="Reprocess even if the COGs exist")
@click.option(
    "--distributed",
    "-d",
    is_flag=True,
    help="Submit to Coiled Batch, one VM per tile, instead of running locally",
)
@click.option(
    "--wait",
    is_flag=True,
    help="With --distributed, block until the run finishes and reconcile it",
)
@click.option(
    "--run-id",
    default=None,
    help="Run token. Set by the batch task so each VM reports into the same run.",
)
@click.option("--limit", type=int, help="Process at most N tiles from the job list")
def process(
    *,
    year: int | None,
    end_year: int | None,
    tiles: tuple[str, ...],
    dry_run: bool,
    force: bool,
    distributed: bool,
    wait: bool,
    run_id: str | None,
    limit: int | None,
) -> None:
    """Process Landsat data to COG composites.

    With no --year, processes the production window (2021-2025). Passing --year
    alone builds a single-year composite; add --end-year for a custom window.
    With --distributed, tiles are submitted to Coiled Batch and this command
    returns immediately; run `landsat-lst reconcile RUN_ID` afterwards to write
    the manifest, or pass --wait to do both in one go.
    """
    jobs = _build_jobs(year, end_year, tiles)
    if limit is not None:
        jobs = jobs[:limit]
    console.print(f"[bold]Processing window {jobs[0].window_label}[/bold]")
    if tiles:
        console.print(f"  Tiles: {', '.join(tiles)}")
    else:
        console.print(f"  Tiles: {len(jobs)} land tiles")

    if force:
        console.print("  [yellow]Force mode: reprocessing existing COGs[/yellow]")

    if dry_run:
        console.print("[yellow]Dry run - no processing performed[/yellow]")
        mode = "on Coiled" if distributed else "locally"
        for job in jobs[:5]:
            console.print(f"    Would process {mode}: {job.tile.name} {job.window_label}")
        if len(jobs) > 5:
            console.print(f"    ... and {len(jobs) - 5} more")
        return

    if distributed:
        _process_distributed(jobs, force=force, run_id=run_id, wait=wait)
    else:
        _process_local(jobs, force=force, run_id=run_id)


def _process_local(jobs: list, *, force: bool, run_id: str | None = None) -> None:
    """Process jobs sequentially in this process with a progress bar.

    This is also the code path a Coiled Batch VM runs, with one tile and a
    ``run_id``. The progress bar renders to the task's log; the run record it
    writes for that tile is what reconciliation reads.
    """
    from landsat_lst.job import process_tile_job

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing {len(jobs)} tiles...", total=len(jobs))

        completed = 0
        skipped = 0
        failed = 0

        for job in jobs:
            progress.update(task, description=f"Processing {job.tile.name}...")
            result = process_tile_job(job, force=force, run_id=run_id)

            if result.status == "completed":
                completed += 1
            elif result.status == "skipped":
                skipped += 1
            else:
                failed += 1
                console.print(f"[red]Failed: {job.tile.name} - {result.error}[/red]")

            progress.advance(task)

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Completed: [green]{completed}[/green]")
    console.print(f"  Skipped: [yellow]{skipped}[/yellow]")
    if failed:
        console.print(f"  Failed: [red]{failed}[/red]")
        # A batch task must exit non-zero on failure, or Coiled's retries and
        # the task state a manifest falls back on both report success.
        raise SystemExit(1)


def _print_results(results: list) -> None:
    """Render reconciled per-tile outcomes."""
    completed = sum(1 for r in results if r.status == "completed")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = [r for r in results if r.status == "failed"]

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Completed: [green]{completed}[/green]")
    console.print(f"  Skipped: [yellow]{skipped}[/yellow]")
    if failed:
        console.print(f"  Failed: [red]{len(failed)}[/red]")
        for r in failed:
            console.print(f"    [red]{r.job.tile.name}: {r.error}[/red]")


def _process_distributed(jobs: list, *, force: bool, run_id: str | None, wait: bool) -> None:
    """Submit jobs to Coiled Batch and hand back the run token.

    No per-tile progress bar and no held connection: the Coiled dashboard is
    the live progress UI, and the manifest built by `reconcile` is the durable
    record. Closing this shell does not touch the run.
    """
    from landsat_lst.batch import reconcile_run, submit_batch, wait_for_batch
    from landsat_lst.config import settings

    console.print(
        f"  Region: {settings.coiled_region}  "
        f"VMs: up to {settings.coiled_max_workers}  "
        f"Types: {', '.join(settings.coiled_vm_types)}"
    )

    submission = submit_batch(jobs, force=force, run_id=run_id)
    console.print(f"  Run id: [bold]{submission.run_id}[/bold]")

    if submission.cluster_id is None:
        console.print("  [yellow]Every tile is already complete; no cluster started.[/yellow]")
        _print_results(reconcile_run(submission.run_id))
        return

    console.print(f"  Cluster: {submission.cluster_id}  Tasks: {len(submission.submitted_tiles)}")
    console.print(f"  Progress: {submission.dashboard_url}")

    if not wait:
        console.print(
            f"\n  Submitted. This shell is free to close.\n"
            f"  When it finishes: [bold]landsat-lst reconcile {submission.run_id}[/bold]"
        )
        return

    console.print("\n  Waiting for the run to finish (Ctrl-C is safe; the run continues)...")
    state = wait_for_batch(submission.run_id)
    console.print(f"  Final job state: {state or 'unknown'}")
    _print_results(reconcile_run(submission.run_id))
    console.print(f"  Manifest: {settings.manifest_dir / (submission.run_id + '.json')}")


@main.command()
@click.argument("run_id")
def reconcile(run_id: str) -> None:
    """Build the run manifest for a submitted batch run.

    Reads the COG listing for what finished, the per-tile run records for
    duration, scene count, and peak memory, and Coiled's task states for why a
    tile without output has none. Safe to run more than once, and safe to run
    while tasks are still going: tiles with no COGs yet report as failed.
    """
    from landsat_lst.batch import reconcile_run
    from landsat_lst.config import settings

    console.print(f"[bold]Reconciling run {run_id}[/bold]")
    results = reconcile_run(run_id)
    _print_results(results)
    console.print(f"  Manifest: {settings.manifest_dir / (run_id + '.json')}")


@main.command()
def list_tiles() -> None:
    """List all tiles in the global grid."""
    from landsat_lst.tiling import generate_global_tiles

    tiles = list(generate_global_tiles())
    console.print(f"[bold]Total tiles: {len(tiles)}[/bold]")

    for t in tiles[:10]:
        console.print(f"  {t.name}: {t.bbox}")

    if len(tiles) > 10:
        console.print(f"  ... and {len(tiles) - 10} more")


@main.command()
@click.argument("tile_name")
def tile_info(tile_name: str) -> None:
    """Show information about a specific tile."""
    console.print(f"[bold]Tile: {tile_name}[/bold]")
    console.print("[red]Not yet implemented[/red]")


@main.group()
def catalog() -> None:
    """Build and validate the published STAC catalog."""


@catalog.command("build")
@click.option("--source", required=True, help="Directory or s3:// prefix of finished COGs")
@click.option("--out", default="./catalog", help="Directory to write the catalog into")
@click.option("--window", default=None, help="Observation window label, e.g. 2021-2025")
@click.option("--tiles", default=None, help="Comma-separated tile names to include")
@click.option(
    "--thumbnail",
    default=None,
    help="PNG to register as the thumbnail, instead of rendering one from the tiles",
)
@click.option(
    "--metadata-only",
    is_flag=True,
    help=(
        "Write a JSON-only staging tree: read every COG header but do not "
        "copy or download the COGs beside their items. For s3:// sources "
        "whose COGs already sit at their published paths."
    ),
)
def catalog_build(
    source: str,
    out: str,
    *,
    window: str | None,
    tiles: str | None,
    thumbnail: str | None,
    metadata_only: bool,
) -> None:
    """Build a Portolan-compliant STAC catalog from finished per-tile COGs."""
    from landsat_lst.catalog import build_catalog
    from landsat_lst.catalog.spec import DEFAULT_SPEC, spec_for_window

    spec = DEFAULT_SPEC if window is None else spec_for_window(window)
    wanted = tuple(name.strip() for name in tiles.split(",")) if tiles else None
    console.print(f"[bold]Building catalog for {spec.window}[/bold] from {source}")
    root = build_catalog(
        source,
        out,
        spec,
        tiles=wanted,
        thumbnail=thumbnail,
        place_assets=not metadata_only,
    )
    console.print(f"  Wrote [green]{root}[/green]")
    if metadata_only:
        console.print(
            "  [yellow]Metadata-only tree: validator byte checks will skip "
            "the COGs. Sample real tiles locally for byte coverage.[/yellow]"
        )


def _print_report(report: Report, unaccepted: set[str]) -> None:
    """Render a validation report for a person reading a terminal."""
    for finding in report.errors:
        console.print(f"[red]{finding.rule_id}[/red] {finding.path}: {finding.message}")
    for summary in report.by_rule():
        colour = "red" if summary.severity.value == "error" else "yellow"
        console.print(
            f"  [{colour}]{summary.rule_id}[/{colour}] x{summary.count}  {summary.description}"
        )
    if unaccepted:
        console.print(f"[red]Warnings outside the accepted baseline: {sorted(unaccepted)}[/red]")


@catalog.command("validate")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit the full report as JSON")
@click.option("--live", is_flag=True, help="Also probe the hosting server (requires network)")
@click.option("--live-base-url", default=None, help="https URL the catalog root is published under")
def catalog_validate(path: Path, as_json: bool, live: bool, live_base_url: str | None) -> None:
    """Validate a built catalog against the Portolan profile.

    The default run is offline. --live adds the hosting pass, which probes the
    published server for range support, CORS, and Content-Length. The catalog's
    hrefs are relative, so that pass needs --live-base-url to know what to probe.
    """
    import json as json_module

    from landsat_lst.catalog.validation import unaccepted_warnings, validate_catalog

    if live and live_base_url is None:
        msg = "--live needs --live-base-url: the catalog's asset hrefs are relative"
        raise click.UsageError(msg)
    # --live-base-url alone turns the pass on too, so naming a URL is never a
    # no-op that quietly reports an offline verdict.
    report = validate_catalog(path, live_base_url=live_base_url)
    unaccepted = unaccepted_warnings(report)
    if as_json:
        payload = report.to_dict()
        payload["unaccepted_warnings"] = sorted(unaccepted)
        click.echo(json_module.dumps(payload, indent=2))
    else:
        _print_report(report, unaccepted)
    if report.errors or unaccepted:
        raise SystemExit(1)


@catalog.command("publish")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--remote", required=True, help="s3://bucket/prefix/ to publish the tree under")
@click.option("--dry-run", is_flag=True, help="Print the upload plan without uploading")
@click.option("--profile", default=None, help="AWS named profile to authenticate with")
def catalog_publish(path: Path, remote: str, dry_run: bool, profile: str | None) -> None:
    """Sync a built catalog tree to the S3 prefix it is published under.

    Each object is uploaded with the media type its extension declares. An
    asset whose remote size already matches is skipped; JSON and markdown are
    always re-sent, because an equal-sized metadata edit is realistic and they
    cost kilobytes.
    """
    from landsat_lst.catalog.publish import publish_catalog

    console.print(f"[bold]Publishing {path}[/bold] to {remote}")
    summary = publish_catalog(path, remote, dry_run=dry_run, profile=profile)

    if dry_run:
        for upload in summary.planned:
            console.print(f"    upload  {upload}")
        console.print(
            f"[yellow]Dry run - nothing uploaded.[/yellow] Would upload "
            f"{len(summary.planned)} objects ({summary.planned_bytes} bytes), "
            f"skipping {summary.skipped_count} unchanged."
        )
        return

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Uploaded: [green]{summary.uploaded_count}[/green]")
    console.print(f"  Skipped:  [yellow]{summary.skipped_count}[/yellow]")
    console.print(f"  Bytes:    {summary.uploaded_bytes}")


if __name__ == "__main__":
    main()
