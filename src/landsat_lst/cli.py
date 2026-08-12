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


def _build_jobs(year: int | None, end_year: int | None, tile: str | None) -> list:
    """Resolve CLI options into the job list to run.

    Omitting ``--year`` selects the production window. Passing ``--year`` alone
    keeps the single-year behavior that predates multi-year windows.
    """
    from landsat_lst.job import DEFAULT_WINDOW, generate_jobs
    from landsat_lst.models import ProcessingJob
    from landsat_lst.tiling import LAND_TILES, parse_tile_name

    if year is None:
        year, end_year = DEFAULT_WINDOW

    if tile:
        if tile not in LAND_TILES:
            console.print(f"[red]Warning: {tile} is not in the land tiles set[/red]")
        return [ProcessingJob(tile=parse_tile_name(tile), year=year, end_year=end_year)]

    if end_year:
        return generate_jobs(window=(year, end_year))
    return generate_jobs([year])


@main.command()
@click.option("--year", "-y", type=int, help="Start year. Omit to use the default window.")
@click.option("--end-year", type=int, help="End year (inclusive) for a multi-year window")
@click.option("--tile", "-t", type=str, help="Specific tile to process (e.g., N40W075)")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without running")
@click.option("--force", "-f", is_flag=True, help="Reprocess even if the COGs exist")
def process(
    year: int | None,
    end_year: int | None,
    tile: str | None,
    dry_run: bool,
    force: bool,
) -> None:
    """Process Landsat data to COG composites.

    With no --year, processes the production window (2021-2025). Passing --year
    alone builds a single-year composite; add --end-year for a custom window.
    """
    from landsat_lst.job import process_tile_job

    jobs = _build_jobs(year, end_year, tile)
    console.print(f"[bold]Processing window {jobs[0].window_label}[/bold]")
    if tile:
        console.print(f"  Tile: {tile}")
    else:
        console.print(f"  Tiles: {len(jobs)} land tiles")

    if force:
        console.print("  [yellow]Force mode: reprocessing existing COGs[/yellow]")

    if dry_run:
        console.print("[yellow]Dry run - no processing performed[/yellow]")
        for job in jobs[:5]:
            console.print(f"    Would process: {job.tile.name} {job.window_label}")
        if len(jobs) > 5:
            console.print(f"    ... and {len(jobs) - 5} more")
        return

    # Process tiles
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
            result = process_tile_job(job, force=force)

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
def catalog_build(
    source: str,
    out: str,
    window: str | None,
    tiles: str | None,
    thumbnail: str | None,
) -> None:
    """Build a Portolan-compliant STAC catalog from finished per-tile COGs."""
    from landsat_lst.catalog import build_catalog
    from landsat_lst.catalog.spec import DEFAULT_SPEC, spec_for_window

    spec = DEFAULT_SPEC if window is None else spec_for_window(window)
    wanted = tuple(name.strip() for name in tiles.split(",")) if tiles else None
    console.print(f"[bold]Building catalog for {spec.window}[/bold] from {source}")
    root = build_catalog(source, out, spec, tiles=wanted, thumbnail=thumbnail)
    console.print(f"  Wrote [green]{root}[/green]")


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
def catalog_validate(path: Path, as_json: bool) -> None:
    """Validate a built catalog against the Portolan profile."""
    import json as json_module

    from landsat_lst.catalog.validation import unaccepted_warnings, validate_catalog

    report = validate_catalog(path)
    unaccepted = unaccepted_warnings(report)
    if as_json:
        payload = report.to_dict()
        payload["unaccepted_warnings"] = sorted(unaccepted)
        click.echo(json_module.dumps(payload, indent=2))
    else:
        _print_report(report, unaccepted)
    if report.errors or unaccepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
