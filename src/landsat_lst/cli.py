"""Command-line interface for Landsat LST pipeline."""

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """Landsat Land Surface Temperature annual composites."""


@main.command()
@click.option("--year", "-y", type=int, required=True, help="Year to process")
@click.option("--tile", "-t", type=str, help="Specific tile to process (e.g., N40W075)")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without running")
@click.option("--force", "-f", is_flag=True, help="Reprocess even if COG exists")
def process(year: int, tile: str | None, dry_run: bool, force: bool) -> None:
    """Process Landsat data to annual composites."""
    from landsat_lst.job import generate_jobs, process_tile_job
    from landsat_lst.models import ProcessingJob
    from landsat_lst.tiling import LAND_TILES, parse_tile_name

    console.print(f"[bold]Processing year {year}[/bold]")

    if tile:
        if tile not in LAND_TILES:
            console.print(f"[red]Warning: {tile} is not in the land tiles set[/red]")
        tile_id = parse_tile_name(tile)
        jobs = [ProcessingJob(tile=tile_id, year=year)]
        console.print(f"  Tile: {tile}")
    else:
        jobs = generate_jobs([year])
        console.print(f"  Tiles: {len(jobs)} land tiles")

    if force:
        console.print("  [yellow]Force mode: reprocessing existing COGs[/yellow]")

    if dry_run:
        console.print("[yellow]Dry run - no processing performed[/yellow]")
        for job in jobs[:5]:
            console.print(f"    Would process: {job.tile.name} {job.year}")
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


@main.command()
@click.option("--year", "-y", type=int, required=True, help="Year to virtualize")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing virtual datacube")
def virtualize(year: int, force: bool) -> None:
    """Create virtual datacube from existing COGs for a year."""
    from landsat_lst.storage import get_storage
    from landsat_lst.tiling import LAND_TILES
    from landsat_lst.virtual import create_virtual_datacube

    storage = get_storage()

    tile_paths = []
    tile_names = []
    for tile_name in sorted(LAND_TILES):
        if storage.cog_exists(year, tile_name):
            tile_paths.append(storage.cog_path(year, tile_name))
            tile_names.append(tile_name)

    if not tile_paths:
        console.print(f"[red]No COGs found for {year}[/red]")
        raise SystemExit(1)

    console.print(f"[bold]Virtualizing {len(tile_paths)} tiles for {year}[/bold]")

    if force:
        console.print("[yellow]Force mode: overwriting existing datacube[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Creating virtual datacube...", total=None)
        snapshot_id = create_virtual_datacube(tile_paths, tile_names, year, storage)

    console.print(f"[green]✓ Created virtual datacube: {snapshot_id[:12]}...[/green]")


if __name__ == "__main__":
    main()
