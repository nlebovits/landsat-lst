"""Command-line interface for Landsat LST pipeline."""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """Landsat Land Surface Temperature annual composites."""


@main.command()
@click.option("--year", "-y", type=int, required=True, help="Year to process")
@click.option("--tile", "-t", type=str, help="Specific tile to process (e.g., N40W075)")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without running")
def process(year: int, tile: str | None, dry_run: bool) -> None:
    """Process Landsat data to annual composites."""
    console.print(f"[bold]Processing year {year}[/bold]")

    if tile:
        console.print(f"  Tile: {tile}")
    else:
        console.print("  Tiles: all land tiles")

    if dry_run:
        console.print("[yellow]Dry run - no processing performed[/yellow]")
        return

    console.print("[red]Processing not yet implemented[/red]")


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


if __name__ == "__main__":
    main()
