"""Build the compact global ASTER GED gap-cell artifact.

Scans a local archive of AG100 v3 granules (1x1 degree, 100x100 cells of
0.01 degree) and writes one compressed ``.npz`` holding every ``NumObs == 0``
cell as sparse global indices plus a 1-degree granule-coverage grid. The
pipeline prefers this artifact over the granules (``settings.ged_artifact``),
which is what lets a fleet VM ship one small file instead of 8,776 HDF5
granules. Gap cells are stored unbuffered; ``settings.ged_gap_buffer_cells``
is applied at load time, so a buffer change never needs a rebuild.

Usage:
    uv run python scripts/build_ged_gap_mask.py
    uv run python scripts/build_ged_gap_mask.py \\
        --ged-dir data/aster_ged \\
        --out data/ged_gap_mask.npz

See src/landsat_lst/ged.py, docs/findings-aster-ged-gaps.md, and the
2026-08-23 S30W065 verification (results/ged-mask-check/).
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from landsat_lst.config import settings
from landsat_lst.ged import build_artifact


@click.command()
@click.option(
    "--ged-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Granule archive to scan (default: settings.ged_dir).",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Artifact path to write (default: settings.ged_artifact).",
)
def main(ged_dir: Path | None, out: Path | None) -> None:
    """Scan the granule archive and write the global gap-cell artifact."""
    ged_dir = ged_dir or settings.ged_dir
    out = out or settings.ged_artifact
    started = time.monotonic()
    report = build_artifact(ged_dir, out)
    size_mb = out.stat().st_size / 1e6
    click.echo(
        f"scanned {report['granules']} granules from {ged_dir} in {time.monotonic() - started:.1f}s"
    )
    click.echo(f"gap cells (NumObs == 0): {report['gap_cells']}")
    click.echo(f"wrote {out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
