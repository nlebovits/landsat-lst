"""Build the compact global ASTER GED gap-cell artifact, deterministically.

Scans a local archive of AG100 v3 granules (1x1 degree, 100x100 cells of
0.01 degree) and writes one compressed ``.npz`` holding every ``NumObs == 0``
cell as sparse global indices, plus the manifest of granules it consumed with
their sha256 digests, the manifest production *expects*, the expected
granules the collection itself lacks (from the persisted CMR inventory that
``scripts/fetch_ged_granules.py`` writes), and a canonical content hash. Gap cells are stored unbuffered;
``settings.ged_gap_buffer_cells`` is applied at load time, so a buffer change
never needs a rebuild.

The consumed manifest is the point. An artifact built from a partial archive
reads, cell for cell, exactly like one built from a complete archive over a
region that has no gaps: both say "nothing to mask". Recording what was
consumed is what lets :func:`landsat_lst.ged.gap_mask_for_geobox` raise
instead of silently shipping an unmasked tile.

The content hash is taken over canonically ordered array bytes, never the
``.npz`` file, because a zip embeds timestamps and two byte-identical builds
would otherwise digest differently. Building twice on one archive reproduces
it; ``tests/unit/test_ged_artifact.py`` asserts that.

Usage:
    uv run --extra analysis python scripts/build_ged_gap_mask.py
    uv run --extra analysis python scripts/build_ged_gap_mask.py \\
        --ged-dir data/aster_ged \\
        --upstream-inventory results/decision/ged_upstream_inventory.json \\
        --require-complete \\
        --out src/landsat_lst/data/ged_gap_mask.npz

See src/landsat_lst/ged.py, src/landsat_lst/ged_coverage.py,
docs/findings-aster-ged-gaps.md, and results/decision/ged_gap_s30w065.json.
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from landsat_lst.config import settings
from landsat_lst.ged import ARTIFACT_FORMAT_VERSION, build_artifact
from landsat_lst.ged_coverage import build_report

DEFAULT_INVENTORY = Path("results/decision/ged_upstream_inventory.json")


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
@click.option(
    "--upstream-inventory",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_INVENTORY,
    show_default=True,
    help="The collection inventory scripts/fetch_ged_granules.py persisted. "
    "Expected granules it lacks are recorded as absent upstream and served "
    "with a warning rather than refused. Without it no build can be complete.",
)
@click.option(
    "--require-complete",
    is_flag=True,
    help="Exit non-zero unless the archive covers every granule the 700 "
    "production land tiles need and the collection holds. Use this before "
    "packaging an artifact: a partial one masks nothing over the gaps it "
    "never saw.",
)
def main(
    ged_dir: Path | None, out: Path | None, upstream_inventory: Path, require_complete: bool
) -> None:
    """Scan the granule archive and write the global gap-cell artifact."""
    ged_dir = ged_dir or settings.ged_dir
    out = out or settings.ged_artifact
    inventory_path = upstream_inventory if upstream_inventory.exists() else None
    if inventory_path is None:
        click.echo(
            f"no upstream inventory at {upstream_inventory}; every absence counts "
            "against completeness. Run scripts/fetch_ged_granules.py to write one."
        )

    click.echo("deriving the expected granule manifest from the production tile list...")
    coverage = build_report(ged_dir=ged_dir, upstream_inventory=inventory_path)
    counts = coverage.counts()
    click.echo(
        f"  {counts['tiles']} land tiles need {counts['expected']} granules "
        f"(buffer {coverage.buffer_cells} cell); archive holds "
        f"{counts['consumed_of_expected']} of them; the collection lacks "
        f"{counts['absent_upstream']} ({counts['absent_upstream_core']} inside a tile)"
    )
    if not coverage.complete:
        click.echo(
            f"  INCOMPLETE: {counts['fetchable']} fetchable granules absent, "
            f"{counts['fetchable_core']} of them inside a tile rather than its "
            f"margin, affecting {counts['tiles_missing_core']} tiles"
        )

    started = time.monotonic()
    report = build_artifact(
        ged_dir,
        out,
        expected=coverage.expected,
        absent_upstream=coverage.absent_upstream,
        inventory=None if coverage.inventory is None else coverage.inventory.identity(),
    )
    size_mb = out.stat().st_size / 1e6
    click.echo(
        f"scanned {report['granules']} granules from {ged_dir} in {time.monotonic() - started:.1f}s"
    )
    click.echo(f"gap cells (NumObs == 0): {report['gap_cells']}")
    click.echo(f"absent upstream:   {report['absent_upstream']}")
    click.echo(f"format version:    {ARTIFACT_FORMAT_VERSION}")
    click.echo(f"content sha256:    {report['content_sha256']}")
    click.echo(f"build code sha256: {report['build_code_sha256']}")
    click.echo(f"wrote {out} ({size_mb:.2f} MB)")

    if not report["complete"]:
        click.echo(
            "\nThis artifact is PARTIAL. It is safe to use -- a geobox reaching "
            "a granule it never consumed raises MissingGranuleError rather than "
            "reading zero gaps -- but it must not be packaged into the wheel as "
            "the production mask. Run `landsat-lst ged-coverage` for the "
            "shopping list."
        )
        if require_complete:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
