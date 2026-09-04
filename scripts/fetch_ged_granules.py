#!/usr/bin/env python3
"""Fetch every ASTER GED granule the production tile list needs, and record
what the collection holds.

The gap mask's artifact must be built from an archive that covers all 700
land tiles, and ``ged_coverage`` can only judge that against the collection's
own inventory: 2,374 of the 19,300 expected granules are open ocean or island
groups AG100 never covered, and no offline listing can tell those from a
granule that was never downloaded. So this script does both halves in one
run: it queries CMR once for the whole ``AG1km`` v003 collection, persists
that listing as ``results/decision/ged_upstream_inventory.json``, derives the
expected manifest, and downloads whatever is expected, present upstream, and
not yet on disk.

It is idempotent. A second run re-queries CMR, rewrites the inventory, and
finds nothing left to fetch.

Usage:
    uv run --extra analysis python scripts/fetch_ged_granules.py
    uv run --extra analysis python scripts/fetch_ged_granules.py --dry-run
    uv run --extra analysis python scripts/fetch_ged_granules.py \\
        --ged-dir data/aster_ged \\
        --inventory-out results/decision/ged_upstream_inventory.json

Requires NASA Earthdata credentials once:
    uv run python -c "import earthaccess; earthaccess.login(persist=True)"

Granules are ~0.47 MB each; the full production shortfall on 2026-09-04 was
8,482 granules, ~3.9 GB, latency-bound at 32 threads. No cloud compute.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import click
import structlog

from landsat_lst.config import settings
from landsat_lst.ged_coverage import build_report, write_upstream_inventory
from landsat_lst.logging_config import configure_logging

log = structlog.get_logger()

SHORT_NAME = "AG1km"
VERSION = "003"
DEFAULT_INVENTORY = Path("results/decision/ged_upstream_inventory.json")

#: Granules are tiny but each fetch pays the LP DAAC auth redirect, so
#: throughput is latency-bound. earthaccess defaults to 8.
DOWNLOAD_THREADS = 32


def _require_credentials(earthaccess: Any) -> None:
    """Fail with the fix rather than a stack trace when Earthdata auth is missing.

    Never uses the interactive strategy: an unattended run should exit with
    instructions instead of blocking on a password prompt.
    """
    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy, persist=False)
        except Exception as exc:
            log.debug("earthaccess_login_failed", strategy=strategy, error=str(exc))
            continue
        if getattr(auth, "authenticated", False):
            log.info("earthaccess_authenticated", strategy=strategy)
            return
    raise SystemExit(
        "NASA Earthdata credentials not found. Run once, interactively:\n"
        '    uv run python -c "import earthaccess; earthaccess.login(persist=True)"'
    )


def _granule_filename(granule: Any) -> str:
    for link in granule.data_links():
        name = link.rsplit("/", 1)[-1]
        if name.endswith(".h5"):
            return name
    return f"{granule['umm']['GranuleUR']}.h5"


@click.command()
@click.option(
    "--ged-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Granule archive to fill (default: settings.ged_dir).",
)
@click.option(
    "--inventory-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_INVENTORY,
    show_default=True,
    help="Where to persist the collection inventory.",
)
@click.option("--threads", type=int, default=DOWNLOAD_THREADS, show_default=True)
@click.option(
    "--dry-run", is_flag=True, help="Query and report; write the inventory; fetch nothing."
)
def main(ged_dir: Path | None, inventory_out: Path, threads: int, dry_run: bool) -> None:
    """Persist the AG1km v003 inventory and fetch the production shortfall."""
    configure_logging()
    import earthaccess  # noqa: PLC0415

    ged_dir = ged_dir or settings.ged_dir
    ged_dir.mkdir(parents=True, exist_ok=True)
    _require_credentials(earthaccess)

    log.info("cmr_search", short_name=SHORT_NAME, version=VERSION)
    query = earthaccess.granule_query().parameters(short_name=SHORT_NAME, version=VERSION)
    cmr_hits = query.hits()
    results = query.get(cmr_hits)
    by_name = {_granule_filename(g): g for g in results}
    if len(results) != cmr_hits or len(by_name) != cmr_hits:
        raise SystemExit(
            f"CMR reported {cmr_hits} hits, but the query returned {len(results)} records "
            f"and {len(by_name)} unique granule filenames; refusing to write a truncated "
            "inventory"
        )
    inventory = write_upstream_inventory(
        inventory_out,
        names=set(by_name),
        short_name=SHORT_NAME,
        version=VERSION,
        cmr_hits=cmr_hits,
    )
    click.echo(f"collection holds {inventory.granule_count} granules; wrote {inventory_out}")

    # Interrupted earthaccess downloads leave partial_* files; they would
    # never match the granule glob but they clutter the archive listing.
    for stray in ged_dir.glob("partial_*"):
        stray.unlink()

    report = build_report(ged_dir=ged_dir, upstream_inventory=inventory)
    counts = report.counts()
    click.echo(
        f"{counts['tiles']} tiles expect {counts['expected']} granules: "
        f"{counts['consumed_of_expected']} held, "
        f"{counts['absent_upstream']} absent upstream "
        f"({counts['absent_upstream_core']} inside a tile), "
        f"{counts['fetchable']} to fetch (~{counts['fetchable'] * 0.47 / 1024:.1f} GB)"
    )
    if dry_run or not report.fetchable:
        click.echo("nothing fetched" if dry_run else "archive is complete; nothing to fetch")
        return

    pending = [by_name[name] for name in report.fetchable]
    started = time.monotonic()
    earthaccess.download(pending, str(ged_dir), threads=threads)
    after = build_report(ged_dir=ged_dir, upstream_inventory=inventory)
    click.echo(
        f"fetched in {time.monotonic() - started:.0f}s; "
        f"still fetchable: {after.counts()['fetchable']}; complete: {after.complete}"
    )
    if not after.complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
