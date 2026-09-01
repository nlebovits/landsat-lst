"""Which ASTER GED granules production needs, and which one has.

``settings.ged_gap_mask`` defaults on, so every one of the 700 land tiles
reaches for a gap mask. The mask has two sources and, until this module
existed, neither could say whether it held enough to answer. The granule path
at least raised on an absent file; the artifact path read an unconsumed
granule as "no gap cells here", which is byte-identical to a genuinely
gap-free region -- a partial archive would have shipped 700 tiles that looked
successful and were unmasked.

The expected manifest is pure local arithmetic: the production tile list, the
global grid, the configured buffer, and the granule naming grammar. No
fetching, no network, no credentials.

**On upstream absence.** The AG100 collection genuinely has no granule over
open ocean, so "expected but not on disk" is not by itself a defect. Telling
the two apart needs an authoritative collection inventory, and there is none
offline: the collection total quoted in docs (24,873) is a hand-transcription
of a log line from a search whose results were never persisted, and the
build's old "coverage grid" was a listing of the local directory wearing an
inventory's clothes. So this module classifies honestly and refuses to guess
-- every absence is :data:`ABSENT_UNVERIFIED` unless a *fetch-domain* grid
narrows it, and even then the labels describe what was requested, never what
exists upstream. Establishing upstream truth requires a CMR query
(``earthaccess.search_data(short_name="AG1km", version="003", count=-1)``)
and that answer belongs in a persisted manifest, not in a log line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from landsat_lst import ged
from landsat_lst.config import settings
from landsat_lst.tiling import geobox_for_bbox

if TYPE_CHECKING:
    from pathlib import Path

#: Bumped when the classification or the report layout changes.
COVERAGE_VERSION = "1.1.0"

#: How many missing granule names the committed record carries. The full list
#: is five figures long; the counts are the finding and the names regenerate.
SAMPLE_SIZE = 25

#: The granule is on disk / in the artifact's consumed manifest.
CONSUMED = "consumed"

#: Absent locally, and nothing offline says whether it exists upstream.
ABSENT_UNVERIFIED = "absent-unverified-upstream"

#: Absent locally, and its 1-degree cell was never in the fetch request, so
#: nobody has ever asked the collection about it. Upstream status unknown.
ABSENT_OUTSIDE_FETCH_DOMAIN = "absent-outside-fetch-domain"

#: Absent locally although its cell *was* requested. The search that produced
#: the archive ran over the whole collection and returned nothing here, which
#: is evidence of upstream absence -- evidence, not proof, because a failed
#: download looks the same.
ABSENT_WITHIN_FETCH_DOMAIN = "absent-within-fetch-domain"


@dataclass(frozen=True)
class CoverageReport:
    """Expected vs consumed, per granule and in aggregate."""

    tiles: int
    buffer_cells: int
    expected: tuple[str, ...]
    consumed: tuple[str, ...]
    missing: tuple[str, ...]
    missing_core: tuple[str, ...]
    classification: dict[str, str]
    tiles_missing_core: tuple[str, ...]
    tiles_missing_any: tuple[str, ...]
    extra: tuple[str, ...]
    fetch_domain_source: str | None

    @property
    def complete(self) -> bool:
        """True when every expected granule was consumed."""
        return not self.missing

    def counts(self) -> dict[str, int]:
        """The headline numbers, for a report or a JSON record."""
        by_class: dict[str, int] = {}
        for label in self.classification.values():
            by_class[label] = by_class.get(label, 0) + 1
        return {
            "tiles": self.tiles,
            "expected": len(self.expected),
            "consumed_of_expected": len(self.expected) - len(self.missing),
            "missing": len(self.missing),
            "missing_core": len(self.missing_core),
            "tiles_missing_core": len(self.tiles_missing_core),
            "tiles_missing_any": len(self.tiles_missing_any),
            "extra_not_expected": len(self.extra),
            **{f"class_{k}": v for k, v in sorted(by_class.items())},
        }

    def as_dict(self) -> dict[str, object]:
        """The machine-readable record."""
        return {
            "coverage_version": COVERAGE_VERSION,
            "product": ged.GED_PRODUCT,
            "buffer_cells": self.buffer_cells,
            "fetch_domain_source": self.fetch_domain_source,
            "upstream_inventory": (
                "none available offline; every absence is classified by what was "
                "requested locally, never by what the collection holds. A CMR "
                "query is required to establish upstream absence."
            ),
            "complete": self.complete,
            "counts": self.counts(),
            # Capped on purpose: the full missing list runs to five figures and
            # this file is committed. The counts are the finding; the names are
            # regenerable in seconds from `landsat-lst ged-coverage`.
            "missing_core_sample": list(self.missing_core[:SAMPLE_SIZE]),
            "missing_core_sample_size": min(SAMPLE_SIZE, len(self.missing_core)),
            "tiles_missing_core": list(self.tiles_missing_core),
        }


def granules_for_tile(tile_name: str, buffer_cells: int) -> tuple[set[str], set[str]]:
    """The granules one tile touches: ``(all, core_only)``.

    Derived from the tile's share of the production global grid, so this is
    the same window :func:`landsat_lst.ged.gap_mask_for_geobox` will ask for.
    """
    from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415

    geobox = geobox_for_bbox(parse_tile_name(tile_name).bbox)
    _, _, core, padded = ged.cell_window_for_geobox(geobox, buffer_cells)
    row0, row1, col0, col1 = padded
    touched = ged.granules_for_window(row0=row0, row1=row1, col0=col0, col1=col1, core=core)
    return {n for n, _, _, _ in touched}, {n for n, _, _, is_core in touched if is_core}


def production_tiles() -> list[str]:
    """Every land tile a production run processes, from the job generator.

    Read through :func:`landsat_lst.job.generate_jobs` rather than
    ``LAND_TILES`` directly, so the expected manifest tracks whatever the
    fleet would actually submit.
    """
    from landsat_lst.job import generate_jobs  # noqa: PLC0415

    return [job.tile.name for job in generate_jobs()]


def _load_fetch_domain(path: Path | None) -> tuple[np.ndarray | None, str | None]:
    """The 1-degree grid of cells the archive's fetch ever requested.

    ``scripts/aster_gap_urban_analysis.py`` caches its urban-cell selection as
    a 180x360 boolean grid. It is a record of what was *asked for*, which is
    the only offline handle on why a granule is absent -- and it is not an
    upstream inventory, which is why the labels it produces say "fetch
    domain" and not "exists".
    """
    if path is None or not path.exists():
        return None, None
    grid = np.load(path)
    if grid.shape != (180, 360):
        msg = f"fetch-domain grid {path} is {grid.shape}, expected (180, 360)"
        raise ValueError(msg)
    return grid.astype(bool), str(path)


def _classify(missing: set[str], fetch_domain: np.ndarray | None) -> dict[str, str]:
    if fetch_domain is None:
        return dict.fromkeys(sorted(missing), ABSENT_UNVERIFIED)
    out: dict[str, str] = {}
    for name in sorted(missing):
        parts = name.split(".")
        lat_top, lon_west = int(parts[2]), int(parts[3])
        requested = bool(fetch_domain[90 - lat_top, lon_west + 180])
        out[name] = ABSENT_WITHIN_FETCH_DOMAIN if requested else ABSENT_OUTSIDE_FETCH_DOMAIN
    return out


def build_report(
    *,
    ged_dir: Path | None = None,
    artifact: Path | None = None,
    buffer_cells: int | None = None,
    tiles: list[str] | None = None,
    fetch_domain: Path | None = None,
) -> CoverageReport:
    """Compare what production needs against what a source holds.

    Args:
        ged_dir: Granule archive to measure. Defaults to ``settings.ged_dir``.
        artifact: Measure this artifact's consumed manifest instead of a
            directory listing. Takes precedence over ``ged_dir``.
        buffer_cells: Margin ring, defaulting to
            ``settings.ged_gap_buffer_cells``. The expected set must include
            it: the dilation reads gap cells one cell outside the tile.
        tiles: Tile names, defaulting to every production land tile.
        fetch_domain: Optional 180x360 ``.npy`` of the 1-degree cells the
            archive's fetch requested, used only to refine *why* a granule
            is absent.

    Returns:
        The comparison, per granule and in aggregate.
    """
    if buffer_cells is None:
        buffer_cells = settings.ged_gap_buffer_cells
    names = production_tiles() if tiles is None else list(tiles)

    expected: set[str] = set()
    core_by_tile: dict[str, set[str]] = {}
    all_by_tile: dict[str, set[str]] = {}
    for name in names:
        touched, core = granules_for_tile(name, buffer_cells)
        all_by_tile[name], core_by_tile[name] = touched, core
        expected |= touched

    if artifact is not None:
        with np.load(artifact) as data:
            held = {str(x) for x in data["consumed"]}
    else:
        directory = ged_dir if ged_dir is not None else settings.ged_dir
        held = {p.name for p in directory.glob("AG1km.v003.*.0010.h5")}

    missing = expected - held
    missing_core = {n for tile in names for n in core_by_tile[tile] if n in missing}
    grid, source = _load_fetch_domain(fetch_domain)
    return CoverageReport(
        tiles=len(names),
        buffer_cells=buffer_cells,
        expected=tuple(sorted(expected)),
        consumed=tuple(sorted(held)),
        missing=tuple(sorted(missing)),
        missing_core=tuple(sorted(missing_core)),
        classification=_classify(missing, grid),
        tiles_missing_core=tuple(sorted(t for t in names if core_by_tile[t] & missing)),
        tiles_missing_any=tuple(sorted(t for t in names if all_by_tile[t] & missing)),
        extra=tuple(sorted(held - expected)),
        fetch_domain_source=source,
    )
