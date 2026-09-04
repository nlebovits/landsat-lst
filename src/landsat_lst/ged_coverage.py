"""Which ASTER GED granules production needs, and which ones a source has.

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

**On upstream absence.** The AG100 collection has no granule over open ocean
and over a few island groups, so "expected but not on disk" is not by itself
a defect. Telling the two apart needs the collection's own inventory, which
``scripts/fetch_ged_granules.py`` persists from one CMR query
(``earthaccess.search_data(short_name="AG1km", version="003", count=-1)``)
to ``results/decision/ged_upstream_inventory.json``. Given that inventory,
an expected granule is one of three things: consumed, absent upstream, or
fetchable. Only the third counts against completeness. Without an inventory
every absence stays :data:`ABSENT_UNVERIFIED` and the report cannot be
complete, because nothing offline can say what the collection holds.

Measured on 2026-09-04 against the 24,873-granule inventory: the 700 tiles
expect 19,300 granules, of which 2,374 do not exist upstream (1,480 inside a
tile core). 1,396 of those core cells hold no land at all; the rest are
Pacific islands and Kerguelen, 5.53 square degrees of land in total.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from landsat_lst import ged
from landsat_lst.config import settings
from landsat_lst.tiling import geobox_for_bbox

#: Bumped when the classification or the report layout changes.
COVERAGE_VERSION = "2.0.0"

#: Layout of newly written ``ged_upstream_inventory.json`` records. Schema v2
#: binds the returned listing to CMR's independently reported hit count and to
#: a canonical content hash. The one committed v1 record predates those fields;
#: it is accepted only when its names match the pinned snapshot below.
INVENTORY_SCHEMA_VERSION = 2
LEGACY_INVENTORY_SCHEMA_VERSION = 1

#: The only collection an inventory may authorize absences for.
INVENTORY_SHORT_NAME = "AG1km"
INVENTORY_VERSION = "003"

#: Canonical name-set digest of the committed 2026-09-04 v1 inventory. A v1
#: record has no CMR hit-count attestation, so accepting any other v1 listing
#: would let a truncated or unrelated file classify real granules as absent.
LEGACY_INVENTORY_CONTENT_SHA256 = "928e828f393dacae164c6c3ecbe0e4ca5cc70e6fe37ab52a7113cf256fb94440"

#: How many missing granule names the committed record carries. The full list
#: is five figures long; the counts are the finding and the names regenerate.
SAMPLE_SIZE = 25

#: The granule is on disk / in the artifact's consumed manifest.
CONSUMED = "consumed"

#: Absent locally, and the collection inventory says it does not exist.
ABSENT_UPSTREAM = "absent-upstream"

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
class UpstreamInventory:
    """Every granule the AG100 collection holds, from one persisted CMR query.

    The identity fields travel into the artifact and its content hash, so an
    artifact built against a different inventory is a different product.
    """

    short_name: str
    version: str
    queried_at: str
    granule_count: int
    names: frozenset[str]
    cmr_hits: int
    content_sha256: str

    def identity(self) -> dict[str, object]:
        """The fields an artifact records about the inventory it used."""
        return {
            "short_name": self.short_name,
            "version": self.version,
            "queried_at": self.queried_at,
            "granule_count": self.granule_count,
        }


def inventory_content_hash(names: set[str] | frozenset[str]) -> str:
    """Canonical digest of an inventory's granule-name set."""
    h = hashlib.sha256()
    h.update(b"ged-upstream-inventory-v1\n")
    for name in sorted(names):
        h.update(f"{name}\n".encode())
    return h.hexdigest()


def _validate_inventory(inventory: UpstreamInventory) -> UpstreamInventory:
    """Refuse an inventory that cannot establish a complete AG1km v003 query."""
    if inventory.short_name != INVENTORY_SHORT_NAME or inventory.version != INVENTORY_VERSION:
        msg = (
            "upstream inventory is for "
            f"{inventory.short_name} v{inventory.version}, expected "
            f"{INVENTORY_SHORT_NAME} v{INVENTORY_VERSION}"
        )
        raise ValueError(msg)
    try:
        stamp = datetime.fromisoformat(inventory.queried_at)
    except ValueError as exc:
        raise ValueError(
            f"upstream inventory queried_at is not an ISO timestamp: {inventory.queried_at!r}"
        ) from exc
    if stamp.tzinfo is None:
        raise ValueError("upstream inventory queried_at must include a timezone")
    if inventory.granule_count <= 0:
        raise ValueError("upstream inventory is empty; it cannot establish collection absence")
    if inventory.granule_count != len(inventory.names):
        msg = (
            f"upstream inventory says {inventory.granule_count} granules but contains "
            f"{len(inventory.names)} unique names"
        )
        raise ValueError(msg)
    if inventory.cmr_hits != inventory.granule_count:
        msg = (
            f"upstream inventory retrieved {inventory.granule_count} unique granules but CMR "
            f"reported {inventory.cmr_hits} hits; the listing may be truncated"
        )
        raise ValueError(msg)
    for name in inventory.names:
        try:
            lat_top, lon_west = _cell_of(name)
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid AG1km granule name in upstream inventory: {name!r}") from exc
        if (
            not -89 <= lat_top <= 90
            or not -180 <= lon_west <= 179
            or ged.granule_name(lat_top, lon_west) != name
        ):
            raise ValueError(f"invalid AG1km granule name in upstream inventory: {name!r}")
    actual_hash = inventory_content_hash(inventory.names)
    if inventory.content_sha256 != actual_hash:
        msg = (
            f"upstream inventory stores content hash {inventory.content_sha256}, but its names "
            f"hash to {actual_hash}"
        )
        raise ValueError(msg)
    return inventory


def write_upstream_inventory(
    path: Path,
    *,
    names: set[str],
    short_name: str,
    version: str,
    cmr_hits: int,
    queried_at: datetime | None = None,
) -> UpstreamInventory:
    """Persist a CMR granule listing as the compact inventory record.

    Stored as ``[lat_top, lon_west]`` cell pairs rather than names, which
    keeps 24,873 granules under 300 KB and inside the repository's size cap.
    Names are rebuilt through :func:`landsat_lst.ged.granule_name`, so the
    record shares the grammar every other consumer uses.
    """
    frozen_names = frozenset(names)
    stamp = (queried_at or datetime.now(UTC)).isoformat(timespec="seconds")
    inventory = _validate_inventory(
        UpstreamInventory(
            short_name=short_name,
            version=version,
            queried_at=stamp,
            granule_count=len(frozen_names),
            names=frozen_names,
            cmr_hits=cmr_hits,
            content_sha256=inventory_content_hash(frozen_names),
        )
    )
    record = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        **inventory.identity(),
        "cmr_hits": inventory.cmr_hits,
        "content_sha256": inventory.content_sha256,
        "cells": sorted(_cell_of(n) for n in inventory.names),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    return inventory


def load_upstream_inventory(path: Path) -> UpstreamInventory:
    """Read the inventory record and rebuild the granule names from its cells."""
    record = json.loads(Path(path).read_text())
    schema = record.get("schema_version")
    if schema not in (LEGACY_INVENTORY_SCHEMA_VERSION, INVENTORY_SCHEMA_VERSION):
        msg = (
            f"upstream inventory {path} is schema v{schema}, this code reads "
            f"v{INVENTORY_SCHEMA_VERSION}; regenerate it with scripts/fetch_ged_granules.py"
        )
        raise ValueError(msg)
    names = frozenset(ged.granule_name(int(lat), int(lon)) for lat, lon in record["cells"])
    count = int(record["granule_count"])
    if count != len(names):
        msg = f"upstream inventory {path} says {count} granules but lists {len(names)} cells"
        raise ValueError(msg)
    content_sha256 = inventory_content_hash(names)
    if schema == LEGACY_INVENTORY_SCHEMA_VERSION:
        if content_sha256 != LEGACY_INVENTORY_CONTENT_SHA256:
            msg = (
                f"legacy upstream inventory {path} has unpinned content hash "
                f"{content_sha256}; regenerate it with scripts/fetch_ged_granules.py"
            )
            raise ValueError(msg)
        cmr_hits = count
    else:
        cmr_hits = int(record["cmr_hits"])
        stored_hash = str(record["content_sha256"])
        if stored_hash != content_sha256:
            msg = (
                f"upstream inventory {path} stores content hash {stored_hash}, but its names "
                f"hash to {content_sha256}"
            )
            raise ValueError(msg)
    return _validate_inventory(
        UpstreamInventory(
            short_name=str(record["short_name"]),
            version=str(record["version"]),
            queried_at=str(record["queried_at"]),
            granule_count=count,
            names=names,
            cmr_hits=cmr_hits,
            content_sha256=content_sha256,
        )
    )


def _cell_of(name: str) -> tuple[int, int]:
    parts = name.split(".")
    return int(parts[2]), int(parts[3])


@dataclass(frozen=True)
class CoverageReport:
    """Expected vs consumed, per granule and in aggregate."""

    tiles: int
    buffer_cells: int
    expected: tuple[str, ...]
    consumed: tuple[str, ...]
    missing: tuple[str, ...]
    missing_core: tuple[str, ...]
    absent_upstream: tuple[str, ...]
    absent_upstream_core: tuple[str, ...]
    classification: dict[str, str]
    tiles_missing_core: tuple[str, ...]
    tiles_missing_any: tuple[str, ...]
    extra: tuple[str, ...]
    fetch_domain_source: str | None
    inventory: UpstreamInventory | None
    artifact_inventory: dict[str, object] | None = None

    @property
    def fetchable(self) -> tuple[str, ...]:
        """Expected, not held, and present upstream: what a fetch must still get."""
        absent = set(self.absent_upstream)
        return tuple(n for n in self.missing if n not in absent)

    @property
    def fetchable_core(self) -> tuple[str, ...]:
        absent = set(self.absent_upstream)
        return tuple(n for n in self.missing_core if n not in absent)

    @property
    def complete(self) -> bool:
        """True when every expected granule the collection holds was consumed.

        An artifact's validated embedded absence set is evidence equivalent
        to the inventory used to build it; a directory still needs an explicit
        inventory before any missing granule can be excused.
        """
        return not self.fetchable

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
            "absent_upstream": len(self.absent_upstream),
            "absent_upstream_core": len(self.absent_upstream_core),
            "fetchable": len(self.fetchable),
            "fetchable_core": len(self.fetchable_core),
            "tiles_missing_core": len(self.tiles_missing_core),
            "tiles_missing_any": len(self.tiles_missing_any),
            "extra_not_expected": len(self.extra),
            **{f"class_{k}": v for k, v in sorted(by_class.items())},
        }

    def as_dict(self) -> dict[str, object]:
        """The machine-readable record."""
        inventory: dict[str, object] | str
        if self.inventory is not None:
            inventory = self.inventory.identity()
        elif self.artifact_inventory is not None:
            inventory = self.artifact_inventory
        else:
            inventory = (
                "none given; every absence is classified by what was requested "
                "locally, never by what the collection holds. Pass the record "
                "scripts/fetch_ged_granules.py writes to establish upstream absence."
            )
        return {
            "coverage_version": COVERAGE_VERSION,
            "product": ged.GED_PRODUCT,
            "buffer_cells": self.buffer_cells,
            "fetch_domain_source": self.fetch_domain_source,
            "upstream_inventory": inventory,
            "complete": self.complete,
            "counts": self.counts(),
            # Capped on purpose: the full missing list runs to five figures and
            # this file is committed. The counts are the finding; the names are
            # regenerable in seconds from `landsat-lst ged-coverage`.
            "fetchable_core_sample": list(self.fetchable_core[:SAMPLE_SIZE]),
            "fetchable_core_sample_size": min(SAMPLE_SIZE, len(self.fetchable_core)),
            "absent_upstream_core_sample": list(self.absent_upstream_core[:SAMPLE_SIZE]),
            "absent_upstream_core_sample_size": min(SAMPLE_SIZE, len(self.absent_upstream_core)),
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


def expected_granules(buffer_cells: int | None = None) -> set[str]:
    """Every granule the production tile list needs, buffer ring included."""
    if buffer_cells is None:
        buffer_cells = settings.ged_gap_buffer_cells
    expected: set[str] = set()
    for name in production_tiles():
        touched, _ = granules_for_tile(name, buffer_cells)
        expected |= touched
    return expected


def _load_fetch_domain(path: Path | None) -> tuple[np.ndarray | None, str | None]:
    """The 1-degree grid of cells the archive's fetch ever requested.

    ``scripts/aster_gap_urban_analysis.py`` caches its urban-cell selection as
    a 180x360 boolean grid. It is a record of what was *asked for*, which is
    the only offline handle on why a granule is absent -- and it is not an
    upstream inventory, which is why the labels it produces say "fetch
    domain" and not "exists". The recorded source is relative to the working
    directory so the committed record carries no machine-specific path.
    """
    if path is None or not path.exists():
        return None, None
    grid = np.load(path)
    if grid.shape != (180, 360):
        msg = f"fetch-domain grid {path} is {grid.shape}, expected (180, 360)"
        raise ValueError(msg)
    return grid.astype(bool), Path(os.path.relpath(path)).as_posix()


def _classify(
    missing: set[str],
    fetch_domain: np.ndarray | None,
    inventory: UpstreamInventory | None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in sorted(missing):
        if inventory is not None and name not in inventory.names:
            out[name] = ABSENT_UPSTREAM
        elif fetch_domain is None:
            out[name] = ABSENT_UNVERIFIED
        else:
            lat_top, lon_west = _cell_of(name)
            requested = bool(fetch_domain[90 - lat_top, lon_west + 180])
            out[name] = ABSENT_WITHIN_FETCH_DOMAIN if requested else ABSENT_OUTSIDE_FETCH_DOMAIN
    return out


def _source_manifests(
    *, ged_dir: Path | None, artifact: Path | None
) -> tuple[set[str], set[str], dict[str, object] | None]:
    """Validated held/absent manifests and inventory identity for one source."""
    if artifact is not None:
        data = ged._load_artifact(artifact)
        return set(data.consumed), set(data.absent_upstream), data.inventory
    directory = ged_dir if ged_dir is not None else settings.ged_dir
    return {p.name for p in directory.glob("AG1km.v003.*.0010.h5")}, set(), None


def _established_absences(
    *,
    missing: set[str],
    artifact: Path | None,
    embedded_absent: set[str],
    inventory: UpstreamInventory | None,
) -> set[str]:
    """Upstream absences established by an artifact or explicit inventory."""
    if artifact is None:
        return set() if inventory is None else missing - inventory.names

    absent = missing & embedded_absent
    if inventory is None:
        return absent

    inventory_absent = missing - inventory.names
    if inventory_absent != absent:
        msg = (
            f"artifact {artifact} records {len(absent)} upstream absences, "
            f"but the supplied inventory establishes {len(inventory_absent)}"
        )
        raise ValueError(msg)
    return absent


def build_report(
    *,
    ged_dir: Path | None = None,
    artifact: Path | None = None,
    buffer_cells: int | None = None,
    tiles: list[str] | None = None,
    fetch_domain: Path | None = None,
    upstream_inventory: Path | UpstreamInventory | None = None,
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
        upstream_inventory: The collection inventory, as a path to the record
            ``scripts/fetch_ged_granules.py`` writes or an already loaded one.
            With it, a missing granule the collection lacks is
            :data:`ABSENT_UPSTREAM` and does not count against completeness.

    Returns:
        The comparison, per granule and in aggregate.
    """
    if buffer_cells is None:
        buffer_cells = settings.ged_gap_buffer_cells
    names = production_tiles() if tiles is None else list(tiles)
    inventory = (
        load_upstream_inventory(upstream_inventory)
        if isinstance(upstream_inventory, Path)
        else upstream_inventory
    )
    if inventory is not None:
        inventory = _validate_inventory(inventory)

    expected: set[str] = set()
    core_by_tile: dict[str, set[str]] = {}
    all_by_tile: dict[str, set[str]] = {}
    for name in names:
        touched, core = granules_for_tile(name, buffer_cells)
        all_by_tile[name], core_by_tile[name] = touched, core
        expected |= touched

    held, embedded_absent, artifact_inventory = _source_manifests(
        ged_dir=ged_dir, artifact=artifact
    )
    missing = expected - held
    missing_core = {n for tile in names for n in core_by_tile[tile] if n in missing}
    absent_upstream = _established_absences(
        missing=missing,
        artifact=artifact,
        embedded_absent=embedded_absent,
        inventory=inventory,
    )
    fetchable = missing - absent_upstream
    grid, source = _load_fetch_domain(fetch_domain)
    classification = _classify(missing, grid, inventory)
    for name in absent_upstream:
        classification[name] = ABSENT_UPSTREAM
    return CoverageReport(
        tiles=len(names),
        buffer_cells=buffer_cells,
        expected=tuple(sorted(expected)),
        consumed=tuple(sorted(held)),
        missing=tuple(sorted(missing)),
        missing_core=tuple(sorted(missing_core)),
        absent_upstream=tuple(sorted(absent_upstream)),
        absent_upstream_core=tuple(sorted(missing_core & absent_upstream)),
        classification=classification,
        tiles_missing_core=tuple(sorted(t for t in names if core_by_tile[t] & fetchable)),
        tiles_missing_any=tuple(sorted(t for t in names if all_by_tile[t] & fetchable)),
        extra=tuple(sorted(held - expected)),
        fetch_domain_source=source,
        inventory=inventory,
        artifact_inventory=artifact_inventory,
    )
