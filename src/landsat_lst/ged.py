"""ASTER GED emissivity-gap masking.

USGS's Landsat Collection 2 surface temperature is retrieved against ASTER
GED v3 (AG100) emissivity, which USGS interpolates where GED has no
observations (``NumObs == 0``). The tracked per-pixel cross-tab of the
published S30W065 tile (``landsat-lst ged-analyze``, recorded in
results/decision/ged_gap_s30w065.json) measures a strong
**spatial association** with those cells: they hold 79.9% of the tile's
>= 70 degC pixels at 369x the tile's base rate, and 99.6% of its missing
pixels, while ``NumObs >= 4`` cells hold none of the tail at all. The
association is not a causal trace -- nothing here follows which ASTER
observations produced a given pixel's emissivity -- so the mask is justified
by where the artifacts sit, not by a demonstrated mechanism.

The adopted rule masks ``NumObs == 0`` cells plus a 1 km buffer, which on
S30W065 removes 0.8642% of valid pixels and 92.45% of the >= 70 degC tail
(whose maximum is 77.87 degC, from the published COG's own band statistics).
Note 0.8642%, not the 0.863% quoted before: the earlier pass clipped its
dilation at the tile edge, and the production form pads the cell window
instead. See docs/findings-aster-ged-gaps.md and docs/methodology.md.

The mask applies to the **composite output only** -- masked pixels become
nodata in the LST P95 COG, exactly like the land mask. It never reaches
offset estimation (the land mask does; this one must not), and ``qa_count``
is left untouched: zero observations is data, and the count layer stays the
evidence behind every P95 value.

Two sources build the same mask:

- **The artifact** (:func:`build_artifact`'s ``.npz``): one compact global
  record of every gap cell plus the manifest of granules the build actually
  consumed, built by ``scripts/build_ged_gap_mask.py`` from a granule
  archive. This is the production path -- the wheel carries one small file,
  not thousands of granules. An earlier version stored a 1-degree "coverage
  grid" and read a cell outside it as holding no granule upstream; that was
  circular, because the grid was built by listing the local directory, so an
  undownloaded granule and an ocean cell set the same bit. The consumed
  manifest replaces it, and a geobox reaching outside it now raises
  :class:`MissingGranuleError` rather than quietly reading zero gaps.
- **The granules** (AG100 v3 HDF5, 1x1 degree, 100x100 cells of 0.01 deg):
  read directly when no artifact exists. A granule that is *absent* on this
  path is an error naming the granule ids, never a silent skip -- a missing
  granule means an unmasked tile. Build the artifact to process tiles whose
  bounding box reaches cells the collection genuinely does not cover.

Both paths were verified to produce identical masks on S30W065.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

from landsat_lst.config import settings

log = structlog.get_logger()

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from odc.geo.geobox import GeoBox

#: GED cell density: 100 cells per degree, i.e. 0.01-degree (~1 km) cells.
GED_CELLS_PER_DEGREE = 100

#: Cells per granule edge (1 degree at 100 cells per degree).
GRANULE_CELLS = 100

#: Global GED cell grid: row 0 is the cell whose north edge is +90 latitude,
#: column 0 the cell whose west edge is -180 longitude.
GLOBAL_CELL_ROWS = 180 * GED_CELLS_PER_DEGREE
GLOBAL_CELL_COLS = 360 * GED_CELLS_PER_DEGREE

#: Sentinel for a cell the AG100 collection holds no granule for. Distinct from
#: ``NumObs == 0`` (a granule that exists and saw nothing), which is the gap.
NUMOBS_ABSENT = -1

#: Bumped when the artifact layout changes; a mismatched artifact is refused
#: rather than reinterpreted. v2 adds the consumed manifest, its per-granule
#: digests, the expected manifest, and the canonical content hash -- without
#: which a partial archive is indistinguishable from a gap-free region. v3
#: adds the granules the collection itself lacks (``absent_upstream``, from a
#: persisted CMR inventory) and that inventory's identity, so completeness
#: can be judged against what exists rather than against every cell a tile
#: touches -- 2,374 of the 19,300 expected granules are open ocean or island
#: groups AG100 never covered.
ARTIFACT_FORMAT_VERSION = 3

#: Recorded in the artifact and hashed into its content digest.
GED_PRODUCT = "ASTER GED AG100 v003 (AG1km.v003, 0010)"

#: The content hash of the *published* artifact, verified on load. ``None``
#: means no artifact is published with this package. Pin the digest here in
#: the same commit that adds ``src/landsat_lst/data/ged_gap_mask.npz``, and
#: only from a build that ``scripts/build_ged_gap_mask.py --require-complete``
#: accepted: a partial artifact masks nothing over the gaps it never saw while
#: looking successful. See ``landsat-lst ged-coverage``.
GED_ARTIFACT_CONTENT_SHA256: str | None = (
    "62e9ca8f22e3bd0810f1a0034197ca327c20ef0ffef94a20a75d7ac291ac058f"
)


class MissingGranuleError(FileNotFoundError):
    """A tile needs GED granules that are not on disk.

    Raised by the granule path only. Never caught to skip: a missing granule
    would leave the tile's pixels unmasked over exactly the cells whose
    provenance is in question. The remedy is either to fetch the granules or
    to build the global artifact (``scripts/build_ged_gap_mask.py``), whose
    coverage grid knows which cells the collection genuinely has no granule
    for.
    """

    def __init__(self, granules: list[str], source: Path, *, source_kind: str = "granules") -> None:
        self.granules = granules
        self.source_kind = source_kind
        listing = ", ".join(granules)
        if source_kind == "artifact":
            detail = (
                f"the artifact {source} was built without them. An unconsumed "
                "granule contributes no gap cells, which is byte-identical to a "
                "granule that has none -- so continuing would mask nothing here "
                "and the tile would ship looking successful. Rebuild the "
                "artifact from an archive that covers these granules "
                "(scripts/build_ged_gap_mask.py); `landsat-lst ged-coverage` "
                "lists everything the 700 production tiles need."
            )
        else:
            detail = (
                f"they are missing from {source}. A missing granule means an "
                "unmasked tile, so this is an error, never a skip. Fetch the "
                "granules, or build the global artifact with "
                "scripts/build_ged_gap_mask.py from an archive that has them."
            )
        super().__init__(f"{len(granules)} ASTER GED granule(s) needed: {listing}. {detail}")


def granule_name(lat_top: int, lon_west: int) -> str:
    """The AG100 v3 filename for the granule at (north-edge lat, west-edge lon).

    Verified against the archive and each file's own ``Geolocation`` arrays:
    ``AG1km.v003.-30.-065.0010.h5`` spans latitude [-31, -30] and longitude
    [-65, -64], so the name carries the granule's *north* edge and *west*
    edge. Non-negative latitudes are zero-padded to two digits, longitudes to
    three, with a leading minus for negatives (``00``, ``-01``, ``008``,
    ``-065``).
    """
    lat = f"{lat_top:02d}" if lat_top >= 0 else f"-{-lat_top:02d}"
    lon = f"{lon_west:03d}" if lon_west >= 0 else f"-{-lon_west:03d}"
    return f"AG1km.v003.{lat}.{lon}.0010.h5"


def read_granule_numobs(path: Path) -> np.ndarray:
    """Read a granule's NumObs layer, oriented north-up and west-left.

    The archive's files already store row 0 north and column 0 west, but the
    orientation is checked against the granule's own ``Geolocation`` arrays
    rather than assumed -- a flipped granule would silently move every gap
    cell to the wrong side of the tile.

    ``h5py`` lives in the optional ``analysis`` extra: the granule path is a
    build-time and local-dev concern, while production VMs read the artifact.
    """
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        msg = (
            "reading ASTER GED granules needs h5py (the 'analysis' extra); "
            "production should read the artifact built by "
            "scripts/build_ged_gap_mask.py instead"
        )
        raise ImportError(msg) from exc

    with h5py.File(path) as f:
        lat = f["Geolocation/Latitude"][:]
        lon = f["Geolocation/Longitude"][:]
        numobs = f["Observations/NumObs"][:]
    if lat[0, 0] < lat[-1, 0]:
        numobs = np.flipud(numobs)
    if lon[0, 0] > lon[0, -1]:
        numobs = np.fliplr(numobs)
    return numobs


def _build_code_digest() -> str:
    """A digest of the code that decides artifact *content*.

    Narrow on purpose: the granule reader, the name grammar, and the build
    itself. Hashing the whole module would churn on every docstring edit and
    make the digest meaningless as a "would this rebuild differ" signal.
    """
    import inspect  # noqa: PLC0415

    src = "".join(
        inspect.getsource(fn)
        for fn in (granule_name, read_granule_numobs, granules_for_window, build_artifact)
    )
    return hashlib.sha256(src.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_count(value: object) -> int:
    """An inventory granule count from whatever an identity mapping carries."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    return 0


def canonical_content_hash(
    *,
    gap_rows: np.ndarray,
    gap_cols: np.ndarray,
    consumed: Sequence[str],
    consumed_sha256: Sequence[str],
    product: str,
    absent_upstream: Sequence[str] = (),
    inventory: Mapping[str, object] | None = None,
) -> str:
    """Hash the artifact's *content*, independent of how it was serialized.

    Over canonically ordered array bytes rather than the ``.npz`` file: a zip
    embeds timestamps and per-member compression state, so two byte-identical
    builds produce different file digests and a file digest can never be
    pinned. Gap cells are sorted and written as fixed big-endian ``int32``,
    so the digest is stable across platforms as well as across rebuilds.

    The absent-upstream list and the inventory identity are part of the
    content: they decide which tiles the artifact will *serve* rather than
    refuse, so two builds that differ only there are different products.
    """
    order = np.lexsort((gap_cols, gap_rows))
    h = hashlib.sha256()
    h.update(f"ged-artifact-v{ARTIFACT_FORMAT_VERSION}\n".encode())
    h.update(f"{product}\n".encode())
    for name, digest in sorted(zip(consumed, consumed_sha256, strict=True)):
        h.update(f"{name} {digest}\n".encode())
    for name in sorted(absent_upstream):
        h.update(f"absent-upstream {name}\n".encode())
    for key in ("short_name", "version", "queried_at", "granule_count"):
        value = "" if inventory is None else inventory.get(key, "")
        if key == "granule_count":
            # Stored as an int64 and read back as one, so an absent count
            # must hash the same way whether it arrives as "" or 0.
            value = _as_count(value)
        h.update(f"inventory {key}={value}\n".encode())
    h.update(np.asarray(gap_rows, dtype=">i4")[order].tobytes())
    h.update(np.asarray(gap_cols, dtype=">i4")[order].tobytes())
    return h.hexdigest()


def build_artifact(
    ged_dir: Path,
    out_path: Path,
    *,
    expected: Sequence[str] | None = None,
    absent_upstream: Sequence[str] | None = None,
    inventory: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Scan every granule under ``ged_dir`` into one compact global artifact.

    The artifact stores the *unbuffered* gap cells (``NumObs == 0``) as global
    cell indices. The buffer is applied at load time from
    ``settings.ged_gap_buffer_cells``, so a buffer change never needs a
    rebuild. Gap cells are rare (0.24% of cells on S30W065), which is why a
    sparse index list beats the 0.93 GB dense global grid.

    It also stores **what it consumed**, and that is the load-bearing part.
    An artifact built from a partial archive is indistinguishable, cell for
    cell, from one built from a complete archive over a region with no gaps:
    both say "no gap cells here". The consumed manifest is what lets
    :func:`gap_mask_for_geobox` tell those apart and refuse the first.

    Args:
        ged_dir: Directory of AG100 v003 granules to consume.
        out_path: Where to write the ``.npz``.
        expected: The manifest a complete artifact would need, from
            :func:`landsat_lst.ged_coverage.expected_granules`. Recorded
            alongside the consumed set so the artifact carries its own
            completeness verdict rather than relying on a separate report.
        absent_upstream: Expected granules the collection does not hold,
            established against a persisted CMR inventory. These are removed
            from ``missing_expected``: a tile touching one is served with a
            warning rather than refused, exactly as the granule path treats
            a margin absence, because nothing can ever be fetched for it.
        inventory: Identity of that inventory (``short_name``, ``version``,
            ``queried_at``, ``granule_count``), recorded and hashed.

    Returns:
        The build report, including the canonical content hash.
    """
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    granules = sorted(ged_dir.glob("AG1km.v003.*.0010.h5"))
    if not granules:
        msg = f"no AG1km.v003.*.0010.h5 granules under {ged_dir}"
        raise FileNotFoundError(msg)

    consumed: list[str] = []
    digests: list[str] = []
    for path in granules:
        parts = path.name.split(".")
        lat_top, lon_west = int(parts[2]), int(parts[3])
        numobs = read_granule_numobs(path)
        consumed.append(path.name)
        digests.append(_file_sha256(path))
        gap_r, gap_c = np.nonzero(numobs == 0)
        if gap_r.size:
            row0 = (90 - lat_top) * GRANULE_CELLS
            col0 = (lon_west + 180) * GRANULE_CELLS
            rows.append((gap_r + row0).astype(np.int32))
            cols.append((gap_c + col0).astype(np.int32))

    gap_rows = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    gap_cols = np.concatenate(cols) if cols else np.empty(0, dtype=np.int32)
    expected_list = sorted(expected) if expected is not None else []
    absent_list = sorted(absent_upstream) if absent_upstream is not None else []
    inventory_identity = dict(inventory) if inventory is not None else {}
    missing = sorted(set(expected_list) - set(consumed) - set(absent_list))
    content = canonical_content_hash(
        gap_rows=gap_rows,
        gap_cols=gap_cols,
        consumed=consumed,
        consumed_sha256=digests,
        product=GED_PRODUCT,
        absent_upstream=absent_list,
        inventory=inventory_identity,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        format_version=np.int32(ARTIFACT_FORMAT_VERSION),
        gap_rows=gap_rows,
        gap_cols=gap_cols,
        product=np.str_(GED_PRODUCT),
        consumed=np.array(consumed, dtype=np.str_),
        consumed_sha256=np.array(digests, dtype=np.str_),
        expected=np.array(expected_list, dtype=np.str_),
        missing_expected=np.array(missing, dtype=np.str_),
        absent_upstream=np.array(absent_list, dtype=np.str_),
        inventory_short_name=np.str_(str(inventory_identity.get("short_name", ""))),
        inventory_version=np.str_(str(inventory_identity.get("version", ""))),
        inventory_queried_at=np.str_(str(inventory_identity.get("queried_at", ""))),
        inventory_granule_count=np.int64(_as_count(inventory_identity.get("granule_count", 0))),
        content_sha256=np.str_(content),
        build_code_sha256=np.str_(_build_code_digest()),
    )
    return {
        "granules": len(consumed),
        "gap_cells": int(gap_rows.size),
        "expected": len(expected_list),
        "missing_expected": len(missing),
        "absent_upstream": len(absent_list),
        "content_sha256": content,
        "build_code_sha256": _build_code_digest(),
        "complete": expected is not None and not missing,
    }


@dataclass(frozen=True)
class _Artifact:
    """A loaded, verified artifact: its gap cells and what it was built from."""

    gap_rows: np.ndarray
    gap_cols: np.ndarray
    consumed: frozenset[str]
    absent_upstream: frozenset[str]
    inventory: dict[str, object]


def _load_artifact(path: Path) -> _Artifact:
    with np.load(path) as data:
        version = int(data["format_version"])
        if version != ARTIFACT_FORMAT_VERSION:
            msg = (
                f"GED gap artifact {path} is format v{version}, this code reads "
                f"v{ARTIFACT_FORMAT_VERSION}; rebuild it with "
                "scripts/build_ged_gap_mask.py"
            )
            raise ValueError(msg)
        gap_rows = data["gap_rows"]
        gap_cols = data["gap_cols"]
        consumed = [str(x) for x in data["consumed"]]
        digests = [str(x) for x in data["consumed_sha256"]]
        stored = str(data["content_sha256"])
        product = str(data["product"])
        absent = [str(x) for x in data["absent_upstream"]]
        inventory = {
            "short_name": str(data["inventory_short_name"]),
            "version": str(data["inventory_version"]),
            "queried_at": str(data["inventory_queried_at"]),
            "granule_count": int(data["inventory_granule_count"]),
        }

    recomputed = canonical_content_hash(
        gap_rows=gap_rows,
        gap_cols=gap_cols,
        consumed=consumed,
        consumed_sha256=digests,
        product=product,
        absent_upstream=absent,
        inventory=inventory,
    )
    if recomputed != stored:
        msg = (
            f"GED gap artifact {path} is self-inconsistent: it stores content "
            f"hash {stored} but its arrays hash to {recomputed}. The file is "
            "corrupt or was edited after the build; rebuild it with "
            "scripts/build_ged_gap_mask.py."
        )
        raise ValueError(msg)
    if GED_ARTIFACT_CONTENT_SHA256 is not None and stored != GED_ARTIFACT_CONTENT_SHA256:
        msg = (
            f"GED gap artifact {path} has content hash {stored}, but this code "
            f"pins {GED_ARTIFACT_CONTENT_SHA256}. A mask built from different "
            "granules is a different product; update GED_ARTIFACT_CONTENT_SHA256 "
            "in landsat_lst/ged.py deliberately, or point LST_GED_ARTIFACT at "
            "the pinned artifact."
        )
        raise ValueError(msg)
    return _Artifact(
        gap_rows=gap_rows,
        gap_cols=gap_cols,
        consumed=frozenset(consumed),
        absent_upstream=frozenset(absent),
        inventory=inventory,
    )


def _cells_from_artifact(
    artifact: Path,
    *,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    core: tuple[int, int, int, int],
) -> np.ndarray:
    """Cut the [row0:row1, col0:col1] window of gap cells out of the artifact.

    Refuses first. A granule the build never consumed contributes no gap
    cells, which is byte-identical to a granule that genuinely has none -- so
    an artifact built from a partial archive would mask a tile's gaps away
    silently, and the tile would ship looking successful. The consumed
    manifest turns that into a :class:`MissingGranuleError` naming the ids,
    matching what the granule path already does for an absent file.
    """
    data = _load_artifact(artifact)
    consumed = data.consumed
    missing_core: list[str] = []
    missing_margin: list[str] = []
    absent_core: list[str] = []
    for name, _, _, touches_core in granules_for_window(
        row0=row0, row1=row1, col0=col0, col1=col1, core=core
    ):
        if name in consumed:
            continue
        if name in data.absent_upstream:
            if touches_core:
                absent_core.append(name)
            continue
        (missing_core if touches_core else missing_margin).append(name)
    if missing_core:
        raise MissingGranuleError(missing_core, artifact, source_kind="artifact")
    if absent_core:
        # The collection has no granule here, verified against its inventory
        # at build time, so nothing can be fetched and refusing would fail the
        # tile forever. No NumObs means no ``NumObs == 0``: the cell
        # contributes no gap and the rule is unchanged. On the production
        # tile list these are open ocean and a few island groups.
        log.warning(
            "ged_upstream_granules_absent",
            granules=absent_core,
            note="the AG100 collection holds no granule here; no gap contribution",
        )
    if missing_margin:
        log.warning(
            "ged_margin_granules_absent",
            granules=missing_margin,
            note="margin-ring only; outside the artifact's consumed manifest",
        )

    cells = np.zeros((row1 - row0, col1 - col0), dtype=bool)
    rows = data.gap_rows.astype(np.int64)
    cols = data.gap_cols.astype(np.int64)
    # A window past the antimeridian addresses columns modulo the globe.
    for shift in (0, GLOBAL_CELL_COLS, -GLOBAL_CELL_COLS):
        c = cols + shift
        inside = (rows >= row0) & (rows < row1) & (c >= col0) & (c < col1)
        cells[rows[inside] - row0, c[inside] - col0] = True
    return cells


def granules_for_window(
    *,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    core: tuple[int, int, int, int] | None = None,
) -> list[tuple[str, int, int, bool]]:
    """Every granule a cell window touches, as ``(name, row0, col0, core)``.

    The one place the window-to-granule grammar lives. :func:`numobs_window`
    reads with it, :mod:`landsat_lst.ged_coverage` derives the expected
    manifest with it, and the artifact's coverage check tests against it, so
    a tile can never need a granule the completeness report did not count.

    ``core`` is the un-padded window; a granule overlapping it is flagged,
    because a *core* absence leaves unmasked pixels inside the tile while a
    *margin* absence only forgoes a buffer contribution. Columns wrap the
    antimeridian modulo the globe.

    Returns:
        One entry per granule: its filename, its global cell origin, and
        whether it overlaps ``core`` (always True when ``core`` is None).
    """
    core_row0, core_row1, core_col0, core_col1 = (
        core
        if core is not None
        else (
            row0,
            row1,
            col0,
            col1,
        )
    )
    out: list[tuple[str, int, int, bool]] = []
    for grow in range(row0 // GRANULE_CELLS, (row1 - 1) // GRANULE_CELLS + 1):
        lat_top = 90 - grow
        g_r0 = grow * GRANULE_CELLS
        for gcol in range(col0 // GRANULE_CELLS, (col1 - 1) // GRANULE_CELLS + 1):
            lon_west = (gcol % 360) - 180
            g_c0 = gcol * GRANULE_CELLS
            touches_core = (
                g_r0 < core_row1
                and g_r0 + GRANULE_CELLS > core_row0
                and g_c0 < core_col1
                and g_c0 + GRANULE_CELLS > core_col0
            )
            out.append((granule_name(lat_top, lon_west), g_r0, g_c0, touches_core))
    return out


def cell_window_for_geobox(
    geobox: GeoBox, buffer_cells: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    """The padded cell window a geobox needs, and its un-padded core.

    Returns ``(row_cells, col_cells, core, padded)``, where the cell index
    arrays are global (subtract the padded origin to index a window array).
    Shared so the mask, the NumObs reader, and the coverage report all agree
    on which cells -- and therefore which granules -- a tile reaches.
    """
    row_cells, col_cells = cell_indices_for_geobox(geobox)
    core = (
        int(row_cells[0]),
        int(row_cells[-1]) + 1,
        int(col_cells[0]),
        int(col_cells[-1]) + 1,
    )
    padded = (
        max(core[0] - buffer_cells, 0),
        min(core[1] + buffer_cells, GLOBAL_CELL_ROWS),
        core[2] - buffer_cells,
        core[3] + buffer_cells,
    )
    return row_cells, col_cells, core, padded


def numobs_window(
    ged_dir: Path,
    *,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    core: tuple[int, int, int, int],
) -> np.ndarray:
    """Mosaic NumObs over the cell window, reading granules from disk.

    Returns ``int16`` observation counts, with :data:`NUMOBS_ABSENT` where the
    collection holds no granule. The gap mask is ``result == 0``, so an absent
    granule contributes no gap -- matching the artifact path exactly.

    ``core`` is the un-padded cell window of the geobox itself. A granule
    covering any core cell must exist; absences are collected and raised
    together as one :class:`MissingGranuleError`, so the operator sees the
    whole shopping list at once -- a missing core granule means unmasked
    pixels inside the tile. A granule touching only the *margin* ring (the
    ``buffer_cells`` pad, where a neighbouring tile's gap cell could buffer
    across the edge) may be absent: the AG100 collection has no granule over
    open ocean and a few island groups (2,374 of the 19,300 cells production
    touches, per the persisted CMR inventory), and failing the tile for a
    fringe the artifact path also cannot see would make the two paths
    disagree. The absence is logged, never silent. This granule path knows
    nothing about the inventory, so a *core* granule the collection lacks
    raises here; only the packaged artifact, which records the absent set,
    serves such a tile.
    """
    cells = np.full((row1 - row0, col1 - col0), NUMOBS_ABSENT, dtype=np.int16)
    missing_core: list[str] = []
    missing_margin: list[str] = []
    for name, g_r0, g_c0, touches_core in granules_for_window(
        row0=row0, row1=row1, col0=col0, col1=col1, core=core
    ):
        path = ged_dir / name
        if not path.exists():
            (missing_core if touches_core else missing_margin).append(name)
            continue
        numobs = read_granule_numobs(path).astype(np.int16, copy=False)
        # Granule's global cell window, clipped to the requested one.
        r_lo, r_hi = max(g_r0, row0), min(g_r0 + GRANULE_CELLS, row1)
        c_lo, c_hi = max(g_c0, col0), min(g_c0 + GRANULE_CELLS, col1)
        cells[r_lo - row0 : r_hi - row0, c_lo - col0 : c_hi - col0] = numobs[
            r_lo - g_r0 : r_hi - g_r0, c_lo - g_c0 : c_hi - g_c0
        ]
    if missing_core:
        raise MissingGranuleError(missing_core, ged_dir)
    if missing_margin:
        log.warning(
            "ged_margin_granules_absent",
            granules=missing_margin,
            note="margin-ring only; no gap contribution, matching the artifact path",
        )
    return cells


def _cells_from_granules(
    ged_dir: Path,
    *,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    core: tuple[int, int, int, int],
) -> np.ndarray:
    """The ``NumObs == 0`` gap cells of :func:`numobs_window`."""
    return numobs_window(ged_dir, row0=row0, row1=row1, col0=col0, col1=col1, core=core) == 0


def dilate_cells(cells: np.ndarray, buffer_cells: int) -> np.ndarray:
    """Binary-dilate by ``buffer_cells`` with a square structuring element.

    ``buffer_cells=1`` is an 8-connected one-cell (~1 km) dilation, matching
    the S30W065 verification pass (scipy ``binary_dilation`` with a 3x3
    structure of ones); implemented as shifted ORs so scipy stays out of the
    runtime dependency set.
    """
    if buffer_cells <= 0 or not cells.any():
        return cells
    out = np.zeros_like(cells)
    height, width = cells.shape
    b = buffer_cells
    for dr in range(-b, b + 1):
        for dc in range(-b, b + 1):
            src = cells[max(-dr, 0) : height - max(dr, 0), max(-dc, 0) : width - max(dc, 0)]
            out[max(dr, 0) : height + min(dr, 0), max(dc, 0) : width + min(dc, 0)] |= src
    return out


def cell_indices_for_geobox(geobox: GeoBox) -> tuple[np.ndarray, np.ndarray]:
    """Global GED cell index of every pixel center, as ``(rows, cols)``.

    This is *the* grid mapping: :func:`gap_mask_for_geobox` masks with it and
    any analysis cross-tabbing output pixels by ``NumObs`` must use it too, so
    the two can never drift apart.

    Pixel centers come from ``geobox.transform`` -- the grid's own affine, so a
    row band's indices are the exact slice of its tile's. Each center maps to
    its 0.01-degree cell by floor division; centers sit at odd multiples of
    1/7200 degree while cell edges sit at multiples of 1/100, so a center can
    never land on a cell boundary and the mapping has no float knife-edge.
    """
    height, width = int(geobox.shape[0]), int(geobox.shape[1])
    t = geobox.transform
    lon_centers = t.c + t.a * (np.arange(width) + 0.5)
    lat_centers = t.f + t.e * (np.arange(height) + 0.5)
    row_cells = np.floor((90.0 - lat_centers) * GED_CELLS_PER_DEGREE).astype(np.int64)
    col_cells = np.floor((lon_centers + 180.0) * GED_CELLS_PER_DEGREE).astype(np.int64)
    return row_cells, col_cells


def numobs_for_geobox(
    geobox: GeoBox, ged_dir: Path, *, pad_cells: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumObs over the geobox's cell window, plus the per-pixel cell indices.

    The production mask is a boolean and the artifact stores only the gap
    cells, so tiering pixels by observation count (0, 1, 2, 3, >= 4) needs the
    granules. The window is returned as *cells*, not pixels: a five-degree tile
    is 500x500 cells against 18,000x18,000 pixels, so the caller expands it one
    row band at a time instead of materialising a full-tile array.

    Args:
        geobox: The grid the composite was computed on.
        ged_dir: Directory of AG100 v3 granules.
        pad_cells: Cells of margin beyond the geobox, for a dilation that must
            see gap cells just outside the tile.

    Returns:
        ``(numobs, row_cells, col_cells)``. ``numobs`` is ``int16`` over the
        padded window with :data:`NUMOBS_ABSENT` where the collection has no
        granule; ``row_cells`` and ``col_cells`` are *window-relative* indices,
        one per geobox row and column.
    """
    row_cells, col_cells, core, padded = cell_window_for_geobox(geobox, pad_cells)
    row0, row1, col0, col1 = padded
    numobs = numobs_window(ged_dir, row0=row0, row1=row1, col0=col0, col1=col1, core=core)
    return numobs, row_cells - row0, col_cells - col0


#: Where a published artifact ships inside the wheel. Nothing is packaged
#: there today (see :data:`GED_ARTIFACT_CONTENT_SHA256`); the resolver looks
#: for it regardless, so publishing one is a build step and not a code change.
PACKAGED_ARTIFACT = ("landsat_lst", "data/ged_gap_mask.npz")


def packaged_artifact_path() -> Path | None:
    """The artifact shipped inside the wheel, or None if none is packaged.

    ``importlib.resources``, not a path relative to the working directory:
    ``settings.ged_artifact``'s default is ``data/ged_gap_mask.npz``, which
    resolves against the *CWD*. A fleet VM runs from wherever Coiled drops it
    and never has the repo, so that default can only ever find the file on a
    developer laptop sitting in the checkout.
    """
    from importlib.resources import as_file, files  # noqa: PLC0415

    package, name = PACKAGED_ARTIFACT
    # joinpath on a relative path, so no __init__.py is needed under data/.
    try:
        resource = files(package).joinpath(name)
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    if not resource.is_file():
        return None
    with as_file(resource) as path:
        return Path(path)


def _resolve_source() -> tuple[str, Path]:
    """Pick a mask source, in a fixed order, and refuse silence.

    1. ``settings.ged_artifact`` if that file exists -- the explicit override.
    2. The artifact packaged in the wheel, which is what a VM has.
    3. ``settings.ged_dir`` granules, the local-dev and build-time path.
    4. Otherwise raise, naming all three, because the alternative is a
       composite that ships unmasked without saying so.
    """
    artifact = settings.ged_artifact
    if artifact.exists():
        return "artifact", artifact
    packaged = packaged_artifact_path()
    if packaged is not None:
        return "artifact", packaged
    ged_dir = settings.ged_dir
    if ged_dir.is_dir():
        return "granules", ged_dir
    packaged_name = "/".join(PACKAGED_ARTIFACT)
    msg = (
        "no GED gap-mask source. Tried, in order: the configured artifact "
        f"{artifact} (absent); the packaged artifact {packaged_name} (not "
        f"shipped in this build); the granule directory {ged_dir} (absent). "
        "The composite must not ship unmasked, so this is an error. Set "
        "LST_GED_ARTIFACT or LST_GED_DIR, build an artifact with "
        "scripts/build_ged_gap_mask.py, or disable the mask explicitly with "
        "LST_GED_GAP_MASK=false."
    )
    raise FileNotFoundError(msg)


def gap_mask_for_geobox(
    geobox: GeoBox,
    *,
    buffer_cells: int | None = None,
) -> np.ndarray:
    """Boolean mask on the geobox's own grid: True where a pixel must be dropped.

    Pixel centers come from ``geobox.transform`` -- the grid's own affine, so
    a row band's mask is the exact slice of its tile's (the same argument as
    :func:`landsat_lst.masks.get_land_mask_for_geobox`). Each center maps to
    its 0.01-degree GED cell by floor division; centers sit at odd multiples
    of 1/7200 degree while cell edges sit at multiples of 1/100, so a center
    can never land on a cell boundary and the mapping has no float knife-edge.

    The cell window is padded by ``buffer_cells`` beyond the geobox, because
    a gap cell just outside the tile buffers into it. The 2026-08-23
    verification pass clipped its dilation at the tile edge instead, and the
    two do **not** agree on S30W065: its margin ring carries 6 gap cells, 2 of
    which buffer into the tile, so the padded form removes 2 more cells (up to
    2,592 pixels). That is the whole of the 0.863% / 0.864% gap between the
    figure this docstring used to quote and ``landsat-lst ged-analyze``'s.
    The padded form is the correct one, here and for the general tile.

    Args:
        geobox: The grid the composite was computed on -- native tiles, row
            bands, anything cut from the global grid.
        buffer_cells: Dilation radius in GED cells (~1 km each). Defaults to
            ``settings.ged_gap_buffer_cells``.

    Returns:
        Boolean array shaped like the geobox, True on ``NumObs == 0`` cells
        and their buffer.
    """
    if buffer_cells is None:
        buffer_cells = settings.ged_gap_buffer_cells

    row_cells, col_cells, core, padded = cell_window_for_geobox(geobox, buffer_cells)
    row0, row1, col0, col1 = padded

    kind, source = _resolve_source()
    if kind == "artifact":
        cells = _cells_from_artifact(source, row0=row0, row1=row1, col0=col0, col1=col1, core=core)
    else:
        cells = _cells_from_granules(source, row0=row0, row1=row1, col0=col0, col1=col1, core=core)

    cells = dilate_cells(cells, buffer_cells)
    return cells[np.ix_(row_cells - row0, col_cells - col0)]
