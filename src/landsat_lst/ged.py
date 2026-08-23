"""ASTER GED emissivity-gap masking.

USGS's Landsat Collection 2 surface temperature is retrieved against ASTER
GED v3 (AG100) emissivity. Where GED has no observations (``NumObs == 0``)
the emissivity is interpolated, and the per-pixel check on S30W065
(2026-08-23, results/ged-mask-check/) showed the consequence: the gap cores
produce ST fill (the missing blobs) and the contaminated fringe produces
spurious 70-78 degC retrievals that the P95 promotes. The confirmed rule is
to mask ``NumObs == 0`` cells plus a 1 km buffer, which on S30W065 removes
0.863% of valid pixels and 92.45% of the >= 70 degC artifact tail. See
docs/findings-aster-ged-gaps.md and docs/methodology.md.

The mask applies to the **composite output only** -- masked pixels become
nodata in the LST P95 COG, exactly like the land mask. It never reaches
offset estimation (the land mask does; this one must not), and ``qa_count``
is left untouched: zero observations is data, and the count layer stays the
evidence behind every P95 value.

Two sources build the same mask:

- **The artifact** (:func:`build_artifact`'s ``.npz``): one compact global
  record of every gap cell plus a 1-degree granule-coverage grid, built once
  by ``scripts/build_ged_gap_mask.py`` from the local granule archive. This
  is the production path -- a fleet VM ships one small file, not 8,776
  granules. A 1-degree cell outside the coverage grid holds no granule in
  the collection (open ocean; the land mask owns it) and contributes no gap.
- **The granules** (AG100 v3 HDF5, 1x1 degree, 100x100 cells of 0.01 deg):
  read directly when no artifact exists. A granule that is *absent* on this
  path is an error naming the granule ids, never a silent skip -- a missing
  granule means an unmasked tile. Build the artifact to process tiles whose
  bounding box reaches cells the collection genuinely does not cover.

Both paths were verified to produce identical masks on S30W065.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

from landsat_lst.config import settings

log = structlog.get_logger()

if TYPE_CHECKING:
    from pathlib import Path

    from odc.geo.geobox import GeoBox

#: GED cell density: 100 cells per degree, i.e. 0.01-degree (~1 km) cells.
GED_CELLS_PER_DEGREE = 100

#: Cells per granule edge (1 degree at 100 cells per degree).
GRANULE_CELLS = 100

#: Global GED cell grid: row 0 is the cell whose north edge is +90 latitude,
#: column 0 the cell whose west edge is -180 longitude.
GLOBAL_CELL_ROWS = 180 * GED_CELLS_PER_DEGREE
GLOBAL_CELL_COLS = 360 * GED_CELLS_PER_DEGREE

#: Bumped when the artifact layout changes; a mismatched artifact is refused
#: rather than reinterpreted.
ARTIFACT_FORMAT_VERSION = 1


class MissingGranuleError(FileNotFoundError):
    """A tile needs GED granules that are not on disk.

    Raised by the granule path only. Never caught to skip: a missing granule
    would leave the tile's pixels unmasked over exactly the cells whose
    provenance is in question. The remedy is either to fetch the granules or
    to build the global artifact (``scripts/build_ged_gap_mask.py``), whose
    coverage grid knows which cells the collection genuinely has no granule
    for.
    """

    def __init__(self, granules: list[str], ged_dir: Path) -> None:
        self.granules = granules
        listing = ", ".join(granules)
        super().__init__(
            f"{len(granules)} ASTER GED granule(s) missing from {ged_dir}: "
            f"{listing}. A missing granule means an unmasked tile, so this is "
            "an error, never a skip. Fetch the granules, or build the global "
            "artifact with scripts/build_ged_gap_mask.py -- its coverage grid "
            "distinguishes a granule the collection never had (open ocean, no "
            "gap contribution) from one this machine is missing."
        )


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


def build_artifact(ged_dir: Path, out_path: Path) -> dict[str, int]:
    """Scan every granule under ``ged_dir`` into one compact global artifact.

    The artifact stores the *unbuffered* gap cells (``NumObs == 0``) as
    global cell indices plus a 1-degree coverage grid of which granules
    exist. The buffer is applied at load time from
    ``settings.ged_gap_buffer_cells``, so a buffer change never needs a
    rebuild. Gap cells are rare (0.24% of cells on S30W065), which is why a
    sparse index list beats the 0.93 GB dense global grid.

    Returns:
        Counts for the build report: granules scanned and gap cells found.
    """
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    coverage = np.zeros((180, 360), dtype=np.uint8)

    granules = sorted(ged_dir.glob("AG1km.v003.*.0010.h5"))
    if not granules:
        msg = f"no AG1km.v003.*.0010.h5 granules under {ged_dir}"
        raise FileNotFoundError(msg)

    for path in granules:
        parts = path.name.split(".")
        lat_top, lon_west = int(parts[2]), int(parts[3])
        coverage[90 - lat_top, lon_west + 180] = 1
        numobs = read_granule_numobs(path)
        gap_r, gap_c = np.nonzero(numobs == 0)
        if gap_r.size:
            row0 = (90 - lat_top) * GRANULE_CELLS
            col0 = (lon_west + 180) * GRANULE_CELLS
            rows.append((gap_r + row0).astype(np.int32))
            cols.append((gap_c + col0).astype(np.int32))

    gap_rows = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    gap_cols = np.concatenate(cols) if cols else np.empty(0, dtype=np.int32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        format_version=np.int32(ARTIFACT_FORMAT_VERSION),
        gap_rows=gap_rows,
        gap_cols=gap_cols,
        coverage=coverage,
        granule_count=np.int32(len(granules)),
    )
    return {"granules": len(granules), "gap_cells": int(gap_rows.size)}


def _load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        version = int(data["format_version"])
        if version != ARTIFACT_FORMAT_VERSION:
            msg = (
                f"GED gap artifact {path} is format v{version}, this code reads "
                f"v{ARTIFACT_FORMAT_VERSION}; rebuild it with "
                "scripts/build_ged_gap_mask.py"
            )
            raise ValueError(msg)
        return {"gap_rows": data["gap_rows"], "gap_cols": data["gap_cols"]}


def _cells_from_artifact(
    artifact: Path, *, row0: int, row1: int, col0: int, col1: int
) -> np.ndarray:
    """Cut the [row0:row1, col0:col1] window of gap cells out of the artifact."""
    data = _load_artifact(artifact)
    cells = np.zeros((row1 - row0, col1 - col0), dtype=bool)
    rows, cols = data["gap_rows"].astype(np.int64), data["gap_cols"].astype(np.int64)
    # A window past the antimeridian addresses columns modulo the globe.
    for shift in (0, GLOBAL_CELL_COLS, -GLOBAL_CELL_COLS):
        c = cols + shift
        inside = (rows >= row0) & (rows < row1) & (c >= col0) & (c < col1)
        cells[rows[inside] - row0, c[inside] - col0] = True
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
    """Mosaic NumObs == 0 over the cell window, reading granules from disk.

    ``core`` is the un-padded cell window of the geobox itself. A granule
    covering any core cell must exist; absences are collected and raised
    together as one :class:`MissingGranuleError`, so the operator sees the
    whole shopping list at once -- a missing core granule means unmasked
    pixels inside the tile. A granule touching only the *margin* ring (the
    ``buffer_cells`` pad, where a neighbouring tile's gap cell could buffer
    across the edge) may be absent: the AG100 collection genuinely lacks some
    1-degree cells (e.g. ``AG1km.v003.-34.-066`` beside S30W065, with both
    its neighbours present), and failing the tile for a fringe the artifact
    path also cannot see would make the two paths disagree. The absence is
    logged, never silent.
    """
    core_row0, core_row1, core_col0, core_col1 = core
    cells = np.zeros((row1 - row0, col1 - col0), dtype=bool)
    missing_core: list[str] = []
    missing_margin: list[str] = []
    for grow in range(row0 // GRANULE_CELLS, (row1 - 1) // GRANULE_CELLS + 1):
        lat_top = 90 - grow
        g_r0 = grow * GRANULE_CELLS
        for gcol in range(col0 // GRANULE_CELLS, (col1 - 1) // GRANULE_CELLS + 1):
            lon_west = (gcol % 360) - 180
            g_c0 = gcol * GRANULE_CELLS
            name = granule_name(lat_top, lon_west)
            path = ged_dir / name
            if not path.exists():
                touches_core = (
                    g_r0 < core_row1
                    and g_r0 + GRANULE_CELLS > core_row0
                    and g_c0 < core_col1
                    and g_c0 + GRANULE_CELLS > core_col0
                )
                (missing_core if touches_core else missing_margin).append(name)
                continue
            numobs = read_granule_numobs(path)
            gap = numobs == 0
            # Granule's global cell window, clipped to the requested one.
            r_lo, r_hi = max(g_r0, row0), min(g_r0 + GRANULE_CELLS, row1)
            c_lo, c_hi = max(g_c0, col0), min(g_c0 + GRANULE_CELLS, col1)
            cells[r_lo - row0 : r_hi - row0, c_lo - col0 : c_hi - col0] = gap[
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


def _resolve_source() -> tuple[str, Path]:
    """Pick the artifact when present, granules otherwise; refuse silence."""
    artifact = settings.ged_artifact
    if artifact.exists():
        return "artifact", artifact
    ged_dir = settings.ged_dir
    if ged_dir.is_dir():
        return "granules", ged_dir
    msg = (
        f"no GED gap-mask source: artifact {artifact} does not exist and "
        f"granule directory {ged_dir} does not exist. The composite must not "
        "ship unmasked; build the artifact with scripts/build_ged_gap_mask.py, "
        "or set LST_GED_ARTIFACT / LST_GED_DIR, or disable the mask explicitly "
        "with LST_GED_GAP_MASK=false."
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
    a gap cell just outside the tile buffers into it. The verification pass
    clipped its dilation at the tile edge instead; on S30W065 the two agree
    everywhere because no edge granule carries a boundary-adjacent gap cell,
    and the padded form is the correct one for the general tile.

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

    height, width = int(geobox.shape[0]), int(geobox.shape[1])
    t = geobox.transform
    lon_centers = t.c + t.a * (np.arange(width) + 0.5)
    lat_centers = t.f + t.e * (np.arange(height) + 0.5)

    row_cells = np.floor((90.0 - lat_centers) * GED_CELLS_PER_DEGREE).astype(np.int64)
    col_cells = np.floor((lon_centers + 180.0) * GED_CELLS_PER_DEGREE).astype(np.int64)

    core = (
        int(row_cells[0]),
        int(row_cells[-1]) + 1,
        int(col_cells[0]),
        int(col_cells[-1]) + 1,
    )
    row0 = max(core[0] - buffer_cells, 0)
    row1 = min(core[1] + buffer_cells, GLOBAL_CELL_ROWS)
    col0 = core[2] - buffer_cells
    col1 = core[3] + buffer_cells

    kind, source = _resolve_source()
    if kind == "artifact":
        cells = _cells_from_artifact(source, row0=row0, row1=row1, col0=col0, col1=col1)
    else:
        cells = _cells_from_granules(source, row0=row0, row1=row1, col0=col0, col1=col1, core=core)

    cells = dilate_cells(cells, buffer_cells)
    return cells[np.ix_(row_cells - row0, col_cells - col0)]
