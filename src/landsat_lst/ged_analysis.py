"""Cross-tab a published LST tile against ASTER GED observation counts.

The ``NumObs == 0`` output mask shipped in #116 rests on one per-pixel pass
over S30W065. That pass lived in an agent worktree and depended on a scratch
path, so its numbers could not be re-derived. This module is that calculation,
tracked: it takes a published COG and a GED granule archive, states both
identities, and walks every output pixel once.

**It measures spatial association, not causation.** Every pixel is placed in
the ~1 km GED cell it falls inside, and the table says how the valid, missing,
and hot-tail pixels distribute across observation counts. It does not trace
which ASTER observations USGS used to retrieve any given Landsat pixel's
emissivity, so it cannot on its own show that interpolated emissivity *caused*
a hot retrieval. Any claim at that level needs observation-level provenance
this calculation does not have.

The grid mapping is :func:`landsat_lst.ged.cell_indices_for_geobox` -- the same
function :func:`landsat_lst.ged.gap_mask_for_geobox` masks with, so the analysis
and production can never disagree about which cell a pixel is in.

Memory is bounded by the row band, not the tile. An 18,000-squared tile is
walked in ``block_rows``-row strips; nothing full-tile is ever materialised at
more than one byte per pixel (the GED window itself is 500x500 cells).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from landsat_lst import ged
from landsat_lst.tiling import geobox_for_bbox, parse_tile_name

if TYPE_CHECKING:
    from pathlib import Path

#: Bumped when the tiering, the denominators, or the record layout change. A
#: stored result carrying a different version was produced by different code.
ANALYSIS_VERSION = "1.0.0"

#: The artifact tail the mask was adopted against (docs/findings-aster-ged-gaps.md).
DEFAULT_HOT_THRESHOLD_C = 70.0

#: Rows are walked in strips this tall by default. 512 rows of an 18,000-wide
#: tile is 9.2 M pixels, so every per-block intermediate stays under ~75 MB.
DEFAULT_BLOCK_ROWS = 512

#: Tier labels, in report order. ``absent`` is not an observation count: it is
#: a cell the AG100 collection holds no granule for, kept as its own row so the
#: tier counts sum to the tile exactly rather than silently absorbing it.
TIER_LABELS = ("0", "1", "2", "3", ">=4", "absent")
_TIER_ABSENT = 5

#: Pixel states, in the order they are packed into the histogram.
_STATE_MISSING, _STATE_VALID_COOL, _STATE_VALID_HOT = 0, 1, 2
_N_STATES = 3


class AnalysisInputError(ValueError):
    """The raster and the requested tile do not describe the same grid."""


def tier_codes(numobs: np.ndarray) -> np.ndarray:
    """Map NumObs values onto :data:`TIER_LABELS` indices.

    ``0``, ``1``, ``2``, ``3`` are their own tiers, every count of four or more
    folds into ``>=4``, and :data:`landsat_lst.ged.NUMOBS_ABSENT` becomes
    ``absent``. The mapping is total: every input value lands in exactly one
    tier, which is what makes the cross-tab conserve counts.
    """
    codes = np.full(numobs.shape, _TIER_ABSENT, dtype=np.uint8)
    present = numobs >= 0
    codes[present] = np.minimum(numobs[present], 4).astype(np.uint8)
    return codes


def threshold_dn(threshold_c: float, scale: float, offset: float) -> int:
    """The smallest stored DN whose physical value reaches ``threshold_c``.

    The COG stores ``value = DN * scale + offset``, so the comparison is done
    in integers rather than floats: at scale 0.01 and offset -50, 70 degC is
    DN 12000 exactly, and testing ``dn >= 12000`` cannot be moved by a rounding
    step the way ``dn * 0.01 - 50 >= 70`` can.
    """
    if scale <= 0:
        msg = f"raster scale must be positive, got {scale}"
        raise AnalysisInputError(msg)
    exact = (threshold_c - offset) / scale
    # Nudge before the ceiling so a value that is integral in exact arithmetic
    # is not pushed up one DN by float representation error.
    return math.ceil(exact - 1e-9)


@dataclass(frozen=True)
class RasterIdentity:
    """Everything about the input raster a reader needs to re-derive the table."""

    source: str
    width: int
    height: int
    crs: str
    transform: list[float]
    bounds: list[float]
    dtype: str
    nodata: float | None
    scale: float
    offset: float
    block_shape: list[int]
    overviews: list[int]
    scene_count: int | None
    scene_count_source: str
    tags: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        """The record's ``raster`` block, JSON-ready."""
        return {
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "transform": self.transform,
            "bounds": self.bounds,
            "dtype": self.dtype,
            "nodata": self.nodata,
            "scale": self.scale,
            "offset": self.offset,
            "block_shape": self.block_shape,
            "overviews": self.overviews,
            "scene_count": self.scene_count,
            "scene_count_source": self.scene_count_source,
            "tags": self.tags,
        }


@dataclass
class _Accumulator:
    """Histogram over (tier, state), plus the mask-rule tallies."""

    hist: np.ndarray = field(
        default_factory=lambda: np.zeros(len(TIER_LABELS) * _N_STATES, dtype=np.int64)
    )
    rules: dict[str, np.ndarray] = field(default_factory=dict)

    def rule(self, name: str) -> np.ndarray:
        """The ``(cool, hot, missing)`` tally for one candidate mask rule."""
        return self.rules.setdefault(name, np.zeros(3, dtype=np.int64))


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _pct(rate: float | None) -> float | None:
    return None if rate is None else 100.0 * rate


def _enrichment(tier_rate: float | None, base_rate: float | None) -> float | None:
    if tier_rate is None or not base_rate:
        return None
    return tier_rate / base_rate


def _identity(src: Any, source: str) -> RasterIdentity:
    tags = dict(src.tags())
    raw = tags.get("scene_count")
    scene_count: int | None = None
    scene_count_source = (
        "unknown: the COG carries no scene_count tag and the scene list is not "
        "recoverable from the published asset"
    )
    if raw is not None:
        try:
            scene_count = int(raw)
        except ValueError:
            scene_count_source = f"unknown: scene_count tag {raw!r} is not an integer"
        else:
            scene_count_source = "GDAL dataset tag 'scene_count' on the published COG"
    scales = src.scales or (1.0,)
    offsets = src.offsets or (0.0,)
    return RasterIdentity(
        source=source,
        width=int(src.width),
        height=int(src.height),
        crs=str(src.crs),
        transform=[float(v) for v in src.transform[:6]],
        bounds=[float(v) for v in src.bounds],
        dtype=str(src.dtypes[0]),
        nodata=None if src.nodata is None else float(src.nodata),
        scale=float(scales[0]),
        offset=float(offsets[0]),
        block_shape=[int(v) for v in src.block_shapes[0]],
        overviews=[int(v) for v in src.overviews(1)],
        scene_count=scene_count,
        scene_count_source=scene_count_source,
        tags=tags,
    )


def _geobox_from_raster(src: Any) -> Any:
    """The grid the raster's pixels are actually on.

    Cells are derived from this rather than from the tile name, so the table
    describes the file that was read. The tile name is a separate assertion
    (:func:`_check_grid`), not the source of the geometry.
    """
    from odc.geo.geobox import GeoBox  # noqa: PLC0415

    return GeoBox((int(src.height), int(src.width)), src.transform, src.crs)


def _check_grid(identity: RasterIdentity, tile: str) -> dict[str, Any]:
    """Confirm the raster sits on the tile's share of the global grid.

    The cross-tab derives cells from the raster's *own* affine, so this is not
    load-bearing for the numbers -- but a raster that is not the tile it claims
    to be is a different question being answered, and the record should say so.
    """
    geobox = geobox_for_bbox(parse_tile_name(tile).bbox)
    expected = [float(v) for v in geobox.transform[:6]]
    shape_ok = (int(geobox.shape[0]), int(geobox.shape[1])) == (
        identity.height,
        identity.width,
    )
    transform_ok = all(
        math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
        for a, b in zip(expected, identity.transform, strict=True)
    )
    if not (shape_ok and transform_ok):
        msg = (
            f"raster {identity.source} does not sit on tile {tile}'s grid: "
            f"shape {(identity.height, identity.width)} vs "
            f"{(int(geobox.shape[0]), int(geobox.shape[1]))}, transform "
            f"{identity.transform} vs {expected}"
        )
        raise AnalysisInputError(msg)
    return {
        "tile": tile,
        "expected_shape": [int(geobox.shape[0]), int(geobox.shape[1])],
        "expected_transform": expected,
        "matches_global_grid": True,
    }


def _walk(
    src: Any,
    *,
    cell_tiers: np.ndarray,
    row_cells: np.ndarray,
    col_cells: np.ndarray,
    rule_cells: dict[str, np.ndarray],
    hot_dn: int,
    fill_dn: int,
    block_rows: int,
) -> _Accumulator:
    """One windowed pass over the raster, accumulating every tally."""
    from rasterio.windows import Window  # noqa: PLC0415

    acc = _Accumulator()
    n_codes = len(TIER_LABELS) * _N_STATES
    width = int(src.width)
    for row0 in range(0, int(src.height), block_rows):
        rows = min(block_rows, int(src.height) - row0)
        # from_slices, not the Window constructor: attrs generates the latter's
        # signature at runtime, so a static checker sees no parameters at all.
        window = Window.from_slices((row0, row0 + rows), (0, width))
        dn = src.read(1, window=window)
        index = np.ix_(row_cells[row0 : row0 + rows], col_cells)

        missing = dn == fill_dn
        hot = dn >= hot_dn
        state = np.where(
            missing, _STATE_MISSING, np.where(hot, _STATE_VALID_HOT, _STATE_VALID_COOL)
        )
        code = cell_tiers[index].astype(np.int64) * _N_STATES + state
        acc.hist += np.bincount(code.ravel(), minlength=n_codes)

        for name, cells in rule_cells.items():
            removed = cells[index]
            tally = acc.rule(name)
            tally[0] += int(np.count_nonzero(removed & ~missing & ~hot))
            tally[1] += int(np.count_nonzero(removed & hot))
            tally[2] += int(np.count_nonzero(removed & missing))
    return acc


def _cross_tab(acc: _Accumulator) -> dict[str, Any]:
    hist = acc.hist.reshape(len(TIER_LABELS), _N_STATES)
    totals = hist.sum(axis=1)
    tile_total = int(totals.sum())
    tile_missing = int(hist[:, _STATE_MISSING].sum())
    tile_hot = int(hist[:, _STATE_VALID_HOT].sum())
    tile_valid = tile_total - tile_missing
    base_hot = _rate(tile_hot, tile_valid)
    base_missing = _rate(tile_missing, tile_total)

    rows = []
    for i, label in enumerate(TIER_LABELS):
        total = int(totals[i])
        missing = int(hist[i, _STATE_MISSING])
        hot = int(hist[i, _STATE_VALID_HOT])
        valid = total - missing
        rows.append(
            {
                "tier": label,
                "total_pixels": total,
                "valid_pixels": valid,
                "missing_pixels": missing,
                "hot_pixels": hot,
                "share_of_tile_pct": _pct(_rate(total, tile_total)),
                "valid_pct_of_tier": _pct(_rate(valid, total)),
                "missing_pct_of_tier": _pct(_rate(missing, total)),
                "hot_pct_of_tier_valid": _pct(_rate(hot, valid)),
                "share_of_tile_hot_pct": _pct(_rate(hot, tile_hot)),
                "share_of_tile_missing_pct": _pct(_rate(missing, tile_missing)),
                "hot_enrichment_vs_tile": _enrichment(_rate(hot, valid), base_hot),
                "missing_enrichment_vs_tile": _enrichment(_rate(missing, total), base_missing),
            }
        )

    return {
        "tile_totals": {
            "total_pixels": tile_total,
            "valid_pixels": tile_valid,
            "missing_pixels": tile_missing,
            "hot_pixels": tile_hot,
            "valid_pct": _pct(_rate(tile_valid, tile_total)),
            "missing_pct": _pct(_rate(tile_missing, tile_total)),
            "hot_pct_of_valid": _pct(base_hot),
        },
        "by_numobs_tier": rows,
    }


def _tradeoffs(acc: _Accumulator, totals: dict[str, Any]) -> list[dict[str, Any]]:
    tile_valid = totals["valid_pixels"]
    tile_hot = totals["hot_pixels"]
    tile_missing = totals["missing_pixels"]
    out = []
    for name, tally in acc.rules.items():
        cool_removed, hot_removed, missing_annotated = (int(v) for v in tally)
        valid_removed = cool_removed + hot_removed
        out.append(
            {
                "rule": name,
                "valid_pixels_removed": valid_removed,
                "valid_pixels_removed_pct": _pct(_rate(valid_removed, tile_valid)),
                "hot_pixels_removed": hot_removed,
                "hot_pixels_removed_pct": _pct(_rate(hot_removed, tile_hot)),
                "hot_pixels_remaining": tile_hot - hot_removed,
                "cool_valid_pixels_removed": cool_removed,
                "missing_pixels_annotated": missing_annotated,
                "missing_pixels_annotated_pct": _pct(_rate(missing_annotated, tile_missing)),
                "cool_pixels_lost_per_hot_pixel_removed": (
                    None if hot_removed == 0 else cool_removed / hot_removed
                ),
            }
        )
    return out


def analyze(
    *,
    raster: str,
    tile: str | None,
    ged_dir: Path,
    hot_threshold_c: float = DEFAULT_HOT_THRESHOLD_C,
    buffer_cells: int = 1,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    ged_version: str = "ASTER GED AG100 v003 (AG1km.v003, 0010)",
) -> dict[str, Any]:
    """Cross-tab every pixel of ``raster`` by the GED NumObs tier it falls in.

    Args:
        raster: Path or URL of the published single-band LST COG. Read
            windowed, so an HTTP source transfers only what it is asked for.
        tile: Tile name the raster is expected to be, e.g. ``S30W065``. The
            raster's grid is asserted against that tile's share of the global
            grid; ``None`` skips the assertion, for a raster that is a
            deliberate cut-out rather than a published tile. Either way the
            cells come from the raster's own affine.
        ged_dir: Directory of AG100 v003 granules.
        hot_threshold_c: Lower bound of the artifact tail, in Celsius.
        buffer_cells: Dilation radius for the candidate mask rules.
        block_rows: Rows per windowed read.
        ged_version: Recorded verbatim; the granules carry no version dataset.

    Returns:
        The machine-readable record: inputs, cross-tab, and mask tradeoffs.
    """
    import rasterio  # noqa: PLC0415

    with rasterio.open(raster) as src:
        identity = _identity(src, str(raster))
        grid = None if tile is None else _check_grid(identity, tile)
        if identity.nodata is None:
            msg = f"{raster} declares no nodata; missing pixels are not identifiable"
            raise AnalysisInputError(msg)

        geobox = _geobox_from_raster(src)
        numobs, row_cells, col_cells = ged.numobs_for_geobox(
            geobox, ged_dir, pad_cells=buffer_cells
        )
        cell_tiers = tier_codes(numobs)
        gap = numobs == 0
        low = (numobs >= 0) & (numobs <= 2)
        rule_cells = {
            "numobs==0": gap,
            f"numobs==0 + {buffer_cells}-cell buffer": ged.dilate_cells(gap, buffer_cells),
            "numobs<=2": low,
            f"numobs<=2 + {buffer_cells}-cell buffer": ged.dilate_cells(low, buffer_cells),
        }
        hot_dn = threshold_dn(hot_threshold_c, identity.scale, identity.offset)
        acc = _walk(
            src,
            cell_tiers=cell_tiers,
            row_cells=row_cells,
            col_cells=col_cells,
            rule_cells=rule_cells,
            hot_dn=hot_dn,
            fill_dn=int(identity.nodata),
            block_rows=block_rows,
        )

    table = _cross_tab(acc)
    return {
        "analysis_version": ANALYSIS_VERSION,
        "association_only": (
            "Pixels are cross-tabbed by the GED cell they fall inside. This is a "
            "spatial association between output pixels and ASTER observation "
            "counts, not a trace of which observations produced any pixel's "
            "emissivity, and it does not on its own establish causation."
        ),
        "raster": identity.as_dict(),
        "grid_check": grid,
        "ged": {
            "version": ged_version,
            "source_kind": "granules",
            "cells_per_degree": ged.GED_CELLS_PER_DEGREE,
            "cell_window_shape": [int(numobs.shape[0]), int(numobs.shape[1])],
            "buffer_cells": buffer_cells,
            "absent_cells": int(np.count_nonzero(numobs == ged.NUMOBS_ABSENT)),
            "gap_cells": int(np.count_nonzero(gap)),
        },
        "threshold": {
            "hot_threshold_c": hot_threshold_c,
            "hot_threshold_dn": hot_dn,
            "fill_dn": int(identity.nodata),
            "rule": "valid pixels with DN >= hot_threshold_dn",
        },
        **table,
        "mask_tradeoffs": _tradeoffs(acc, table["tile_totals"]),
    }
