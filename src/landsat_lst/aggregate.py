"""Spatial aggregation from the source grid to the delivered nominal ~100 m grid.

USGS acquires TIRS thermal radiance at 100 m and delivers Collection 2 ``ST_B10``
resampled onto a 30 m grid. Delivery spacing is not resolving power, so V1
publishes one nominal ~100 m product: every masked solar-day observation is
reduced from aligned 3x3 source blocks to one delivered cell *before* the
temporal percentile. See ADR-017 and issue #120.

Three properties define the contract, and each is pinned in
``tests/unit/test_aggregate.py``:

- **Exact alignment.** ``output_pixels_per_degree`` divides
  ``pixels_per_degree``, and both grids are cut from global arrays sharing an
  origin, so an output cell covers source cells ``[3i, 3i+3)`` with no
  remainder and no interpolation. A ragged edge is an error, never a trimmed
  partial block.
- **Valid cells only.** The reducer is an area-weighted mean over the *valid*
  source cells. A masked cell contributes nothing to the numerator and nothing
  to the denominator; a fill value or a zero can never enter the mean. Below
  ``min_valid_source_cells`` the whole block is nodata for that observation.
- **Area weights, from the cells' real areas.** In EPSG:4326 a cell's area is
  ``R^2 * dlon * (sin(lat_top) - sin(lat_bottom))``, so within one block the
  three source *rows* differ and the three columns do not. The variation is
  tiny: relative spread across a block is ``tan(lat) * (factor - 1) * dlat``,
  which at 60 degrees -- the edge of the published latitude band, and the worst
  case -- is 1.7e-5, and less than half that at 40 degrees. An unweighted mean
  would agree to well inside float32. The weights are computed anyway, because
  a reducer whose correctness rests on the error being small is a reducer
  nobody can check, and because the same code has to stay right if the factor
  or the grid ever moves. ``tests/unit/test_aggregate.py`` measures the spread
  rather than restating it here.

**This module never touches the offset estimator.** Offsets are estimated on a
coarsening of the *source* grid (``settings.offset_resolution``), the accuracy
of that choice was calibrated against the source grid
(docs/findings-offset-subsampling.md), and the correction is a per-scene scalar
applied after aggregation. A scalar commutes with a weighted mean exactly, so
the ordering costs nothing and keeps the two grids independent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog
import xarray as xr

from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

log = structlog.get_logger()

#: Bumped when a change to this module would change the delivered values.
#: Enters ``shards.TilePlan.digest``, so a plan cut under one aggregation
#: contract cannot be completed by a shard running another.
AGGREGATION_VERSION = 1

_TIME_DIM = "time"


def spatial_dims(lst: xr.DataArray) -> list[str]:
    """The ``(row, column)`` dims of a stack: every dim but time.

    Spelled the same way :func:`landsat_lst.normalization._spatial_dims` spells
    it, and for the same reason. A real load names them ``latitude`` and
    ``longitude``; synthetic and benchmark stacks name them ``y`` and ``x``,
    and the reducer must not care which.

    Raises:
        ValueError: Unless there are exactly two.
    """
    dims = [str(d) for d in lst.dims if d != _TIME_DIM]
    if len(dims) != 2:
        msg = (
            f"aggregation needs exactly two spatial dims, got {dims} from "
            f"{lst.dims}. A stack with more axes than (time, row, column) is "
            "not something an aligned block reducer can interpret."
        )
        raise ValueError(msg)
    return dims


def aligned_source_chunk(size: int, factor: int | None = None) -> int:
    """Round a source-grid chunk edge up to a whole number of output cells.

    A chunk that straddles an output cell forces ``coarsen`` to rechunk the
    whole stack before it can reduce anything, and the rechunk it picks is
    uneven: a 512 chunk at factor 3 came back as ``(170, 85, 86, 1)`` output
    chunks against the ``(171, 171)`` an aligned 513 gives. Correctness is
    unaffected either way -- this is about not paying a shuffle for a reduction
    that has none.

    Args:
        size: Requested chunk edge, in source pixels.
        factor: Source cells per output cell edge. Defaults to the configured
            :attr:`~landsat_lst.config.Settings.spatial_aggregation_factor`.

    Returns:
        The smallest multiple of ``factor`` that is at least ``size``.
    """
    factor = settings.spatial_aggregation_factor if factor is None else factor
    if factor <= 1:
        return size
    return factor * -(-size // factor)


def row_area_weights(lat_edges: np.ndarray) -> np.ndarray:
    """Relative area of each row of cells, from its latitude edges.

    A cell spanning constant ``dlon`` between two latitudes has area
    proportional to ``sin(lat_top) - sin(lat_bottom)`` on a sphere. That is
    exact rather than the ``cos(lat)`` small-angle form, and it costs nothing.

    Args:
        lat_edges: ``height + 1`` latitude edges in degrees, in grid order
            (descending for the north-down grids this project uses).

    Returns:
        ``height`` positive weights, in the same order. Only their ratios
        matter; the reducer normalizes.
    """
    sines = np.sin(np.deg2rad(np.asarray(lat_edges, dtype=np.float64)))
    return np.abs(np.diff(sines))


def source_row_weights(latitudes: xr.DataArray | np.ndarray, spacing: float) -> np.ndarray:
    """Area weights for source rows given their centre latitudes and spacing.

    ``spacing`` is signed as the grid's affine has it (negative for north-down),
    and only its magnitude is used.
    """
    centres = np.asarray(latitudes, dtype=np.float64)
    half = abs(float(spacing)) / 2.0
    edges = np.empty(centres.size + 1, dtype=np.float64)
    descending = centres.size > 1 and centres[1] < centres[0]
    if descending:
        edges[:-1] = centres + half
        edges[-1] = centres[-1] - half
    else:
        edges[:-1] = centres - half
        edges[-1] = centres[-1] + half
    return row_area_weights(edges)


def _spacing(latitudes: xr.DataArray) -> float:
    """Latitude step of a loaded grid, taken from the coordinate itself."""
    values = np.asarray(latitudes, dtype=np.float64)
    if values.size >= 2:
        return float(values[1] - values[0])
    return -settings.source_resolution


def _weights_for(lst: xr.DataArray, row_dim: str) -> xr.DataArray | None:
    """Per-row area weights, or ``None`` when the rows carry no latitude.

    A stack loaded through ``geobox=`` always carries its row coordinate, so
    production always weights. A synthetic stack built for a graph measurement
    often carries no coordinate at all, and there is no latitude to weight by;
    equal weights are then the only defensible reading, and they differ from
    the weighted answer by at most about ``tan(lat) * dlat``, which is 8.4e-6
    at 60 degrees. Returning ``None`` rather than a vector of ones also keeps
    that case out of the graph entirely.
    """
    if row_dim not in lst.coords:
        return None
    rows = lst[row_dim]
    if not np.issubdtype(np.asarray(rows).dtype, np.number):
        return None
    return xr.DataArray(
        source_row_weights(rows, _spacing(rows)).astype(lst.dtype, copy=False),
        dims=[row_dim],
        coords={row_dim: rows},
    )


def aggregate_to_output_grid(
    lst: xr.DataArray,
    *,
    factor: int | None = None,
    min_valid_cells: int | None = None,
    coords: Mapping[str, np.ndarray] | None = None,
) -> xr.DataArray:
    """Reduce a masked source-grid stack onto the delivered grid.

    Call this **after** QA masking, fill handling, scaling, and the
    plausibility clamp, and **before** the per-scene correction and the
    temporal percentile. Computing a percentile on the source grid and
    downsampling afterwards is a different statistic and is explicitly
    non-compliant with the V1 contract (issue #120).

    The time axis is untouched: one masked solar-day observation in, one
    delivered solar-day observation out, on the same time coordinate. That is
    what lets the offsets -- estimated on a different grid entirely -- still
    join by time value.

    Args:
        lst: Celsius LST with dims ``(time, latitude, longitude)``, NaN where
            masked. A 2-D array without a time dim is aggregated the same way.
        factor: Source cells per output cell edge. Defaults to the configured
            aggregation factor. ``1`` returns ``lst`` unchanged.
        min_valid_cells: Valid source cells an output cell needs. Defaults to
            ``settings.min_valid_source_cells``. The sensitivity arms pass 1,
            5, and 9 here rather than mutating the setting.
        coords: Delivered-grid coordinate values by dim name, normally from
            ``geobox_coords(output_geobox_for_bbox(...))``. Without them the
            labels are ``coarsen``'s per-block coordinate means, which land on
            the right cell centres only up to float64 round-off. That is enough
            to look right and not enough to *align*: xarray joins a mask to a
            stack on exact index equality, so a mask labelled from the geobox
            and a stack labelled from three averaged source centres can refuse
            to align at all. The grid definition is authoritative here for the
            same reason ``pixels_per_degree`` is an integer (ADR-008).

    Returns:
        A DataArray on the delivered grid, NaN where support fell below the
        threshold.

    Raises:
        ValueError: If a spatial dimension is not a whole multiple of
            ``factor``, or if ``min_valid_cells`` is outside ``1..factor**2``.
    """
    factor = settings.spatial_aggregation_factor if factor is None else factor
    if factor == 1:
        return lst
    if factor < 1:
        msg = f"aggregation factor must be positive, got {factor}"
        raise ValueError(msg)

    cells = factor * factor
    min_valid_cells = (
        settings.min_valid_source_cells if min_valid_cells is None else min_valid_cells
    )
    if not 1 <= min_valid_cells <= cells:
        msg = (
            f"min_valid_cells {min_valid_cells} is outside 1..{cells}, the source "
            f"cells an output cell covers at factor {factor}. A threshold of 0 "
            "would emit a temperature for a block with no observation behind it."
        )
        raise ValueError(msg)

    row_dim, col_dim = spatial_dims(lst)
    for dim in (row_dim, col_dim):
        size = int(lst.sizes[dim])
        if size % factor:
            msg = (
                f"{dim} is {size} source cells, not a whole multiple of {factor}. "
                "A delivered cell must cover a complete aligned block; trimming "
                "or padding the edge would put a partial block on the shared "
                "global grid. Load onto a geobox cut from the source grid whose "
                f"{dim} extent is a multiple of {factor}."
            )
            raise ValueError(msg)

    windows: dict[Hashable, int] = {row_dim: factor, col_dim: factor}
    valid = lst.notnull()
    weights = _weights_for(lst, row_dim)

    # fillna(0) is safe precisely because the same mask zeroes the denominator:
    # a masked cell adds 0 to the numerator and 0 to the weight sum, so it
    # neither pulls the mean toward zero nor counts as support. Multiplying the
    # mask by the weights rather than summing weights unconditionally is the
    # whole of the "fill values never enter the mean" rule.
    support = valid.coarsen(windows).sum()  # ty: ignore[unresolved-attribute]
    if weights is None:
        numerator = lst.fillna(0).coarsen(windows).sum()  # ty: ignore[unresolved-attribute]
        weight_sum = support
    else:
        numerator = (lst.fillna(0) * weights).coarsen(windows).sum()  # ty: ignore[unresolved-attribute]
        weight_sum = (valid * weights).coarsen(windows).sum()  # ty: ignore[unresolved-attribute]

    # Masking the denominator rather than the quotient keeps 0/0 out of the
    # graph entirely: below the threshold the divisor is NaN, so the result is
    # NaN with no invalid-value warning fired per block.
    aggregated = numerator / weight_sum.where(support >= min_valid_cells)
    aggregated = aggregated.astype(lst.dtype, copy=False)

    if coords:
        for dim, values in coords.items():
            if dim in aggregated.dims:
                expected = int(aggregated.sizes[dim])
                if len(values) != expected:
                    msg = (
                        f"delivered coordinate {dim!r} has {len(values)} values "
                        f"against {expected} aggregated cells. The geobox handed "
                        "in is not the aggregation of the one the stack loaded on."
                    )
                    raise ValueError(msg)
                aggregated = aggregated.assign_coords({dim: values})

    aggregated.attrs.update(lst.attrs)
    aggregated.attrs["aggregation_factor"] = factor
    aggregated.attrs["min_valid_source_cells"] = min_valid_cells
    aggregated.attrs["aggregation_version"] = AGGREGATION_VERSION
    return aggregated


def support_fraction(
    lst: xr.DataArray,
    *,
    factor: int | None = None,
) -> xr.DataArray:
    """Valid source cells per delivered cell, as a fraction of ``factor**2``.

    The diagnostic behind the valid-area rule: what the sensitivity sweep
    thresholds and what a coverage report describes. Kept beside the reducer so
    the two cannot disagree about which cells count as valid.
    """
    factor = settings.spatial_aggregation_factor if factor is None else factor
    if factor == 1:
        return lst.notnull().astype(np.float32)
    row_dim, col_dim = spatial_dims(lst)
    windows: dict[Hashable, int] = {row_dim: factor, col_dim: factor}
    return lst.notnull().coarsen(windows).sum() / float(factor * factor)  # ty: ignore[unresolved-attribute]
