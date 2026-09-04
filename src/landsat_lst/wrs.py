"""WRS-2 path geometry: swath polygons, per-pixel blend weights, step labels.

The composite pools every scene that overlaps a pixel, so where two WRS-2
paths overlap the P95 draws on both. Measured on S30W065, one path runs
+2.2 to +4.8 C warmer than its neighbour in the upper tail on identical ground
at matched observation counts, and the pooled percentile therefore steps where
that path's coverage stops. This module supplies the geometry that lets the
composite build one P95 per path and cross-fade them continuously instead.

Three rules hold this together:

**The swath polygon is a property of the tile, never of a row band.** Every
band derives it from the same item geometries at the same reduced resolution,
so bands cannot disagree and invent a seam at their own boundary. That is the
argument :func:`landsat_lst.masks.get_land_mask_for_geobox` and
:func:`landsat_lst.ged.gap_mask_for_geobox` already make about rasterising
against a geobox's own affine, one level up.

**The swath is the median footprint of each (path, row) quad**, the ground at
least half that quad's scenes reach -- not the union. A union overreaches: one
outlying scene extends the swath past where the path actually contributes, and
on the S30W065 diagnostic crop path 229's union covered 100% of the area and
left no edge to ramp toward. Grouping by quad rather than by path is what
makes the threshold meaningful, since a path's scenes span five rows and no
pixel is reached by more than one or two of them.

**Weights come from geometry alone.** Never from temperature, observation
count, or a fitted seam position.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import rasterio
import shapely
import structlog
import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.features import MergeAlg, rasterize, shapes
from rasterio.warp import Resampling, reproject
from shapely.geometry import shape
from shapely.ops import unary_union

from landsat_lst.config import settings

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

log = structlog.get_logger()

#: Marks a solar-day step whose items span more than one path.
MIXED_PATH = ""

_ROW_BLOCK = 256

#: Smallest coarse edge worth using. Below this the ramp is computed exactly.
#: Deliberately well under any production band: at 64 the floor sat exactly on
#: a 512-row band, so a tile whose bands came out a few rows shorter would drop
#: silently onto the exact path and pay 180 s instead of 4 s. This guard exists
#: to protect a degenerate grid, not to adjudicate production geometry.
_MIN_COARSE_EDGE = 16


def _property(item, name: str) -> str:
    try:
        return str(item.properties[name])
    except KeyError:
        msg = (
            f"STAC item {getattr(item, 'id', '?')!r} carries no {name!r}. "
            "Every landsat-c2-l2 item does; feathering cannot assign a path "
            "without it. Set settings.wrs_feather=False to composite pooled."
        )
        raise KeyError(msg) from None


def path_of(item) -> str:
    """The item's WRS-2 path as a string, zero padding preserved."""
    return _property(item, "landsat:wrs_path")


def _row_of(item) -> str:
    return _property(item, "landsat:wrs_row")


def quad_groups(items: list) -> dict[tuple[str, str], list]:
    """Items grouped by ``(path, row)``."""
    out: dict[tuple[str, str], list] = defaultdict(list)
    for item in items:
        out[(path_of(item), _row_of(item))].append(item)
    return dict(out)


def _footprint_geobox(items: list, tile_geobox: GeoBox) -> GeoBox:
    """A coarse grid covering every footprint, not just the tile.

    Rasterising on the tile's own extent would clip each swath at the tile
    border, and the polygon's boundary would then include that border. Weights
    would ramp toward the edge of the tile and two neighbouring tiles would
    disagree along their shared edge -- a new seam, on the tile grid instead of
    the WRS grid. Covering the footprints' own bounds keeps every swath edge a
    real acquisition edge.
    """
    bounds = unary_union([shape(i.geometry) for i in items]).bounds
    west, south, east, north = bounds
    step = float(abs(tile_geobox.transform.a)) * settings.wrs_swath_factor
    west, south = west - step, south - step
    east, north = east + step, north + step
    height = max(int(np.ceil((north - south) / step)), 1)
    width = max(int(np.ceil((east - west) / step)), 1)
    return GeoBox.from_bbox(
        (west, south, west + width * step, south + height * step),
        crs=tile_geobox.crs,
        shape=(height, width),
    )


def swath_polygons(items: list, tile_geobox: GeoBox) -> dict[str, BaseGeometry]:
    """Median-footprint polygon per WRS path, on the whole tile.

    Each ``(path, row)`` quad is rasterised at
    ``settings.wrs_swath_factor`` and thresholded at half its own scene count;
    a path's swath is the union of its quads' regions. The reduced grid keeps
    this cheap and, more importantly, keeps it identical for every row band of
    the tile.

    Returns:
        ``{path: polygon}``, empty for a path no pixel of the tile retains.
    """
    coarse = _footprint_geobox(items, tile_geobox)
    height, width = coarse.shape[0], coarse.shape[1]
    by_path: dict[str, list[BaseGeometry]] = defaultdict(list)

    for (path, row), members in sorted(quad_groups(items).items()):
        count = np.zeros((height, width), dtype=np.uint16)
        rasterize(
            ((shape(i.geometry), 1) for i in members),
            out=count,
            transform=coarse.transform,
            merge_alg=MergeAlg.add,
        )
        # "at least half this quad's scenes reach here"
        keep = count >= max(len(members) / 2.0, 1.0)
        if not keep.any():
            log.info("wrs_quad_empty", path=path, row=row, scenes=len(members))
            continue
        polys = [
            shape(geom)
            for geom, value in shapes(keep.astype(np.uint8), mask=keep, transform=coarse.transform)
            if value == 1
        ]
        if polys:
            by_path[path].append(unary_union(polys))

    return {path: unary_union(parts) for path, parts in sorted(by_path.items()) if parts}


def _exact_distances(
    geobox: GeoBox, polygons, paths, inside: np.ndarray, multi: np.ndarray
) -> np.ndarray:
    """Point-to-boundary distance on ``geobox``'s own grid, where it matters."""
    n, height, width = inside.shape
    a = geobox.transform
    dist = np.zeros((n, height, width), dtype=np.float32)
    boundaries = [polygons[p].boundary for p in paths]
    lon = a.c + a.a * (np.arange(width, dtype=np.float64) + 0.5)
    for y0 in range(0, height, _ROW_BLOCK):
        y1 = min(y0 + _ROW_BLOCK, height)
        block = multi[y0:y1]
        if not block.any():
            continue
        lat = a.f + a.e * (np.arange(y0, y1, dtype=np.float64) + 0.5)
        yy, xx = np.nonzero(block)
        px = shapely.points(lon[xx], lat[yy])
        for j in range(n):
            sel = inside[j, y0:y1][yy, xx]
            if not sel.any():
                continue
            d = np.zeros(px.size, dtype=np.float64)
            d[sel] = shapely.distance(px[sel], boundaries[j])
            tgt = dist[j, y0:y1]
            tgt[yy, xx] = d.astype(np.float32)
    return dist


def _coarse_distances(geobox: GeoBox, polygons, paths, factor: int) -> np.ndarray:
    """The distance ramp computed coarse and resampled onto ``geobox``.

    The ramp is smooth over tens of kilometres, so a grid ``factor`` pixels
    coarser carries it to well under 1% of a weight while cutting the point
    count by ``factor**2``.
    """
    coarse = geobox.zoom_out(factor)
    ch, cw = coarse.shape[0], coarse.shape[1]
    n = len(paths)
    c_inside = np.zeros((n, ch, cw), dtype=bool)
    for j, path in enumerate(paths):
        m = np.zeros((ch, cw), dtype=np.uint8)
        rasterize([(polygons[path], 1)], out=m, transform=coarse.transform)
        c_inside[j] = m.astype(bool)
    c_multi = c_inside.sum(axis=0) >= 2
    c_dist = _exact_distances(coarse, polygons, paths, c_inside, c_multi)

    height, width = geobox.shape[0], geobox.shape[1]
    out = np.zeros((n, height, width), dtype=np.float32)
    for j in range(n):
        reproject(
            source=c_dist[j],
            destination=out[j],
            src_transform=coarse.transform,
            src_crs=rasterio.crs.CRS.from_user_input(str(coarse.crs)),
            dst_transform=geobox.transform,
            dst_crs=rasterio.crs.CRS.from_user_input(str(geobox.crs)),
            resampling=Resampling.bilinear,
        )
    return out


def _blend(
    weight: np.ndarray,
    dist: np.ndarray,
    inside: np.ndarray,
    multi: np.ndarray,
    k: np.ndarray,
) -> None:
    """Turn distances into shares in place: ``w_j = d_j / sum_i d_i``.

    Renormalised on the exact containment masks, so an interpolated ramp still
    sums to one and still gives a single-path pixel exactly its own weight. On
    a boundary every distance is zero; equal shares are the answer there rather
    than a division by zero, and the pixel is a measure-zero line either way.
    """
    n = weight.shape[0]
    total = dist.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        for j in range(n):
            share = np.where(total > 0, dist[j] / np.where(total > 0, total, 1.0), 0.0)
            weight[j] = np.where(multi & inside[j], share.astype(np.float32), weight[j])
    wsum = weight.sum(axis=0)
    fix = multi & (wsum > 0)
    for j in range(n):
        weight[j] = np.where(fix, weight[j] / np.where(wsum > 0, wsum, 1.0), weight[j])
    degenerate = multi & (wsum <= 0)
    if degenerate.any():
        for j in range(n):
            sel = degenerate & inside[j]
            weight[j][sel] = 1.0 / k[sel]


@dataclass(frozen=True)
class PathWeights:
    """Per-path blend weights on one geobox.

    Attributes:
        paths: Path labels in ascending order. The order is canonical so that
            the weighted sum is evaluated in a fixed sequence and permuting the
            input items cannot change a floating-point result.
        weight: ``(n_paths, height, width)`` float32. Sums to 1 where any path
            covers, 0 elsewhere.
        covered: ``(height, width)`` bool, true where at least one path covers.
        n_paths_at_pixel: ``(height, width)`` uint8 count of covering paths.
    """

    paths: tuple[str, ...]
    weight: np.ndarray
    covered: np.ndarray
    n_paths_at_pixel: np.ndarray


def path_weights(
    geobox: GeoBox,
    polygons: dict[str, BaseGeometry],
    *,
    factor: int | None = None,
) -> PathWeights:
    """Cross-fade weights for ``polygons`` on ``geobox``'s own grid.

    Inside a pixel's covering set the weight is its distance to its own swath
    boundary over the sum of those distances,
    ``w_j = d_j / sum_i d_i``. With one covering path that is exactly 1, so a
    single-path pixel is untouched. With two it is the linear cross-fade that
    reaches 0 at one edge and 1 at the other. With three or more it stays
    continuous, which matters because 5.4% of S30W065 is reached by three.

    Distances are measured only where they can matter. A pixel reached by one
    path needs no distance at all, and only the covered pixels of a
    multi-path region are handed to shapely.

    ``factor`` (default :attr:`~landsat_lst.config.Settings.wrs_weight_factor`)
    computes the ramp on a grid that many pixels coarser and interpolates it
    back. Exact distance was 99.8% of the geometry cost -- 152-180 s a band
    against 0.04 s for containment -- because a swath polygon carries thousands
    of vertices and shapely walks them per point. Containment stays exact
    whatever the factor, so membership, single-path pixels and the sum-to-one
    property are unaffected; only the ratio between overlapping paths is
    interpolated. Pass ``factor=1`` for the exact ramp.
    """
    if factor is None:
        factor = settings.wrs_weight_factor
    height, width = geobox.shape[0], geobox.shape[1]
    paths = tuple(sorted(polygons))
    n = len(paths)
    if n == 0:
        return PathWeights(
            paths=(),
            weight=np.zeros((0, height, width), np.float32),
            covered=np.zeros((height, width), bool),
            n_paths_at_pixel=np.zeros((height, width), np.uint8),
        )

    inside = np.zeros((n, height, width), dtype=bool)
    for j, path in enumerate(paths):
        m = np.zeros((height, width), dtype=np.uint8)
        rasterize([(polygons[path], 1)], out=m, transform=geobox.transform)
        inside[j] = m.astype(bool)

    k = inside.sum(axis=0).astype(np.uint8)
    covered = k > 0
    weight = np.zeros((n, height, width), dtype=np.float32)

    # One covering path: weight 1, no geometry needed.
    single = k == 1
    if single.any():
        for j in range(n):
            weight[j][single & inside[j]] = 1.0

    multi = k >= 2
    # A grid too small to coarsen would carry the ramp on a handful of cells,
    # which is worse than the cost it saves. Below the floor, compute exactly.
    if multi.any():
        coarse_enough = (
            factor and factor > 1 and min(height // factor, width // factor) >= _MIN_COARSE_EDGE
        )
        dist = (
            _coarse_distances(geobox, polygons, paths, factor)
            if coarse_enough
            else _exact_distances(geobox, polygons, paths, inside, multi)
        )
        _blend(weight, dist, inside, multi, k)

    return PathWeights(paths=paths, weight=weight, covered=covered, n_paths_at_pixel=k)


def path_of_steps(items: list, bbox, time, resolution_factor: int = 1) -> xr.DataArray:
    """The WRS path behind each solar-day step of a loaded stack.

    ``load_scenes`` groups by solar day, so the loaded axis is steps and not
    items. This reproduces odc-stac's grouping the way
    :func:`landsat_lst.pipeline.scene_cloud_cover` does, and checks the derived
    stamps against the axis actually loaded rather than trusting the rule.

    Returned **on the time axis as a coordinate**, not as a bare array. The
    stack that reaches the composite is not the stack that was loaded: de-
    striping drops rejected scenes, so a 1,031-step load reaches
    ``_composite_graph`` with 912 steps. Labels carried positionally would
    then address the wrong scenes, or run off the end. Joining by coordinate
    value is the same rule ``normalization.debias_with_offsets`` follows for
    the offsets themselves.

    A step whose items span more than one path is labelled :data:`MIXED_PATH`.
    That is rare -- 3 of 1,031 steps on S30W065 -- but it is not zero, so the
    caller must decide rather than assume. Two paths can land on one solar day
    at opposite edges of a five-degree tile, and more often as they converge
    toward the latitude limits.
    """
    from odc.stac._mdtools import parse_items  # noqa: PLC0415
    from odc.stac._stac_load import _extract_timestamps, _group_items  # noqa: PLC0415

    from landsat_lst.tiling import geobox_for_bbox  # noqa: PLC0415

    parsed = list(parse_items(items))
    gbox = geobox_for_bbox(bbox, resolution_factor)
    ((mid_lon, _),) = gbox.extent.centroid.to_crs("epsg:4326").points
    grouped = _group_items(items, parsed, "solar_day", mid_lon)
    stamps = np.array(
        _extract_timestamps([[parsed[i] for i in g] for g in grouped]), dtype="datetime64[ns]"
    )
    loaded = time.values.astype("datetime64[ns]")
    if stamps.shape != loaded.shape or not np.array_equal(stamps, loaded):
        msg = (
            f"solar-day grouping does not reproduce the loaded time axis "
            f"({stamps.size} derived vs {loaded.size} loaded). "
            "odc-stac's grouping rule has changed; path_of_steps must follow it."
        )
        raise ValueError(msg)

    labels = []
    for group in grouped:
        found = {path_of(items[i]) for i in group}
        labels.append(found.pop() if len(found) == 1 else MIXED_PATH)
    return xr.DataArray(np.array(labels, dtype=object), dims=["time"], coords={"time": stamps})
