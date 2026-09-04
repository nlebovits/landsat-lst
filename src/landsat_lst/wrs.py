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
import shapely
import structlog
from odc.geo.geobox import GeoBox
from rasterio.features import MergeAlg, rasterize, shapes
from shapely.geometry import shape
from shapely.ops import unary_union

from landsat_lst.config import settings

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

log = structlog.get_logger()

#: Marks a solar-day step whose items span more than one path.
MIXED_PATH = ""

_ROW_BLOCK = 256


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


def path_weights(geobox: GeoBox, polygons: dict[str, BaseGeometry]) -> PathWeights:
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
    """
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
    if multi.any():
        a = geobox.transform
        dist = np.zeros((n, height, width), dtype=np.float32)
        boundaries = [polygons[p].boundary for p in paths]
        cols = np.arange(width, dtype=np.float64)
        lon = a.c + a.a * (cols + 0.5)
        for y0 in range(0, height, _ROW_BLOCK):
            y1 = min(y0 + _ROW_BLOCK, height)
            block = multi[y0:y1]
            if not block.any():
                continue
            rows = np.arange(y0, y1, dtype=np.float64)
            lat = a.f + a.e * (rows + 0.5)
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
        total = dist.sum(axis=0)
        # On a boundary every distance is 0. Fall back to equal shares rather
        # than dividing by zero; the pixel is a measure-zero line either way.
        degenerate = multi & (total <= 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            for j in range(n):
                share = np.where(total > 0, dist[j] / np.where(total > 0, total, 1.0), 0.0)
                weight[j] = np.where(multi, share.astype(np.float32), weight[j])
        if degenerate.any():
            for j in range(n):
                weight[j][degenerate & inside[j]] = 1.0 / k[degenerate & inside[j]]

    return PathWeights(paths=paths, weight=weight, covered=covered, n_paths_at_pixel=k)


def path_of_steps(items: list, bbox, time, resolution_factor: int = 1) -> np.ndarray:
    """The WRS path behind each solar-day step of a loaded stack.

    ``load_scenes`` groups by solar day, so the loaded axis is steps and not
    items. This reproduces odc-stac's grouping the way
    :func:`landsat_lst.pipeline.scene_cloud_cover` does, and checks the derived
    stamps against the axis actually loaded rather than trusting the rule.

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
    return np.array(labels, dtype=object)
