"""Discover finished COG pairs and read the facts the catalog needs from them.

A tile is publishable only when both of its assets exist. Half a tile is a
processing failure, not a catalog shape, so :func:`scan_source` refuses to
build and names the gaps rather than emitting an item with one asset.

Every number the items carry -- bounding box, raster size, band statistics,
byte count -- is read here, from the COG headers, so the catalog cannot drift
from the files it describes.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import rasterio

from landsat_lst.catalog.spec import LST_ASSET_KEY, QA_ASSET_KEY

_TIF_SUFFIX = ".tif"

# GDAL will happily read statistics out of a .aux.xml sidecar that never ships
# with the data. Portolan validators read with sidecars disabled, so the
# catalog is built from the same embedded tags they will check.
_HEADER_ENV = {"GDAL_PAM_ENABLED": "NO"}

_STAT_TAGS = {
    "minimum": "STATISTICS_MINIMUM",
    "maximum": "STATISTICS_MAXIMUM",
    "mean": "STATISTICS_MEAN",
    "stddev": "STATISTICS_STDDEV",
}


class IncompleteTileError(RuntimeError):
    """A tile carries one of its two assets, so the catalog cannot be built."""


@dataclass(frozen=True)
class SourceFile:
    """One finished COG in the source tree, wherever it lives."""

    uri: str
    name: str
    size: int


@dataclass(frozen=True)
class BandStats:
    """One band's embedded statistics, in the units the file stores."""

    description: str | None
    minimum: float | None
    maximum: float | None
    mean: float | None
    stddev: float | None


@dataclass(frozen=True)
class CogHeader:
    """What one COG says about itself."""

    file: SourceFile
    bbox: tuple[float, float, float, float]
    width: int
    height: int
    data_type: str
    nodata: float | None
    bands: tuple[BandStats, ...]


@dataclass(frozen=True)
class TilePair:
    """A tile's two assets, both confirmed present."""

    tile: str
    lst: CogHeader
    qa: CogHeader


def _list_local(root: Path) -> list[SourceFile]:
    """Every GeoTIFF under a local directory, at any depth."""
    return [
        SourceFile(uri=str(path), name=path.name, size=path.stat().st_size)
        for path in sorted(root.rglob(f"*{_TIF_SUFFIX}"))
        if path.is_file()
    ]


def _list_s3(source: str) -> list[SourceFile]:
    """Every GeoTIFF under an ``s3://bucket/prefix`` location."""
    import boto3  # noqa: PLC0415 - only the s3 path pays for the SDK import

    parsed = urlparse(source)
    prefix = parsed.path.lstrip("/")
    pages = boto3.client("s3").get_paginator("list_objects_v2")
    found: list[SourceFile] = []
    for page in pages.paginate(Bucket=parsed.netloc, Prefix=prefix):
        for obj in page.get("Contents", ()):
            key = obj["Key"]
            if key.endswith(_TIF_SUFFIX):
                name = key.rsplit("/", 1)[-1]
                uri = f"s3://{parsed.netloc}/{key}"
                found.append(SourceFile(uri=uri, name=name, size=obj["Size"]))
    return sorted(found, key=lambda item: item.uri)


def list_source(source: str | Path) -> list[SourceFile]:
    """Every GeoTIFF under a local directory or an ``s3://`` prefix."""
    text = str(source)
    if text.startswith("s3://"):
        return _list_s3(text)
    root = Path(text)
    if not root.is_dir():
        msg = f"source is not a directory: {root}"
        raise NotADirectoryError(msg)
    return _list_local(root)


def _tile_of(name: str, prefix: str, window: str) -> str | None:
    """The tile a filename names, or ``None`` when it is not one of ours."""
    head = f"{prefix}_{window}_"
    if not name.startswith(head) or not name.endswith(_TIF_SUFFIX):
        return None
    return name[len(head) : -len(_TIF_SUFFIX)]


def _index_by_tile(files: list[SourceFile], prefix: str, window: str) -> dict[str, SourceFile]:
    """Map tile name to the file of one asset kind."""
    return {
        tile: file for file in files if (tile := _tile_of(file.name, prefix, window)) is not None
    }


def _check_complete(lst: dict[str, SourceFile], qa: dict[str, SourceFile]) -> None:
    """Raise when any tile has exactly one of its two assets."""
    gaps = sorted(
        f"{tile} (missing {QA_ASSET_KEY if tile in lst else LST_ASSET_KEY})"
        for tile in set(lst) ^ set(qa)
    )
    if gaps:
        msg = (
            f"{len(gaps)} tile(s) carry only one of the two required assets; "
            f"finish or remove them before building: {', '.join(gaps)}"
        )
        raise IncompleteTileError(msg)


def _band_stats(src: rasterio.DatasetReader, index: int) -> BandStats:
    """One band's description and embedded statistics."""
    tags = src.tags(index)
    values = {field: float(tags[tag]) if tag in tags else None for field, tag in _STAT_TAGS.items()}
    return BandStats(description=src.descriptions[index - 1], **values)


def read_header(file: SourceFile) -> CogHeader:
    """Read one COG's footprint, shape, and per-band statistics."""
    with rasterio.Env(**_HEADER_ENV), rasterio.open(file.uri) as src:
        bounds = src.bounds
        return CogHeader(
            file=file,
            bbox=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            width=src.width,
            height=src.height,
            data_type=str(src.dtypes[0]),
            nodata=src.nodata,
            bands=tuple(_band_stats(src, index) for index in src.indexes),
        )


def _place_local(uri: str, dest: Path) -> None:
    """Hard-link the source file into place, copying when that is impossible."""
    dest.unlink(missing_ok=True)
    try:
        dest.hardlink_to(uri)
    except OSError:
        # Different filesystem, or one that has no hard links at all.
        shutil.copy2(uri, dest)


def place_file(file: SourceFile, dest: Path) -> Path:
    """Materialise one source COG beside the item that declares it.

    A destination that already holds the right number of bytes is left alone,
    so re-running a build over an existing catalog is cheap.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == file.size:
        return dest
    if file.uri.startswith("s3://"):
        import boto3  # noqa: PLC0415 - only the s3 path pays for the SDK import

        parsed = urlparse(file.uri)
        boto3.client("s3").download_file(parsed.netloc, parsed.path.lstrip("/"), str(dest))
    else:
        _place_local(file.uri, dest)
    return dest


def scan_source(
    source: str | Path, window: str, tiles: tuple[str, ...] | None = None
) -> list[TilePair]:
    """Discover the complete tiles under ``source`` and read their headers.

    Args:
        source: A local directory or an ``s3://bucket/prefix`` location holding
            the finished COGs, at any depth.
        window: Window label in the filenames, e.g. ``"2021-2025"``.
        tiles: Restrict the result to these tile names. Tiles named here but
            absent from the source are simply not returned.

    Returns:
        One :class:`TilePair` per complete tile, ordered by tile name.

    Raises:
        IncompleteTileError: A tile carries exactly one of its two assets.
    """
    files = list_source(source)
    lst = _index_by_tile(files, LST_ASSET_KEY, window)
    qa = _index_by_tile(files, QA_ASSET_KEY, window)
    _check_complete(lst, qa)
    wanted = sorted(lst if tiles is None else set(lst) & set(tiles))
    return [
        TilePair(tile=tile, lst=read_header(lst[tile]), qa=read_header(qa[tile])) for tile in wanted
    ]
