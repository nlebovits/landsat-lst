"""A real tile's pixels, fetched once and read back locally forever after.

Memory and runtime need geometry, and :mod:`landsat_lst.benchmarks` supplies
that from ``dask.array.random`` without touching the network. Accuracy needs
real pixels. Comparing two offset estimators means running both over the same
scenes, and today that is a cloud round trip per iteration: a STAC query for
2,912 items, then hundreds of gigabytes of coarse reads, for an answer that is
600 floats. The first fetch here is slow. The next hundred cost seconds.

**Read the size arithmetic before choosing a factor.** A five-degree tile at
``destripe_offset_resolution_factor=2`` is a 9,000 squared grid, and 300 scenes
of two ``uint16`` bands over it is 97 GB. That is a storage array, not a laptop
fixture. Factor 8 is 6.1 GB and factor 16 is 1.5 GB.

Coarsening costs accuracy in absolute terms -- offset error grows linearly in
the factor -- but this fixture exists for a *relative* question. #93 asks whether
restructuring the two medians in ``offset_graph`` changes the offsets it
returns. Both estimators read the same fixture, so the comparison is exact at
any factor, and the factor only has to be fine enough that the offsets are not
noise. What a fixture cannot answer is the memory question: below the streaming
regime the stack fits in RAM and dask never streams, which is the behaviour
under test there. Use the synthetic tier for that.

    landsat-lst fixture --tile N40W075 --factor 8
    landsat-lst fixture --list

The store is plain ``.npy`` per band, deliberately. Uncompressed means
:func:`load_fixture` can memory-map it and hand dask a lazy array at production
chunking, so a fixture read builds the same graph a real load does rather than
materializing the stack up front.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

from landsat_lst.config import settings

if TYPE_CHECKING:
    import xarray as xr

    from landsat_lst.models import TileId

log = structlog.get_logger()

#: Bands a fixture stores. The two :func:`~landsat_lst.pipeline.load_scenes`
#: returns, and the only two anything downstream reads.
BANDS = ("lwir11", "qa_pixel")

#: Refuse to fetch more than this without being told twice. Eight gigabytes is
#: a long download and a real dent in a laptop's disk; 97 is a mistake.
DEFAULT_MAX_GB = 8.0

#: Bumped when the on-disk layout changes in a way a reader cannot detect.
FORMAT_VERSION = 1


def fixture_root() -> Path:
    """Where fixtures live. Beside the manifests, not in the package."""
    return Path(settings.manifest_dir).parent / "fixtures"


@dataclass(frozen=True)
class FixtureSpec:
    """Which pixels a fixture holds, and everything that changes them.

    Every field is part of the directory name, so two fixtures that differ in
    any of them cannot collide and a stale one cannot be mistaken for a fresh
    one. This is the same reasoning as :class:`landsat_lst.offsets.OffsetKey`,
    at a coarser grain: a fixture is inputs, not an estimate, so no algorithm
    version applies.
    """

    tile: str
    year: int = 2021
    end_year: int | None = 2025
    max_scenes: int | None = 300
    factor: int = 8

    @property
    def window(self) -> str:
        return f"{self.year}-{self.end_year}" if self.end_year else str(self.year)

    @property
    def name(self) -> str:
        scenes = "all" if self.max_scenes is None else str(self.max_scenes)
        return f"{self.tile}_{self.window}_n{scenes}_f{self.factor}"

    @property
    def path(self) -> Path:
        return fixture_root() / self.name

    def band_path(self, band: str) -> Path:
        return self.path / f"{band}.npy"

    @property
    def meta_path(self) -> Path:
        return self.path / "meta.json"

    @property
    def items_path(self) -> Path:
        return self.path / "items.json"

    def exists(self) -> bool:
        return self.meta_path.exists() and all(self.band_path(b).exists() for b in BANDS)


@dataclass
class FixtureMeta:
    """What a built fixture records about itself."""

    spec: dict
    shape: tuple[int, int, int]
    times: list[str]
    latitude: list[float] = field(default_factory=list)
    longitude: list[float] = field(default_factory=list)
    stac_url: str = ""
    scene_count: int = 0
    bytes_on_disk: int = 0
    format_version: int = FORMAT_VERSION


def grid_shape(spec: FixtureSpec) -> tuple[int, int]:
    """Pixels per side at this factor, from the shared global grid.

    ``round`` rather than ``int``: ``int(5 / (1/3600))`` is 17,999, not 18,000.
    """
    from landsat_lst.tiling import geobox_for_bbox, parse_tile_name  # noqa: PLC0415

    box = geobox_for_bbox(parse_tile_name(spec.tile).bbox, spec.factor)
    return int(box.shape[0]), int(box.shape[1])


def estimate_bytes(spec: FixtureSpec, scenes: int | None = None) -> int:
    """Bytes the stack will occupy on disk, before fetching any of it.

    Arithmetic over the grid, so it is exact for a known scene count and costs
    nothing. Knowing this before the download is the difference between a
    fixture and a surprise.
    """
    height, width = grid_shape(spec)
    n = scenes if scenes is not None else (spec.max_scenes or 0)
    return height * width * n * len(BANDS) * np.dtype("uint16").itemsize


def list_fixtures() -> list[FixtureMeta]:
    """Every built fixture, newest layout first. Skips anything unreadable."""
    root = fixture_root()
    if not root.is_dir():
        return []

    found = []
    for meta_path in sorted(root.glob("*/meta.json")):
        try:
            found.append(FixtureMeta(**json.loads(meta_path.read_text())))
        except (OSError, TypeError, json.JSONDecodeError) as e:
            log.warning("fixture_unreadable", path=str(meta_path), error=str(e))
    return found


def build_fixture(
    spec: FixtureSpec,
    *,
    max_gb: float = DEFAULT_MAX_GB,
    force: bool = False,
    progress=None,
) -> FixtureMeta:
    """Fetch a tile's coarse stack once and write it where a laptop can reread it.

    Args:
        spec: Which pixels to fetch.
        max_gb: Refuse a fetch whose stack would exceed this. Raise it
            deliberately; the default is a laptop's patience, not a limit of
            the format.
        force: Rebuild even when the fixture is already on disk.
        progress: Optional callable taking a status string, for a CLI to render.

    Returns:
        The written :class:`FixtureMeta`.

    Raises:
        ValueError: If the stack would exceed ``max_gb``, or the query returned
            no scenes.
    """
    import dask.array as da  # noqa: PLC0415

    from landsat_lst.models import ProcessingJob  # noqa: PLC0415
    from landsat_lst.pipeline import (  # noqa: PLC0415
        TIME_CHUNK,
        _patch_url_for,
        _sample_scenes,
        load_scenes,
        query_stac,
    )
    from landsat_lst.tiling import parse_tile_name  # noqa: PLC0415

    def _say(msg: str) -> None:
        if progress is not None:
            progress(msg)

    if spec.exists() and not force:
        _say(f"{spec.name} already built; pass --force to refetch")
        return FixtureMeta(**json.loads(spec.meta_path.read_text()))

    planned = estimate_bytes(spec)
    if planned / 1e9 > max_gb:
        height, width = grid_shape(spec)
        msg = (
            f"{spec.name} would write {planned / 1e9:.1f} GB "
            f"({height}x{width} x {spec.max_scenes} scenes x {len(BANDS)} uint16 bands), "
            f"past the {max_gb:.1f} GB ceiling. Raise --factor to coarsen "
            f"(each doubling divides the stack by four), cut --max-scenes, or "
            f"raise --max-gb deliberately."
        )
        raise ValueError(msg)

    tile: TileId = parse_tile_name(spec.tile)
    job = ProcessingJob(
        tile=tile, year=spec.year, end_year=spec.end_year, max_scenes=spec.max_scenes
    )

    _say(f"querying {settings.stac_url}")
    items = query_stac(job)
    if not items:
        msg = f"No scenes for {spec.tile} over {spec.window}"
        raise ValueError(msg)
    if spec.max_scenes is not None:
        items = _sample_scenes(items, spec.max_scenes)
    _say(f"{len(items)} scenes; writing {estimate_bytes(spec, len(items)) / 1e9:.1f} GB")

    data = load_scenes(
        items,
        tile.bbox,
        _patch_url_for(items),
        fail_on_error=False,
        resolution_factor=spec.factor,
    )

    spec.path.mkdir(parents=True, exist_ok=True)
    scenes, height, width = (int(n) for n in data[BANDS[0]].shape)

    # open_memmap writes a real .npy header, so load_fixture can mmap it back
    # without holding the stack in RAM at either end. da.store streams block by
    # block into it, which is the only way a stack larger than memory lands.
    sources, targets = [], []
    for band in BANDS:
        array = data[band].data
        target = np.lib.format.open_memmap(
            spec.band_path(band), mode="w+", dtype=array.dtype, shape=array.shape
        )
        sources.append(array)
        targets.append(target)

    _say("fetching (this is the slow one; every later read is local)")
    da.store(sources, targets)
    for target in targets:
        target.flush()

    # Item ids rather than full items: the fixture is the pixels, and the ids
    # are what make a later run reproducible or a discrepancy attributable.
    spec.items_path.write_text(
        json.dumps([{"id": item.id, "datetime": str(item.datetime)} for item in items], indent=2)
    )

    meta = FixtureMeta(
        spec={
            "tile": spec.tile,
            "year": spec.year,
            "end_year": spec.end_year,
            "max_scenes": spec.max_scenes,
            "factor": spec.factor,
        },
        shape=(scenes, height, width),
        times=[str(t) for t in np.asarray(data["time"].values)],
        latitude=[float(data["latitude"].values[0]), float(data["latitude"].values[-1])],
        longitude=[float(data["longitude"].values[0]), float(data["longitude"].values[-1])],
        stac_url=settings.stac_url,
        scene_count=len(items),
        bytes_on_disk=sum(spec.band_path(b).stat().st_size for b in BANDS),
    )
    spec.meta_path.write_text(json.dumps(meta.__dict__, indent=2))

    log.info(
        "fixture_built",
        name=spec.name,
        scenes=meta.scene_count,
        gb=round(meta.bytes_on_disk / 1e9, 2),
        chunk=TIME_CHUNK,
    )
    _say(f"wrote {spec.path} ({meta.bytes_on_disk / 1e9:.1f} GB)")
    return meta


def load_fixture(spec: FixtureSpec) -> xr.Dataset:
    """Read a fixture back as the Dataset ``load_scenes`` would have returned.

    Memory-mapped and wrapped in dask at production chunking, so the graph built
    on top of it is the graph a real tile builds. Materializing the stack here
    instead would make every downstream measurement describe a pipeline that
    does not stream.

    Raises:
        FileNotFoundError: If the fixture has not been built.
    """
    import dask.array as da  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    from landsat_lst.pipeline import TIME_CHUNK  # noqa: PLC0415

    if not spec.exists():
        msg = f"No fixture at {spec.path}. Build it with: landsat-lst fixture --tile {spec.tile}"
        raise FileNotFoundError(msg)

    meta = FixtureMeta(**json.loads(spec.meta_path.read_text()))
    csize = settings.load_chunk_size
    scenes, height, width = meta.shape

    arrays = {}
    for band in BANDS:
        mapped = np.load(spec.band_path(band), mmap_mode="r")
        arrays[band] = (
            ("time", "latitude", "longitude"),
            da.from_array(
                mapped, chunks=(min(TIME_CHUNK, scenes), min(csize, height), min(csize, width))
            ),
        )

    return xr.Dataset(
        arrays,
        coords={
            "time": pd.to_datetime(meta.times),
            # Endpoints rather than the full vectors: the grid is regular, and
            # storing 9,000 floats per axis in JSON buys nothing.
            "latitude": np.linspace(meta.latitude[0], meta.latitude[1], height),
            "longitude": np.linspace(meta.longitude[0], meta.longitude[1], width),
        },
    )
