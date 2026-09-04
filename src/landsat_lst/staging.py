"""Stage phase A's coarse observations so phase B does not read the sources twice.

The offset estimator makes two passes over the same coarse stack. Phase A
shards over space and reduces every scene into a monthly climatology. Phase B
shards over scene and takes one spatial median per scene. The axes are
orthogonal, so the passes cannot share a traversal, and the second pass has
always gone back to the Landsat sources for bytes the first pass already
decoded and threw away.

At production geometry each pass decodes 949.3 GB, and the retained anchor run
``shard-S30W065-2021-2025-20260823T102135Z`` measured phase B at 488 to 687 s
per shard against 2.6 s of compute. Phase B is read time.

This module carries those observations forward instead. Phase A writes what it
reads as ``uint16`` DN, and phase B assembles its scenes from that stage. Two
properties make it exact rather than approximate:

- The DN carries the observation losslessly. ``qa.celsius_stack`` reconstructs
  the float32 Celsius array the estimator has always read, bit for bit, so no
  estimator, no threshold, and no output changes. The staged form is half the
  bytes because the source pass carries two ``uint16`` bands and the stage
  carries one.
- Only blocks phase A actually reads are staged. A block holding no land is
  never read, never staged, and reconstructed as all-NaN, which is precisely
  what the land-masked source yields there. A land-free block contributes
  nothing to a spatial median and exactly zero to ``n_valid``, so phase B's
  arithmetic is unchanged. A block the plan says holds land but whose object is
  missing is an error, never a silent gap: reading a missing object as
  no-observation would quietly thin ``n_valid``.

The stage is scratch. It is keyed by the tile's :class:`~landsat_lst.offsets.OffsetKey`
digest and algorithm version, so a stage written for a different scene set,
offset factor, clamp, or estimator version lives under a different prefix and
can never be read back into this one. The driver sweeps it once the merged
offset record exists, because that record is the durable output and a leftover
object under the run prefix is an object a later listing reads as finished work.

See issue #125 and ADR-020.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog
import xarray as xr

from landsat_lst.qa import celsius_stack

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from landsat_lst.offsets import OffsetKey
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: Bumped when the stored bytes stop meaning what an older reader thinks. It
#: sits in the prefix beside the estimator's own version, so a format change
#: cannot be mistaken for a scene-set change, or read by the wrong reader.
STAGE_FORMAT_VERSION = 1

#: Ceiling on the DN buffer one phase-A read holds, in bytes. Phase A already
#: holds a float32 block for the whole window; this rides on top of it, so it
#: is sized to disappear against that rather than to be fast. 256 MB against a
#: 12.3 GB block is 2%.
_READ_BUFFER_BYTES = 256 * 1024 * 1024

Span = tuple[int, int, int, int]


@dataclass(frozen=True)
class StageKey:
    """Where one tile-window's stage lives, and what it is allowed to answer for.

    Built from the :class:`~landsat_lst.offsets.OffsetKey` rather than from
    loose parts, so the stage inherits every term that decides an offset: the
    scene ids, the offset factor, the clamp bounds, and
    ``offsets.ALGORITHM_VERSION``. A stale stage is therefore unreachable
    rather than merely unlikely -- it is at a prefix nobody looks under.
    """

    root: str
    algorithm_version: int
    digest: str
    fmt: int = STAGE_FORMAT_VERSION

    @classmethod
    def from_offset_key(cls, root: str, key: OffsetKey) -> StageKey:
        return cls(root=root, algorithm_version=key.algorithm_version, digest=key.digest)

    @property
    def prefix(self) -> str:
        """The listing prefix, and the unit the driver sweeps."""
        return f"{self.root}/stage/f{self.fmt}-v{self.algorithm_version}-{self.digest}/"

    def object_key(self, block: int, batch: int) -> str:
        """One staged object: block ``block``, scene batch ``batch``.

        A pure function of the two indexes, like every other shard artifact, so
        a writer and a reader in different processes agree without negotiating.
        """
        return f"{self.prefix}b{block:04d}.s{batch:05d}.npy"


class CoarseStage:
    """Read and write staged coarse observations for one tile-window.

    Objects are raw uncompressed ``.npy``, for the reason
    ``shard_tasks._assemble_ref`` stores climatology blocks that way: the
    reader wants the bytes straight into a slice of an array it already
    allocated, and a compressor would buy storage this thing holds for minutes
    at the cost of CPU on the critical path.
    """

    def __init__(self, storage: StorageBackend, key: StageKey) -> None:
        self.storage = storage
        self.key = key

    def write(self, block: int, batch: int, values: np.ndarray) -> str:
        """Publish one ``(scenes, y, x)`` ``uint16`` chunk. Idempotent by key."""
        if values.dtype != np.uint16:
            msg = f"the stage carries uint16 DN, not {values.dtype}"
            raise TypeError(msg)
        key = self.key.object_key(block, batch)
        scratch = Path(tempfile.mkdtemp(prefix="lst_stage_w_"))
        try:
            local = scratch / "chunk.npy"
            np.save(local, values)
            self.storage.upload(local, key)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        return key

    def read(self, block: int, batch: int) -> np.ndarray | None:
        """One staged chunk, or ``None`` when the object is not there."""
        key = self.key.object_key(block, batch)
        scratch = Path(tempfile.mkdtemp(prefix="lst_stage_r_"))
        try:
            local = scratch / "chunk.npy"
            if not self.storage.download(key, local):
                return None
            return np.load(local)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def cleanup(self) -> int:
        """Delete the whole stage. Returns how many objects went."""
        removed = self.storage.delete_prefix(self.key.prefix)
        log.info("coarse_stage_cleaned", prefix=self.key.prefix, deleted=removed)
        return removed


def stage_batches(lst: xr.DataArray, time_dim: str = "time") -> list[tuple[int, int]]:
    """The scene spans one staged object covers, aligned to the source's chunks.

    Alignment is not tuning. A batch cut anywhere but a source chunk edge makes
    its boundary chunk materialize twice, which is the same rule
    ``normalization._scene_batches`` follows for the estimator's own reads. It
    also guarantees that a phase-B group, which is a union of whole source
    chunks, is a union of whole staged objects.
    """
    sizes: Sequence[int] | None = None
    if lst.chunks is not None:
        sizes = lst.chunks[lst.dims.index(time_dim)]
    if not sizes:
        sizes = (int(lst.sizes[time_dim]),)
    spans, start = [], 0
    for size in sizes:
        spans.append((start, start + int(size)))
        start += int(size)
    return spans


def _as_celsius(dn: np.ndarray, dims: tuple[str, ...]) -> np.ndarray:
    """The estimator's float32 input for a staged DN chunk.

    Routed through :func:`landsat_lst.qa.celsius_stack` rather than reimplemented
    so there is one definition of the reconstruction, and the test that pins it
    bit-identical pins this too.
    """
    return np.asarray(celsius_stack(xr.DataArray(dn, dims=dims)).values, dtype=np.float32)


def staging_block_reader(
    dn: xr.DataArray,
    stage: CoarseStage,
    *,
    block_index: Callable[[Span], int],
    batches: list[tuple[int, int]],
    read_values: Callable[[xr.DataArray, np.dtype], np.ndarray],
    dims: tuple[str, ...] = ("time", "latitude", "longitude"),
) -> Callable[[Span], np.ndarray]:
    """Phase A's reader: decode the sources once, stage the DN, return Celsius.

    The staging rides inside the read the pass was going to do anyway. Scenes
    are decoded in groups of whole staged batches, bounded by
    :data:`_READ_BUFFER_BYTES`, and each batch is published from that buffer
    before the float32 block absorbs it. Peak memory is therefore the float32
    block phase A already held, plus one bounded DN buffer.
    """
    y_dim, x_dim = dims[1], dims[2]
    n_scenes = int(dn.sizes[dims[0]])
    f32 = np.dtype(np.float32)

    def read(span: Span) -> np.ndarray:
        y0, y1, x0, x1 = span
        block = block_index(span)
        out = np.empty((n_scenes, y1 - y0, x1 - x0), dtype=f32)
        per_scene = (y1 - y0) * (x1 - x0) * 2
        group = max(1, _READ_BUFFER_BYTES // max(per_scene, 1))
        pending: list[tuple[int, int, int]] = []  # (batch index, start, stop)
        for bi, (start, stop) in enumerate(batches):
            pending.append((bi, start, stop))
            spans_scenes = pending[-1][2] - pending[0][1]
            last = bi == len(batches) - 1
            if spans_scenes < group and not last:
                continue
            lo, hi = pending[0][1], pending[-1][2]
            chunk = read_values(
                dn.isel({dims[0]: slice(lo, hi), y_dim: slice(y0, y1), x_dim: slice(x0, x1)}),
                np.dtype(np.uint16),
            )
            for sub_bi, sub_lo, sub_hi in pending:
                stage.write(block, sub_bi, chunk[sub_lo - lo : sub_hi - lo])
            out[lo:hi] = _as_celsius(chunk, dims)
            del chunk
            pending = []
        return out

    return read


def staged_batch_reader(
    stage: CoarseStage,
    *,
    blocks: list[Span],
    block_has_land: list[bool],
    batches: list[tuple[int, int]],
    shape: tuple[int, int],
    dims: tuple[str, ...] = ("time", "latitude", "longitude"),
) -> Callable[[tuple[int, int]], np.ndarray]:
    """Phase B's reader: rebuild a scene batch from the stage, not the sources.

    The result is the array ``normalization._read_values`` would have produced
    from the land-masked coarse stack. Pixels outside a staged block are NaN,
    which is what the land mask already made them, so the spatial median sees
    the same samples and ``n_valid`` counts the same pixels.

    Raises:
        FileNotFoundError: If a block the plan marks as holding land has no
            staged object. Treating that as no-observation would thin
            ``n_valid`` without saying so.
    """
    height, width = shape

    def read(group: tuple[int, int]) -> np.ndarray:
        start, stop = group
        out = np.full((stop - start, height, width), np.nan, dtype=np.float32)
        wanted = [(bi, lo, hi) for bi, (lo, hi) in enumerate(batches) if lo >= start and hi <= stop]
        for block, (y0, y1, x0, x1) in enumerate(blocks):
            if not block_has_land[block]:
                continue
            for bi, lo, hi in wanted:
                chunk = stage.read(block, bi)
                if chunk is None:
                    msg = (
                        f"staged block {block} batch {bi} is missing at "
                        f"{stage.key.object_key(block, bi)}; phase A published a "
                        "climatology block for it, so the stage should hold it"
                    )
                    raise FileNotFoundError(msg)
                out[lo - start : hi - start, y0:y1, x0:x1] = _as_celsius(chunk, dims)
                del chunk
        return out

    return read
