"""Persisted per-scene de-striping offsets, keyed on everything that changes them.

Estimating the offsets is the expensive half of a tile. On the 300-scene
``N40W075`` sample of run ``2021-2025-sample300-20260813T123249Z`` it ran for
26.9 minutes and produced roughly 600 float64 values -- a few kilobytes bought
with 598,604 dask tasks. Every experiment downstream of that pass recomputes it
from scratch, including the experiments where the offsets provably cannot
change: sweeping ``destripe_max_offset_c``, moving either sparse floor, touching
the land mask, the encoding, or the COG writer, and every rerun after a crash,
a spot preemption, or a timeout.

This module makes that pass a cache lookup. See issue #77 item 2.

**The cache is keyed, not versioned.** A stale result never satisfies a fresh
request, because every input that moves the offsets is in the key:

- the scene ids, hashed. ``--max-scenes`` changes which scenes are pooled, and
  each offset is measured against a monthly climatology built from the whole
  set, so a different scene list is a different answer for every scene in it.
- ``destripe_offset_resolution_factor``, which decides the grid the median rests
  on.
- the physical-plausibility clamp, which decides what reaches that median.
- :data:`ALGORITHM_VERSION`, for the code changes a hash cannot see.

**A cache miss must never fail a tile.** Every read and write here is
best-effort, the same rule the heartbeat follows: a failure is logged and
swallowed, and the tile recomputes. Losing the cache costs 27 minutes. Failing
the tile costs the whole run.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import xarray as xr

from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Iterable

    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()

#: Bump when a code change moves the offsets a hash cannot see: the reduction in
#: :func:`landsat_lst.normalization.offset_graph`, the QA bits
#: :func:`landsat_lst.qa.create_qa_mask` masks, or the DN-to-Celsius conversion.
#: The scene ids, the resolution factor, and the plausibility clamp are hashed
#: into the key and need no bump. See ADR-007.
#:
#: "Moves the offsets" is the operative clause. The 2026-08-15 month-loop
#: reformulation of ``offset_graph`` rebuilt the dask graph and left the
#: values bit-identical -- pinned in the unit tests against the preserved
#: groupby formulation and measured at max |delta| = 0.0 over 300 real
#: scenes -- so it takes no bump: cached estimates stay valid answers.
ALGORITHM_VERSION = 1

#: Hex characters kept from the input digest. 16 is 64 bits: collision odds stay
#: negligible across every tile-window this project will ever build, and the key
#: stays readable in a bucket listing.
_DIGEST_CHARS = 16


def _finite_or_none(values: Iterable[float]) -> list[float | None]:
    """NaN as ``null``, so the record is JSON a strict parser accepts.

    ``json.dumps`` emits a bare ``NaN`` token by default, which round-trips
    through Python and fails everywhere else. A rejected scene's offset is
    genuinely absent, and ``null`` is how JSON says so.
    """
    return [None if not math.isfinite(v) else float(v) for v in values]


def _times_iso(time_coord: xr.DataArray) -> list[str]:
    """The time coordinate as ISO strings, for an exact equality check on read."""
    return [np.datetime_as_string(t, unit="s") for t in np.asarray(time_coord.values)]


@dataclass(frozen=True)
class OffsetKey:
    """Everything that decides the offsets, in the object key that holds them.

    Built by :meth:`build` rather than by hand, so a caller cannot forget a term
    and key a result on less than produced it.
    """

    tile: str
    window: str
    factor: int
    algorithm_version: int
    digest: str

    @classmethod
    def build(
        cls,
        *,
        tile: str,
        window: str,
        factor: int,
        scene_ids: Iterable[str],
        algorithm_version: int = ALGORITHM_VERSION,
    ) -> OffsetKey:
        """Hash the scene set and the settings that shape the estimate.

        Scene ids are sorted before hashing: STAC returns them in whatever order
        the search paged them, and two identical scene sets must key alike.

        The clamp bounds are in the digest because
        :func:`landsat_lst.qa.convert_to_celsius` applies them before the median
        ever sees a pixel, so moving either one moves the offsets.
        """
        material = "\n".join(
            [
                f"tile={tile}",
                f"window={window}",
                f"factor={factor}",
                f"algorithm_version={algorithm_version}",
                f"lst_valid_min={settings.lst_valid_min}",
                f"lst_valid_max={settings.lst_valid_max}",
                *sorted(scene_ids),
            ]
        )
        digest = hashlib.sha256(material.encode()).hexdigest()[:_DIGEST_CHARS]
        return cls(
            tile=tile,
            window=window,
            factor=factor,
            algorithm_version=algorithm_version,
            digest=digest,
        )

    @property
    def storage_key(self) -> str:
        """Backend-relative key, as laid out in :mod:`landsat_lst.storage`."""
        from landsat_lst.storage import offset_cache_key  # noqa: PLC0415

        return offset_cache_key(
            tile=self.tile,
            window=self.window,
            factor=self.factor,
            algorithm_version=self.algorithm_version,
            digest=self.digest,
        )


class OffsetCache:
    """Read and write one tile-window's offsets, never raising either way.

    Constructed with the key already resolved, so the caller has committed to a
    scene set before anything is looked up. :meth:`read` returns ``None`` on a
    miss, on a malformed record, and on a record whose time coordinate does not
    match the stack in hand.

    The two halves switch independently, because the two reasons to distrust a
    cache are different. ``read=False`` with ``write=True`` is ``--force``: the
    inputs are unchanged but you want the estimate rebuilt, and the fresh answer
    should replace the old one. ``enabled=False`` is ``--no-offset-cache``:
    neither half runs, and the record on disk is left exactly as it was.
    """

    def __init__(
        self,
        *,
        storage: StorageBackend,
        key: OffsetKey,
        enabled: bool = True,
        read: bool = True,
        write: bool = True,
    ) -> None:
        self.storage = storage
        self.key = key
        self.enabled = enabled
        self.read_enabled = read
        self.write_enabled = write
        #: Whether the last :meth:`read` was served from cache. ``None`` until
        #: one has been attempted. Recorded rather than inferred from how long
        #: the pass took: a caller reporting "cached" should be reading a fact,
        #: not timing one.
        self.last_read_hit: bool | None = None

    def read(self, time_coord: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray] | None:
        """The cached ``(offset, n_valid)`` for this stack, or ``None``.

        The stored time coordinate is compared against ``time_coord`` before the
        arrays are handed back. The digest should already have guaranteed the
        match, so a mismatch here means the key is under-specified rather than
        that the cache is cold, and it is logged as the defect it is.
        """
        if not (self.enabled and self.read_enabled):
            self.last_read_hit = False
            return None

        try:
            raw = self.storage.read_text(self.key.storage_key)
        except Exception as e:
            log.warning("offset_cache_read_failed", key=self.key.storage_key, error=str(e))
            self.last_read_hit = False
            return None

        if raw is None:
            log.info("offset_cache_miss", key=self.key.storage_key)
            self.last_read_hit = False
            return None

        try:
            record = json.loads(raw)
            stored_times = list(record["times"])
            offset_values = [np.nan if v is None else float(v) for v in record["offset"]]
            n_valid_values = [int(v) for v in record["n_valid"]]
        except (ValueError, KeyError, TypeError) as e:
            log.warning("offset_cache_malformed", key=self.key.storage_key, error=str(e))
            self.last_read_hit = False
            return None

        if stored_times != _times_iso(time_coord):
            log.warning(
                "offset_cache_time_mismatch",
                key=self.key.storage_key,
                stored=len(stored_times),
                live=int(time_coord.size),
                note="key is under-specified; recomputing",
            )
            self.last_read_hit = False
            return None

        log.info(
            "offset_cache_hit",
            key=self.key.storage_key,
            scenes=len(stored_times),
            saved_s=record.get("duration_s"),
        )
        self.last_read_hit = True
        coords = {"time": time_coord}
        return (
            xr.DataArray(np.array(offset_values, dtype="float64"), dims=["time"], coords=coords),
            xr.DataArray(np.array(n_valid_values, dtype="int64"), dims=["time"], coords=coords),
        )

    def write(
        self,
        offset: xr.DataArray,
        n_valid: xr.DataArray,
        *,
        duration_s: float | None = None,
    ) -> None:
        """Persist the arrays this run computed. Never raises."""
        if not (self.enabled and self.write_enabled):
            return

        payload: dict[str, Any] = {
            "tile": self.key.tile,
            "window": self.key.window,
            "offset_resolution_factor": self.key.factor,
            "algorithm_version": self.key.algorithm_version,
            "digest": self.key.digest,
            "scenes": int(offset.sizes.get("time", 0)),
            "times": _times_iso(offset.time),
            "offset": _finite_or_none(np.asarray(offset.values, dtype="float64")),
            "n_valid": [int(v) for v in np.asarray(n_valid.values)],
            "duration_s": None if duration_s is None else round(duration_s, 1),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            self.storage.write_text(self.key.storage_key, json.dumps(payload, indent=2))
            log.info(
                "offset_cache_written",
                key=self.key.storage_key,
                scenes=payload["scenes"],
                duration_s=payload["duration_s"],
            )
        except Exception as e:
            log.warning("offset_cache_write_failed", key=self.key.storage_key, error=str(e))


def cache_for_items(
    *,
    tile: str,
    window: str,
    items: Iterable[Any],
    factor: int,
    storage: StorageBackend | None = None,
    enabled: bool = True,
    read: bool = True,
) -> OffsetCache:
    """Build the cache for a resolved STAC item list.

    Args:
        tile: Tile name (``"N40W075"``).
        window: Window label, ``ProcessingJob.window_label``. A sampled window
            carries its own ``-sampleN`` token, so a sample cannot be served the
            full window's offsets even before the scene hash disagrees.
        items: The STAC items the tile will load. Only their ``id`` is read.
        factor: ``settings.destripe_offset_resolution_factor`` for this run.
        storage: Backend the record lives in. Defaults to the configured one.
        enabled: ``False`` disables both halves, for ``--no-offset-cache``.
        read: ``False`` skips the lookup but still writes, for ``--force``.

    Returns:
        A cache ready to :meth:`~OffsetCache.read`.
    """
    from landsat_lst.storage import get_storage  # noqa: PLC0415

    key = OffsetKey.build(
        tile=tile,
        window=window,
        factor=factor,
        scene_ids=[str(item.id) for item in items],
    )
    return OffsetCache(storage=storage or get_storage(), key=key, enabled=enabled, read=read)
