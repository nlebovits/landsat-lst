"""A tile plan small enough to hold in a test, and a fleet that never bills.

The shard tests need two things that are awkward to build inline: a
:class:`~landsat_lst.shards.TilePlan` whose geometry is a few pixels rather than
18,000, and something that plays the part of Coiled Batch by writing the
artifacts a stage's shards would have written. Both live here so the driver
tests and the task tests agree about what a shard's output looks like -- two
fixtures disagreeing about that would let a driver test pass against keys no
task writes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from landsat_lst import shards
from landsat_lst.config import settings
from landsat_lst.offsets import _times_iso
from landsat_lst.shard_driver import _expected_keys
from landsat_lst.storage import PRODUCTS

TILE = "N40W075"
WINDOW = "2021-2025"
RUN_ID = "shard-test"
SCENES = 4

#: 1,024 rows is 512 x 2, so two row bands each start on a COG block row -- the
#: alignment ``band_edges`` guarantees on the production grid.
NATIVE = (1024, 1024)
COARSE = (8, 8)
BLOCK = 4


def scene_times(n: int = SCENES) -> list[str]:
    """A frozen time axis, spelled exactly as the offset records spell it."""
    times = pd.date_range("2021-07-04", periods=n, freq="61D").values
    coord = SimpleNamespace(values=times)
    # ``np.datetime_as_string`` yields ``np.str_``; the plan's own round trip
    # through JSON does this cast, so the fixture does it up front.
    return [str(stamp) for stamp in _times_iso(coord)]  # type: ignore[arg-type]


def make_plan(*, ref_shards: int = 2, scene_shards: int = 2, band_shards: int = 2):
    """A four-scene, four-block, two-band plan.

    One block is marked land-free on purpose: the phase-A stage publishes a
    zero-byte marker for it rather than a plane of NaN, and both the shard and
    the driver have to agree on which key that is.
    """
    blocks = shards.block_spans(COARSE, BLOCK)
    return shards.TilePlan(
        tile=TILE,
        window=WINDOW,
        scene_ids=[f"scene-{i}" for i in range(SCENES)],
        scene_times=scene_times(),
        offset_factor=settings.destripe_offset_resolution_factor,
        coarse_shape=COARSE,
        native_shape=NATIVE,
        block_edge=BLOCK,
        blocks=blocks,
        block_has_land=[True, True, True, False],
        scene_batches=[(0, 2), (2, 4)],
        bands=shards.band_edges(NATIVE[0], band_shards, settings.cog_blocksize),
        ref_shards=ref_shards,
        scene_shards=scene_shards,
        band_shards=band_shards,
    )


def make_items(plan) -> list[dict]:
    """Serialized STAC items, as ``pipeline.items_to_dicts`` would leave them."""
    import pystac

    out = []
    for scene_id, stamp in zip(plan.scene_ids, plan.scene_times, strict=True):
        item = pystac.Item(
            id=scene_id,
            geometry={
                "type": "Polygon",
                "coordinates": [[[-75, 35], [-70, 35], [-70, 40], [-75, 40], [-75, 35]]],
            },
            bbox=[-75.0, 35.0, -70.0, 40.0],
            datetime=pd.Timestamp(stamp).to_pydatetime(),
            properties={},
        )
        out.append(item.to_dict())
    return out


def publish_plan(storage, plan, *, run_id: str = RUN_ID) -> str:
    """Write the plan and its items, as the resolve stage would.

    ``apply_shard_settings`` first, exactly as the real planner does: the
    digest covers ``load_chunk_size``, and a plan stamped under the ambient
    value would be refused by every shard that pins the sharded one.
    """
    from landsat_lst.shard_tasks import apply_shard_settings

    apply_shard_settings()
    root = shards.shard_root(run_id, plan.tile)
    storage.write_text(shards.items_key(root), json.dumps(make_items(plan)))
    storage.write_text(shards.plan_key(root), json.dumps(plan.to_dict(), indent=2))
    return root


def write_offset_cache(storage, plan, *, offset: float = 0.5, n_valid: int = 1000):
    """Seed the merged offsets a composite shard reads back.

    Written through :class:`~landsat_lst.offsets.OffsetCache` at the canonical
    key rather than into a shard-shaped location, because that is the seam: the
    merge writes the ordinary ADR-012 record and every band reads it.
    """
    import xarray as xr

    from landsat_lst.offsets import OffsetCache
    from landsat_lst.shard_tasks import _offset_key, _time_coord

    coord = _time_coord(plan)
    cache = OffsetCache(storage=storage, key=_offset_key(plan))
    cache.write(
        xr.DataArray(
            np.full(coord.size, offset, dtype=np.float32), dims=["time"], coords={"time": coord}
        ),
        xr.DataArray(
            np.full(coord.size, n_valid, dtype="int64"), dims=["time"], coords={"time": coord}
        ),
    )
    return cache


class FakeFleet:
    """Stands in for ``submit_shard_stage``: writes the artifacts, bills nothing.

    Records every call so a test can assert on the stage order and on *which
    indexes* a resubmission carried -- the point of the barrier is that it
    resends only what is missing, and a fleet that quietly resent everything
    would still make the tile finish.
    """

    def __init__(
        self,
        storage,
        plan,
        *,
        run_id: str = RUN_ID,
        never: set | None = None,
        heal: bool = False,
    ) -> None:
        self.storage = storage
        self.plan = plan
        self.run_id = run_id
        self.root = shards.shard_root(run_id, plan.tile)
        #: ``(stage, index)`` pairs this fleet refuses to complete, standing in
        #: for a shard that keeps dying.
        self.never = set(never or ())
        #: With ``heal``, each of those fails once and succeeds on the next
        #: submission -- a preempted VM rather than a broken one.
        self.heal = heal
        self.calls: list[tuple[str, list[int]]] = []

    def __call__(self, *, stage, run_id, tile, indexes, job=None):
        self.calls.append((stage, list(indexes)))
        for index in indexes:
            if (stage, index) in self.never:
                if self.heal:
                    self.never.discard((stage, index))
                continue
            self._write(stage, index)
        return SimpleNamespace(cluster_id=len(self.calls), job_id=1)

    @property
    def stages(self) -> list[str]:
        return [stage for stage, _ in self.calls]

    def _write(self, stage: str, index: int) -> None:
        if stage == "resolve":
            publish_plan(self.storage, self.plan, run_id=self.run_id)
            return
        if stage == "export":
            for product in PRODUCTS:
                self.storage.write_text(
                    self.storage.cog_key(self.plan.window, self.plan.tile, product), "tif"
                )
            return
        if stage == "offsets":
            self._write_partial(index)
            return
        for key in _expected_keys(self.plan, stage, self.root)[index]:
            self.storage.write_text(key, "")

    def _write_partial(self, index: int) -> None:
        """A real partial, so the driver's in-process merge has something to merge."""
        from landsat_lst.shard_tasks import offsets_group

        group = offsets_group(self.plan, index)
        start, stop = group[0][0], group[-1][1]
        payload = {
            "times": self.plan.scene_times[start:stop],
            "offset": [0.25] * (stop - start),
            "n_valid": [1000] * (stop - start),
        }
        self.storage.write_text(
            shards.scene_partial_key(self.root, start, stop), json.dumps(payload)
        )
