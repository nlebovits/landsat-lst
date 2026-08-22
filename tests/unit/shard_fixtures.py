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
import time
from types import SimpleNamespace
from typing import ClassVar

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


def scene_time_values(n: int = SCENES) -> np.ndarray:
    """A frozen time axis with real sub-second components.

    Landsat solar-day stamps carry them, and every stamp here round-trips
    through JSON before a coordinate join reads it back. Whole seconds made the
    serializer's truncation invisible to every test in this repo while the
    composite failed on every shard of S30W065.
    """
    base = pd.date_range("2021-07-04T13:45:12", periods=n, freq="61D")
    return (base + pd.to_timedelta(482_915 + 137 * np.arange(n), unit="us")).values


def scene_times(n: int = SCENES) -> list[str]:
    """That axis, spelled exactly as the offset records spell it."""
    coord = SimpleNamespace(values=scene_time_values(n))
    return _times_iso(coord)  # type: ignore[arg-type]


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


def publish_legacy_plan(storage, plan, *, run_id: str = RUN_ID, items=None) -> str:
    """A plan as the pre-2026-08-22 planner wrote it: stamps at second precision.

    The items keep their full precision, because they always did -- the
    truncation happened in ``_times_iso`` on the way into the plan, not in the
    catalog. That asymmetry is the whole of the bug and the whole of the
    recovery, so a fixture that truncated both would prove nothing.
    """
    from landsat_lst.shard_tasks import apply_shard_settings

    apply_shard_settings()
    root = shards.shard_root(run_id, plan.tile)
    payload = plan.to_dict()
    payload["scene_times"] = [stamp.split(".")[0] for stamp in plan.scene_times]
    storage.write_text(shards.items_key(root), json.dumps(items or make_items(plan)))
    storage.write_text(shards.plan_key(root), json.dumps(payload, indent=2))
    return root


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
        claims_export: bool = True,
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
        #: Every cluster name this fleet was asked to start, in order. A round
        #: that reused a previous round's name is what Coiled refuses outright.
        self.names: list[str] = []
        #: Scene partials present at each call, to pin the overlap's ordering.
        self.partials_at_call: list[int] = []
        #: Whether a composite worker claims the export once every band exists,
        #: as :func:`landsat_lst.shard_tasks.claim_export` does. ``False`` plays
        #: the VM that was preempted between claiming and running.
        self.claims_export = claims_export

    def __call__(self, *, stage, run_id, tile, indexes, job=None, units=None, submission_round=1):
        from landsat_lst.batch import stage_cluster_name

        self.calls.append((stage, list(indexes)))
        self.names.append(stage_cluster_name(run_id, tile, stage, submission_round))
        # How much of phase B existed when this fleet was asked for. The overlap
        # is only worth anything if the composite starts *before* the offsets
        # stage finishes, and a fleet that started after would still produce a
        # correct tile -- so the ordering has to be recorded, not inferred.
        self.partials_at_call.append(len(self.storage.list_prefix(f"{self.root}/offsets/scene/")))
        for index in indexes:
            if (stage, index) in self.never:
                if self.heal:
                    self.never.discard((stage, index))
                continue
            self._write(stage, index)
        return SimpleNamespace(
            cluster_id=len(self.calls),
            job_id=1,
            name=self.names[-1],
            submission_round=submission_round,
        )

    @property
    def stages(self) -> list[str]:
        return [stage for stage, _ in self.calls]

    def _write(self, stage: str, index: int) -> None:
        if stage == "resolve":
            publish_plan(self.storage, self.plan, run_id=self.run_id)
            return
        if stage == "export":
            self._write_cogs()
            return
        if stage == "offsets":
            # Fused: shard 0 resolves before anyone reduces anything, every
            # shard publishes its blocks, and then its scene partial.
            publish_plan(self.storage, self.plan, run_id=self.run_id)
            if index < self.plan.ref_shards:
                for key in _expected_keys(self.plan, "climatology", self.root)[index]:
                    self.storage.write_text(key, "")
            if index < self.plan.scene_shards:
                self._write_partial(index)
            return

        for key in _expected_keys(self.plan, stage, self.root)[index]:
            self.storage.write_text(key, "")

        if stage == "composite" and self.claims_export:
            self._maybe_claim_export()

    def _write_cogs(self) -> None:
        for product in PRODUCTS:
            self.storage.write_text(
                self.storage.cog_key(self.plan.window, self.plan.tile, product), "tif"
            )

    def _maybe_claim_export(self) -> None:
        """What ``claim_export`` does: the last band written runs the export."""
        present = set(self.storage.list_prefix(f"{self.root}/composite/"))
        wanted = {
            key
            for index in range(len(self.plan.bands))
            for key in _expected_keys(self.plan, "composite", self.root)[index]
        }
        if wanted - present:
            return
        claim = shards.export_claim_key(self.root)
        if self.storage.read_text(claim) is not None:
            return
        self.storage.write_text(claim, json.dumps({"tile": self.plan.tile}))
        self._write_cogs()

    def all_indexes(self, stage: str) -> list[int]:
        """Every shard index this stage has, from the plan."""
        counts = {
            "resolve": 1,
            "climatology": self.plan.ref_shards,
            "offsets": self.plan.scene_shards,
            "composite": len(self.plan.bands),
            "export": 1,
        }
        return list(range(counts[stage]))

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


def record_in_flight(
    storage, plan, stage, indexes, *, round_no: int = 1, age_s: float = 0.0, run_id: str = RUN_ID
) -> str:
    """Publish a submission record, as a driver does just before it submits.

    ``age_s`` backdates it: a record older than ``shard_barrier_timeout_s``
    describes a cluster whose barrier has expired, and a fresh one describes
    shards that may still be booting -- the state the artifacts alone cannot
    distinguish from "nobody has started this".
    """
    root = shards.shard_root(run_id, plan.tile)
    storage.write_text(
        shards.stage_submission_key(root, stage, round_no),
        json.dumps(
            {
                "run_id": run_id,
                "tile": plan.tile,
                "stage": stage,
                "round": round_no,
                "indexes": list(indexes),
                "cluster_name": f"lst-fake-{stage}-r{round_no}",
                "cluster_id": 1,
                "submitted_at": time.time() - age_s,
            }
        ),
    )
    return root


class LandsOnPoll:
    """A backend whose Nth listing is when somebody else's shards finish.

    The in-flight case cannot be tested by waiting it out: a driver that adopts
    a fresh submission watches until that round's deadline, and a deadline
    short enough for a test would not be fresh. So the artifacts arrive *during*
    the watch, which is exactly what an adopting driver is waiting for -- and
    what the driver in the observed failure never got to see, because it
    submitted instead and Coiled refused the duplicate cluster name.
    """

    #: Where each stage's artifacts are listed from. Only listings of *this*
    #: prefix are counted, so the trigger cannot drift when an earlier stage
    #: adds or drops a listing.
    PREFIXES: ClassVar[dict[str, str]] = {
        "climatology": "offsets/ref/",
        "offsets": "offsets/scene/",
        "composite": "composite/",
        # The export's artifacts are the COGs, which live outside the shard
        # prefix entirely -- completion is the canonical key, unchanged.
        "export": None,  # type: ignore[dict-item]
    }

    def __init__(
        self,
        storage,
        plan,
        stage,
        *,
        after: int = 2,
        when=None,
        run_id: str = RUN_ID,
    ) -> None:
        self._storage = storage
        self._plan = plan
        self._stage = stage
        self._after = after
        #: Land on a *condition* rather than a poll count. Counting polls is
        #: brittle once several helpers list the same prefix; a condition says
        #: what the test means -- "this stage finishes only after that happened"
        #: -- and makes the run deadlock if it never does.
        self._when = when
        self._run_id = run_id
        self._landed = False
        suffix = self.PREFIXES[stage]
        self._watched = (
            storage.cog_key(plan.window, plan.tile, PRODUCTS[0]).rsplit("/", 1)[0] + "/"
            if suffix is None
            else f"{shards.shard_root(run_id, plan.tile)}/{suffix}"
        )
        #: Listings of the watched prefix, so a test can show the watch polled.
        self.polls = 0

    def __getattr__(self, name):
        return getattr(self._storage, name)

    def list_prefix(self, prefix):
        if prefix != self._watched:
            return self._storage.list_prefix(prefix)

        self.polls += 1
        ready = self._when() if self._when is not None else self.polls >= self._after
        if ready and not self._landed:
            self._landed = True
            fleet = FakeFleet(self._storage, self._plan, run_id=self._run_id)
            fleet(
                stage=self._stage,
                run_id=self._run_id,
                tile=self._plan.tile,
                indexes=fleet.all_indexes(self._stage),
            )
        return self._storage.list_prefix(prefix)
