"""What one shard does, and what it refuses to do.

The sharded pipeline's failure mode is not a crash. It is a tile assembled from
pieces that were each computed correctly against a different set of
assumptions, which merges into a plausible raster with the wrong pixels in it.
Everything here is a check on the assumptions rather than on the arithmetic:
the arithmetic is pinned with zero tolerance by
``tests/integration/test_shard_merge_equivalence.py``.

Three defences, in order of how quietly they would fail without a test. A plan
cut under a different configuration must be refused, because nothing
downstream would notice. A shard that finds its artifact already present must
exit rather than recompute, because the driver resubmits indexes that may
still be running. And a merge missing a partial must raise, because filling
the gap with NaN would turn a lost shard into a silently thinner composite.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from landsat_lst import shard_tasks, shards
from landsat_lst.config import settings
from landsat_lst.models import ProcessingJob
from landsat_lst.offsets import _times_iso
from landsat_lst.shard_tasks import (
    _offset_key,
    _time_coord,
    claim_export,
    climatology_group,
    job_for_window,
    load_context,
    merge_offsets,
    offsets_group,
    run_climatology_shard,
    run_composite_shard,
    run_offsets_stage,
)
from landsat_lst.storage import PRODUCTS, LocalStorage
from landsat_lst.tiling import parse_tile_name
from tests.unit.shard_fixtures import (
    COARSE,
    RUN_ID,
    TILE,
    WINDOW,
    FakeFleet,
    make_items,
    make_plan,
    publish_legacy_plan,
    publish_plan,
    stub_tile_geoboxes,
    write_offset_cache,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(output_dir=tmp_path / "bucket")


@pytest.fixture
def plan():
    return make_plan()


@pytest.fixture
def published(storage, plan):
    return publish_plan(storage, plan)


@pytest.fixture
def job():
    return ProcessingJob(tile=parse_tile_name(TILE), year=2021, end_year=2025)


class _PeerArrivesDuringTheWait:
    """A backend whose Nth block listing is when the peer shard finishes.

    Concurrency without threads. Running two fused tasks in real threads
    deadlocks against the estimator's own bounded pools, and the thing under
    test is not the pools -- it is whether this shard proceeds before its
    peer's blocks exist.
    """

    def __init__(self, storage, plan, *, after: int = 2) -> None:
        self._storage = storage
        self._plan = plan
        self._after = after
        self._polls = 0
        self.landed = False

    def __getattr__(self, name):
        return getattr(self._storage, name)

    def list_prefix(self, prefix: str):
        root = shards.shard_root(RUN_ID, self._plan.tile)
        if prefix == f"{root}/offsets/ref/":
            self._polls += 1
            if self._polls >= self._after and not self.landed:
                self.landed = True
                run_climatology_shard(RUN_ID, self._plan.tile, 1, storage=self._storage)
        return self._storage.list_prefix(prefix)


def _iso(stamps: list[str]) -> list[str]:
    """Spell a list of timestamps exactly as a plan stores them."""
    coord = SimpleNamespace(values=pd.to_datetime(stamps).values)
    return [str(stamp) for stamp in _times_iso(coord)]


def _record_phases(monkeypatch) -> list[str]:
    """Every phase the fused task reports, in order.

    The sub-phases are the point of the consolidation's observability clause:
    an acceptance run has to be able to attribute wall clock to resolve, to the
    climatology, to the barrier wait, and to the offsets separately, or the
    next round of tuning is guesswork again.
    """
    seen: list[str] = []
    import landsat_lst.progress as progress_module

    real = progress_module.report_phase

    def record(phase: str, **counts) -> None:
        seen.append(phase)
        real(phase, **counts)

    monkeypatch.setattr("landsat_lst.shard_tasks.report_phase", record)
    monkeypatch.setattr(progress_module, "report_phase", record)
    return seen


class TestWindowLabels:
    """The plan carries the label, not a second copy of the job."""

    @pytest.mark.parametrize(
        "job",
        [
            ProcessingJob(tile=parse_tile_name(TILE), year=2024),
            ProcessingJob(tile=parse_tile_name(TILE), year=2021, end_year=2025),
            ProcessingJob(tile=parse_tile_name(TILE), year=2021, end_year=2025, max_scenes=300),
        ],
    )
    def test_a_window_label_rebuilds_its_job(self, job):
        assert job_for_window(TILE, job.window_label) == job

    def test_a_label_this_project_does_not_produce_is_refused(self):
        with pytest.raises(ValueError, match="not a window label"):
            job_for_window(TILE, "last-summer")


class TestPlanDigest:
    """A shard whose configuration drifted must refuse the plan, not merge into it."""

    def test_a_plan_cut_under_a_different_configuration_is_refused(
        self, storage, plan, published, monkeypatch
    ):
        """The clamp is in the digest because it runs before the median sees a pixel."""
        monkeypatch.setattr(settings, "lst_valid_max", settings.lst_valid_max + 1.0)

        with pytest.raises(ValueError, match="cut under a different configuration"):
            load_context(RUN_ID, TILE, storage=storage)

    def test_a_plan_whose_scene_set_moved_is_refused(self, storage, plan, published):
        payload = json.loads(storage.read_text(shards.plan_key(published)))
        payload["scene_ids"] = [*payload["scene_ids"], "scene-99"]
        storage.write_text(shards.plan_key(published), json.dumps(payload))

        with pytest.raises(ValueError, match="cut under a different configuration"):
            load_context(RUN_ID, TILE, storage=storage)

    def test_a_matching_plan_loads(self, storage, plan, published):
        ctx = load_context(RUN_ID, TILE, storage=storage)

        assert ctx.plan.tile == TILE
        assert ctx.plan.window == WINDOW
        assert [item.id for item in ctx.items] == plan.scene_ids
        assert ctx.job.year == 2021

    def test_a_run_with_no_plan_says_which_stage_is_missing(self, storage):
        with pytest.raises(FileNotFoundError, match="resolve stage has not published"):
            load_context(RUN_ID, TILE, storage=storage)


class TestWorkAssignment:
    """Every shard's slice is a pure function of the plan and its index."""

    def test_the_climatology_groups_partition_the_blocks_exactly(self, plan):
        seen = []
        for index in range(plan.ref_shards):
            start, group = climatology_group(plan, index)
            assert plan.blocks[start : start + len(group)] == group
            seen.extend(group)
        assert seen == plan.blocks

    def test_the_scene_groups_partition_the_batches_exactly(self, plan):
        seen = []
        for index in range(plan.scene_shards):
            seen.extend(offsets_group(plan, index))
        assert seen == plan.scene_batches


class TestMergeOffsets:
    """The seam: partials in, one ordinary ADR-012 cache record out."""

    def test_a_missing_partial_is_an_error_not_a_thinner_answer(self, storage, plan, published):
        fleet = FakeFleet(storage, plan)
        fleet(stage="offsets", run_id=RUN_ID, tile=TILE, indexes=[0])

        with pytest.raises(ValueError, match="no partial"):
            merge_offsets(RUN_ID, TILE, storage=storage)

    def test_full_coverage_writes_the_canonical_offset_key(self, storage, plan, published):
        fleet = FakeFleet(storage, plan)
        fleet(stage="offsets", run_id=RUN_ID, tile=TILE, indexes=[0, 1])

        key = merge_offsets(RUN_ID, TILE, storage=storage)

        assert key.storage_key.startswith("_offsets/")
        record = json.loads(storage.read_text(key.storage_key))
        assert record["times"] == plan.scene_times
        assert record["offset"] == [0.25] * len(plan.scene_times)

    def test_a_second_merge_leaves_the_record_alone(self, storage, plan, published):
        fleet = FakeFleet(storage, plan)
        fleet(stage="offsets", run_id=RUN_ID, tile=TILE, indexes=[0, 1])
        key = merge_offsets(RUN_ID, TILE, storage=storage)
        first = storage.read_text(key.storage_key)

        merge_offsets(RUN_ID, TILE, storage=storage)

        assert storage.read_text(key.storage_key) == first


class TestClimatologyShard:
    """Blocks out, and a marker where a block holds no land."""

    def test_it_publishes_a_block_per_span_and_a_marker_for_ocean(
        self, storage, plan, published, monkeypatch
    ):
        """The marker is not a shortcut: it is the same answer, unwritten.

        ``climatology_by_blocks`` fills a land-free block with NaN without
        reading it, so a plane of NaN would cost more to upload than the block
        cost to produce.
        """
        _stub_coarse_load(monkeypatch, plan)
        root = shards.shard_root(RUN_ID, TILE)

        written = []
        for index in range(plan.ref_shards):
            written.extend(run_climatology_shard(RUN_ID, TILE, index, storage=storage))

        assert written == [
            shards.ref_block_key(root, 0),
            shards.ref_block_key(root, 1),
            shards.ref_block_key(root, 2),
            shards.ref_marker_key(root, 3),
        ]
        assert storage.read_text(shards.ref_marker_key(root, 3)) == ""

    def test_a_shard_that_finds_its_blocks_present_exits(
        self, storage, plan, published, monkeypatch
    ):
        """The driver resubmits indexes that may still be running."""
        _stub_coarse_load(monkeypatch, plan)
        run_climatology_shard(RUN_ID, TILE, 0, storage=storage)

        def explode(*args, **kwargs):
            raise AssertionError("an idempotent shard must not load anything")

        monkeypatch.setattr("landsat_lst.pipeline.load_scenes", explode)

        assert run_climatology_shard(RUN_ID, TILE, 0, storage=storage) == []


class TestCompositeShard:
    """One row band, both products, at the keys the export expects."""

    def test_it_writes_both_band_slabs(self, storage, plan, published, monkeypatch):
        _stub_native_load(monkeypatch, plan)
        write_offset_cache(storage, plan)
        root = shards.shard_root(RUN_ID, TILE)

        written = run_composite_shard(RUN_ID, TILE, 0, storage=storage)

        assert written == [shards.band_key(root, product, 0) for product in PRODUCTS]
        for key in written:
            assert storage.read_text is not None
            assert (storage.output_dir / key).stat().st_size > 0

    def test_a_band_with_its_slabs_already_present_exits(
        self, storage, plan, published, monkeypatch
    ):
        _stub_native_load(monkeypatch, plan)
        write_offset_cache(storage, plan)
        run_composite_shard(RUN_ID, TILE, 0, storage=storage)

        def explode(*args, **kwargs):
            raise AssertionError("an idempotent shard must not load anything")

        monkeypatch.setattr("landsat_lst.pipeline.load_scenes", explode)

        assert run_composite_shard(RUN_ID, TILE, 0, storage=storage) == []

    # A composite shard without merged offsets now *waits* rather than refusing
    # -- its VM was started early on purpose. See TestOffsetRecordWait.


class TestFusedOffsetsStage:
    """Four sub-phases, one boot.

    A shard computed for about six minutes while its stage held a fleet for
    about thirty: boots and queueing dominated, so the boundaries between
    resolve, climatology, and offsets became in-process waits. The order still
    matters -- phase B measures scenes against the *whole* climatology -- and so
    does skipping what a retry already finished, since the barrier resubmits
    indexes that may merely be slow.
    """

    @pytest.fixture(autouse=True)
    def _quick_polls(self, monkeypatch):
        monkeypatch.setattr(settings, "shard_unit_poll_s", 0.001)
        monkeypatch.setattr(settings, "shard_plan_wait_s", 1)
        monkeypatch.setattr(settings, "shard_block_wait_s", 1)

    def test_one_shard_runs_every_sub_phase_in_order(self, storage, plan, monkeypatch, job):
        """Resolve, then its own blocks, then the barrier, then its scenes.

        The peer's blocks are pre-published, so this shard's barrier clears
        immediately; what the barrier does when they are *not* there is the
        next test.
        """
        _stub_coarse_load(monkeypatch, plan)
        root = shards.shard_root(RUN_ID, TILE)

        def resolve(*_args, **_kwargs):
            """Shard 0's resolve, plus the peer that would be running alongside."""
            publish_plan(storage, plan)
            run_climatology_shard(RUN_ID, TILE, 1, storage=storage)

        monkeypatch.setattr(shard_tasks, "resolve_tile_plan", resolve)

        seen = _record_phases(monkeypatch)
        key = run_offsets_stage(RUN_ID, TILE, 0, job=job, storage=storage)

        assert seen.index("shard_resolve") < seen.index("shard_plan_wait")
        assert seen.index("shard_plan_wait") < seen.index("shard_barrier_wait")
        assert seen.index("shard_barrier_wait") < seen.index("destripe_offsets")
        assert key is not None
        assert len(storage.list_prefix(f"{root}/offsets/ref/")) == len(plan.blocks)

    def test_no_offset_is_estimated_until_the_whole_climatology_exists(
        self, storage, plan, published, monkeypatch
    ):
        """Phase B measures each scene against the *whole* climatology.

        The peer arrives mid-wait, which is what a fleet looks like from inside
        one shard. Without the barrier this shard would estimate against a half
        -built reference and the spy would see two blocks rather than four --
        an answer that is wrong and that nothing downstream inspects.
        """
        _stub_coarse_load(monkeypatch, plan)
        monkeypatch.setattr(settings, "shard_block_wait_s", 30)
        root = shards.shard_root(RUN_ID, TILE)
        blocks_when_estimated: list[int] = []

        peer = _PeerArrivesDuringTheWait(storage, plan, after=2)
        real = shard_tasks.run_offsets_shard

        def spy(*args, **kwargs):
            blocks_when_estimated.append(len(storage.list_prefix(f"{root}/offsets/ref/")))
            return real(*args, **kwargs)

        monkeypatch.setattr(shard_tasks, "run_offsets_shard", spy)

        run_offsets_stage(RUN_ID, TILE, 0, storage=peer)

        assert peer.landed, "the barrier must have actually waited"
        assert blocks_when_estimated == [len(plan.blocks)]

    def test_a_shard_that_is_not_zero_waits_for_the_plan(self, storage, plan, monkeypatch):
        """And fails loudly rather than hanging when shard 0 never publishes."""
        _stub_coarse_load(monkeypatch, plan)

        with pytest.raises(RuntimeError, match="never published"):
            run_offsets_stage(RUN_ID, TILE, 1, storage=storage)

    def test_shard_zero_without_a_plan_or_a_job_says_so(self, storage, plan, monkeypatch):
        _stub_coarse_load(monkeypatch, plan)

        with pytest.raises(ValueError, match="no job to resolve one from"):
            run_offsets_stage(RUN_ID, TILE, 0, storage=storage)

    def test_shard_zero_does_not_re_resolve_a_plan_that_exists(
        self, storage, plan, published, monkeypatch
    ):
        """Which is what makes a retry, and every shard of a resume, work."""
        _stub_coarse_load(monkeypatch, plan)
        run_climatology_shard(RUN_ID, TILE, 1, storage=storage)

        def explode(*args, **kwargs):
            raise AssertionError("a second resolve would query a live catalog again")

        monkeypatch.setattr(shard_tasks, "resolve_tile_plan", explode)

        assert run_offsets_stage(RUN_ID, TILE, 0, storage=storage) is not None

    def test_phase_b_waits_for_every_peers_blocks(self, storage, plan, published, monkeypatch):
        """A scene's offset is measured against the *whole* climatology.

        Shard 1 publishes only its own blocks, so the barrier must not clear.
        """
        _stub_coarse_load(monkeypatch, plan)

        with pytest.raises(RuntimeError, match="phase-A climatology"):
            run_offsets_stage(RUN_ID, TILE, 1, storage=storage)

        root = shards.shard_root(RUN_ID, TILE)
        assert storage.list_prefix(f"{root}/offsets/scene/") == {}

    def test_a_retry_skips_the_sub_phases_it_already_finished(
        self, storage, plan, published, monkeypatch
    ):
        """The driver resubmits indexes that may still be running."""
        _stub_coarse_load(monkeypatch, plan)
        for index in range(plan.ref_shards):
            run_climatology_shard(RUN_ID, TILE, index, storage=storage)
        for index in range(plan.scene_shards):
            run_offsets_stage(RUN_ID, TILE, index, storage=storage)

        def explode(*args, **kwargs):
            raise AssertionError("a retried fused task must not recompute finished work")

        monkeypatch.setattr("landsat_lst.pipeline.load_scenes", explode)

        assert run_offsets_stage(RUN_ID, TILE, 0, storage=storage) is None

    def test_a_shard_past_the_work_skips_its_phases(self, storage, plan, published, monkeypatch):
        """The fleet's width is fixed before the plan exists, so it can exceed it."""
        _stub_coarse_load(monkeypatch, plan)
        for index in range(plan.ref_shards):
            run_climatology_shard(RUN_ID, TILE, index, storage=storage)

        assert run_offsets_stage(RUN_ID, TILE, plan.scene_shards + 3, storage=storage) is None


class TestExportClaim:
    """The last band written runs the export, rather than a fleet booting to."""

    def test_the_last_band_claims_and_runs_it(self, storage, plan, published, monkeypatch):
        _stub_native_load(monkeypatch, plan)
        write_offset_cache(storage, plan)

        for index in range(len(plan.bands)):
            run_composite_shard(RUN_ID, TILE, index, storage=storage)

        root = shards.shard_root(RUN_ID, TILE)
        assert storage.read_text(shards.export_claim_key(root)) is not None
        assert storage.cog_exists(plan.window, TILE)

    def test_an_earlier_band_claims_nothing(self, storage, plan, published, monkeypatch):
        from landsat_lst.shard_tasks import load_context

        _stub_native_load(monkeypatch, plan)
        write_offset_cache(storage, plan)

        run_composite_shard(RUN_ID, TILE, 0, storage=storage)

        root = shards.shard_root(RUN_ID, TILE)
        assert storage.read_text(shards.export_claim_key(root)) is None
        assert not storage.cog_exists(plan.window, TILE)
        assert claim_export(load_context(RUN_ID, TILE, storage=storage), 0) is False

    def test_a_second_claimant_does_not_run_it_again(self, storage, plan, published, monkeypatch):
        """Wasted work, never corruption -- which is why this is a note, not a lock."""
        from landsat_lst.shard_tasks import load_context

        _stub_native_load(monkeypatch, plan)
        write_offset_cache(storage, plan)
        for index in range(len(plan.bands)):
            run_composite_shard(RUN_ID, TILE, index, storage=storage)

        ctx = load_context(RUN_ID, TILE, storage=storage)

        assert claim_export(ctx, 0) is False


class TestOffsetRecordWait:
    """A composite shard boots early on purpose, so it waits rather than refusing."""

    def test_it_polls_for_the_merged_record(self, storage, plan, published, monkeypatch):
        _stub_native_load(monkeypatch, plan)
        monkeypatch.setattr(settings, "shard_unit_poll_s", 0.001)
        monkeypatch.setattr(settings, "shard_offsets_record_wait_s", 1)

        with pytest.raises(FileNotFoundError, match="never merged"):
            run_composite_shard(RUN_ID, TILE, 0, storage=storage)

    def test_a_record_that_arrives_during_the_wait_is_used(
        self, storage, plan, published, monkeypatch
    ):
        _stub_native_load(monkeypatch, plan)
        monkeypatch.setattr(settings, "shard_unit_poll_s", 0.001)
        monkeypatch.setattr(settings, "shard_offsets_record_wait_s", 5)

        calls = {"n": 0}
        real = storage.read_text

        def read_text(key: str):
            calls["n"] += 1
            if calls["n"] > 3 and "_offsets/" in key and real(key) is None:
                write_offset_cache(storage, plan)
            return real(key)

        monkeypatch.setattr(storage, "read_text", read_text)

        assert run_composite_shard(RUN_ID, TILE, 0, storage=storage)


class TestLegacyPlanStamps:
    """A plan written before the nanosecond fix truncated its own time axis.

    The record-side fix was not enough. ``_time_coord`` rebuilds the offset axis
    from ``plan.scene_times``, so a legacy plan hands the join a
    second-precision axis and the composite fails exactly as it did before --
    which is what the packing probe hit on every arm, and what the S30W065
    acceptance rerun would hit again, because a resume reads the legacy plan
    rather than writing a new one.

    The items were never truncated: the loss happened on the way *into* the
    plan. That asymmetry is what makes recovery possible, and verifiable.
    """

    def test_the_axis_is_recovered_from_the_items(self, storage, plan):
        publish_legacy_plan(storage, plan)

        ctx = load_context(RUN_ID, TILE, storage=storage)

        assert ctx.plan.scene_times == plan.scene_times
        assert all("." in stamp for stamp in ctx.plan.scene_times)

    def test_the_recovered_axis_joins_against_a_full_precision_stack(
        self, storage, plan, monkeypatch
    ):
        """The probe's exact failure, end to end.

        Before the recovery this raised ``lst carries a time step the offsets
        do not ... ("not all values found in index 'time'")`` from
        ``keep.sel({time: lst.time})`` -- once per composite shard, on every arm.
        """
        publish_legacy_plan(storage, plan)
        _stub_native_load(monkeypatch, plan)
        write_offset_cache(storage, plan)

        written = run_composite_shard(RUN_ID, TILE, 0, storage=storage)

        assert written

    def test_the_digest_does_not_move(self, storage, plan):
        """It covers the scene ids and the settings, never the stamps.

        Which is what lets a legacy plan verify against a current process at
        all -- and why the recovery needs no re-signing.
        """
        root = publish_legacy_plan(storage, plan)
        stored = json.loads(storage.read_text(shards.plan_key(root)))["digest"]

        ctx = load_context(RUN_ID, TILE, storage=storage)

        assert ctx.plan.digest == stored

    def test_an_ambiguous_truncation_is_a_hard_error(self, storage):
        """Two *time steps* inside one second: the stored axis fits two real ones.

        Nothing in the items can decide which, so this is a refusal rather than
        a reading -- the same rule ``offsets._truncation_of`` follows.
        """
        crowded = replace(
            make_plan(),
            scene_times=_iso(
                [
                    "2021-07-04T13:45:12.100000",
                    "2021-07-04T13:45:12.900000",
                    "2021-09-03T13:45:13.482915",
                    "2021-11-03T13:45:14.482915",
                ]
            ),
        )
        publish_legacy_plan(storage, crowded)

        with pytest.raises(ValueError, match="truncates ambiguously"):
            load_context(RUN_ID, TILE, storage=storage)

    def test_extra_items_sharing_a_second_do_not_block_recovery(self, storage, plan):
        """Which is the ordinary case, not an exotic one.

        ``items.json`` holds one entry per scene and the axis one per solar-day
        group, so adjacent WRS rows of a single overpass land seconds -- often
        the same second -- apart. The group's timestamp is the earliest of
        them, because odc-stac sorts each group by ``nominal_datetime`` and
        takes the first.
        """
        items = make_items(plan)
        neighbour = json.loads(json.dumps(items[0]))
        neighbour["id"] = "scene-0-row-next"
        first = pd.Timestamp(plan.scene_times[0])
        neighbour["properties"]["datetime"] = (
            first + pd.Timedelta(microseconds=250_000)
        ).isoformat() + "Z"
        publish_legacy_plan(storage, plan, items=[*items, neighbour])

        ctx = load_context(RUN_ID, TILE, storage=storage)

        assert ctx.plan.scene_times == plan.scene_times

    def test_a_stamp_no_item_matches_is_a_hard_error(self, storage, plan):
        """The plan and the item list disagree; guessing would invent an axis."""
        wrong = make_items(plan)
        wrong[0]["properties"]["datetime"] = "2019-01-01T00:00:00.123456Z"
        publish_legacy_plan(storage, plan, items=wrong)

        with pytest.raises(ValueError, match=r"no item in items\.json matches"):
            load_context(RUN_ID, TILE, storage=storage)

    def test_a_current_plan_is_left_alone(self, storage, plan, published, monkeypatch):
        """The upgrade path must not touch a plan that never lost anything."""
        from landsat_lst import shard_tasks as tasks

        def explode(*args, **kwargs):
            raise AssertionError("a full-precision plan needs no recovery")

        monkeypatch.setattr(tasks, "_item_time_values", explode)

        assert load_context(RUN_ID, TILE, storage=storage).plan.scene_times == plan.scene_times

    def test_the_fused_offsets_stage_runs_against_a_legacy_plan(self, storage, plan, monkeypatch):
        """The resume case: S30W065 comes back to a plan it did not just write."""
        publish_legacy_plan(storage, plan)
        _stub_coarse_load(monkeypatch, plan)
        monkeypatch.setattr(settings, "shard_unit_poll_s", 0.001)
        monkeypatch.setattr(settings, "shard_block_wait_s", 5)
        for index in range(plan.ref_shards):
            run_climatology_shard(RUN_ID, TILE, index, storage=storage)

        key = run_offsets_stage(RUN_ID, TILE, 0, storage=storage)

        assert key is not None
        partial = json.loads(storage.read_text(key))
        assert all("." in stamp for stamp in partial["times"])

    def test_the_merge_of_legacy_partials_lands_on_the_recovered_axis(self, storage, plan):
        """Old partials carry old stamps; the merge already tolerates that."""
        publish_legacy_plan(storage, plan)
        root = shards.shard_root(RUN_ID, TILE)
        for index in range(plan.scene_shards):
            group = offsets_group(plan, index)
            start, stop = group[0][0], group[-1][1]
            storage.write_text(
                shards.scene_partial_key(root, start, stop),
                json.dumps(
                    {
                        "times": [s.split(".")[0] for s in plan.scene_times[start:stop]],
                        "offset": [0.25] * (stop - start),
                        "n_valid": [1000] * (stop - start),
                    }
                ),
            )

        key = merge_offsets(RUN_ID, TILE, storage=storage)

        record = json.loads(storage.read_text(key.storage_key))
        assert record["times"] == plan.scene_times


class TestAttemptNumbers:
    """Keyed by attempt, for the reason ``runs.py`` documents at length."""

    def test_the_first_attempt_is_one(self, storage):
        root = shards.shard_root(RUN_ID, TILE)
        assert shards.resolve_shard_attempt(storage, root, "climatology", 0) == 1

    def test_a_log_alone_counts(self, storage):
        """A VM preempted before it published state still uploaded a log."""
        root = shards.shard_root(RUN_ID, TILE)
        storage.write_text(shards.shard_log_key(root, "climatology", 0, 1), "boom")

        assert shards.resolve_shard_attempt(storage, root, "climatology", 0) == 2

    def test_one_shard_does_not_collect_another_shards_attempts(self, storage):
        root = shards.shard_root(RUN_ID, TILE)
        for attempt in (1, 2, 3):
            storage.write_text(shards.shard_state_key(root, "climatology", 10, attempt), "{}")

        assert shards.resolve_shard_attempt(storage, root, "climatology", 1) == 1
        assert shards.resolve_shard_attempt(storage, root, "climatology", 10) == 4


class TestFleetSizing:
    """A shard with no work is a VM that boots to bill a minute."""

    def test_widths_are_clamped_to_the_work_available(self, monkeypatch):
        monkeypatch.setattr(settings, "shard_climatology_vms", 99)
        monkeypatch.setattr(settings, "shard_offset_vms", 99)
        monkeypatch.setattr(settings, "shard_composite_vms", 99)

        assert shards.stage_shard_counts(blocks=4, scene_batches=2, block_rows=3) == (4, 2, 3)

    def test_zero_means_the_measured_projection(self, monkeypatch):
        from landsat_lst.projection import tile_projection

        monkeypatch.setattr(settings, "shard_climatology_vms", 0)
        monkeypatch.setattr(settings, "shard_composite_vms", 0)
        projected = tile_projection()

        ref, _scene, band = shards.stage_shard_counts(
            blocks=10_000, scene_batches=10_000, block_rows=10_000
        )

        assert ref == max(1, round(projected.n_vms_offsets))
        assert band == max(1, round(projected.n_vms_composite))


# --------------------------------------------------------------------------
# Stubs. Nothing here reaches a catalog or an object store beyond tmp_path.
# --------------------------------------------------------------------------


def _dataset(shape: tuple[int, int], times) -> xr.Dataset:
    """Raw-DN Landsat-like stack, which is what the shard paths consume.

    Raw DN rather than Celsius: both stages apply the QA mask and the
    conversion themselves, and handing them Celsius would skip the two steps
    the clamp and the offsets interact with.
    """
    height, width = shape
    rng = np.random.default_rng(7)
    celsius = 25.0 + rng.normal(0.0, 2.0, (len(times), height, width))
    dn = ((celsius + 273.15) - 149.0) / 0.00341802
    return xr.Dataset(
        {
            "lwir11": (["time", "latitude", "longitude"], dn.astype(np.float32)),
            "qa_pixel": (
                ["time", "latitude", "longitude"],
                np.full((len(times), height, width), 21824, dtype="uint16"),
            ),
        },
        coords={
            "time": times,
            "latitude": np.linspace(40.0, 35.0, height),
            "longitude": np.linspace(-75.0, -70.0, width),
        },
    )


def _stub_loader(monkeypatch, plan, shape: tuple[int, int], *, follow_geobox: bool = True):
    """Replace the scene load and the land mask with arrays of a known shape.

    ``follow_geobox`` is how the composite path gets a *band's* shape from the
    row-sliced geobox it was handed. The coarse path must not follow it: the
    real offsets geobox is the production 9,000 squared grid, and the fixture
    plan's blocks and ``coarse_shape`` describe an 8 squared one. Letting the
    two disagree gave the phase-B reduction a 9,000 squared scene and an 8
    squared climatology, which broadcast rather than failing anywhere useful.
    """
    times = np.array(plan.scene_times, dtype="datetime64[ns]")

    def load_scenes(items, bbox, **kwargs):
        geobox = kwargs.get("geobox")
        size = (
            (int(geobox.shape[0]), int(geobox.shape[1]))
            if follow_geobox and geobox is not None
            else shape
        )
        return _dataset(size, times)

    def build_land_mask(geobox, latitude, longitude):
        return xr.DataArray(
            np.ones((latitude.size, longitude.size), dtype=bool),
            dims=["latitude", "longitude"],
            coords={"latitude": latitude, "longitude": longitude},
        )

    monkeypatch.setattr("landsat_lst.pipeline.load_scenes", load_scenes)
    monkeypatch.setattr("landsat_lst.pipeline._build_land_mask", build_land_mask)
    monkeypatch.setattr("landsat_lst.pipeline._patch_url_for", lambda _items: None)
    # And the grids the tasks derive from the real tile name, which are the
    # production 18,000-column ones however small the plan is. See
    # stub_tile_geoboxes.
    stub_tile_geoboxes(monkeypatch, plan)


def _stub_coarse_load(monkeypatch, plan):
    _stub_loader(monkeypatch, plan, COARSE, follow_geobox=False)
    # The fixture's blocks are 4 px, far below the chunk edge ``_io_block_edge``
    # would insist on, so the block edge comes from the plan either way.
    monkeypatch.setattr(settings, "destripe_compute_panel", 2)


def _stub_native_load(monkeypatch, plan):
    _stub_loader(monkeypatch, plan, plan.native_shape)


def test_the_time_coordinate_round_trips_through_the_plan(plan):
    """The axis every per-scene answer is joined on, spelled one way.

    A plan whose stamps did not re-parse to the estimator's own would make
    every cache read a miss and every merge an error about an unknown scene.
    """
    from landsat_lst.offsets import _times_iso

    assert [str(s) for s in _times_iso(_time_coord(plan))] == plan.scene_times


def test_the_plan_keeps_the_sub_second_component_of_every_stamp(plan):
    """Which is where S30W065 died.

    The plan froze the axis at second precision, ``_time_coord`` rebuilt a
    truncated axis from it, and every composite shard then failed to join that
    estimate onto a stack loaded at full precision:
    ``lst carries a time step the offsets do not``. Real solar-day stamps have
    sub-second components; the synthetic fixtures did not, so nothing here
    could see it.
    """
    assert all("." in stamp for stamp in plan.scene_times)
    rebuilt = np.asarray(_time_coord(plan).values)
    assert (rebuilt.astype("datetime64[ns]") != rebuilt.astype("datetime64[s]")).any()


def test_the_offset_key_is_the_one_a_whole_tile_would_write(plan):
    from landsat_lst.offsets import OffsetKey

    assert _offset_key(plan) == OffsetKey.build(
        tile=plan.tile,
        window=plan.window,
        factor=plan.offset_factor,
        scene_ids=plan.scene_ids,
    )
