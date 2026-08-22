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

import numpy as np
import pytest
import xarray as xr

from landsat_lst import shards
from landsat_lst.config import settings
from landsat_lst.models import ProcessingJob
from landsat_lst.shard_tasks import (
    _offset_key,
    _time_coord,
    climatology_group,
    job_for_window,
    load_context,
    merge_offsets,
    offsets_group,
    run_climatology_shard,
    run_composite_shard,
)
from landsat_lst.storage import PRODUCTS, LocalStorage
from landsat_lst.tiling import parse_tile_name
from tests.unit.shard_fixtures import (
    COARSE,
    RUN_ID,
    TILE,
    WINDOW,
    FakeFleet,
    make_plan,
    publish_plan,
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

    def test_it_refuses_to_composite_without_merged_offsets(
        self, storage, plan, published, monkeypatch
    ):
        """Estimating per band instead would seam the tile at every boundary."""
        _stub_native_load(monkeypatch, plan)

        with pytest.raises(FileNotFoundError, match="no merged offsets"):
            run_composite_shard(RUN_ID, TILE, 0, storage=storage)


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


def _stub_loader(monkeypatch, plan, shape: tuple[int, int]):
    """Replace the scene load and the land mask with arrays of a known shape."""
    times = np.array(plan.scene_times, dtype="datetime64[ns]")

    def load_scenes(items, bbox, **kwargs):
        geobox = kwargs.get("geobox")
        size = (int(geobox.shape[0]), int(geobox.shape[1])) if geobox is not None else shape
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


def _stub_coarse_load(monkeypatch, plan):
    _stub_loader(monkeypatch, plan, COARSE)
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


def test_the_offset_key_is_the_one_a_whole_tile_would_write(plan):
    from landsat_lst.offsets import OffsetKey

    assert _offset_key(plan) == OffsetKey.build(
        tile=plan.tile,
        window=plan.window,
        factor=plan.offset_factor,
        scene_ids=plan.scene_ids,
    )
