"""Unit tests for static graph planning and dask profiling.

Nothing here loads a pixel, queries STAC, or starts a cluster. Every graph is
built against :func:`landsat_lst.profiling.synthetic_dataset`, which is the
point of the module under test: task counts follow from array shape and
chunking, so they are checkable against hand arithmetic.

Shapes stay small. The property under test is that a count tracks the geometry,
not that it reproduces a production tile, and a production tile would make this
suite minutes long.
"""

import json
import time

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.models import ProcessingJob
from landsat_lst.normalization import offset_graph
from landsat_lst.pipeline import TIME_CHUNK, compute_annual_composite
from landsat_lst.profiling import (
    GIB,
    MAX_PLAN_TASKS,
    MONTHS,
    PHASE_COMPOSITE,
    PHASE_OFFSETS,
    PRODUCTION_SCENES,
    PlanTooLarge,
    destripe_disabled,
    estimate_raw_tasks,
    graph_stats,
    plan_memory,
    plan_memory_record,
    plan_tile,
    predict_peak,
    profile_compute,
    sweep_plan,
    synthetic_dataset,
)
from landsat_lst.progress import TileHeartbeat
from landsat_lst.qa import apply_qa_mask, convert_to_celsius
from landsat_lst.storage import LocalStorage
from landsat_lst.tiling import parse_tile_name

pytestmark = pytest.mark.unit

TILE = "N40W075"


# ---------------------------------------------------------------- graph_stats


def test_graph_stats_rejects_a_collection_with_no_graph():
    """An eagerly loaded array has nothing to plan, and says so."""
    eager = xr.DataArray(np.zeros((4, 4)))
    with pytest.raises(TypeError, match="no dask graph"):
        graph_stats(eager)


def test_graph_stats_counts_blocks_from_the_chunking():
    """Block count is geometry: 4x4 chunks over 8x8 is four blocks."""
    array = da.zeros((8, 8), chunks=(4, 4))
    stats = graph_stats(xr.DataArray(array))
    assert stats.blocks == 4
    assert stats.tasks >= 4
    assert stats.layers >= 1


def test_graph_stats_prefixes_partition_the_task_count():
    """Every task lands in exactly one prefix bucket, so the parts sum."""
    array = da.zeros((64, 64), chunks=(16, 16)) + 1
    stats = graph_stats(xr.DataArray(array).mean())
    assert sum(p.tasks for p in stats.by_prefix) == stats.tasks
    # Sorted largest first, which is what `top` promises.
    assert list(stats.by_prefix) == sorted(stats.by_prefix, key=lambda p: -p.tasks)
    assert len(stats.top(2)) <= 2


def test_graph_stats_grows_with_the_number_of_blocks():
    """The count this module exists to predict tracks chunking, not values."""
    coarse = graph_stats(xr.DataArray(da.zeros((64, 64), chunks=(32, 32))))
    fine = graph_stats(xr.DataArray(da.zeros((64, 64), chunks=(8, 8))))
    assert fine.blocks == 64
    assert coarse.blocks == 4
    assert fine.tasks > coarse.tasks


def test_graph_stats_handles_a_dataset_whose_variables_disagree_on_chunks():
    """A reduction that rechunks time sits beside one that does not."""
    data = synthetic_dataset(shape=(64, 64), scenes=8, chunk_size=32)
    lst = convert_to_celsius(apply_qa_mask(data)["lwir11"])
    offset, n_valid = offset_graph(lst)
    stats = graph_stats(xr.Dataset({"offset": offset, "n_valid": n_valid}))
    assert stats.tasks > 0
    assert stats.blocks > 0


def _offsets(scenes=8, shape=(64, 64), chunk_size=32):
    """The offset pair for a small synthetic stack, as a single Dataset."""
    data = synthetic_dataset(shape=shape, scenes=scenes, chunk_size=chunk_size)
    offset, n_valid = offset_graph(convert_to_celsius(apply_qa_mask(data)["lwir11"]))
    return xr.Dataset({"offset": offset, "n_valid": n_valid})


def test_graph_stats_counts_the_fused_graph_by_default():
    """Fusion removes tasks, so the headline count sits below the raw one.

    This is the whole reason the default is not the raw graph: on a 300-scene
    N40W075 offset pass the raw graph holds 905,923 tasks where the run itself
    reported 598,604, and fusing brings the plan to 613,240.
    """
    stats = graph_stats(_offsets())
    assert stats.optimized is True
    assert stats.tasks < stats.raw_tasks
    assert stats.fusion > 1.0


def test_graph_stats_can_skip_fusion_and_says_so():
    """--fast trades comparability for speed, and marks the result."""
    raw = graph_stats(_offsets(), optimize=False)
    fused = graph_stats(_offsets())
    assert raw.optimized is False
    # With fusion skipped, the headline is the raw count itself.
    assert raw.tasks == raw.raw_tasks
    assert raw.fusion == 1.0
    assert raw.raw_tasks == fused.raw_tasks


def test_graph_stats_fusion_is_not_a_constant_factor():
    """Why a raw count cannot be corrected after the fact.

    Measured on real geometry: 1.48x on the offset graph at 300 scenes, 1.59x
    at 1,000, and 2.71x on the composite. A single divisor would be wrong
    somewhere, so the graph has to be fused rather than scaled.
    """
    shallow = graph_stats(_offsets(scenes=8)).fusion
    deep = graph_stats(_offsets(scenes=40)).fusion
    assert shallow != pytest.approx(deep, rel=0.01)


# --------------------------------------------------------------- predict_peak


def test_predict_peak_matches_hand_arithmetic():
    """The floor is three named terms, each checkable without running anything."""
    peak = predict_peak(
        scenes=2930, chunk_size=512, threads=4, height=9000, width=9000, baseline_gib=2.0
    )
    assert peak.stack_bytes == 4 * 512 * 512 * 2930 * 4
    assert peak.climatology_bytes == MONTHS * 9000 * 9000 * 4
    assert peak.baseline_bytes == int(2.0 * GIB)
    assert peak.total_bytes == (peak.stack_bytes + peak.climatology_bytes + peak.baseline_bytes)


def test_predict_peak_stack_term_is_linear_in_threads_and_scenes():
    """Halving threads halves the concurrent stacks. So does halving scenes."""
    base = predict_peak(scenes=800, chunk_size=512, threads=8, height=1024, width=1024)
    fewer = predict_peak(scenes=800, chunk_size=512, threads=4, height=1024, width=1024)
    shorter = predict_peak(scenes=400, chunk_size=512, threads=8, height=1024, width=1024)
    assert fewer.stack_bytes * 2 == base.stack_bytes
    assert shorter.stack_bytes * 2 == base.stack_bytes


def test_predict_peak_stack_term_is_quadratic_in_chunk_size():
    """512 -> 256 is a 4x cut, which is why the lever is worth having."""
    base = predict_peak(scenes=100, chunk_size=512, threads=2, height=1024, width=1024)
    small = predict_peak(scenes=100, chunk_size=256, threads=2, height=1024, width=1024)
    assert small.stack_bytes * 4 == base.stack_bytes


def test_predict_peak_climatology_does_not_grow_with_the_window():
    """Twelve buckets hold five years as readily as one. See ADR-005."""
    one_year = predict_peak(scenes=600, chunk_size=512, threads=4, height=900, width=900)
    five_year = predict_peak(scenes=2930, chunk_size=512, threads=4, height=900, width=900)
    assert one_year.climatology_bytes == five_year.climatology_bytes


def test_predict_peak_fits_in_compares_against_a_vm():
    """A configuration whose floor exceeds the VM is disqualified for free."""
    heavy = predict_peak(scenes=2930, chunk_size=512, threads=64, height=9000, width=9000)
    light = predict_peak(scenes=300, chunk_size=256, threads=2, height=9000, width=9000)
    assert not heavy.fits_in(64.0)
    assert light.fits_in(64.0)
    assert heavy.as_dict()["floor_gib"] == pytest.approx(heavy.total_gib, abs=0.01)


# ----------------------------------------------------------- synthetic_dataset


def test_synthetic_dataset_matches_what_load_scenes_returns():
    """Bands, dtypes, dims, and chunking are the contract the graph rests on.

    Shape and chunk edge are both rounded up to a whole number of delivered
    cells, exactly as ``load_scenes`` rounds its own native chunk. A real tile
    shape already is one; a hand-picked benchmark geometry is not, and the
    graph must be the same either way (ADR-017).
    """
    data = synthetic_dataset(shape=(1024, 512), scenes=40, chunk_size=256)
    assert set(data.data_vars) == {"lwir11", "qa_pixel"}
    assert data["lwir11"].dtype == np.uint16
    assert data["qa_pixel"].dtype == np.uint16
    assert data["lwir11"].dims == ("time", "latitude", "longitude")
    assert data.sizes == {"time": 40, "latitude": 1026, "longitude": 513}
    # Chunked exactly as pipeline.load_scenes chunks a real load.
    assert data["lwir11"].chunks[0][0] == TIME_CHUNK
    assert data["lwir11"].chunks[1][0] == 258


def test_synthetic_dataset_spans_every_calendar_month():
    """groupby("time.month") needs all twelve buckets filled to build the real graph."""
    data = synthetic_dataset(shape=(32, 32), scenes=60, start_year=2021, end_year=2025)
    months = set(np.unique(data["time"].dt.month.values).tolist())
    assert months == set(range(1, 13))


def test_synthetic_dataset_values_survive_the_plausibility_clamp():
    """DN bounds are chosen so convert_to_celsius keeps the stack, not drops it."""
    data = synthetic_dataset(shape=(64, 64), scenes=10, chunk_size=32)
    celsius = convert_to_celsius(data["lwir11"]).compute()
    assert np.isfinite(celsius).any()
    assert float(np.nanmin(celsius)) >= settings.lst_valid_min
    assert float(np.nanmax(celsius)) <= settings.lst_valid_max


def test_synthetic_dataset_qa_actually_masks_something():
    """A QA band that flagged nothing would build a graph the real one does not."""
    data = synthetic_dataset(shape=(64, 64), scenes=10, chunk_size=32, cloud_percent=50)
    masked = apply_qa_mask(data)["lwir11"].compute()
    assert bool(np.isnan(masked).any())


@pytest.mark.parametrize("shape,scenes", [((0, 32), 10), ((32, 0), 10), ((32, 32), 0)])
def test_synthetic_dataset_rejects_an_empty_geometry(shape, scenes):
    with pytest.raises(ValueError, match="positive shape and scene count"):
        synthetic_dataset(shape=shape, scenes=scenes)


def test_synthetic_dataset_feeds_the_real_composite_lazily():
    """The planner's whole premise: build the real graph, compute none of it."""
    data = synthetic_dataset(shape=(128, 128), scenes=24, chunk_size=64)
    with destripe_disabled():
        composite = compute_annual_composite(data)
    assert set(composite.data_vars) == {"lst_p95", "qa_count"}
    assert composite["lst_p95"].chunks is not None
    assert graph_stats(composite).tasks > 0


def test_destripe_disabled_restores_the_setting():
    original = settings.destripe
    with destripe_disabled():
        assert settings.destripe is False
    assert settings.destripe is original


def test_destripe_disabled_restores_after_an_exception():
    original = settings.destripe
    with pytest.raises(RuntimeError), destripe_disabled():
        raise RuntimeError("boom")
    assert settings.destripe is original


# ------------------------------------------------------------------ plan_tile


@pytest.fixture(scope="module")
def planned():
    """Both phases of a real tile, at a scene count that keeps the suite fast."""
    return plan_tile(tile=parse_tile_name(TILE), scenes=12, threads=4, optimize=False)


def test_plan_tile_reports_both_phases_in_order(planned):
    assert [p.name for p in planned] == [PHASE_OFFSETS, PHASE_COMPOSITE]
    assert all(p.graph.tasks > 0 for p in planned)


def test_plan_tile_uses_the_real_tile_grid(planned):
    """Native is the 18,000 squared of ADR-008; offsets run a factor coarser."""
    offsets, composite = planned
    assert (composite.height, composite.width) == (18000, 18000)
    factor = settings.destripe_offset_resolution_factor
    assert offsets.height == 18000 // factor
    assert offsets.width == 18000 // factor


def test_plan_tile_composite_blocks_follow_the_chunking(planned):
    """36 chunks of 512 per side covers 18,000 px, so 1,296 spatial blocks."""
    composite = planned[1]
    per_side = -(-18000 // settings.load_chunk_size)
    assert composite.graph.blocks >= per_side * per_side


def test_plan_tile_honours_an_explicit_chunk_size():
    """A smaller chunk means more blocks, and so a larger graph."""
    tile = parse_tile_name(TILE)
    coarse = plan_tile(tile=tile, scenes=10, chunk_size=1024, threads=2, optimize=False)
    fine = plan_tile(tile=tile, scenes=10, chunk_size=512, threads=2, optimize=False)
    assert fine[1].graph.tasks > coarse[1].graph.tasks
    assert fine[1].peak.stack_bytes < coarse[1].peak.stack_bytes


def test_plan_phase_serializes_for_the_json_flag(planned):
    payload = planned[0].as_dict()
    assert payload["phase"] == PHASE_OFFSETS
    assert payload["graph"]["tasks"] > 0
    assert payload["memory"]["floor_gib"] > 0
    json.dumps(payload)  # must round-trip for `landsat-lst plan --json`


# ---------------------------------------------------------------- plan_memory


def test_plan_memory_agrees_with_plan_tile(planned):
    """The anti-drift test: two pricing paths, one answer.

    A submission stores what plan_memory says and reconcile diffs against it,
    so a floor that disagreed with `landsat-lst plan` would make both useless.
    """
    offsets, composite = plan_memory(tile=parse_tile_name(TILE), scenes=12, threads=4)
    assert offsets == planned[0].peak
    assert composite == planned[1].peak


def test_plan_memory_charges_a_climatology_only_to_the_offsets_phase():
    """De-striping holds a resident float32 monthly reference. The composite does not.

    Its twelve-month band is a uint8 `qa_count` streamed to the writer, so
    charging it a climatology would put 14.5 GiB into every configuration.
    """
    offsets, composite = plan_memory(tile=parse_tile_name(TILE), scenes=300, threads=4)
    assert offsets.months == MONTHS
    assert offsets.climatology_bytes > 0
    assert composite.months == 0
    assert composite.climatology_bytes == 0


def test_plan_memory_prices_the_offsets_phase_on_the_coarse_grid():
    """Only the offset pass reads coarse, so only its floor moves with the factor."""
    tile = parse_tile_name(TILE)
    fine = plan_memory(tile=tile, scenes=300, threads=4, offset_factor=2)
    coarser = plan_memory(tile=tile, scenes=300, threads=4, offset_factor=4)

    assert coarser[0].height < fine[0].height
    assert coarser[0].climatology_bytes < fine[0].climatology_bytes
    assert coarser[0].total_bytes < fine[0].total_bytes
    assert coarser[1] == fine[1]


def test_plan_memory_builds_no_graph():
    """Pure arithmetic, so a 700-tile submission can afford to price every tile.

    Building the graphs for one production-scene tile runs into minutes, which
    is why the submission record carries floors and not task counts.
    """
    start = time.perf_counter()
    offsets, composite = plan_memory(tile=parse_tile_name(TILE), scenes=PRODUCTION_SCENES)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5
    assert offsets.scenes == PRODUCTION_SCENES
    assert composite.total_gib > 0


def test_plan_memory_record_round_trips_through_json():
    """It is stored on a submission record, so it has to survive a dump and a load."""
    record = plan_memory_record(tile=parse_tile_name(TILE), scenes=300, threads=4, chunk_size=512)
    restored = json.loads(json.dumps(record))

    assert restored == record
    assert restored["scenes"] == 300
    assert restored["chunk_size"] == 512
    assert restored["threads"] == 4
    assert restored["offset_factor"] == settings.destripe_offset_resolution_factor
    assert set(restored["phases"]) == {PHASE_OFFSETS, PHASE_COMPOSITE}
    assert restored["phases"][PHASE_OFFSETS]["floor_gib"] > 0
    assert restored["phases"][PHASE_COMPOSITE]["climatology_gib"] == 0


def test_plan_memory_record_matches_plan_memory():
    """The serializer reports the same floors it was handed, not a second estimate."""
    tile = parse_tile_name(TILE)
    offsets, composite = plan_memory(tile=tile, scenes=300, threads=4)
    record = plan_memory_record(tile=tile, scenes=300, threads=4)

    assert record["phases"][PHASE_OFFSETS] == offsets.as_dict()
    assert record["phases"][PHASE_COMPOSITE] == composite.as_dict()


# ---------------------------------------------------------------- build guard


def test_estimate_raw_tasks_tracks_blocks_and_time_chunks():
    """Blocks squared times time chunks: the shape of the allocation."""
    small = estimate_raw_tasks(height=1024, width=1024, chunk_size=512, scenes=10)
    # 2x2 blocks, 1 time chunk.
    assert small == 2 * 2 * 1 * 95

    # Halving the chunk quadruples the blocks.
    finer = estimate_raw_tasks(height=1024, width=1024, chunk_size=256, scenes=10)
    assert finer == small * 4

    # Ten times the scenes is ten times the time chunks.
    longer = estimate_raw_tasks(height=1024, width=1024, chunk_size=512, scenes=100)
    assert longer == small * 10


def test_guard_refuses_the_configuration_that_crashed_a_desktop():
    """`plan --sweep` at 2,930 scenes reached chunk 128 and took the machine down.

    The estimate has to catch it before a single task object is allocated, which
    is the whole point of checking geometry rather than measuring memory.
    """
    estimated = estimate_raw_tasks(height=18000, width=18000, chunk_size=128, scenes=2930)
    assert estimated > MAX_PLAN_TASKS

    with pytest.raises(PlanTooLarge, match="over the"):
        plan_tile(tile=parse_tile_name(TILE), scenes=2930, chunk_size=128, optimize=False)


def test_guard_allows_the_production_default():
    """The configuration people actually plan must not be blocked by the guard."""
    for height, width in ((9000, 9000), (18000, 18000)):
        estimated = estimate_raw_tasks(height=height, width=width, chunk_size=512, scenes=2930)
        assert estimated < MAX_PLAN_TASKS


def test_plan_too_large_names_the_levers():
    """A refusal that does not say what to change is only half an answer."""
    with pytest.raises(PlanTooLarge) as excinfo:
        plan_tile(tile=parse_tile_name(TILE), scenes=2930, chunk_size=128, optimize=False)
    message = str(excinfo.value)
    assert "--chunk" in message
    assert "--scenes" in message
    assert "--max-tasks" in message


def test_guard_checks_before_building_anything(monkeypatch):
    """The refusal must beat the allocation, or it buys nothing."""
    import landsat_lst.profiling as profiling_module

    def explode(**_kwargs):
        raise AssertionError("synthetic_dataset was called despite the guard")

    monkeypatch.setattr(profiling_module, "synthetic_dataset", explode)
    with pytest.raises(PlanTooLarge):
        plan_tile(tile=parse_tile_name(TILE), scenes=2930, chunk_size=128, optimize=False)


def test_sweep_drops_an_unplannable_chunk_rather_than_failing():
    """A sweep finds viable configurations; an unviable one is not a crash."""
    rows = sweep_plan(
        tile=parse_tile_name(TILE),
        scenes=10,
        chunk_sizes=(512, 8),
        thread_counts=(2,),
        optimize=False,
    )
    assert {r.chunk_size for r in rows} == {512}


# ----------------------------------------------------------------- sweep_plan


@pytest.fixture(scope="module")
def swept():
    """Ranking configurations needs only the floor, so fusion is skipped here."""
    return sweep_plan(
        tile=parse_tile_name(TILE),
        scenes=10,
        chunk_sizes=(1024, 512),
        thread_counts=(2, 8),
        vm_gib=64.0,
        optimize=False,
    )


def test_sweep_plan_covers_every_combination(swept):
    assert len(swept) == 4
    assert {(r.chunk_size, r.threads) for r in swept} == {(1024, 2), (1024, 8), (512, 2), (512, 8)}


def test_sweep_plan_orders_by_floor(swept):
    assert [r.floor_gib for r in swept] == sorted(r.floor_gib for r in swept)


def test_sweep_plan_builds_one_graph_per_chunk_size(swept):
    """Task count follows chunking; thread count only changes blocks in flight.

    Equal counts across thread rows are the observable consequence of building
    each chunk size's graph once rather than once per row.
    """
    for chunk_size in (1024, 512):
        rows = [r for r in swept if r.chunk_size == chunk_size]
        assert len({r.offsets_tasks for r in rows}) == 1
        assert len({r.composite_tasks for r in rows}) == 1


def test_sweep_plan_fewer_threads_lowers_the_floor(swept):
    for chunk_size in (1024, 512):
        rows = {r.threads: r for r in swept if r.chunk_size == chunk_size}
        assert rows[2].floor_gib < rows[8].floor_gib
        assert rows[2].as_dict()["fits"] is rows[2].fits


# ------------------------------------------------------------- profile_compute


@pytest.fixture
def profiling_on(monkeypatch, tmp_path):
    """Turn profiling on and point its local fallback at a scratch directory."""
    monkeypatch.setattr(settings, "profile_dask", True)
    monkeypatch.setattr(settings, "profile_dask_interval_s", 0.05)
    monkeypatch.setattr(settings, "manifest_dir", tmp_path / "runs")
    return tmp_path


def _small_compute():
    (da.ones((64, 64), chunks=(16, 16)) * 2).sum().compute(scheduler="threads")


def test_profile_compute_is_inert_when_disabled(monkeypatch, tmp_path):
    """Off by default, and off means it writes nothing at all."""
    monkeypatch.setattr(settings, "profile_dask", False)
    monkeypatch.setattr(settings, "manifest_dir", tmp_path / "runs")
    with profile_compute("noop"):
        _small_compute()
    assert not (tmp_path / "runs" / "profiles").exists()


def test_profile_compute_writes_a_summary_without_a_heartbeat(profiling_on):
    """A local run has no run prefix, so the dump falls back to disk."""
    with profile_compute("local_label"):
        _small_compute()

    dump = profiling_on / "runs" / "profiles" / "local_label.profile.json"
    payload = json.loads(dump.read_text())
    assert payload["label"] == "local_label"
    assert payload["run_id"] is None
    assert payload["tasks"]["total"] > 0
    assert payload["tasks"]["by_prefix"]
    # CacheProfiler stays off unless asked for separately.
    assert "cache" not in payload


def test_profile_compute_attributes_wall_time_to_task_prefixes(profiling_on):
    """The gap GraphProgress cannot close: which tasks own the hour."""
    with profile_compute("attribution"):
        _small_compute()

    payload = json.loads(
        (profiling_on / "runs" / "profiles" / "attribution.profile.json").read_text()
    )
    top = payload["tasks"]["by_prefix"][0]
    assert set(top) == {"prefix", "tasks", "seconds", "mean_s", "share"}
    assert 0.0 <= top["share"] <= 1.0
    # Sorted by seconds, because the question is who owns the clock.
    shares = [row["share"] for row in payload["tasks"]["by_prefix"]]
    assert shares == sorted(shares, reverse=True)


def test_profile_compute_adds_cache_records_only_when_asked(profiling_on, monkeypatch):
    """Gated separately: it retains one record per task. See the ADR-011 risk."""
    monkeypatch.setattr(settings, "profile_dask_cache", True)
    with profile_compute("with_cache"):
        _small_compute()

    payload = json.loads(
        (profiling_on / "runs" / "profiles" / "with_cache.profile.json").read_text()
    )
    assert payload["cache"]["entries"] > 0
    assert payload["cache"]["peak_bytes"] > 0


def test_profile_compute_publishes_beside_the_heartbeat(profiling_on, monkeypatch):
    """Inside a batch tile the dump lands in the run prefix reconciliation reads.

    It carries the heartbeat's attempt number, so the profile of the attempt
    that ran long is not overwritten by the retry that ran short.
    """
    storage = LocalStorage(output_dir=profiling_on / "cogs")
    job = ProcessingJob(tile=parse_tile_name(TILE), year=2021, end_year=2025)
    with (
        TileHeartbeat(run_id="run-1", job=job, storage=storage, attempt=2, interval_s=3600),
        profile_compute("destripe_offsets"),
    ):
        _small_compute()

    key = storage.profile_key("run-1", TILE, "destripe_offsets", 2)
    payload = json.loads(storage.read_text(key))
    assert payload["run_id"] == "run-1"
    assert payload["tile"] == TILE
    assert payload["resource"]["samples"] >= 0
    # The state object the same process published, under the same number.
    assert storage.read_text(storage.run_record_key("run-1", TILE, 2)) is not None
    assert storage.read_text(storage.profile_key("run-1", TILE, "destripe_offsets", 1)) is None


def test_profile_compute_never_fails_the_work_it_wraps(profiling_on, monkeypatch):
    """A dump that cannot be written is logged and dropped, not raised."""

    def explode(*_args, **_kwargs):
        msg = "bucket on fire"
        raise OSError(msg)

    monkeypatch.setattr(LocalStorage, "write_text", explode)
    with profile_compute("doomed"):
        _small_compute()  # must complete regardless


def test_profile_compute_propagates_the_bodys_own_exception(profiling_on):
    """Instrumentation must not swallow a real failure either."""
    with pytest.raises(RuntimeError, match="tile died"), profile_compute("failing"):
        msg = "tile died"
        raise RuntimeError(msg)

    # The partial profile is still written, which is the point of dumping in a
    # finally: a tile that died is exactly the one worth profiling.
    assert (profiling_on / "runs" / "profiles" / "failing.profile.json").exists()
