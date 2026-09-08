"""The futures driver against an in-memory executor, in milliseconds.

What is pinned is the shape of the run, not any number: which futures depend
on which, that a published shard is never recomputed on resume, that an error
is classified only after the bucket is checked, that the worker cap binds
every stage, that a bounded run names its bands and submits no export, and
that the deadline fires at every blocking point. Nothing here imports coiled,
frisky, or distributed, and a test asserts the driver does not either.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from landsat_lst import futures_driver, shards
from landsat_lst.config import settings
from landsat_lst.futures_driver import (
    Deadline,
    NullObserver,
    ShardFuturesFailed,
    classify_future_error,
    derive_run_timeout_s,
    drive_tile_futures,
    reconcile,
    resume_tile_futures,
)
from landsat_lst.models import ProcessingJob
from landsat_lst.shard_driver import ShardBackendMismatch
from landsat_lst.storage import PRODUCTS
from landsat_lst.tiling import parse_tile_name
from tests.unit.futures_fixtures import (
    RUN_ID,
    TILE,
    FakeClock,
    FakeExecutor,
    FakeTask,
    fresh_bucket,
)
from tests.unit.shard_fixtures import make_plan

pytestmark = pytest.mark.unit


@pytest.fixture
def storage(tmp_path):
    return fresh_bucket(tmp_path)


@pytest.fixture
def plan():
    return make_plan()


@pytest.fixture
def job():
    return ProcessingJob(tile=parse_tile_name(TILE), year=2021, end_year=2025)


@pytest.fixture(autouse=True)
def _fast(monkeypatch, fast_barriers):
    monkeypatch.setattr(settings, "futures_observer_poll_s", 1.0)
    monkeypatch.setattr(settings, "futures_offsets_stall_s", 0)
    monkeypatch.setattr(settings, "futures_retries", 2)
    monkeypatch.setattr(settings, "futures_max_workers", 16)


class RecordingObserver(NullObserver):
    def __init__(self) -> None:
        self.registered: list = []
        self.completions: list = []
        self.stalled: list = []
        self.verified: list[str] = []
        self.finalized = False
        self.ticks = 0

    def register(self, ref):
        self.registered.append(ref)

    def note_completion(self, key, *, result, error):
        self.completions.append((key, result, error))

    def note_stalled(self, keys):
        self.stalled.append(list(keys))

    def tick(self):
        self.ticks += 1

    def verify(self, phase):
        self.verified.append(phase)
        return {"phase": phase}

    def finalize(self):
        self.finalized = True
        return {"passed": True, "failures": [], "required": ["x"]}


def _drive(storage, plan, job, *, executor=None, task=None, **kwargs):
    executor = executor or FakeExecutor(clock=FakeClock())
    task = task or FakeTask(storage, plan)
    summary = drive_tile_futures(
        job,
        executor=executor,
        run_id=RUN_ID,
        storage=storage,
        task=task,
        clock=executor.clock,
        **kwargs,
    )
    return summary, executor, task


def _by_stage(executor, stage):
    return [s for s in executor.submissions if f"-{stage}-" in s.key]


# ---------------------------------------------------------------------------
# The edges
# ---------------------------------------------------------------------------


class TestDependencyWiring:
    def test_every_stage_depends_on_the_one_before(self, storage, plan, job):
        summary, executor, _task = _drive(storage, plan, job)

        resolve = _by_stage(executor, "resolve")
        offsets = _by_stage(executor, "offsets")
        merge = _by_stage(executor, "merge")
        composite = _by_stage(executor, "composite")
        export = _by_stage(executor, "export")
        assert len(resolve) == 1 and len(merge) == 1 and len(export) == 1
        assert len(offsets) == settings.shard_offset_vms
        assert len(composite) == len(plan.bands)
        assert all([d.key for d in s.deps] == [resolve[0].key] for s in offsets)
        assert sorted(d.key for d in merge[0].deps) == sorted(s.key for s in offsets)
        assert all([d.key for d in s.deps] == [merge[0].key] for s in composite)
        assert sorted(d.key for d in export[0].deps) == sorted(s.key for s in composite)
        assert summary.completed
        assert summary.status == "completed_unobserved", "no observer: correct pixels, no gate"

    def test_every_future_is_submitted_with_the_configured_retries(self, storage, plan, job):
        _summary, executor, _task = _drive(storage, plan, job)
        assert {s.retries for s in executor.submissions} == {settings.futures_retries}

    def test_the_plan_future_is_skipped_when_a_plan_exists(self, storage, plan, job):
        from tests.unit.shard_fixtures import publish_plan

        publish_plan(storage, plan)
        _summary, executor, _task = _drive(storage, plan, job)
        assert not _by_stage(executor, "resolve")
        offsets = _by_stage(executor, "offsets")
        assert all(s.deps == [] for s in offsets)


class TestWorkerCap:
    def test_offsets_then_composite_width_never_above_the_cap(
        self, storage, plan, job, monkeypatch
    ):
        monkeypatch.setattr(settings, "futures_max_workers", 1)
        monkeypatch.setattr(settings, "shard_offset_vms", 1)
        _summary, executor, _task = _drive(storage, plan, job)
        assert executor.ensure_calls == [1, 1]

    def test_composite_scales_to_the_band_count_under_the_cap(self, storage, plan, job):
        _summary, executor, _task = _drive(storage, plan, job)
        assert executor.ensure_calls == [settings.shard_offset_vms, len(plan.bands)]

    def test_an_offsets_width_above_the_cap_is_refused_before_anything_submits(
        self, storage, plan, job, monkeypatch
    ):
        monkeypatch.setattr(settings, "futures_max_workers", 1)
        monkeypatch.setattr(settings, "shard_offset_vms", 2)
        executor = FakeExecutor(clock=FakeClock())
        with pytest.raises(ShardFuturesFailed) as info:
            _drive(storage, plan, job, executor=executor)
        assert info.value.terminal
        assert info.value.cause == "offsets_width_exceeds_cap"
        assert executor.submissions == []


# ---------------------------------------------------------------------------
# Bounded runs
# ---------------------------------------------------------------------------


class TestBoundedRuns:
    def test_only_the_named_bands_are_submitted_and_no_export_without_finalize(
        self, storage, plan, job
    ):
        summary, executor, task = _drive(storage, plan, job, bands=[1], finalize=False)
        composite = _by_stage(executor, "composite")
        assert [s.key.split("-")[-2] for s in composite] == ["0001"]
        assert not _by_stage(executor, "export")
        assert summary.bands_requested == [1]
        assert summary.completed, "the requested bands landed"
        assert summary.missing["composite"] == []
        assert summary.missing["export"] == [0], "finalization was declined, not done"
        assert ("composite", 0) not in task.computed
        assert not storage.cog_exists(plan.window, TILE)

    def test_an_unknown_band_is_refused(self, storage, plan, job):
        with pytest.raises(ShardFuturesFailed) as info:
            _drive(storage, plan, job, bands=[99], finalize=False)
        assert info.value.cause == "unknown_bands"
        assert info.value.terminal


# ---------------------------------------------------------------------------
# Errors, retries, resume
# ---------------------------------------------------------------------------


class TestErrors:
    def test_a_transient_failure_is_retried_by_the_scheduler_and_the_tile_completes(
        self, storage, plan, job
    ):
        executor = FakeExecutor(clock=FakeClock(), scripts={"-composite-": "fail_once"})
        summary, executor, _task = _drive(storage, plan, job, executor=executor)
        assert summary.completed
        assert all(s.attempts == 2 for s in _by_stage(executor, "composite"))
        assert summary.errors == [], "a scheduler-retried error never reaches the driver"

    def test_a_permanent_failure_names_the_missing_band_and_leaves_the_others_published(
        self, storage, plan, job
    ):
        executor = FakeExecutor(clock=FakeClock(), scripts={"-composite-N40W075-0001-": "never"})
        with pytest.raises(ShardFuturesFailed) as info:
            _drive(storage, plan, job, executor=executor)
        failure = info.value
        assert not failure.terminal
        assert failure.missing["composite"] == [1]
        root = shards.shard_root(RUN_ID, TILE)
        assert storage.read_text(shards.band_key(root, PRODUCTS[0], 0)) is not None
        assert "resume" in str(failure)
        assert failure.summary is not None and failure.summary.status == "incomplete"
        assert any(e["stage"] == "composite" and e["index"] == 1 for e in failure.summary.errors)

    def test_a_shard_that_published_and_then_died_is_a_success(self, storage, plan, job):
        executor = FakeExecutor(clock=FakeClock(), scripts={"-composite-": "fail_after"})
        summary, _executor, _task = _drive(storage, plan, job, executor=executor)
        assert summary.completed
        assert summary.missing["composite"] == []
        assert all(e["stage"] == "composite" for e in summary.errors)

    def test_a_terminal_error_releases_every_pending_future(self, storage, plan, job):
        executor = FakeExecutor(clock=FakeClock(), scripts={"-composite-N40W075-0000-": "quota"})
        with pytest.raises(ShardFuturesFailed) as info:
            _drive(storage, plan, job, executor=executor)
        assert info.value.terminal
        assert info.value.cause == "composite_terminal"
        export = executor.futures[_by_stage(executor, "export")[0].key]
        assert export.released
        assert info.value.summary is not None
        assert info.value.summary.stopped_by == "composite_terminal"

    def test_a_dependency_failure_is_transient_for_the_dependent(self, storage, plan, job):
        executor = FakeExecutor(clock=FakeClock(), scripts={"-merge-": "never"})
        with pytest.raises(ShardFuturesFailed) as info:
            _drive(storage, plan, job, executor=executor)
        assert not info.value.terminal
        assert info.value.missing["composite"] == list(range(len(plan.bands)))


class TestResume:
    def test_a_resume_submits_everything_and_recomputes_nothing(self, storage, plan, job):
        _summary, _executor, task = _drive(storage, plan, job)
        assert task.computed
        again = FakeExecutor(clock=FakeClock())
        task2 = FakeTask(storage, plan)
        summary = resume_tile_futures(
            RUN_ID, TILE, executor=again, storage=storage, task=task2, clock=again.clock
        )
        assert summary.completed
        assert task2.computed == [("merge", 0)], (
            "the merge is idempotent and cheap; nothing else ran"
        )
        assert len(again.submissions) == 1 + settings.shard_offset_vms + len(plan.bands) + 1
        skipped = [r for r in summary.results if r.get("skipped")]
        assert len(skipped) == len(summary.results) - 1

    def test_a_resume_before_the_plan_exists_says_so(self, storage):
        with pytest.raises(FileNotFoundError, match="published no plan"):
            resume_tile_futures(RUN_ID, TILE, executor=FakeExecutor(), storage=storage)


# ---------------------------------------------------------------------------
# The deadline and the stall report
# ---------------------------------------------------------------------------


class TestDeadline:
    def test_the_deadline_fires_in_the_settle_loop_and_releases_the_rest(self, storage, plan, job):
        clock = FakeClock()
        executor = FakeExecutor(clock=clock, scripts={"-composite-N40W075-0001-": "hang"})
        deadline = Deadline(clock=clock, total_s=30.0)
        with pytest.raises(ShardFuturesFailed) as info:
            _drive(storage, plan, job, executor=executor, deadline=deadline)
        assert info.value.cause == "run_timeout"
        assert not info.value.terminal
        assert "waiting for" in info.value.detail
        hung = executor.futures[_by_stage(executor, "composite")[1].key]
        assert hung.released
        assert info.value.summary is not None and info.value.summary.stopped_by == "run_timeout"

    def test_the_deadline_fires_while_waiting_for_workers(self, storage, plan, job):
        clock = FakeClock()

        class SlowBoot(FakeExecutor):
            def ensure_workers(self, n, *, timeout_s):
                self.clock.advance(timeout_s + 1)

        executor = SlowBoot(clock=clock)
        deadline = Deadline(clock=clock, total_s=10.0)
        with pytest.raises(ShardFuturesFailed) as info:
            _drive(storage, plan, job, executor=executor, deadline=deadline)
        assert info.value.cause == "run_timeout"
        assert "workers" in info.value.detail
        assert executor.submissions == []

    def test_the_deadline_fires_while_waiting_for_the_plan(self, storage, plan, job):
        clock = FakeClock()
        executor = FakeExecutor(clock=clock, scripts={"-resolve-": "hang"})
        deadline = Deadline(clock=clock, total_s=5.0)
        with pytest.raises(ShardFuturesFailed) as info:
            _drive(storage, plan, job, executor=executor, deadline=deadline)
        assert info.value.cause == "run_timeout"
        assert "plan" in info.value.detail

    def test_derived_timeout_covers_boot_stages_and_waves(self, plan, monkeypatch):
        monkeypatch.setattr(settings, "futures_run_timeout_s", None)
        one_wave = derive_run_timeout_s(plan, bands=2, workers=2)
        two_waves = derive_run_timeout_s(plan, bands=4, workers=2)
        assert two_waves > one_wave
        monkeypatch.setattr(settings, "futures_run_timeout_s", 1234)
        assert derive_run_timeout_s(plan, bands=4, workers=2) == 1234.0
        assert derive_run_timeout_s(None, bands=1, workers=1) > 0


class TestStalledOffsets:
    def test_an_offsets_future_pending_while_peers_run_is_reported(self, storage, plan, job):
        clock = FakeClock()
        executor = FakeExecutor(clock=clock, scripts={"-offsets-N40W075-0001-": "hang"})
        observer = RecordingObserver()
        deadline = Deadline(clock=clock, total_s=20.0)
        with pytest.raises(ShardFuturesFailed):
            _drive(storage, plan, job, executor=executor, deadline=deadline, observer=observer)
        assert observer.stalled, "the stalled offsets index was reported"
        assert any("-offsets-N40W075-0001-" in k for keys in observer.stalled for k in keys)


# ---------------------------------------------------------------------------
# Observer contract, summary, reconciliation, classification
# ---------------------------------------------------------------------------


class TestObserverAndSummary:
    def test_the_observer_sees_every_future_and_the_gate_decides_the_status(
        self, storage, plan, job
    ):
        observer = RecordingObserver()
        summary, executor, _task = _drive(storage, plan, job, observer=observer)
        assert {r.key for r in observer.registered} == {s.key for s in executor.submissions}
        assert len(observer.completions) == len(executor.submissions)
        assert "submitted" in observer.verified and "complete" in observer.verified
        assert observer.finalized
        assert summary.status == "accepted"
        assert summary.observability["passed"] is True

    def test_a_raising_observer_never_fails_the_tile(self, storage, plan, job):
        class Exploding(RecordingObserver):
            def register(self, ref):
                raise RuntimeError("observer down")

            def finalize(self):
                raise RuntimeError("observer down")

        summary, _executor, _task = _drive(storage, plan, job, observer=Exploding())
        assert summary.completed
        assert summary.status == "completed_unobserved"

    def test_the_summary_is_published_to_the_bucket(self, storage, plan, job):
        summary, _executor, _task = _drive(storage, plan, job)
        key = f"{shards.fleet_root(RUN_ID)}/{TILE}/state/futures.summary.json"
        payload = json.loads(storage.read_text(key))
        assert payload["status"] == summary.status
        assert payload["completed"] is True
        assert {s["stage"] for s in payload["stages"]} == {
            "resolve",
            "offsets",
            "merge",
            "composite",
            "export",
        }

    def test_reconcile_reads_the_bucket_per_stage(self, storage, plan):
        from tests.unit.shard_fixtures import publish_plan

        publish_plan(storage, plan)
        root = shards.shard_root(RUN_ID, TILE)
        missing = reconcile(storage, plan, root)
        assert missing["composite"] == list(range(len(plan.bands)))
        assert missing["export"] == [0]
        storage.write_text(shards.band_key(root, PRODUCTS[0], 0), "x")
        storage.write_text(shards.band_key(root, PRODUCTS[1], 0), "x")
        assert reconcile(storage, plan, root, bands=[0])["composite"] == []
        assert reconcile(storage, plan, root)["composite"] == [1]


class TestStorageRefusal:
    def test_a_real_executor_on_local_storage_is_refused(self, storage, plan, job):
        executor = FakeExecutor(clock=FakeClock())
        executor.scheduler = "frisky"
        with pytest.raises(ShardBackendMismatch):
            drive_tile_futures(job, executor=executor, run_id=RUN_ID, storage=storage)


class TestClassification:
    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (RuntimeError("you have reached the workspace quota"), "terminal"),
            (RuntimeError(""), "transient"),
            (type("KilledWorker", (RuntimeError,), {})("lost"), "transient"),
            (type("CommClosedError", (OSError,), {})("closed"), "transient"),
            (ValueError("plan digest drift: settings moved"), "terminal"),
            (RuntimeError("inner scheduler is not the configured one"), "terminal"),
            (FileNotFoundError("band slab composite/lst_p95/band003.tif is missing"), "terminal"),
            (OSError("Connection reset"), "transient"),
        ],
    )
    def test_terminal_versus_transient(self, exc, expected):
        assert classify_future_error(exc) == expected


def test_the_driver_imports_nothing_backend_specific():
    """The driver is backend-neutral by construction, not by discipline."""
    source = Path(futures_driver.__file__).read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"coiled", "frisky", "distributed"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names & {"coiled", "frisky", "distributed"}, "no backend object is touched in code"


def test_a_spot_policy_or_vm_type_never_appears_in_the_driver():
    source = Path(futures_driver.__file__).read_text()
    assert "spot" not in source.lower()
    assert "r6i" not in source and "m6i" not in source
