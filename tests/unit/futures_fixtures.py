"""An in-memory executor for the futures driver, credential-less.

``FakeExecutor`` runs each submitted callable synchronously the moment every
future it depends on is done, honours ``retries`` by re-invoking on an
exception, and lets a test script a task to fail once, fail always, or
disappear like a lost worker. ``FakeTask`` writes the same artifacts a real
shard writes (through the shard fixtures' key grammar) so the driver's
reconciliation reads a bucket that looks real.

Everything is deterministic and runs in milliseconds; there is no thread, no
socket, and no scheduler. What is exercised is the driver's own logic: the
edges it submits, the order it awaits, what it does with an error, what it
counts as done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from landsat_lst import shards
from landsat_lst.futures_tasks import ShardResult
from landsat_lst.shard_driver import Clock
from landsat_lst.storage import PRODUCTS, LocalStorage
from tests.unit.shard_fixtures import RUN_ID, TILE, publish_plan, write_offset_cache


class FakeClock(Clock):
    """A clock that only moves when told to, or a little on every sleep."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += max(0.0, seconds)

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeFuture:
    def __init__(self, key: str) -> None:
        self.key = key
        self._done = False
        self._result: Any = None
        self._error: BaseException | None = None
        self.released = False
        self.status = "pending"

    def done(self) -> bool:
        return self._done

    def result(self, timeout: float | None = None) -> Any:
        if self._error is not None:
            raise self._error
        return self._result

    def release(self) -> None:
        self.released = True

    def _finish(self, result: Any) -> None:
        self._done = True
        self._result = result
        self.status = "finished"

    def _fail(self, error: BaseException) -> None:
        self._done = True
        self._error = error
        self.status = "error"


@dataclass
class Submission:
    key: str
    fn: Any
    args: tuple
    kwargs: dict
    retries: int
    deps: list[FakeFuture]
    attempts: int = 0


class FakeExecutor:
    """Synchronous, dependency-aware, scriptable.

    ``scripts`` maps a substring of a future key to a behaviour:
    ``"fail_once"`` raises on the first attempt only; ``"never"`` raises on
    every attempt; ``"killed"`` raises a ``KilledWorker``-named error on the
    first attempt; ``"hang"`` never completes (the deadline test);
    ``"fail_after"`` runs the task and then raises on every attempt, which is
    a shard that published and died; ``"quota"`` raises a terminal control-plane
    error.
    """

    scheduler = "fake"
    dashboard_url = None
    cluster_id = 7

    def __init__(
        self,
        *,
        n_workers: int = 2,
        clock: FakeClock | None = None,
        scripts: dict[str, str] | None = None,
    ) -> None:
        self.n_workers = n_workers
        self.clock = clock or FakeClock()
        self.scripts = scripts or {}
        self.submissions: list[Submission] = []
        self.futures: dict[str, FakeFuture] = {}
        self.ensure_calls: list[int] = []
        self.ran: list[str] = []
        self.hung: set[str] = set()

    def submit(self, fn, *args, key: str, retries: int, **kw) -> FakeFuture:
        deps = [a for a in args if isinstance(a, FakeFuture)]
        future = FakeFuture(key)
        self.futures[key] = future
        self.submissions.append(Submission(key, fn, args, kw, retries, deps))
        return future

    def _behaviour(self, key: str) -> str | None:
        for needle, behaviour in self.scripts.items():
            if needle in key:
                return behaviour
        return None

    def _run(self, submission: Submission) -> None:
        future = self.futures[submission.key]
        behaviour = self._behaviour(submission.key)
        if behaviour == "hang":
            self.hung.add(submission.key)
            return
        args = tuple((a.result() if isinstance(a, FakeFuture) else a) for a in submission.args)
        while True:
            submission.attempts += 1
            try:
                if behaviour == "never":
                    raise RuntimeError(f"scripted permanent failure of {submission.key}")
                if behaviour == "fail_once" and submission.attempts == 1:
                    raise RuntimeError(f"scripted transient failure of {submission.key}")
                if behaviour == "killed" and submission.attempts == 1:
                    raise type("KilledWorker", (RuntimeError,), {})(
                        f"worker lost under {submission.key}"
                    )
                if behaviour == "quota":
                    raise RuntimeError("you have reached the workspace quota of 400 Coiled credits")
                self.ran.append(submission.key)
                value = submission.fn(*args, **submission.kwargs)
                if behaviour == "fail_after":
                    # The shard published, then the worker died on the way out.
                    raise RuntimeError(f"scripted failure after publishing {submission.key}")
                future._finish(value)
                return
            except Exception as exc:
                if submission.attempts > submission.retries:
                    future._fail(exc)
                    return

    def _runnable(self, pending: list[FakeFuture]) -> Submission | None:
        for submission in self.submissions:
            future = self.futures[submission.key]
            if future.done() or future not in pending or submission.key in self.hung:
                continue
            if all(d.done() and not self.futures[d.key]._error for d in submission.deps):
                return submission
            if any(self.futures[d.key]._error for d in submission.deps):
                # A dependency failed: the scheduler fails the dependent too.
                dep = next(d for d in submission.deps if self.futures[d.key]._error)
                future._fail(
                    RuntimeError(f"dependency {dep.key} failed: {self.futures[dep.key]._error}")
                )
                return None
        return None

    def as_completed(self, futures, *, timeout_s=None):
        pending = [f for f in futures if not f.done()]
        for future in futures:
            if future.done():
                yield future
                return
        submission = self._runnable(pending)
        if submission is None:
            newly = [f for f in futures if f.done()]
            if newly:
                yield newly[0]
                return
            self.clock.advance(timeout_s or 1.0)
            raise TimeoutError
        self._run(submission)
        future = self.futures[submission.key]
        if future.done():
            yield future
            return
        self.clock.advance(timeout_s or 1.0)
        raise TimeoutError

    def ensure_workers(self, n: int, *, timeout_s: float) -> None:
        self.ensure_calls.append(n)
        self.n_workers = n

    def worker_addresses(self) -> list[str]:
        return [f"fake-worker-{i}" for i in range(self.n_workers)]


@dataclass
class FakeTask:
    """Writes what a real shard writes, and records what it computed."""

    storage: LocalStorage
    plan: Any
    computed: list[tuple[str, int]] = field(default_factory=list)
    skipped: list[tuple[str, int]] = field(default_factory=list)

    def __call__(
        self, stage: str, run_id: str, tile: str, index: int, *_deps, **kwargs
    ) -> ShardResult:
        root = shards.shard_root(run_id, tile)
        handler = {
            "resolve": self._resolve,
            "offsets": self._offsets,
            "merge": self._merge,
            "composite": self._composite,
            "export": self._export,
        }[stage]
        keys, did = handler(root, tile, index)
        (self.computed if did else self.skipped).append((stage, index))
        return ShardResult(
            stage=stage,
            index=index,
            key=kwargs.get("outer_key"),
            keys=keys,
            skipped=not did,
            wall_s=1.0,
            peak_rss_gb=0.1,
            worker="fake-worker-0",
            pid=1,
            hostname="fake",
            inner_scheduler=str(kwargs.get("inner_scheduler", "threads")),
            inner_threads=None,
            trace_id="0x1",
            sink_error=None,
        )

    def _write_once(self, key: str, body: str = "x") -> bool:
        if self.storage.read_text(key) is not None:
            return False
        self.storage.write_text(key, body, content_type="application/octet-stream")
        return True

    def _resolve(self, root: str, _tile: str, _index: int) -> tuple[list[str], bool]:
        did = False
        if self.storage.read_text(shards.plan_key(root)) is None:
            publish_plan(self.storage, self.plan)
            did = True
        return [shards.plan_key(root)], did

    def _offsets(self, root: str, _tile: str, index: int) -> tuple[list[str], bool]:
        from landsat_lst.shard_tasks import offsets_group

        for offset in range(len(self.plan.blocks)):
            block_key = (
                shards.ref_block_key(root, offset)
                if self.plan.block_has_land[offset]
                else shards.ref_marker_key(root, offset)
            )
            self._write_once(block_key, "")
        if index >= self.plan.scene_shards:
            return [], False
        group = offsets_group(self.plan, index)
        key = shards.scene_partial_key(root, group[0][0], group[-1][1])
        return [key], self._write_once(key, "{}")

    def _merge(self, _root: str, _tile: str, _index: int) -> tuple[list[str], bool]:
        write_offset_cache(self.storage, self.plan)
        return ["_offsets/merged"], True

    def _composite(self, root: str, _tile: str, index: int) -> tuple[list[str], bool]:
        keys = [shards.band_key(root, product, index) for product in PRODUCTS]
        did = [self._write_once(key, "slab") for key in keys]
        return keys, any(did)

    def _export(self, _root: str, tile: str, _index: int) -> tuple[list[str], bool]:
        keys = [self.storage.cog_key(self.plan.window, tile, product) for product in PRODUCTS]
        did = [self._write_once(key, "cog") for key in keys]
        return keys, any(did)


def fresh_bucket(tmp_path) -> LocalStorage:
    return LocalStorage(output_dir=tmp_path / "bucket")


__all__ = [
    "RUN_ID",
    "TILE",
    "FakeClock",
    "FakeExecutor",
    "FakeFuture",
    "FakeTask",
    "fresh_bucket",
]
