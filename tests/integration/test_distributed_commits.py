"""Integration test for per-tile Icechunk commits under concurrency.

Validates the "uncooperative distributed writes" pattern from ADR-001 Section 16:
- Multiple workers commit independently to the same branch
- ConflictError triggers retry with fresh session
- All commits eventually succeed

NOTE: This tests the per-tile commit pattern, NOT merge_sessions.
The original issue #7 described merge_sessions, but ADR-001 chose
per-tile commits for better fault tolerance in 10,000+ job pipelines.

IMPORTANT: LocalFileSystem storage is NOT safe for high-concurrency commits.
These tests use local storage for CI convenience. Production uses S3.
The 4-worker tests reliably pass; higher concurrency may lose commits on local FS.

Run with: pytest -m integration tests/integration/test_distributed_commits.py -v
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import icechunk
import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path


def _get_storage(path: Path) -> icechunk.Storage:
    """Create local Icechunk storage at path."""
    path.mkdir(parents=True, exist_ok=True)
    return icechunk.local_filesystem_storage(str(path))


def _worker_commit(
    storage_path: str,
    worker_id: int,
    max_retries: int = 10,
) -> tuple[int, str, int]:
    """Worker function that commits to Icechunk with conflict retry.

    This simulates what each tile processing worker does:
    opens own session, writes data, commits with retry on conflict.

    Args:
        storage_path: Path to Icechunk storage
        worker_id: Unique worker identifier
        max_retries: Max conflict retries

    Returns:
        Tuple of (worker_id, commit_id, attempts)
    """
    storage = icechunk.local_filesystem_storage(storage_path)

    for attempt in range(max_retries):
        try:
            repo = icechunk.Repository.open(storage)
            session = repo.writable_session("main")

            root = zarr.open_group(store=session.store, mode="r+")

            tile_data = np.full((10, 10), worker_id, dtype="f4")
            root.create_array(
                f"tile_{worker_id}",
                data=tile_data,
                chunks=(5, 5),
                overwrite=True,
            )

            commit_id = session.commit(f"Worker {worker_id} commit")
            return (worker_id, commit_id, attempt + 1)

        except icechunk.ConflictError:
            if attempt == max_retries - 1:
                raise
            continue

    msg = f"Worker {worker_id} exceeded max retries"
    raise RuntimeError(msg)


def _worker_commit_or_fail(
    storage_path: str,
    worker_id: int,
    fail_id: int = 2,
) -> tuple[int, str | None, int]:
    """Worker that fails for a specific worker_id, succeeds for others."""
    if worker_id == fail_id:
        raise ValueError(f"Simulated failure for worker {worker_id}")
    return _worker_commit(storage_path, worker_id)


@pytest.fixture
def icechunk_repo(tmp_path: Path) -> tuple[Path, str]:
    """Create initialized Icechunk repo, return (storage_path, initial_commit)."""
    storage_path = tmp_path / "icechunk"
    storage = _get_storage(storage_path)

    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")

    root = zarr.group(store=session.store, overwrite=True)
    root.attrs["initialized"] = True

    initial_commit = session.commit("Initialize repository")
    return (storage_path, initial_commit)


@pytest.mark.integration
class TestConcurrentCommits:
    """Test per-tile commit pattern under real concurrency."""

    def test_sequential_commits_baseline(self, icechunk_repo: tuple[Path, str]):
        """Baseline: sequential commits should always succeed."""
        storage_path, _ = icechunk_repo

        results = []
        for worker_id in range(4):
            result = _worker_commit(str(storage_path), worker_id)
            results.append(result)

        assert len(results) == 4
        assert all(r[2] == 1 for r in results)  # All first attempt

        storage = icechunk.local_filesystem_storage(str(storage_path))
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")
        root = zarr.open_group(store=session.store, mode="r")

        for i in range(4):
            assert f"tile_{i}" in root

    def test_concurrent_commits_with_conflicts(self, icechunk_repo: tuple[Path, str]):
        """Concurrent commits should succeed via conflict retry."""
        storage_path, _ = icechunk_repo
        num_workers = 4

        with contextlib.suppress(RuntimeError):
            mp.set_start_method("forkserver", force=True)

        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_worker_commit, str(storage_path), i) for i in range(num_workers)
            ]
            for future in as_completed(futures):
                results.append(future.result())

        assert len(results) == num_workers

        worker_ids = {r[0] for r in results}
        assert worker_ids == set(range(num_workers))

        total_attempts = sum(r[2] for r in results)
        assert total_attempts >= num_workers

        storage = icechunk.local_filesystem_storage(str(storage_path))
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")
        root = zarr.open_group(store=session.store, mode="r")

        for i in range(num_workers):
            assert f"tile_{i}" in root, f"Missing tile_{i} array"
            arr = root[f"tile_{i}"][:]
            assert arr[0, 0] == i, f"tile_{i} has wrong data"

    def test_partial_failure_isolation(self, icechunk_repo: tuple[Path, str]):
        """Failed workers don't affect successful ones - validates isolation.

        Uses sequential execution to test failure isolation cleanly.
        The key property: one job failing doesn't corrupt others.
        """
        storage_path, _ = icechunk_repo

        results = []
        errors = []

        for worker_id in range(4):
            try:
                if worker_id == 2:
                    raise ValueError(f"Simulated failure for worker {worker_id}")
                result = _worker_commit(str(storage_path), worker_id)
                results.append(result)
            except ValueError as e:
                errors.append((worker_id, str(e)))

        assert len(results) == 3
        assert len(errors) == 1
        assert errors[0][0] == 2

        storage = icechunk.local_filesystem_storage(str(storage_path))
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")
        root = zarr.open_group(store=session.store, mode="r")

        for i in [0, 1, 3]:
            assert f"tile_{i}" in root
        assert "tile_2" not in root

    @pytest.mark.skip(
        reason="LocalFileSystem unreliable at high concurrency - use S3 in production"
    )
    def test_high_concurrency_stress(self, icechunk_repo: tuple[Path, str]):
        """Stress test with 16 concurrent workers.

        SKIPPED: LocalFileSystem storage can lose commits under high concurrency.
        This test documents expected production behavior with S3 storage.
        """
        storage_path, _ = icechunk_repo
        num_workers = 16

        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_worker_commit, str(storage_path), i) for i in range(num_workers)
            ]
            for future in as_completed(futures):
                results.append(future.result())

        assert len(results) == num_workers

        storage = icechunk.local_filesystem_storage(str(storage_path))
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")
        root = zarr.open_group(store=session.store, mode="r")

        for i in range(num_workers):
            assert f"tile_{i}" in root

        ancestry = list(repo.ancestry(branch="main"))
        assert len(ancestry) == num_workers + 2
