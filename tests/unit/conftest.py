"""Unit-scope fixtures.

The shard fixtures live here rather than in a helper module because a fixture
imported by name and then taken as a test argument reads as a redefinition, and
because ``_restore_shard_settings`` has to run for anything that touches shard
code whether or not the test asked for it.
"""

from __future__ import annotations

import pytest

from landsat_lst.config import settings


@pytest.fixture(autouse=True)
def _restore_shard_settings():
    """Put back the settings a shard process pins globally.

    Every shard pins ``load_chunk_size`` to ``shard_composite_chunk`` so the
    plan digest agrees across the fleet (``shard_tasks.apply_shard_settings``).
    In a VM that is the point; in a test process it is one global mutation
    leaking into whatever runs next.
    """
    original = settings.load_chunk_size
    yield
    settings.load_chunk_size = original


@pytest.fixture
def fast_barriers(monkeypatch):
    """No sleeping and no waiting: a fake fleet finishes before the first poll."""
    monkeypatch.setattr(settings, "shard_barrier_timeout_s", 0)
    monkeypatch.setattr(settings, "shard_driver_poll_s", 0.001)
    monkeypatch.setattr(settings, "shard_barrier_rounds", 2)
