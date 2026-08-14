"""Fused task counts for the two graphs a tile builds.

Task count is arithmetic over array shape and chunking, so it is exactly
reproducible: the pinned configuration returned 2,668 and 828 on three
consecutive runs. The bands below are nonetheless wide, because the number worth
catching is a restructured graph, not a rounding difference.

Every count here is taken **after** ``dask.optimize``. That is the graph the
scheduler runs and the one ``GraphProgress`` counts against, and fusion is not a
constant factor -- 1.48x on the offset graph at 300 scenes, 2.71x on the
composite. A raw count cannot be scaled into a real one.
"""

from __future__ import annotations

import pytest

from landsat_lst.benchmarks import (
    GRAPH_COMPOSITE,
    GRAPH_OFFSETS,
    Geometry,
)

pytestmark = pytest.mark.benchmark

#: Measured on the pinned configuration (24 scenes, 4x4 blocks of 256, 2
#: threads). Deterministic to the task across runs *for a given dependency set*,
#: which is not the same as constant: upgrading dask to 2026.7.1 and xarray to
#: 2026.7.0 moved the composite from 828 to 845 while the offset graph did not
#: budge. That is 2.1%, against a band that tolerates 40% and fails at the 1.60x
#: a deleted rechunk costs -- the bands are sized for exactly this. Left at the
#: originally measured values rather than re-pinned on every upgrade, since a
#: constant that chases the measurement stops being a reference.
OFFSET_TASKS = 2_668
COMPOSITE_TASKS = 828

#: Half to double, for the offset graph. A graph that restructures moves task
#: count by more than this; nothing else does. Chosen so that a 3% drift never
#: fails a build and a doubling always does.
BAND = 2.0

#: Tighter, for the composite. Its known failure mode is smaller than a
#: doubling: deleting the shared time rechunk from ``_composite_graph`` takes
#: this configuration from 828 tasks to 1,326, a factor of 1.60. A 2.0 band
#: would let that through, and it is the exact regression ADR-013 exists to
#: prevent. Task count is deterministic arithmetic over shape and chunking, so
#: 1.4 still leaves room for every change that is not structural.
COMPOSITE_BAND = 1.4


def _assert_band(actual: int, expected: int, what: str, band: float = BAND) -> None:
    low, high = expected / band, expected * band
    assert low <= actual <= high, (
        f"{what}: {actual:,} tasks against a pinned {expected:,} "
        f"(band {low:,.0f}-{high:,.0f}). The graph restructured. Confirm the "
        f"change was intended, re-run scripts/synthetic_scaling.py on a VM for "
        f"the production number, and re-pin this constant."
    )


def test_offset_graph_task_count_is_pinned(pinned):
    """``offset_graph`` holds the task count it was measured at."""
    _assert_band(pinned.offset_tasks, OFFSET_TASKS, "offset_graph")


def test_composite_graph_task_count_is_pinned(pinned):
    """``_composite_graph`` holds the task count it was measured at.

    The band is the one that catches a deleted time rechunk. See
    :data:`COMPOSITE_BAND` and :mod:`tests.benchmark.test_native_passes`.
    """
    _assert_band(pinned.composite_tasks, COMPOSITE_TASKS, "_composite_graph", COMPOSITE_BAND)


def test_offset_graph_dominates_the_composite(pinned):
    """The offset pass is the expensive graph, and has to stay the one to beat.

    27 of the ~35 minutes in a 300-scene tile went to ``scene_offsets``, which is
    why its whole output is cached. If the composite ever overtakes it, the
    cache is aimed at the wrong phase and ADR-012's arithmetic no longer holds.
    """
    assert pinned.offset_tasks > pinned.composite_tasks, (
        f"composite ({pinned.composite_tasks:,}) has overtaken offsets "
        f"({pinned.offset_tasks:,}); ADR-012 caches the wrong phase now"
    )


@pytest.mark.parametrize(
    ("graph", "attr"),
    [(GRAPH_OFFSETS, "offset_tasks"), (GRAPH_COMPOSITE, "composite_tasks")],
)
def test_task_count_grows_with_scenes(measure_one, graph, attr):
    """Both graphs grow with the time axis, and neither grows faster than it.

    A graph whose task count grows super-linearly in scene count would make a
    five-year window quadratically more expensive than a one-year one, which is
    the shape that turns a planned run into an unplanned one.
    """
    small = measure_one(
        f"{graph}_12", Geometry(scenes=12, blocks=4, chunk=256, threads=2, graph=graph)
    )
    large = measure_one(
        f"{graph}_48", Geometry(scenes=48, blocks=4, chunk=256, threads=2, graph=graph)
    )

    small_tasks, large_tasks = getattr(small, attr), getattr(large, attr)
    assert large_tasks > small_tasks, f"{graph} task count did not grow with scenes"
    # 4x the scenes must not cost more than 8x the tasks.
    assert large_tasks <= 8 * small_tasks, (
        f"{graph} grew {large_tasks / small_tasks:.1f}x for 4x the scenes; "
        "something in the graph scales worse than the time axis"
    )
