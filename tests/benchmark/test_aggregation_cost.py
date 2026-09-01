"""What aggregating to nominal ~100 m costs, and what it must not cost.

ADR-017 moves the composite onto a grid with nine times fewer cells. The
tempting summary -- "so everything falls by nine" -- is wrong in both
directions, and this file exists to keep the real shape of the change visible
rather than assumed:

- **The read does not fall at all.** Every delivered cell is reduced from nine
  source cells, and those nine are still fetched and decoded. ADR-013's single
  native pass has to survive the extra reduction, and that is the first test
  here.
- **The graph gets bigger, not smaller.** Three coarsen reductions over the
  full stack (weighted sum, weighted count, unweighted count) add tasks. What
  shrinks is everything *downstream* of them.
- **The estimator must not move.** Its grid is a coarsening of the source grid
  and has nothing to do with the output. An offset task count that differed
  between the two arms would be the exact defect issue #120 warns against.

Bands, not values, for the reason ``tests/benchmark/`` states throughout: a
benchmark that fails on drift gets disabled within a month.
"""

from __future__ import annotations

import pytest

from landsat_lst.benchmarks import (
    GRAPH_COMPOSITE,
    GRAPH_OFFSETS,
    Geometry,
)

pytestmark = pytest.mark.benchmark

#: The pinned CI geometry in both arms. Same scenes, same blocks, same chunk,
#: same threads: the only difference is the delivered grid.
AGGREGATED = Geometry(scenes=24, blocks=4, chunk=256, threads=2, graph=GRAPH_COMPOSITE)
SOURCE_GRID = Geometry(
    scenes=24, blocks=4, chunk=256, threads=2, graph=GRAPH_COMPOSITE, aggregate=False
)

#: Measured on this geometry, 2026-09-01: 828 composite tasks on the source
#: grid against 2,094 aggregated, a factor of 2.53. The three coarsen
#: reductions are the whole of it.
TASK_RATIO = 2.53
TASK_RATIO_BAND = 1.4

#: Measured peak RSS: 295 MB source-grid, 266 MB aggregated. Nearly flat at CI
#: geometry, because a 1,026 squared stack of 24 scenes never approaches the
#: percentile working set that dominates a production tile. The assertion is
#: therefore one-sided -- aggregation must not cost memory -- rather than a
#: claim that it saves a particular amount here.
PEAK_HEADROOM = 1.25


def test_aggregation_keeps_the_single_native_pass(measure_one):
    """ADR-013 survives ADR-017: the source stack is still read once.

    The reduction sits between the load and both consumers, so a source block
    could plausibly be materialized once per coarsen. It is not: within one
    ``dask.compute`` each key is produced once whatever is downstream.
    """
    m = measure_one("composite_100m", AGGREGATED)

    assert m.native_passes == pytest.approx(1.0), (
        f"{m.native_passes:.2f} passes over the source stack with aggregation on. "
        "Aggregating to the delivered grid must not cost an extra read; every "
        "delivered cell already descends from the nine source cells read once."
    )


def test_the_source_grid_arm_also_reads_once(measure_one):
    """The other side of the comparison, so 1.0 above means something."""
    m = measure_one("composite_source_grid", SOURCE_GRID)

    assert m.native_passes == pytest.approx(1.0)


def test_the_offset_graph_is_identical_on_both_grids(measure_one):
    """The estimator does not move because the output moved.

    Its grid is ``source / destripe_offset_resolution_factor``, its accuracy
    bound was calibrated there, and ``destripe_min_scene_pixels`` counts its
    pixels. Issue #120 names this as the thing not to break by accident. Task
    count is deterministic arithmetic over shape and chunking, so the two arms
    must agree exactly rather than within a band.
    """
    aggregated = measure_one(
        "offsets_with_aggregation",
        Geometry(scenes=24, blocks=4, chunk=256, threads=2, graph=GRAPH_OFFSETS),
    )
    source_grid = measure_one(
        "offsets_without_aggregation",
        Geometry(scenes=24, blocks=4, chunk=256, threads=2, graph=GRAPH_OFFSETS, aggregate=False),
    )

    assert aggregated.offset_tasks == source_grid.offset_tasks, (
        f"offset graph moved from {source_grid.offset_tasks:,} to "
        f"{aggregated.offset_tasks:,} tasks when the output grid changed. The "
        "estimator reads the source grid in both arms and must not notice."
    )


def test_aggregation_costs_tasks_and_not_memory(measure_one):
    """The honest shape of the change: more graph, no more peak.

    Asserting the task *ratio* rather than either count keeps this readable as
    what it is -- a statement about what the reduction adds -- and keeps it
    valid when a dependency upgrade moves both counts together, which they have
    done before (828 to 845 on one dask bump).
    """
    aggregated = measure_one("composite_100m_tasks", AGGREGATED)
    source_grid = measure_one("composite_source_grid_tasks", SOURCE_GRID)

    ratio = aggregated.composite_tasks / source_grid.composite_tasks
    low, high = TASK_RATIO / TASK_RATIO_BAND, TASK_RATIO * TASK_RATIO_BAND
    assert low <= ratio <= high, (
        f"aggregation multiplied composite tasks by {ratio:.2f} "
        f"({source_grid.composite_tasks:,} -> {aggregated.composite_tasks:,}), "
        f"outside the pinned {TASK_RATIO} x{TASK_RATIO_BAND}. Either a coarsen "
        "reduction was added or removed, or the delivered grid moved."
    )

    assert aggregated.peak_rss_mb <= source_grid.peak_rss_mb * PEAK_HEADROOM, (
        f"aggregated peak {aggregated.peak_rss_mb:.0f} MB against "
        f"{source_grid.peak_rss_mb:.0f} MB on the source grid. Reducing before "
        "the percentile must not cost memory; if it does, the reduction is "
        "holding the stack rather than streaming it."
    )
