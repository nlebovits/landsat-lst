"""One native pass per tile, measured with a memory number beside it.

``tests/integration/test_cog.py`` already pins this property from both sides,
and keeps doing so: those are fast correctness checks and belong where they are.
What they cannot report is what the second pass costs, and cost is why ADR-013
exists. A tile used to read the full native stack three times; on a synthetic
tile that was exactly 3.0x one pass.

The pass count is exact and scale-free, which makes it the most reliable guard
in this directory. Counting block executions rather than task keys is what
survives fusion, since fusion renames keys freely.
"""

from __future__ import annotations

import pytest

from landsat_lst.benchmarks import (
    GRAPH_COMPOSITE,
    GRAPH_EXPORT,
    GRAPH_EXPORT_SEPARATE,
    Geometry,
)

pytestmark = pytest.mark.benchmark

#: Small enough that a COG write is seconds. The property is a ratio, so the
#: geometry only has to produce more blocks than threads.
EXPORT_GEOMETRY = Geometry(scenes=8, blocks=4, chunk=256, threads=2, graph=GRAPH_EXPORT)


def test_cog_export_costs_one_pass_over_the_sources(measure_one):
    """Both COG products come out of one pass. See ADR-013 and issue #80."""
    m = measure_one("export_fused", EXPORT_GEOMETRY)

    assert m.native_passes == pytest.approx(1.0), (
        f"{m.native_passes:.2f} passes over the native stack. "
        "cog_export must hand both deferred stores to one dask.compute; "
        "exporting one product at a time gives the second pass back."
    )


def test_separate_exports_still_cost_a_pass_each(measure_one):
    """The other side of the same property, so 1.0 means something.

    Without this, a change that stopped the tally counting would read as a pass
    saved rather than as a broken benchmark.
    """
    m = measure_one(
        "export_separate",
        Geometry(scenes=8, blocks=4, chunk=256, threads=2, graph=GRAPH_EXPORT_SEPARATE),
    )

    assert m.native_passes == pytest.approx(2.0), (
        f"{m.native_passes:.2f} passes from two separate exports, expected 2.0. "
        "The read tally is no longer counting what it thinks it counts."
    )


#: Composite-only, at the pinned geometry. Re-measured 266 MB on 2026-09-01,
#: after ADR-017 moved the composite onto the delivered grid; it was 306-310 MB
#: on the source grid. The regression this guards -- deleting the shared time
#: rechunk -- took the source-grid graph to 842 MB, a factor of 2.73.
COMPOSITE_PEAK_MB = 266.0

#: Composite-only task count at the same geometry: 2,094 with aggregation,
#: against 828 without it.
#:
#: **The 1.60x figure below was measured on the pre-ADR-017 graph and has not
#: been re-derived.** Deleting the rechunk no longer produces a comparable
#: number: ``nanquantile_last`` needs a whole time core dimension, and without
#: the rechunk the composite child now fails outright rather than building a
#: more expensive graph. That is a stricter failure than the one this band was
#: sized for, so the band is retained rather than widened, and the memory
#: assertion above remains the sharper of the two guards.
COMPOSITE_ONLY_TASKS = 2_094

#: Wide enough for a different runner, narrow enough for a 2.73x regression.
COMPOSITE_PEAK_BAND = 2.0


def test_composite_graph_shares_one_time_rechunk(measure_one):
    """The rechunk in ``_composite_graph`` is what keeps the fused write cheap.

    ``quantile`` needs the whole time series per pixel and inserts the rechunk
    itself; ``groupby("time.month").sum()`` does not. Two differently chunked
    consumers means every source block is materialized twice, and the fused
    write then has no block order satisfying both -- it fans out and holds the
    stack. Removing the shared rechunk took a 4096 squared x 120 synthetic tile
    from 1.60 GB to 10.88 GB.

    Asserted on task count and peak RSS rather than on the read tally. The tally
    stays at 1.0 either way: both consumers descend from the same source keys,
    and within one ``dask.compute`` each key is produced once whatever is
    downstream of it. That is exactly why this needed measuring -- an earlier
    draft of this test asserted the pass count, passed with the rechunk deleted,
    and would have shipped the regression it was written to catch.
    """
    m = measure_one(
        "composite_only",
        Geometry(scenes=24, blocks=4, chunk=256, threads=2, graph=GRAPH_COMPOSITE),
    )

    assert m.composite_tasks <= COMPOSITE_ONLY_TASKS * 1.4, (
        f"{m.composite_tasks:,} composite tasks against a pinned "
        f"{COMPOSITE_ONLY_TASKS:,}. The two outputs are consuming differently "
        "chunked views of the stack; restore the shared time rechunk in "
        "_composite_graph."
    )
    assert m.peak_rss_mb <= COMPOSITE_PEAK_MB * COMPOSITE_PEAK_BAND, (
        f"peak RSS {m.peak_rss_mb:.0f} MB against a pinned "
        f"{COMPOSITE_PEAK_MB:.0f} MB. The composite is holding more of the "
        "stack than it did; check that lst_p95 and qa_count still share one "
        "time-contiguous view."
    )
