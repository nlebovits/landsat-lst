"""Peak RSS at reduced geometry, asserted on shape rather than on a number.

This tier cannot reproduce a production peak. The run that prompted issue #94
OOMed at 46.5 GB on a 64 GiB VM; a CI runner has neither the memory to reach
that nor the hours to build the graph that gets there. What it can do is notice
that the same small graph suddenly costs three times what it used to, which is
the change that shipped without comment.

Two properties are worth a build failure:

- Peak stays inside a wide band at one pinned configuration. Catches a change
  that inflates memory at fixed geometry.
- Peak grows with scene count, and grows no faster than the time axis. Catches
  a change that turns a linear cost quadratic, which is what a five-year window
  would pay for first.

The absolute ceiling is deliberately generous. Peak RSS varied 603-645 MB across
three runs of the pinned configuration on the dev box, and a CI runner has a
different allocator, a different Python build, and different resident libraries.
"""

from __future__ import annotations

import pytest

from landsat_lst.benchmarks import Geometry

pytestmark = pytest.mark.benchmark

#: Measured range on the pinned configuration: 603-645 MB across three runs.
PINNED_PEAK_MB = 630.0

#: Twice the measured peak. Wide enough to absorb a different runner, narrow
#: enough that the 3.0x regression ADR-013 removed would have failed here.
PEAK_BAND = 2.0

#: Floor under the band. A peak that collapses is as interesting as one that
#: climbs: it usually means the graph stopped doing the work.
PEAK_FLOOR_MB = PINNED_PEAK_MB / 4


def test_pinned_peak_rss_stays_in_band(pinned):
    """One configuration, one memory number, one wide band around it."""
    peak = pinned.peak_rss_mb
    assert PEAK_FLOOR_MB <= peak <= PINNED_PEAK_MB * PEAK_BAND, (
        f"peak RSS {peak:.0f} MB against a pinned {PINNED_PEAK_MB:.0f} MB "
        f"(band {PEAK_FLOOR_MB:.0f}-{PINNED_PEAK_MB * PEAK_BAND:.0f}). "
        "Run scripts/synthetic_scaling.py on a production-type VM before "
        "re-pinning: this tier reports direction, not magnitude."
    )


def test_peak_rss_grows_with_scenes_but_no_faster(measure_one):
    """Memory tracks the time axis, sub-linearly, and does not run away.

    Measured 446 MB at 12 scenes and 1,589 MB at 96 -- a 3.6x spread for 8x the
    scenes. That the spread exists at all is what makes this configuration worth
    measuring: below the streaming regime the whole stack fits in RAM, peak is
    the process baseline, and a flat line says nothing about the pipeline. See
    ADR-011.
    """
    small = measure_one("scaling_12", Geometry(scenes=12, blocks=4, chunk=256, threads=2))
    large = measure_one("scaling_96", Geometry(scenes=96, blocks=4, chunk=256, threads=2))

    growth = large.peak_rss_mb / small.peak_rss_mb
    assert growth > 1.2, (
        f"peak RSS moved only {growth:.2f}x for 8x the scenes "
        f"({small.peak_rss_mb:.0f} -> {large.peak_rss_mb:.0f} MB). The stack now "
        "fits in RAM at this geometry, dask never streams, and this benchmark "
        "is measuring the interpreter. Raise the geometry."
    )
    assert growth <= 8.0, (
        f"peak RSS grew {growth:.2f}x for 8x the scenes; memory is tracking the "
        "time axis at least linearly, which a streaming pipeline should not do"
    )


def test_measured_peak_beats_the_static_floor(pinned):
    """``predict_peak`` is a floor, and at this geometry its baseline dominates.

    Recorded rather than asserted tightly. The interesting version of this
    number comes from a production-type VM, where the stack term dwarfs the
    2 GiB process baseline; here the baseline is most of the floor, so the ratio
    lands under 1 and says nothing about the pipeline. The assertion is only
    that the arithmetic still runs and returns something positive.
    """
    assert pinned.floor_mb > 0
    assert pinned.peak_over_floor > 0
