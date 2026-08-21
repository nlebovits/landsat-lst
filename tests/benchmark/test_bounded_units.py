"""Regression guards for the bounded-work-unit offset pass (C1, ADR-015).

C1 exists to make one property true: memory during the offset pass is bounded
by one work unit rather than by the window. The architecture it replaced held a
scene-independent ~21 GB plateau and could not build its graph at all above
2,000 scenes. Two guards here would catch a regression back toward that, and a
third pins the structure that makes the bound hold.

Bands are wide on purpose, matching the rest of this tier. A benchmark that
fails on a few percent of drift gets disabled within a month; one that fails
when peak RSS grows with the window is the one worth having.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.benchmark

#: Grid the memory guards run on. Small enough for a CI runner, large enough
#: that the block loop runs more than one block.
GRID = 512
BLOCK_BUDGET_GB = 0.02

#: Peak RSS must not grow by more than this when the window doubles. The claim
#: is that it does not grow at all; the band leaves room for allocator noise and
#: for the climatology, which is fixed in the window and so cannot cause growth.
MEMORY_BAND = 1.35

#: Phase B reads one source time-chunk per batch over the whole footprint, so
#: its graph is the same size whatever the window. Measured flat at 2,592 tasks
#: across 150 / 500 / 1,000 / 2,930 scenes at production geometry.
BATCH_TASK_BAND = 1.25


def _run(source: str) -> dict:
    """Execute a snippet in a fresh interpreter and read its last JSON line.

    Fresh because ``getrusage`` reports a high-water mark for the life of a
    process: a second configuration in the first one's interpreter inherits its
    peak and draws a flat curve whatever the truth is.
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, f"child failed:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


_MEMORY_CHILD = """
    import json, resource
    import numpy as np, pandas as pd, xarray as xr, dask, dask.array as da
    dask.config.set(scheduler="threads", num_workers=2)
    from landsat_lst.config import settings
    settings.destripe_bounded_units = True
    settings.destripe_unit_memory_gb = {budget}
    settings.destripe_compute_panel = 128
    settings.destripe_unit_workers = {workers}
    # Pinned like num_workers above: the guard measures what a unit holds,
    # not the read pool's in-flight chunk set. At the production default (32)
    # the pool's per-task overhead rides the ratio up to ~1.33 at this toy
    # geometry -- against the band, run-to-run flaky, and not the property
    # under test.
    settings.destripe_io_threads = 2
    from landsat_lst.normalization import offsets_as_units

    n = {scenes}
    times = pd.date_range("2021-01-01", periods=n, freq="11D").values
    rng = np.random.default_rng(0)
    data = rng.normal(25.0, 5.0, (n, {grid}, {grid})).astype("float32")
    data[rng.random(data.shape) < 0.3] = np.nan
    lst = xr.DataArray(
        da.from_array(data, chunks=(10, 128, 128)),
        dims=["time", "latitude", "longitude"],
        coords={{"time": times,
                "latitude": np.linspace(40, 35, {grid}),
                "longitude": np.linspace(-75, -70, {grid})}},
    )
    off, nval = offsets_as_units(lst)
    print(json.dumps({{
        "scenes": n,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "n_offsets": int(np.isfinite(np.asarray(off.values)).sum()),
        "n_valid_total": int(np.asarray(nval.values).sum()),
    }}))
"""


#: Same doubling, with the unit pool at its default width. In-flight memory is
#: workers x unit rather than 1 x unit, and a unit's bytes grow with the
#: window (the block edge only shrinks at the budget boundary), so the slope
#: is legitimately steeper than the one-worker case. The band still catches a
#: worker holding the window, which doubles the whole curve.
PARALLEL_MEMORY_BAND = 1.6


def _peak_for(scenes: int, workers: int = 1) -> dict:
    return _run(
        _MEMORY_CHILD.format(scenes=scenes, grid=GRID, budget=BLOCK_BUDGET_GB, workers=workers)
    )


def test_peak_memory_does_not_grow_with_the_window():
    """The whole point of C1: doubling the window must not double memory.

    The graph form this replaced held a plateau independent of scene count only
    because it had already materialized the stack; growth here would mean a
    work unit is holding the window rather than a block of it. Pinned at one
    unit worker so the guard isolates the per-unit property; the parallel
    envelope has its own guard below.
    """
    small, large = _peak_for(24), _peak_for(48)
    ratio = large["peak_rss_mb"] / small["peak_rss_mb"]
    assert ratio <= MEMORY_BAND, (
        f"peak RSS grew {ratio:.2f}x when the window doubled "
        f"({small['peak_rss_mb']:.0f} -> {large['peak_rss_mb']:.0f} MB, band "
        f"{MEMORY_BAND}). A work unit is holding the window, not a unit of it. "
        f"Check normalization._io_block_edge and offsets_by_scene."
    )
    # Both runs must have produced a real answer, or the ratio is meaningless.
    assert small["n_offsets"] == 24
    assert large["n_offsets"] == 48


def test_parallel_units_keep_the_memory_envelope():
    """The unit pool multiplies in-flight units, never what a unit holds.

    With eight workers, in-flight memory is eight units, and each unit's bytes
    scale with the window until the block edge shrinks at the budget boundary.
    That gives a steeper-but-bounded doubling curve. A regression where a
    *worker* holds the window (the pre-C1 failure mode, reintroduced through
    the pool) doubles the whole curve and blows the band.
    """
    small, large = _peak_for(24, workers=8), _peak_for(48, workers=8)
    ratio = large["peak_rss_mb"] / small["peak_rss_mb"]
    assert ratio <= PARALLEL_MEMORY_BAND, (
        f"peak RSS grew {ratio:.2f}x when the window doubled at 8 unit "
        f"workers ({small['peak_rss_mb']:.0f} -> {large['peak_rss_mb']:.0f} MB, "
        f"band {PARALLEL_MEMORY_BAND}). A worker is holding the window, not "
        f"one unit. Check normalization._unit_workers and _read_values."
    )
    assert small["n_offsets"] == 24
    assert large["n_offsets"] == 48


def test_phase_b_batch_graph_is_constant_in_the_window():
    """Phase B reads one source time-chunk per batch, whatever the window.

    Measured flat at 2,592 tasks across 150 / 500 / 1,000 / 2,930 scenes at
    production geometry. If ``_scene_batches`` stops aligning to source chunks,
    a batch straddles a boundary, the chunk materializes twice, and the offset
    pass silently pays roughly a quarter of an extra read of the whole stack.
    """
    source = """
        import json
        import dask, dask.array as da, numpy as np, pandas as pd, xarray as xr
        from landsat_lst.config import settings
        from landsat_lst.normalization import _scene_batches
        out = []
        for n in (60, 240, 960):
            arr = da.zeros((n, 1024, 1024), chunks=(10, 256, 256), dtype="float32")
            lst = xr.DataArray(
                arr, dims=["time", "latitude", "longitude"],
                coords={"time": pd.date_range("2021-01-01", periods=n, freq="11D").values,
                        "latitude": np.arange(1024), "longitude": np.arange(1024)})
            lo, hi = _scene_batches(lst, settings.destripe_scene_batch)[0]
            (opt,) = dask.optimize(lst.isel(time=slice(lo, hi)).data)
            out.append({"scenes": n, "batch_scenes": hi - lo,
                        "tasks": len(dict(opt.dask))})
        print(json.dumps({"rows": out}))
    """
    rows = _run(source)["rows"]
    counts = [r["tasks"] for r in rows]
    lo, hi = min(counts), max(counts)
    assert hi / lo <= BATCH_TASK_BAND, (
        f"phase B batch graph moved with the window: {counts} tasks for "
        f"{[r['scenes'] for r in rows]} scenes (band {BATCH_TASK_BAND}). "
        f"_scene_batches must group whole source time-chunks."
    )
    # Every batch is one source time-chunk, so its scene span is also constant.
    spans = {r["batch_scenes"] for r in rows}
    assert len(spans) == 1, f"batch span drifted with the window: {spans}"


def test_block_edge_shrinks_as_the_window_deepens():
    """Unit memory is held flat by shrinking the block, not by luck.

    Cheap arithmetic rather than a measurement, but it is the mechanism the
    memory guard above depends on, and a change to the budget rule would break
    this long before anyone noticed the RSS band drifting.
    """
    source = """
        import json
        import dask.array as da, numpy as np, pandas as pd, xarray as xr
        from landsat_lst.normalization import _io_block_edge
        out = {}
        for n in (50, 500, 5000):
            arr = da.zeros((n, 4096, 4096), chunks=(10, 256, 256), dtype="float32")
            lst = xr.DataArray(
                arr, dims=["time", "latitude", "longitude"],
                coords={"time": pd.date_range("2021-01-01", periods=n, freq="11D").values,
                        "latitude": np.arange(4096), "longitude": np.arange(4096)})
            out[str(n)] = _io_block_edge(lst, 4.0)
        print(json.dumps(out))
    """
    edges = _run(source)
    e50, e500, e5000 = edges["50"], edges["500"], edges["5000"]
    assert e50 >= e500 >= e5000, f"block edge did not shrink with depth: {edges}"
    for edge in (e50, e500, e5000):
        assert edge & (edge - 1) == 0, f"block edge {edge} is not a power of two"
    # Resident block stays within the stated budget at every depth.
    for scenes, edge in ((50, e50), (500, e500), (5000, e5000)):
        gb = edge * edge * scenes * 4 / 1024**3
        assert gb <= 4.0 * 1.05, f"{scenes} scenes: block is {gb:.2f} GB over a 4.0 GB budget"


def test_unit_form_matches_the_graph_form_end_to_end():
    """Equivalence, at the tier that would notice a slow drift.

    ``tests/unit`` pins this too. It is repeated here because the unit tier runs
    a 40 px grid and this one runs a multi-block geometry with real chunking,
    which is where a blocking bug would surface.
    """
    source = """
        import json
        import numpy as np, pandas as pd, xarray as xr, dask, dask.array as da
        dask.config.set(scheduler="threads", num_workers=2)
        from landsat_lst.config import settings
        from landsat_lst.normalization import offset_graph, offsets_as_units
        n = 36
        rng = np.random.default_rng(11)
        data = rng.normal(25.0, 5.0, (n, 384, 384)).astype("float32")
        data[rng.random(data.shape) < 0.25] = np.nan
        lst = xr.DataArray(
            da.from_array(data, chunks=(10, 128, 128)),
            dims=["time", "latitude", "longitude"],
            coords={"time": pd.date_range("2021-01-01", periods=n, freq="17D").values,
                    "latitude": np.linspace(40, 35, 384),
                    "longitude": np.linspace(-75, -70, 384)})
        settings.destripe_unit_memory_gb = 0.01
        settings.destripe_compute_panel = 64
        g_off, g_n = dask.compute(*offset_graph(lst))
        u_off, u_n = offsets_as_units(lst)
        g_off = np.asarray(g_off.values, dtype="float64")
        u_off = np.asarray(u_off.values, dtype="float64")
        both = np.isfinite(g_off) & np.isfinite(u_off)
        print(json.dumps({
            "max_abs_delta": float(np.abs(g_off[both] - u_off[both]).max()) if both.any() else 0.0,
            "exact": bool(np.array_equal(g_off, u_off, equal_nan=True)),
            "n_valid_equal": bool(np.array_equal(np.asarray(g_n.values), np.asarray(u_n.values))),
            "n_scored": int(both.sum()),
        }))
    """
    r = _run(source)
    assert r["n_scored"] > 0, "nothing was compared"
    assert r["exact"], f"unit form diverged, max |delta| = {r['max_abs_delta']}"
    assert r["n_valid_equal"], "valid-pixel counts diverged"
