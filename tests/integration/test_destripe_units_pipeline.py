"""The bounded-unit offset pass, through the pipeline rather than beside it.

``tests/unit`` pins :func:`offsets_as_units` against :func:`offset_graph`
directly. That proves the estimator, not the wiring. These run the composite
the way ``process_tile`` runs it -- QA mask, Celsius conversion, land mask,
de-striping, pooled P95, monthly ``qa_count`` -- with the only difference being
which offset implementation ``scene_offsets`` dispatches to.

No network. The stack is synthetic and the land mask is passed in, so this is
an integration test of the code path rather than of the data source, and it can
run on every CI build instead of nightly.
"""

from __future__ import annotations

from unittest.mock import patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from landsat_lst.config import settings
from landsat_lst.pipeline import compute_annual_composite

pytestmark = pytest.mark.integration

#: Source-grid side. A multiple of the aggregation factor, as a real source
#: stack always is.
GRID = 96
#: Delivered side the composite comes back on.
OUTPUT_GRID = GRID // 3


def _dataset(*, scenes: int = 36, seed: int = 5, bias: np.ndarray | None = None):
    """Raw-DN Landsat-like Dataset, which is what the composite consumes.

    ``compute_annual_composite`` applies the QA mask and the DN-to-Celsius
    conversion itself, so handing it Celsius would skip the two steps most
    likely to interact with the offset path.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2021-01-05", periods=scenes, freq="34D").values
    doy = pd.DatetimeIndex(times).dayofyear.values.astype("float64")

    # DN scale is 0.00341802 K per count, offset 149 K.
    celsius = 25.0 + 12.0 * np.sin(2 * np.pi * (doy - 15) / 365)
    celsius = celsius[:, None, None] + rng.normal(0, 3, (1, GRID, GRID))
    if bias is not None:
        celsius = celsius + bias[:, None, None]
    dn = ((celsius + 273.15) - 149.0) / 0.00341802

    qa = np.zeros((scenes, GRID, GRID), dtype=np.uint16)
    # Cloud bit 3 on a scattered fifth, so the QA mask has real work to do.
    qa[rng.random(qa.shape) < 0.2] = 1 << 3

    return xr.Dataset(
        {
            "lwir11": (
                ["time", "latitude", "longitude"],
                da.from_array(dn.astype(np.float32), chunks=(10, 48, 48)),
            ),
            "qa_pixel": (["time", "latitude", "longitude"], da.from_array(qa, chunks=(10, 48, 48))),
        },
        coords={
            "time": times,
            "latitude": np.linspace(-33.4, -34.4, GRID),
            "longitude": np.linspace(-61.1, -60.1, GRID),
        },
    )


def _land_mask(fraction: float = 0.75) -> xr.DataArray:
    """Land over most of the grid, ocean along the eastern edge.

    On the DELIVERED grid, because ``compute_annual_composite`` aggregates
    before it applies a land mask (ADR-017). Its coordinates reproduce what the
    aggregator will label its result with when no geobox is handed in: the
    per-block means of the source coordinates.
    """
    mask = np.zeros((OUTPUT_GRID, OUTPUT_GRID), dtype=bool)
    mask[:, : int(OUTPUT_GRID * fraction)] = True
    source = {
        "latitude": np.linspace(-33.4, -34.4, GRID),
        "longitude": np.linspace(-61.1, -60.1, GRID),
    }
    factor = settings.spatial_aggregation_factor
    return xr.DataArray(
        mask,
        dims=["latitude", "longitude"],
        coords={
            dim: values.reshape(OUTPUT_GRID, factor).mean(axis=1) for dim, values in source.items()
        },
    )


def _composite(data, *, bounded: bool, land_mask=None, panel: int = 32):
    with (
        patch.object(settings, "destripe_bounded_units", bounded),
        patch.object(settings, "destripe_compute_panel", panel),
        patch.object(settings, "destripe_unit_memory_gb", 0.005),
    ):
        return compute_annual_composite(data, land_mask=land_mask)


class TestCompositeIsUnchangedByTheSplit:
    """Same composite, whichever offset implementation ran."""

    def test_p95_and_qa_count_are_identical(self):
        data = _dataset()
        graph = _composite(data, bounded=False)
        units = _composite(data, bounded=True)

        for name in ("lst_p95", "qa_count"):
            g = np.asarray(graph[name].values)
            u = np.asarray(units[name].values)
            assert g.shape == u.shape, f"{name}: shape changed"
            np.testing.assert_array_equal(
                np.isfinite(g), np.isfinite(u), err_msg=f"{name}: NaN pattern changed"
            )
            np.testing.assert_allclose(
                g[np.isfinite(g)],
                u[np.isfinite(u)],
                rtol=0,
                atol=0,
                err_msg=f"{name}: values changed",
            )

    def test_identical_under_a_land_mask(self):
        """Offsets are estimated over land only; the split must respect that."""
        data, mask = _dataset(), _land_mask()
        graph = _composite(data, bounded=False, land_mask=mask)
        units = _composite(data, bounded=True, land_mask=mask)
        np.testing.assert_array_equal(
            np.asarray(graph["lst_p95"].values),
            np.asarray(units["lst_p95"].values),
        )
        # The mask really did remove something, or the test proves nothing.
        # Masked ocean is filled with the nodata sentinel, not NaN, so compare
        # against the unmasked composite rather than testing for finiteness.
        unmasked = _composite(data, bounded=True)
        masked_vals = np.asarray(units["lst_p95"].values)
        assert not np.array_equal(masked_vals, np.asarray(unmasked["lst_p95"].values))
        ocean = masked_vals[:, int(OUTPUT_GRID * 0.75) :]
        assert len(np.unique(ocean)) == 1, "ocean should be a single fill value"

    def test_rejection_decisions_agree_on_a_biased_scene(self):
        """A scene past the cap must be discarded by both paths alike."""
        scenes = 36
        bias = np.zeros(scenes)
        bias[7] = -60.0  # far outside destripe_max_offset_c
        data = _dataset(bias=bias)

        graph = _composite(data, bounded=False)
        units = _composite(data, bounded=True)
        np.testing.assert_array_equal(
            np.asarray(graph["qa_count"].values),
            np.asarray(units["qa_count"].values),
        )
        # qa_count is built from the surviving stack, so a discarded scene has
        # to show up as fewer observations than scenes.
        assert int(np.asarray(units["qa_count"].values).max()) < scenes


class TestBoundedUnitsRunTheRealPath:
    """Guards on wiring that an equivalence check cannot see."""

    def test_the_unit_path_is_the_default(self):
        assert settings.destripe_bounded_units is True

    def test_phases_are_reported_for_a_watcher(self):
        """A stall must be attributable to a phase, and to a unit within it."""
        from landsat_lst import normalization

        seen: list[tuple[str, dict]] = []

        def record(phase, **counts):
            seen.append((phase, counts))

        with patch.object(normalization, "report_phase", record):
            _composite(_dataset(), bounded=True)

        phases = {p for p, _ in seen}
        assert "destripe_climatology" in phases
        assert "destripe_offsets" in phases

        blocks = [c for p, c in seen if p == "destripe_climatology" and c]
        scenes = [c for p, c in seen if p == "destripe_offsets" and c]
        assert blocks and blocks[-1]["blocks_done"] == blocks[-1]["blocks_total"]
        assert scenes and scenes[-1]["scenes_done"] == scenes[-1]["scenes_total"]

    def test_more_than_one_block_actually_ran(self):
        """A single-block run would pass every equivalence test vacuously."""
        from landsat_lst.normalization import _io_block_edge

        data = _dataset()
        with patch.object(settings, "destripe_unit_memory_gb", 0.005):
            edge = _io_block_edge(data["lwir11"], settings.destripe_unit_memory_gb)
        assert edge < GRID, f"block edge {edge} covers the whole {GRID} px grid"
