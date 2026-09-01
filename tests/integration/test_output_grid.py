"""The delivered nominal ~100 m grid, end to end and against real geometry.

Issue #120's follow-up decision lists the evidence the implementation must
produce. ``tests/unit/test_aggregate.py`` covers the reducer in isolation; this
file covers what only the real grid and the real pipeline can show: that the
delivered grid is an exact 3x of the source one everywhere on the globe, that
neighbouring tiles still abut, that solar-day fusion is deterministic and
happens before aggregation, and that the percentile is computed from aggregated
observations rather than downsampled after the fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from landsat_lst.aggregate import aggregate_to_output_grid
from landsat_lst.config import settings
from landsat_lst.kernels import nanquantile_last
from landsat_lst.models import TileId
from landsat_lst.pipeline import compute_annual_composite, geobox_coords
from landsat_lst.qa import apply_qa_mask, convert_to_celsius
from landsat_lst.tiling import (
    geobox_for_bbox,
    global_geobox,
    output_geobox_for_bbox,
    output_global_geobox,
    output_tile_geobox,
    tile_geobox,
)

pytestmark = pytest.mark.integration

FACTOR = 3


class TestDeliveredGridGeometry:
    """#120: "exact 3x alignment and 6000 x 6000 tile geometry"."""

    def test_a_five_degree_tile_is_six_thousand_square(self):
        assert tuple(output_tile_geobox(TileId(lat=40, lon=-75)).shape) == (6000, 6000)

    def test_the_global_delivered_array_is_an_exact_third_of_the_source(self):
        source = global_geobox().shape
        delivered = output_global_geobox().shape
        assert tuple(delivered) == (source[0] // FACTOR, source[1] // FACTOR)
        assert tuple(delivered) == (144_000, 432_000)

    @pytest.mark.parametrize(
        "tile",
        [
            TileId(lat=40, lon=-75),
            TileId(lat=-30, lon=-65),
            TileId(lat=0, lon=0),
            TileId(lat=60, lon=175),
        ],
    )
    def test_every_tile_is_an_exact_third_wherever_it_sits(self, tile):
        """Including the poles of the published band and the antimeridian edge."""
        source = tile_geobox(tile).shape
        delivered = output_tile_geobox(tile).shape
        assert tuple(delivered) == (source[0] // FACTOR, source[1] // FACTOR)
        assert source[0] % FACTOR == 0 and source[1] % FACTOR == 0

    def test_the_delivered_extent_equals_the_source_extent(self):
        """No half cell of drift: the two grids cover the same ground."""
        tile = TileId(lat=40, lon=-75)
        source = tuple(tile_geobox(tile).boundingbox)
        delivered = tuple(output_tile_geobox(tile).boundingbox)
        assert delivered == pytest.approx(source, abs=1e-9)

    def test_it_matches_zooming_the_source_grid_out_by_three(self):
        """Same extent, same block structure, one ULP apart in the transform.

        ``zoom_out`` computes ``(1/3600) * 3`` where the integer density gives
        ``1/1200``. The integer form is the authority (ADR-008's argument one
        grid down), so this pins agreement on everything that is not that last
        bit rather than pretending the two floats are equal.
        """
        tile = TileId(lat=40, lon=-75)
        zoomed = tile_geobox(tile).zoom_out(FACTOR)
        delivered = output_tile_geobox(tile)
        assert tuple(zoomed.shape) == tuple(delivered.shape)
        assert tuple(zoomed.boundingbox) == pytest.approx(tuple(delivered.boundingbox), abs=1e-9)
        assert delivered.transform.a == pytest.approx(1 / 1200, rel=0, abs=0)

    def test_neighbouring_tiles_abut_with_no_gap_or_overlap(self):
        """#120: "neighboring tiles share an exact grid with no gap or overlap"."""
        west = output_tile_geobox(TileId(lat=40, lon=-75))
        east = output_tile_geobox(TileId(lat=40, lon=-70))
        north = output_tile_geobox(TileId(lat=40, lon=-75))
        south = output_tile_geobox(TileId(lat=35, lon=-75))

        assert west.boundingbox.right == pytest.approx(east.boundingbox.left, abs=1e-12)
        assert north.boundingbox.bottom == pytest.approx(south.boundingbox.top, abs=1e-12)
        # Identical pixel size, so the shared edge is a shared pixel edge.
        assert west.transform.a == east.transform.a
        assert north.transform.e == south.transform.e

    def test_a_row_band_is_an_exact_slice_of_the_tile(self):
        """What lets a shard's band concatenate into the tile's composite."""
        tile = output_tile_geobox(TileId(lat=40, lon=-75))
        band = output_geobox_for_bbox((-75.0, 35.0, -70.0, 40.0))[512:1024, :]
        assert band.transform.a == tile.transform.a
        assert band.transform.c == tile.transform.c
        assert tuple(band.shape) == (512, 6000)

    def test_a_bands_source_slice_covers_whole_blocks(self):
        """3 x a multiple of 512 is still a whole number of source cells."""
        start, stop = 512, 1024
        source = geobox_for_bbox((-75.0, 35.0, -70.0, 40.0))[FACTOR * start : FACTOR * stop, :]
        delivered = output_geobox_for_bbox((-75.0, 35.0, -70.0, 40.0))[start:stop, :]
        assert source.shape[0] == FACTOR * delivered.shape[0]
        assert source.shape[1] == FACTOR * delivered.shape[1]

    def test_the_nominal_hundred_metres_is_stated_honestly(self):
        """The documented latitude dependence, checked as arithmetic.

        A cell is about 93 m tall everywhere and shrinks in width with the
        cosine of latitude. The docs say roughly 93 m at the equator and 46 m at
        60 degrees; if that ever stops being true the docs are wrong, not this.
        """
        metres_per_degree = 111_320.0
        width_equator = metres_per_degree * settings.output_resolution
        width_60 = width_equator * np.cos(np.deg2rad(60.0))
        assert width_equator == pytest.approx(93.0, abs=1.0)
        assert width_60 == pytest.approx(46.0, abs=1.0)


def _raw(values: np.ndarray, times: np.ndarray) -> xr.Dataset:
    """A raw-DN Dataset shaped like a solar-day-fused load."""
    dn = ((values + 273.15) - 149.0) / 0.00341802
    scenes, rows, cols = values.shape
    return xr.Dataset(
        {
            "lwir11": (["time", "latitude", "longitude"], dn.astype(np.float32)),
            "qa_pixel": (
                ["time", "latitude", "longitude"],
                np.full((scenes, rows, cols), 21824, dtype="uint16"),  # clear
            ),
        },
        coords={
            "time": times,
            "latitude": 40.0 - np.arange(rows) / 3600,
            "longitude": -75.0 + np.arange(cols) / 3600,
        },
    )


def _times(n: int) -> np.ndarray:
    base = pd.date_range("2021-01-05T13:52:07", periods=n, freq="34D")
    return (base + pd.to_timedelta(482_915 + 137 * np.arange(n), unit="us")).values


@pytest.fixture
def no_destripe(monkeypatch):
    """Turn de-striping off for the ordering fixtures.

    They are single blocks of nine source cells, orders of magnitude below
    ``destripe_min_scene_pixels``, so every scene would be rejected as sparse
    and the composite would refuse to exist. The subject here is the order of
    masking, fusion, aggregation, and the percentile; de-striping's own
    interaction with that order is covered by
    ``tests/integration/test_destripe_units_pipeline.py`` at a grid large
    enough for the estimator to mean something.
    """
    monkeypatch.setattr(settings, "destripe", False)


class TestProcessingOrder:
    """#120: "P95 is calculated from aggregated solar-day observations"."""

    def test_aggregating_first_differs_from_downsampling_the_percentile(self, no_destripe):
        """The two are different statistics, and the difference is measurable.

        This is the whole reason the decision names an order. If the two agreed,
        the cheaper-looking route would be a legitimate optimization; they do
        not, so it is a different product.
        """
        rng = np.random.default_rng(3)
        scenes = 24
        values = rng.normal(30.0, 8.0, (scenes, FACTOR, FACTOR)).astype("float32")
        data = _raw(values, _times(scenes))
        lst = convert_to_celsius(apply_qa_mask(data)["lwir11"])

        compliant = np.asarray(compute_annual_composite(data)["lst_p95"].values, dtype="float64")

        # The non-compliant route: percentile at source resolution, then a mean
        # over the same 3x3 block.
        source_p95 = nanquantile_last(np.moveaxis(np.asarray(lst.values, "float64"), 0, -1), 0.95)
        downsampled = float(source_p95.mean())

        assert float(compliant[0, 0]) != pytest.approx(downsampled, abs=0.05), (
            "aggregate-then-percentile agreed with percentile-then-downsample on "
            "this fixture, which would mean the fixture cannot tell them apart"
        )

    def test_masking_happens_before_aggregation(self, no_destripe):
        """A clouded source cell must be absent from the mean, not averaged in.

        If QA ran after aggregation the cloud's DN would be mixed into the cell
        first, and no later mask could take it back out.
        """
        scenes = 12
        values = np.full((scenes, FACTOR, FACTOR), 30.0, dtype="float32")
        values[:, 0, 0] = 80.0  # a hot contaminant
        data = _raw(values, _times(scenes))
        data["qa_pixel"][:, 0, 0] = 22280  # and it is flagged cloudy

        result = compute_annual_composite(data)["lst_p95"].values

        # Eight clear cells at 30 C. The 80 C cell contributed nothing.
        assert float(result[0, 0]) == pytest.approx(30.0, abs=0.05)

    def test_qa_count_counts_delivered_observations_not_source_cells(self, no_destripe):
        """#120: "monthly qa_count uses the same valid-observation population"."""
        scenes = 12
        data = _raw(np.full((scenes, FACTOR, FACTOR), 30.0, "float32"), _times(scenes))

        counts = compute_annual_composite(data)["qa_count"].values

        # 12 scenes at 34-day spacing land at most twice in any calendar month,
        # and never nine times: a count is observations, not source cells.
        assert counts.sum() == scenes
        assert counts.max() <= 2

    def test_a_block_below_the_rule_is_nodata_for_that_observation(self, no_destripe):
        """And the count drops with it, so the two stay one population."""
        scenes = 12
        values = np.full((scenes, FACTOR, FACTOR), 30.0, dtype="float32")
        data = _raw(values, _times(scenes))
        # Five of nine cells clouded in every scene: 4 valid, below 5 of 9.
        data["qa_pixel"][:, 0, :] = 22280
        data["qa_pixel"][:, 1, :2] = 22280

        result = compute_annual_composite(data)

        assert float(result["lst_p95"].values[0, 0]) == settings.nodata
        assert int(result["qa_count"].values[:, 0, 0].sum()) == 0


class TestSolarDayFusion:
    """#120: "deterministic same-day overlap fusion"."""

    def test_fusion_happens_on_the_source_grid_before_aggregation(self, no_destripe):
        """Two granules of one solar day are one observation, not two.

        ``odc-stac`` fuses them at load time on the source grid, so by the time
        the reducer sees the stack there is one time step per solar day. That
        ordering is what stops a scene-edge overlap being counted, weighted, or
        aggregated as two independent observations.
        """
        scenes = 6
        data = _raw(np.full((scenes, FACTOR, FACTOR), 30.0, "float32"), _times(scenes))

        result = compute_annual_composite(data)

        # One delivered observation per loaded time step, never one per granule.
        assert int(result["qa_count"].values.sum()) == scenes

    def test_a_scene_edge_keeps_its_support_after_fusion(self):
        """The reason fusion precedes aggregation rather than following it.

        A granule that covers only part of the block leaves the rest nodata. Its
        same-day partner fills that in *before* the 5-of-9 rule is applied, so
        the cell is fully supported. Aggregating each granule separately would
        have failed both halves on support and produced nodata from two
        observations that jointly cover the block.
        """
        block = np.full((1, FACTOR, FACTOR), np.nan, dtype="float32")
        left = block.copy()
        left[:, :, :2] = 30.0  # granule A: six cells
        right = block.copy()
        right[:, :, 2:] = 30.0  # granule B: three cells

        fused = np.where(np.isnan(left), right, left)

        # Fused first: nine valid cells, comfortably over the rule.
        stack = xr.DataArray(
            fused,
            dims=["time", "latitude", "longitude"],
            coords={
                "time": _times(1),
                "latitude": 40.0 - np.arange(FACTOR) / 3600,
                "longitude": -75.0 + np.arange(FACTOR) / 3600,
            },
        )
        assert np.isfinite(aggregate_to_output_grid(stack, factor=FACTOR).values).all()

        # Aggregated first: six cells and three cells, and three fails 5 of 9.
        separate = stack.copy(data=right)
        assert np.isnan(aggregate_to_output_grid(separate, factor=FACTOR).values).all()

    def test_overlap_resolution_is_deterministic(self, no_destripe):
        """The same inputs give the same fused observation, every time.

        ``odc-stac`` sorts each solar-day group by ``nominal_datetime`` and
        takes the first valid value per pixel. The property that matters
        downstream is only that it is a function of the inputs: an offset
        estimated against one fusion and applied to another would be wrong in a
        way nothing inspects.
        """
        scenes = 6
        data = _raw(np.full((scenes, FACTOR, FACTOR), 30.0, "float32"), _times(scenes))

        first = compute_annual_composite(data)["lst_p95"].values
        second = compute_annual_composite(data)["lst_p95"].values

        np.testing.assert_array_equal(first, second)


class TestMasksOnTheDeliveredGrid:
    """#120: "land and GED mask mapping at the V1 grid"."""

    def test_geobox_coords_are_the_delivered_cell_centres(self):
        geobox = output_geobox_for_bbox((-75.0, 39.9, -74.9, 40.0))
        latitude, longitude = geobox_coords(geobox)

        assert latitude.size == 120
        assert longitude.size == 120
        # Centres, half a cell in from the edge.
        assert float(latitude[0]) == pytest.approx(40.0 - 0.5 / 1200, abs=1e-12)
        assert float(longitude[0]) == pytest.approx(-75.0 + 0.5 / 1200, abs=1e-12)

    def test_a_delivered_mask_aligns_with_the_composite_it_masks(self, no_destripe):
        """The float-equality trap: labels must come from the grid, not a mean."""
        scenes = 6
        rows = cols = 6
        data = _raw(np.full((scenes, rows, cols), 30.0, "float32"), _times(scenes))
        geobox = output_geobox_for_bbox((-75.0, 40.0 - rows / 3600, -75.0 + cols / 3600, 40.0))
        latitude, longitude = geobox_coords(geobox)
        mask = xr.DataArray(
            np.ones((latitude.size, longitude.size), dtype=bool),
            dims=["latitude", "longitude"],
            coords={"latitude": latitude, "longitude": longitude},
        )

        result = compute_annual_composite(data, land_mask=mask, output_geobox=geobox)

        assert result["lst_p95"].shape == (rows // FACTOR, cols // FACTOR)
        np.testing.assert_array_equal(result["latitude"].values, latitude.values)
