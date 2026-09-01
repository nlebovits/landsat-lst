"""The tracked GED cross-tab: grid, mapping, threshold, and count conservation.

Synthetic throughout. The rasters are cut from the production global grid via
``geobox_for_bbox`` so one GED cell is exactly 36 x 36 native pixels (3,600
px/deg over 100 cells/deg), but they are 0.1 degree rather than five, because
an 18,000-squared float intermediate is 2.6 GB and this suite runs beside
others (see CLAUDE.md, Testing).

The granules are written the way the archive stores them, Geolocation arrays
included, since the reader orients against those rather than trusting file
order.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from landsat_lst import ged
from landsat_lst.ged_analysis import (
    ANALYSIS_VERSION,
    TIER_LABELS,
    AnalysisInputError,
    analyze,
    threshold_dn,
    tier_codes,
)
from landsat_lst.tiling import geobox_for_bbox

h5py = pytest.importorskip("h5py")

#: A 0.1-degree box in the NW corner of tile N40W075: 360 x 360 native pixels,
#: 10 x 10 GED cells, entirely inside granule (40, -75).
BBOX = (-75.0, 39.9, -74.9, 40.0)

#: Native pixels per GED cell edge: 3600 px/deg over 100 cells/deg.
PX_PER_CELL = 36

#: The published encoding: DN * 0.01 - 50 degC, 0 is fill.
SCALE, OFFSET, FILL = 0.01, -50.0, 0


def write_granule(ged_dir, lat_top: int, lon_west: int, numobs: np.ndarray | None = None) -> None:
    """Write one synthetic AG100 granule, north-up and west-left."""
    if numobs is None:
        numobs = np.full((100, 100), 7, dtype=np.int16)
    lat = np.linspace(lat_top, lat_top - 1, 100)[:, None] * np.ones((1, 100))
    lon = np.ones((100, 1)) * np.linspace(lon_west, lon_west + 1, 100)[None, :]
    ged_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(ged_dir / ged.granule_name(lat_top, lon_west), "w") as f:
        f["Geolocation/Latitude"] = lat
        f["Geolocation/Longitude"] = lon
        f["Observations/NumObs"] = numobs.astype(np.int16)


def write_raster(path, dn: np.ndarray, *, bbox=BBOX, tags: dict | None = None):
    """Write ``dn`` as a published-shaped LST COG on the global grid."""
    geobox = geobox_for_bbox(bbox)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=dn.shape[0],
        width=dn.shape[1],
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=Affine(*[float(v) for v in geobox.transform[:6]]),
        nodata=FILL,
    ) as dst:
        dst.write(dn.astype(np.uint16), 1)
        dst.scales = (SCALE,)
        dst.offsets = (OFFSET,)
        if tags:
            dst.update_tags(**tags)
    return path


@pytest.fixture
def scene(tmp_path):
    """A 360x360 raster over BBOX plus the granules a buffered read touches."""
    ged_dir = tmp_path / "aster_ged"
    numobs = np.full((100, 100), 7, dtype=np.int16)
    write_granule(ged_dir, 40, -75, numobs)
    for lat_top, lon_west in ((40, -76), (41, -75), (41, -76)):
        write_granule(ged_dir, lat_top, lon_west)
    return ged_dir, numobs, tmp_path


class TestTierCodes:
    def test_every_count_lands_in_exactly_one_tier(self):
        codes = tier_codes(np.array([[0, 1, 2, 3, 4, 99, ged.NUMOBS_ABSENT]], dtype=np.int16))
        assert codes.tolist() == [[0, 1, 2, 3, 4, 4, 5]]

    def test_absent_is_not_folded_into_zero(self):
        """A cell with no granule is not a cell that saw nothing."""
        codes = tier_codes(np.array([ged.NUMOBS_ABSENT, 0], dtype=np.int16))
        assert codes[0] == TIER_LABELS.index("absent")
        assert codes[1] == TIER_LABELS.index("0")

    def test_high_counts_saturate_at_the_top_tier(self):
        codes = tier_codes(np.arange(4, 200, dtype=np.int16))
        assert np.all(codes == TIER_LABELS.index(">=4"))


class TestThreshold:
    def test_seventy_celsius_is_dn_twelve_thousand(self):
        assert threshold_dn(70.0, SCALE, OFFSET) == 12000

    def test_the_boundary_dn_is_inclusive_and_its_neighbour_is_not(self):
        dn = threshold_dn(70.0, SCALE, OFFSET)
        assert dn * SCALE + OFFSET == pytest.approx(70.0)
        assert (dn - 1) * SCALE + OFFSET < 70.0

    def test_float_error_does_not_push_an_exact_value_up_one_dn(self):
        """0.07 / 0.01 is 7.000000000000001, so a bare ceiling returns 8."""
        import math

        assert math.ceil(0.07 / SCALE) == 8
        assert threshold_dn(0.07, SCALE, 0.0) == 7

    def test_a_non_representable_threshold_rounds_up_not_down(self):
        """DN 12034 is 70.34 C; 70.335 must not be satisfied by 70.33."""
        assert threshold_dn(70.335, SCALE, OFFSET) == 12034

    def test_identity_encoding(self):
        assert threshold_dn(70.0, 1.0, 0.0) == 70

    def test_a_non_positive_scale_is_refused(self):
        with pytest.raises(AnalysisInputError, match="scale must be positive"):
            threshold_dn(70.0, 0.0, OFFSET)


class TestGridAndMapping:
    def test_one_ged_cell_covers_exactly_thirty_six_pixels(self, scene):
        ged_dir, numobs, tmp_path = scene
        # The cell at the geobox's NW corner: granule row 0, column 0.
        numobs[0, 0] = 0
        write_granule(ged_dir, 40, -75, numobs)
        dn = np.full((360, 360), 13000, dtype=np.uint16)
        raster = write_raster(tmp_path / "lst.tif", dn)

        record = analyze(raster=str(raster), tile=None, ged_dir=ged_dir, buffer_cells=0)
        rows = {r["tier"]: r for r in record["by_numobs_tier"]}
        assert rows["0"]["total_pixels"] == PX_PER_CELL * PX_PER_CELL
        assert rows["0"]["hot_pixels"] == PX_PER_CELL * PX_PER_CELL

    def test_orientation_north_up_west_left(self, scene):
        """A gap in the granule's NW-most cell lands in the raster's NW corner.

        A flipped mapping would put these 36x36 hot pixels in another corner,
        which is exactly the silent failure ged.read_granule_numobs guards.
        """
        ged_dir, numobs, tmp_path = scene
        numobs[0, 0] = 0  # geobox NW corner: lat 39.99-40.0, lon -75.0
        write_granule(ged_dir, 40, -75, numobs)
        dn = np.full((360, 360), 11000, dtype=np.uint16)
        dn[:PX_PER_CELL, :PX_PER_CELL] = 13000  # hot in the NW corner only
        raster = write_raster(tmp_path / "lst.tif", dn)

        record = analyze(raster=str(raster), tile=None, ged_dir=ged_dir, buffer_cells=0)
        rows = {r["tier"]: r for r in record["by_numobs_tier"]}
        assert rows["0"]["hot_pixels"] == PX_PER_CELL * PX_PER_CELL
        assert rows[">=4"]["hot_pixels"] == 0

    @pytest.mark.parametrize("buffer_cells", [0, 1, 2])
    def test_the_mapping_is_the_production_mask(self, scene, monkeypatch, buffer_cells):
        """The analysis footprint is `gap_mask_for_geobox`'s, pixel for pixel.

        This is the point of routing both through
        `ged.cell_indices_for_geobox`: if the two ever disagreed, the table
        would describe pixels the mask does not drop.
        """
        from landsat_lst.config import settings

        ged_dir, numobs, _ = scene
        numobs[0:3, 0:3] = 0
        numobs[5, 7] = 0
        write_granule(ged_dir, 40, -75, numobs)
        monkeypatch.setattr(settings, "ged_dir", ged_dir)
        monkeypatch.setattr(settings, "ged_artifact", ged_dir / "absent.npz")

        geobox = geobox_for_bbox(BBOX)
        window, row_cells, col_cells = ged.numobs_for_geobox(
            geobox, ged_dir, pad_cells=buffer_cells
        )
        mine = ged.dilate_cells(window == 0, buffer_cells)[np.ix_(row_cells, col_cells)]
        theirs = ged.gap_mask_for_geobox(geobox, buffer_cells=buffer_cells)
        assert mine.any()
        assert np.array_equal(mine, theirs)

    def test_a_raster_off_the_tile_grid_is_refused(self, scene):
        ged_dir, _, tmp_path = scene
        dn = np.zeros((360, 360), dtype=np.uint16)
        raster = write_raster(tmp_path / "lst.tif", dn)
        with pytest.raises(AnalysisInputError, match="does not sit on tile"):
            analyze(raster=str(raster), tile="N40W075", ged_dir=ged_dir)

    def test_a_raster_without_nodata_is_refused(self, scene):
        ged_dir, _, tmp_path = scene
        path = tmp_path / "no_nodata.tif"
        geobox = geobox_for_bbox(BBOX)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=360,
            width=360,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=Affine(*[float(v) for v in geobox.transform[:6]]),
        ) as dst:
            dst.write(np.ones((360, 360), dtype=np.uint16), 1)
        with pytest.raises(AnalysisInputError, match="declares no nodata"):
            analyze(raster=str(path), tile=None, ged_dir=ged_dir)


class TestCountConservation:
    @pytest.fixture
    def mixed(self, scene):
        """A raster mixing every tier with fill, cool, and hot pixels."""
        ged_dir, numobs, tmp_path = scene
        numobs[0:10, 0:10] = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
        write_granule(ged_dir, 40, -75, numobs)
        rng = np.random.default_rng(0)
        dn = rng.choice(
            np.array([FILL, 9000, 11999, 12000, 13000], dtype=np.uint16), size=(360, 360)
        )
        raster = write_raster(tmp_path / "lst.tif", dn, tags={"scene_count": "4403"})
        return analyze(raster=str(raster), tile=None, ged_dir=ged_dir, buffer_cells=1), dn

    def test_tier_totals_sum_to_the_tile(self, mixed):
        record, dn = mixed
        rows = record["by_numobs_tier"]
        assert sum(r["total_pixels"] for r in rows) == dn.size
        assert record["tile_totals"]["total_pixels"] == dn.size

    def test_valid_and_missing_partition_every_tier(self, mixed):
        record, _ = mixed
        for row in record["by_numobs_tier"]:
            assert row["valid_pixels"] + row["missing_pixels"] == row["total_pixels"]
            assert row["hot_pixels"] <= row["valid_pixels"]

    def test_each_column_sums_to_its_tile_total(self, mixed):
        record, _ = mixed
        rows, totals = record["by_numobs_tier"], record["tile_totals"]
        for key in ("valid_pixels", "missing_pixels", "hot_pixels"):
            assert sum(r[key] for r in rows) == totals[key]

    def test_the_counts_are_the_raw_arrays_counts(self, mixed):
        record, dn = mixed
        totals = record["tile_totals"]
        assert totals["missing_pixels"] == int(np.count_nonzero(dn == FILL))
        assert totals["hot_pixels"] == int(np.count_nonzero(dn >= 12000))

    def test_the_boundary_dn_counts_as_hot_and_the_one_below_does_not(self, mixed):
        record, dn = mixed
        assert record["threshold"]["hot_threshold_dn"] == 12000
        # 11999 is present and is not counted; only 12000 and 13000 are.
        assert np.count_nonzero(dn == 11999) > 0
        assert record["tile_totals"]["hot_pixels"] == int(
            np.count_nonzero((dn == 12000) | (dn == 13000))
        )

    def test_the_unbuffered_gap_rule_removes_exactly_tier_zero(self, mixed):
        """The rule tally and the tier row are two paths to one number."""
        record, _ = mixed
        tier0 = next(r for r in record["by_numobs_tier"] if r["tier"] == "0")
        rule = next(r for r in record["mask_tradeoffs"] if r["rule"] == "numobs==0")
        assert rule["valid_pixels_removed"] == tier0["valid_pixels"]
        assert rule["hot_pixels_removed"] == tier0["hot_pixels"]
        assert rule["missing_pixels_annotated"] == tier0["missing_pixels"]

    def test_the_buffer_only_ever_removes_more(self, mixed):
        record, _ = mixed
        rules = {r["rule"]: r for r in record["mask_tradeoffs"]}
        for bare, buffered in (
            ("numobs==0", "numobs==0 + 1-cell buffer"),
            ("numobs<=2", "numobs<=2 + 1-cell buffer"),
        ):
            for key in ("valid_pixels_removed", "hot_pixels_removed"):
                assert rules[buffered][key] >= rules[bare][key]

    def test_the_low_confidence_rule_subsumes_the_gap_rule(self, mixed):
        record, _ = mixed
        rules = {r["rule"]: r for r in record["mask_tradeoffs"]}
        assert (
            rules["numobs<=2"]["valid_pixels_removed"]
            >= (rules["numobs==0"]["valid_pixels_removed"])
        )


class TestRecord:
    def test_scene_count_comes_from_the_tag(self, scene):
        ged_dir, _, tmp_path = scene
        raster = write_raster(
            tmp_path / "lst.tif",
            np.full((360, 360), 9000, dtype=np.uint16),
            tags={"scene_count": "4403"},
        )
        record = analyze(raster=str(raster), tile=None, ged_dir=ged_dir)
        assert record["raster"]["scene_count"] == 4403
        assert "scene_count" in record["raster"]["scene_count_source"]

    def test_a_missing_scene_count_is_recorded_as_unknown_with_a_reason(self, scene):
        ged_dir, _, tmp_path = scene
        raster = write_raster(tmp_path / "lst.tif", np.full((360, 360), 9000, dtype=np.uint16))
        record = analyze(raster=str(raster), tile=None, ged_dir=ged_dir)
        assert record["raster"]["scene_count"] is None
        assert record["raster"]["scene_count_source"].startswith("unknown")

    def test_an_unparseable_scene_count_is_unknown_not_a_crash(self, scene):
        ged_dir, _, tmp_path = scene
        raster = write_raster(
            tmp_path / "lst.tif",
            np.full((360, 360), 9000, dtype=np.uint16),
            tags={"scene_count": "many"},
        )
        record = analyze(raster=str(raster), tile=None, ged_dir=ged_dir)
        assert record["raster"]["scene_count"] is None
        assert "many" in record["raster"]["scene_count_source"]

    def test_the_record_states_its_inputs_and_its_limits(self, scene):
        ged_dir, _, tmp_path = scene
        raster = write_raster(tmp_path / "lst.tif", np.full((360, 360), 9000, dtype=np.uint16))
        record = analyze(raster=str(raster), tile=None, ged_dir=ged_dir, hot_threshold_c=70.0)
        assert record["analysis_version"] == ANALYSIS_VERSION
        assert "not a trace" in record["association_only"]
        raster_block = record["raster"]
        assert raster_block["source"] == str(raster)
        assert raster_block["nodata"] == 0.0
        assert raster_block["scale"] == SCALE
        assert raster_block["offset"] == OFFSET
        assert raster_block["crs"] == "EPSG:4326"
        assert len(raster_block["transform"]) == 6
        assert len(raster_block["bounds"]) == 4
        assert record["ged"]["cells_per_degree"] == 100
        assert record["threshold"]["hot_threshold_dn"] == 12000

    def test_block_rows_do_not_change_the_answer(self, scene):
        """A windowed walk must be a partition, not a sampling."""
        ged_dir, numobs, tmp_path = scene
        numobs[0:5, 0:5] = 0
        write_granule(ged_dir, 40, -75, numobs)
        rng = np.random.default_rng(1)
        dn = rng.choice(np.array([FILL, 9000, 13000], dtype=np.uint16), size=(360, 360))
        raster = write_raster(tmp_path / "lst.tif", dn)
        answers = [
            analyze(raster=str(raster), tile=None, ged_dir=ged_dir, block_rows=n)["by_numobs_tier"]
            for n in (7, 128, 360, 1024)
        ]
        assert all(a == answers[0] for a in answers)
