"""Unit tests for the network-free logic in the ASTER GED gap analysis.

The end-to-end path (GHSL download, CMR search, Earthdata auth, pipeline run)
is exercised manually via the documented run command; here we test the pure
helpers on synthetic data.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load scripts/aster_gap_urban_analysis.py (scripts/ is not a package).
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "aster_gap_urban_analysis.py"
_spec = importlib.util.spec_from_file_location("aster_gap_urban_analysis", _SCRIPT)
aga = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = aga
_spec.loader.exec_module(aga)

pytestmark = pytest.mark.unit


# --- classify_tiers ---------------------------------------------------------


def test_classify_tiers_by_observation_count():
    num_obs = np.array([[0, 1, 2, 3, 50]])
    lwmap = np.full_like(num_obs, aga.LWMAP_LAND)

    tiers = aga.classify_tiers(num_obs, lwmap)

    assert tiers.tolist() == [
        [
            aga.TIER_GAP,
            aga.TIER_LOW_CONFIDENCE,
            aga.TIER_LOW_CONFIDENCE,
            aga.TIER_NORMAL,
            aga.TIER_NORMAL,
        ]
    ]
    assert tiers.dtype == np.uint8


def test_classify_tiers_excludes_water():
    """Water must never be counted as a gap, however few observations it has.

    LWmap codes land as 0 and water as 1. Getting this backwards silently
    inverts the whole analysis, so pin the real codes rather than a sentinel.
    """
    num_obs = np.array([[0, 0, 0]])
    lwmap = np.array([[0, 1, -9999]])

    tiers = aga.classify_tiers(num_obs, lwmap)

    assert tiers[0, 0] == aga.TIER_GAP  # land
    assert tiers[0, 1] == aga.TIER_NODATA  # water
    assert tiers[0, 2] == aga.TIER_NODATA  # fill


def test_classify_tiers_treats_negative_fill_as_gap():
    num_obs = np.array([[-9999, 4]])
    lwmap = np.full_like(num_obs, aga.LWMAP_LAND)

    tiers = aga.classify_tiers(num_obs, lwmap)

    assert tiers.tolist() == [[aga.TIER_GAP, aga.TIER_NORMAL]]


# --- cell keys --------------------------------------------------------------


@pytest.mark.parametrize(
    ("lat", "lon", "expected"),
    [
        (0.5, 0.5, (0, 0)),
        (-11.5, 154.5, (-12, 154)),
        (-0.1, -0.1, (-1, -1)),
        (45.0, -75.0, (45, -75)),
    ],
)
def test_cell_key(lat, lon, expected):
    assert aga.cell_key(lat, lon) == expected


def test_granule_cell_from_cmr_polygon():
    """Real AG1km granule shape: the polygon, not the id, fixes the cell."""
    umm = {
        "SpatialExtent": {
            "HorizontalSpatialDomain": {
                "Geometry": {
                    "GPolygons": [
                        {
                            "Boundary": {
                                "Points": [
                                    {"Latitude": -12.0, "Longitude": 155.0},
                                    {"Latitude": -11.0, "Longitude": 155.0},
                                    {"Latitude": -11.0, "Longitude": 154.0},
                                    {"Latitude": -12.0, "Longitude": 154.0},
                                ]
                            }
                        }
                    ]
                }
            }
        }
    }

    assert aga.granule_cell(umm) == (-12, 154)


@pytest.mark.parametrize(
    "umm",
    [
        {},
        {"SpatialExtent": {}},
        {"SpatialExtent": {"HorizontalSpatialDomain": {"Geometry": {"GPolygons": []}}}},
        {
            "SpatialExtent": {
                "HorizontalSpatialDomain": {
                    "Geometry": {"GPolygons": [{"Boundary": {"Points": []}}]}
                }
            }
        },
    ],
)
def test_granule_cell_returns_none_without_usable_polygon(umm):
    assert aga.granule_cell(umm) is None


def test_cells_from_mask_maps_grid_indices_to_corners():
    mask = np.zeros((180, 360), dtype=bool)
    mask[0, 0] = True  # top-left: 89-90N, 180-179W
    mask[179, 359] = True  # bottom-right: 90-89S, 179-180E
    mask[45, 105] = True  # 44-45N, 75-74W

    assert aga.cells_from_mask(mask) == {(89, -180), (-90, 179), (44, -75)}


def test_cells_from_mask_empty():
    assert aga.cells_from_mask(np.zeros((180, 360), dtype=bool)) == set()


# --- area accounting --------------------------------------------------------


def test_tier_area_table_converts_counts_to_km2():
    counts = {
        (30, aga.TIER_GAP): 10,
        (30, aga.TIER_NORMAL): 90,
        (11, aga.TIER_LOW_CONFIDENCE): 5,
    }

    table = aga.tier_area_table(counts, pixel_km2=1.0)

    assert set(table.columns) == {"smod_class", "smod_label", "tier", "km2"}
    urban_gap = table[(table["smod_class"] == 30) & (table["tier"] == "gap")]
    assert urban_gap["km2"].item() == 10.0
    assert urban_gap["smod_label"].item() == "Urban centre"


def test_tier_area_table_scales_by_pixel_area():
    table = aga.tier_area_table({(30, aga.TIER_GAP): 4}, pixel_km2=0.25)

    assert table["km2"].item() == 1.0


def test_tier_area_table_drops_nodata_tier():
    counts = {(30, aga.TIER_NODATA): 100, (30, aga.TIER_GAP): 1}

    table = aga.tier_area_table(counts, pixel_km2=1.0)

    assert table["tier"].tolist() == ["gap"]


def test_summarize_rolls_classes_together():
    counts = {
        (30, aga.TIER_GAP): 10,
        (30, aga.TIER_NORMAL): 90,
        (21, aga.TIER_LOW_CONFIDENCE): 50,
        (21, aga.TIER_NORMAL): 50,
        (11, aga.TIER_GAP): 1000,  # rural, must not leak into the urban roll-up
    }
    table = aga.tier_area_table(counts, pixel_km2=1.0)

    urban = aga.summarize(table, aga.URBAN_CLASSES)

    assert urban["land_km2"] == 200.0
    assert urban["gap_km2"] == 10.0
    assert urban["gap_pct"] == pytest.approx(5.0)
    assert urban["low_confidence_pct"] == pytest.approx(25.0)


def test_summarize_handles_absent_class():
    table = aga.tier_area_table({(30, aga.TIER_GAP): 1}, pixel_km2=1.0)

    stats = aga.summarize(table, (22,))

    assert stats["land_km2"] == 0.0
    assert stats["gap_pct"] == 0.0


# --- granule reading --------------------------------------------------------


def test_read_granule_tiers_builds_whole_degree_transform(tmp_path):
    """Bounds come from the cell, not from pixel-centre coordinates."""
    h5py = pytest.importorskip("h5py")

    path = tmp_path / "AG1km.v003.-11.154.0010.h5"
    size = 100
    num_obs = np.full((size, size), 7, dtype=np.int16)
    num_obs[0, 0] = 0
    lwmap = np.full((size, size), aga.LWMAP_LAND, dtype=np.int16)
    # Pixel centres sit half a pixel inside the 154-155E, 12-11S cell.
    lat = np.linspace(-11.005, -11.995, size)
    lon = np.linspace(154.005, 154.995, size)

    with h5py.File(path, "w") as h5:
        h5.create_dataset(aga.H5_NUM_OBS, data=num_obs)
        h5.create_dataset(aga.H5_LWMAP, data=lwmap)
        h5.create_dataset(aga.H5_LAT, data=lat)
        h5.create_dataset(aga.H5_LON, data=lon)

    tiers, transform = aga.read_granule_tiers(path)

    assert tiers.shape == (size, size)
    assert tiers[0, 0] == aga.TIER_GAP
    assert tiers[1, 1] == aga.TIER_NORMAL
    assert transform.c == pytest.approx(154.0)
    assert transform.f == pytest.approx(-11.0)
    assert transform.a == pytest.approx(0.01)
    assert transform.e == pytest.approx(-0.01)


# --- predicted gap ----------------------------------------------------------


def _write_granule(path: Path, north: int, west: int, gap_rows: int) -> None:
    """Write a synthetic AG1km granule whose first `gap_rows` rows are gaps."""
    h5py = pytest.importorskip("h5py")

    size = 100
    num_obs = np.full((size, size), 7, dtype=np.int16)
    num_obs[:gap_rows, :] = 0
    lwmap = np.full((size, size), aga.LWMAP_LAND, dtype=np.int16)
    lat = np.linspace(north - 0.005, north - 0.995, size)
    lon = np.linspace(west + 0.005, west + 0.995, size)

    with h5py.File(path, "w") as h5:
        h5.create_dataset(aga.H5_NUM_OBS, data=num_obs)
        h5.create_dataset(aga.H5_LWMAP, data=lwmap)
        h5.create_dataset(aga.H5_LAT, data=lat)
        h5.create_dataset(aga.H5_LON, data=lon)


def test_predicted_gap_stays_inside_the_granule_footprint(tmp_path, monkeypatch):
    """Area with no granule must stay false.

    Reprojecting with a src_nodata the source never contains floods the whole
    destination with that sentinel, which ORs every pixel true and reports the
    entire tile as a gap.
    """
    monkeypatch.setattr(aga, "ASTER_DIR", tmp_path)
    # One cell of a 2x2-degree bbox, gaps in the top tenth of that cell.
    _write_granule(tmp_path / "AG1km.v003.1.0.0010.h5", north=1, west=0, gap_rows=10)

    predicted = aga._predicted_gap_for_bbox((0.0, 0.0, 2.0, 2.0), (200, 200))

    assert predicted.dtype == np.bool_
    # The granule covers 0-1E, 0-1S -> the lower-left quadrant of the bbox.
    assert not predicted[:100, :].any()  # northern half has no granule
    assert not predicted[:, 100:].any()  # eastern half has no granule
    # Gaps occupy the top tenth of that quadrant, so ~10 rows of the 100.
    assert 0 < predicted.sum() < 200 * 200 * 0.10


def test_predicted_gap_is_empty_without_granules(tmp_path, monkeypatch):
    monkeypatch.setattr(aga, "ASTER_DIR", tmp_path)

    predicted = aga._predicted_gap_for_bbox((0.0, 0.0, 2.0, 2.0), (50, 50))

    assert not predicted.any()
