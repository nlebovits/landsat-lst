"""Unit tests for the network-free logic in the urban seasonality diagnostic.

The end-to-end path (STAC + WFS) is exercised manually via the documented run
command; here we test the pure helpers on synthetic data.
"""

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_bounds
from shapely.geometry import box

# Load scripts/urban_seasonality_diagnostic.py (scripts/ is not a package).
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "urban_seasonality_diagnostic.py"
_spec = importlib.util.spec_from_file_location("urban_seasonality_diagnostic", _SCRIPT)
usd = importlib.util.module_from_spec(_spec)
# Register before exec so dataclass introspection can resolve the module.
sys.modules[_spec.name] = usd
_spec.loader.exec_module(usd)

pytestmark = pytest.mark.unit


def test_scene_day_counts():
    # 2 Jan (summer), 1 Jun (winter), 1 Sep (spring)
    months = np.array([1, 1, 6, 9])
    sd = usd.scene_day_counts(months)
    assert sd.total == 4
    assert sd.per_month[1] == 2
    assert sd.per_month[6] == 1
    assert sd.per_month[3] == 0
    assert sd.per_season["summer"] == 2
    assert sd.per_season["winter"] == 1
    assert sd.per_season["spring"] == 1
    assert sd.per_season["autumn"] == 0


def test_aggregate_monthly_counts():
    # Jan & Feb (summer) high counts; Jun (winter) low.
    months = np.array([1, 1, 2, 6])
    counts = {
        "urbana": np.array([100.0, 200.0, 150.0, 30.0]),
        "rural": np.array([1000.0, 1000.0, 1000.0, 500.0]),
    }
    monthly, summer_vs = usd.aggregate_monthly_counts(months, counts)

    by_month = {mc.month: mc.counts for mc in monthly}
    assert by_month[1]["urbana"] == pytest.approx(150.0)  # (100+200)/2
    assert by_month[6]["urbana"] == pytest.approx(30.0)
    assert set(by_month) == {1, 2, 6}  # only months present

    # summer = Jan/Feb -> urbana mean of [100,200,150]=150; non-summer = Jun=30
    assert summer_vs["summer"]["urbana"] == pytest.approx(150.0)
    assert summer_vs["non_summer"]["urbana"] == pytest.approx(30.0)


def test_compute_p95_by_class_medians_and_nan_handling():
    # 2x3 grid: row0 = urbana (code 1), row1 = rural (code 4)
    class_raster = np.array([[1, 1, 1], [4, 4, 4]], dtype="uint8")
    annual = np.array([[40.0, 42.0, np.nan], [44.0, 46.0, 48.0]])
    summer = np.array([[43.0, 45.0, 50.0], [49.0, 50.0, 51.0]])

    result = {c.clazz: c for c in usd.compute_p95_by_class(class_raster, annual, summer)}

    # urbana: annual median of [40,42] (nan dropped) = 41; summer [43,45,50]=45
    assert result["urbana"].annual_p95 == pytest.approx(41.0)
    assert result["urbana"].summer_p95 == pytest.approx(45.0)
    assert result["urbana"].delta == pytest.approx(4.0)
    assert result["urbana"].n_pixels == 3  # mask counts all 3 cells

    # rural: annual median [44,46,48]=46; summer [49,50,51]=50
    assert result["rural"].annual_p95 == pytest.approx(46.0)
    assert result["rural"].delta == pytest.approx(4.0)

    # classes with no pixels are skipped
    assert "periurbana" not in result
    assert "urbano en ruralidad" not in result


def test_rasterize_classes_codes_precedence_and_rural():
    # 10x10 grid over lon/lat [0,1]; pixel size 0.1
    out_shape = (10, 10)
    transform = from_bounds(0, 0, 1, 1, 10, 10)

    # periurbana covers [0,0.5]^2 (5x5=25 px); urbana covers [0,0.3]^2 (3x3=9 px)
    # and must WIN on the overlap. dept covers the whole grid -> rest is rural.
    urban = gpd.GeoDataFrame(
        {"clasificacion": ["periurbana", "urbana"]},
        geometry=[box(0, 0, 0.5, 0.5), box(0, 0, 0.3, 0.3)],
        crs="EPSG:4326",
    )
    dept = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")

    raster = usd.rasterize_classes(out_shape, transform, urban, dept)

    assert int((raster == usd.CLASS_CODES["urbana"]).sum()) == 9
    assert int((raster == usd.CLASS_CODES["periurbana"]).sum()) == 25 - 9
    assert int((raster == usd.CLASS_CODES["rural"]).sum()) == 100 - 25
    assert int((raster == 0).sum()) == 0  # dept covers everything


def test_rasterize_classes_outside_dept_is_zero():
    out_shape = (10, 10)
    transform = from_bounds(0, 0, 1, 1, 10, 10)
    urban = gpd.GeoDataFrame(
        {"clasificacion": ["urbana"]},
        geometry=[box(0, 0, 0.2, 0.2)],
        crs="EPSG:4326",
    )
    # department only covers the left half -> right half stays 0 (outside)
    dept = gpd.GeoDataFrame(geometry=[box(0, 0, 0.5, 1)], crs="EPSG:4326")

    raster = usd.rasterize_classes(out_shape, transform, urban, dept)

    assert int((raster == usd.CLASS_CODES["urbana"]).sum()) == 4  # 2x2
    assert int((raster == 0).sum()) == 50  # right half outside department
