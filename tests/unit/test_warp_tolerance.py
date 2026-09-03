"""The exact-transform switch reaches rasterio.warp.reproject, and only when on."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio.warp
from rasterio.transform import from_origin

from landsat_lst import pipeline
from landsat_lst.config import settings

pytestmark = pytest.mark.unit


def _warp_once():
    src = np.arange(16, dtype=np.uint16).reshape(4, 4)
    dst = np.zeros((4, 4), dtype=np.uint16)
    t = from_origin(500000, 4000000, 30, 30)
    rasterio.warp.reproject(
        src, dst, src_transform=t, src_crs="EPSG:32620", dst_transform=t, dst_crs="EPSG:32620"
    )
    return dst


@pytest.fixture
def capture(monkeypatch):
    """Install the wrapper around a spy, so the kwargs it forwards are visible."""
    real = pipeline._WARP_ORIGINAL or rasterio.warp.reproject
    calls: list[dict] = []

    def spy(*args, **kwargs):
        calls.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_WARP_ORIGINAL", None)
    monkeypatch.setattr(rasterio.warp, "reproject", spy)
    pipeline._install_warp_tolerance()
    yield calls
    monkeypatch.setattr(pipeline, "_WARP_ORIGINAL", None)
    monkeypatch.setattr(rasterio.warp, "reproject", real)


def test_off_forwards_no_tolerance_and_the_warp_still_runs(capture, monkeypatch):
    monkeypatch.setattr(settings, "warp_exact_transform", False)
    dst = _warp_once()
    assert "tolerance" not in capture[-1]
    assert dst[0, 0] == 0 and dst[3, 3] == 15


def test_on_supplies_tolerance_zero(capture, monkeypatch):
    monkeypatch.setattr(settings, "warp_exact_transform", True)
    _warp_once()
    assert capture[-1]["tolerance"] == 0.0


def test_an_explicit_tolerance_is_kept(capture, monkeypatch):
    monkeypatch.setattr(settings, "warp_exact_transform", True)
    src = np.zeros((2, 2), dtype=np.uint16)
    t = from_origin(0, 60, 30, 30)
    rasterio.warp.reproject(
        src,
        src.copy(),
        src_transform=t,
        src_crs="EPSG:32620",
        dst_transform=t,
        dst_crs="EPSG:32620",
        tolerance=0.5,
    )
    assert capture[-1]["tolerance"] == 0.5


def test_install_is_idempotent(capture):
    first = rasterio.warp.reproject
    pipeline._install_warp_tolerance()
    assert rasterio.warp.reproject is first
