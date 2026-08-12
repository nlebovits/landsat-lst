"""Integration tests for COG export (lst_p95 + 12-band monthly qa_count)."""

import calendar

import numpy as np
import pytest
import rasterio
import xarray as xr
from rio_cogeo.cogeo import cog_validate

from landsat_lst.cog import cog_export
from landsat_lst.encoding import LST_OFFSET, LST_SCALE


def _native(n: int = 64) -> xr.Dataset:
    """Synthetic native level: uint16 LST DN + uint8 (month, lat, lon) qa_count."""
    lat = np.linspace(-30.0, -35.0, n)  # descending (north-down)
    lon = np.linspace(-65.0, -60.0, n)
    # DN for 35 degC: (35 - (-50)) / 0.01 = 8500.
    lst = xr.DataArray(
        np.full((n, n), 8500, dtype=np.uint16),
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
    )
    qa = xr.DataArray(
        (np.arange(1, 13)[:, None, None] * np.ones((12, n, n))).astype(np.uint8),
        dims=["month", "latitude", "longitude"],
        coords={"month": np.arange(1, 13), "latitude": lat, "longitude": lon},
    )
    return xr.Dataset({"lst_p95": lst, "qa_count": qa})


@pytest.mark.integration
def test_cog_export_produces_valid_cogs(tmp_path):
    lst_cog = tmp_path / "lst.tif"
    qa_cog = tmp_path / "qa.tif"

    cog_export(_native(), lst_cog, qa_cog)

    assert cog_validate(str(lst_cog))[0]
    assert cog_validate(str(qa_cog))[0]


@pytest.mark.integration
def test_lst_cog_single_band_with_scale_offset(tmp_path):
    lst_cog = tmp_path / "lst.tif"
    qa_cog = tmp_path / "qa.tif"
    cog_export(_native(), lst_cog, qa_cog)

    with rasterio.open(lst_cog) as src:
        assert src.count == 1
        assert src.scales[0] == pytest.approx(LST_SCALE)
        assert src.offsets[0] == pytest.approx(LST_OFFSET)
        assert src.nodata == 0
        # DN 8500 decodes to ~35 degC.
        assert src.read(1)[0, 0] * LST_SCALE + LST_OFFSET == pytest.approx(35.0)


@pytest.mark.integration
def test_qa_cog_twelve_bands_named_by_month(tmp_path):
    lst_cog = tmp_path / "lst.tif"
    qa_cog = tmp_path / "qa.tif"
    cog_export(_native(), lst_cog, qa_cog)

    with rasterio.open(qa_cog) as src:
        assert src.count == 12
        assert list(src.descriptions) == [calendar.month_name[m] for m in range(1, 13)]
        # Band m holds value m (from the synthetic fixture).
        assert src.read(1)[0, 0] == 1
        assert src.read(12)[0, 0] == 12
