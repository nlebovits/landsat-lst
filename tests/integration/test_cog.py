"""Integration tests for COG export (lst_p95 + 12-band monthly qa_count)."""

import calendar

import numpy as np
import pytest
import rasterio
import xarray as xr
from rio_cogeo.cogeo import cog_validate

from landsat_lst.cog import cog_export
from landsat_lst.encoding import LST_OFFSET, LST_SCALE

# The five keys the Portolan validator requires on every band (PTL-DAT-009/010).
STATISTIC_KEYS = {
    "STATISTICS_MINIMUM",
    "STATISTICS_MAXIMUM",
    "STATISTICS_MEAN",
    "STATISTICS_STDDEV",
    "STATISTICS_VALID_PERCENT",
}

# Attrs that pipeline.process_tile stamps onto the composite.
PROVENANCE = {"tile": "S30W065", "year": 2021, "window": "2021-2025", "scene_count": 390}


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
    return xr.Dataset({"lst_p95": lst, "qa_count": qa}, attrs=dict(PROVENANCE))


def _tags(path, bidx=None) -> dict[str, str]:
    """Read tags with PAM disabled, the way the Portolan validator does."""
    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(path) as src:
        return src.tags() if bidx is None else src.tags(bidx)


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


@pytest.mark.integration
def test_both_cogs_embed_statistics_on_every_band(tmp_path):
    lst_cog = tmp_path / "lst.tif"
    qa_cog = tmp_path / "qa.tif"
    cog_export(_native(), lst_cog, qa_cog)

    assert _tags(lst_cog, 1).keys() >= STATISTIC_KEYS
    for bidx in range(1, 13):
        assert _tags(qa_cog, bidx).keys() >= STATISTIC_KEYS, f"qa band {bidx}"
    # No PAM sidecar: the statistics must live in the TIFF itself.
    assert not lst_cog.with_suffix(".tif.aux.xml").exists()
    assert not qa_cog.with_suffix(".tif.aux.xml").exists()


@pytest.mark.integration
def test_qa_statistics_match_the_per_band_constant(tmp_path):
    lst_cog = tmp_path / "lst.tif"
    qa_cog = tmp_path / "qa.tif"
    cog_export(_native(), lst_cog, qa_cog)

    # Band m is uniformly m, so its mean is m and its spread is zero.
    for bidx in range(1, 13):
        tags = _tags(qa_cog, bidx)
        assert float(tags["STATISTICS_MEAN"]) == pytest.approx(bidx)
        assert float(tags["STATISTICS_STDDEV"]) == pytest.approx(0.0)
        assert float(tags["STATISTICS_VALID_PERCENT"]) == pytest.approx(100.0)


@pytest.mark.integration
def test_band_descriptions_survive_alongside_forwarded_band_tags(tmp_path):
    """``forward_band_tags=True`` must not cost us the month labels."""
    lst_cog = tmp_path / "lst.tif"
    qa_cog = tmp_path / "qa.tif"
    cog_export(_native(), lst_cog, qa_cog)

    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(qa_cog) as src:
        assert list(src.descriptions) == [calendar.month_name[m] for m in range(1, 13)]
        assert src.tags(1).keys() >= STATISTIC_KEYS


@pytest.mark.integration
def test_composite_provenance_lands_in_dataset_tags(tmp_path):
    """pipeline.process_tile stamps tile/year/window/scene_count; both COGs keep them."""
    lst_cog = tmp_path / "lst.tif"
    qa_cog = tmp_path / "qa.tif"
    cog_export(_native(), lst_cog, qa_cog)

    for path in (lst_cog, qa_cog):
        tags = _tags(path)
        for key, value in PROVENANCE.items():
            assert tags[key] == str(value), f"{path.name}:{key}"


@pytest.mark.integration
def test_cog_export_streams_a_dask_backed_composite(tmp_path):
    """The production input is chunked; the streamed write must be lossless."""
    native = _native(256).chunk({"latitude": 100, "longitude": 100, "month": 4})
    assert native["lst_p95"].chunks is not None
    lst_cog = tmp_path / "lst.tif"
    qa_cog = tmp_path / "qa.tif"

    cog_export(native, lst_cog, qa_cog)

    assert cog_validate(str(lst_cog))[0]
    assert cog_validate(str(qa_cog))[0]
    with rasterio.open(lst_cog) as src:
        assert set(src.block_shapes) == {(512, 512)}
        np.testing.assert_array_equal(src.read(1), np.full((256, 256), 8500, dtype=np.uint16))
    with rasterio.open(qa_cog) as src:
        assert src.count == 12
        for bidx in range(1, 13):
            np.testing.assert_array_equal(src.read(bidx), np.full((256, 256), bidx, dtype=np.uint8))
