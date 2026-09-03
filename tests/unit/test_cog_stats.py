"""Statistics and overview semantics of the COG writer.

These tests pin the two properties that a downstream consumer cannot repair:
the embedded ``STATISTICS_*`` band tags the Portolan validator requires
(PTL-DAT-009/010), and what the overview pyramid does to fill pixels.

Every fixture is synthetic and a few hundred kilobytes on disk -- the largest is
2048 x 2048 of highly repetitive DN, which deflate crushes to nothing.
"""

import dask
import dask.array as da
import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.enums import MaskFlags
from structlog.testing import capture_logs

from landsat_lst import cog as cog_module
from landsat_lst.cog import _coverage_summary, export_lst_cog, export_qa_cog
from landsat_lst.encoding import LST_OFFSET, LST_SCALE

pytestmark = pytest.mark.unit

# The five keys the Portolan validator requires on every band.
STATISTIC_KEYS = {
    "STATISTICS_MINIMUM",
    "STATISTICS_MAXIMUM",
    "STATISTICS_MEAN",
    "STATISTICS_STDDEV",
    "STATISTICS_VALID_PERCENT",
}

# DN for 35 C and 0 C: (degC - (-50)) / 0.01.
DN_35C = 8500
DN_0C = 5000

# Physical plausibility bounds for decoded LST, matching the pipeline clamp.
LST_MIN_C = -50.0
LST_MAX_C = 80.0


def _coords(n: int) -> dict[str, np.ndarray]:
    return {
        "latitude": np.linspace(-30.0, -35.0, n),  # descending (north-down)
        "longitude": np.linspace(-65.0, -60.0, n),
    }


def _lst_dataset(values: np.ndarray, **attrs: object) -> xr.Dataset:
    n = values.shape[-1]
    da = xr.DataArray(values, dims=["latitude", "longitude"], coords=_coords(n))
    return xr.Dataset({"lst_p95": da}, attrs=attrs)


def _qa_dataset(values: np.ndarray) -> xr.Dataset:
    n = values.shape[-1]
    da = xr.DataArray(
        values,
        dims=["month", "latitude", "longitude"],
        coords={"month": np.arange(1, 13), **_coords(n)},
    )
    return xr.Dataset({"qa_count": da})


def _striped_lst(n: int = 2048) -> np.ndarray:
    """Odd rows carry data, even rows are fill; the right half is all fill.

    Every 2x2 (and 4x4) overview window over the left half therefore mixes fill
    with real DN, which is exactly the case that would betray a resampler that
    averages nodata in. The two distinct DNs also give the band a non-zero
    standard deviation to check.
    """
    values = np.zeros((n, n), dtype=np.uint16)
    values[1::2, : n // 4] = DN_35C
    values[1::2, n // 4 : n // 2] = DN_0C
    return values


def _striped_qa(n: int = 1024, high: int = 8) -> np.ndarray:
    """Odd rows carry ``high`` observations, even rows carry a genuine zero."""
    values = np.zeros((12, n, n), dtype=np.uint8)
    values[:, 1::2, :] = high
    return values


def _band_tags(path, bidx: int) -> dict[str, str]:
    """Read band tags the way the Portolan validator does: with PAM disabled.

    Reading with PAM enabled would happily pick up a ``.aux.xml`` sidecar and
    prove nothing about what is inside the TIFF.
    """
    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(path) as src:
        return src.tags(bidx)


def _overview(path, bidx: int, level_index: int) -> np.ndarray:
    """Read one overview level directly, bypassing on-the-fly decimation."""
    with rasterio.open(path, OVERVIEW_LEVEL=level_index) as src:
        return src.read(bidx)


@pytest.fixture(scope="module")
def lst_cog(tmp_path_factory) -> object:
    path = tmp_path_factory.mktemp("lst_stats") / "lst.tif"
    dataset = _lst_dataset(
        _striped_lst(),
        tile="S30W065",
        year=2021,
        window="2021-2025",
        scene_count=390,
    )
    return export_lst_cog(dataset, path)


@pytest.fixture(scope="module")
def qa_cog(tmp_path_factory) -> object:
    path = tmp_path_factory.mktemp("qa_stats") / "qa.tif"
    return export_qa_cog(_qa_dataset(_striped_qa()), path)


# ---------------------------------------------------------------------------
# Embedded statistics
# ---------------------------------------------------------------------------


def test_lst_band_carries_all_statistics_without_pam(lst_cog):
    assert _band_tags(lst_cog, 1).keys() >= STATISTIC_KEYS


def test_qa_every_band_carries_all_statistics_without_pam(qa_cog):
    with rasterio.open(qa_cog) as src:
        band_count = src.count
    assert band_count == 12
    for bidx in range(1, band_count + 1):
        assert _band_tags(qa_cog, bidx).keys() >= STATISTIC_KEYS, f"band {bidx}"


def test_statistics_live_in_the_tiff_not_a_sidecar(lst_cog, qa_cog):
    for path in (lst_cog, qa_cog):
        assert not path.with_suffix(".tif.aux.xml").exists()


def test_statistics_are_exact_so_no_approximate_flag(lst_cog):
    assert "STATISTICS_APPROXIMATE" not in _band_tags(lst_cog, 1)


def test_lst_statistics_describe_raw_dn_excluding_fill(lst_cog):
    tags = _band_tags(lst_cog, 1)
    # Valid pixels are half the rows of the left half: a quarter of the array,
    # split evenly between DN 8500 and DN 5000.
    assert float(tags["STATISTICS_MINIMUM"]) == pytest.approx(DN_0C)
    assert float(tags["STATISTICS_MAXIMUM"]) == pytest.approx(DN_35C)
    assert float(tags["STATISTICS_MEAN"]) == pytest.approx((DN_35C + DN_0C) / 2)
    assert float(tags["STATISTICS_STDDEV"]) == pytest.approx((DN_35C - DN_0C) / 2)
    assert float(tags["STATISTICS_VALID_PERCENT"]) == pytest.approx(25.0)


def test_qa_statistics_count_zeros_as_valid(qa_cog):
    tags = _band_tags(qa_cog, 1)
    # qa_count has no nodata: 0 means "no observations", which is data.
    assert float(tags["STATISTICS_VALID_PERCENT"]) == pytest.approx(100.0)
    assert float(tags["STATISTICS_MINIMUM"]) == pytest.approx(0.0)
    assert float(tags["STATISTICS_MAXIMUM"]) == pytest.approx(8.0)
    assert float(tags["STATISTICS_MEAN"]) == pytest.approx(4.0)


def test_statistics_match_numpy_on_the_valid_pixels(lst_cog):
    values = _striped_lst()
    valid = values[values != 0].astype(np.float64)
    tags = _band_tags(lst_cog, 1)
    assert float(tags["STATISTICS_MEAN"]) == pytest.approx(valid.mean())
    assert float(tags["STATISTICS_STDDEV"]) == pytest.approx(valid.std())
    assert float(tags["STATISTICS_VALID_PERCENT"]) == pytest.approx(
        100.0 * valid.size / values.size
    )


def test_all_fill_band_reports_zero_valid_percent(tmp_path):
    path = export_lst_cog(_lst_dataset(np.zeros((64, 64), dtype=np.uint16)), tmp_path / "empty.tif")
    tags = _band_tags(path, 1)
    assert tags.keys() >= STATISTIC_KEYS
    assert float(tags["STATISTICS_VALID_PERCENT"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The nodata contract
# ---------------------------------------------------------------------------


def test_qa_declares_no_nodata_on_the_dataset_or_any_band(qa_cog):
    """0 observations is data, so ``qa_count`` must claim no absent value.

    The shipped S30W065 tile carried ``nodata=0.0`` while its own statistics
    said ``VALID_PERCENT`` 100 and ``MINIMUM`` 0 -- a header contradicting its
    own tags. It leaked in three places at once: ``qa_count`` reaches the
    writer carrying a ``nodata`` attr off the loaded stack, ``merge_bands``
    copies band 0's profile into a sharded tile, and
    ``cog_translate(nodata=None)`` declines to set a nodata rather than
    clearing one. So assert the whole surface, not just the dataset.
    """
    with rasterio.open(qa_cog) as src:
        assert src.nodata is None
        assert set(src.nodatavals) == {None}
        assert src.count == 12
        for bidx in range(1, 13):
            assert src.mask_flag_enums[bidx - 1] == [MaskFlags.all_valid], f"band {bidx}"


def test_lst_still_declares_its_fill_as_nodata(lst_cog):
    """The QA fix must not reach the LST product, whose DN 0 really is absent."""
    with rasterio.open(lst_cog) as src:
        assert src.nodata == 0
        assert src.mask_flag_enums[0] == [MaskFlags.nodata]


def test_a_nodata_attr_on_the_input_does_not_reach_the_qa_cog(tmp_path):
    """The exact leak: an attr the pipeline leaves on ``qa_count``.

    ``rio.to_raster`` reads ``rio.nodata`` off the array, so without the
    explicit strip in ``qa_product`` this attr alone reinstates the defect --
    on the intermediate, on every sharded band slab, and from there on the COG.
    """
    dataset = _qa_dataset(_striped_qa(n=64))
    dataset["qa_count"].attrs["nodata"] = 0

    path = export_qa_cog(dataset, tmp_path / "qa.tif")

    with rasterio.open(path) as src:
        assert src.nodata is None
        assert set(src.nodatavals) == {None}


def test_translate_forces_bigtiff_so_a_deep_pyramid_cannot_overrun(tmp_path):
    """A white-box guard, because the real failure costs 3.62 GiB to reproduce.

    ``cog_translate`` builds the pyramid in an *uncompressed* scratch raster.
    A production ``qa_count`` is 12 x 18,000 x 18,000 uint8 = 3.62 GiB there,
    just under the 4 GiB ceiling of a classic TIFF's 32-bit offsets, so GDAL's
    default ``IF_NEEDED`` declines to promote it -- and then the overviews
    append past the end of the addressable file. The shipped tile lost levels
    4 through 64 entirely and ``cog_validate`` still returned True.

    No affordable fixture reproduces that, so this pins the option instead.
    ``tests/integration/test_cog.py`` pins the cascade's semantics.
    """
    captured: dict[str, object] = {}
    real = cog_module.cog_translate

    def spy(src, dst, profile, **kwargs):
        captured["profile"] = dict(profile)
        return real(src, dst, profile, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cog_module, "cog_translate", spy)
        export_qa_cog(_qa_dataset(_striped_qa(n=64)), tmp_path / "qa.tif")

    assert captured["profile"]["BIGTIFF"] == "IF_SAFER"


# ---------------------------------------------------------------------------
# Overview semantics
# ---------------------------------------------------------------------------


def test_lst_overviews_are_built_on_a_multiblock_fixture(lst_cog):
    with rasterio.open(lst_cog) as src:
        assert src.overviews(1) == [2, 4]


def test_lst_overview_average_does_not_drag_values_toward_fill(lst_cog):
    """The empirical settlement: GDAL average resampling skips nodata.

    Each overview window over the left half of the fixture is half fill (DN 0)
    and half a known DN. If average counted the fill, DN 8500 (35 C) would
    collapse to DN 4250 (-7.5 C). It does not, which is why the writer keeps
    ``average`` for LST rather than falling back to ``nearest``.
    """
    with rasterio.open(lst_cog) as src:
        levels = src.overviews(1)
    for level_index in range(len(levels)):
        overview = _overview(lst_cog, 1, level_index)
        quarter = overview.shape[1] // 4
        hot = overview[:, :quarter] * LST_SCALE + LST_OFFSET
        mild = overview[:, quarter : 2 * quarter] * LST_SCALE + LST_OFFSET
        assert np.all(hot == pytest.approx(35.0)), f"level {levels[level_index]}"
        assert np.all(mild == pytest.approx(0.0)), f"level {levels[level_index]}"


def test_lst_overview_keeps_all_fill_regions_as_fill(lst_cog):
    for level_index in (0, 1):
        overview = _overview(lst_cog, 1, level_index)
        half = overview.shape[1] // 2
        assert np.all(overview[:, half:] == 0)


def test_lst_overview_values_stay_physically_plausible(lst_cog):
    with rasterio.open(lst_cog) as src:
        levels = src.overviews(1)
    for level_index in range(len(levels)):
        overview = _overview(lst_cog, 1, level_index)
        decoded = overview[overview != 0] * LST_SCALE + LST_OFFSET
        assert decoded.min() >= LST_MIN_C
        assert decoded.max() <= LST_MAX_C


def test_qa_overview_averages_genuine_zeros(qa_cog):
    """For observation counts, averaging zeros is the correct semantic."""
    overview = _overview(qa_cog, 1, 0)
    assert np.all(overview == 4)  # mean of a 2x2 holding {0, 0, 8, 8}


# ---------------------------------------------------------------------------
# Block layout
# ---------------------------------------------------------------------------


def test_blocks_are_512_square(lst_cog, qa_cog):
    for path in (lst_cog, qa_cog):
        with rasterio.open(path) as src:
            assert set(src.block_shapes) == {(512, 512)}


def test_tiny_fixture_keeps_the_512_blocksize_and_gains_no_overviews(tmp_path):
    """GDAL reports the requested blocksize even when the image is smaller."""
    path = export_lst_cog(
        _lst_dataset(np.full((64, 64), DN_35C, dtype=np.uint16)), tmp_path / "tiny.tif"
    )
    with rasterio.open(path) as src:
        assert src.block_shapes == [(512, 512)]
        # One block covers the whole image, so there is nothing to decimate.
        assert src.overviews(1) == []


# ---------------------------------------------------------------------------
# Streaming write path
# ---------------------------------------------------------------------------


def test_dask_backed_input_streams_and_round_trips(tmp_path):
    """A chunked input must survive the ``dask.array.store`` path intact.

    Chunks are deliberately not aligned to the 512-pixel blocking, which is the
    layout the writer will actually see from a composite.
    """
    values = np.arange(1024 * 1024, dtype=np.uint32).reshape(1024, 1024)
    values = (values % 60000 + 1).astype(np.uint16)  # never 0, so nothing is fill
    dataset = _lst_dataset(values).chunk({"latitude": 300, "longitude": 300})
    assert dataset["lst_p95"].chunks is not None

    path = export_lst_cog(dataset, tmp_path / "chunked.tif")

    with rasterio.open(path) as src:
        np.testing.assert_array_equal(src.read(1), values)
    tags = _band_tags(path, 1)
    assert float(tags["STATISTICS_MEAN"]) == pytest.approx(values.astype(np.float64).mean())
    assert float(tags["STATISTICS_VALID_PERCENT"]) == pytest.approx(100.0)


def test_dask_backed_qa_input_streams_all_twelve_bands(tmp_path):
    values = _striped_qa(n=512, high=6)
    dataset = _qa_dataset(values).chunk({"month": 3, "latitude": 200, "longitude": 200})
    path = export_qa_cog(dataset, tmp_path / "chunked_qa.tif")

    with rasterio.open(path) as src:
        assert src.count == 12
        for bidx in range(1, 13):
            np.testing.assert_array_equal(src.read(bidx), values[bidx - 1])


def test_bounded_writer_computes_only_each_longitude_group(tmp_path, monkeypatch):
    """Each compute reaches only its own source chunks and writes one full TIFF."""
    values = np.arange(20, dtype=np.uint16).reshape(4, 5) + 1
    observed_blocks: list[int] = []

    def observe(block, block_info=None):
        observed_blocks.append(block_info[None]["chunk-location"][-1])
        return block

    base = da.from_array(values, chunks=(4, 2), name="bounded-source")
    source = base.map_blocks(observe, dtype=base.dtype, meta=np.array((), dtype=base.dtype))
    array = xr.DataArray(
        source,
        dims=["latitude", "longitude"],
        coords={
            "latitude": np.linspace(-30.0, -35.0, 4),
            "longitude": np.linspace(-65.0, -60.0, 5),
        },
    )
    array = cog_module._prep(array).rio.write_nodata(0)
    executed: list[set[int]] = []
    real_compute = dask.compute

    def recording_compute(*collections):
        before = len(observed_blocks)
        result = real_compute(*collections)
        executed.append(set(observed_blocks[before:]))
        return result

    monkeypatch.setattr(cog_module.dask, "compute", recording_compute)
    path = tmp_path / "bounded.tif"

    cog_module.write_intermediates_bounded([(array, path)], longitude_group=4)

    assert len(executed) == 2
    assert executed == [{0, 1}, {2}]
    with rasterio.open(path) as src:
        assert (src.height, src.width, src.count) == (4, 5, 1)
        np.testing.assert_array_equal(src.read(1), values)


# ---------------------------------------------------------------------------
# Coverage diagnostic (issue #80)
# ---------------------------------------------------------------------------


def _coverage_reference(values: np.ndarray) -> dict[str, float]:
    """What the retired eager reduction in ``process_tile`` used to report."""
    obs = values.sum(axis=0, dtype=np.int64)
    return {
        "min": int(obs.min()),
        "median": float(np.median(obs)),
        "max": int(obs.max()),
        "zero_frac": round(float((obs == 0).mean()), 3),
    }


@pytest.mark.parametrize("n", [64, 65])  # even and odd pixel counts per band
def test_coverage_summary_matches_the_reduction_it_replaced(n):
    """The histogram must reproduce numpy exactly, median included.

    An odd pixel count exercises the single middle order statistic; an even one
    exercises the mean of the two straddling it, which is where a cumulative
    lookup is easiest to get wrong.
    """
    rng = np.random.default_rng(7)
    values = rng.integers(0, 9, size=(12, n, n), dtype=np.uint8)
    values[:, 0, :] = 0  # a row of pixels with no observations at all

    histogram = np.bincount(
        values.sum(axis=0, dtype=np.int64).ravel(), minlength=255 * 12 + 1
    ).astype(np.int64)

    assert _coverage_summary(histogram) == _coverage_reference(values)


def test_coverage_summary_of_an_empty_raster_is_all_zero():
    assert _coverage_summary(np.zeros(3061, dtype=np.int64)) == {
        "min": 0,
        "median": 0.0,
        "max": 0,
        "zero_frac": 0.0,
    }


def test_qa_export_logs_coverage_without_a_second_pass(tmp_path):
    """Exporting the QA COG is where the coverage line comes from now."""
    values = _striped_qa(n=256, high=5)
    with capture_logs() as logs:
        export_qa_cog(_qa_dataset(values), tmp_path / "qa.tif")

    coverage = [entry for entry in logs if entry["event"] == "valid_coverage_obs_per_pixel"]
    assert len(coverage) == 1
    reported = {key: coverage[0][key] for key in ("min", "median", "max", "zero_frac")}
    assert reported == _coverage_reference(values)


def test_lst_export_logs_no_coverage(tmp_path):
    """Only the observation counts carry coverage; DN sums would be meaningless."""
    with capture_logs() as logs:
        export_lst_cog(_lst_dataset(_striped_lst(n=256)), tmp_path / "lst.tif")

    assert not [e for e in logs if e["event"] == "valid_coverage_obs_per_pixel"]
