"""Integration tests for COG export (lst_p95 + 12-band monthly qa_count)."""

import calendar

import dask.array as da
import numpy as np
import pytest
import rasterio
import xarray as xr
from rio_cogeo import cogeo
from rio_cogeo.cogeo import cog_validate

from landsat_lst.cog import cog_export, export_lst_cog, export_qa_cog
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


def _shared_source(reads: list, n: int = 64) -> tuple[xr.Dataset, int]:
    """A composite whose two products descend from one stack, as production's do.

    Every source block passes through ``_tally`` on its way into the graph, so
    ``len(reads)`` counts how many times a block was produced -- the number of
    passes over the scenes, in blocks. Counting executions rather than task keys
    is what makes this survive graph fusion, which renames keys freely.
    """

    def _tally(block: np.ndarray) -> np.ndarray:
        reads.append(1)  # list.append is atomic; the scheduler is threaded
        return block

    stack = da.random.default_rng(0).random((8, n, n), chunks=(4, n // 2, n // 2))
    # An explicit meta keeps dask from calling _tally once on a sample block to
    # infer the output type, which would show up as a phantom extra read.
    stack = stack.map_blocks(_tally, dtype=stack.dtype, meta=np.array((), dtype=stack.dtype))
    total = stack.sum(axis=0)
    coords = {"latitude": np.linspace(-30.0, -35.0, n), "longitude": np.linspace(-65.0, -60.0, n)}
    lst = xr.DataArray(
        (total * 1000).astype(np.uint16), dims=["latitude", "longitude"], coords=coords
    )
    qa = xr.DataArray(
        da.broadcast_to(total.astype(np.uint8), (12, n, n)).rechunk((12, n // 2, n // 2)),
        dims=["month", "latitude", "longitude"],
        coords={"month": np.arange(1, 13), **coords},
    )
    native = xr.Dataset({"lst_p95": lst, "qa_count": qa}, attrs=dict(PROVENANCE))
    return native, stack.npartitions


@pytest.mark.integration
def test_cog_export_reads_each_source_block_once(tmp_path):
    """One export, one pass over the scenes. See issue #80."""
    reads: list[int] = []
    native, blocks = _shared_source(reads)

    cog_export(native, tmp_path / "lst.tif", tmp_path / "qa.tif")

    assert len(reads) == blocks, f"{len(reads) / blocks:.1f} passes over the sources"


@pytest.mark.integration
def test_separate_exports_cost_a_pass_each(tmp_path):
    """Why :func:`cog_export` exists: exporting one product at a time doubles the reads."""
    reads: list[int] = []
    native, blocks = _shared_source(reads)

    export_lst_cog(native, tmp_path / "lst.tif")
    export_qa_cog(native, tmp_path / "qa.tif")

    assert len(reads) == 2 * blocks


@pytest.mark.integration
def test_cog_export_handles_a_half_lazy_composite(tmp_path):
    """One dask product and one numpy product in the same export.

    The numpy side writes during graph construction and contributes nothing to
    the shared compute, so the deferred-store list has a hole in it.
    """
    native = _native(128)
    native["lst_p95"] = native["lst_p95"].chunk({"latitude": 64, "longitude": 64})
    assert native["lst_p95"].chunks is not None
    assert native["qa_count"].chunks is None

    lst_cog, qa_cog = tmp_path / "lst.tif", tmp_path / "qa.tif"
    cog_export(native, lst_cog, qa_cog)

    assert cog_validate(str(lst_cog))[0]
    assert cog_validate(str(qa_cog))[0]
    with rasterio.open(lst_cog) as src:
        np.testing.assert_array_equal(src.read(1), np.full((128, 128), 8500, dtype=np.uint16))
    with rasterio.open(qa_cog) as src:
        np.testing.assert_array_equal(src.read(7), np.full((128, 128), 7, dtype=np.uint8))


# ---------------------------------------------------------------------------
# The overview cascade
# ---------------------------------------------------------------------------

#: Smallest square that forces four overview levels through the production
#: profile. ``get_maximum_overview_level`` halves until ``min(w, h)`` reaches the
#: 512 blocksize, so levels follow the *shorter* side and a cheap thin strip
#: gets no pyramid at all: 8192 -> 4096, 2048, 1024, 512.
DEEP_PYRAMID_SIZE = 8192

#: Three of the twelve months. The cascade is per band, so band count changes
#: nothing it asserts, and the full product would cost 805 MB of fixture.
DEEP_PYRAMID_BANDS = 3


def _deep_pyramid_native() -> xr.Dataset:
    """Alternating zero and ``2 * band`` columns, so every 2x2 block is mixed.

    Dask-backed and built from constant blocks so the fixture streams rather
    than materializing.

    The interleave is what gives the test its second edge. Every 2x2 window
    holds two genuine zero counts and two of the constant, so averaging *all*
    of them lands on ``band`` exactly, at every level, while averaging only the
    nonzero ones would land on ``2 * band``. A block-aligned half-and-half
    split cannot tell those apart -- an all-nodata block averages to nodata,
    which is 0, so its mean survives either way -- and an earlier draft of this
    fixture was exactly that and passed with ``nodata=0`` reinstated.
    """
    n = DEEP_PYRAMID_SIZE
    chunks = (DEEP_PYRAMID_BANDS, 1024, 1024)
    column = np.zeros(n, dtype=np.uint8)
    column[::2] = 1  # 1 where a count lands, 0 where it genuinely does not
    bands = [
        da.from_array(np.broadcast_to(column * 2 * (m + 1), (n, n)), chunks=chunks[1:])
        for m in range(DEEP_PYRAMID_BANDS)
    ]
    qa = xr.DataArray(
        da.stack(bands),
        dims=["month", "latitude", "longitude"],
        coords={
            "month": np.arange(1, DEEP_PYRAMID_BANDS + 1),
            "latitude": np.linspace(-30.0, -35.0, n),
            "longitude": np.linspace(-65.0, -60.0, n),
        },
    )
    return xr.Dataset({"qa_count": qa}, attrs=dict(PROVENANCE))


@pytest.mark.integration
def test_qa_deep_overview_levels_survive_and_average_zeros_as_data(tmp_path, monkeypatch):
    """Every level keeps its counts; none collapses to zero.

    The shipped S30W065 tile failed both halves of this: level 2 was truncated
    at row 2048 of 9000 and levels 4 through 64 were entirely zero, so a viewer
    zoomed past 1:4 saw a blank QA layer. ``cog_validate`` passed it, because
    it checks structure and not content -- which is why the assertion here is
    on the values at every level rather than on the pyramid's existence.

    The scratch raster that the pyramid is built in is forced onto the
    temp-file path, which is the branch a production 18,000-square tile always
    takes and the one whose 4 GiB ceiling the truncation ran into.
    """
    monkeypatch.setattr(cogeo, "IN_MEMORY_THRESHOLD", 1)

    qa_cog = export_qa_cog(_deep_pyramid_native(), tmp_path / "qa.tif")

    assert cog_validate(str(qa_cog))[0]
    with rasterio.open(qa_cog) as src:
        assert src.overviews(1) == [2, 4, 8, 16], "fixture must force four levels"
        for bidx in range(1, DEEP_PYRAMID_BANDS + 1):
            # Half the columns hold 2*bidx and half hold 0, so the mean is bidx
            # at every level -- but only if the zeros are averaged in as data.
            expected = float(bidx)
            native = src.read(bidx)
            assert (native != 0).mean() == pytest.approx(0.5), "fixture zeros are real"

            for factor in [1, *src.overviews(bidx)]:
                level = src.read(bidx, out_shape=(src.height // factor, src.width // factor))
                assert level.any(), f"band {bidx} level {factor} collapsed to all-zero"
                assert level.mean() == pytest.approx(expected, abs=0.01), (
                    f"band {bidx} level {factor}"
                )


@pytest.mark.integration
def test_qa_cog_declares_no_nodata_while_lst_keeps_its_fill(tmp_path):
    """The two products' nodata contracts, asserted side by side.

    ``qa_count`` must claim no absent value -- 0 observations is data -- while
    ``lst_p95`` must keep DN 0 as nodata. The shipped tile had both wrong in
    the same direction, so pinning them together is what keeps a future fix to
    one from silently reaching the other.
    """
    lst_cog, qa_cog = tmp_path / "lst.tif", tmp_path / "qa.tif"
    cog_export(_native(), lst_cog, qa_cog)

    with rasterio.open(qa_cog) as src:
        assert src.nodata is None
        assert set(src.nodatavals) == {None}
    with rasterio.open(lst_cog) as src:
        assert src.nodata == 0
