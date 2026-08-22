"""A tile cut into shards and put back together must be the same tile.

Stage 3 is worth nothing if the seams show. Every check here is an equality
with **zero tolerance**: sharded offsets against whole-tile offsets, a
concatenation of row-band composites against the whole-tile composite, a COG
merged from per-band GeoTIFFs against one written in a single process. A
tolerance would let a real drift hide -- the failure mode is a horizontal seam
at a band boundary, which is a few hundredths of a degree over a few rows and
invisible to anything looser than ``array_equal``.

Single-process and synthetic on purpose. There is no Coiled here and no
network: what is under test is the arithmetic of the cut, not the distribution
of it, and that arithmetic should fail in CI rather than in a five-hour tile.

Modelled on ``tests/integration/test_destripe_units_pipeline.py``, which does
the same thing one layer down for the bounded-unit offset pass.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from shapely.geometry import Polygon

from landsat_lst import normalization
from landsat_lst.cog import (
    cog_export,
    finish_product,
    lst_product,
    merge_bands,
    qa_product,
    write_intermediates,
)
from landsat_lst.config import settings
from landsat_lst.encoding import encode_lst_uint16
from landsat_lst.masks import get_land_mask_for_geobox
from landsat_lst.normalization import (
    _scene_batches,
    climatology_by_blocks,
    offsets_as_units,
    offsets_by_scene,
)
from landsat_lst.offsets import (
    OffsetCache,
    OffsetKey,
    merge_scene_partials,
    partial_payload,
)
from landsat_lst.pipeline import compute_annual_composite
from landsat_lst.shards import band_edges, block_spans, partition
from landsat_lst.storage import LocalStorage
from landsat_lst.tiling import geobox_for_bbox

pytestmark = pytest.mark.integration

#: Offset-equivalence geometry. Small: the estimate is a scalar per scene, and
#: what is under test is the sharding of the reduction, not its scale.
GRID = 96

#: Composite geometry. 1,536 rows is 512 x 3, so three row bands each start on
#: a COG block row -- the alignment ``shards.band_edges`` guarantees on the
#: production grid and that ``merge_bands`` copies along.
BAND_H, BAND_W = 1536, 512
BLOCKSIZE = 512

STATISTIC_KEYS = (
    "STATISTICS_MINIMUM",
    "STATISTICS_MAXIMUM",
    "STATISTICS_MEAN",
    "STATISTICS_STDDEV",
    "STATISTICS_VALID_PERCENT",
)


def _dataset(
    *, scenes: int = 24, height: int = GRID, width: int = GRID, seed: int = 5, chunk: int = 32
):
    """Raw-DN Landsat-like Dataset, which is what the composite consumes.

    ``compute_annual_composite`` applies the QA mask and the DN-to-Celsius
    conversion itself, so handing it Celsius would skip two steps that the
    offset path interacts with.
    """
    rng = np.random.default_rng(seed)
    # Sub-second components, because real solar-day stamps have them and every
    # offset here round-trips through JSON before a coordinate join reads it
    # back. On whole seconds this whole file passed while the composite failed
    # on every shard of S30W065: the serializer truncated the axis and the join
    # could not find a single stamp.
    base = pd.date_range("2021-01-05T13:52:07", periods=scenes, freq="34D")
    times = (base + pd.to_timedelta(482_915 + 137 * np.arange(scenes), unit="us")).values
    doy = pd.DatetimeIndex(times).dayofyear.values.astype("float64")

    celsius = 25.0 + 12.0 * np.sin(2 * np.pi * (doy - 15) / 365)
    field = celsius[:, None, None] + rng.normal(0.0, 1.5, (scenes, height, width))
    # Per-scene bias, which is what de-striping exists to take back out.
    field += rng.normal(0.0, 3.0, (scenes, 1, 1))

    # DN scale 0.00341802 K per count, offset 149 K.
    dn = ((field + 273.15) - 149.0) / 0.00341802
    qa = np.full((scenes, height, width), 21824, dtype="uint16")  # clear
    qa[rng.random(qa.shape) < 0.15] = 22280  # cloud

    return xr.Dataset(
        {
            "lwir11": (
                ["time", "latitude", "longitude"],
                da.from_array(dn.astype(np.float32), chunks=(10, chunk, chunk)),
            ),
            "qa_pixel": (
                ["time", "latitude", "longitude"],
                da.from_array(qa, chunks=(10, chunk, chunk)),
            ),
        },
        coords={
            "time": times,
            "latitude": np.linspace(-33.4, -34.4, height),
            "longitude": np.linspace(-61.1, -60.1, width),
        },
    )


def _land_mask(height: int = GRID, width: int = GRID, fraction: float = 0.75) -> xr.DataArray:
    """Land over most of the grid, ocean along the eastern edge."""
    mask = np.zeros((height, width), dtype=bool)
    mask[:, : int(width * fraction)] = True
    return xr.DataArray(
        mask,
        dims=["latitude", "longitude"],
        coords={
            "latitude": np.linspace(-33.4, -34.4, height),
            "longitude": np.linspace(-61.1, -60.1, width),
        },
    )


def _celsius_stack(data: xr.Dataset, land: xr.DataArray) -> xr.DataArray:
    """The masked Celsius stack the offset estimator actually sees."""
    from landsat_lst.qa import apply_qa_mask, convert_to_celsius

    return convert_to_celsius(apply_qa_mask(data)["lwir11"]).where(land)


class TestShardedOffsets:
    """Phase A over span groups, phase B over scene ranges, then merged."""

    def test_sharded_offsets_equal_the_whole_tile_estimate(self):
        """Zero tolerance, on the offsets *and* the valid-pixel counts.

        The counts matter as much as the offsets: they drive the sparse-scene
        rejection, so a count assembled wrongly discards a different set of
        scenes and changes the composite without changing a single offset.
        """
        data = _dataset()
        land = _land_mask()
        lst = _celsius_stack(data, land)

        with (
            patch.object(settings, "destripe_unit_memory_gb", 0.005),
            patch.object(settings, "destripe_compute_panel", 32),
        ):
            whole_offset, whole_valid = offsets_as_units(lst, land_mask=land)

            block = normalization._io_block_edge(lst, settings.destripe_unit_memory_gb)
            spans = block_spans((GRID, GRID), block)
            assert len(spans) >= 3, "the split has to be a split"

            # Phase A: three disjoint span groups, as three processes would.
            ref = None
            months = None
            for group in partition(spans, 3):
                part, months = climatology_by_blocks(lst, block=block, land_mask=land, spans=group)
                if ref is None:
                    ref = np.empty_like(part)
                for y0, y1, x0, x1 in group:
                    ref[:, y0:y1, x0:x1] = part[:, y0:y1, x0:x1]

            # Phase B: two scene ranges, published as partials and merged on
            # the time coordinate rather than on the ranges in their keys.
            batches = _scene_batches(lst, settings.destripe_scene_batch)
            partials = []
            for group in partition(batches, 2):
                off, valid = offsets_by_scene(lst, ref, months, batches=group)
                # Through JSON, because that is what a shard publishes.
                partials.append(json.loads(json.dumps(partial_payload(off, valid))))

            merged_offset, merged_valid = merge_scene_partials(partials, lst.time)

        assert np.array_equal(
            np.asarray(whole_offset.values), np.asarray(merged_offset.values), equal_nan=True
        )
        assert np.array_equal(np.asarray(whole_valid.values), np.asarray(merged_valid.values))
        assert merged_offset.dtype == whole_offset.dtype

    def test_a_missing_partial_is_an_error_not_a_thinner_answer(self):
        """A preempted shard is ordinary; silently emitting NaN for it is not."""
        data = _dataset(scenes=12)
        lst = _celsius_stack(data, _land_mask())
        payload = partial_payload(
            *offsets_by_scene(
                lst,
                *climatology_by_blocks(lst, block=64),
                batches=_scene_batches(lst, settings.destripe_scene_batch)[:1],
            )
        )

        with pytest.raises(ValueError, match="no partial"):
            merge_scene_partials([payload], lst.time)

    def test_a_partial_from_another_scene_set_is_refused(self):
        """The digest should have caught it; this catches it if the digest did not."""
        lst = _celsius_stack(_dataset(scenes=12), _land_mask())
        payload = {
            "times": ["1999-01-01T00:00:00.000000000"],
            "offset": [1.0],
            "n_valid": [10],
        }

        with pytest.raises(ValueError, match="not on the planned time axis"):
            merge_scene_partials([payload], lst.time)


class TestShardedComposite:
    """Row bands, each through the full per-band path, then concatenated."""

    @staticmethod
    def _encode(composite: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(encode_lst_uint16(composite["lst_p95"]).values),
            np.asarray(composite["qa_count"].values),
        )

    def test_bands_concatenate_into_the_whole_tile_composite(self):
        """Identical uint16 and uint8 arrays -- the shipped bytes, not floats.

        Every band applies the *same* tile-wide offsets, which is the whole
        reason the concatenation works. Re-estimating per band would give each
        one its own reference climatology and leave a seam at every boundary.
        """
        data = _dataset(scenes=18, height=BAND_H, width=BAND_W, chunk=256)
        land = _land_mask(BAND_H, BAND_W)
        lst = _celsius_stack(data, land)

        with (
            patch.object(settings, "destripe_unit_memory_gb", 0.02),
            patch.object(settings, "destripe_compute_panel", 128),
        ):
            offsets = offsets_as_units(lst, land_mask=land)

            whole = compute_annual_composite(data, land_mask=land, offsets=offsets)

            bands = band_edges(BAND_H, 3, BLOCKSIZE)
            pieces = [
                compute_annual_composite(
                    data.isel(latitude=slice(start, stop)),
                    land_mask=land.isel(latitude=slice(start, stop)),
                    offsets=offsets,
                )
                for start, stop in bands
            ]

        assert [stop - start for start, stop in bands] == [512, 512, 512]

        stitched = xr.concat(pieces, dim="latitude")
        whole_lst, whole_qa = self._encode(whole)
        band_lst, band_qa = self._encode(stitched)

        np.testing.assert_array_equal(whole_lst, band_lst)
        np.testing.assert_array_equal(whole_qa, band_qa)
        assert whole_lst.dtype == np.uint16
        assert whole_qa.dtype == np.uint8

    def test_offsets_are_applied_by_time_coordinate_not_position(self):
        """A band that lost a time step must still get the right offsets.

        A spatial subset can drop time steps, and index alignment would then
        apply scene k's offset to scene k+1 from the first gap onward: a
        plausible, wrong correction that nothing downstream inspects.
        """
        data = _dataset(scenes=18)
        land = _land_mask()
        lst = _celsius_stack(data, land)

        with (
            patch.object(settings, "destripe_unit_memory_gb", 0.005),
            patch.object(settings, "destripe_compute_panel", 32),
        ):
            offset, n_valid = offsets_as_units(lst, land_mask=land)

            keep_times = [0, 1, 2, 5, 9, 14, 17]
            thinned = data.isel(time=keep_times)

            partial = compute_annual_composite(thinned, land_mask=land, offsets=(offset, n_valid))
            # The same scenes, with the offsets already restricted to them:
            # position and coordinate agree here, so the two must match.
            reference = compute_annual_composite(
                thinned,
                land_mask=land,
                offsets=(offset.isel(time=keep_times), n_valid.isel(time=keep_times)),
            )

        np.testing.assert_array_equal(
            np.asarray(partial["lst_p95"].values),
            np.asarray(reference["lst_p95"].values),
        )

    def test_a_stack_the_offsets_do_not_cover_is_refused(self):
        data = _dataset(scenes=12)
        land = _land_mask()
        lst = _celsius_stack(data, land)

        with (
            patch.object(settings, "destripe_unit_memory_gb", 0.005),
            patch.object(settings, "destripe_compute_panel", 32),
        ):
            offset, n_valid = offsets_as_units(lst, land_mask=land)
            short = (offset.isel(time=slice(0, 6)), n_valid.isel(time=slice(0, 6)))

            with pytest.raises(ValueError, match="time step the offsets do not"):
                compute_annual_composite(data, land_mask=land, offsets=short)


def _native(height: int = BAND_H, width: int = BAND_W) -> xr.Dataset:
    """Synthetic encoded native level: uint16 LST DN and 12-month uint8 counts."""
    lat = np.linspace(-30.0, -35.0, height)  # descending (north-down)
    lon = np.linspace(-65.0, -60.0, width)
    rng = np.random.default_rng(11)
    lst = xr.DataArray(
        rng.integers(7000, 9000, (height, width), dtype=np.uint16),
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
    ).chunk({"latitude": 512, "longitude": 512})
    qa = xr.DataArray(
        rng.integers(0, 40, (12, height, width), dtype=np.uint8),
        dims=["month", "latitude", "longitude"],
        coords={"month": np.arange(1, 13), "latitude": lat, "longitude": lon},
    ).chunk({"month": 12, "latitude": 512, "longitude": 512})
    native = xr.Dataset({"lst_p95": lst, "qa_count": qa})
    native.attrs.update({"tile": "S30W065", "window": "2021-2025", "scene_count": 390})
    return native


def _band_tags(path, bidx: int) -> dict:
    with rasterio.open(path) as src:
        return src.tags(bidx)


class TestMergedCog:
    """A COG assembled from bands must be the COG a single VM would have written."""

    def test_merged_cog_matches_a_single_process_export(self, tmp_path):
        """Arrays *and* every ``STATISTICS_*`` tag, on every band.

        The tags are the sharp half. Two rasters holding identical pixels can
        still carry different statistics if the merge writes a band twice or
        drops an edge row, and the Portolan validator reads the tags, not the
        pixels.
        """
        native = _native()
        bands = band_edges(BAND_H, 3, BLOCKSIZE)

        single_lst = tmp_path / "single_lst.tif"
        single_qa = tmp_path / "single_qa.tif"
        cog_export(native, single_lst, single_qa)

        # Per band: both products in ONE dask.compute, so ADR-013's single
        # native pass holds inside a shard as it does inside a whole tile.
        lst_parts, qa_parts = [], []
        for i, (start, stop) in enumerate(bands):
            piece = native.isel(latitude=slice(start, stop))
            piece.attrs.update(native.attrs)
            lst_tif = tmp_path / f"band{i}_lst.tif"
            qa_tif = tmp_path / f"band{i}_qa.tif"
            products = [lst_product(piece, lst_tif), qa_product(piece, qa_tif)]
            write_intermediates(
                [(p.da, path) for p, path in zip(products, (lst_tif, qa_tif), strict=True)]
            )
            lst_parts.append(lst_tif)
            qa_parts.append(qa_tif)

        merged_lst = merge_bands(lst_parts, tmp_path / "merged_lst_src.tif", bands)
        merged_qa = merge_bands(qa_parts, tmp_path / "merged_qa_src.tif", bands)

        out_lst = finish_product(
            merged_lst, lst_product(native, tmp_path / "out_lst.tif"), native.attrs
        )
        out_qa = finish_product(
            merged_qa, qa_product(native, tmp_path / "out_qa.tif"), native.attrs
        )

        with rasterio.open(single_lst) as a, rasterio.open(out_lst) as b:
            np.testing.assert_array_equal(a.read(1), b.read(1))
            assert a.count == b.count == 1
        with rasterio.open(single_qa) as a, rasterio.open(out_qa) as b:
            assert a.count == b.count == 12
            assert list(a.descriptions) == list(b.descriptions)
            for bidx in range(1, 13):
                np.testing.assert_array_equal(a.read(bidx), b.read(bidx), err_msg=f"band {bidx}")

        for single, merged, count in ((single_lst, out_lst, 1), (single_qa, out_qa, 12)):
            for bidx in range(1, count + 1):
                one, other = _band_tags(single, bidx), _band_tags(merged, bidx)
                for key in STATISTIC_KEYS:
                    assert one[key] == other[key], f"{merged.name} band {bidx} {key}"

    def test_a_band_that_lies_about_its_rows_is_refused(self, tmp_path):
        native = _native(height=1024)
        bands = [(0, 512), (512, 1024)]
        paths = []
        for i, (start, stop) in enumerate(bands):
            path = tmp_path / f"b{i}.tif"
            products = [lst_product(native.isel(latitude=slice(start, stop)), path)]
            write_intermediates([(products[0].da, path)])
            paths.append(path)

        with pytest.raises(ValueError, match="claims rows"):
            merge_bands(paths, tmp_path / "merged.tif", [(0, 512), (512, 2048)])


class TestBandLandMask:
    """A band's mask must be the *slice* of the tile's, not an approximation."""

    @staticmethod
    def _polygons():
        import geopandas as gpd

        # A blob straddling several band boundaries, with an edge that does not
        # fall on a pixel centre -- which is where a rebuilt transform diverges.
        poly = Polygon(
            [(-74.7, 37.3), (-71.4, 38.15), (-70.9, 41.63), (-73.2, 42.9), (-74.7, 40.1)]
        )
        return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")

    def test_a_row_slice_of_the_geobox_gives_a_slice_of_the_mask(self):
        """Exactly equal, because ``geobox[a:b, :]`` only moves the origin.

        Rasterizing against a transform rebuilt from the band's bounds instead
        divides a different span by a different pixel count, and the last-bit
        difference is enough to flip a pixel on a polygon edge. The seam
        between two bands would then carry a one-pixel land/ocean
        disagreement that nothing downstream looks for.
        """
        from landsat_lst.tiling import parse_tile_name

        polygons = self._polygons()
        geobox = geobox_for_bbox(parse_tile_name("N40W075").bbox, 60)
        tile_mask = get_land_mask_for_geobox(geobox, polygons)

        assert tile_mask.any(), "the fixture polygon has to intersect the tile"
        for start, stop in band_edges(int(geobox.shape[0]), 3, 60):
            band_mask = get_land_mask_for_geobox(geobox[start:stop, :], polygons)
            np.testing.assert_array_equal(
                band_mask, tile_mask[start:stop, :], err_msg=f"rows [{start}, {stop})"
            )

    def test_a_band_with_no_land_is_all_false(self):
        from landsat_lst.tiling import parse_tile_name

        geobox = geobox_for_bbox(parse_tile_name("N40W075").bbox, 60)
        # The southern edge of N40W075 sits below the fixture polygon entirely.
        empty = get_land_mask_for_geobox(geobox[-10:, :], self._polygons())

        assert empty.shape == (10, int(geobox.shape[1]))
        assert not empty.any()


class TestOffsetCacheDtype:
    """A warm cache must not quietly widen the whole composite."""

    @staticmethod
    def _cache(tmp_path) -> OffsetCache:
        return OffsetCache(
            storage=LocalStorage(output_dir=tmp_path),
            key=OffsetKey.build(tile="N40W075", window="2021-2025", factor=2, scene_ids=("a", "b")),
        )

    def test_the_round_trip_preserves_float32(self, tmp_path):
        """It rebuilt them as float64, and ``lst - offset`` takes the wider type.

        Whether every intermediate the P95 holds was 4 or 8 bytes wide came
        down to whether a lookup hit.
        """
        cache = self._cache(tmp_path)
        times = pd.to_datetime(["2021-07-04", "2022-07-11"]).values
        offset = xr.DataArray(
            np.array([1.5, -2.0], dtype=np.float32), dims=["time"], coords={"time": times}
        )
        n_valid = xr.DataArray(
            np.array([900, 800], dtype="int64"), dims=["time"], coords={"time": times}
        )
        cache.write(offset, n_valid, duration_s=1.0)

        hit = cache.read(offset.time)

        assert hit is not None
        assert hit[0].dtype == np.float32
        np.testing.assert_array_equal(np.asarray(hit[0].values), np.asarray(offset.values))
        assert hit[1].dtype == np.int64

    def test_a_warm_cache_composite_matches_a_cold_one(self, tmp_path):
        """Bit-for-bit, and still float32 on the way out."""
        data = _dataset(scenes=18)
        land = _land_mask()
        storage = LocalStorage(output_dir=tmp_path)
        key = OffsetKey.build(
            tile="N40W075", window="2021-2025", factor=1, scene_ids=("a", "b", "c")
        )

        with (
            patch.object(settings, "destripe_unit_memory_gb", 0.005),
            patch.object(settings, "destripe_compute_panel", 32),
        ):
            cold = compute_annual_composite(
                data, land_mask=land, offset_cache=OffsetCache(storage=storage, key=key)
            )
            warm_cache = OffsetCache(storage=storage, key=key)
            warm = compute_annual_composite(data, land_mask=land, offset_cache=warm_cache)

        assert warm_cache.last_read_hit is True, "the second run must have hit the cache"
        assert warm["lst_p95"].dtype == np.float32
        np.testing.assert_array_equal(
            np.asarray(cold["lst_p95"].values), np.asarray(warm["lst_p95"].values)
        )
        np.testing.assert_array_equal(
            np.asarray(cold["qa_count"].values), np.asarray(warm["qa_count"].values)
        )
