"""Cloud-Optimized GeoTIFF (COG) export for LST composites.

Derives QGIS-ready COGs from a native-resolution composite level:

- ``lst_p95``  -> single-band uint16 COG. GDAL band scale/offset are embedded so
  viewers (QGIS, ``gdalinfo``) auto-decode DN to Celsius: ``degC = DN*0.01 - 50``.
  Fill value 0 is written as nodata.
- ``qa_count`` -> 12-band uint8 COG, one band per calendar month (Jan..Dec), band
  descriptions set accordingly. **No nodata** is set: a value of 0 is meaningful
  (no valid observations that month) and must stay visible for gap diagnosis.

The input is the encoded native-resolution dataset (``uint16`` LST DN + ``uint8``
per-month counts), as produced by the pipeline before export.

Four properties of the writer are load-bearing and easy to regress:

**Both intermediates are written by one shared compute.** ``lst_p95`` and
``qa_count`` descend from the same de-biased stack, so writing them one after
the other reads every scene twice: dask holds nothing between two ``compute``
calls. :func:`cog_export` therefore defers both writes (``compute=False``) and
hands them to a single :func:`dask.compute`, which retires the shared source
blocks once. Measured on a synthetic tile, that is 1.0x one pass over the
sources where sequential writes cost 2.0x, and the bytes written are identical.
Exporting through :func:`export_lst_cog` and :func:`export_qa_cog` separately
still costs two passes, which is why the pipeline calls :func:`cog_export`.
See issue #80.

**The intermediate GeoTIFF is streamed, never materialized.** ``rio.to_raster``
only routes a dask array through ``dask.array.store`` when ``lock`` is truthy
(``rioxarray/raster_writer.py:315``); with the default ``lock=None`` it calls
``.values`` and pulls the whole array into RAM — about 3.9 GB for a production
12-band QA tile. ``_write_intermediate`` therefore always passes a real lock,
and compresses the intermediate so a worker's scratch disk survives too.

**Exact per-band statistics are embedded in the TIFF itself.** The Portolan
validator (PTL-DAT-009/010) requires the five ``STATISTICS_*`` keys and reads
them with PAM disabled, so a ``.aux.xml`` sidecar does not count. Every band is
accumulated in a single windowed walk of the intermediate -- one walk, not one
per band -- and the same walk sums the twelve monthly counts per pixel into the
histogram behind ``valid_coverage_obs_per_pixel``. That log line used to be an
eager ``.values`` on ``qa_count``, a third full pass over the native stack for
four numbers; reading them off the written raster costs a local file scan the
statistics already pay for. They are written as band tags, which
``cog_translate`` forwards **only** when ``forward_band_tags=True``
(default ``False`` silently drops them; ``rio_cogeo/cogeo.py:397``). Per GDAL
convention the statistics describe raw DN, not decoded Celsius, and they are
exact — hence no ``STATISTICS_APPROXIMATE`` tag.

**Overviews use ``average`` on both products, including LST.** That is not an
oversight about nodata: it was measured. A half-fill/half-constant fixture
translated through ``cog_translate(nodata=0, overview_resampling="average")``
decodes to exactly the constant temperature at every overview level, and
all-fill regions stay at DN 0 — GDAL's average resampling skips nodata rather
than averaging it in. ``nearest`` is actively worse here, since it can latch
onto the fill pixel and report -50 C. ``tests/unit/test_cog_stats.py`` pins this
behaviour, so a GDAL upgrade that changed it would fail the suite rather than
quietly cool the pyramid. For ``qa_count`` averaging is the plain correct
semantic anyway: zeros are genuine observation counts.

**Every translate forces ``BIGTIFF=IF_SAFER``, and ``qa_count`` is why.**
:func:`cog_translate` builds the pyramid in a *temporary* raster and then copies
it out, and that scratch file is **uncompressed** however the output is
compressed (``rio_cogeo/cogeo.py:322`` pops ``compress`` unless
``allow_intermediate_compression``). A production ``qa_count`` tile is
12 x 18,000 x 18,000 uint8 = 3.62 GiB, which sits just *under* the 4 GiB ceiling
of a classic TIFF's 32-bit offsets — so GDAL's default ``BIGTIFF=IF_NEEDED``
declines to promote it, and then ``build_overviews`` appends a 972 MB level 2 on
top and runs off the end of the addressable file. The shipped S30W065 tile lost
its pyramid exactly there: level 2 truncated at row 2048 of 9000, levels 4
through 64 never written at all, so anything zoomed past 1:4 rendered a blank QA
layer. ``cog_validate`` returns **True** on that file — it checks structure, not
content — which is how the damage shipped. Reproduced and fixed at full 18,000²
scale; ``tests/integration/test_cog.py`` pins a smaller multi-level case.

``IF_SAFER`` rather than ``YES`` so the decision stays GDAL's and tracks the
actual product: an LST tile (one uint16 band, 648 MB) still writes a classic
TIFF, byte-for-byte what it wrote before. Do not "simplify" this to a bare
translate — the failure is silent, size-dependent, and validates clean.

**Nodata is applied to the raster explicitly, never inherited.** A
:class:`Product` declaring ``nodata=None`` is not sufficient on its own, and
``qa_count`` proved it twice over. It reaches the writer carrying a ``nodata``
attr off the loaded stack, so ``rio.to_raster`` stamped 0.0 onto the
intermediate; :func:`merge_bands` copies band 0's profile verbatim, so a sharded
tile inherited it too; and ``cog_translate(nodata=None)`` does not *clear* a
source's nodata, it merely declines to set one. The shipped tile therefore
carried a header saying 0 is absent data while its own ``STATISTICS_*`` tags
(``VALID_PERCENT`` 100, ``MINIMUM`` 0) said 0 is data. Zero observations **is**
data. So :func:`qa_product` strips it from the array and :func:`finish_product`
assigns ``src.nodata = product.nodata`` on the open raster — the one point both
the whole-tile and the merge path pass through.
"""

from __future__ import annotations

import calendar
import shutil
import tempfile
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask
import dask.array as dask_array
import numpy as np
import rasterio
import rioxarray  # noqa: F401 - needed for .rio accessor
import structlog
from rasterio.windows import Window
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles
from rioxarray.raster_writer import RasterioWriter

from landsat_lst.encoding import LST_FILL_VALUE, LST_OFFSET, LST_SCALE

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    import xarray as xr

log = structlog.get_logger()

# Month names for qa_count band descriptions (index 1..12).
_MONTH_NAMES = tuple(calendar.month_name[m] for m in range(1, 13))

# Matches the blocking of ``cog_profiles.get("deflate")``, so the intermediate's
# windows line up with the output's tiles.
_BLOCKSIZE = 512

# Empirically verified to honour nodata for both products; see module docstring.
_OVERVIEW_RESAMPLING = "average"

# The pyramid is built in an *uncompressed* scratch raster, so a 12-band uint8
# tile crosses the 4 GiB classic-TIFF ceiling there and nowhere else. Without
# this the deep overview levels are silently lost; see the module docstring.
_BIGTIFF = "IF_SAFER"

_STATISTIC_KEYS = (
    "STATISTICS_MINIMUM",
    "STATISTICS_MAXIMUM",
    "STATISTICS_MEAN",
    "STATISTICS_STDDEV",
    "STATISTICS_VALID_PERCENT",
)


def _prep(da: xr.DataArray) -> xr.DataArray:
    """Attach CRS + spatial dims so rioxarray can write a georeferenced raster."""
    return da.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")


@contextmanager
def _scratch_tif(prefix: str) -> Iterator[Path]:
    """Yield a path for the intermediate GeoTIFF, removing its directory after."""
    scratch = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield scratch / "src.tif"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _write_intermediate(da: xr.DataArray, src_tif: Path, *, compute: bool = True) -> Any:
    """Write the pre-COG GeoTIFF, streaming block by block for dask inputs.

    The lock is what selects the streaming path: without it rioxarray materializes
    the full array in memory before writing anything.

    Args:
        da: Array to write. May be dask-backed or already in memory.
        src_tif: Destination for the intermediate GeoTIFF.
        compute: When ``False`` and ``da`` is dask-backed, write the header and
            return the deferred store instead of running it, so several
            intermediates can share one :func:`dask.compute`. A numpy-backed
            array ignores this and writes eagerly, returning ``None``.

    Returns:
        The deferred store when one was created, else ``None``.
    """
    return da.rio.to_raster(
        src_tif,
        lock=threading.Lock(),
        tiled=True,
        blockxsize=_BLOCKSIZE,
        blockysize=_BLOCKSIZE,
        compress="deflate",
        compute=compute,
    )


def write_intermediates(pairs: Sequence[tuple[xr.DataArray, Path]]) -> None:
    """Write every ``(array, path)`` pair, sharing one pass over their sources.

    The arrays here descend from a common stack, so their graphs share source
    keys; handing the deferred stores to one :func:`dask.compute` retires each
    of those keys once rather than once per output. Numpy-backed arrays have
    already been written by :func:`_write_intermediate` and contribute nothing
    to the compute.
    """
    deferred = [_write_intermediate(da, path, compute=False) for da, path in pairs]
    pending = [d for d in deferred if d is not None]
    if pending:
        dask.compute(*pending)


def write_intermediates_bounded(
    pairs: Sequence[tuple[xr.DataArray, Path]], *, longitude_group: int
) -> None:
    """Write full-shape rasters through sequential longitude-slice computes.

    Each destination header is created once from the full encoded array. Every
    compute then sees only one lazy longitude slice from that already-built
    graph, so Dask culls source reads outside the group and can release the
    group intermediate results before the next one starts. Both products stay
    in the same compute so their common source tasks are still shared.
    """
    if longitude_group <= 0:
        raise ValueError("longitude_group must be positive")
    if not pairs:
        return

    widths = {int(da.sizes["longitude"]) for da, _path in pairs}
    if len(widths) != 1:
        raise ValueError("bounded intermediate arrays must share a longitude width")
    width = widths.pop()

    # ``to_raster(compute=False)`` writes the full-shape header before it builds
    # the deferred store. Discard that full store graph: the loop below builds
    # only the region stores that are actually executed.
    lazy_pairs = []
    for da, path in pairs:
        deferred = _write_intermediate(da, path, compute=False)
        if deferred is not None:
            lazy_pairs.append((da, path))
        del deferred
    if not lazy_pairs:
        return

    target_lock = threading.Lock()
    for start in range(0, width, longitude_group):
        stop = min(start + longitude_group, width)
        slices = [da.isel(longitude=slice(start, stop)) for da, _path in lazy_pairs]
        sources = [part.data for part in slices]
        targets: list[Any] = [RasterioWriter(path) for _da, path in lazy_pairs]
        regions = [
            (slice(None), slice(start, stop))
            if part.ndim == 2
            else (slice(None), slice(None), slice(start, stop))
            for part in slices
        ]
        stores = dask_array.store(
            sources,
            targets,
            regions=regions,
            lock=target_lock,
            compute=False,
        )
        if isinstance(stores, tuple | list):
            dask.compute(*stores)
        else:
            dask.compute(stores)
        del stores, targets, sources, slices


def _dataset_tags(attrs: Mapping[str, Any]) -> dict[str, str]:
    """Render dataset attrs (tile, year, window, scene_count, ...) as TIFF tags."""
    return {str(key): str(value) for key, value in attrs.items() if value is not None}


def _statistics_tags(
    *,
    valid: float,
    total: int,
    total_sum: float,
    total_sumsq: float,
    minimum: float | None,
    maximum: float | None,
) -> dict[str, str]:
    """Format accumulated moments as GDAL ``STATISTICS_*`` tag values."""
    if valid == 0 or minimum is None or maximum is None:
        # An all-nodata band has no statistics to report, but the keys are
        # mandatory, so record the emptiness explicitly via valid_percent.
        return dict.fromkeys(_STATISTIC_KEYS, "0")
    mean = total_sum / valid
    # Guard against a tiny negative from floating-point cancellation.
    variance = max(total_sumsq / valid - mean * mean, 0.0)
    values = (minimum, maximum, mean, variance**0.5, 100.0 * valid / total)
    return {key: repr(float(value)) for key, value in zip(_STATISTIC_KEYS, values, strict=True)}


class _BandMoments:
    """Running moments for one band, exact in float64.

    Accumulators are float64 so a full 18,000 x 18,000 uint16 band sums without
    loss. Statistics are of raw DN, per GDAL convention.
    """

    __slots__ = ("maximum", "minimum", "total", "total_sum", "total_sumsq", "valid")

    def __init__(self) -> None:
        self.valid = self.total_sum = self.total_sumsq = 0.0
        self.total = 0
        self.minimum: float | None = None
        self.maximum: float | None = None

    def update(self, block: np.ndarray, nodata: float | None) -> None:
        """Fold one window of one band in, ignoring ``nodata`` if there is one."""
        self.total += block.size
        if nodata is not None:
            block = block[block != nodata]
        if block.size == 0:
            return
        self.valid += block.size
        self.total_sum += float(block.sum())
        self.total_sumsq += float(np.square(block).sum())
        block_min, block_max = float(block.min()), float(block.max())
        self.minimum = block_min if self.minimum is None else min(self.minimum, block_min)
        self.maximum = block_max if self.maximum is None else max(self.maximum, block_max)

    def tags(self) -> dict[str, str]:
        """Render the accumulated moments as GDAL ``STATISTICS_*`` tag values."""
        return _statistics_tags(
            valid=self.valid,
            total=self.total,
            total_sum=self.total_sum,
            total_sumsq=self.total_sumsq,
            minimum=self.minimum,
            maximum=self.maximum,
        )


def _coverage_bins(src: rasterio.io.DatasetWriter) -> int:
    """Number of histogram bins needed for the per-pixel sum over all bands."""
    return int(np.iinfo(src.dtypes[0]).max) * src.count + 1


def _embed_statistics(
    src: rasterio.io.DatasetWriter, nodata: float | None, *, coverage: bool = False
) -> np.ndarray | None:
    """Attach exact ``STATISTICS_*`` band tags to an open, already-written raster.

    One walk of the raster serves every band. Reading band by band would scan a
    12-band QA intermediate twelve times for the same bytes.

    Args:
        src: Open raster, already written, in a mode that permits tag updates.
        nodata: Value to exclude from the statistics, or ``None`` to count every
            pixel as valid.
        coverage: Also accumulate the distribution of the per-pixel sum across
            bands. Only meaningful where the bands are counts (``qa_count``),
            and it requires an integer dtype.

    Returns:
        The coverage histogram when ``coverage`` is set, else ``None``. Index
        ``i`` holds the number of pixels whose bands sum to ``i``.
    """
    moments = [_BandMoments() for _ in range(src.count)]
    histogram = np.zeros(_coverage_bins(src), dtype=np.int64) if coverage else None
    for _, window in src.block_windows(1):
        block = src.read(window=window)
        for band, band_moments in zip(block, moments, strict=True):
            band_moments.update(band.astype(np.float64), nodata)
        if histogram is not None:
            per_pixel = block.sum(axis=0, dtype=np.int64).ravel()
            histogram += np.bincount(per_pixel, minlength=histogram.size)
    for bidx, band_moments in enumerate(moments, start=1):
        src.update_tags(bidx, **band_moments.tags())
    return histogram


def _coverage_summary(histogram: np.ndarray) -> dict[str, float]:
    """Reduce the per-pixel observation histogram to the four reported numbers.

    The median is the exact order statistic ``numpy.median`` would return, taken
    from the cumulative counts rather than from a materialized array.
    """
    total = int(histogram.sum())
    if total == 0:
        return {"min": 0, "median": 0.0, "max": 0, "zero_frac": 0.0}
    cumulative = np.cumsum(histogram)
    half = total // 2
    if total % 2:
        median = float(np.searchsorted(cumulative, half + 1))
    else:
        lower = int(np.searchsorted(cumulative, half))
        upper = int(np.searchsorted(cumulative, half + 1))
        median = (lower + upper) / 2
    populated = np.nonzero(histogram)[0]
    return {
        "min": int(populated[0]),
        "median": median,
        "max": int(populated[-1]),
        "zero_frac": round(float(histogram[0] / total), 3),
    }


def _to_cog(src_tif: Path, cog_path: Path, *, nodata: float | None) -> None:
    """Translate a plain GeoTIFF to a validated deflate COG with average overviews."""
    cog_path.parent.mkdir(parents=True, exist_ok=True)
    # rio_cogeo merges the profile into the scratch raster's creation options as
    # well as the output's, which is the half that matters: the pyramid is built
    # there, uncompressed, and that is where a 12-band tile overruns a classic
    # TIFF's 32-bit offsets.
    profile = cog_profiles.get("deflate")
    profile["BIGTIFF"] = _BIGTIFF
    cog_translate(
        str(src_tif),
        str(cog_path),
        profile,
        overview_resampling=_OVERVIEW_RESAMPLING,
        nodata=nodata,
        # Without this the STATISTICS_* band tags are dropped on the floor.
        forward_band_tags=True,
        quiet=True,
    )
    is_valid, errors, _warnings = cog_validate(str(cog_path))
    if not is_valid:
        msg = f"output is not a valid COG: {cog_path} errors={errors}"
        raise RuntimeError(msg)


@dataclass(frozen=True)
class Product:
    """One COG to write: the array, its nodata, and how to finish the raster."""

    da: xr.DataArray
    nodata: float | None
    cog_path: Path
    scratch_prefix: str
    #: Accumulate the per-pixel observation histogram during the statistics walk.
    coverage: bool
    #: Band-level touches applied to the open intermediate before statistics.
    describe: Callable[[rasterio.io.DatasetWriter], None]


def lst_product(native: xr.Dataset, cog_path: Path, var: str = "lst_p95") -> Product:
    """Describe a single-band LST COG (uint16 DN with scale/offset).

    ``var`` selects the variable so the pooled all-path baseline can ride the
    identical writer, encoding, scale, offset and nodata as the shipped
    product. Two assets that differ in anything but their pixels would not be
    comparable in a viewer.
    """

    def describe(src: rasterio.io.DatasetWriter) -> None:
        # Embed GDAL band scale/offset so viewers auto-decode DN -> Celsius.
        src.scales = (LST_SCALE,)
        src.offsets = (LST_OFFSET,)

    return Product(
        da=_prep(native[var]).rio.write_nodata(LST_FILL_VALUE),
        nodata=LST_FILL_VALUE,
        cog_path=cog_path,
        scratch_prefix="lst_cog_",
        coverage=False,
        describe=describe,
    )


def qa_product(native: xr.Dataset, cog_path: Path) -> Product:
    """Describe the 12-band ``qa_count`` COG (uint8, one band per calendar month)."""
    qa = _prep(native["qa_count"])
    if "month" in qa.dims:
        qa = qa.transpose("month", "latitude", "longitude")

    def describe(src: rasterio.io.DatasetWriter) -> None:
        # Label each band with its month so QGIS shows Jan..Dec, not Band 1..12.
        for band_idx in range(1, src.count + 1):
            src.set_band_description(band_idx, _MONTH_NAMES[band_idx - 1])

    # No nodata anywhere on this product: 0 observations is data, not absence of
    # data, and it has to stay visible for gap diagnosis. The explicit strip is
    # load-bearing -- qa_count arrives carrying a nodata attr off the loaded
    # stack, and rio.to_raster would otherwise stamp 0.0 onto the intermediate,
    # from where merge_bands and cog_translate both inherit it.
    return Product(
        da=qa.rio.write_nodata(None),
        nodata=None,
        cog_path=cog_path,
        scratch_prefix="qa_cog_",
        coverage=True,
        describe=describe,
    )


def _export(native: xr.Dataset, products: Sequence[Product]) -> None:
    """Write every product, sharing one pass over the sources they have in common.

    All the intermediates are written first, in one compute, and only then
    finished one at a time. Interleaving would put a ``cog_translate`` between
    the two writes and cost the second one its shared source blocks. The price
    is that every intermediate is on scratch disk at once rather than one at a
    time, which is why they are deflate-compressed.
    """
    with ExitStack() as stack:
        scratch = [stack.enter_context(_scratch_tif(p.scratch_prefix)) for p in products]
        write_intermediates([(p.da, tif) for p, tif in zip(products, scratch, strict=True)])
        for product, src_tif in zip(products, scratch, strict=True):
            finish_product(src_tif, product, native.attrs)


def finish_product(src_tif: Path, product: Product, attrs: Mapping[str, Any]) -> Path:
    """Turn one written intermediate into its finished COG.

    Split out of :func:`_export` so that a merge step can run it over an
    intermediate it assembled from row bands rather than one it computed. The
    tail is identical either way, and it has to be: the band descriptions, the
    scale/offset pair, and the exact ``STATISTICS_*`` moments are what a
    sharded tile has to match a single-VM one on, and a second implementation
    of this sequence would drift on one of them without failing anything.

    Args:
        src_tif: A written plain GeoTIFF, opened here in ``r+``.
        product: What it is and how to finish it.
        attrs: Dataset attrs to stamp on as TIFF tags.

    Returns:
        The COG path written.
    """
    with rasterio.open(src_tif, "r+") as src:
        product.describe(src)
        # Assert the product's nodata rather than trusting what the intermediate
        # happens to carry. A merged tile inherits band 0's profile, and
        # cog_translate(nodata=None) declines to set one rather than clearing
        # one, so this is the only point that holds for both paths.
        src.nodata = product.nodata
        src.update_tags(**_dataset_tags(attrs))
        histogram = _embed_statistics(src, product.nodata, coverage=product.coverage)
    if histogram is not None:
        _log_coverage(attrs, histogram)
    _to_cog(src_tif, product.cog_path, nodata=product.nodata)
    return product.cog_path


def merge_bands(
    band_paths: Sequence[Path],
    dst_tif: Path,
    band_windows: Sequence[tuple[int, int]],
) -> Path:
    """Stack per-band GeoTIFFs into one full-tile intermediate, window by window.

    Each band file holds the same columns and a contiguous run of the output's
    rows. The copy walks the *source's* block windows and writes each one at
    its destination row offset, so nothing larger than a block is resident and
    the tile is assembled from a stream rather than a concatenation. Band
    boundaries are multiples of the block height
    (:func:`landsat_lst.shards.band_edges`), which is what makes every write
    land on a destination block edge instead of forcing GDAL into a
    read-modify-write of a straddling block.

    The result is a **plain tiled GeoTIFF, never a COG**. Overviews belong to
    the assembled tile: a pyramid built per band would resample across a
    boundary the output does not have, and the merged file exists precisely so
    :func:`finish_product` can build one pyramid over the whole thing. The
    profile matches ``_write_intermediate``'s so the finished COG's blocking is
    the same whichever way the tile was produced.

    Args:
        band_paths: One GeoTIFF per row band, in band order.
        dst_tif: Destination intermediate.
        band_windows: ``(row_start, row_stop)`` for each band, in the same
            order. The last stop is the output height.

    Returns:
        ``dst_tif``.

    Raises:
        ValueError: If the bands disagree in width, band count, or dtype, or if
            a band's height does not match the window it claims.
    """
    if len(band_paths) != len(band_windows):
        msg = f"{len(band_paths)} bands but {len(band_windows)} windows"
        raise ValueError(msg)

    with rasterio.open(band_paths[0]) as first:
        profile = dict(first.profile)
        reference = (first.width, first.count, first.dtypes[0])

    profile.update(
        height=band_windows[-1][1],
        tiled=True,
        blockxsize=_BLOCKSIZE,
        blockysize=_BLOCKSIZE,
        compress="deflate",
    )

    dst_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_tif, "w", **profile) as dst:
        for path, (row_start, row_stop) in zip(band_paths, band_windows, strict=True):
            with rasterio.open(path) as src:
                if (src.width, src.count, src.dtypes[0]) != reference:
                    msg = (
                        f"{path.name} does not match band 0: "
                        f"{(src.width, src.count, src.dtypes[0])} vs {reference}"
                    )
                    raise ValueError(msg)
                if src.height != row_stop - row_start:
                    msg = (
                        f"{path.name} is {src.height} rows but claims rows "
                        f"[{row_start}, {row_stop})"
                    )
                    raise ValueError(msg)
                for _, window in src.block_windows(1):
                    dst.write(
                        src.read(window=window),
                        window=Window.from_slices(
                            (
                                window.row_off + row_start,
                                window.row_off + row_start + window.height,
                            ),
                            (window.col_off, window.col_off + window.width),
                        ),
                    )
    return dst_tif


def _log_coverage(attrs: Mapping[str, Any], histogram: np.ndarray) -> None:
    """Report valid observations per pixel, the check on silent nodata fill.

    ``load_scenes`` runs with ``fail_on_error=False``, so a read failure fills a
    scene with nodata instead of aborting the tile. Occasional fill is the point;
    mass fill is a broken run, and a low median or a high zero fraction is what
    distinguishes them. Deriving it from the written raster keeps the check
    without the full native pass the old eager reduction cost (issue #80).
    """
    log.info(
        "valid_coverage_obs_per_pixel",
        tile=attrs.get("tile"),
        window=attrs.get("window"),
        **_coverage_summary(histogram),
    )


def export_lst_cog(native: xr.Dataset, cog_path: Path) -> Path:
    """Write the single-band ``lst_p95`` COG (uint16 DN with scale/offset)."""
    _export(native, [lst_product(native, cog_path)])
    return cog_path


def export_qa_cog(native: xr.Dataset, cog_path: Path) -> Path:
    """Write the 12-band ``qa_count`` COG (uint8, one band per calendar month)."""
    _export(native, [qa_product(native, cog_path)])
    return cog_path


def cog_export(native: xr.Dataset, lst_cog: Path, qa_cog: Path) -> tuple[Path, Path]:
    """Export both LST and per-month QA COGs from a native composite level.

    Both products come out of one pass over the source scenes. Calling
    :func:`export_lst_cog` and :func:`export_qa_cog` in sequence instead would
    read every scene twice, which at production geometry is roughly half an hour
    of the tile thrown away (issue #80).

    Args:
        native: Decoded native-resolution level with ``lst_p95`` (uint16 DN) and
            ``qa_count`` (uint8, dims ``(month, latitude, longitude)``). Any
            dataset attrs (``tile``, ``year``, ``window``, ``scene_count``) are
            carried into both COGs as dataset-level TIFF tags.
        lst_cog: Output path for the single-band LST COG.
        qa_cog: Output path for the 12-band monthly QA COG.

    Returns:
        ``(lst_cog, qa_cog)`` paths written.
    """
    _export(native, [lst_product(native, lst_cog), qa_product(native, qa_cog)])
    return lst_cog, qa_cog
