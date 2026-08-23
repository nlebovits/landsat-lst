"""Re-finish a shipped ``qa_count`` COG from its band slabs, without recomputing it.

The S30W065 tile published on 2026-08-23 carries two defects that live entirely
in the COG tail, not in the pixels:

* ``nodata=0.0``, though zero observations is data -- the header contradicts the
  tile's own ``STATISTICS_VALID_PERCENT`` of 100 and ``STATISTICS_MINIMUM`` of 0.
* a destroyed overview cascade: level 2 truncated at row 2048 of 9000 and levels
  4 through 64 never written, so anything zoomed past 1:4 renders a blank layer.

Both are fixed in :mod:`landsat_lst.cog`. Neither needs the composite recomputed:
the per-band slabs under ``_shards/`` hold the finished native pixels, and
:func:`~landsat_lst.cog.merge_bands` plus :func:`~landsat_lst.cog.finish_product`
are exactly what the original export ran over them. So this re-runs the tail.

    # what it would do, fetching and rebuilding but uploading nothing
    uv run python scripts/reexport_qa_count.py

    # another tile, another run
    uv run python scripts/reexport_qa_count.py --tile N40W075 --run-id shard-...

    # actually replace the object
    uv run python scripts/reexport_qa_count.py --apply

**Dry run is the default.** ``--apply`` is the only thing that writes, and it
runs only after the rebuilt COG passes the same checks that would have caught
the original defect -- an unverified re-export would replace one broken object
with another.

``lst_p95`` is not touched. Its overviews are intact and its DN 0 really is
absent data.

Needs ``LST_STORAGE_BACKEND=s3`` (plus a live session) to reach a published
tile; the slabs are ~2 GB for a five-degree tile, so budget the download.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import rasterio

from landsat_lst import shards
from landsat_lst.cog import finish_product, merge_bands, qa_product
from landsat_lst.shard_tasks import _placeholder_native, job_for_window
from landsat_lst.storage import S3Storage, get_storage

if TYPE_CHECKING:
    from landsat_lst.storage import StorageBackend

PRODUCT = "qa_count"

DEFAULT_TILE = "S30W065"
DEFAULT_WINDOW = "2021-2025"

#: How far an overview's mean may drift from its band's native mean before the
#: cascade is judged broken rather than merely resampled.
#:
#: Wide on purpose. Averaging into uint8 rounds at every level and the bias
#: accumulates with depth -- a healthy rebuild drifted 1.7% by 4x on random
#: counts. The signature being caught is nothing like that: the shipped tile's
#: level 2 read 3.57 against a native 17.58, 80% low, and levels 4 through 64
#: were flat zero. A tight band would only reject good rebuilds.
LEVEL_MEAN_RELATIVE_TOLERANCE = 0.2

#: Floor for the band above, so a tile whose counts are genuinely near zero
#: does not get an impossibly narrow tolerance.
LEVEL_MEAN_ABSOLUTE_FLOOR = 1.0


@dataclass(frozen=True)
class LevelStats:
    """What one overview level of one band looks like."""

    factor: int
    mean: float
    nonzero: float

    def describe(self) -> str:
        return f"level {self.factor:>3}: mean {self.mean:8.4f}  nonzero {self.nonzero:.4f}"


def _band_levels(path: Path | str, bidx: int = 1) -> list[LevelStats]:
    """Every *overview* level of one band, as a mean and a nonzero fraction.

    The native level is deliberately not read. A five-degree tile is 324 MB per
    band and 3.9 GB across twelve, which is a wasteful way to learn a number
    :func:`_native_mean` reads out of a tag -- and ruinous against a remote
    object, where it is the difference between a header fetch and a full
    download. Overview reads are decimated, so each costs its own size.
    """
    stats = []
    with rasterio.open(str(path)) as src:
        for factor in src.overviews(bidx):
            values = src.read(bidx, out_shape=(src.height // factor, src.width // factor))
            stats.append(LevelStats(factor, float(values.mean()), float((values != 0).mean())))
    return stats


def _native_mean(path: Path | str, bidx: int) -> float | None:
    """The band's native mean, from the ``STATISTICS_MEAN`` tag the export embeds.

    Exact, not approximate: ``cog._embed_statistics`` accumulates it in float64
    over every pixel. Reading the tag rather than the raster is what makes the
    health check affordable.
    """
    with rasterio.open(str(path)) as src:
        raw = src.tags(bidx).get("STATISTICS_MEAN")
    return float(raw) if raw is not None else None


def _healthy(path: Path | str) -> list[str]:
    """Every reason the rebuilt COG is not fit to publish. Empty means fit.

    This is the gate the original export did not have. ``cog_validate`` returns
    True on the broken file -- it checks structure, not content -- so the checks
    that matter are on the header's nodata and on the values at each level.
    """
    problems = []
    with rasterio.open(str(path)) as src:
        if src.nodata is not None:
            problems.append(f"dataset declares nodata={src.nodata}, expected none")
        if set(src.nodatavals) != {None}:
            problems.append(f"bands declare nodata={set(src.nodatavals)}, expected none")
        band_count = src.count
        overviews = src.overviews(1)

    if not overviews:
        problems.append("no overviews were built")

    for bidx in range(1, band_count + 1):
        native = _native_mean(path, bidx)
        if native is None:
            problems.append(f"band {bidx} carries no STATISTICS_MEAN tag")
            continue
        tolerance = max(LEVEL_MEAN_ABSOLUTE_FLOOR, LEVEL_MEAN_RELATIVE_TOLERANCE * native)
        for level in _band_levels(path, bidx):
            if level.nonzero == 0.0:
                problems.append(f"band {bidx} {level.factor}x collapsed to all-zero")
            elif abs(level.mean - native) > tolerance:
                problems.append(
                    f"band {bidx} {level.factor}x mean {level.mean:.4f} "
                    f"vs native {native:.4f} (tolerance {tolerance:.4f})"
                )
    return problems


def discover_run_id(storage: StorageBackend, tile: str, window: str) -> str:
    """Newest run that published shards for this tile-window.

    Run ids embed the tile and window (``shard-S30W065-2021-2025-<stamp>``), so
    this is one targeted listing rather than a scan of every run, and the stamp
    sorts lexically.
    """
    prefix = f"{shards.SHARD_PREFIX}/shard-{tile}-{window}-"
    run_ids = {
        key[len(shards.SHARD_PREFIX) + 1 :].split("/", 1)[0] for key in storage.list_prefix(prefix)
    }
    if not run_ids:
        msg = (
            f"no shard run found under {prefix!r}; pass --run-id explicitly, and "
            "check LST_STORAGE_BACKEND=s3"
        )
        raise SystemExit(msg)
    return max(run_ids)


def fetch_slabs(storage: StorageBackend, root: str, scratch: Path) -> list[Path]:
    """Download every published band slab of the product, in band order.

    The count comes from the bucket rather than from the plan: the slabs are
    what the merge actually consumes, and a plan that disagreed with them would
    be the more suspect of the two.
    """
    prefix = f"{root}/composite/{PRODUCT}/"
    keys = sorted(storage.list_prefix(prefix))
    if not keys:
        msg = f"no band slabs under {prefix!r}"
        raise SystemExit(msg)

    paths = []
    for index, key in enumerate(keys):
        local = scratch / f"band{index:03d}.tif"
        if not storage.download(key, local):
            msg = f"band slab {key} vanished mid-download"
            raise SystemExit(msg)
        paths.append(local)
        print(f"  [{index + 1:>2}/{len(keys)}] {key}  {local.stat().st_size / 1e6:.1f} MB")
    return paths


def band_windows(slabs: list[Path]) -> list[tuple[int, int]]:
    """``(row_start, row_stop)`` per slab, accumulated from their own heights.

    Derived from the rasters rather than read from the plan so that the windows
    cannot disagree with the pixels they describe. ``merge_bands`` re-checks
    each one against its slab anyway.
    """
    windows, row = [], 0
    for slab in slabs:
        with rasterio.open(slab) as src:
            windows.append((row, row + src.height))
            row += src.height
    return windows


def tile_attrs(storage: StorageBackend, root: str, tile: str, window: str) -> dict[str, object]:
    """The dataset tags the original export stamped on.

    Mirrors ``shard_tasks._tile_attrs`` but reads the plan as plain JSON, so a
    settings default that has drifted since the run cannot make a *repair*
    refuse. Nothing here feeds a computation; these are provenance tags.
    """
    raw = storage.read_text(shards.plan_key(root))
    scene_count = len(json.loads(raw)["scene_ids"]) if raw else None
    return {
        "tile": tile,
        "year": job_for_window(tile, window).year,
        "window": window,
        "scene_count": scene_count,
    }


def describe_published(storage: StorageBackend, key: str) -> None:
    """Print the state of the object that would be replaced.

    Header and decimated reads only, so this costs range requests rather than
    the object. Best-effort: a tile that has not shipped yet is not an error,
    and neither is a backend this cannot address by URI.
    """
    if not isinstance(storage, S3Storage):
        print(f"  (existing object not inspected: {type(storage).__name__})")
        return
    uri = f"s3://{storage.bucket}/{storage._full_key(key)}"
    try:
        with rasterio.Env(AWS_REGION=storage.region):
            with rasterio.open(uri) as src:
                print(f"  {uri}")
                print(f"  nodata={src.nodata}  overviews={src.overviews(1)}")
            native = _native_mean(uri, 1)
            print(f"    native   : mean {native:8.4f}  (from STATISTICS_MEAN)")
            for level in _band_levels(uri):
                print(f"    {level.describe()}")
    except Exception as e:
        print(f"  (could not read {uri}: {type(e).__name__}: {e})")


def reexport(
    tile: str,
    window: str,
    *,
    run_id: str | None,
    apply: bool,
    scratch_root: Path | None,
    keep_scratch: bool,
) -> int:
    """Rebuild one tile's QA COG from its slabs. Returns a process exit code."""
    storage = get_storage()
    run_id = run_id or discover_run_id(storage, tile, window)
    root = shards.shard_root(run_id, tile)
    key = storage.cog_key(window, tile, PRODUCT)

    print(f"tile      {tile}")
    print(f"window    {window}")
    print(f"run       {run_id}")
    print(f"target    {key}")
    print(f"mode      {'APPLY (will upload)' if apply else 'dry run (uploads nothing)'}\n")

    print("currently published:")
    describe_published(storage, key)

    scratch = Path(tempfile.mkdtemp(prefix="reexport_qa_", dir=scratch_root))
    try:
        print(f"\ndownloading band slabs to {scratch}:")
        slabs = fetch_slabs(storage, root, scratch)
        windows = band_windows(slabs)
        print(f"  {len(slabs)} slabs covering rows 0..{windows[-1][1]}")

        print("\nmerging:")
        merged = merge_bands(slabs, scratch / "merged.tif", windows)
        print(f"  {merged}  {merged.stat().st_size / 1e6:.1f} MB")

        print("\nfinishing (statistics, tags, pyramid):")
        attrs = tile_attrs(storage, root, tile, window)
        print(f"  attrs {attrs}")
        product = qa_product(_placeholder_native(), scratch / f"{PRODUCT}.tif")
        rebuilt = finish_product(merged, product, attrs)
        print(f"  {rebuilt}  {rebuilt.stat().st_size / 1e6:.1f} MB")

        with rasterio.open(rebuilt) as src:
            print(f"  nodata={src.nodata}  overviews={src.overviews(1)}")
        print(f"    native   : mean {_native_mean(rebuilt, 1):8.4f}  (from STATISTICS_MEAN)")
        for level in _band_levels(rebuilt):
            print(f"    {level.describe()}")

        problems = _healthy(rebuilt)
        if problems:
            print("\nREFUSING: the rebuilt COG is not fit to publish")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("\nchecks passed: no nodata anywhere, every overview level intact")

        if not apply:
            print(f"\nWOULD REPLACE {key}")
            print("  nothing was uploaded; re-run with --apply to publish")
            return 0

        print(f"\nuploading -> {key}")
        storage.upload(rebuilt, key)
        print("done")
        return 0
    finally:
        if keep_scratch:
            print(f"\nscratch kept at {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tile", default=DEFAULT_TILE, help="Tile name (default: %(default)s)")
    parser.add_argument(
        "--window", default=DEFAULT_WINDOW, help="Window label (default: %(default)s)"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Shard run holding the band slabs. Default: newest run for the tile-window.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Upload the rebuilt COG. Without this nothing is written.",
    )
    parser.add_argument(
        "--scratch", type=Path, default=None, help="Parent directory for the working files."
    )
    parser.add_argument(
        "--keep-scratch", action="store_true", help="Leave the working files for inspection."
    )
    args = parser.parse_args()

    return reexport(
        args.tile,
        args.window,
        run_id=args.run_id,
        apply=args.apply,
        scratch_root=args.scratch,
        keep_scratch=args.keep_scratch,
    )


if __name__ == "__main__":
    sys.exit(main())
