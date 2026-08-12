#!/usr/bin/env python3
"""Decision run: 1yr / 3yr / 5yr P95 composites for Pergamino (tile S30W065).

Purpose: decide whether multi-year P95 composites meaningfully reduce striping and
cloud gaps versus a single year. For each window we produce the published COG pair:
a single-band uint16 LST COG and a 12-band uint8 monthly QA COG, the latter a
climatology of valid-observation counts.

We also emit ``results/decision/report.md`` with per-window gap/striping metrics and
the **measured** compression ratio, then re-derive the global storage estimate from it.

Runs LOCALLY against Planetary Computer (free, no egress -- per CLAUDE.md). The Dask
client is hijacked by Frisky when installed (``pip install frisky``), else plain Dask.

Usage:
    uv run python scripts/pergamino_multiyear_decision.py
    uv run python scripts/pergamino_multiyear_decision.py --windows 2024      # subset
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog
import xarray as xr

from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings

# Local rule (CLAUDE.md): Planetary Computer endpoint. Set BEFORE any pipeline import
# that reads settings.stac_url at call time; also guards against an .env override.
settings.stac_url = STAC_PLANETARY_COMPUTER

from landsat_lst.cog import cog_export  # noqa: E402
from landsat_lst.encoding import encode_lst_uint16  # noqa: E402
from landsat_lst.models import ProcessingJob  # noqa: E402
from landsat_lst.pipeline import process_tile  # noqa: E402
from landsat_lst.tiling import parse_tile_name  # noqa: E402

log = structlog.get_logger()

TILE_NAME = "S30W065"  # 5-degree tile containing Pergamino, Argentina (-33.9, -60.6)

# Gut-check AOI: ~1 degree around Pergamino (~3600x3600 px) -- big enough to see
# striping/gaps, ~25x fewer pixels (and far fewer scenes) than the full 5deg tile.
# Use --full-tile to process the whole S30W065 tile instead (slow; overnight).
AOI_BBOX = (-61.1, -34.4, -60.1, -33.4)  # (west, south, east, north)

OUT_DIR = Path("results/decision")

# (year, end_year|None, label) for each composite window.
WINDOWS: list[tuple[int, int | None]] = [
    (2024, None),  # 1-year
    (2022, 2024),  # 3-year
    (2020, 2024),  # 5-year
]

# Global extrapolation constant (see plan). 5-degree tiles, 30 m pixels.
# No separate overview term: the measured ratios divide native uncompressed bytes by a
# COG file size that already contains its internal overviews.
GLOBAL_LAND_PIXELS = 1.656e11  # ~149 Mkm2 land / (30 m)^2


@dataclass
class WindowResult:
    """Per-window outputs and metrics."""

    label: str
    scene_count: int = 0
    duration_s: float = 0.0
    land_pixels: int = 0
    gap_pixels: int = 0
    gap_fraction: float = 0.0
    per_month_median: list[float] = field(default_factory=list)
    per_month_pct_zero: list[float] = field(default_factory=list)
    striping_cv: float = 0.0
    lst_ratio: float = 0.0
    qa_ratio: float = 0.0
    lst_cog: str = ""
    qa_cog: str = ""
    error: str | None = None


def setup_client(use_frisky: bool = True):
    """Create a Dask LocalCluster+Client, hijacked by Frisky when requested.

    Frisky is fast at scheduling but (as an early "art project") has proven
    unreliable gathering large results here -- its websocket closes mid-gather on
    the multi-GB composite. Pass ``use_frisky=False`` (``--no-frisky``) to run on
    plain Dask, which is the reliable path for actually producing the datasets.
    """
    from dask.distributed import Client, LocalCluster  # noqa: PLC0415

    cluster = LocalCluster(
        n_workers=settings.dask_workers,
        threads_per_worker=settings.dask_threads_per_worker,
        memory_limit=settings.dask_memory_limit,
        dashboard_address=":8787",
    )
    client = Client(cluster)
    scheduler = "dask"
    if use_frisky:
        try:
            import frisky  # noqa: PLC0415

            client = frisky.hijack(client)  # experimental Rust scheduler
            scheduler = "frisky"
        except Exception as e:  # never let an experimental scheduler block the run
            log.warning("frisky_unavailable_using_dask", error=str(e))
    log.info("scheduler_ready", scheduler=scheduler, dashboard=":8787")
    return cluster, client, scheduler


def _cog_ratio(path: Path) -> float:
    """Uncompressed native bytes over the COG's on-disk size.

    The file size includes the internal overviews, so the ratio describes the
    artifact that actually gets published rather than the native band alone.
    """
    import rasterio  # noqa: PLC0415

    try:
        with rasterio.open(path) as src:
            itemsize = np.dtype(src.dtypes[0]).itemsize
            uncompressed = src.width * src.height * src.count * itemsize
        return uncompressed / path.stat().st_size
    except Exception as e:  # measurement is best-effort, never fail the run
        log.warning("cog_ratio_failed", path=str(path), error=str(e))
        return 0.0


def measure_compression(lst_cog: Path, qa_cog: Path) -> tuple[float, float]:
    """Compression ratio per COG. Returns ``(lst_ratio, qa_ratio)``."""
    return _cog_ratio(lst_cog), _cog_ratio(qa_cog)


def composite_for_bbox(job: ProcessingJob, bbox: tuple[float, float, float, float]) -> xr.Dataset:
    """Build a P95 + monthly-QA composite over a custom bbox (fast gut-check path).

    Uses the low-level pipeline (STAC query -> load -> compute_annual_composite) with
    an arbitrary bbox instead of process_tile's fixed 5deg tile. No land mask (the AOI
    is inland Pergamino), matching scripts/smoke_small_tile_cog.py.
    """
    import pystac_client  # noqa: PLC0415

    from landsat_lst.azure_auth import enable_pc_azure_refresh  # noqa: PLC0415
    from landsat_lst.pipeline import compute_annual_composite, load_scenes  # noqa: PLC0415

    catalog = pystac_client.Client.open(settings.stac_url)
    items = list(
        catalog.search(
            collections=[settings.collection],
            bbox=bbox,
            datetime=job.datetime_range,
            query={
                "eo:cloud_cover": {"lt": settings.max_cloud_cover},
                "platform": {"in": ["landsat-8", "landsat-9"]},
            },
        ).items()
    )
    if not items:
        msg = f"no scenes for {job.window_label} in bbox {bbox}"
        raise ValueError(msg)
    patch_url = enable_pc_azure_refresh(items)
    data = load_scenes(items, bbox, patch_url=patch_url)
    composite = compute_annual_composite(data)
    composite.attrs["scene_count"] = len(items)
    composite.attrs["window"] = job.window_label
    return composite


def compute_metrics(composite: xr.Dataset, res: WindowResult) -> None:
    """Fill gap/striping/per-month metrics from the in-memory float composite."""
    lst = composite["lst_p95"]
    qa = composite["qa_count"]  # (month, lat, lon) uint8, ocean = 0

    # Land pixels: everything the land mask kept (lst is NaN only over ocean).
    land = ~np.isnan(lst)
    land_count = int(land.sum())
    # Gap = land pixel that never got a valid observation in the window.
    gap = land & (lst == settings.nodata)
    gap_count = int(gap.sum())

    res.land_pixels = land_count
    res.gap_pixels = gap_count
    res.gap_fraction = gap_count / land_count if land_count else 0.0

    # Per-month coverage over land pixels.
    qa_land = qa.where(land)  # ocean/non-land -> NaN, excluded from stats
    for m in range(1, 13):
        month_slice = qa_land.sel(month=m)
        median = float(month_slice.median(skipna=True))
        pct_zero = float((month_slice == 0).sum() / land_count) if land_count else 0.0
        res.per_month_median.append(median)
        res.per_month_pct_zero.append(pct_zero)

    # Striping proxy: CV of per-column (longitude) mean total coverage over land.
    qa_total = qa.sum("month").where(land)
    col_mean = qa_total.mean(dim="latitude", skipna=True)
    mu = float(col_mean.mean(skipna=True))
    sigma = float(col_mean.std(skipna=True))
    res.striping_cv = sigma / mu if mu else 0.0


def process_window(
    year: int,
    end_year: int | None,
    bbox: tuple[float, float, float, float] | None,
) -> WindowResult:
    """Full pipeline + COG + metrics for one window.

    ``bbox=None`` processes the whole 5deg tile (production); a bbox runs the fast
    AOI gut-check path.
    """
    job = ProcessingJob(tile=parse_tile_name(TILE_NAME), year=year, end_year=end_year)
    res = WindowResult(label=job.window_label)
    t0 = time.perf_counter()
    log.info(
        "window_start",
        window=job.window_label,
        datetime_range=job.datetime_range,
        mode="full_tile" if bbox is None else "aoi",
    )

    try:
        composite = process_tile(job) if bbox is None else composite_for_bbox(job, bbox)
        res.scene_count = int(composite.attrs.get("scene_count", 0))
        composite = composite.compute()
        log.info("compute_done", window=job.window_label, size_gb=round(composite.nbytes / 1e9, 2))

        # Metrics from the in-memory float composite (before encoding).
        compute_metrics(composite, res)

        # Encode to the published uint16 contract, export both COGs, measure them.
        native = xr.Dataset(
            {
                "lst_p95": encode_lst_uint16(composite["lst_p95"]),
                "qa_count": composite["qa_count"],
            }
        )
        lst_cog, qa_cog = cog_export(
            native,
            OUT_DIR / f"lst_{res.label}_{job.tile.name}.tif",
            OUT_DIR / f"qa_count_{res.label}_{job.tile.name}.tif",
        )
        res.lst_ratio, res.qa_ratio = measure_compression(lst_cog, qa_cog)
        res.lst_cog, res.qa_cog = str(lst_cog), str(qa_cog)

    except Exception as e:  # record and continue to next window
        log.exception("window_failed", window=res.label, error=str(e))
        res.error = str(e)

    res.duration_s = time.perf_counter() - t0
    log.info("window_done", window=res.label, duration_s=round(res.duration_s, 1))
    return res


def _gb(n: float) -> float:
    return n / 1e9


def write_report(results: list[WindowResult]) -> Path:
    """Write the side-by-side comparison + measured global storage extrapolation."""
    ok = [r for r in results if r.error is None]
    # Use the mean measured ratio across successful windows for extrapolation.
    lst_ratio = float(np.mean([r.lst_ratio for r in ok])) if ok else 0.0
    qa_ratio = float(np.mean([r.qa_ratio for r in ok])) if ok else 0.0

    lines: list[str] = []
    lines.append("# Pergamino multi-year P95 decision run\n")
    lines.append(f"Tile **{TILE_NAME}** | endpoint `{settings.stac_url}`\n")

    lines.append("## Gap & striping comparison\n")
    lines.append("| Window | Scenes | Land px | Gap px | Gap % | Striping CV | Runtime |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for r in results:
        if r.error:
            lines.append(f"| {r.label} | — | — | — | FAILED | — | — |")
            continue
        lines.append(
            f"| {r.label} | {r.scene_count} | {r.land_pixels:,} | {r.gap_pixels:,} | "
            f"{r.gap_fraction * 100:.3f}% | {r.striping_cv:.3f} | {r.duration_s:.0f}s |"
        )
    lines.append("")

    lines.append("## Per-month median coverage (land pixels)\n")
    months = "".join(f" {m} |" for m in range(1, 13))
    lines.append(f"| Window |{months}")
    lines.append("|---|" + "--:|" * 12)
    for r in ok:
        cells = "".join(f" {v:.0f} |" for v in r.per_month_median)
        lines.append(f"| {r.label} |{cells}")
    lines.append("")

    lines.append("## Per-month % zero-coverage (land pixels)\n")
    lines.append(f"| Window |{months}")
    lines.append("|---|" + "--:|" * 12)
    for r in ok:
        cells = "".join(f" {v * 100:.1f} |" for v in r.per_month_pct_zero)
        lines.append(f"| {r.label} |{cells}")
    lines.append("")

    lines.append("## Measured compression (COG, overviews included)\n")
    lines.append("| Window | LST ratio | QA ratio |")
    lines.append("|---|--:|--:|")
    for r in ok:
        lines.append(f"| {r.label} | {r.lst_ratio:.1f}x | {r.qa_ratio:.1f}x |")
    lines.append(
        f"\n**Mean ratios used for extrapolation:** LST {lst_ratio:.1f}x, QA {qa_ratio:.1f}x\n"
    )

    # Global extrapolation with MEASURED ratios.
    px = GLOBAL_LAND_PIXELS
    lst_c = _gb(px * 2 / lst_ratio) if lst_ratio else 0.0
    qa1_c = _gb(px * 1 / qa_ratio) if qa_ratio else 0.0
    lines.append("## Global land storage estimate (measured ratios)\n")
    lines.append("| Layer | Uncompressed | Compressed |")
    lines.append("|---|--:|--:|")
    lines.append(f"| lst_p95 (uint16) | {_gb(px * 2):.0f} GB | {lst_c:.0f} GB |")
    lines.append(f"| QA single (uint8) | {_gb(px * 1):.0f} GB | {qa1_c:.0f} GB |")
    lines.append(f"| QA quarterly (4x) | {_gb(px * 4):.0f} GB | {qa1_c * 4:.0f} GB |")
    lines.append(f"| QA monthly (12x) | {_gb(px * 12):.0f} GB | {qa1_c * 12:.0f} GB |")
    lines.append(
        f"\n**Per global product (LST + monthly QA):** "
        f"{_gb(px * 2 + px * 12):.0f} GB uncompressed / "
        f"{lst_c + qa1_c * 12:.0f} GB compressed — identical for 1/3/5yr "
        f"(climatological QA is 12 bands regardless of window).\n"
    )
    lines.append(
        f"**Incremental monthly-over-single QA:** "
        f"{_gb(px * 11):.0f} GB uncompressed / {qa1_c * 11:.0f} GB compressed.\n"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = OUT_DIR / "report.md"
    report.write_text("\n".join(lines))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        nargs="*",
        help="Window labels to run (e.g. 2024 2022-2024). Default: all three.",
    )
    parser.add_argument(
        "--no-frisky",
        action="store_true",
        help="Force plain Dask instead of the Frisky scheduler (reliable for large gathers).",
    )
    parser.add_argument(
        "--full-tile",
        action="store_true",
        help="Process the whole 5deg S30W065 tile (slow) instead of the fast ~1deg AOI.",
    )
    args = parser.parse_args()
    bbox = None if args.full_tile else AOI_BBOX

    # Select windows.
    selected = WINDOWS
    if args.windows:
        wanted = set(args.windows)
        selected = [(y, e) for (y, e) in WINDOWS if (str(y) if e is None else f"{y}-{e}") in wanted]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cluster, client, _scheduler = setup_client(use_frisky=not args.no_frisky)

    results: list[WindowResult] = []
    try:
        for year, end_year in selected:
            results.append(process_window(year, end_year, bbox))
    finally:
        client.close()
        cluster.close()

    report = write_report(results)
    print(f"\nReport: {report}")
    for r in results:
        status = "OK" if r.error is None else f"FAILED: {r.error}"
        print(f"  {r.label}: {status}")

    return 0 if all(r.error is None for r in results) else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
