#!/usr/bin/env python3
"""Season-aware normalized P95 (de-stripe without cooling).

Removes each scene's per-scene bias *relative to a per-pixel monthly climatology*
(not the annual mean), so the seasonal/hot signal is preserved while the
atmospheric-correction offset that jumps at scene-footprint boundaries is removed:

    ref(x,y,month) = median over time-in-month of LST          # per-pixel seasonal expectation
    anomaly_i      = scene_i - ref(x, y, month_of_i)           # deviation from expected
    offset_i       = spatial median of anomaly_i               # per-scene bias (season removed)
    scene_i'       = scene_i - offset_i
    P95'           = 95th percentile over time of scene_i'

Contrast with the blunt version (reference = annual median), whose per-scene offsets
were dominated by real season (std ~11 degC) and cooled the composite. Here the
offsets should be small (~1-5 degC) because season is already accounted for.

Default window is 3-year (2022-2024), where the monthly climatology is well-sampled.

Usage:
    LST_LOAD_CHUNK_SIZE=256 uv run python scripts/season_aware_p95_test.py
    uv run python scripts/season_aware_p95_test.py --selftest   # fast synthetic check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

MIN_SCENE_PIXELS = 500  # below this valid-pixel count, don't trust a scene's offset


def seasonal_debias(lst: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Return (debiased LST, per-scene offset) using a per-pixel monthly climatology.

    ``lst`` has dims (time, latitude, longitude) with a datetime ``time`` coord.
    Offsets are computed relative to each pixel's monthly median, so real seasonal
    variation is NOT removed -- only each scene's bulk deviation from expectation.
    """
    ref_month = lst.groupby("time.month").median(skipna=True)  # (month, y, x)
    anomaly = lst.groupby("time.month") - ref_month  # (time, y, x)
    offset = anomaly.median(dim=["latitude", "longitude"], skipna=True)  # (time,)
    n_valid = lst.notnull().sum(dim=["latitude", "longitude"])
    offset = offset.where(n_valid > MIN_SCENE_PIXELS, 0.0).fillna(0.0)
    return lst - offset, offset


def _selftest() -> int:
    """Synthetic check: recover injected per-scene bias, preserve seasonal signal."""
    rng = np.random.default_rng(0)
    import pandas as pd  # noqa: PLC0415

    # 3-year sampling (~6 obs/month) -- matches the target window, where the monthly
    # climatology is well enough sampled to isolate per-scene bias (corr ~sqrt(1-1/k)).
    times = pd.to_datetime(
        [f"{y}-{m:02d}-{d:02d}" for y in (2022, 2023, 2024) for m in range(1, 13) for d in (5, 20)]
    )
    ny = nx = 40
    doy = times.dayofyear.values.astype("float64")  # ty: ignore[unresolved-attribute]
    season = 15 * np.sin(2 * np.pi * (doy - 15) / 365)  # +-15 degC seasonal swing
    spatial = rng.normal(0, 3, (ny, nx))  # real spatial pattern
    bias = rng.normal(0, 2, len(times))  # per-scene atmospheric bias (std 2)
    data = (
        30
        + season[:, None, None]
        + spatial[None]
        + bias[:, None, None]
        + rng.normal(0, 0.3, (len(times), ny, nx))
    )
    lst = xr.DataArray(
        data,
        dims=["time", "latitude", "longitude"],
        coords={"time": times, "latitude": np.arange(ny), "longitude": np.arange(nx)},
    )
    deb, offset = seasonal_debias(lst)
    ov = offset.values
    # (1) Offsets must be bias-sized (~2), NOT the seasonal amplitude (~11) -- the exact
    #     failure of the blunt annual-mean version.
    print(
        f"[selftest] injected bias std=2.0  ->  recovered offset std={ov.std():.2f} "
        f"min={ov.min():.2f} max={ov.max():.2f}  (blunt version was ~11)"
    )
    # (2) Positively correlated with the true injected bias (~sqrt(1-1/k) ceiling).
    corr = np.corrcoef(ov, bias)[0, 1]
    print(f"[selftest] corr(offset, injected bias) = {corr:.3f}")
    # (3) Seasonal signal preserved: de-biased seasonal amplitude ~ original.
    amp_orig = float(lst.mean(["latitude", "longitude"]).std())
    amp_deb = float(deb.mean(["latitude", "longitude"]).std())
    print(f"[selftest] seasonal amplitude preserved: orig={amp_orig:.1f} debiased={amp_deb:.1f}")
    ok = ov.std() < 5 and corr > 0.75 and amp_deb > 0.9 * amp_orig
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _run(args: argparse.Namespace) -> int:
    import pystac_client  # noqa: PLC0415
    import structlog  # noqa: PLC0415

    from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings  # noqa: PLC0415

    settings.stac_url = STAC_PLANETARY_COMPUTER
    from dask.distributed import Client, LocalCluster  # noqa: PLC0415

    from landsat_lst.azure_auth import enable_pc_azure_refresh  # noqa: PLC0415
    from landsat_lst.cog import export_lst_cog  # noqa: PLC0415
    from landsat_lst.encoding import encode_lst_uint16  # noqa: PLC0415
    from landsat_lst.pipeline import load_scenes  # noqa: PLC0415
    from landsat_lst.qa import apply_qa_mask, convert_to_celsius  # noqa: PLC0415

    log = structlog.get_logger()
    window = f"{args.year}-{args.end_year}"
    bbox = tuple(args.bbox)
    out = Path(f"results/decision/percentile_test/lst_p95_seasonnorm_{window}_{args.label}.tif")
    out.parent.mkdir(parents=True, exist_ok=True)

    def load_lst() -> xr.DataArray:
        catalog = pystac_client.Client.open(settings.stac_url)
        items = list(
            catalog.search(
                collections=[settings.collection],
                bbox=bbox,
                datetime=f"{args.year}-01-01/{args.end_year}-12-31",
                query={
                    "eo:cloud_cover": {"lt": settings.max_cloud_cover},
                    "platform": {"in": ["landsat-8", "landsat-9"]},
                },
            ).items()
        )
        log.info("scenes_found", n=len(items))
        if not items:
            msg = "no scenes"
            raise ValueError(msg)
        patch_url = enable_pc_azure_refresh(items)
        # Tolerate the occasional transient Azure read failure across the ~1.9k
        # scenes in a 5-year window; a dropped read becomes nodata rather than
        # aborting the whole load (negligible vs a P95 over the full stack).
        data = load_scenes(items, bbox, patch_url=patch_url, fail_on_error=False)
        return convert_to_celsius(apply_qa_mask(data)["lwir11"])

    cluster = LocalCluster(
        n_workers=settings.dask_workers,
        threads_per_worker=settings.dask_threads_per_worker,
        memory_limit=settings.dask_memory_limit,
        dashboard_address=":8787",
    )
    client = Client(cluster)
    log.info("dask_ready", window=window, chunk=settings.load_chunk_size)
    try:
        lst = load_lst()

        lst_deb, offset = seasonal_debias(lst)
        valid = lst.notnull().sum(dim="time")
        p95 = lst_deb.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")
        p95 = p95.where(valid > 0, settings.nodata)

        log.info("computing")
        p95, ov, nobs = p95.compute(), offset.compute().values, valid.compute().values
        # Coverage sanity check: with fail_on_error=False a dropped read is
        # silently filled with nodata, so confirm we retained dense coverage
        # (a low median / high zero_frac would mean reads failed en masse).
        log.info(
            "valid_coverage_obs_per_pixel",
            min=int(nobs.min()),
            median=int(np.median(nobs)),
            max=int(nobs.max()),
            zero_frac=round(float((nobs == 0).mean()), 3),
        )
        log.info(
            "per_scene_offsets_degC",
            std=round(float(np.nanstd(ov)), 2),
            min=round(float(np.nanmin(ov)), 2),
            max=round(float(np.nanmax(ov)), 2),
        )
        ds = xr.Dataset(
            {"lst_p95": encode_lst_uint16(p95)},
            coords={"latitude": p95.latitude, "longitude": p95.longitude},
        )
        export_lst_cog(ds, out)
        vc = p95.where(p95 != settings.nodata)
        log.info(
            "wrote_cog",
            path=str(out),
            degc_min=round(float(vc.min()), 1),
            degc_mean=round(float(vc.mean()), 1),
            degc_max=round(float(vc.max()), 1),
        )
    finally:
        client.close()
        cluster.close()

    print(f"\nSeason-aware normalized P95 COG: {out}")
    print("Compare in QGIS vs: lst_p95_masked_2024 (masked, un-normalized)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run synthetic math check only")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        default=[-61.1, -34.4, -60.1, -33.4],
        help="AOI bbox in EPSG:4326 lon/lat order (default: Pergamino S30W065 sub-box)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="S30W065",
        help="Site label used in the output COG filename",
    )
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
