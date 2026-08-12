"""Compare P95 composites: raw, natively de-striped, and coarse-offset (issue #46).

Two questions, both about the composite rather than the offsets:

1. Does estimating offsets from a coarse grid change the output map? The
   offsets agree to a median of 0.002 degC at factor 2, but the sparse guards
   differ between grids, so the two paths keep slightly different scene sets.
   Agreement on offsets does not by itself prove agreement on composites.
2. Does de-striping preserve the hot signal? The mean-preservation numbers
   quoted during the investigation (41.0 at 3 years, 42.0 at 5, against a 40.6
   baseline) all predate scene rejection existing, so the effect of discarding
   ~22% of scenes on the actual P95 has never been measured.

The offsets and per-scene valid counts are read from the sweep output rather
than recomputed. They are deterministic (the sweep reproduced the committed
calibration to 0.0005 degC), and recomputing would cost several extra full
passes over the stack for numbers already on disk. So this loads the stack once
and derives all three composites from it in a single dask.compute, letting the
graphs share the read.

    uv run python scripts/compare_destripe_composites.py

Planetary Computer per CLAUDE.md. The AOI is inland, so no land mask is applied
here; ocean handling is process_tile's job and is exercised elsewhere.
"""

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_BBOX = [-61.1, -34.4, -60.1, -33.4]
SWEEP_JSON = Path("results/decision/offset_subsampling_validation.json")

# A pixel-level difference below this is not visible in a product whose signal
# is urban-versus-rural contrast of several degrees.
MATERIAL_C = 0.1


def _keep_mask(offsets, n_valid, floor, cap):
    """Reproduce the production keep rule for one estimation grid."""
    o = np.asarray(offsets, dtype="float64")
    n = np.asarray(n_valid)
    return np.isfinite(o) & (n >= floor) & (np.abs(o) <= cap)


def _stats(name, arr, log):
    v = arr[np.isfinite(arr)]
    out = {
        "name": name,
        "mean": round(float(v.mean()), 3),
        "median": round(float(np.median(v)), 3),
        "p05": round(float(np.percentile(v, 5)), 3),
        "p95": round(float(np.percentile(v, 95)), 3),
        "valid_frac": round(float(v.size / arr.size), 4),
    }
    log.info("composite", **out)
    return out


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--cogs", action="store_true", help="Export COGs for QGIS")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/decision/destripe_composite_comparison.json"),
    )
    args = parser.parse_args()

    import dask  # noqa: PLC0415
    import pystac_client  # noqa: PLC0415
    import structlog  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415
    from dask.distributed import Client, LocalCluster  # noqa: PLC0415

    from landsat_lst.azure_auth import enable_pc_azure_refresh  # noqa: PLC0415
    from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings  # noqa: PLC0415
    from landsat_lst.pipeline import load_scenes  # noqa: PLC0415
    from landsat_lst.qa import apply_qa_mask, convert_to_celsius  # noqa: PLC0415

    settings.stac_url = STAC_PLANETARY_COMPUTER
    # Three quantiles share each block, so shrink the block to keep peak memory
    # in range: a 390-step time stack at 512x512 is ~400 MB per chunk per graph.
    settings.load_chunk_size = args.chunk
    log = structlog.get_logger()
    bbox = tuple(args.bbox)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sweep = json.loads(SWEEP_JSON.read_text())
    runs = {r["factor"]: r for r in sweep["runs"]}
    cap = sweep["cap_c"]
    native, coarse = runs[1], runs[args.factor]

    keep_native = _keep_mask(
        native["offsets"], native["n_valid"], settings.destripe_min_scene_pixels, cap
    )
    keep_coarse = _keep_mask(
        coarse["offsets"], coarse["n_valid"], settings.destripe_min_offset_samples, cap
    )
    off_native = np.nan_to_num(np.asarray(native["offsets"], dtype="float64"))
    off_coarse = np.nan_to_num(np.asarray(coarse["offsets"], dtype="float64"))

    disagree = np.flatnonzero(keep_native != keep_coarse)
    log.info(
        "scene_sets",
        n_scenes=len(keep_native),
        kept_native=int(keep_native.sum()),
        kept_coarse=int(keep_coarse.sum()),
        disagreements=int(disagree.size),
    )
    for i in disagree:
        log.info(
            "keep_disagreement",
            idx=int(i),
            native="keep" if keep_native[i] else "drop",
            coarse="keep" if keep_coarse[i] else "drop",
            off_native=round(float(off_native[i]), 2),
            off_coarse=round(float(off_coarse[i]), 2),
        )

    cluster = LocalCluster(
        n_workers=settings.dask_workers,
        threads_per_worker=settings.dask_threads_per_worker,
        memory_limit=settings.dask_memory_limit,
    )
    client = Client(cluster)

    try:
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
        patch_url = enable_pc_azure_refresh(items)
        data = load_scenes(items, bbox, patch_url=patch_url, fail_on_error=False)
        lst = convert_to_celsius(apply_qa_mask(data)["lwir11"])

        n_time = lst.sizes["time"]
        if n_time != len(keep_native):
            msg = f"scene count moved: {n_time} now vs {len(keep_native)} in the sweep"
            raise ValueError(msg)
        log.info("stack_ready", scenes=n_time, chunk=args.chunk)

        def composite(keep: np.ndarray, offsets: np.ndarray):
            idx = np.flatnonzero(keep)
            sub = lst.isel(time=idx)
            shifted = sub - xr.DataArray(offsets[idx], dims=["time"], coords={"time": sub.time})
            valid = shifted.notnull().sum(dim="time")
            p95 = shifted.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")
            return p95.where(valid > 0), valid

        all_keep = np.ones_like(keep_native)
        raw_p95, raw_n = composite(all_keep, np.zeros_like(off_native))
        nat_p95, nat_n = composite(keep_native, off_native)
        crs_p95, crs_n = composite(keep_coarse, off_coarse)

        log.info("computing_composites")
        raw, rawn, nat, natn, crs, crsn = dask.compute(
            raw_p95, raw_n, nat_p95, nat_n, crs_p95, crs_n
        )
        raw, nat, crs = raw.values, nat.values, crs.values

        report = {
            "window": f"{args.year}-{args.end_year}",
            "factor": args.factor,
            "n_scenes": int(n_time),
            "kept_native": int(keep_native.sum()),
            "kept_coarse": int(keep_coarse.sum()),
            "keep_disagreements": int(disagree.size),
            "composites": [
                _stats("raw", raw, log),
                _stats("destriped_native", nat, log),
                _stats(f"destriped_coarse_f{args.factor}", crs, log),
            ],
        }

        # Question 1: does the coarse path change the map?
        both = np.isfinite(nat) & np.isfinite(crs)
        d = crs[both] - nat[both]
        a = np.abs(d)
        report["coarse_vs_native"] = {
            "mean_delta": round(float(d.mean()), 4),
            "median_abs": round(float(np.median(a)), 4),
            "p99_abs": round(float(np.percentile(a, 99)), 4),
            "max_abs": round(float(a.max()), 4),
            "frac_pixels_over_0.1C": round(float((a > MATERIAL_C).mean()), 5),
            "spatial_corr": round(float(np.corrcoef(nat[both], crs[both])[0, 1]), 6),
            "coverage_delta_mean": round(float((crsn.values - natn.values).mean()), 3),
        }
        log.info("coarse_vs_native", **report["coarse_vs_native"])

        # Question 2: does de-striping preserve the hot signal?
        both2 = np.isfinite(nat) & np.isfinite(raw)
        report["destriped_vs_raw"] = {
            "mean_shift": round(float((nat[both2] - raw[both2]).mean()), 3),
            "spatial_corr": round(float(np.corrcoef(raw[both2], nat[both2])[0, 1]), 4),
            "rejection_frac": round(1 - float(keep_native.mean()), 4),
            "coverage_mean_raw": round(float(rawn.values.mean()), 1),
            "coverage_mean_destriped": round(float(natn.values.mean()), 1),
        }
        log.info("destriped_vs_raw", **report["destriped_vs_raw"])

        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nWritten to {args.out}")

        if args.cogs:
            from landsat_lst.cog import export_lst_cog  # noqa: PLC0415
            from landsat_lst.encoding import encode_lst_uint16  # noqa: PLC0415

            outdir = Path("results/decision/percentile_test")
            outdir.mkdir(parents=True, exist_ok=True)
            for name, arr in (
                ("raw", raw),
                ("destriped_native", nat),
                (f"destriped_coarse_f{args.factor}", crs),
            ):
                da = xr.DataArray(
                    arr,
                    dims=["latitude", "longitude"],
                    coords={"latitude": lst.latitude, "longitude": lst.longitude},
                )
                ds = xr.Dataset({"lst_p95": encode_lst_uint16(da)})
                path = outdir / f"cmp_{name}_{args.year}-{args.end_year}.tif"
                export_lst_cog(ds, path)
                print(f"  COG: {path}")
    finally:
        client.close()
        cluster.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
