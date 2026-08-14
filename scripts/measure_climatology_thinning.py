"""Measure the second-order cost of a cloud filter: a thinner climatology (issue #81).

``scripts/analyze_cloud_cover_filter.py`` prices a threshold in the observations
it removes directly. It cannot see the indirect cost, which is the one issue #81
warns about: every scene's offset is measured against a per-pixel monthly
climatology built from the surviving scenes, so removing scenes moves the
reference, which moves the offsets of the scenes that stay, which can move
keep/reject decisions. Sampling 300 scenes from a five-year window is what put
that on the record -- it pushed the rejection rate from 21.8% to 69%.

A cloud filter is not random sampling, so the magnitude has to be measured
rather than assumed. This loads the stack once, computes offsets on the full
scene set and on each filtered subset, and reports how far the survivors moved.

The comparison is paired and same-session, for the reasons
``validate_offset_subsampling.py`` documents: offsets for every subset go into
one ``dask.compute`` so the graphs share the loaded chunks, and each subset is
scored only on the scenes it has in common with the full set.

Runs at ``--factor 2`` by default, the shipped offset resolution. The question
is how a scene's offset moves between two scene sets on the same grid, so the
grid cancels; paying for a native pass to ask it would buy nothing.

    uv run python scripts/measure_climatology_thinning.py
    uv run python scripts/measure_climatology_thinning.py --thresholds 90 80 70

Planetary Computer per CLAUDE.md: Earth Search costs egress from a laptop.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DEFAULT_BBOX = [-61.1, -34.4, -60.1, -33.4]


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[90, 80, 70])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/decision/climatology_thinning.json"),
    )
    args = parser.parse_args()

    import dask  # noqa: PLC0415
    import pystac_client  # noqa: PLC0415
    import structlog  # noqa: PLC0415
    from dask.distributed import Client, LocalCluster  # noqa: PLC0415

    from landsat_lst.azure_auth import enable_pc_azure_refresh  # noqa: PLC0415
    from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings  # noqa: PLC0415
    from landsat_lst.normalization import offset_graph, scene_keep_mask  # noqa: PLC0415
    from landsat_lst.pipeline import load_scenes, scene_cloud_cover  # noqa: PLC0415
    from landsat_lst.qa import apply_qa_mask, convert_to_celsius  # noqa: PLC0415

    settings.stac_url = STAC_PLANETARY_COMPUTER
    log = structlog.get_logger()
    bbox = tuple(args.bbox)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cluster = LocalCluster(
        n_workers=settings.dask_workers,
        threads_per_worker=settings.dask_threads_per_worker,
        memory_limit=settings.dask_memory_limit,
        dashboard_address=":8787",
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
        log.info("scenes_found", n=len(items))
        patch_url = enable_pc_azure_refresh(items)

        data = load_scenes(
            items, bbox, patch_url=patch_url, fail_on_error=False, resolution_factor=args.factor
        )
        lst = convert_to_celsius(apply_qa_mask(data)["lwir11"])
        cover = scene_cloud_cover(items, bbox, lst.time, args.factor)
        log.info("stack_built", scenes=int(lst.sizes["time"]), shape=lst.shape[1:])

        # Subsetting the time axis is what rebuilds the climatology: the monthly
        # median inside offset_graph is taken over whatever scenes it is given.
        subsets = {0: lst}
        for t in args.thresholds:
            subsets[t] = lst.isel(time=np.flatnonzero(cover.values < t))

        log.info("computing_offsets", subsets=list(subsets))
        t0 = time.perf_counter()
        results = dask.compute(*[offset_graph(s) for s in subsets.values()])
        log.info("offsets_computed", secs=round(time.perf_counter() - t0, 1))

        computed = dict(zip(subsets, results, strict=True))
        floor = settings.destripe_min_offset_samples
        cap = settings.destripe_max_offset_c

        full_offset, full_valid = computed[0]
        full_keep = scene_keep_mask(full_offset, full_valid, max_offset_c=cap, floor=floor).values
        full_times = full_offset.time.values

        print(f"\nfull scene set: {full_times.size} scenes, ", end="")
        print(f"{int(full_keep.sum())} kept ({100 * (1 - full_keep.mean()):.1f}% rejected)")
        header = (
            f"\n{'thresh':>7} {'scenes':>7} {'rejected':>9} {'med|d off|':>11} "
            f"{'p99|d off|':>11} {'max|d off|':>11} {'decision flips':>15}"
        )
        print(header)
        print("-" * (len(header) - 1))

        rows = []
        for t in args.thresholds:
            offset, n_valid = computed[t]
            keep = scene_keep_mask(offset, n_valid, max_offset_c=cap, floor=floor).values
            times = offset.time.values

            # Score only the scenes both sets contain. A scene the filter
            # removed has no offset to compare, and its removal is the direct
            # cost that analyze_cloud_cover_filter.py already priced.
            shared = np.isin(full_times, times)
            here = np.isin(times, full_times)
            a = full_offset.values[shared]
            b = offset.values[here]
            both = np.isfinite(a) & np.isfinite(b)
            delta = np.abs(b[both] - a[both])
            flips = int((full_keep[shared] != keep[here]).sum())

            row = {
                "threshold": t,
                "scenes": int(times.size),
                "rejected_frac": float(1 - keep.mean()),
                "median_abs_delta": float(np.median(delta)) if delta.size else 0.0,
                "p99_abs_delta": float(np.percentile(delta, 99)) if delta.size else 0.0,
                "max_abs_delta": float(delta.max()) if delta.size else 0.0,
                "decision_flips": flips,
                "decision_flip_frac": flips / max(1, int(shared.sum())),
            }
            rows.append(row)
            print(
                f"{t:>7} {row['scenes']:>7} {100 * row['rejected_frac']:>8.1f}% "
                f"{row['median_abs_delta']:>11.4f} {row['p99_abs_delta']:>11.4f} "
                f"{row['max_abs_delta']:>11.3f} {flips:>15}"
            )

        args.out.write_text(
            json.dumps(
                {
                    "window": f"{args.year}-{args.end_year}",
                    "bbox": list(bbox),
                    "factor": args.factor,
                    "n_items": len(items),
                    "full_scenes": int(full_times.size),
                    "full_rejected_frac": float(1 - full_keep.mean()),
                    "cap_c": cap,
                    "min_offset_samples": floor,
                    "thresholds": rows,
                },
                indent=2,
            )
        )
        print(f"\nWritten to {args.out}")
    finally:
        client.close()
        cluster.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
