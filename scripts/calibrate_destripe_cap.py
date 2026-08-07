"""Calibrate the de-striping offset cap against a real AOI (issue #46).

``settings.destripe_max_offset_c`` ships as a provisional 15 C. The measured
offset spread at Pergamino is wide (std 11-13 C) because the monthly reference
absorbs day-to-day weather alongside the per-scene bias it is meant to isolate,
so a cap chosen from that std alone could discard a large share of good scenes.

This script loads a window once, computes the per-scene offsets once, then
sweeps candidate caps over the result and reports what each would reject. Run
it before treating the default as settled.

    uv run python scripts/calibrate_destripe_cap.py

Uses Planetary Computer per CLAUDE.md: Earth Search costs egress from a laptop.
"""

import argparse
import json
from pathlib import Path

import numpy as np

CANDIDATE_CAPS = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0]

# Pergamino, Argentina -- the AOI every de-striping run so far has used.
DEFAULT_BBOX = [-61.1, -34.4, -60.1, -33.4]


def _load_lst(bbox: tuple, year: int, end_year: int):
    """Load a QA-masked Celsius LST stack for the window."""
    import pystac_client  # noqa: PLC0415
    import structlog  # noqa: PLC0415

    from landsat_lst.azure_auth import enable_pc_azure_refresh  # noqa: PLC0415
    from landsat_lst.config import settings  # noqa: PLC0415
    from landsat_lst.pipeline import load_scenes  # noqa: PLC0415
    from landsat_lst.qa import apply_qa_mask, convert_to_celsius  # noqa: PLC0415

    log = structlog.get_logger()
    catalog = pystac_client.Client.open(settings.stac_url)
    items = list(
        catalog.search(
            collections=[settings.collection],
            bbox=bbox,
            datetime=f"{year}-01-01/{end_year}-12-31",
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
    data = load_scenes(items, bbox, patch_url=patch_url, fail_on_error=False)
    return convert_to_celsius(apply_qa_mask(data)["lwir11"])


def _sweep(ov, nv, min_scene_pixels: int, log) -> list[dict]:
    """Report what each candidate cap would reject."""
    sweep = []
    for cap in CANDIDATE_CAPS:
        keep = np.isfinite(ov) & (nv >= min_scene_pixels) & (np.abs(ov) <= cap)
        row = {
            "cap_c": cap,
            "kept": int(keep.sum()),
            "rejected": int((~keep).sum()),
            "rejected_frac": round(float(1.0 - keep.mean()), 4),
        }
        sweep.append(row)
        log.info("cap_sweep", **row)
    return sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    parser.add_argument("--label", type=str, default="S30W065")
    parser.add_argument(
        "--out", type=Path, default=Path("results/decision/destripe_cap_calibration.json")
    )
    args = parser.parse_args()

    import structlog  # noqa: PLC0415
    from dask.distributed import Client, LocalCluster  # noqa: PLC0415

    from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings  # noqa: PLC0415
    from landsat_lst.normalization import scene_offsets  # noqa: PLC0415

    settings.stac_url = STAC_PLANETARY_COMPUTER
    log = structlog.get_logger()

    window = f"{args.year}-{args.end_year}"
    bbox = tuple(args.bbox)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cluster = LocalCluster(
        n_workers=settings.dask_workers,
        threads_per_worker=settings.dask_threads_per_worker,
        memory_limit=settings.dask_memory_limit,
        dashboard_address=":8787",
    )
    client = Client(cluster)
    log.info("dask_ready", window=window, chunk=settings.load_chunk_size)

    try:
        lst = _load_lst(bbox, args.year, args.end_year)

        log.info("computing_offsets")
        offset, n_valid = scene_offsets(lst)

        ov = np.asarray(offset.values, dtype="float64")
        nv = np.asarray(n_valid.values)
        finite = ov[np.isfinite(ov)]

        sparse = int((nv < settings.destripe_min_scene_pixels).sum())
        log.info(
            "offset_distribution",
            n_scenes=int(ov.size),
            n_sparse=sparse,
            std=round(float(finite.std()), 2),
            min=round(float(finite.min()), 2),
            max=round(float(finite.max()), 2),
            **{
                f"p{p}": round(float(np.percentile(finite, p)), 2)
                for p in (1, 5, 25, 50, 75, 95, 99)
            },
        )

        sweep = _sweep(ov, nv, settings.destripe_min_scene_pixels, log)

        payload = {
            "window": window,
            "label": args.label,
            "bbox": list(bbox),
            "n_scenes": int(ov.size),
            "n_sparse": sparse,
            "offsets": [round(float(v), 3) for v in ov],
            "sweep": sweep,
        }
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nCalibration written to {args.out}")
    finally:
        client.close()
        cluster.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
