"""Validate that coarse-grid per-scene offsets match full-resolution ones (issue #46).

De-striping needs one scalar per scene, but currently derives it from a
full-resolution per-pixel monthly climatology. That costs a second full read of
the stack (~9.5 min for a 1 degree AOI, 390 scenes). The source COGs carry
internal overviews at [2,4,8,16,32,64], so the same scalar can be estimated
from a coarse read that fetches ~factor**2 fewer bytes.

This script measures whether the cheap estimate is the same number.

Design notes, because the result is worthless if either is wrong:

* The comparison is **paired**. Every coarse stack is compared scene-by-scene
  against a full-resolution reference computed in the same session, not against
  a previous run. Offsets for all factors go into one ``dask.compute`` call so
  the graphs can share work.
* The time axes are **asserted equal** before comparing. All loads use the same
  items and ``groupby="solar_day"``, so they must line up; a silent misalignment
  would let xarray broadcast and quietly invent agreement.
* The reference is **cross-checked** against the committed calibration output
  (``results/decision/destripe_cap_calibration.json``). If the factor=1 pass
  cannot reproduce the offsets behind the shipped 15 C cap, nothing downstream
  is comparable to it.

    uv run python scripts/validate_offset_subsampling.py

Planetary Computer per CLAUDE.md: Earth Search costs egress from a laptop.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Powers of two so GDAL lands on a stored overview level rather than
# resampling between two of them.
FACTORS = [1, 2, 4, 8, 16, 32, 64]

DEFAULT_BBOX = [-61.1, -34.4, -60.1, -33.4]
REFERENCE_JSON = Path("results/decision/destripe_cap_calibration.json")

# Fixed before seeing results. An offset error lands directly in the P95 of
# every pixel its scene touches, and the urban contrasts this product exists to
# show are a few degrees, so the tolerance is tight.
ACCEPT = {
    "median_abs": 0.05,
    "p99_abs": 0.25,
    "max_abs": 0.5,
    "bias": 0.05,
    "flip_frac": 0.01,
    "flip_margin_c": 1.0,
}


@dataclass
class FactorRun:
    """One resolution factor's offsets and how they compare to the reference."""

    factor: int
    shape: tuple[int, int] = (0, 0)
    n_pixels: int = 0
    bytes_proxy: int = 0
    load_secs: float = 0.0
    offsets: list[float] = field(default_factory=list)
    n_valid: list[int] = field(default_factory=list)
    # Comparison against factor=1, filled in later.
    n_scored: int = 0
    n_admissible: int = 0
    median_abs: float = 0.0
    p99_abs: float = 0.0
    max_abs: float = 0.0
    bias: float = 0.0
    corr: float = 0.0
    slope: float = 0.0
    flips: int = 0
    flip_frac: float = 0.0
    worst_flip_margin_c: float = 0.0
    sparse_flips: int = 0
    passes: bool = False
    error: str | None = None


def _compare(
    ref: np.ndarray,
    cur: np.ndarray,
    cap: float,
    ref_ok: np.ndarray,
    cur_ok: np.ndarray,
) -> dict:
    """Score a coarse estimate against the full-resolution reference.

    Both metrics are conditioned on the sparse guards, because a scene neither
    path would use cannot affect the composite. Skipping that conditioning
    reads as failure where there is none: at Pergamino a scene with 2 valid
    native pixels showed a 17 C disagreement and registered as a decision flip,
    while both paths were in fact discarding it as too sparse to estimate.

    Flips are counted over scenes both guards admit, since the cap is what
    flips. Accuracy is measured over scenes both paths actually keep, since an
    offset that is never applied cannot be wrong in any way that matters.
    """
    both = np.isfinite(ref) & np.isfinite(cur)
    admissible = both & ref_ok & cur_ok

    # A flip near the cap is a coin toss either way; one far from it means the
    # estimate genuinely broke, which is why the margin is reported alongside.
    keep_ref = np.abs(ref) <= cap
    keep_cur = np.abs(cur) <= cap
    flipped = (keep_ref != keep_cur) & admissible
    margins = np.abs(np.abs(ref[flipped]) - cap) if flipped.any() else np.array([0.0])

    scored = admissible & keep_ref & keep_cur
    d = cur[scored] - ref[scored]
    a = np.abs(d)

    n = int(scored.sum())
    slope = float(np.polyfit(ref[scored], cur[scored], 1)[0]) if n > 2 else 0.0
    corr = float(np.corrcoef(ref[scored], cur[scored])[0, 1]) if n > 2 else 0.0

    return {
        "n_scored": n,
        "median_abs": float(np.median(a)) if n else 0.0,
        "p99_abs": float(np.percentile(a, 99)) if n else 0.0,
        "max_abs": float(a.max()) if n else 0.0,
        "bias": float(np.median(d)) if n else 0.0,
        "corr": corr,
        "slope": slope,
        "n_admissible": int(admissible.sum()),
        "flips": int(flipped.sum()),
        "flip_frac": float(flipped.sum() / max(1, admissible.sum())),
        "worst_flip_margin_c": float(margins.max()),
    }


def _verdict(r: FactorRun) -> bool:
    return (
        r.median_abs <= ACCEPT["median_abs"]
        and r.p99_abs <= ACCEPT["p99_abs"]
        and r.max_abs <= ACCEPT["max_abs"]
        and abs(r.bias) <= ACCEPT["bias"]
        and r.flip_frac <= ACCEPT["flip_frac"]
        and r.worst_flip_margin_c <= ACCEPT["flip_margin_c"]
    )


def _cross_check_reference(ref: np.ndarray, log) -> None:
    """Confirm factor=1 reproduces the committed calibration offsets."""
    if not REFERENCE_JSON.exists():
        log.warning("reference_json_missing", path=str(REFERENCE_JSON))
        return
    saved = np.array(json.loads(REFERENCE_JSON.read_text())["offsets"], dtype="float64")
    if saved.size != ref.size:
        log.error("reference_scene_count_mismatch", saved=saved.size, current=ref.size)
        return
    both = np.isfinite(saved) & np.isfinite(ref)
    delta = float(np.abs(saved[both] - ref[both]).max())
    log.info(
        "reference_cross_check",
        max_abs_delta=round(delta, 4),
        reproduces="yes" if delta < 0.01 else "NO -- not comparable to the shipped cap",
    )


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    parser.add_argument("--factors", type=int, nargs="+", default=FACTORS)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/decision/offset_subsampling_validation.json"),
    )
    args = parser.parse_args()

    import dask  # noqa: PLC0415
    import pystac_client  # noqa: PLC0415
    import structlog  # noqa: PLC0415
    from dask.distributed import Client, LocalCluster  # noqa: PLC0415

    from landsat_lst.azure_auth import enable_pc_azure_refresh  # noqa: PLC0415
    from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings  # noqa: PLC0415
    from landsat_lst.normalization import _spatial_dims, scene_offsets  # noqa: PLC0415
    from landsat_lst.pipeline import load_scenes  # noqa: PLC0415
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
        if not items:
            msg = "no scenes"
            raise ValueError(msg)
        patch_url = enable_pc_azure_refresh(items)

        runs: list[FactorRun] = []
        stacks = {}
        for f in args.factors:
            t0 = time.perf_counter()
            data = load_scenes(
                items, bbox, patch_url=patch_url, fail_on_error=False, resolution_factor=f
            )
            lst = convert_to_celsius(apply_qa_mask(data)["lwir11"])
            spatial = _spatial_dims(lst)
            shape = (lst.sizes[spatial[0]], lst.sizes[spatial[1]])
            npix = shape[0] * shape[1]
            runs.append(
                FactorRun(
                    factor=f,
                    shape=shape,
                    n_pixels=npix,
                    # uint16 source, two bands; same proxy benchmark_coiled.py uses.
                    bytes_proxy=npix * lst.sizes["time"] * 2 * 2,
                    load_secs=round(time.perf_counter() - t0, 2),
                )
            )
            stacks[f] = lst
            log.info("stack_built", factor=f, shape=shape, pixels=npix)

        # Every factor must describe the same scenes, or the pairing is a lie.
        base_time = stacks[args.factors[0]].time
        for f, lst in stacks.items():
            if not lst.time.equals(base_time):
                msg = f"time axis mismatch at factor {f}: {lst.sizes['time']} vs {base_time.size}"
                raise ValueError(msg)
        log.info("time_axes_aligned", n_scenes=int(base_time.size))

        log.info("computing_offsets", factors=args.factors)
        t0 = time.perf_counter()
        # One compute call so dask can share whatever the graphs have in common.
        results = dask.compute(*[scene_offsets(stacks[f]) for f in args.factors])
        log.info("offsets_computed", secs=round(time.perf_counter() - t0, 1))

        for run, (offset, n_valid) in zip(runs, results, strict=True):
            run.offsets = [float(v) for v in np.asarray(offset.values)]
            run.n_valid = [int(v) for v in np.asarray(n_valid.values)]

        ref_run = next(r for r in runs if r.factor == args.factors[0])
        ref = np.array(ref_run.offsets, dtype="float64")
        _cross_check_reference(ref, log)

        cap = settings.destripe_max_offset_c
        # Native path keeps a scene on min_scene_pixels; a coarse path keeps it
        # on min_offset_samples, stated on its own grid. These are the real
        # production rules, so the comparison uses them rather than a scaling.
        ref_ok = np.array(ref_run.n_valid) >= settings.destripe_min_scene_pixels
        for run in runs:
            cur = np.array(run.offsets, dtype="float64")
            if run.factor == args.factors[0]:
                run.passes = True
                log.info("factor_reference", factor=run.factor, n_kept=int(ref_ok.sum()))
                continue
            cur_ok = np.array(run.n_valid) >= settings.destripe_min_offset_samples
            for k, v in _compare(ref, cur, cap, ref_ok, cur_ok).items():
                setattr(run, k, v)
            run.sparse_flips = int((cur_ok != ref_ok).sum())
            run.passes = _verdict(run)
            log.info(
                "factor_scored",
                factor=run.factor,
                median_abs=round(run.median_abs, 4),
                p99_abs=round(run.p99_abs, 4),
                max_abs=round(run.max_abs, 3),
                bias=round(run.bias, 4),
                flips=run.flips,
                worst_flip_margin_c=round(run.worst_flip_margin_c, 2),
                sparse_flips=run.sparse_flips,
                passes=run.passes,
            )

        print(
            f"\n{'factor':>7} {'grid':>13} {'MB':>8} {'med|d|':>8} "
            f"{'p99|d|':>8} {'max|d|':>8} {'bias':>7} {'flips':>6} {'pass':>5}"
        )
        print("-" * 80)
        for r in runs:
            mb = r.bytes_proxy / 1e6
            grid = f"{r.shape[0]}x{r.shape[1]}"
            if r.factor == args.factors[0]:
                print(f"{r.factor:>7} {grid:>13} {mb:>8.0f} {'(reference)':>44}")
            else:
                print(
                    f"{r.factor:>7} {grid:>13} {mb:>8.0f} {r.median_abs:>8.4f} "
                    f"{r.p99_abs:>8.4f} {r.max_abs:>8.3f} {r.bias:>7.3f} "
                    f"{r.flips:>6} {'YES' if r.passes else 'no':>5}"
                )

        best = max((r.factor for r in runs if r.passes), default=1)
        print(f"\nLargest passing factor: {best}")
        print(f"Recommended (one step more conservative): {max(1, best // 2)}")

        args.out.write_text(
            json.dumps(
                {
                    "window": f"{args.year}-{args.end_year}",
                    "bbox": list(bbox),
                    "n_items": len(items),
                    "cap_c": cap,
                    "acceptance": ACCEPT,
                    "largest_passing_factor": best,
                    "recommended_factor": max(1, best // 2),
                    "runs": [asdict(r) for r in runs],
                },
                indent=2,
            )
        )
        print(f"Written to {args.out}")
    finally:
        client.close()
        cluster.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
