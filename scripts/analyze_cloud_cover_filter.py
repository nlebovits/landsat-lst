"""Price a scene-level cloud-cover filter in the evidence it destroys (issue #81).

``settings.max_cloud_cover`` is 100, so no scene is ever filtered out and the
pipeline relies on pixel-level QA. Raising the filter would cut scene count, and
scene count is the linear term in every phase, so the saving is proportional
across the whole tile rather than confined to the offset pass.

The question is what it costs. ``eo:cloud_cover`` describes a whole Landsat
footprint, roughly 185 km across, while a tile sees only part of one. A scene
reported at 90% cloud can therefore be clear over the AOI, or entirely clouded,
and the STAC property alone cannot tell the two apart. What settles it is the
valid-pixel count the QA mask actually produces for that scene, which
``scripts/validate_offset_subsampling.py`` already computes and now records
alongside each scene's cloud cover.

So this reads that file rather than loading anything. Every number below is
measured over the same scene set, in the same session, on the native grid.

Two quantities per candidate threshold:

* **What it saves**: the fraction of scenes it never reads. Scene count is
  linear in every phase, so this is the proportional I/O cut.
* **What it costs**: the fraction of surviving valid observations it discards.
  Those observations are the evidence behind the P95 and behind ``qa_count``.
  Scenes de-striping rejects anyway are free to drop and are excluded from the
  cost, which is the "removes scenes de-striping would have rejected anyway"
  case stated in the issue.

    uv run python scripts/analyze_cloud_cover_filter.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_INPUT = Path("results/decision/offset_subsampling_validation_f124.json")
THRESHOLDS = (30, 40, 50, 60, 70, 80, 90, 95, 100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/decision/cloud_cover_filter_analysis.json"),
    )
    parser.add_argument("--thresholds", type=int, nargs="+", default=list(THRESHOLDS))
    args = parser.parse_args()

    record = json.loads(args.input.read_text())
    cover = np.array(record["scene_cloud_cover"], dtype="float64")

    # The native pass, because the composite is built on the native stack and
    # the sparse floor is stated in native pixels.
    native = next(r for r in record["runs"] if r["factor"] == 1)
    offset = np.array(native["offsets"], dtype="float64")
    n_valid = np.array(native["n_valid"], dtype="float64")

    cap = record["cap_c"]
    floor = record["min_scene_pixels"]

    # The production rule, imported rather than restated. A second copy of it
    # here would drift from the one the pipeline applies.
    import xarray as xr  # noqa: PLC0415

    from landsat_lst.normalization import scene_keep_mask  # noqa: PLC0415

    keep = scene_keep_mask(
        xr.DataArray(offset, dims=["time"]),
        xr.DataArray(n_valid, dims=["time"]),
        max_offset_c=cap,
        floor=floor,
    ).values

    total_obs = float(n_valid[keep].sum())
    n_scenes = cover.size

    print(f"\n{args.input}")
    print(f"{n_scenes} scenes, {int(keep.sum())} kept by de-striping ", end="")
    print(f"({100 * (1 - keep.mean()):.1f}% rejected), cap {cap} C, floor {floor} px")
    print(f"valid observations behind the composite: {total_obs / 1e6:.1f}M pixel-scenes\n")

    header = (
        f"{'thresh':>7} {'scenes cut':>11} {'I/O saved':>10} "
        f"{'of those, already rejected':>27} {'obs lost':>10} {'obs lost %':>11}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for t in args.thresholds:
        dropped = cover >= t
        # A scene de-striping already discards costs nothing to skip earlier.
        wasted = dropped & ~keep
        costly = dropped & keep
        obs_lost = float(n_valid[costly].sum())
        row = {
            "threshold": t,
            "scenes_dropped": int(dropped.sum()),
            "scenes_dropped_frac": float(dropped.mean()),
            "dropped_already_rejected": int(wasted.sum()),
            "dropped_already_rejected_frac": float(wasted.sum() / max(1, dropped.sum())),
            "obs_lost": obs_lost,
            "obs_lost_frac": obs_lost / total_obs if total_obs else 0.0,
        }
        rows.append(row)
        print(
            f"{t:>7} {row['scenes_dropped']:>11} {100 * row['scenes_dropped_frac']:>9.1f}% "
            f"{100 * row['dropped_already_rejected_frac']:>26.0f}% "
            f"{obs_lost / 1e6:>9.1f}M {100 * row['obs_lost_frac']:>10.1f}%"
        )

    # The scene-level property is only a useful proxy if it predicts what the
    # QA mask does over this AOI. A weak relationship means a threshold cuts
    # good scenes and bad ones alike.
    valid_frac = n_valid / n_valid.max()
    both = np.isfinite(cover) & np.isfinite(valid_frac)
    corr = float(np.corrcoef(cover[both], valid_frac[both])[0, 1])
    print(f"\ncorr(eo:cloud_cover, valid-pixel fraction) = {corr:+.3f}")

    # The scenes a threshold cannot see: heavily clouded by the footprint
    # statistic, but well covered here.
    for t in (70, 90):
        clear_but_cloudy = (cover >= t) & keep & (valid_frac > 0.5)
        print(
            f"  scenes >= {t}% cloud that de-striping keeps and that cover "
            f">50% of the AOI: {int(clear_but_cloudy.sum())}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": str(args.input),
                "bbox": record["bbox"],
                "window": record["window"],
                "n_scenes": n_scenes,
                "n_kept_by_destripe": int(keep.sum()),
                "cap_c": cap,
                "min_scene_pixels": floor,
                "total_valid_obs": total_obs,
                "cloud_cover_vs_valid_fraction_corr": corr,
                "thresholds": rows,
            },
            indent=2,
        )
    )
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
