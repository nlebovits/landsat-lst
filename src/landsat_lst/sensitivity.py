"""The pre-registered valid-area sensitivity check: 1/9, 5/9, 9/9.

Issue #120 locks 5 of 9 as the V1 default and requires one bounded check
against 1 and 9 before acceptance, on fixed crops, reporting coverage, P95
change, hotspot and rank stability, and visible artifacts. The thresholds are
declared here, in this order, and are not arguments a caller can tune:

    **Do not tune the threshold after looking.** If the arms disagree
    materially, the answer is to stop and report, not to adopt whichever crop
    looked best.

This module is the machinery, not the result. It computes nothing on import,
reaches no network, and holds no measured numbers. What it produces is a
:class:`SensitivityReport` from inputs a caller supplies -- normally a cached
:mod:`landsat_lst.fixture`, which is why the check can run on a laptop against
retained scenes rather than as a cloud run.

The three arms share one loaded stack and one offset estimate. That is the
point: the arms must differ in the valid-area rule and in nothing else, so a
difference between them cannot be a difference in scene set, in masking, or in
correction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from landsat_lst.aggregate import aggregate_to_output_grid
from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import xarray as xr

log = structlog.get_logger()

#: The arms, in the order the decision names them. The default sits between the
#: extremes deliberately: 1 accepts a single clear source cell as a delivered
#: observation, 9 demands a wholly clear block, and the question is whether the
#: product moves between them.
THRESHOLDS: tuple[int, ...] = (1, 5, 9)

#: Quantile defining a "hotspot" for the rank-stability report. The product is
#: a P95, so a hotspot is the hot tail of that percentile field.
HOTSPOT_QUANTILE = 0.95


@dataclass(frozen=True)
class ArmResult:
    """One threshold's answer over one crop."""

    min_valid_cells: int
    #: Delivered cells carrying a P95 at all, as a fraction of the crop.
    coverage: float
    #: Delivered solar-day observations that met the rule, summed over the crop.
    observations: int
    p95_mean: float | None
    p95_p95: float | None
    #: Cells in the hot tail of this arm's own P95 field.
    hotspot_cells: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArmComparison:
    """One arm measured against the 5/9 default, over the cells both resolve."""

    min_valid_cells: int
    #: Delivered cells this arm resolves that the default does not, and vice
    #: versa. Coverage is the first thing a threshold moves.
    cells_gained: int
    cells_lost: int
    #: P95 difference over the cells both arms resolve.
    shared_cells: int
    mean_abs_delta_c: float | None
    max_abs_delta_c: float | None
    #: Spearman rank correlation of the P95 field over those shared cells. Rank
    #: preservation is what a heat-ranking application actually depends on, and
    #: it can hold while absolute values move.
    rank_correlation: float | None
    #: Share of the default's hotspot cells this arm also calls a hotspot.
    hotspot_agreement: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SensitivityReport:
    """Every arm over every crop, plus the machinery's own verdict.

    ``stable`` is a mechanical reading of the numbers against
    :data:`STABILITY_BOUNDS`, not a decision. The decision is a person's, and
    the report exists so that person sees the numbers rather than a verdict.
    """

    crops: list[str] = field(default_factory=list)
    arms: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    comparisons: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "thresholds": list(THRESHOLDS),
            "default": settings.min_valid_source_cells,
            "hotspot_quantile": HOTSPOT_QUANTILE,
            "crops": self.crops,
            "arms": self.arms,
            "comparisons": self.comparisons,
            "stability_bounds": STABILITY_BOUNDS,
            "stable": self.stable(),
            "notes": self.notes,
        }

    def stable(self) -> bool | None:
        """Whether every arm stayed inside the pre-registered bounds.

        ``None`` when there is nothing to judge. A ``False`` is an instruction
        to stop and report, never an instruction to pick another threshold.
        """
        rows = [row for rows in self.comparisons.values() for row in rows]
        if not rows:
            return None
        return all(
            (
                row["rank_correlation"] is None
                or row["rank_correlation"] >= STABILITY_BOUNDS["min_rank_correlation"]
            )
            and (
                row["hotspot_agreement"] is None
                or row["hotspot_agreement"] >= STABILITY_BOUNDS["min_hotspot_agreement"]
            )
            and (
                row["max_abs_delta_c"] is None
                or row["max_abs_delta_c"] <= STABILITY_BOUNDS["max_abs_delta_c"]
            )
            for row in rows
        )

    def write(self, path: Path) -> Path:
        """Persist the report as JSON. The evidence, not a summary of it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))
        return path


#: Pre-registered, and stated before any arm runs. Widened after looking is the
#: failure mode this constant exists to make visible in a diff.
STABILITY_BOUNDS: dict[str, float] = {
    "min_rank_correlation": 0.99,
    "min_hotspot_agreement": 0.90,
    "max_abs_delta_c": 1.0,
}


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman correlation, computed from ranks with no scipy dependency."""
    if a.size < 2:
        return None
    ranks_a = np.argsort(np.argsort(a)).astype(np.float64)
    ranks_b = np.argsort(np.argsort(b)).astype(np.float64)
    sd_a, sd_b = ranks_a.std(), ranks_b.std()
    if sd_a == 0 or sd_b == 0:
        return None
    return float(np.mean((ranks_a - ranks_a.mean()) * (ranks_b - ranks_b.mean())) / (sd_a * sd_b))


def _hotspot_mask(values: np.ndarray) -> np.ndarray:
    """The hot tail of a P95 field, as a boolean mask over finite cells."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=bool)
    cut = float(np.quantile(finite, HOTSPOT_QUANTILE))
    return np.isfinite(values) & (values >= cut)


def _arm(field_values: np.ndarray, observations: int, threshold: int) -> ArmResult:
    """Summarize one arm's delivered P95 field."""
    finite = field_values[np.isfinite(field_values)]
    return ArmResult(
        min_valid_cells=threshold,
        coverage=round(float(finite.size / field_values.size), 6) if field_values.size else 0.0,
        observations=int(observations),
        p95_mean=round(float(finite.mean()), 4) if finite.size else None,
        p95_p95=round(float(np.quantile(finite, 0.95)), 4) if finite.size else None,
        hotspot_cells=int(_hotspot_mask(field_values).sum()),
    )


def _compare(default: np.ndarray, other: np.ndarray, threshold: int) -> ArmComparison:
    """One arm against the 5/9 default, over the cells both resolve."""
    default_ok = np.isfinite(default)
    other_ok = np.isfinite(other)
    shared = default_ok & other_ok

    delta = other[shared] - default[shared]
    default_hot = _hotspot_mask(default)
    other_hot = _hotspot_mask(other)
    hot_total = int(default_hot.sum())

    return ArmComparison(
        min_valid_cells=threshold,
        cells_gained=int((other_ok & ~default_ok).sum()),
        cells_lost=int((default_ok & ~other_ok).sum()),
        shared_cells=int(shared.sum()),
        mean_abs_delta_c=round(float(np.abs(delta).mean()), 4) if delta.size else None,
        max_abs_delta_c=round(float(np.abs(delta).max()), 4) if delta.size else None,
        rank_correlation=(
            round(v, 6)
            if (v := _rank_correlation(default[shared], other[shared])) is not None
            else None
        ),
        hotspot_agreement=(
            round(float((default_hot & other_hot).sum() / hot_total), 6) if hot_total else None
        ),
    )


def run_threshold_sweep(
    lst: xr.DataArray,
    *,
    crop: str = "crop",
    thresholds: Sequence[int] = THRESHOLDS,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Aggregate one masked source stack under each threshold and compare.

    ``lst`` must already be QA-masked, scaled, clamped, and corrected -- the
    state :func:`~landsat_lst.aggregate.aggregate_to_output_grid` expects. It is
    aggregated once per threshold and reduced to a P95 the same way the product
    is, so the arms differ in the valid-area rule and in nothing else.

    Args:
        lst: Celsius source-grid stack, dims ``(time, row, column)``.
        crop: Name this crop appears under in the report.
        thresholds: Arms to run. Defaults to the pre-registered three.

    Returns:
        ``(arms, comparisons)``, each keyed by crop name.
    """
    from landsat_lst.kernels import nanquantile_last  # noqa: PLC0415

    fields: dict[int, np.ndarray] = {}
    arms: list[dict[str, Any]] = []

    for threshold in thresholds:
        aggregated = aggregate_to_output_grid(lst, min_valid_cells=threshold)
        values = np.asarray(aggregated.compute().values, dtype=np.float64)
        # Same statistic the product publishes: a pooled percentile over the
        # delivered observations, never a percentile of source cells.
        p95 = nanquantile_last(np.moveaxis(values, 0, -1), 0.95)
        observations = int(np.isfinite(values).sum())
        fields[threshold] = p95
        arms.append(_arm(p95, observations, threshold).as_dict())
        log.info(
            "sensitivity_arm",
            crop=crop,
            min_valid_cells=threshold,
            coverage=arms[-1]["coverage"],
            observations=observations,
        )

    default = settings.min_valid_source_cells
    comparisons: list[dict[str, Any]] = []
    if default in fields:
        for threshold, values in fields.items():
            if threshold != default:
                comparisons.append(_compare(fields[default], values, threshold).as_dict())

    return {crop: arms}, {crop: comparisons}
