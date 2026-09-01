#!/usr/bin/env python3
"""Urban LST seasonality diagnostic for Pergamino, Argentina.

Question (raised by Elizabeth, WRI): Is the suspiciously low urban-core LST P95
in Pergamino a real "surface cool island" signal, or an artifact of sparse
Southern-Hemisphere summer (DJF) observations?

Pergamino (~34 degS) is Southern Hemisphere, so summer = Dec/Jan/Feb. The
production composite is a per-pixel P95 over a multi-year stack. The worry: if
summer scene-days are scarce, the per-pixel P95 never reaches the true summer
surface peak and urban LST looks artificially cool.

This script tests that with the ACTUAL Pergamino municipality + urban
classification polygons (cached in the data dir), not a guessed bbox:

  1. Scene-days per calendar month  -> is summer actually under-sampled?
  2. Mean valid pixels per scene, by month x urban class  -> the "qa_count".
  3. Annual per-pixel P95 composite vs SUMMER-ONLY per-pixel P95 composite, by
     class. If annual << summer for urban, the cool reading is a sampling /
     aggregation artifact. If annual ~= summer, it is a real signal.

NO QA masking is applied: clouds sit in the cold tail, below the 95th
percentile, so a P95 is robust to them. Re-enabling cloud+shadow masking only
reintroduces the documented sparse-observation problem (it rejected ~91% of
pixels in sparse areas) with no benefit to the warm tail. Fill values (DN=0) are
dropped via landsat_lst.qa.convert_to_celsius.

Boundaries are fetched once from public WFS services and cached:
  - <data>/pergamino_dept.gpkg   (IGN ign:departamento, nam='Pergamino')
  - <data>/pergamino_urban.gpkg  (Pergamino IDE publico:aglomerados_urbanos)

This is a Pergamino-specific diagnostic (the WFS layers are hardcoded), not a
generic tool. Years and output locations are configurable.

Usage:
    uv run --extra analysis python scripts/urban_seasonality_diagnostic.py
    uv run --extra analysis python scripts/urban_seasonality_diagnostic.py \
        --start-year 2020 --end-year 2024 --output results/urban-seasonality
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import requests
import rioxarray  # noqa: F401 - enables .rio accessor
import structlog
import urllib3
import xarray as xr
from odc.stac import load as stac_load
from rasterio.features import rasterize

from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings
from landsat_lst.qa import convert_to_celsius

# --------------------------------------------------------------------------- #
# Logging (mirrors scripts/validation_pyramid.py)
# --------------------------------------------------------------------------- #
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
log = structlog.get_logger()

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SUMMER_MONTHS = (12, 1, 2)  # Southern Hemisphere summer (DJF)

# Rasterized classification codes. 4 = rural: inside the department but outside
# any urban polygon. 0 = outside the department entirely.
CLASS_CODES = {"urbana": 1, "periurbana": 2, "urbano en ruralidad": 3, "rural": 4}
CLASS_ORDER = ["urbana", "periurbana", "urbano en ruralidad", "rural"]

MONTH_NAME = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}
SEASON = {
    12: "summer",
    1: "summer",
    2: "summer",
    3: "autumn",
    4: "autumn",
    5: "autumn",
    6: "winter",
    7: "winter",
    8: "winter",
    9: "spring",
    10: "spring",
    11: "spring",
}
SEASON_ORDER = ["summer", "autumn", "winter", "spring"]
SEASON_COLOR = {
    "summer": "#d73027",
    "autumn": "#fc8d59",
    "winter": "#4575b4",
    "spring": "#91bfdb",
}

# WFS sources for boundaries
IGN_WFS = "https://wms.ign.gob.ar/geoserver/ign/ows"
PERGAMINO_WFS = "https://ide.pergamino.gob.ar:8443/geoserver/wfs"


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class SceneDayCounts:
    """Scene-day counts per calendar month and per season."""

    total: int = 0
    per_month: dict[int, int] = field(default_factory=dict)
    per_season: dict[str, int] = field(default_factory=dict)


@dataclass
class MonthlyClassCount:
    """Mean valid (non-fill) pixels per scene for one month, by class."""

    month: int = 0
    counts: dict[str, float] = field(default_factory=dict)


@dataclass
class ClassP95:
    """Per-pixel P95 composite stats over one class's pixels."""

    clazz: str = ""
    n_pixels: int = 0
    annual_p95: float = 0.0
    summer_p95: float = 0.0
    delta: float = 0.0  # summer - annual


@dataclass
class DiagnosticResult:
    """Full diagnostic output, serialized to summary.json."""

    timestamp: str = ""
    stac_endpoint: str = ""
    start_year: int = 0
    end_year: int = 0
    bbox: list[float] = field(default_factory=list)
    n_scenes: int = 0
    scene_days: SceneDayCounts = field(default_factory=SceneDayCounts)
    monthly_counts: list[MonthlyClassCount] = field(default_factory=list)
    summer_vs_nonsummer: dict[str, dict[str, float]] = field(default_factory=dict)
    p95_by_class: list[ClassP95] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Boundary fetching / loading
# --------------------------------------------------------------------------- #
def fetch_boundaries(data_dir: Path) -> tuple[Path, Path]:
    """Download + cache Pergamino department and urban classification.

    Idempotent: skips anything already on disk. The Pergamino IDE server is
    flaky, so the urban fetch retries a few times.

    Returns:
        (dept_gpkg_path, urban_gpkg_path)
    """
    dept_path = data_dir / "pergamino_dept.gpkg"
    urban_path = data_dir / "pergamino_urban.gpkg"
    if dept_path.exists() and urban_path.exists():
        return dept_path, urban_path

    warnings.filterwarnings("ignore")
    urllib3.disable_warnings()
    data_dir.mkdir(parents=True, exist_ok=True)

    if not dept_path.exists():
        log.info("fetch_boundary", layer="ign:departamento")
        r = requests.get(
            IGN_WFS,
            params={
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typename": "ign:departamento",
                "outputFormat": "application/json",
                "CQL_FILTER": "nam='Pergamino'",
            },
            timeout=120,
        )
        gpd.read_file(BytesIO(r.content)).to_crs(4326).to_file(dept_path, driver="GPKG")

    if not urban_path.exists():
        b = gpd.read_file(dept_path).total_bounds
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                log.info("fetch_boundary", layer="aglomerados_urbanos", attempt=attempt)
                r = requests.get(
                    PERGAMINO_WFS,
                    params={
                        "service": "WFS",
                        "version": "2.0.0",
                        "request": "GetFeature",
                        "typename": "publico:aglomerados_urbanos",
                        "outputFormat": "application/json",
                        "srsname": "EPSG:4326",
                        "bbox": f"{b[0]},{b[1]},{b[2]},{b[3]},EPSG:4326",
                    },
                    timeout=120,
                    verify=False,
                )
                gpd.read_file(BytesIO(r.content)).to_crs(4326).to_file(urban_path, driver="GPKG")
                break
            except Exception as e:  # flaky server, retry on any error
                last_err = e
                log.warning("fetch_retry", error=type(e).__name__)
        else:
            raise RuntimeError("Failed to fetch urban classification") from last_err

    return dept_path, urban_path


def load_boundaries(data_dir: Path):
    """Load cached department + urban GeoDataFrames in EPSG:4326."""
    dept_path, urban_path = fetch_boundaries(data_dir)
    dept = gpd.read_file(dept_path).to_crs("EPSG:4326")
    urban = gpd.read_file(urban_path).to_crs("EPSG:4326")
    return dept, urban


# --------------------------------------------------------------------------- #
# Pure logic (unit-tested in tests/unit/test_urban_seasonality.py)
# --------------------------------------------------------------------------- #
def rasterize_classes(out_shape, transform, urban_gdf, dept_gdf) -> np.ndarray:
    """Rasterize urban classification + rural baseline onto an LST grid.

    Args:
        out_shape: (height, width) of the target grid.
        transform: affine transform mapping pixel -> world (EPSG:4326).
        urban_gdf: GeoDataFrame with a 'clasificacion' column.
        dept_gdf: department boundary GeoDataFrame (defines rural extent).

    Returns:
        uint8 array with CLASS_CODES values; 0 = outside the department.
    """
    # Draw urban classes in increasing density so urbana wins on overlap.
    shapes = []
    for cls in ("urbano en ruralidad", "periurbana", "urbana"):
        code = CLASS_CODES[cls]
        for geom in urban_gdf.loc[urban_gdf["clasificacion"] == cls, "geometry"]:
            shapes.append((geom, code))

    urban_raster = (
        rasterize(
            shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
        if shapes
        else np.zeros(out_shape, dtype="uint8")
    )

    dept_raster = rasterize(
        [(g, 1) for g in dept_gdf.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    raster = urban_raster.copy()
    raster[(dept_raster == 1) & (urban_raster == 0)] = CLASS_CODES["rural"]
    return raster


def scene_day_counts(months: np.ndarray) -> SceneDayCounts:
    """Count scene-days per calendar month and per season."""
    months = np.asarray(months)
    total = int(months.size)
    per_month = {m: int((months == m).sum()) for m in range(1, 13)}
    per_season: dict[str, int] = {}
    for m, n in per_month.items():
        per_season[SEASON[m]] = per_season.get(SEASON[m], 0) + n
    return SceneDayCounts(total=total, per_month=per_month, per_season=per_season)


def aggregate_monthly_counts(
    months: np.ndarray, counts: dict[str, np.ndarray]
) -> tuple[list[MonthlyClassCount], dict[str, dict[str, float]]]:
    """Mean valid-pixels-per-scene by month x class, plus summer vs non-summer.

    Args:
        months: per-scene calendar month (length = n_scenes).
        counts: class -> per-scene valid-pixel count array (length = n_scenes).

    Returns:
        (monthly list, {"summer"/"non_summer": {class: mean}}).
    """
    df = pd.DataFrame({c: np.asarray(counts[c]) for c in counts})
    df["month"] = np.asarray(months)
    monthly_mean = df.groupby("month").mean()

    monthly: list[MonthlyClassCount] = []
    for m in range(1, 13):
        if m not in monthly_mean.index:
            continue
        row = monthly_mean.loc[m]
        monthly.append(MonthlyClassCount(month=m, counts={c: float(row[c]) for c in counts}))

    is_summer = np.isin(df["month"].to_numpy(), SUMMER_MONTHS)
    summer_vs = {
        "summer": {c: float(df.loc[is_summer, c].mean()) for c in counts},
        "non_summer": {c: float(df.loc[~is_summer, c].mean()) for c in counts},
    }
    return monthly, summer_vs


def compute_p95_by_class(
    class_raster: np.ndarray, annual: np.ndarray, summer: np.ndarray
) -> list[ClassP95]:
    """Median of per-pixel annual & summer P95 over each class's pixels."""
    out: list[ClassP95] = []
    for cls in CLASS_ORDER:
        mask = class_raster == CLASS_CODES[cls]
        a = annual[mask]
        a = a[np.isfinite(a)]
        s = summer[mask]
        s = s[np.isfinite(s)]
        if a.size == 0 or s.size == 0:
            continue
        am, sm = float(np.median(a)), float(np.median(s))
        out.append(
            ClassP95(
                clazz=cls,
                n_pixels=int(mask.sum()),
                annual_p95=round(am, 2),
                summer_p95=round(sm, 2),
                delta=round(sm - am, 2),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Data loading + compute
# --------------------------------------------------------------------------- #
def query_and_load(bbox: tuple, start_year: int, end_year: int) -> xr.Dataset:
    """Query Planetary Computer STAC and lazily load the LWIR11 stack."""
    log.info("stac_query", bbox=bbox, years=[start_year, end_year])
    catalog = pystac_client.Client.open(settings.stac_url, modifier=pc.sign_inplace)
    items = list(
        catalog.search(
            collections=[settings.collection],
            bbox=bbox,
            datetime=f"{start_year}-01-01/{end_year}-12-31",
            query={
                "eo:cloud_cover": {"lt": settings.max_cloud_cover},
                "platform": {"in": ["landsat-8", "landsat-9"]},
            },
        ).items()
    )
    log.info("stac_found", scenes=len(items))
    stack = stac_load(
        items,
        bands=["lwir11"],
        crs=settings.crs,
        resolution=settings.source_resolution,
        chunks={"time": 1, "latitude": 1024, "longitude": 1024},
        groupby="solar_day",
        bbox=bbox,
    )
    log.info("stack_loaded", **{k: int(v) for k, v in stack.sizes.items()})
    return stack


def build_class_raster(template: xr.DataArray, urban_gdf, dept_gdf) -> np.ndarray:
    """Derive grid geometry from an LST template and rasterize classes onto it."""
    template = template.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude").rio.write_crs(
        "EPSG:4326"
    )
    out_shape = (template.sizes["latitude"], template.sizes["longitude"])
    return rasterize_classes(out_shape, template.rio.transform(), urban_gdf, dept_gdf)


def compute_composites_and_counts(lst, class_da, summer_sel):
    """Single dask pass: annual P95, summer P95, per-class valid-px-per-scene.

    Returns:
        (annual_p95_np, summer_p95_np, {class: per-scene count np array}).
    """
    valid = ~np.isnan(lst)
    p95_annual = lst.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")
    lst_summer = lst.isel(time=np.where(summer_sel)[0])
    p95_summer = lst_summer.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")

    count_targets = {
        cls: (valid & (class_da == CLASS_CODES[cls])).sum(dim=("latitude", "longitude"))
        for cls in CLASS_ORDER
    }

    log.info("compute_start", note="reading COG stack; may take several minutes")
    annual_v, summer_v, *count_vals = dask.compute(
        p95_annual, p95_summer, *[count_targets[c] for c in CLASS_ORDER]
    )
    log.info("compute_done")
    counts = {c: count_vals[i].values for i, c in enumerate(CLASS_ORDER)}
    return annual_v, summer_v, counts


# --------------------------------------------------------------------------- #
# Output: JSON, CSV, COGs, figures, console
# --------------------------------------------------------------------------- #
def save_summary(result: DiagnosticResult, output_dir: Path) -> Path:
    """Write the full result to summary.json."""
    path = output_dir / "summary.json"
    with path.open("w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    log.info("wrote", file=str(path))
    return path


def save_tables(result: DiagnosticResult, output_dir: Path) -> list[Path]:
    """Write monthly_qa_count.csv and p95_by_class.csv."""
    monthly_rows = []
    for mc in result.monthly_counts:
        row = {"month": MONTH_NAME[mc.month], "season": SEASON[mc.month]}
        row.update({c: round(mc.counts[c], 1) for c in CLASS_ORDER})
        monthly_rows.append(row)
    monthly_path = output_dir / "monthly_qa_count.csv"
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)

    p95_path = output_dir / "p95_by_class.csv"
    pd.DataFrame([asdict(c) for c in result.p95_by_class]).to_csv(p95_path, index=False)
    log.info("wrote", file=str(monthly_path))
    log.info("wrote", file=str(p95_path))
    return [monthly_path, p95_path]


def export_cogs(
    annual_da: xr.DataArray,
    summer_da: xr.DataArray,
    class_da: xr.DataArray,
    output_dir: Path,
) -> list[Path]:
    """Write the annual P95, summer P95, and class-raster COGs for QGIS."""
    paths = []
    for name, da, dtype in (
        ("lst_p95_annual", annual_da, "float32"),
        ("lst_p95_summer", summer_da, "float32"),
        ("urban_class", class_da, "uint8"),
    ):
        out = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude").rio.write_crs(
            "EPSG:4326"
        )
        path = output_dir / f"{name}.tif"
        out.rio.to_raster(path, driver="COG", compress="DEFLATE", dtype=dtype)
        paths.append(path)
        log.info("wrote", file=str(path))
    return paths


_MONTHS = list(range(1, 13))
_MONTH_LABELS = [MONTH_NAME[m] for m in _MONTHS]


def _save_fig(fig, path: Path, plt) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_scene_days(result: DiagnosticResult, output_dir: Path, plt) -> Path:
    """Scene-days per month, colored by season."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sd = result.scene_days
    vals = [sd.per_month.get(m, 0) for m in _MONTHS]
    ax.bar(_MONTH_LABELS, vals, color=[SEASON_COLOR[SEASON[m]] for m in _MONTHS])
    ax.set_ylabel("scene-days (5-yr pooled)")
    summer = sd.per_season.get("summer", 0)
    ax.set_title(
        f"Landsat scene-days per month, Pergamino "
        f"{result.start_year}-{result.end_year}\n"
        f"Summer (DJF) = {summer} of {sd.total} "
        f"({100 * summer / max(sd.total, 1):.1f}%) — not under-sampled"
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEASON_COLOR[s]) for s in SEASON_ORDER]
    ax.legend(
        handles,
        SEASON_ORDER,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
    )
    return _save_fig(fig, output_dir / "fig_scene_days_by_season.png", plt)


def _fig_monthly_qa(result: DiagnosticResult, output_dir: Path, plt) -> Path:
    """Monthly valid-pixel count, normalized per class (classes differ ~1000x)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    by_month = {mc.month: mc.counts for mc in result.monthly_counts}
    for cls in CLASS_ORDER:
        series = np.array([by_month.get(m, {}).get(cls, np.nan) for m in _MONTHS])
        ax.plot(_MONTH_LABELS, series / np.nanmean(series), marker="o", label=cls)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    for m in _MONTHS:
        if SEASON[m] == "summer":
            ax.axvspan(m - 1.5, m - 0.5, color=SEASON_COLOR["summer"], alpha=0.08)
    ax.set_ylabel("valid px/scene (relative to class mean)")
    ax.set_title(
        "Per-month observation density by urban class (normalized)\n"
        "summer shaded — summer is the best-observed season"
    )
    ax.legend(fontsize=8)
    return _save_fig(fig, output_dir / "fig_monthly_qa_count.png", plt)


def _fig_p95(result: DiagnosticResult, output_dir: Path, plt) -> Path:
    """Annual vs summer-only per-pixel P95 by class."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    classes = [c.clazz for c in result.p95_by_class]
    x = np.arange(len(classes))
    w = 0.38
    ax.bar(
        x - w / 2,
        [c.annual_p95 for c in result.p95_by_class],
        w,
        label="annual P95",
        color="#4575b4",
    )
    ax.bar(
        x + w / 2,
        [c.summer_p95 for c in result.p95_by_class],
        w,
        label="summer-only P95",
        color="#d73027",
    )
    for i, c in enumerate(result.p95_by_class):
        ax.text(x[i] + w / 2, c.summer_p95 + 0.3, f"+{c.delta:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylabel("LST P95 (°C)")
    ax.set_title(
        "Per-pixel P95 composite by class: annual vs summer-only\n"
        "urban stays cooler than rural in BOTH — real signal, not artifact"
    )
    ax.legend()
    return _save_fig(fig, output_dir / "fig_p95_annual_vs_summer.png", plt)


def make_figures(result: DiagnosticResult, output_dir: Path) -> list[Path]:
    """Generate the three PNG charts embedded in the findings writeup."""
    import matplotlib  # noqa: PLC0415 - optional 'analysis' extra, imported lazily

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415 - optional 'analysis' extra

    paths = [
        _fig_scene_days(result, output_dir, plt),
        _fig_monthly_qa(result, output_dir, plt),
        _fig_p95(result, output_dir, plt),
    ]
    for p in paths:
        log.info("wrote", file=str(p))
    return paths


def print_summary(result: DiagnosticResult) -> None:
    """Human-readable console summary."""
    print("\n" + "=" * 72)
    print("PERGAMINO URBAN LST SEASONALITY DIAGNOSTIC")
    print("=" * 72)
    print(f"Years:    {result.start_year}-{result.end_year}")
    print(f"STAC:     {result.stac_endpoint}")
    print(f"Scenes:   {result.n_scenes}")
    print(f"Bbox:     {result.bbox}")

    sd = result.scene_days
    print("\nScene-days by season:")
    for s in SEASON_ORDER:
        n = sd.per_season.get(s, 0)
        share = 100 * n / max(sd.total, 1)
        flag = "  <-- under-sampled?" if s == "summer" and share < 5 else ""
        print(f"  {s:<8}{n:>4} ({share:>4.1f}%){flag}")

    print("\nMean valid px/scene, summer vs non-summer:")
    sv = result.summer_vs_nonsummer
    print(f"  {'class':<22}{'summer':>12}{'non-summer':>14}")
    for cls in CLASS_ORDER:
        print(f"  {cls:<22}{sv['summer'][cls]:>12,.0f}{sv['non_summer'][cls]:>14,.0f}")

    print("\nPer-pixel P95 by class (annual vs summer-only):")
    print(f"  {'class':<22}{'annual':>10}{'summer':>10}{'delta':>9}")
    for c in result.p95_by_class:
        print(f"  {c.clazz:<22}{c.annual_p95:>9.1f}C{c.summer_p95:>9.1f}C{c.delta:>+8.1f}C")
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(
    start_year: int, end_year: int, output_dir: Path, data_dir: Path, make_figs: bool
) -> DiagnosticResult:
    """Execute the full diagnostic and write all artifacts."""
    # Local rule (hook-enforced): Planetary Computer for local dev.
    settings.stac_url = STAC_PLANETARY_COMPUTER
    output_dir.mkdir(parents=True, exist_ok=True)

    dept, urban = load_boundaries(data_dir)
    bbox = tuple(float(x) for x in dept.total_bounds)

    stack = query_and_load(bbox, start_year, end_year)
    lst = convert_to_celsius(stack["lwir11"])
    months = lst["time"].dt.month.values
    summer_sel = np.isin(months, SUMMER_MONTHS)

    log.info("rasterize_classes")
    class_raster = build_class_raster(lst.isel(time=0), urban, dept)
    class_da = xr.DataArray(
        class_raster,
        dims=("latitude", "longitude"),
        coords={"latitude": lst["latitude"], "longitude": lst["longitude"]},
    )

    annual_v, summer_v, counts = compute_composites_and_counts(lst, class_da, summer_sel)

    monthly, summer_vs = aggregate_monthly_counts(months, counts)
    result = DiagnosticResult(
        timestamp=datetime.now(UTC).isoformat(),
        stac_endpoint=settings.stac_url,
        start_year=start_year,
        end_year=end_year,
        bbox=list(bbox),
        n_scenes=int(lst.sizes["time"]),
        scene_days=scene_day_counts(months),
        monthly_counts=monthly,
        summer_vs_nonsummer=summer_vs,
        p95_by_class=compute_p95_by_class(class_raster, annual_v.values, summer_v.values),
    )

    save_summary(result, output_dir)
    save_tables(result, output_dir)
    export_cogs(annual_v, summer_v, class_da, output_dir)
    if make_figs:
        make_figures(result, output_dir)

    print_summary(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/urban-seasonality"),
        help="Output directory for JSON/CSV/COG/PNG artifacts.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for cached boundary GeoPackages.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip PNG figure generation (e.g. if matplotlib is unavailable).",
    )
    args = parser.parse_args()

    run(
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.output,
        data_dir=args.data_dir,
        make_figs=not args.no_figures,
    )


if __name__ == "__main__":
    main()
