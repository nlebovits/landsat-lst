#!/usr/bin/env python3
"""Quantify permanent ASTER GED coverage gaps over GHSL urban areas.

Landsat Collection 2 Level-2 Surface Temperature is derived using emissivity
from the ASTER Global Emissivity Dataset (GED), built from clear-sky ASTER
scenes acquired 2000-2008. Where ASTER never caught clear sky, there is no
emissivity and therefore no Surface Temperature -- in every year, permanently.
This script measures how much of the world's urban land sits in such a gap.

Method:
  1. Download GHS-SMOD R2023A (1 km, Mollweide), the urban reference.
  2. Reduce SMOD to the set of 1-degree cells that contain settlement.
  3. Fetch only the ASTER GED AG1km granules covering those cells.
  4. Classify every land pixel by its clear-sky observation count:
       NumObs == 0   -> no emissivity, permanent gap
       NumObs 1-2    -> low confidence (USGS: "only one or two scenes")
       NumObs >= 3   -> normal
  5. Warp the tier raster onto the SMOD grid. Mollweide is equal-area at 1 km,
     so a pixel count converts to km2 directly, then cross-tabulate tier
     against settlement class.

Because every settled pixel lies inside a fetched cell, the settlement
denominators are complete and the percentages are exact. The global land gap
figure is NOT measurable this way -- cite USGS (~96.3% coverage) instead.

Usage:
    uv run --extra analysis python scripts/aster_gap_urban_analysis.py
    uv run --extra analysis python scripts/aster_gap_urban_analysis.py --skip-download
    uv run --extra analysis python scripts/aster_gap_urban_analysis.py --validate-tile S25E030

Requires NASA Earthdata credentials once:
    uv run python -c "import earthaccess; earthaccess.login(persist=True)"

Output (results/aster-gaps/):
    gap_tiers_smod_grid.tif   tier raster on the SMOD grid (large, gitignored)
    gap_by_class.csv          km2 per settlement class and tier
    summary.json              headline numbers
    fig_gap_by_class.png      gap share by settlement class
    report.md                 human-readable summary
    validation_<tile>.json    predicted vs observed gap (--validate-tile)
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.request import urlopen

import numpy as np
import pandas as pd
import rasterio
import structlog
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds

if TYPE_CHECKING:
    from affine import Affine

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
log = structlog.get_logger()

# --- GHSL -------------------------------------------------------------------

SMOD_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/"
    "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/V1-0/"
    "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0.zip"
)
SMOD_TIF = "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0.tif"

# GHS-SMOD "Degree of Urbanisation" level-2 classes.
SMOD_LABELS = {
    10: "Water",
    11: "Very low density rural",
    12: "Low density rural",
    13: "Rural cluster",
    21: "Suburban or peri-urban",
    22: "Semi-dense urban cluster",
    23: "Dense urban cluster",
    30: "Urban centre",
}
URBAN_CLASSES = (21, 22, 23, 30)
RURAL_CLASSES = (11, 12, 13)
SETTLED_CLASSES = RURAL_CLASSES + URBAN_CLASSES

# --- ASTER GED --------------------------------------------------------------

AG1KM_SHORT_NAME = "AG1km"
AG1KM_VERSION = "003"

# HDF5 paths inside an AG1km v003 granule. The Land Water Map group name
# genuinely contains spaces.
H5_NUM_OBS = "Observations/NumObs"
H5_LWMAP = "Land Water Map/LWmap"
H5_LAT = "Geolocation/Latitude"
H5_LON = "Geolocation/Longitude"

LWMAP_LAND = 1

# Tier codes written to the mosaic. 0 doubles as nodata, so water and
# unevaluated pixels drop out of every count.
TIER_NODATA = 0
TIER_NORMAL = 1
TIER_LOW_CONFIDENCE = 2
TIER_GAP = 3
TIER_LABELS = {
    TIER_NORMAL: "normal",
    TIER_LOW_CONFIDENCE: "low_confidence",
    TIER_GAP: "gap",
}
# USGS: "in some regions, only one or two scenes were available to produce GED".
LOW_CONFIDENCE_MAX_OBS = 2

# --- Paths ------------------------------------------------------------------

DATA_DIR = Path("data")
GHSL_DIR = DATA_DIR / "ghsl"
ASTER_DIR = DATA_DIR / "aster_ged"
OUT_DIR = Path("results/aster-gaps")

# Ordinal blue ramp, validated light-mode: severity reads light -> dark.
COLOR_LOW_CONFIDENCE = "#5598e7"
COLOR_GAP = "#104281"
COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#52514e"
COLOR_SURFACE = "#fcfcfb"


# =============================================================================
# Pure helpers (unit-tested)
# =============================================================================


def classify_tiers(num_obs: np.ndarray, lwmap: np.ndarray) -> np.ndarray:
    """Classify ASTER GED pixels into coverage tiers.

    Water is excluded via the land/water map so it is never counted as a gap.

    Args:
        num_obs: Clear-sky ASTER observation count per pixel.
        lwmap: Land water map, 1 = land, 2 = water.

    Returns:
        uint8 array of TIER_* codes; TIER_NODATA over water.
    """
    tiers = np.full(num_obs.shape, TIER_NODATA, dtype=np.uint8)
    land = lwmap == LWMAP_LAND
    tiers[land & (num_obs >= LOW_CONFIDENCE_MAX_OBS + 1)] = TIER_NORMAL
    tiers[land & (num_obs >= 1) & (num_obs <= LOW_CONFIDENCE_MAX_OBS)] = TIER_LOW_CONFIDENCE
    tiers[land & (num_obs <= 0)] = TIER_GAP
    return tiers


def cell_key(lat: float, lon: float) -> tuple[int, int]:
    """Return the (south, west) integer key of the 1-degree cell holding a point."""
    return (int(np.floor(lat)), int(np.floor(lon)))


def granule_cell(umm: dict[str, Any]) -> tuple[int, int] | None:
    """Extract the (south, west) 1-degree cell key from CMR granule metadata.

    Keys off the granule's own polygon rather than parsing its id. Ids look
    like ``AG1km.v003.-11.154.0010``, whose trailing field is undocumented;
    the polygon is authoritative.

    Args:
        umm: The ``umm`` block of a CMR granule record.

    Returns:
        Cell key, or None when the record carries no usable polygon.
    """
    try:
        geometry = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        points = geometry["GPolygons"][0]["Boundary"]["Points"]
    except (KeyError, IndexError, TypeError):
        return None
    if not points:
        return None
    lats = [p["Latitude"] for p in points]
    lons = [p["Longitude"] for p in points]
    return cell_key(min(lats), min(lons))


def cells_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    """Convert a 180x360 boolean grid of 1-degree cells into (south, west) keys.

    Row 0 is the 89-90N band and column 0 the 180-179W band, matching a
    north-up EPSG:4326 grid at 1-degree resolution.
    """
    rows, cols = np.nonzero(mask)
    return {(int(89 - r), int(-180 + c)) for r, c in zip(rows, cols, strict=True)}


def tier_area_table(counts: dict[tuple[int, int], int], pixel_km2: float) -> pd.DataFrame:
    """Build the per-class area table from raw (smod_class, tier) pixel counts."""
    rows = [
        {
            "smod_class": smod_class,
            "smod_label": SMOD_LABELS.get(smod_class, f"class {smod_class}"),
            "tier": TIER_LABELS[tier],
            "km2": count * pixel_km2,
        }
        for (smod_class, tier), count in sorted(counts.items())
        if tier in TIER_LABELS
    ]
    return pd.DataFrame(rows, columns=["smod_class", "smod_label", "tier", "km2"])


def summarize(table: pd.DataFrame, classes: tuple[int, ...]) -> dict[str, float]:
    """Roll a per-class area table up into headline gap shares."""
    subset = table[table["smod_class"].isin(classes)]
    total = float(subset["km2"].sum())
    gap = float(subset.loc[subset["tier"] == "gap", "km2"].sum())
    low = float(subset.loc[subset["tier"] == "low_confidence", "km2"].sum())
    return {
        "land_km2": total,
        "gap_km2": gap,
        "low_confidence_km2": low,
        "gap_pct": 100.0 * gap / total if total else 0.0,
        "low_confidence_pct": 100.0 * low / total if total else 0.0,
    }


# =============================================================================
# Stage 1: GHS-SMOD
# =============================================================================


def fetch_smod(*, skip_download: bool) -> Path:
    """Download and unzip GHS-SMOD unless it is already on disk."""
    tif = GHSL_DIR / SMOD_TIF
    if tif.exists():
        log.info("smod_cached", path=str(tif))
        return tif
    if skip_download:
        raise FileNotFoundError(f"{tif} missing and --skip-download was passed")

    GHSL_DIR.mkdir(parents=True, exist_ok=True)
    archive = GHSL_DIR / "smod.zip"
    log.info("smod_download", url=SMOD_URL)
    with urlopen(SMOD_URL) as response, archive.open("wb") as handle:
        handle.write(response.read())
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(GHSL_DIR)
    log.info("smod_ready", path=str(tif))
    return tif


def urban_cell_set(smod_path: Path, classes: tuple[int, ...]) -> set[tuple[int, int]]:
    """Find every 1-degree cell containing at least one pixel of the given classes.

    Reprojects a boolean settlement mask onto a 1-degree EPSG:4326 grid with
    ``Resampling.max``, so a cell is selected when any source pixel qualifies.
    """
    cells_path = GHSL_DIR / f"cells_1deg_{'_'.join(str(c) for c in classes)}.npy"
    if cells_path.exists():
        mask = np.load(cells_path)
        log.info("cells_cached", path=str(cells_path), cells=int(mask.sum()))
        return cells_from_mask(mask)

    mask_path = GHSL_DIR / f"mask_1km_{'_'.join(str(c) for c in classes)}.tif"
    if not mask_path.exists():
        _write_class_mask(smod_path, classes, mask_path)

    grid = np.zeros((180, 360), dtype=np.uint8)
    with rasterio.open(mask_path) as src:
        log.info("cells_reproject", classes=classes)
        reproject(
            source=rasterio.band(src, 1),
            destination=grid,
            dst_transform=from_bounds(-180, -90, 180, 90, 360, 180),
            dst_crs="EPSG:4326",
            resampling=Resampling.max,
            src_nodata=0,
            dst_nodata=0,
        )

    mask = grid.astype(bool)
    np.save(cells_path, mask)
    log.info("cells_ready", cells=int(mask.sum()))
    return cells_from_mask(mask)


def _write_class_mask(smod_path: Path, classes: tuple[int, ...], out_path: Path) -> None:
    """Write a 1-bit-ish uint8 mask of the requested SMOD classes, block by block."""
    with rasterio.open(smod_path) as src:
        profile = src.profile | {
            "dtype": "uint8",
            "nodata": 0,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        log.info("class_mask_build", path=str(out_path), shape=src.shape)
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                block = src.read(1, window=window)
                dst.write(np.isin(block, classes).astype(np.uint8), 1, window=window)


# =============================================================================
# Stage 2: ASTER GED granules
# =============================================================================


def search_aster_granules(cells: set[tuple[int, int]]) -> list[Any]:
    """Search CMR for AG1km granules and keep those covering the given cells."""
    import earthaccess

    _require_credentials(earthaccess)
    log.info("aster_search", short_name=AG1KM_SHORT_NAME, version=AG1KM_VERSION)
    results = earthaccess.search_data(short_name=AG1KM_SHORT_NAME, version=AG1KM_VERSION, count=-1)

    keep: dict[tuple[int, int], Any] = {}
    unplaced = 0
    for granule in results:
        cell = granule_cell(granule.get("umm", {}))
        if cell is None:
            unplaced += 1
            continue
        if cell in cells and cell not in keep:
            keep[cell] = granule

    log.info(
        "aster_search_done",
        total=len(results),
        matched=len(keep),
        wanted=len(cells),
        unplaced=unplaced,
    )
    return list(keep.values())


def _require_credentials(earthaccess: Any) -> None:
    """Fail with the fix rather than a stack trace when Earthdata auth is missing.

    Never uses the interactive strategy: an unattended run should exit with
    instructions instead of blocking on a password prompt.
    """
    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy, persist=False)
        except Exception as exc:
            log.debug("earthaccess_login_failed", strategy=strategy, error=str(exc))
            continue
        if getattr(auth, "authenticated", False):
            log.info("earthaccess_authenticated", strategy=strategy)
            return

    raise SystemExit(
        "NASA Earthdata credentials not found. Run once, interactively:\n"
        '    uv run python -c "import earthaccess; earthaccess.login(persist=True)"'
    )


def download_granules(granules: list[Any]) -> list[Path]:
    """Download granules that are not already on disk; return every local path."""
    import earthaccess

    ASTER_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in ASTER_DIR.glob("*.h5")}
    pending = [g for g in granules if _granule_filename(g) not in existing]

    log.info("aster_download", pending=len(pending), cached=len(granules) - len(pending))
    if pending:
        earthaccess.download(pending, str(ASTER_DIR))

    wanted = {_granule_filename(g) for g in granules}
    return sorted(p for p in ASTER_DIR.glob("*.h5") if p.name in wanted)


def _granule_filename(granule: Any) -> str:
    """Best-effort local filename for a CMR granule record."""
    for link in granule.data_links():
        name = link.rsplit("/", 1)[-1]
        if name.endswith(".h5"):
            return name
    return f"{granule['umm']['GranuleUR']}.h5"


def read_granule_tiers(path: Path) -> tuple[np.ndarray, Affine]:
    """Read one AG1km granule and return its tier raster plus geotransform."""
    import h5py

    with h5py.File(path, "r") as h5:
        num_obs = np.asarray(h5[H5_NUM_OBS][:])
        lwmap = np.asarray(h5[H5_LWMAP][:])
        lat = np.asarray(h5[H5_LAT][:])
        lon = np.asarray(h5[H5_LON][:])

    tiers = classify_tiers(num_obs, lwmap)
    # Granules tile the globe on whole degrees; derive exact bounds from the
    # cell rather than from pixel-centre coordinates.
    south, west = cell_key(float(lat.min()), float(lon.min()))
    height, width = tiers.shape
    transform = from_bounds(west, south, west + 1, south + 1, width, height)
    return tiers, transform


# =============================================================================
# Stage 3: mosaic onto the SMOD grid
# =============================================================================


def build_tier_mosaic(granule_paths: list[Path], smod_path: Path, out_path: Path) -> Path:
    """Warp every granule's tier raster into a single raster on the SMOD grid.

    The SMOD grid is Mollweide, which is equal-area, so downstream pixel counts
    convert straight to km2. SMOD itself is never resampled; only the
    categorical tier raster moves, by nearest neighbour.
    """
    with rasterio.open(smod_path) as smod:
        profile = smod.profile | {
            "dtype": "uint8",
            "nodata": TIER_NODATA,
            "count": 1,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        dst_crs = smod.crs
        dst_transform = smod.transform
        dst_shape = smod.shape

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("mosaic_build", granules=len(granule_paths), path=str(out_path))

    with rasterio.open(out_path, "w+", **profile) as dst:
        for index, path in enumerate(granule_paths, start=1):
            _burn_granule(dst, path, dst_crs, dst_transform, dst_shape)
            if index % 500 == 0:
                log.info("mosaic_progress", done=index, total=len(granule_paths))

    log.info("mosaic_ready", path=str(out_path))
    return out_path


def _burn_granule(
    dst: rasterio.io.DatasetWriter,
    path: Path,
    dst_crs: Any,
    dst_transform: Affine,
    dst_shape: tuple[int, int],
) -> None:
    """Reproject one granule into its window of the mosaic, preserving neighbours."""
    tiers, src_transform = read_granule_tiers(path)
    west, south, east, north = rasterio.transform.array_bounds(*tiers.shape, src_transform)
    bounds = transform_bounds("EPSG:4326", dst_crs, west, south, east, north, densify_pts=21)

    window = window_from_bounds(*bounds, transform=dst_transform).round_offsets().round_lengths()
    window = window.intersection(Window(0, 0, dst_shape[1], dst_shape[0]))
    if window.width <= 0 or window.height <= 0:
        return

    patch = np.zeros((int(window.height), int(window.width)), dtype=np.uint8)
    reproject(
        source=tiers,
        destination=patch,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        src_nodata=TIER_NODATA,
        dst_transform=dst.window_transform(window),
        dst_crs=dst_crs,
        dst_nodata=TIER_NODATA,
        resampling=Resampling.nearest,
    )
    if not patch.any():
        return

    # Windows overlap slightly at cell edges after the Mollweide bbox padding;
    # keep whatever a neighbouring granule already wrote.
    prior = dst.read(1, window=window)
    dst.write(np.where(patch != TIER_NODATA, patch, prior), 1, window=window)


def crosstab_tiers(tier_path: Path, smod_path: Path) -> pd.DataFrame:
    """Cross-tabulate coverage tier against settlement class over the whole grid."""
    # Pack (smod_class, tier) into one integer key so each block reduces with a
    # single bincount; a Python-level loop over ~650M pixels is not viable.
    stride = max(TIER_LABELS) + 1
    totals = np.zeros((max(SMOD_LABELS) + 1) * stride, dtype=np.int64)

    with rasterio.open(tier_path) as tiers, rasterio.open(smod_path) as smod:
        pixel_km2 = abs(tiers.transform.a * tiers.transform.e) / 1_000_000.0
        for _, window in tiers.block_windows(1):
            tier_block = tiers.read(1, window=window)
            if not tier_block.any():
                continue
            smod_block = smod.read(1, window=window)
            valid = (tier_block != TIER_NODATA) & (smod_block > 0)
            if not valid.any():
                continue
            keys = smod_block[valid].astype(np.int64) * stride + tier_block[valid]
            totals += np.bincount(keys, minlength=totals.size)

    counts = {
        (int(key // stride), int(key % stride)): int(count)
        for key, count in enumerate(totals)
        if count
    }
    log.info("crosstab_done", pairs=len(counts), pixel_km2=pixel_km2)
    return tier_area_table(counts, pixel_km2)


# =============================================================================
# Stage 4: outputs
# =============================================================================


def write_outputs(table: pd.DataFrame, out_dir: Path, classes: tuple[int, ...]) -> dict[str, Any]:
    """Write the CSV, JSON summary, figure, and report for the evaluated classes.

    Only cells containing ``classes`` were fetched, so any other class in the
    cross-tabulation is counted over an arbitrary subset of its true extent.
    Those rows are dropped rather than published: a partial number that looks
    whole is worse than no number.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluated = tuple(sorted(classes))

    dropped = sorted(set(table["smod_class"]) - set(evaluated))
    if dropped:
        log.info("dropping_partial_classes", classes=dropped)
    complete = table[table["smod_class"].isin(evaluated)].reset_index(drop=True)
    complete.to_csv(out_dir / "gap_by_class.csv", index=False)

    urban = tuple(c for c in evaluated if c in URBAN_CLASSES)
    summary = {
        "evaluated_classes": list(evaluated),
        "all_evaluated": summarize(complete, evaluated),
        "urban": summarize(complete, urban),
        "by_class": {SMOD_LABELS.get(c, str(c)): summarize(complete, (c,)) for c in evaluated},
        "notes": (
            "Gap = ASTER GED NumObs == 0 over land, meaning no emissivity and "
            "therefore no Landsat C2 L2 Surface Temperature. Areas are km2 on the "
            "equal-area GHS-SMOD Mollweide grid. Only 1-degree cells containing the "
            "evaluated classes were fetched, so these figures are complete for those "
            "classes and no global land total can be derived from them."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plot_gap_by_class(complete, out_dir / "fig_gap_by_class.png", evaluated)
    _write_report(complete, summary, out_dir / "report.md", evaluated)
    return summary


def plot_gap_by_class(table: pd.DataFrame, out_path: Path, evaluated: tuple[int, ...]) -> None:
    """Horizontal stacked bars: share of each settlement class in each gap tier."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = [c for c in SETTLED_CLASSES if c in evaluated and c in set(table["smod_class"])]
    labels = [SMOD_LABELS[c] for c in classes]
    stats = [summarize(table, (c,)) for c in classes]
    gap = [s["gap_pct"] for s in stats]
    low = [s["low_confidence_pct"] for s in stats]

    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    y = np.arange(len(classes))

    ax.barh(y, gap, height=0.6, color=COLOR_GAP, label="No emissivity (permanent gap)")
    ax.barh(
        y,
        low,
        height=0.6,
        left=gap,
        color=COLOR_LOW_CONFIDENCE,
        label="Low confidence (1-2 scenes)",
    )

    for index, (g, lo) in enumerate(zip(gap, low, strict=True)):
        ax.text(
            g + lo + 0.25,
            index,
            f"{g:.1f}%",
            va="center",
            fontsize=9,
            color=COLOR_TEXT,
        )

    ax.set_yticks(y, labels, fontsize=9, color=COLOR_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Share of class land area (%)", fontsize=9, color=COLOR_MUTED)
    ax.tick_params(axis="x", labelsize=9, colors=COLOR_MUTED)
    ax.set_title(
        "ASTER GED coverage gaps by settlement class",
        fontsize=11,
        color=COLOR_TEXT,
        loc="left",
        pad=12,
    )
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d6d5d0")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=COLOR_TEXT)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=COLOR_SURFACE)
    plt.close(fig)
    log.info("figure_written", path=str(out_path))


def _write_report(
    table: pd.DataFrame, summary: dict[str, Any], out_path: Path, evaluated: tuple[int, ...]
) -> None:
    """Write a short markdown summary next to the machine-readable outputs."""
    lines = [
        "# ASTER GED coverage gaps over settlement",
        "",
        "Gap = ASTER GED `NumObs == 0` over land: no emissivity, so no Landsat",
        "Collection 2 Level-2 Surface Temperature, permanently.",
        "",
        "| Settlement class | Land km2 | Gap km2 | Gap % | Low confidence % |",
        "|---|---:|---:|---:|---:|",
    ]
    for smod_class in SETTLED_CLASSES:
        if smod_class not in evaluated or smod_class not in set(table["smod_class"]):
            continue
        stats = summarize(table, (smod_class,))
        lines.append(
            f"| {SMOD_LABELS[smod_class]} | {stats['land_km2']:,.0f} | "
            f"{stats['gap_km2']:,.0f} | {stats['gap_pct']:.2f} | "
            f"{stats['low_confidence_pct']:.2f} |"
        )

    urban = summary["urban"]
    lines += [
        "",
        f"**Urban domain (classes 21, 22, 23, 30)**: {urban['gap_pct']:.2f}% of "
        f"{urban['land_km2']:,.0f} km2 has no emissivity "
        f"({urban['gap_km2']:,.0f} km2).",
        "",
    ]
    out_path.write_text("\n".join(lines))


# =============================================================================
# Validation against produced rasters
# =============================================================================


def validate_tile(tile_name: str, year: int, out_dir: Path) -> dict[str, Any]:
    """Compare the ASTER-predicted gap against measured Landsat ST coverage.

    The ASTER tier raster predicts where Surface Temperature is unavailable; it
    is not a measurement of it. This runs the real pipeline for one tile and
    reports both fractions over the same land mask.
    """
    from landsat_lst.config import STAC_PLANETARY_COMPUTER, settings

    settings.stac_url = STAC_PLANETARY_COMPUTER

    from landsat_lst.masks import get_land_mask_for_bbox, load_land_polygons
    from landsat_lst.models import ProcessingJob
    from landsat_lst.pipeline import process_tile
    from landsat_lst.tiling import parse_tile_name

    tile = parse_tile_name(tile_name)
    log.info("validate_start", tile=tile_name, year=year, bbox=tile.bbox)

    composite = process_tile(ProcessingJob(tile=tile, year=year))
    shape = (len(composite.latitude), len(composite.longitude))
    land = get_land_mask_for_bbox(
        tile.bbox, settings.resolution, load_land_polygons(), target_shape=shape
    )

    # process_tile zeroes qa_count over ocean, so the land mask -- not
    # qa_count itself -- has to supply the denominator.
    observed_gap = (composite["qa_count"].sum("month").to_numpy() == 0) & land
    observed_pct = 100.0 * observed_gap.sum() / land.sum()

    predicted = _predicted_gap_for_bbox(tile.bbox, shape)
    predicted_gap = predicted & land
    predicted_pct = 100.0 * predicted_gap.sum() / land.sum()

    both = int((observed_gap & predicted_gap).sum())
    union = int((observed_gap | predicted_gap).sum())
    result = {
        "tile": tile_name,
        "year": year,
        "land_pixels": int(land.sum()),
        "observed_gap_pct": float(observed_pct),
        "predicted_gap_pct": float(predicted_pct),
        "agreement_iou": float(both / union) if union else 1.0,
        "observed_only_pixels": int((observed_gap & ~predicted_gap).sum()),
        "predicted_only_pixels": int((predicted_gap & ~observed_gap).sum()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"validation_{tile_name}.json").write_text(json.dumps(result, indent=2) + "\n")
    log.info("validate_done", **result)
    return result


def _predicted_gap_for_bbox(
    bbox: tuple[float, float, float, float], shape: tuple[int, int]
) -> np.ndarray:
    """Rasterize the ASTER-predicted gap onto an EPSG:4326 grid for one bbox."""
    west, south, east, north = bbox
    height, width = shape
    dst_transform = from_bounds(west, south, east, north, width, height)
    out = np.zeros(shape, dtype=np.uint8)

    for lat in range(int(np.floor(south)), int(np.ceil(north))):
        for lon in range(int(np.floor(west)), int(np.ceil(east))):
            path = _granule_path_for_cell(lat, lon)
            if path is None:
                continue
            tiers, src_transform = read_granule_tiers(path)
            patch = np.zeros(shape, dtype=np.uint8)
            reproject(
                source=(tiers == TIER_GAP).astype(np.uint8),
                destination=patch,
                src_transform=src_transform,
                src_crs="EPSG:4326",
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest,
                src_nodata=255,
                dst_nodata=0,
            )
            out |= patch

    return out.astype(bool)


def _granule_path_for_cell(south: int, west: int) -> Path | None:
    """Find the downloaded granule covering a 1-degree cell, if present."""
    for path in ASTER_DIR.glob("*.h5"):
        parts = path.stem.split(".")
        if len(parts) < 4:
            continue
        try:
            north_edge, west_edge = int(parts[2]), int(parts[3])
        except ValueError:
            continue
        if north_edge - 1 == south and west_edge == west:
            return path
    return None


# =============================================================================
# Entry point
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=list(URBAN_CLASSES),
        help=(
            "GHS-SMOD classes to evaluate (default: the urban domain 21 22 23 30). "
            "Adding rural classes 11 12 13 pulls in nearly every land cell on Earth, "
            "so the download grows from roughly 9k granules to the full 25k."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Fail rather than fetch anything that is missing from disk.",
    )
    parser.add_argument(
        "--validate-tile",
        help="Run the pipeline for one 5-degree tile and compare against the prediction.",
    )
    parser.add_argument("--validate-year", type=int, default=2024, help="Year for --validate-tile.")
    parser.add_argument("--output", type=Path, default=OUT_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.validate_tile:
        validate_tile(args.validate_tile, args.validate_year, args.output)
        return

    classes = tuple(sorted(args.classes))
    smod_path = fetch_smod(skip_download=args.skip_download)
    cells = urban_cell_set(smod_path, classes)

    granules = search_aster_granules(cells)
    paths = download_granules(granules)
    log.info("granules_local", count=len(paths))

    tier_path = build_tier_mosaic(paths, smod_path, args.output / "gap_tiers_smod_grid.tif")
    table = crosstab_tiers(tier_path, smod_path)
    summary = write_outputs(table, args.output, classes)

    urban = summary["urban"]
    print(
        f"\nUrban land evaluated: {urban['land_km2']:,.0f} km2\n"
        f"Permanent gap:        {urban['gap_km2']:,.0f} km2 ({urban['gap_pct']:.2f}%)\n"
        f"Low confidence:       {urban['low_confidence_km2']:,.0f} km2 "
        f"({urban['low_confidence_pct']:.2f}%)\n"
        f"Output written to:    {args.output}"
    )


if __name__ == "__main__":
    main()
