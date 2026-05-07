"""Main ETL pipeline for Landsat LST composites."""

import numpy as np
import pystac_client
import xarray as xr
from odc.stac import stac_load

from landsat_lst.config import settings
from landsat_lst.models import ProcessingJob
from landsat_lst.qa import apply_qa_mask, convert_to_celsius


def query_stac(job: ProcessingJob) -> list:
    """Query STAC catalog for Landsat scenes.

    Args:
        job: Processing job with tile and year info.

    Returns:
        List of STAC items matching the query.
    """
    catalog = pystac_client.Client.open(settings.stac_url)

    search = catalog.search(
        collections=[settings.collection],
        bbox=job.tile.bbox,
        datetime=job.datetime_range,
        query={
            "eo:cloud_cover": {"lt": settings.max_cloud_cover},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
    )

    return list(search.items())


def load_scenes(items: list, bbox: tuple[float, float, float, float]) -> xr.Dataset:
    """Load Landsat scenes as an xarray Dataset.

    Args:
        items: List of STAC items to load.
        bbox: Bounding box for spatial subsetting.

    Returns:
        Dataset with thermal and QA bands.
    """
    return stac_load(
        items,
        bands=["lwir11", "qa_pixel"],
        crs=settings.crs,
        resolution=settings.resolution,
        chunks={"time": 10, "x": 2048, "y": 2048},
        groupby="solar_day",
        bbox=bbox,
    )


def compute_annual_composite(data: xr.Dataset) -> xr.Dataset:
    """Compute annual LST composite with p50, p95, and observation count.

    Args:
        data: Dataset with thermal and QA bands across time.

    Returns:
        Dataset with lst_p50, lst_p95, and qa_count bands.
    """
    masked = apply_qa_mask(data)

    lst = convert_to_celsius(masked["lwir11"])

    valid_mask = ~np.isnan(lst)
    qa_count = valid_mask.sum(dim="time").astype(np.int16)  # ty: ignore[no-matching-overload]

    lst_p50 = lst.quantile(0.5, dim="time", skipna=True).drop_vars("quantile")
    lst_p95 = lst.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")

    lst_p50 = lst_p50.where(qa_count > 0, settings.nodata)
    lst_p95 = lst_p95.where(qa_count > 0, settings.nodata)

    return xr.Dataset(
        {
            "lst_p50": lst_p50.astype(np.float32),
            "lst_p95": lst_p95.astype(np.float32),
            "qa_count": qa_count,
        }
    )


def process_tile(job: ProcessingJob) -> xr.Dataset:
    """Process a single tile for one year.

    Args:
        job: Processing job specification.

    Returns:
        Dataset with annual composite.
    """
    items = query_stac(job)

    if not items:
        msg = f"No scenes found for {job.tile.name} in {job.year}"
        raise ValueError(msg)

    data = load_scenes(items, job.tile.bbox)

    composite = compute_annual_composite(data)

    composite.attrs["tile"] = job.tile.name
    composite.attrs["year"] = job.year
    composite.attrs["scene_count"] = len(items)

    return composite
