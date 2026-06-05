"""Main ETL pipeline for Landsat LST composites."""

import os

import numpy as np
import pystac_client
import xarray as xr
from odc.stac import stac_load

from landsat_lst.config import settings
from landsat_lst.masks import get_land_mask_for_bbox, load_land_polygons
from landsat_lst.models import ProcessingJob
from landsat_lst.qa import apply_qa_mask, convert_to_celsius

# Planetary Computer URL prefix for conditional signing
_PC_URL_PREFIX = "https://planetarycomputer.microsoft.com"


def _is_planetary_computer() -> bool:
    """Check if using Planetary Computer endpoint."""
    return settings.stac_url.startswith(_PC_URL_PREFIX)


def _configure_requester_pays() -> None:
    """Configure GDAL/rasterio for AWS requester-pays buckets.

    Earth Search serves Landsat data from the usgs-landsat bucket which
    requires requester-pays. These env vars must be set before rasterio
    opens any files.
    """
    os.environ.setdefault("AWS_REQUEST_PAYER", "requester")
    os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "NO")


def query_stac(job: ProcessingJob) -> list:
    """Query STAC catalog for Landsat scenes.

    For Planetary Computer: signs items with SAS tokens for Azure access.
    For Earth Search: configures requester-pays for S3 access.

    Args:
        job: Processing job with tile and year info.

    Returns:
        List of STAC items matching the query.
    """
    modifier = None

    if _is_planetary_computer():
        import planetary_computer as pc  # noqa: PLC0415

        modifier = pc.sign_inplace
    else:
        _configure_requester_pays()

    catalog = pystac_client.Client.open(
        settings.stac_url,
        modifier=modifier,
    )

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
        chunks={"time": 10, "latitude": 512, "longitude": 512},
        groupby="solar_day",
        bbox=bbox,
    )


def compute_annual_composite(data: xr.Dataset) -> xr.Dataset:
    """Compute annual LST composite with p95 and observation count.

    WARNING: This function does NOT apply land masking. For production output
    that excludes ocean pixels, use process_tile() instead. See issue #26.

    Args:
        data: Dataset with thermal and QA bands across time.

    Returns:
        Dataset with lst_p95 and qa_count bands.

    Note:
        P50 (median) was removed per stakeholder feedback - hot season temps
        (P95) are what matter for urban heat applications. See issue #22.
    """
    masked = apply_qa_mask(data)

    lst = convert_to_celsius(masked["lwir11"])

    valid_mask = ~np.isnan(lst)
    qa_count = valid_mask.sum(dim="time").astype(np.int16)  # ty: ignore[no-matching-overload]

    lst_p95 = lst.quantile(0.95, dim="time", skipna=True).drop_vars("quantile")
    lst_p95 = lst_p95.where(qa_count > 0, settings.nodata)

    return xr.Dataset(
        {
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

    # Apply land mask to exclude ocean/water pixels (Natural Earth 10m)
    land_polygons = load_land_polygons()
    land_mask = get_land_mask_for_bbox(
        job.tile.bbox,
        settings.resolution,
        land_polygons,
    )
    # Both rasterio and odc-stac use north-down (descending latitude), no flip needed
    land_mask_da = xr.DataArray(
        land_mask,
        dims=["latitude", "longitude"],
        coords={
            "latitude": composite.latitude,
            "longitude": composite.longitude,
        },
    )
    composite["lst_p95"] = composite["lst_p95"].where(land_mask_da)

    composite.attrs["tile"] = job.tile.name
    composite.attrs["year"] = job.year
    composite.attrs["scene_count"] = len(items)

    return composite
