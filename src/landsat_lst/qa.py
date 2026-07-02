"""QA filtering and masking functions for Landsat data."""

import xarray as xr

from landsat_lst.config import settings


def create_qa_mask(qa_pixel: xr.DataArray) -> xr.DataArray:
    """Create a boolean mask from Landsat QA pixel band.

    Masks pixels flagged as dilated cloud, cirrus, cloud, cloud shadow, or
    snow/ice -- the standard Collection 2 "clear-sky" definition. Dilated cloud
    (bit 1) and cirrus (bit 2) catch the cloud-edge and thin-cloud/haze
    contamination that would otherwise survive as per-scene warm/cool residuals
    and drive scene-footprint striping in the composite.

    Args:
        qa_pixel: QA pixel band from Landsat C2 L2 product.

    Returns:
        Boolean DataArray where True indicates GOOD (usable) pixels.

    Reference:
        Landsat Collection 2 QA_PIXEL bit assignments:
        - Bit 1: Dilated Cloud (1 = near cloud)
        - Bit 2: Cirrus (1 = cirrus)
        - Bit 3: Cloud (1 = cloud)
        - Bit 4: Cloud Shadow (1 = shadow)
        - Bit 5: Snow (1 = snow/ice)
    """
    dilated = (qa_pixel >> 1) & 1
    cirrus = (qa_pixel >> 2) & 1
    cloud = (qa_pixel >> 3) & 1
    shadow = (qa_pixel >> 4) & 1
    snow = (qa_pixel >> 5) & 1

    return (dilated == 0) & (cirrus == 0) & (cloud == 0) & (shadow == 0) & (snow == 0)


def apply_qa_mask(data: xr.Dataset, qa_band: str = "qa_pixel") -> xr.Dataset:
    """Apply QA mask to all data variables in a dataset.

    Args:
        data: Dataset containing Landsat bands including QA.
        qa_band: Name of the QA band.

    Returns:
        Dataset with masked values set to NaN.
    """
    mask = create_qa_mask(data[qa_band])
    return data.where(mask)


def convert_to_celsius(lwir_band: xr.DataArray) -> xr.DataArray:
    """Convert Landsat thermal band to Celsius.

    Applies Landsat Collection 2 Level-2 scaling factors.
    Masks fill value (0) which indicates nodata (scene edges, SLC-off gaps).

    Args:
        lwir_band: LWIR11 band (ST_B10) from Landsat C2 L2.

    Returns:
        Land Surface Temperature in degrees Celsius.

    Reference:
        Scale factor: 0.00341802
        Offset: 149.0
        Result is in Kelvin, then converted to Celsius.
        Fill value: 0 (converts to -124°C if not masked)

    A physical-plausibility clamp drops values outside
    ``[settings.lst_valid_min, settings.lst_valid_max]`` °C. This removes the
    ~-124°C artifacts produced when reprojection interpolates near the DN=0 fill
    at scene edges (not caught by the exact ``!= 0`` test), plus any high-DN
    saturation junk -- both are per-scene-edge contaminants that feed striping.
    """
    # Mask fill value (0) before conversion - these are nodata pixels
    lwir_valid = lwir_band.where(lwir_band != 0)
    lst_kelvin = lwir_valid * 0.00341802 + 149.0
    lst_celsius = lst_kelvin - 273.15
    # Drop physically implausible land-surface temperatures (resampling/fill junk).
    return lst_celsius.where(
        (lst_celsius >= settings.lst_valid_min) & (lst_celsius <= settings.lst_valid_max)
    )
