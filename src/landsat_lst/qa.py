"""QA filtering and masking functions for Landsat data."""

import xarray as xr


def create_qa_mask(qa_pixel: xr.DataArray) -> xr.DataArray:
    """Create a boolean mask from Landsat QA pixel band.

    Masks pixels with cloud, cloud shadow, or snow/ice flags set.

    Args:
        qa_pixel: QA pixel band from Landsat C2 L2 product.

    Returns:
        Boolean DataArray where True indicates GOOD (usable) pixels.

    Reference:
        Landsat Collection 2 QA_PIXEL bit assignments:
        - Bit 3: Cloud (1 = cloud)
        - Bit 4: Cloud Shadow (1 = shadow)
        - Bit 5: Snow (1 = snow/ice)
    """
    cloud = (qa_pixel >> 3) & 1
    shadow = (qa_pixel >> 4) & 1
    snow = (qa_pixel >> 5) & 1

    return (cloud == 0) & (shadow == 0) & (snow == 0)


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

    Args:
        lwir_band: LWIR11 band (ST_B10) from Landsat C2 L2.

    Returns:
        Land Surface Temperature in degrees Celsius.

    Reference:
        Scale factor: 0.00341802
        Offset: 149.0
        Result is in Kelvin, then converted to Celsius.
    """
    lst_kelvin = lwir_band * 0.00341802 + 149.0
    lst_celsius = lst_kelvin - 273.15
    return lst_celsius
