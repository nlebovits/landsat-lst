"""QA filtering and masking functions for Landsat data."""

import numpy as np
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


# ---------------------------------------------------------------------------
# The coarse stack's integer representation (issue #125).
#
# The offset estimator reads a float32 Celsius stack. Its source is uint16 DN,
# and the map between them is lossless in one direction: every float32 Celsius
# value the estimator sees is an affine function of one DN, and every sample it
# drops is droppable from the DN alone. So the observations can be *carried* as
# DN at two bytes per element and handed to the estimator as the float32 it
# always read. That is what lets phase A stage what it reads and phase B reuse
# it instead of reading the sources a second time (ADR-020).
#
# These five names are the lossless half of the representation introduced by
# #136/#138. The quantisation-bearing half of that work -- the whole-DN offset
# shift and the integer P95 -- is deliberately not here: it changes published
# values, and this module's contract is that it does not.
# ---------------------------------------------------------------------------

#: The stack's "no observation" value. DN 0 is the source's fill, so no valid
#: observation can carry it.
DN_SENTINEL: int = 0
DN_MAX: int = 65535


def dn_clamp_bounds() -> tuple[int, int]:
    """DN bounds that reproduce :func:`convert_to_celsius`'s clamp exactly.

    Evaluated by running the float32 conversion over every DN once, so an
    integer stack keeps precisely the samples the Celsius path keeps: the
    clamp is applied to the same float32 value it always was, and the bound
    is read back as a DN. The kept set is contiguous, which is asserted.
    """
    dn = np.arange(1, DN_MAX + 1, dtype=np.uint16)
    celsius = convert_to_celsius(xr.DataArray(dn.astype(np.float32), dims=["dn"]))
    kept = np.flatnonzero(np.isfinite(np.asarray(celsius.values)))
    if kept.size == 0:
        msg = f"No DN survives the clamp [{settings.lst_valid_min}, {settings.lst_valid_max}] C"
        raise ValueError(msg)
    lo, hi = int(dn[kept[0]]), int(dn[kept[-1]])
    if kept.size != hi - lo + 1:
        msg = "The Celsius clamp does not keep a contiguous DN range"
        raise AssertionError(msg)
    return lo, hi


def dn_stack(
    data: xr.Dataset, qa_band: str = "qa_pixel", lwir_band: str = "lwir11"
) -> xr.DataArray:
    """The carried form: ``uint16`` DN with :data:`DN_SENTINEL` for no data.

    Keeps exactly the samples ``convert_to_celsius(apply_qa_mask(data)[lwir])``
    keeps (QA-clear, not fill, inside the plausibility clamp) and writes 0
    everywhere else. Values are the source DN, untouched. Two bytes per
    element instead of the four a float32 Celsius stack costs.

    A float ``lwir11`` is a test fixture, never a load: it is rounded to the
    nearest DN first. Production loads are ``uint16`` end to end.
    """
    lwir = data[lwir_band]
    if not np.issubdtype(lwir.dtype, np.integer):
        lwir = xr.where(lwir.notnull(), lwir.round(), DN_SENTINEL).astype(np.uint16)
    lo, hi = dn_clamp_bounds()
    valid = create_qa_mask(data[qa_band]) & (lwir >= lo) & (lwir <= hi)
    stack = lwir.where(valid, DN_SENTINEL)
    if stack.dtype != np.uint16:  # pragma: no cover - xarray keeps the dtype for an int fill
        stack = stack.astype(np.uint16)
    return stack


def celsius_stack(stack: xr.DataArray) -> xr.DataArray:
    """A float32 Celsius view of a DN stack, bit-identical to the old path.

    ``stack.where(stack != 0)`` is the same float32 DN array
    ``apply_qa_mask`` produced, and :func:`convert_to_celsius` then applies
    the same arithmetic, so an estimator reading this sees the values it
    always saw. Lazy: nothing is materialized until a consumer computes.
    """
    return convert_to_celsius(stack.where(stack != DN_SENTINEL))
