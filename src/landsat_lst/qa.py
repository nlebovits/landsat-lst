"""QA filtering and masking functions for Landsat data."""

from typing import overload

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
# The composite's integer representation (issue #136).
#
# The published LST is uint16 at 0.01 C. The source is uint16 DN at
# DN_SCALE_K = 0.00341802 C per DN, three times finer than the product. So the
# composite stack is kept as DN, with 0 as "no observation", and converted to
# Celsius only on the 2-D P95 result. Every stack-proportional term in a
# composite shard's working set halves against the float32 stack this
# replaced (docs/findings-composite-precision-audit.md). The offset pass is
# untouched: it still estimates from a float32 Celsius stack, and the shard
# applies its estimate rounded to whole DN, which moves the P95 by at most
# half a DN (0.0017 C) and the encoded product by at most one DN.
# ---------------------------------------------------------------------------

#: Landsat Collection 2 Level-2 surface temperature scaling, Kelvin per DN.
DN_SCALE_K: float = 0.00341802
#: Additive term of the DN-to-Celsius map: ``149.0 K`` minus ``273.15``.
DN_OFFSET_C: float = 149.0 - 273.15
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
    """The composite's input: ``uint16`` DN with :data:`DN_SENTINEL` for no data.

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


@overload
def dn_to_celsius(dn: xr.DataArray) -> xr.DataArray: ...
@overload
def dn_to_celsius(dn: np.ndarray) -> np.ndarray: ...
def dn_to_celsius(dn: xr.DataArray | np.ndarray) -> xr.DataArray | np.ndarray:
    """The affine DN-to-Celsius map, for a float64 P95 in DN units."""
    return dn * DN_SCALE_K + DN_OFFSET_C


def celsius_stack(stack: xr.DataArray) -> xr.DataArray:
    """A float32 Celsius view of a DN stack, bit-identical to the old path.

    ``stack.where(stack != 0)`` is the same float32 DN array
    ``apply_qa_mask`` produced, and :func:`convert_to_celsius` then applies
    the same arithmetic, so an estimator reading this sees the values it
    always saw. Lazy: nothing is materialized until a consumer computes.
    """
    return convert_to_celsius(stack.where(stack != DN_SENTINEL))


def offset_dn_shift(offset: xr.DataArray) -> xr.DataArray:
    """Per-scene offsets in Celsius, rounded to whole DN, as ``int32``."""
    values = np.asarray(offset.values, dtype=np.float64)
    shift = np.rint(values / DN_SCALE_K).astype(np.int32)
    return xr.DataArray(shift, dims=offset.dims, coords=offset.coords)


def debias_dn(stack: xr.DataArray, offset: xr.DataArray) -> xr.DataArray:
    """Subtract each scene's offset from a DN stack, staying ``uint16``.

    The shift is ``round(offset / DN_SCALE_K)``. A shifted DN that leaves
    ``[1, 65535]`` becomes the sentinel: it cannot happen inside the clamp
    and the offset cap (-65 to 95 C is DN 17,261 to 64,073), and if it did
    the sample would be one no encoder could represent anyway.
    """
    shifted = stack.astype(np.int32) - offset_dn_shift(offset)
    valid = (stack != DN_SENTINEL) & (shifted >= 1) & (shifted <= DN_MAX)
    return shifted.where(valid, DN_SENTINEL).astype(np.uint16)
