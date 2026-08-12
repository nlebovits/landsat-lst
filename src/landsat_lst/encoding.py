"""LST uint16 encoding contract.

This module is the single home of the encoding constants and the encoder.
Every writer (COG export, tests, analysis scripts) imports from here so the
contract cannot drift between output formats.

Encoding (LST bands only):
- Scale: 0.01, Offset: -50.0
- Decode: celsius = dn * 0.01 + (-50.0)
- Fill value: 0 (uint16)
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# Encoding constants for LST bands (lst_p95)
LST_SCALE: float = 0.01
LST_OFFSET: float = -50.0
LST_NODATA_FLOAT: float = -9999.0
LST_FILL_VALUE: int = 0

# DN 0 is reserved for fill, so the encodable range is 1..65535.
LST_MIN_DN: int = 1
LST_MAX_DN: int = 65535

# Lowest DN carrying a physically meaningful LST. DN 0 is fill, and DN 1 is
# reachable only from values sitting on the encoding floor (-49.99 C). A
# hot-season P95 over land between 60S and 60N never gets that cold, so DN 1
# marks a failed retrieval rather than a real temperature. See issue #24.
LST_MIN_TRUSTED_DN: int = 2

# Coldest Celsius value the pipeline will keep: the bottom of the DN 2 bucket.
# Anything below encodes to DN 0 or DN 1 and is treated as missing.
LST_MIN_TRUSTED_C: float = LST_OFFSET + LST_MIN_TRUSTED_DN * LST_SCALE


def encode_lst_uint16(data: xr.DataArray) -> xr.DataArray:
    """Encode LST float values to uint16 with scale/offset.

    Formula: dn = (celsius - offset) / scale
    Decode:  celsius = dn * scale + offset

    Values outside the encodable range become fill (DN 0) rather than being
    clipped to the nearest representable DN. Clipping would turn an
    out-of-range value such as -124 C into a believable -49.99 C, which is
    exactly how the isolated anomaly pixels in issue #24 were produced.
    ``convert_to_celsius`` already drops implausible observations, so anything
    arriving here out of range signals a defect and is better recorded as
    missing than as a plausible temperature.

    Args:
        data: LST values in Celsius (float32), nodata=-9999.0

    Returns:
        Encoded uint16 values, fill_value=0
    """
    # Convert celsius to DN: dn = (celsius - offset) / scale
    dn = (data - LST_OFFSET) / LST_SCALE

    # Out-of-range values (including NaN, which fails both comparisons) become
    # fill. DN 0 is reserved for fill, so the floor is 1.
    in_range = (dn >= LST_MIN_DN) & (dn <= LST_MAX_DN)
    dn = xr.where(in_range, dn, LST_FILL_VALUE)

    # Set nodata pixels to fill value (0)
    dn = xr.where(data == LST_NODATA_FLOAT, LST_FILL_VALUE, dn)
    dn = xr.where(np.isnan(data), LST_FILL_VALUE, dn)

    return dn.astype(np.uint16)
