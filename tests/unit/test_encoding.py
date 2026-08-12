"""Unit tests for the LST uint16 encoding contract.

These live apart from any writer so the contract is tested once rather than
per output format. They run as unit tests: the encoder is pure numpy/xarray
and needs no I/O.
"""

import numpy as np
import xarray as xr

from landsat_lst.encoding import LST_OFFSET, LST_SCALE, encode_lst_uint16


def test_encode_lst_uint16():
    """Test uint16 encoding preserves values correctly."""
    celsius_values = np.array([0.0, 25.0, 35.0, 50.0, -10.0])
    data = xr.DataArray(celsius_values, dims=["x"])

    encoded = encode_lst_uint16(data)

    assert encoded.dtype == np.uint16

    # Decode and verify
    decoded = encoded.values * LST_SCALE + LST_OFFSET
    np.testing.assert_array_almost_equal(decoded, celsius_values, decimal=2)


def test_encode_lst_uint16_nodata():
    """Test that nodata values are encoded as 0."""
    data = xr.DataArray(
        np.array([25.0, -9999.0, np.nan, 35.0]),
        dims=["x"],
    )

    encoded = encode_lst_uint16(data)

    assert encoded.values[0] != 0  # Valid value
    assert encoded.values[1] == 0  # -9999.0 -> 0
    assert encoded.values[2] == 0  # NaN -> 0
    assert encoded.values[3] != 0  # Valid value


def test_encode_lst_uint16_out_of_range_becomes_fill():
    """Out-of-range values become fill instead of being clipped (issue #24).

    Clipping turned an impossible -124 C into a believable -49.99 C, which is
    how the isolated anomaly pixels of issue #24 reached the output. Values
    that cannot be represented are recorded as missing instead.
    """
    data = xr.DataArray(
        np.array([-124.0, -50.0, 700.0, 25.0]),
        dims=["x"],
    )

    encoded = encode_lst_uint16(data)

    assert encoded.values[0] == 0  # below the encodable floor
    assert encoded.values[1] == 0  # collides with the fill value
    assert encoded.values[2] == 0  # above the uint16 ceiling
    assert encoded.values[3] != 0  # ordinary value is untouched


def test_encode_lst_uint16_never_reports_floor_temperature():
    """No input decodes to -49.99 C, the signature of the issue #24 anomaly."""
    data = xr.DataArray(
        np.array([-200.0, -124.0, -60.0, -50.0, -49.995]),
        dims=["x"],
    )

    decoded = encode_lst_uint16(data).values * LST_SCALE + LST_OFFSET

    assert not np.any(np.isclose(decoded, -49.99)), f"floor value resurfaced: {decoded}"
