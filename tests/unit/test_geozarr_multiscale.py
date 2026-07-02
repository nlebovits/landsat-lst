"""Tier 0 smoke test: GeoZarr conventions + multiscale overviews + compression.

Fast, synthetic, no network. Validates the write path produces:
- GeoZarr `proj`/`spatial`/`multiscales` metadata on the parent and level groups,
- coarsened overview levels with correct shapes/scales,
- Blosc compression on every array,
- an atomic single-commit Icechunk pyramid that reads back correctly,
- the fill-masking invariant (overview LST never collapses toward the -50 degC fill).

See plan: why-not-use-icechunk-deep-flask.md (Tier 0).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.codecs import BloscCodec

from landsat_lst.storage import IcechunkStorage
from landsat_lst.zarr_writer import (
    LST_NODATA_FLOAT,
    LST_OFFSET,
    LST_SCALE,
    build_overviews,
)

FACTORS = [4, 16]
SIZE = 256  # divisible by all factors -> clean coarsening


def _decode(dn: np.ndarray) -> np.ndarray:
    """Decode uint16 DN back to Celsius (valid where dn > 0)."""
    return dn.astype(np.float32) * LST_SCALE + LST_OFFSET


@pytest.fixture
def synthetic_composite() -> xr.Dataset:
    """A small realistic composite: warm land, a NaN ocean block, a -9999 gap block."""
    rng = np.random.default_rng(42)
    lst = rng.uniform(20.0, 45.0, (SIZE, SIZE)).astype(np.float32)
    lst[:32, :32] = np.nan  # ocean (land-masked)
    lst[-32:, -32:] = LST_NODATA_FLOAT  # orbital gap
    qa = rng.integers(1, 60, (SIZE, SIZE)).astype(np.int16)

    latitude = np.linspace(40.0, 35.0, SIZE)
    longitude = np.linspace(-75.0, -70.0, SIZE)
    return xr.Dataset(
        {
            "lst_p95": (["latitude", "longitude"], lst),
            "qa_count": (["latitude", "longitude"], qa),
        },
        coords={"latitude": latitude, "longitude": longitude},
        attrs={"tile": "TEST", "year": 2024},
    )


def test_build_overviews_excludes_fill() -> None:
    """Coarsening must average only valid pixels, never the -9999/NaN fill."""
    lat = np.array([40.0, 39.9])
    lon = np.array([-75.0, -74.9])
    # A single 2x2 block: two valid 25 degC pixels + one gap + one ocean.
    lst = np.array([[25.0, 25.0], [LST_NODATA_FLOAT, np.nan]], dtype=np.float32)
    qa = np.array([[10, 10], [0, 0]], dtype=np.int16)
    comp = xr.Dataset(
        {"lst_p95": (["latitude", "longitude"], lst), "qa_count": (["latitude", "longitude"], qa)},
        coords={"latitude": lat, "longitude": lon},
    )

    levels = build_overviews(comp, [2])
    assert [name for name, _f, _ds in levels] == ["0", "1"]

    _name, factor, overview = levels[1]
    assert factor == 2
    # Mean of the two valid 25s only -> 25, NOT dragged toward fill.
    assert float(overview["lst_p95"].values[0, 0]) == pytest.approx(25.0)


def test_write_geozarr_multiscale_icechunk(tmp_path, synthetic_composite) -> None:
    storage = IcechunkStorage.from_local(tmp_path / "icechunk")
    session = storage.writable_session()
    group = "2024/TEST"

    from landsat_lst.zarr_writer import write_zarr

    write_zarr(synthetic_composite, session, group=group, factors=FACTORS)
    commit_id = session.commit("tier0 multiscale test")
    assert commit_id

    # --- Atomic single commit -------------------------------------------------
    ancestry = list(storage.repo.ancestry(branch="main"))
    assert len(ancestry) == 2, "expected initial + one pyramid commit"

    rs = storage.readonly_session()

    # --- Parent group: GeoZarr conventions -----------------------------------
    parent = zarr.open_group(rs.store, path=group, mode="r")
    assert parent.attrs["proj:code"] == "EPSG:4326"
    assert parent.attrs["spatial:dimensions"] == ["latitude", "longitude"]
    assert len(parent.attrs["spatial:transform"]) == 6
    assert parent.attrs["spatial:shape"] == [SIZE, SIZE]

    layout = parent.attrs["multiscales"]["layout"]
    assert [e["asset"] for e in layout] == ["0", "1", "2"]
    assert layout[0]["transform"]["scale"] == [1.0, 1.0]
    assert layout[1]["transform"]["scale"] == [4.0, 4.0]
    assert layout[1]["derived_from"] == "0"
    assert layout[2]["transform"]["scale"] == [16.0, 16.0]

    # --- Levels: shapes, dtype, compression, per-level GeoZarr attrs ----------
    expected_shape = {"0": SIZE, "1": SIZE // 4, "2": SIZE // 16}
    native_pixel = None
    for name, exp in expected_shape.items():
        ds = xr.open_zarr(rs.store, group=f"{group}/{name}")
        assert ds.sizes["latitude"] == exp
        assert ds.sizes["longitude"] == exp
        assert ds["lst_p95"].dtype == np.uint16
        assert ds["qa_count"].dtype == np.uint8
        assert "proj:code" in ds.attrs
        assert "_CRS" in ds.attrs  # GDAL metadata preserved alongside GeoZarr

        # Compression codec present on the array.
        arr = zarr.open_array(rs.store, path=f"{group}/{name}/lst_p95", mode="r")
        assert any(isinstance(c, BloscCodec) for c in arr.metadata.codecs)

        # Pixel size grows ~4x per level (sanity on the affine transform).
        pixel = abs(float(ds["longitude"][1] - ds["longitude"][0]))
        if name == "0":
            native_pixel = pixel
        elif name == "1":
            assert pixel == pytest.approx(native_pixel * 4, rel=0.05)

    # --- Physical sanity: the fill-mask bug surfaces here ---------------------
    for name in ("0", "1", "2"):
        ds = xr.open_zarr(rs.store, group=f"{group}/{name}")
        dn = ds["lst_p95"].values
        valid = _decode(dn[dn > 0])
        assert valid.size > 0
        assert valid.min() >= -20.0, f"level {name}: fill leaked into overview means"
        assert valid.max() <= 60.0
