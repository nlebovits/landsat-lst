"""Check that a tile's published COGs are real, reachable, and readable.

The storage backend can only answer whether two objects exist. That is not the
question a published dataset has to pass. The question is whether the person who
pastes a URL into QGIS gets a raster that decodes to Celsius, which needs the
object opened over the public HTTPS host with no credentials, and its dtype,
nodata, scale, offset, and overviews inspected.

This is the S1 rehearsal gate in the publication runbook: it runs against bare
COGs, before any catalog exists. ``landsat-lst catalog validate --live`` answers
a different and later question, whether the *hosting* of a built catalog
supports range requests and CORS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from landsat_lst.storage import PRODUCTS, get_storage

if TYPE_CHECKING:
    from landsat_lst.storage import StorageBackend

log = structlog.get_logger()


@dataclass
class AssetCheck:
    """What one published asset turned out to be."""

    product: str
    key: str
    url: str
    exists: bool
    error: str | None = None
    dtype: str | None = None
    shape: tuple[int, int] | None = None
    nodata: float | None = None
    scale: float | None = None
    offset: float | None = None
    overviews: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exists and self.error is None


@dataclass
class TileCheck:
    """Both assets of one tile-window, as a public consumer sees them."""

    tile: str
    window: str
    assets: list[AssetCheck]

    @property
    def ok(self) -> bool:
        return len(self.assets) == len(PRODUCTS) and all(a.ok for a in self.assets)


def public_url(
    window: str, tile: str, product: str, *, storage: StorageBackend | None = None
) -> str:
    """The URL a QGIS or gdalinfo user pastes for one asset."""
    from landsat_lst.catalog.spec import spec_for_window  # noqa: PLC0415

    storage = storage or get_storage()
    read_base = spec_for_window(window).read_base_url
    return f"{read_base}{storage.cog_key(window, tile, product)}"


def _open_asset(product: str, key: str, url: str) -> AssetCheck:
    """Open one asset over public HTTPS and read back what it declares.

    Opened unauthenticated on purpose. A tile that reads only with credentials
    is not published, whatever the bucket listing says.
    """
    import rasterio  # noqa: PLC0415

    try:
        with rasterio.open(url) as src:
            return AssetCheck(
                product=product,
                key=key,
                url=url,
                exists=True,
                dtype=src.dtypes[0],
                shape=(src.height, src.width),
                nodata=src.nodata,
                scale=src.scales[0],
                offset=src.offsets[0],
                overviews=list(src.overviews(1)),
            )
    except Exception as e:
        return AssetCheck(product=product, key=key, url=url, exists=True, error=str(e))


def verify_tile(tile: str, window: str, *, storage: StorageBackend | None = None) -> TileCheck:
    """Check both of one tile's assets, in storage and over public HTTPS.

    Args:
        tile: Tile name (``"N40W075"``).
        window: Window label (``"2021-2025"``).
        storage: Backend holding the objects (default from
            :func:`landsat_lst.storage.get_storage`).

    Returns:
        A :class:`TileCheck` whose ``ok`` is True only when both assets exist
        and both open unauthenticated.
    """
    storage = storage or get_storage()
    logger = log.bind(tile=tile, window=window)
    logger.info("verify_start")

    # One existence call, not one per asset: a tile is complete only with both,
    # so a half-written tile fails as a tile rather than as one stray asset.
    present = storage.cog_exists(window, tile)

    assets = []
    for product in PRODUCTS:
        key = storage.cog_key(window, tile, product)
        url = public_url(window, tile, product, storage=storage)
        if present:
            assets.append(_open_asset(product, key, url))
        else:
            assets.append(
                AssetCheck(
                    product=product,
                    key=key,
                    url=url,
                    exists=False,
                    error="tile incomplete in storage (both assets required)",
                )
            )

    check = TileCheck(tile=tile, window=window, assets=assets)
    logger.info("verify_finished", ok=check.ok)
    return check
