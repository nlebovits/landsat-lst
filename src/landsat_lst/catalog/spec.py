"""Frozen description of the catalog this repository publishes.

Everything that is a naming or policy decision rather than a fact read off the
data lives here: identifiers, human-readable titles, prose, providers, licence,
and the URIs of the extensions the objects declare. :data:`DEFAULT_SPEC`
describes the production dataset; :func:`spec_for_window` rebuilds the same
shape for a different observation window.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The one Portolan profile URI every catalog and collection declares. Items do
#: not declare it: they inherit conformance from their collection.
PORTOLAN_SCHEMA_URI = "https://schemas.portolan-sdi.org/portolan/v0.1.0/schema.json"

#: Declared by items because their assets carry ``file:size``.
FILE_EXTENSION_URI = "https://stac-extensions.github.io/file/v2.1.0/schema.json"

#: Declared by items because their asset bands carry ``raster:*`` fields.
RASTER_EXTENSION_URI = "https://stac-extensions.github.io/raster/v2.0.0/schema.json"

#: The exact media type a Portolan validator recognises as a COG.
COG_MEDIA_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"

#: Asset keys. Both are declared in the collection's ``item_assets``.
LST_ASSET_KEY = "lst_p95"
QA_ASSET_KEY = "qa_count"

DEFAULT_WINDOW = "2021-2025"
SOURCE_COOP_SLUG = "nlebovits/landsat-lst"
SOURCE_COOP_READ_BASE = "https://data.source.coop/nlebovits/landsat-lst/"

_USGS_URL = "https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products"
_MAINTAINER_URL = "https://github.com/nlebovits/landsat-lst"
_MAINTAINER_EMAIL = "nlebovits@pm.me"


@dataclass(frozen=True)
class ProviderSpec:
    """One entry of a collection's ``providers`` array.

    Order is meaningful: Portolan requires exactly one provider carrying the
    ``host`` role and requires it to be the last element.
    """

    name: str
    roles: tuple[str, ...]
    url: str | None = None
    email: str | None = None

    def to_dict(self) -> dict[str, object]:
        """The STAC provider object, omitting the contact fields not set."""
        provider: dict[str, object] = {"name": self.name, "roles": list(self.roles)}
        if self.url is not None:
            provider["url"] = self.url
        if self.email is not None:
            provider["email"] = self.email
        return provider


@dataclass(frozen=True)
class CatalogSpec:
    """Naming and policy decisions for one published observation window."""

    window: str
    catalog_id: str
    catalog_title: str
    catalog_description: str
    collection_id: str
    collection_title: str
    collection_description: str
    license: str
    providers: tuple[ProviderSpec, ...]
    colormap: str
    source_coop_slug: str
    read_base_url: str

    @property
    def start_year(self) -> int:
        """First calendar year of the window."""
        return int(self.window.split("-")[0])

    @property
    def end_year(self) -> int:
        """Last calendar year of the window, inclusive."""
        return int(self.window.split("-")[-1])

    @property
    def start_datetime(self) -> str:
        """RFC 3339 instant every item declares as ``start_datetime``."""
        return f"{self.start_year}-01-01T00:00:00Z"

    @property
    def end_datetime(self) -> str:
        """RFC 3339 instant every item declares as ``end_datetime``."""
        return f"{self.end_year}-12-31T23:59:59Z"


_CATALOG_DESCRIPTION = (
    "Global land surface temperature composites derived from Landsat "
    "Collection 2 Level-2 surface temperature. One collection per observation "
    "window, each cut from a single global 1/3600-degree grid and published as "
    "one cloud-optimized GeoTIFF pair per five-degree tile."
)


def _collection_description(window: str) -> str:
    return (
        f"Hot-season land surface temperature for {window}, expressed as the "
        "95th percentile of every cloud-free Landsat 8/9 observation pooled "
        "across the window, on a 1/3600-degree (about 30 m) WGS84 grid. Each "
        "five-degree tile carries the percentile composite and a twelve-band "
        "monthly count of the observations behind it. Scenes are de-striped "
        "against a per-pixel monthly climatology and scenes whose offset "
        "cannot be trusted are discarded rather than corrected."
    )


def spec_for_window(window: str = DEFAULT_WINDOW) -> CatalogSpec:
    """Build the catalog spec for one observation window, e.g. ``"2021-2025"``."""
    return CatalogSpec(
        window=window,
        catalog_id="landsat-lst",
        catalog_title="Landsat Land Surface Temperature",
        catalog_description=_CATALOG_DESCRIPTION,
        collection_id=f"lst-p95-{window}",
        collection_title=f"Land Surface Temperature 95th Percentile, {window}",
        collection_description=_collection_description(window),
        license="CC0-1.0",
        providers=(
            ProviderSpec(
                name="United States Geological Survey",
                roles=("licensor", "processor"),
                url=_USGS_URL,
            ),
            ProviderSpec(
                name="Nissim Lebovits, Radiant Earth",
                roles=("producer", "host"),
                url=_MAINTAINER_URL,
                email=_MAINTAINER_EMAIL,
            ),
        ),
        colormap="inferno",
        source_coop_slug=SOURCE_COOP_SLUG,
        read_base_url=SOURCE_COOP_READ_BASE,
    )


#: The production dataset: the 2021-2025 window on Source Cooperative.
DEFAULT_SPEC = spec_for_window()
