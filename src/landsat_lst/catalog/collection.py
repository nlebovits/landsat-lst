"""Assemble the root catalog and its single collection.

The shape is deliberately flat: one root catalog, one collection, one item per
tile. Portolan forbids nested collections, and a second level would buy nothing
here -- the tiles are a grid, not a hierarchy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pystac

from landsat_lst.catalog.spec import (
    COG_MEDIA_TYPE,
    LST_ASSET_KEY,
    PORTOLAN_SCHEMA_URI,
    QA_ASSET_KEY,
)

if TYPE_CHECKING:
    from landsat_lst.catalog.spec import CatalogSpec

_MARKDOWN = "text/markdown"

_ITEM_ASSET_TITLES = {
    LST_ASSET_KEY: "Land surface temperature, 95th percentile",
    QA_ASSET_KEY: "Valid observation count per calendar month",
}
_ITEM_ASSET_ROLES = {LST_ASSET_KEY: ["data"], QA_ASSET_KEY: ["data", "quality"]}


def _item_assets() -> dict[str, pystac.ItemAssetDefinition | dict[str, Any]]:
    """The two assets every item in this collection carries."""
    return {
        key: pystac.ItemAssetDefinition.create(
            title=title,
            description=None,
            media_type=COG_MEDIA_TYPE,
            roles=_ITEM_ASSET_ROLES[key],
        )
        for key, title in _ITEM_ASSET_TITLES.items()
    }


def _spatial_extent(items: list[pystac.Item]) -> pystac.SpatialExtent:
    """The union of every item's bounding box."""
    boxes = [item.bbox for item in items if item.bbox is not None]
    if not boxes:
        return pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]])
    return pystac.SpatialExtent(
        [
            [
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            ]
        ]
    )


def _temporal_extent(spec: CatalogSpec) -> pystac.TemporalExtent:
    """The observation window, which every item shares."""
    from pystac.utils import str_to_datetime  # noqa: PLC0415 - keep import cost local

    start = str_to_datetime(spec.start_datetime)
    end = str_to_datetime(spec.end_datetime)
    return pystac.TemporalExtent([[start, end]])


def _add_doc_links(obj: pystac.Catalog) -> None:
    """Point the object at the AGENTS.md and README.md beside it."""
    obj.add_link(pystac.Link(rel="agents", target="./AGENTS.md", media_type=_MARKDOWN))
    obj.add_link(pystac.Link(rel="describedby", target="./README.md", media_type=_MARKDOWN))


def build_collection(items: list[pystac.Item], spec: CatalogSpec) -> pystac.Collection:
    """The one collection, holding every tile item."""
    collection = pystac.Collection(
        id=spec.collection_id,
        title=spec.collection_title,
        description=spec.collection_description,
        extent=pystac.Extent(_spatial_extent(items), _temporal_extent(spec)),
        license=spec.license,
        providers=[pystac.Provider.from_dict(p.to_dict()) for p in spec.providers],
        stac_extensions=[PORTOLAN_SCHEMA_URI],
    )
    collection.item_assets = _item_assets()
    _add_doc_links(collection)
    for item in items:
        collection.add_item(item, title=item.properties["title"])
    return collection


def build_root_catalog(collection: pystac.Collection, spec: CatalogSpec) -> pystac.Catalog:
    """The root catalog, with the collection linked beneath it."""
    catalog = pystac.Catalog(
        id=spec.catalog_id,
        title=spec.catalog_title,
        description=spec.catalog_description,
        stac_extensions=[PORTOLAN_SCHEMA_URI],
    )
    _add_doc_links(catalog)
    catalog.add_child(collection, title=spec.collection_title)
    return catalog
