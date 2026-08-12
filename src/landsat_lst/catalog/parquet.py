"""Mirror the collection's items as one spatially ordered GeoParquet table.

A reader that wants every tile's footprint should not have to fetch seven
hundred JSON documents to get them. Portolan asks a raster collection whose
scenes are items to publish that mirror at ``items.parquet``
(``PORTO-FMT-040``) and to register it as a collection-level asset carrying the
``collection-mirror`` role (``PORTO-FMT-041``).

The mirror is built from the same in-memory items the JSON is written from,
never re-read from disk. Re-parsing would make the mirror a second, independent
reading of the same facts, and two readings can disagree; building both sides
from one object makes agreement structural rather than tested.

Rows are sorted along a Hilbert curve through the tile footprints, which is
what makes the row group's bounding box worth its bytes: a reader filtering by
area can skip a block instead of scanning it. The curve is laid over the whole
globe rather than over the tiles that happen to be published, so a tile's place
in the ordering does not shift when the published set changes.

Null ``datetime`` is the case worth naming. Every item here carries an interval
(``start_datetime``/``end_datetime``) and a null instant, and stac-geoparquet
round-trips that faithfully: the ``datetime`` column comes back a timestamp
column full of nulls while the two interval columns keep their values. That is
why this module writes through stac-geoparquet rather than assembling the
Arrow table by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pystac
from shapely.geometry import shape
from stac_geoparquet.arrow import parse_stac_items_to_arrow, to_parquet

if TYPE_CHECKING:
    from pathlib import Path

#: The filename Portolan looks for, and the only one it recognises as a mirror.
MIRROR_FILENAME = "items.parquet"

#: The media type a client uses to decide it can read the file.
PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"

#: The asset key is the producer's choice -- the stac-geoparquet spec names
#: none -- so this follows the community convention the large public catalogs
#: use, which keeps readers that look the file up by key working.
MIRROR_ASSET_KEY = "geoparquet-items"

#: ``collection-mirror`` is the role Portolan requires; ``stac-items`` is the
#: community one that predates it. Both travel, so either reader finds the file.
MIRROR_ROLES = ("stac-items", "collection-mirror")

_MIRROR_TITLE = "STAC items as GeoParquet"

#: The curve is rescaled into this extent rather than into the extent of the
#: tiles being published, so the ordering of a tile is a property of where it
#: is on Earth and not of which of its neighbours happened to finish.
_GLOBAL_BOUNDS = (-180.0, -90.0, 180.0, 90.0)

#: 2^16 cells per axis over 360 degrees is about 5.5 mdeg, far finer than the
#: five-degree tile it has to separate.
_HILBERT_LEVEL = 16

#: Portolan caps a row group at 150,000 rows. stac-geoparquet writes one row
#: group per record batch, so the batch size is the cap: bounding it here bounds
#: the row group. A global build writes about seven hundred rows, so the limit
#: is never reached, but the file's conformance is then a decision rather than
#: an accident of whatever default the writer happens to carry.
_ROW_GROUP_ROWS = 50_000


def hilbert_order(items: list[pystac.Item]) -> list[pystac.Item]:
    """The items ordered along a Hilbert curve through their footprints.

    ``hilbert_distance`` maps each geometry's midpoint onto the curve, so for
    the five-degree tiles here the ordering is by tile centroid. Ties break on
    the item id, which cannot repeat, so the ordering is total and a rebuild
    over the same tiles produces the same file.
    """
    footprints = gpd.GeoSeries([shape(item.geometry) for item in items], crs="EPSG:4326")
    distances = footprints.hilbert_distance(total_bounds=_GLOBAL_BOUNDS, level=_HILBERT_LEVEL)
    ranked = sorted(
        zip(distances.tolist(), items, strict=True), key=lambda pair: (pair[0], pair[1].id)
    )
    return [item for _distance, item in ranked]


def mirror_asset() -> pystac.Asset:
    """The collection-level asset that registers the mirror."""
    return pystac.Asset(
        href=f"./{MIRROR_FILENAME}",
        title=_MIRROR_TITLE,
        media_type=PARQUET_MEDIA_TYPE,
        roles=list(MIRROR_ROLES),
    )


def write_items_parquet(items: list[pystac.Item], directory: Path) -> Path:
    """Write the items to ``items.parquet`` in ``directory``, spatially ordered.

    ``include_self_link=False`` matches what a self-contained catalog writes to
    disk, so the mirror's ``links`` reproduce the item documents rather than a
    variant of them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MIRROR_FILENAME
    ordered = hilbert_order(items)
    batches = parse_stac_items_to_arrow(
        [item.to_dict(include_self_link=False) for item in ordered],
        chunk_size=_ROW_GROUP_ROWS,
    )
    to_parquet(batches, path)
    return path


def write_item_mirror(
    collection: pystac.Collection, directory: Path, items: list[pystac.Item]
) -> Path:
    """Write the mirror beside the collection and register it as its asset.

    Registration happens on the in-memory collection, so the caller must still
    be holding it before it saves; a mirror on disk that the collection does
    not declare is invisible to every reader and an error under ``PTL-MIR-002``
    the moment anything finds it.
    """
    path = write_items_parquet(items, directory)
    collection.add_asset(MIRROR_ASSET_KEY, mirror_asset())
    return path
