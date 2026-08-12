"""The items.parquet mirror: agreement with the items, and spatial ordering.

Two questions are worth a test here, and they need different fixtures.

Agreement is a per-field question, so it is asked of a real built catalog whose
items come from real COG headers: the mirror's rows are compared against the
item documents on disk, field by field, because those documents are what a
reader would otherwise have fetched.

Ordering is a population question -- Portolan's row-order rule only engages
above two hundred rows -- so it is asked of a metadata-only catalog spread over
three hundred real land tiles. Those items name COGs that were never written,
which is deliberate: an asset whose bytes are absent is skipped by the checks
that read bytes, leaving the Parquet checks to run alone and cheaply. A
deliberately shuffled mirror over the same items is checked too, so a passing
ordering test cannot be one that never had teeth.
"""

from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import pystac
import pytest
from rashid import validate
from shapely import from_wkb
from shapely.geometry import shape
from stac_geoparquet.arrow import parse_stac_items_to_arrow, to_parquet

from landsat_lst.catalog import build_catalog
from landsat_lst.catalog.collection import build_collection, build_root_catalog
from landsat_lst.catalog.parquet import (
    MIRROR_ASSET_KEY,
    MIRROR_FILENAME,
    MIRROR_ROLES,
    PARQUET_MEDIA_TYPE,
    hilbert_order,
    mirror_asset,
)
from landsat_lst.catalog.spec import COG_MEDIA_TYPE, DEFAULT_SPEC, LST_ASSET_KEY
from landsat_lst.tiling import generate_land_tiles, parse_tile_name, tile_geobox
from tests.cog_fixtures import write_source_tree

if TYPE_CHECKING:
    from pathlib import Path

    from landsat_lst.models import TileId

TILES = ("N40W075", "N35W120", "S30W060")

#: Comfortably past the two-hundred-row floor below which Portolan's row-order
#: rule declines to judge (ten chunks of twenty rows).
SPREAD_TILES = 320

#: The rule this file exists to keep satisfied.
ROW_ORDER_RULE = "PTL-DAT-006"

_COORD_TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def built_catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small catalog built the production way, mirror included."""
    root = tmp_path_factory.mktemp("mirror")
    source = write_source_tree(root / "cogs", TILES)
    return build_catalog(source, root / "catalog", tiles=TILES)


def collection_dir(catalog: Path) -> Path:
    return catalog / DEFAULT_SPEC.collection_id


def read_mirror(catalog: Path) -> dict[str, Any]:
    """The mirror's columns as plain Python, keyed by column name."""
    table = pq.read_table(collection_dir(catalog) / MIRROR_FILENAME)
    return {name: table.column(name).to_pylist() for name in table.schema.names}


def read_item_json(catalog: Path, item_id: str) -> dict[str, Any]:
    path = collection_dir(catalog) / item_id / f"{item_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_mirror_is_written_beside_the_collection(built_catalog: Path) -> None:
    assert (collection_dir(built_catalog) / MIRROR_FILENAME).is_file()


def test_mirror_is_registered_as_a_collection_level_asset(built_catalog: Path) -> None:
    """PTL-MIR-002 is an error once the file exists but the roles are wrong."""
    collection = json.loads(
        (collection_dir(built_catalog) / "collection.json").read_text(encoding="utf-8")
    )
    asset = collection["assets"][MIRROR_ASSET_KEY]
    assert asset["href"] == f"./{MIRROR_FILENAME}"
    assert asset["type"] == PARQUET_MEDIA_TYPE
    assert asset["roles"] == list(MIRROR_ROLES)
    assert "collection-mirror" in asset["roles"]


def test_every_item_appears_exactly_once(built_catalog: Path) -> None:
    ids = read_mirror(built_catalog)["id"]
    assert sorted(ids) == sorted(TILES)
    assert len(ids) == len(set(ids))


def test_row_fields_reproduce_the_item_documents(built_catalog: Path) -> None:
    """Id, bbox, geometry and both datetimes, against the JSON on disk."""
    columns = read_mirror(built_catalog)
    for row, item_id in enumerate(columns["id"]):
        item = read_item_json(built_catalog, item_id)
        box = columns["bbox"][row]
        assert [box["xmin"], box["ymin"], box["xmax"], box["ymax"]] == pytest.approx(
            item["bbox"], abs=_COORD_TOLERANCE
        )
        assert from_wkb(columns["geometry"][row]).equals_exact(
            shape(item["geometry"]), tolerance=_COORD_TOLERANCE
        )
        assert columns["start_datetime"][row].isoformat() == item["properties"][
            "start_datetime"
        ].replace("Z", "+00:00")
        assert columns["end_datetime"][row].isoformat() == item["properties"][
            "end_datetime"
        ].replace("Z", "+00:00")


def test_null_instant_survives_the_round_trip(built_catalog: Path) -> None:
    """Items carry an interval and a null ``datetime``; the mirror must too.

    stac-geoparquet writes the column as a timestamp full of nulls rather than
    dropping it or inventing an instant, which is what lets this build write
    the mirror through the library instead of assembling Arrow by hand.
    """
    columns = read_mirror(built_catalog)
    assert columns["datetime"] == [None] * len(TILES)
    for item_id in columns["id"]:
        assert read_item_json(built_catalog, item_id)["properties"]["datetime"] is None


def test_rows_are_written_in_hilbert_order(built_catalog: Path) -> None:
    items = [
        pystac.Item.from_dict(read_item_json(built_catalog, tile), preserve_dict=False)
        for tile in TILES
    ]
    expected = [item.id for item in hilbert_order(items)]
    assert read_mirror(built_catalog)["id"] == expected


def test_hilbert_order_is_independent_of_input_order() -> None:
    """The curve is laid over the globe, so a rebuild reproduces the file."""
    items = [_spread_item(tile.name) for tile in _spread_tiles()]
    shuffled = list(items)
    random.Random(7).shuffle(shuffled)
    assert [item.id for item in hilbert_order(shuffled)] == [
        item.id for item in hilbert_order(items)
    ]


def test_bbox_covering_column_carries_row_group_statistics(built_catalog: Path) -> None:
    """PTL-DAT-007 reads the covering declaration, then the leaves' min/max."""
    parquet = pq.ParquetFile(collection_dir(built_catalog) / MIRROR_FILENAME)
    geo = json.loads(parquet.schema_arrow.metadata[b"geo"])
    assert geo["version"].startswith("1.1")
    covering = geo["columns"][geo["primary_column"]]["covering"]["bbox"]
    assert covering == {
        "xmin": ["bbox", "xmin"],
        "ymin": ["bbox", "ymin"],
        "xmax": ["bbox", "xmax"],
        "ymax": ["bbox", "ymax"],
    }
    group = parquet.metadata.row_group(0)
    leaves = {group.column(j).path_in_schema: group.column(j) for j in range(group.num_columns)}
    for corner in ("xmin", "ymin", "xmax", "ymax"):
        statistics = leaves[f"bbox.{corner}"].statistics
        assert statistics.min is not None
        assert statistics.max is not None


def test_row_group_stays_under_the_portolan_limit(built_catalog: Path) -> None:
    metadata = pq.ParquetFile(collection_dir(built_catalog) / MIRROR_FILENAME).metadata
    assert max(metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)) <= 150_000


# --- ordering over a realistic global spread --------------------------------


def _spread_tiles() -> list[TileId]:
    """The first few hundred real land tiles of the production grid."""
    tiles: list[TileId] = []
    for tile in generate_land_tiles():
        tiles.append(tile)
        if len(tiles) == SPREAD_TILES:
            break
    return tiles


def _spread_item(tile: str) -> pystac.Item:
    """One metadata-only item on a real tile, naming a COG that is not there."""
    box = tile_geobox(parse_tile_name(tile)).boundingbox
    bbox = [box.left, box.bottom, box.right, box.top]
    ring = [
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
        [bbox[0], bbox[1]],
    ]
    item = pystac.Item(
        id=tile,
        geometry={"type": "Polygon", "coordinates": [ring]},
        bbox=bbox,
        datetime=None,
        properties={
            "title": f"{tile} land surface temperature, {DEFAULT_SPEC.window}",
            "start_datetime": DEFAULT_SPEC.start_datetime,
            "end_datetime": DEFAULT_SPEC.end_datetime,
        },
    )
    item.add_asset(
        LST_ASSET_KEY,
        pystac.Asset(href=f"./{tile}.tif", media_type=COG_MEDIA_TYPE, roles=["data"]),
    )
    return item


def _write_spread_catalog(root: Path, *, shuffle: bool) -> Path:
    """A metadata-only catalog over the spread tiles, mirror written last.

    ``shuffle`` writes the same rows in a deliberately unordered sequence, by
    going around :func:`hilbert_order` rather than by feeding it garbage, so
    the ordering rule is exercised against a file that differs from the good
    one in row order alone.
    """
    items = [_spread_item(tile.name) for tile in _spread_tiles()]
    collection = build_collection(items, DEFAULT_SPEC)
    catalog = build_root_catalog(collection, DEFAULT_SPEC)
    catalog.normalize_hrefs(str(root))
    directory = root / DEFAULT_SPEC.collection_id
    directory.mkdir(parents=True, exist_ok=True)

    ordered = list(items)
    if shuffle:
        random.Random(11).shuffle(ordered)
    else:
        ordered = hilbert_order(ordered)
    batches = parse_stac_items_to_arrow([item.to_dict(include_self_link=False) for item in ordered])
    to_parquet(batches, directory / MIRROR_FILENAME)

    collection.add_asset(MIRROR_ASSET_KEY, mirror_asset())
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return root


def _data_findings(catalog: Path) -> list:
    report = validate(catalog, structural=False, schema=False, data=True, live=False)
    return [*report.errors, *report.warnings, *report.infos]


@pytest.fixture(scope="module")
def spread_catalogs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The ordered and shuffled mirrors, built once and read by both tests."""
    root = tmp_path_factory.mktemp("spread")
    ordered = _write_spread_catalog(root / "ordered", shuffle=False)
    shuffled = _write_spread_catalog(root / "shuffled", shuffle=True)
    return ordered, shuffled


def test_the_spread_crosses_the_row_ordering_threshold(spread_catalogs: tuple[Path, Path]) -> None:
    """Below two hundred rows the rule declines to judge, so the fixture must
    clear that floor for either of the next two tests to mean anything."""
    ordered, _shuffled = spread_catalogs
    assert SPREAD_TILES >= 200
    assert len(read_mirror(ordered)["id"]) == SPREAD_TILES


def test_hilbert_ordered_spread_satisfies_the_row_ordering_rule(
    spread_catalogs: tuple[Path, Path],
) -> None:
    ordered, _shuffled = spread_catalogs
    findings = _data_findings(ordered)
    assert [f for f in findings if f.rule_id == ROW_ORDER_RULE] == []


def test_shuffled_spread_trips_the_row_ordering_rule(
    spread_catalogs: tuple[Path, Path],
) -> None:
    """Proof that the test above is measuring something."""
    _ordered, shuffled = spread_catalogs
    findings = _data_findings(shuffled)
    assert [f.rule_id for f in findings if f.rule_id == ROW_ORDER_RULE] == [ROW_ORDER_RULE]


def test_absent_cog_bytes_do_not_fault_the_metadata_only_spread(
    spread_catalogs: tuple[Path, Path],
) -> None:
    """The spread fixture names COGs that were never written.

    Nothing here asserts the catalog is clean -- it is not -- only that the
    missing rasters cost no error the mirror tests would then have to sift
    around: the byte-reading checks skip an asset whose bytes are absent.
    """
    ordered, _shuffled = spread_catalogs
    assert [f.rule_id for f in _data_findings(ordered) if f.rule_id.startswith("PTL-DAT")] == []
