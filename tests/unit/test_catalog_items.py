"""Shape of the JSON the catalog builder writes.

The conformance gate checks that a validator accepts the catalog. These tests
check the decisions a validator has no opinion about: which keys carry the
encoding contract, which roles each asset takes, and the order of the providers
list, where order is meaning rather than style.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from landsat_lst.catalog import build_catalog
from landsat_lst.catalog.items import GridMismatchError, build_item
from landsat_lst.catalog.scan import scan_source
from landsat_lst.catalog.spec import (
    COG_MEDIA_TYPE,
    DEFAULT_SPEC,
    PORTOLAN_SCHEMA_URI,
    spec_for_window,
)
from tests.cog_fixtures import write_source_tree, write_thumbnail

if TYPE_CHECKING:
    from pathlib import Path

TILE = "N40W075"


@pytest.fixture(scope="module")
def catalog_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("items")
    source = write_source_tree(root / "cogs", (TILE, "N35W120"))
    return build_catalog(
        source, root / "catalog", thumbnail=write_thumbnail(root / "thumbnail.png")
    )


def _read(root: Path, *parts: str) -> dict:
    return json.loads(root.joinpath(*parts).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def item(catalog_root: Path) -> dict:
    return _read(catalog_root, DEFAULT_SPEC.collection_id, TILE, f"{TILE}.json")


@pytest.fixture(scope="module")
def collection(catalog_root: Path) -> dict:
    return _read(catalog_root, DEFAULT_SPEC.collection_id, "collection.json")


@pytest.fixture(scope="module")
def root_catalog(catalog_root: Path) -> dict:
    return _read(catalog_root, "catalog.json")


def test_item_declares_an_interval_and_no_instant(item: dict) -> None:
    assert item["properties"]["datetime"] is None
    assert item["properties"]["start_datetime"] == "2021-01-01T00:00:00Z"
    assert item["properties"]["end_datetime"] == "2025-12-31T23:59:59Z"


def test_item_carries_the_encoding_contract(item: dict) -> None:
    assert item["properties"]["lst:scale"] == 0.01
    assert item["properties"]["lst:offset"] == -50.0
    assert item["properties"]["lst:units"] == "celsius"
    assert item["properties"]["lst:nodata"] == 0


def test_item_geometry_and_bbox_match_the_tile(item: dict) -> None:
    assert item["bbox"] == [-75.0, 35.0, -70.0, 40.0]
    ring = item["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    assert {tuple(point) for point in ring} == {
        (-75.0, 35.0),
        (-70.0, 35.0),
        (-70.0, 40.0),
        (-75.0, 40.0),
    }


def test_both_assets_are_cogs_with_the_expected_roles(item: dict) -> None:
    assets = item["assets"]
    assert assets["lst_p95"]["roles"] == ["data"]
    assert assets["qa_count"]["roles"] == ["data", "quality"]
    for asset in assets.values():
        assert asset["type"] == COG_MEDIA_TYPE
        assert asset["href"].startswith("./")
        assert asset["file:size"] > 0
        assert "file:checksum" not in asset


def test_lst_statistics_are_decoded_to_celsius(item: dict) -> None:
    stats = item["assets"]["lst_p95"]["bands"][0]["statistics"]
    assert 15.0 < stats["minimum"] < stats["maximum"] < 50.0
    assert item["assets"]["lst_p95"]["bands"][0]["nodata"] == 0


def test_qa_carries_twelve_monthly_bands(item: dict) -> None:
    bands = item["assets"]["qa_count"]["bands"]
    assert [band["description"] for band in bands][:2] == ["January", "February"]
    assert len(bands) == 12
    assert bands[-1]["description"] == "December"
    assert all("nodata" not in band for band in bands)


def test_only_catalog_and_collection_declare_the_portolan_schema(
    root_catalog: dict, collection: dict, item: dict
) -> None:
    assert root_catalog["stac_extensions"] == [PORTOLAN_SCHEMA_URI]
    assert PORTOLAN_SCHEMA_URI in collection["stac_extensions"]
    assert PORTOLAN_SCHEMA_URI not in item["stac_extensions"]


def test_provider_order_puts_the_single_host_last(collection: dict) -> None:
    providers = collection["providers"]
    assert providers[0]["roles"] == ["licensor", "processor"]
    assert providers[-1]["roles"] == ["producer", "host"]
    assert sum("host" in p["roles"] for p in providers) == 1
    assert providers[-1].get("url") or providers[-1].get("email")


def test_collection_declares_both_item_assets(collection: dict) -> None:
    assert set(collection["item_assets"]) == {"lst_p95", "qa_count"}
    assert collection["license"] == "CC0-1.0"


def test_no_object_carries_a_self_or_upstream_link(
    root_catalog: dict, collection: dict, item: dict
) -> None:
    for obj in (root_catalog, collection, item):
        rels = {link["rel"] for link in obj["links"]}
        assert rels.isdisjoint({"self", "via", "canonical"})


def test_structural_links_are_relative_titled_and_typed(
    root_catalog: dict, collection: dict
) -> None:
    for obj in (root_catalog, collection):
        for link in obj["links"]:
            assert "://" not in link["href"]
            if link["rel"] in ("child", "item"):
                assert link["title"]
                expected = "application/geo+json" if link["rel"] == "item" else "application/json"
                assert link["type"] == expected


def test_markdown_links_point_at_the_sibling_files(root_catalog: dict, collection: dict) -> None:
    for obj in (root_catalog, collection):
        by_rel = {link["rel"]: link for link in obj["links"]}
        assert by_rel["agents"] == {
            "rel": "agents",
            "href": "./AGENTS.md",
            "type": "text/markdown",
        }
        assert by_rel["describedby"]["href"] == "./README.md"


def test_a_raster_off_the_shared_grid_stops_the_build(tmp_path: Path) -> None:
    """A footprint the grid does not predict means the tile was written wrong."""
    source = write_source_tree(tmp_path / "cogs", (TILE,))
    pair = scan_source(source, "2021-2025")[0]
    shifted = pair.lst.__class__(**{**vars(pair.lst), "bbox": (0.0, 0.0, 5.0, 5.0)})
    off_grid = pair.__class__(tile=pair.tile, lst=shifted, qa=pair.qa)
    with pytest.raises(GridMismatchError, match=TILE):
        build_item(off_grid, DEFAULT_SPEC)


def test_a_different_window_renames_the_collection() -> None:
    spec = spec_for_window("2018-2020")
    assert spec.collection_id == "lst-p95-2018-2020"
    assert spec.start_datetime == "2018-01-01T00:00:00Z"
    assert spec.end_datetime == "2020-12-31T23:59:59Z"
