"""The collection preview, and the render declaration it has to agree with.

The preview and the render extension are one decision expressed twice: a
client stretching the same colormap over the same range must see what the
thumbnail shows. These tests pin both to the spec, and check the mosaic where
its arithmetic can go wrong -- which cell a footprint lands in, and whether an
absence stays an absence rather than becoming a colour.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.image import imread

from landsat_lst.catalog import build_catalog
from landsat_lst.catalog.spec import DEFAULT_SPEC, RENDER_EXTENSION_URI
from landsat_lst.catalog.thumbnail import HEIGHT, WIDTH, generate_thumbnail
from landsat_lst.catalog.validation import validate_catalog
from tests.cog_fixtures import tile_bounds, write_flat_lst_cog, write_source_tree

if TYPE_CHECKING:
    from pathlib import Path

TILES = ("N40W075", "N35W120")

# Four pixels to the degree over lon -180..180 and lat -60..60, so the centre
# of a five-degree tile is arithmetic anyone can check by hand.
#   N40W075 spans -75..-70 E and 35..40 N: cols 420..440, rows 80..100.
#   S30W060 spans -60..-55 E and -35..-30 N: cols 480..500, rows 360..380.
_HOT_CENTRE = (90, 430)
_EMPTY_CENTRE = (370, 490)
#: A cell no tile covers: mid-Pacific, well away from either fixture.
_UNCOVERED = (240, 60)

_HOT_CELSIUS = 30.0


def _rgba(path: Path) -> np.ndarray:
    """The PNG as uint8 RGBA, the way it was written."""
    return np.round(imread(path) * 255).astype(np.uint8)


@pytest.fixture(scope="module")
def preview(tmp_path_factory: pytest.TempPathFactory) -> np.ndarray:
    """A mosaic of one flat 30 C tile and one tile that is nothing but fill."""
    root = tmp_path_factory.mktemp("preview")
    hot = write_flat_lst_cog(root / "hot.tif", tile_bounds("N40W075"), _HOT_CELSIUS)
    empty = write_flat_lst_cog(root / "empty.tif", tile_bounds("S30W060"), None)
    return _rgba(generate_thumbnail([hot, empty], root / "thumbnail.png"))


def test_the_preview_is_the_whole_publishable_band(preview: np.ndarray) -> None:
    assert preview.shape == (HEIGHT, WIDTH, 4)
    assert (HEIGHT, WIDTH) == (480, 1440)


def test_a_known_temperature_gets_the_colour_the_spec_names(preview: np.ndarray) -> None:
    """The pixel over the flat tile is the spec's colormap at 30 of 0-55 C."""
    low, high = DEFAULT_SPEC.rescale
    expected = colormaps[DEFAULT_SPEC.colormap]((_HOT_CELSIUS - low) / (high - low), bytes=True)
    assert tuple(preview[_HOT_CENTRE]) == expected
    assert preview[_HOT_CENTRE][3] == 255


def test_fill_and_uncovered_ground_are_both_transparent(preview: np.ndarray) -> None:
    """DN 0 is an absence of data, and so is a tile that was never published."""
    assert preview[_EMPTY_CENTRE][3] == 0
    assert preview[_UNCOVERED][3] == 0


def test_a_sparse_catalog_leaves_most_of_the_world_transparent(preview: np.ndarray) -> None:
    opaque = preview[..., 3] == 255
    assert opaque.sum() == 20 * 20  # one five-degree tile, and only one
    assert opaque[80:100, 420:440].all()


@pytest.fixture(scope="module")
def generated_catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A catalog built the ordinary way: no thumbnail handed in."""
    root = tmp_path_factory.mktemp("generated")
    source = write_source_tree(root / "cogs", TILES)
    return build_catalog(source, root / "catalog")


def _collection(root: Path) -> dict:
    path = root / DEFAULT_SPEC.collection_id / "collection.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_generates_and_registers_a_thumbnail(generated_catalog: Path) -> None:
    png = generated_catalog / DEFAULT_SPEC.collection_id / "thumbnail.png"
    assert png.is_file()
    asset = _collection(generated_catalog)["assets"]["thumbnail"]
    assert asset["type"] == "image/png"
    assert asset["roles"] == ["thumbnail"]
    assert asset["file:size"] == png.stat().st_size
    assert _rgba(png).shape == (HEIGHT, WIDTH, 4)


def test_no_collection_asset_claims_the_visual_role(generated_catalog: Path) -> None:
    """A 'visual' asset would promise a MapLibre style this collection has not got."""
    for asset in _collection(generated_catalog)["assets"].values():
        assert "visual" not in asset.get("roles", [])


def test_the_generated_catalog_raises_no_visualization_finding(generated_catalog: Path) -> None:
    report = validate_catalog(generated_catalog)
    viz = [f for f in report.findings if f.rule_id.startswith("PTL-VIZ")]
    assert viz == [], [f"{f.rule_id} {f.path}: {f.message}" for f in viz]
    assert report.errors == [], [f"{f.rule_id} {f.path}: {f.message}" for f in report.errors]


def _item(root: Path, tile: str) -> dict:
    path = root / DEFAULT_SPEC.collection_id / tile / f"{tile}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_item_declares_the_render_extension_it_uses(generated_catalog: Path) -> None:
    item = _item(generated_catalog, TILES[0])
    assert RENDER_EXTENSION_URI in item["stac_extensions"]
    assert RENDER_EXTENSION_URI.endswith("/render/v2.0.0/schema.json")


def test_the_render_names_the_composite_and_the_spec_s_stretch(generated_catalog: Path) -> None:
    """Render v2 keeps the fields in properties.renders, naming the assets drawn."""
    renders = _item(generated_catalog, TILES[0])["properties"]["renders"]
    assert list(renders) == ["lst"]
    render = renders["lst"]
    assert render["assets"] == ["lst_p95"]
    assert render["colormap_name"] == DEFAULT_SPEC.colormap
    assert render["rescale"] == [list(DEFAULT_SPEC.rescale)]
    assert render["title"]


def test_the_collection_carries_the_same_render(generated_catalog: Path) -> None:
    """A browser that never opens an item still learns how to draw the tiles."""
    collection = _collection(generated_catalog)
    assert RENDER_EXTENSION_URI in collection["stac_extensions"]
    assert collection["renders"] == _item(generated_catalog, TILES[0])["properties"]["renders"]


def test_no_asset_carries_prefixed_render_fields(generated_catalog: Path) -> None:
    """Render v2.0.0 defines no ``render:*`` asset fields, so none are written."""
    for asset in _item(generated_catalog, TILES[0])["assets"].values():
        assert not [key for key in asset if key.startswith("render:")]
