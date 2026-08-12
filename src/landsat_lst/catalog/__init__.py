"""Build a Portolan-compliant STAC catalog from finished per-tile COGs.

The source is a directory or an ``s3://`` prefix holding the two COGs each tile
produces. The output is a self-contained catalog: one root catalog, one
collection, one item per tile, with the COGs materialised beside the items they
belong to and every structural link relative.

    from landsat_lst.catalog import build_catalog

    build_catalog("./cogs", "./catalog")
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pystac

from landsat_lst.catalog.collection import build_collection, build_root_catalog
from landsat_lst.catalog.docs import (
    collection_agents_md,
    collection_readme_md,
    root_agents_md,
    root_readme_md,
)
from landsat_lst.catalog.items import build_item
from landsat_lst.catalog.scan import place_file, scan_source
from landsat_lst.catalog.spec import (
    DEFAULT_SPEC,
    FILE_EXTENSION_URI,
    CatalogSpec,
)

__all__ = ["CatalogSpec", "build_catalog"]

_THUMBNAIL_NAME = "thumbnail.png"


class EmptySourceError(RuntimeError):
    """No complete tile was found, so there is nothing to publish."""


def _write_docs(root: Path, collection_dir: Path, spec: CatalogSpec) -> None:
    """Write the four markdown files the STAC objects link to."""
    for path, text in (
        (root / "AGENTS.md", root_agents_md(spec)),
        (root / "README.md", root_readme_md(spec)),
        (collection_dir / "AGENTS.md", collection_agents_md(spec)),
        (collection_dir / "README.md", collection_readme_md(spec)),
    ):
        path.write_text(text, encoding="utf-8")


def _thumbnail_asset(collection_dir: Path, provided: Path | None) -> pystac.Asset | None:
    """The collection thumbnail, if one has been rendered for this collection.

    A thumbnail handed in is copied into place; otherwise an existing
    ``thumbnail.png`` beside the collection is picked up, so re-running a build
    over a catalog that already has one keeps it registered.
    """
    dest = collection_dir / _THUMBNAIL_NAME
    if provided is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(provided, dest)
    if not dest.is_file():
        return None
    return pystac.Asset(
        href=f"./{_THUMBNAIL_NAME}",
        title="Collection preview",
        media_type=pystac.MediaType.PNG,
        roles=["thumbnail"],
        extra_fields={"file:size": dest.stat().st_size},
    )


def _attach_thumbnail(collection: pystac.Collection, asset: pystac.Asset | None) -> None:
    """Register the thumbnail and the extension its ``file:size`` needs."""
    if asset is None:
        return
    collection.add_asset("thumbnail", asset)
    if FILE_EXTENSION_URI not in collection.stac_extensions:
        collection.stac_extensions.append(FILE_EXTENSION_URI)


def _place_assets(collection_dir: Path, pairs: list, items: list[pystac.Item]) -> None:
    """Copy or link every tile's COGs into the directory of its item."""
    for pair, item in zip(pairs, items, strict=True):
        tile_dir = collection_dir / item.id
        place_file(pair.lst.file, tile_dir / pair.lst.file.name)
        place_file(pair.qa.file, tile_dir / pair.qa.file.name)


def build_catalog(
    source: str | Path,
    out: str | Path,
    spec: CatalogSpec = DEFAULT_SPEC,
    tiles: tuple[str, ...] | None = None,
    thumbnail: str | Path | None = None,
) -> Path:
    """Build the catalog for one observation window.

    Args:
        source: Directory or ``s3://bucket/prefix`` holding the finished COGs.
        out: Directory to write the catalog into; created if absent.
        spec: Naming and policy decisions. Defaults to the production dataset.
        tiles: Restrict the catalog to these tile names.
        thumbnail: PNG to register as the collection thumbnail. When omitted,
            an existing ``thumbnail.png`` beside the collection is used.

    Returns:
        The catalog root directory.

    Raises:
        EmptySourceError: The source holds no complete tile.
        IncompleteTileError: A tile carries exactly one of its two assets.
    """
    pairs = scan_source(source, spec.window, tiles)
    if not pairs:
        msg = f"no complete {spec.window} tile found under {source}"
        raise EmptySourceError(msg)

    items = [build_item(pair, spec) for pair in pairs]
    collection = build_collection(items, spec)
    catalog = build_root_catalog(collection, spec)
    root = Path(out)
    catalog.normalize_hrefs(str(root))

    collection_dir = root / spec.collection_id
    _place_assets(collection_dir, pairs, items)
    _attach_thumbnail(collection, _thumbnail_asset(collection_dir, _as_path(thumbnail)))

    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    _write_docs(root, collection_dir, spec)
    return root


def _as_path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value)
