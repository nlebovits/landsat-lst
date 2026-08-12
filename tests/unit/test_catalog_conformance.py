"""The conformance gate: a built catalog must satisfy the Portolan validator.

Hermetic and offline. The catalog is assembled from synthetic 64x64 COGs and
validated in process through ``rashid.validate``; every pass it runs here reads
either a schema bundled in the rashid wheel or bytes inside the temporary
catalog tree, so nothing reaches a network.

The gate is the warning set, not the warning count. Errors must be zero, and
the *set* of warning rule ids must equal the frozen baseline, so a new class of
defect fails the build even though a second occurrence of an accepted one does
not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from landsat_lst.catalog import build_catalog
from landsat_lst.catalog.scan import IncompleteTileError
from landsat_lst.catalog.spec import DEFAULT_SPEC
from landsat_lst.catalog.validation import (
    ACCEPTED_WARNING_RULE_IDS,
    unaccepted_warnings,
    validate_catalog,
)
from tests.cog_fixtures import write_source_tree, write_thumbnail

if TYPE_CHECKING:
    from pathlib import Path

TILES = ("N40W075", "N35W120", "S30W060")

# Measured against rashid 0.1.4 by running the gate and reading back what it
# emitted. Each id is accepted for a stated reason; anything else is a defect.
#
#   PTL-AST-003  assets carry file:size but no file:checksum, which Portolan
#                makes a SHOULD precisely because recomputing digests over a
#                multi-terabyte republish is not free.
#   PTL-DAT-010  qa_count declares no nodata, because 0 ("no valid observation
#                this month") is a real value. Without nodata the embedded
#                valid-percent statistic is only a SHOULD, so its absence warns
#                rather than errors. lst_p95 does declare nodata and does carry
#                the statistic, so it contributes no finding.
#   PTL-MIR-001  no items.parquet mirror: the stac-geoparquet mirror is a
#                separate publishing artifact, not part of catalog assembly.
EXPECTED_WARNING_RULE_IDS = frozenset({"PTL-AST-003", "PTL-DAT-010", "PTL-MIR-001"})


@pytest.fixture(scope="module")
def built_catalog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A complete catalog built from synthetic COGs, shared across the module."""
    root = tmp_path_factory.mktemp("conformance")
    source = write_source_tree(root / "cogs", TILES)
    thumbnail = write_thumbnail(root / "thumbnail.png")
    return build_catalog(source, root / "catalog", thumbnail=thumbnail)


def test_baseline_matches_the_shipped_constant() -> None:
    """The reasons documented here are the ones the validator applies."""
    assert ACCEPTED_WARNING_RULE_IDS == EXPECTED_WARNING_RULE_IDS


def test_catalog_has_no_validation_errors(built_catalog: Path) -> None:
    report = validate_catalog(built_catalog)
    assert report.errors == [], [f"{f.rule_id} {f.path}: {f.message}" for f in report.errors]
    assert report.passed


def test_warning_set_equals_the_frozen_baseline(built_catalog: Path) -> None:
    report = validate_catalog(built_catalog)
    assert {f.rule_id for f in report.warnings} == EXPECTED_WARNING_RULE_IDS
    assert unaccepted_warnings(report) == set()


def test_every_tile_becomes_an_item(built_catalog: Path) -> None:
    collection_dir = built_catalog / DEFAULT_SPEC.collection_id
    for tile in TILES:
        assert (collection_dir / tile / f"{tile}.json").is_file()
        assert (collection_dir / tile / f"lst_p95_2021-2025_{tile}.tif").is_file()
        assert (collection_dir / tile / f"qa_count_2021-2025_{tile}.tif").is_file()


def test_required_markdown_sits_beside_each_object(built_catalog: Path) -> None:
    for directory in (built_catalog, built_catalog / DEFAULT_SPEC.collection_id):
        for name in ("AGENTS.md", "README.md"):
            text = (directory / name).read_text(encoding="utf-8")
            assert text.startswith("#")


def test_collection_readme_states_license_and_provenance(built_catalog: Path) -> None:
    readme = (built_catalog / DEFAULT_SPEC.collection_id / "README.md").read_text()
    assert "CC0-1.0" in readme
    assert "Landsat Collection 2 Level-2" in readme
    assert "Geological Survey" in readme
    assert "Reading the data" in readme
    assert "celsius = dn * 0.01 - 50" in readme


def test_half_a_tile_refuses_to_build(tmp_path: Path) -> None:
    """A tile with one asset is a processing failure, not a catalog shape."""
    source = write_source_tree(tmp_path / "cogs", ("N40W075",))
    (source / "qa_count_2021-2025_N40W075.tif").unlink()
    with pytest.raises(IncompleteTileError, match="N40W075"):
        build_catalog(source, tmp_path / "catalog")
