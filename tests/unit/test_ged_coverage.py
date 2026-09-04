"""What production needs from ASTER GED, and whether a source has it.

The expected manifest is arithmetic, so most of this is exact: a five-degree
tile spans 5 degrees of granules, the buffer adds a ring, and the count
follows. The one test that is not arithmetic reads a real AG1km granule and
checks the cell convention against the granule's own Geolocation arrays --
every other orientation test in this suite is code agreeing with code.
"""

from __future__ import annotations

import numpy as np
import pytest

from landsat_lst import ged, ged_coverage
from landsat_lst.config import settings
from landsat_lst.ged_coverage import (
    ABSENT_OUTSIDE_FETCH_DOMAIN,
    ABSENT_UNVERIFIED,
    ABSENT_UPSTREAM,
    ABSENT_WITHIN_FETCH_DOMAIN,
    UpstreamInventory,
    build_report,
    granules_for_tile,
    load_upstream_inventory,
    production_tiles,
    write_upstream_inventory,
)

h5py = pytest.importorskip("h5py")


def _real_granules() -> list:
    """AG1km granules on this machine, if any. CI has none."""
    directory = settings.ged_dir
    return sorted(directory.glob("AG1km.v003.*.0010.h5")) if directory.is_dir() else []


class TestExpectedManifest:
    def test_a_five_degree_tile_spans_five_granules_plus_a_ring(self):
        """S30W065 is lat [-35,-30], lon [-65,-60): 5x5 granules, 7x7 buffered."""
        touched, core = granules_for_tile("S30W065", buffer_cells=0)
        assert len(core) == len(touched) == 25
        buffered, core_b = granules_for_tile("S30W065", buffer_cells=1)
        assert len(buffered) == 49
        assert core_b == core
        assert core <= buffered

    def test_the_buffer_ring_is_exactly_the_difference(self):
        bare, _ = granules_for_tile("S30W065", buffer_cells=0)
        buffered, _ = granules_for_tile("S30W065", buffer_cells=1)
        assert len(buffered - bare) == 24

    def test_the_named_granules_are_the_tiles_own_degrees(self):
        _, core = granules_for_tile("S30W065", buffer_cells=0)
        assert ged.granule_name(-30, -65) in core
        assert ged.granule_name(-34, -61) in core
        # One degree outside the tile on either side.
        assert ged.granule_name(-29, -65) not in core
        assert ged.granule_name(-35, -65) not in core

    def test_the_buffered_set_reaches_one_degree_out(self):
        buffered, _ = granules_for_tile("S30W065", buffer_cells=1)
        assert ged.granule_name(-29, -66) in buffered
        assert ged.granule_name(-35, -60) in buffered

    def test_production_is_seven_hundred_land_tiles(self):
        tiles = production_tiles()
        assert len(tiles) == 700
        assert len(set(tiles)) == 700
        assert "S30W065" in tiles

    def test_the_grammar_is_the_readers_grammar(self):
        """granules_for_tile and numobs_window must name the same files."""
        from landsat_lst.tiling import geobox_for_bbox, parse_tile_name

        geobox = geobox_for_bbox(parse_tile_name("S30W065").bbox)
        _, _, core, padded = ged.cell_window_for_geobox(geobox, 1)
        row0, row1, col0, col1 = padded
        named = {
            n
            for n, _, _, _ in ged.granules_for_window(
                row0=row0, row1=row1, col0=col0, col1=col1, core=core
            )
        }
        touched, _ = granules_for_tile("S30W065", buffer_cells=1)
        assert named == touched


class TestReport:
    @pytest.fixture
    def archive(self, tmp_path):
        directory = tmp_path / "aster_ged"
        directory.mkdir()
        touched, _ = granules_for_tile("S30W065", buffer_cells=1)
        for name in sorted(touched)[:10]:
            (directory / name).write_bytes(b"")
        return directory

    def test_a_partial_archive_is_incomplete_and_says_by_how_much(self, archive):
        report = build_report(ged_dir=archive, tiles=["S30W065"], buffer_cells=1)
        counts = report.counts()
        assert report.complete is False
        assert counts["expected"] == 49
        assert counts["consumed_of_expected"] == 10
        assert counts["missing"] == 39
        assert counts["tiles_missing_core"] == 1

    def test_a_complete_archive_is_complete(self, tmp_path):
        directory = tmp_path / "full"
        directory.mkdir()
        touched, _ = granules_for_tile("S30W065", buffer_cells=1)
        for name in touched:
            (directory / name).write_bytes(b"")
        report = build_report(ged_dir=directory, tiles=["S30W065"], buffer_cells=1)
        assert report.complete is True
        assert report.counts()["missing"] == 0

    def test_a_margin_only_absence_does_not_count_as_core(self, tmp_path):
        """A missing margin granule forgoes a buffer; a missing core one
        leaves unmasked pixels inside the tile. The report must separate them."""
        directory = tmp_path / "core_only"
        directory.mkdir()
        _, core = granules_for_tile("S30W065", buffer_cells=1)
        for name in core:
            (directory / name).write_bytes(b"")
        report = build_report(ged_dir=directory, tiles=["S30W065"], buffer_cells=1)
        assert report.counts()["missing"] == 24
        assert report.counts()["missing_core"] == 0
        assert report.tiles_missing_core == ()
        assert report.tiles_missing_any == ("S30W065",)

    def test_counts_conserve(self, archive):
        report = build_report(ged_dir=archive, tiles=["S30W065"], buffer_cells=1)
        counts = report.counts()
        assert counts["consumed_of_expected"] + counts["missing"] == counts["expected"]
        assert len(report.classification) == counts["missing"]

    def test_an_artifact_manifest_can_be_measured_instead_of_a_directory(self, tmp_path):
        touched, _ = granules_for_tile("S30W065", buffer_cells=1)
        artifact = tmp_path / "a.npz"
        np.savez_compressed(artifact, consumed=np.array(sorted(touched), dtype=np.str_))
        report = build_report(artifact=artifact, tiles=["S30W065"], buffer_cells=1)
        assert report.complete is True


class TestClassification:
    def test_without_a_fetch_domain_every_absence_is_unverified(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        report = build_report(ged_dir=directory, tiles=["S30W065"], buffer_cells=0)
        assert set(report.classification.values()) == {ABSENT_UNVERIFIED}

    def test_a_fetch_domain_separates_never_asked_from_asked_and_absent(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        grid = np.zeros((180, 360), dtype=bool)
        # Request exactly the granule at lat_top -30, lon_west -65.
        grid[90 - (-30), -65 + 180] = True
        domain = tmp_path / "domain.npy"
        np.save(domain, grid)
        report = build_report(
            ged_dir=directory, tiles=["S30W065"], buffer_cells=0, fetch_domain=domain
        )
        labels = report.classification
        assert labels[ged.granule_name(-30, -65)] == ABSENT_WITHIN_FETCH_DOMAIN
        assert labels[ged.granule_name(-31, -65)] == ABSENT_OUTSIDE_FETCH_DOMAIN
        assert sum(v == ABSENT_WITHIN_FETCH_DOMAIN for v in labels.values()) == 1

    def test_a_wrong_shaped_grid_is_refused(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        domain = tmp_path / "bad.npy"
        np.save(domain, np.zeros((10, 10), dtype=bool))
        with pytest.raises(ValueError, match=r"expected \(180, 360\)"):
            build_report(ged_dir=directory, tiles=["S30W065"], fetch_domain=domain)

    def test_without_an_inventory_the_record_never_claims_upstream_absence(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        report = build_report(ged_dir=directory, tiles=["S30W065"], buffer_cells=0)
        assert "none given" in str(report.as_dict()["upstream_inventory"])
        assert ABSENT_UPSTREAM not in report.classification.values()
        assert ged_coverage.COVERAGE_VERSION


def _inventory(names: set[str]) -> UpstreamInventory:
    return UpstreamInventory(
        short_name="AG1km",
        version="003",
        queried_at="2026-09-04T07:50:00+00:00",
        granule_count=len(names),
        names=frozenset(names),
    )


class TestUpstreamInventory:
    """Completeness is judged against what the collection holds, and only
    against a persisted inventory -- never a local listing."""

    def test_the_record_round_trips_through_cells(self, tmp_path):
        names = {ged.granule_name(-30, -65), ged.granule_name(40, -75), ged.granule_name(0, 179)}
        path = tmp_path / "inv.json"
        written = write_upstream_inventory(path, names=names, short_name="AG1km", version="003")
        loaded = load_upstream_inventory(path)
        assert loaded == written
        assert loaded.names == frozenset(names)
        assert loaded.granule_count == 3
        assert loaded.queried_at.endswith("+00:00")

    def test_a_foreign_schema_is_refused(self, tmp_path):
        path = tmp_path / "inv.json"
        path.write_text('{"schema_version": 99, "cells": []}')
        with pytest.raises(ValueError, match="schema v99"):
            load_upstream_inventory(path)

    def test_a_count_that_disagrees_with_the_cells_is_refused(self, tmp_path):
        path = tmp_path / "inv.json"
        path.write_text(
            '{"schema_version": 1, "short_name": "AG1km", "version": "003", '
            '"queried_at": "x", "granule_count": 2, "cells": [[-30, -65]]}'
        )
        with pytest.raises(ValueError, match="says 2 granules but lists 1"):
            load_upstream_inventory(path)

    def test_an_absence_the_collection_shares_is_absent_upstream(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        touched, _ = granules_for_tile("S30W065", buffer_cells=0)
        absent = ged.granule_name(-30, -65)
        inventory = _inventory(touched - {absent})
        report = build_report(
            ged_dir=directory, tiles=["S30W065"], buffer_cells=0, upstream_inventory=inventory
        )
        assert report.classification[absent] == ABSENT_UPSTREAM
        assert report.classification[ged.granule_name(-31, -65)] == ABSENT_UNVERIFIED
        assert report.absent_upstream == (absent,)
        assert report.absent_upstream_core == (absent,)
        assert len(report.fetchable) == 24

    def test_complete_means_every_granule_the_collection_holds(self, tmp_path):
        """An archive holding everything upstream is complete even though an
        expected granule is missing, because nothing can ever fetch it."""
        directory = tmp_path / "held"
        directory.mkdir()
        touched, _ = granules_for_tile("S30W065", buffer_cells=1)
        absent = ged.granule_name(-30, -65)
        for name in touched - {absent}:
            (directory / name).write_bytes(b"")
        inventory = _inventory(touched - {absent})
        with_inv = build_report(
            ged_dir=directory, tiles=["S30W065"], buffer_cells=1, upstream_inventory=inventory
        )
        without = build_report(ged_dir=directory, tiles=["S30W065"], buffer_cells=1)
        assert with_inv.complete is True
        assert without.complete is False
        assert with_inv.counts()["missing"] == without.counts()["missing"] == 1
        assert with_inv.counts()["fetchable"] == 0
        assert with_inv.tiles_missing_core == ()
        assert without.tiles_missing_core == ("S30W065",)

    def test_the_record_carries_the_inventory_identity(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        touched, _ = granules_for_tile("S30W065", buffer_cells=0)
        report = build_report(
            ged_dir=directory,
            tiles=["S30W065"],
            buffer_cells=0,
            upstream_inventory=_inventory(touched),
        )
        record = report.as_dict()
        assert record["upstream_inventory"] == {
            "short_name": "AG1km",
            "version": "003",
            "queried_at": "2026-09-04T07:50:00+00:00",
            "granule_count": 25,
        }
        assert record["counts"]["absent_upstream"] == 0
        assert record["counts"]["fetchable"] == 25

    def test_a_path_is_loaded_like_a_record(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        touched, _ = granules_for_tile("S30W065", buffer_cells=0)
        path = tmp_path / "inv.json"
        write_upstream_inventory(path, names=touched, short_name="AG1km", version="003")
        report = build_report(
            ged_dir=directory, tiles=["S30W065"], buffer_cells=0, upstream_inventory=path
        )
        assert report.inventory is not None
        assert report.inventory.granule_count == 25

    def test_the_fetch_domain_source_is_recorded_relative(self, tmp_path, monkeypatch):
        directory = tmp_path / "empty"
        directory.mkdir()
        domain = tmp_path / "domain.npy"
        np.save(domain, np.zeros((180, 360), dtype=bool))
        monkeypatch.chdir(tmp_path)
        report = build_report(
            ged_dir=directory, tiles=["S30W065"], buffer_cells=0, fetch_domain=domain
        )
        assert report.fetch_domain_source == "domain.npy"


@pytest.mark.skipif(not _real_granules(), reason="no local AG1km archive (set LST_GED_DIR)")
class TestAgainstRealGranules:
    """The cell convention, checked against a granule's own geolocation.

    Everything else in this suite is code agreeing with code. This is the one
    place the naming grammar, the north-up orientation, and the 100-cells-per-
    degree pitch are checked against the product itself.
    """

    def test_the_filename_names_the_north_and_west_edges(self):
        path = _real_granules()[0]
        parts = path.name.split(".")
        lat_top, lon_west = int(parts[2]), int(parts[3])
        with h5py.File(path) as f:
            lat = f["Geolocation/Latitude"][:]
            lon = f["Geolocation/Longitude"][:]
        # Geolocation is a 100-point linspace across the degree, so its
        # endpoints are the outermost cell *centres-as-sampled*, i.e. the
        # granule's own edges to within one 1/99-degree step. Assert the
        # bracket rather than equality.
        step = 1.0 / 99.0
        assert lat.max() == pytest.approx(lat_top, abs=step)
        assert lat.min() == pytest.approx(lat_top - 1, abs=step)
        assert lon.min() == pytest.approx(lon_west, abs=step)
        assert lon.max() == pytest.approx(lon_west + 1, abs=step)

    def test_the_granule_is_one_degree_of_one_hundred_cells(self):
        path = _real_granules()[0]
        with h5py.File(path) as f:
            numobs = f["Observations/NumObs"][:]
        assert numobs.shape == (ged.GRANULE_CELLS, ged.GRANULE_CELLS)
        assert ged.GED_CELLS_PER_DEGREE == 100

    def test_the_reader_returns_it_north_up_and_west_left(self):
        """The earlier form asserted ``lat[0,0] > lat[-1,0] or <flipped>``,
        and every real granule is stored north-up, so the ``or`` short-
        circuited and the flip was never exercised (issue #118 audit). This
        form derives the expected orientation from the geolocation arrays
        and compares the whole array; the synthetic single-axis flips live
        in tests/unit/test_ged_mask.py."""
        path = _real_granules()[0]
        with h5py.File(path) as f:
            lat = f["Geolocation/Latitude"][:]
            lon = f["Geolocation/Longitude"][:]
        raw = _raw_numobs(path)
        expected = raw
        if lat[0, 0] < lat[-1, 0]:
            expected = np.flipud(expected)
        if lon[0, 0] > lon[0, -1]:
            expected = np.fliplr(expected)
        oriented = ged.read_granule_numobs(path)
        assert oriented.shape == (100, 100)
        np.testing.assert_array_equal(oriented, expected)
        # And the product's own convention, so the test is not code agreeing
        # with code: the file is north-up and west-left, so no flip applied.
        assert lat[0, 0] > lat[-1, 0]
        assert lon[0, 0] < lon[0, -1]
        np.testing.assert_array_equal(oriented, raw)

    def test_the_name_round_trips_through_the_grammar(self):
        for path in _real_granules()[:50]:
            parts = path.name.split(".")
            assert ged.granule_name(int(parts[2]), int(parts[3])) == path.name


def _raw_numobs(path):
    with h5py.File(path) as f:
        return f["Observations/NumObs"][:]
