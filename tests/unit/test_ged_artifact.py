"""The GED artifact's completeness contract.

The artifact's failure mode is silence: a granule the build never consumed
contributes no gap cells, which is byte-identical to a granule that genuinely
has none. An artifact built from a partial archive would therefore mask
nothing over the gaps it never saw, and every tile would ship looking
successful. These tests pin the machinery that makes that impossible -- the
consumed manifest, the refusal, and a content hash that reproduces.
"""

from __future__ import annotations

import numpy as np
import pytest

from landsat_lst import ged
from landsat_lst.config import settings
from landsat_lst.ged import MissingGranuleError, build_artifact, gap_mask_for_geobox
from landsat_lst.tiling import geobox_for_bbox

h5py = pytest.importorskip("h5py")

#: 0.1 degree inside granule (40, -75) -- see tests/unit/test_ged_mask.py.
BBOX = (-75.0, 39.9, -74.9, 40.0)

#: A box in a granule the fixtures deliberately never write.
BBOX_UNCOVERED = (-70.0, 39.9, -69.9, 40.0)

CORE = (40, -75)
MARGIN = [(40, -76), (41, -75), (41, -76)]


def write_granule(ged_dir, lat_top: int, lon_west: int, numobs: np.ndarray | None = None) -> None:
    """Write one synthetic AG100 granule, north-up and west-left."""
    if numobs is None:
        numobs = np.full((100, 100), 7, dtype=np.int16)
    lat = np.linspace(lat_top, lat_top - 1, 100)[:, None] * np.ones((1, 100))
    lon = np.ones((100, 1)) * np.linspace(lon_west, lon_west + 1, 100)[None, :]
    ged_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(ged_dir / ged.granule_name(lat_top, lon_west), "w") as f:
        f["Geolocation/Latitude"] = lat
        f["Geolocation/Longitude"] = lon
        f["Observations/NumObs"] = numobs.astype(np.int16)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A four-granule archive with gaps, wired into settings."""
    ged_dir = tmp_path / "aster_ged"
    numobs = np.full((100, 100), 7, dtype=np.int16)
    numobs[0:3, 0:3] = 0
    numobs[5, 7] = 0
    write_granule(ged_dir, *CORE, numobs)
    for lat_top, lon_west in MARGIN:
        write_granule(ged_dir, lat_top, lon_west)
    monkeypatch.setattr(settings, "ged_dir", ged_dir)
    monkeypatch.setattr(settings, "ged_artifact", tmp_path / "absent.npz")
    # The wheel now carries the production artifact, which outranks a
    # granule directory by design; these fixtures test the granule path.
    monkeypatch.setattr(ged, "packaged_artifact_path", lambda: None)
    # An artifact these tests build is not the pinned production one.
    monkeypatch.setattr(ged, "GED_ARTIFACT_CONTENT_SHA256", None)
    return ged_dir, tmp_path


class TestDeterminism:
    def test_two_builds_of_one_archive_reproduce_the_content_hash(self, archive):
        ged_dir, tmp_path = archive
        first = build_artifact(ged_dir, tmp_path / "a.npz")
        second = build_artifact(ged_dir, tmp_path / "b.npz")
        assert first["content_sha256"] == second["content_sha256"]
        assert first["build_code_sha256"] == second["build_code_sha256"]

    def test_the_zip_bytes_are_not_the_content_hash(self, archive):
        """npz embeds timestamps, which is why the hash is over the arrays."""
        ged_dir, tmp_path = archive
        build_artifact(ged_dir, tmp_path / "a.npz")
        build_artifact(ged_dir, tmp_path / "b.npz")
        # The files may or may not differ byte for byte depending on the
        # clock; what must hold is that content equality does not depend on it.
        with np.load(tmp_path / "a.npz") as a, np.load(tmp_path / "b.npz") as b:
            assert str(a["content_sha256"]) == str(b["content_sha256"])

    def test_a_changed_granule_changes_the_content_hash(self, archive):
        ged_dir, tmp_path = archive
        before = build_artifact(ged_dir, tmp_path / "a.npz")["content_sha256"]
        moved = np.full((100, 100), 7, dtype=np.int16)
        moved[50:53, 50:53] = 0
        write_granule(ged_dir, *CORE, moved)
        after = build_artifact(ged_dir, tmp_path / "b.npz")["content_sha256"]
        assert before != after

    def test_the_build_records_its_inputs(self, archive):
        ged_dir, tmp_path = archive
        expected = [ged.granule_name(*CORE), "AG1km.v003.10.010.0010.h5"]
        report = build_artifact(ged_dir, tmp_path / "a.npz", expected=expected)
        assert report["granules"] == 4
        assert report["missing_expected"] == 1
        assert report["complete"] is False
        with np.load(tmp_path / "a.npz") as data:
            assert str(data["product"]) == ged.GED_PRODUCT
            assert len(data["consumed"]) == len(data["consumed_sha256"]) == 4
            assert list(data["missing_expected"]) == ["AG1km.v003.10.010.0010.h5"]
            assert all(len(str(d)) == 64 for d in data["consumed_sha256"])
            assert list(data["absent_upstream"]) == []
            assert int(data["format_version"]) == 3

    def test_a_complete_archive_reports_complete(self, archive):
        ged_dir, tmp_path = archive
        names = sorted(p.name for p in ged_dir.glob("*.h5"))
        report = build_artifact(ged_dir, tmp_path / "a.npz", expected=names)
        assert report["complete"] is True
        assert report["missing_expected"] == 0


INVENTORY = {
    "short_name": "AG1km",
    "version": "003",
    "queried_at": "2026-09-04T07:50:00+00:00",
    "granule_count": 24873,
}

#: The core granule BBOX_UNCOVERED needs and the fixtures never write.
UNCOVERED = "AG1km.v003.40.-070.0010.h5"


class TestUpstreamAbsence:
    """A granule the collection lacks is served with a warning; a granule
    the collection has but the build never consumed is still refused."""

    def test_an_absent_upstream_core_granule_is_served_not_refused(self, archive):
        ged_dir, tmp_path = archive
        artifact = tmp_path / "a.npz"
        report = build_artifact(
            ged_dir,
            artifact,
            expected=[UNCOVERED],
            absent_upstream=[UNCOVERED],
            inventory=INVENTORY,
        )
        assert report["complete"] is True
        assert report["missing_expected"] == 0
        assert report["absent_upstream"] == 1
        settings.ged_artifact = artifact
        mask = gap_mask_for_geobox(geobox_for_bbox(BBOX_UNCOVERED), buffer_cells=1)
        assert mask.shape == geobox_for_bbox(BBOX_UNCOVERED).shape
        assert not mask.any()

    def test_the_absence_is_logged_never_silent(self, archive):
        from structlog.testing import capture_logs

        ged_dir, tmp_path = archive
        artifact = tmp_path / "a.npz"
        build_artifact(ged_dir, artifact, absent_upstream=[UNCOVERED], inventory=INVENTORY)
        settings.ged_artifact = artifact
        with capture_logs() as logs:
            gap_mask_for_geobox(geobox_for_bbox(BBOX_UNCOVERED), buffer_cells=1)
        events = [e for e in logs if e["event"] == "ged_upstream_granules_absent"]
        assert len(events) == 1
        assert events[0]["granules"] == [UNCOVERED]
        assert events[0]["log_level"] == "warning"

    def test_a_fetchable_unconsumed_core_granule_is_still_refused(self, archive):
        """Recording *some* upstream absences must not loosen the rule for
        the rest: the artifact only knows what it was told."""
        ged_dir, tmp_path = archive
        artifact = tmp_path / "a.npz"
        build_artifact(
            ged_dir, artifact, absent_upstream=["AG1km.v003.10.010.0010.h5"], inventory=INVENTORY
        )
        settings.ged_artifact = artifact
        with pytest.raises(MissingGranuleError, match=UNCOVERED):
            gap_mask_for_geobox(geobox_for_bbox(BBOX_UNCOVERED), buffer_cells=1)

    def test_the_absent_list_is_part_of_the_content_hash(self, archive):
        ged_dir, tmp_path = archive
        plain = build_artifact(ged_dir, tmp_path / "a.npz")["content_sha256"]
        marked = build_artifact(ged_dir, tmp_path / "b.npz", absent_upstream=[UNCOVERED])
        assert marked["content_sha256"] != plain

    def test_the_inventory_identity_is_part_of_the_content_hash(self, archive):
        ged_dir, tmp_path = archive
        one = build_artifact(ged_dir, tmp_path / "a.npz", inventory=INVENTORY)["content_sha256"]
        other = build_artifact(
            ged_dir, tmp_path / "b.npz", inventory={**INVENTORY, "queried_at": "2027-01-01"}
        )["content_sha256"]
        assert one != other

    def test_the_build_records_the_inventory_it_used(self, archive):
        ged_dir, tmp_path = archive
        build_artifact(ged_dir, tmp_path / "a.npz", inventory=INVENTORY)
        with np.load(tmp_path / "a.npz") as data:
            assert str(data["inventory_short_name"]) == "AG1km"
            assert str(data["inventory_version"]) == "003"
            assert str(data["inventory_queried_at"]) == INVENTORY["queried_at"]
            assert int(data["inventory_granule_count"]) == 24873

    def test_a_v2_artifact_is_refused_not_reinterpreted(self, archive):
        """v2 has no absent-upstream list, so reading it as v3 would either
        refuse every ocean tile or, worse, be patched to serve them blind."""
        _, tmp_path = archive
        artifact = tmp_path / "old.npz"
        np.savez_compressed(
            artifact,
            format_version=np.int32(2),
            gap_rows=np.zeros(0, dtype=np.int32),
            gap_cols=np.zeros(0, dtype=np.int32),
        )
        settings.ged_artifact = artifact
        with pytest.raises(ValueError, match="format v2"):
            gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)


class TestCoverageRefusal:
    def test_a_geobox_inside_the_manifest_is_served(self, archive):
        ged_dir, tmp_path = archive
        artifact = tmp_path / "a.npz"
        build_artifact(ged_dir, artifact)
        settings.ged_artifact = artifact
        mask = gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)
        assert mask.any()

    def test_a_geobox_outside_the_manifest_raises_naming_the_granule(self, archive):
        """The whole point: no silent 'no gap cells here'."""
        ged_dir, tmp_path = archive
        artifact = tmp_path / "a.npz"
        build_artifact(ged_dir, artifact)
        settings.ged_artifact = artifact
        with pytest.raises(MissingGranuleError) as excinfo:
            gap_mask_for_geobox(geobox_for_bbox(BBOX_UNCOVERED), buffer_cells=1)
        assert "AG1km.v003.40.-070.0010.h5" in str(excinfo.value)
        assert excinfo.value.source_kind == "artifact"

    def test_the_refusal_explains_why_silence_would_be_worse(self, archive):
        ged_dir, tmp_path = archive
        artifact = tmp_path / "a.npz"
        build_artifact(ged_dir, artifact)
        settings.ged_artifact = artifact
        with pytest.raises(MissingGranuleError, match="looking successful"):
            gap_mask_for_geobox(geobox_for_bbox(BBOX_UNCOVERED), buffer_cells=1)

    def test_a_margin_only_absence_is_tolerated_like_the_granule_path(self, archive, tmp_path):
        """Both paths must agree, or a tile's mask depends on its source."""
        ged_dir, _ = archive
        partial = tmp_path / "partial"
        partial.mkdir()
        (partial / ged.granule_name(*CORE)).write_bytes(
            (ged_dir / ged.granule_name(*CORE)).read_bytes()
        )
        artifact = tmp_path / "core_only.npz"
        build_artifact(partial, artifact)
        settings.ged_artifact = artifact
        mask = gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)
        assert mask.any()


class TestIntegrity:
    def test_a_tampered_artifact_is_refused(self, archive):
        ged_dir, tmp_path = archive
        artifact = tmp_path / "a.npz"
        build_artifact(ged_dir, artifact)
        with np.load(artifact) as data:
            fields = dict(data)
        fields["gap_rows"] = fields["gap_rows"][:-1]
        fields["gap_cols"] = fields["gap_cols"][:-1]
        np.savez_compressed(artifact, **fields)
        settings.ged_artifact = artifact
        with pytest.raises(ValueError, match="self-inconsistent"):
            gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)

    def test_a_v1_artifact_is_refused_not_reinterpreted(self, archive):
        _, tmp_path = archive
        artifact = tmp_path / "old.npz"
        np.savez_compressed(
            artifact,
            format_version=np.int32(1),
            gap_rows=np.zeros(0, dtype=np.int32),
            gap_cols=np.zeros(0, dtype=np.int32),
        )
        settings.ged_artifact = artifact
        with pytest.raises(ValueError, match="format v1"):
            gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)

    def test_a_pinned_digest_mismatch_is_a_hard_error(self, archive, monkeypatch):
        ged_dir, tmp_path = archive
        artifact = tmp_path / "a.npz"
        build_artifact(ged_dir, artifact)
        monkeypatch.setattr(ged, "GED_ARTIFACT_CONTENT_SHA256", "0" * 64)
        settings.ged_artifact = artifact
        with pytest.raises(ValueError, match=r"pins 0{64}"):
            gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)

    def test_the_matching_pin_passes(self, archive, monkeypatch):
        ged_dir, tmp_path = archive
        artifact = tmp_path / "a.npz"
        report = build_artifact(ged_dir, artifact)
        monkeypatch.setattr(ged, "GED_ARTIFACT_CONTENT_SHA256", report["content_sha256"])
        settings.ged_artifact = artifact
        assert gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1).any()


class TestResolution:
    def test_nothing_available_names_all_three_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "ged_artifact", tmp_path / "absent.npz")
        monkeypatch.setattr(settings, "ged_dir", tmp_path / "absent_dir")
        monkeypatch.setattr(ged, "packaged_artifact_path", lambda: None)
        with pytest.raises(FileNotFoundError) as excinfo:
            gap_mask_for_geobox(geobox_for_bbox(BBOX))
        message = str(excinfo.value)
        assert "absent.npz" in message
        assert "landsat_lst/data/ged_gap_mask.npz" in message
        assert "absent_dir" in message
        assert "LST_GED_GAP_MASK=false" in message

    def test_the_packaged_artifact_outranks_the_granules(self, archive, tmp_path, monkeypatch):
        """A VM has the wheel and no archive; a laptop may have both."""
        ged_dir, _ = archive
        artifact = tmp_path / "packaged.npz"
        build_artifact(ged_dir, artifact)
        monkeypatch.setattr(settings, "ged_artifact", tmp_path / "absent.npz")
        monkeypatch.setattr(ged, "packaged_artifact_path", lambda: artifact)
        kind, source = ged._resolve_source()
        assert kind == "artifact"
        assert source == artifact

    def test_an_explicit_override_outranks_the_packaged_artifact(
        self, archive, tmp_path, monkeypatch
    ):
        ged_dir, _ = archive
        override = tmp_path / "override.npz"
        build_artifact(ged_dir, override)
        monkeypatch.setattr(settings, "ged_artifact", override)
        monkeypatch.setattr(ged, "packaged_artifact_path", lambda: tmp_path / "other.npz")
        kind, source = ged._resolve_source()
        assert (kind, source) == ("artifact", override)

    def test_the_packaged_artifact_exists_and_is_pinned(self):
        """The production mask ships inside the package (#118).

        ``ged_gap_mask`` defaults on and a VM has no archive, so the packaged
        artifact is the only source a fleet tile can find. Its content hash
        is pinned in the same commit that adds it, and the pin must match:
        a rebuilt artifact that was not deliberately re-pinned is a
        different product.
        """
        packaged = ged.packaged_artifact_path()
        assert packaged is not None, "no artifact packaged; see scripts/build_ged_gap_mask.py"
        assert ged.GED_ARTIFACT_CONTENT_SHA256 is not None
        with np.load(packaged) as data:
            assert int(data["format_version"]) == ged.ARTIFACT_FORMAT_VERSION
            assert str(data["content_sha256"]) == ged.GED_ARTIFACT_CONTENT_SHA256
            assert len(data["missing_expected"]) == 0, "a partial artifact is packaged"
            assert len(data["consumed"]) > 0
            assert int(data["inventory_granule_count"]) > 0

    def test_the_packaged_artifact_serves_every_production_tile(self):
        """Every core granule of every land tile is consumed or absent
        upstream, so no production tile can hit MissingGranuleError."""
        from landsat_lst.ged_coverage import build_report

        packaged = ged.packaged_artifact_path()
        assert packaged is not None
        with np.load(packaged) as data:
            consumed = {str(x) for x in data["consumed"]}
            absent = {str(x) for x in data["absent_upstream"]}
        report = build_report(artifact=packaged)
        core_needed = set(report.missing_core) | (set(report.expected) & consumed)
        assert report.complete is True
        assert set(report.missing_core) <= absent
        assert core_needed  # the manifest is not empty
