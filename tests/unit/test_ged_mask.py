"""The ASTER GED emissivity-gap mask: construction, sources, and failure modes.

Synthetic AG100 granules throughout -- 100x100 int16 NumObs over one degree,
written with h5py exactly as the archive stores them (Geolocation arrays
included, since the reader orients against them rather than trusting file
order). The geoboxes come from the production ``geobox_for_bbox``, so every
expected footprint is exact index arithmetic on the real global grid: one GED
cell is exactly 36 x 36 native pixels (3600 px/deg over 100 cells/deg).
"""

from __future__ import annotations

import numpy as np
import pytest

from landsat_lst import ged
from landsat_lst.config import settings
from landsat_lst.ged import (
    MissingGranuleError,
    build_artifact,
    dilate_cells,
    gap_mask_for_geobox,
    granule_name,
)
from landsat_lst.tiling import geobox_for_bbox

h5py = pytest.importorskip("h5py")

#: A 0.1-degree box in the NW corner of tile N40W075: 360 x 360 native pixels,
#: 10 x 10 GED cells, entirely inside granule (40, -75).
BBOX = (-75.0, 39.9, -74.9, 40.0)

#: The granules a buffered mask over BBOX touches: the core plus the margin
#: ring across the north and west granule boundaries.
CORE = (40, -75)
MARGIN = [(40, -76), (41, -75), (41, -76)]


def write_granule(
    ged_dir,
    lat_top: int,
    lon_west: int,
    numobs: np.ndarray | None = None,
    *,
    flip: bool = False,
) -> None:
    """Write one synthetic AG100 granule, north-up unless ``flip``."""
    if numobs is None:
        numobs = np.full((100, 100), 7, dtype=np.int16)
    lat = np.linspace(lat_top, lat_top - 1, 100)[:, None] * np.ones((1, 100))
    lon = np.ones((100, 1)) * np.linspace(lon_west, lon_west + 1, 100)[None, :]
    if flip:
        # Flip both axes consistently: the data and its own geolocation.
        numobs, lat, lon = numobs[::-1, ::-1], np.flipud(lat), np.fliplr(lon)
    ged_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(ged_dir / granule_name(lat_top, lon_west), "w") as f:
        f["Geolocation/Latitude"] = lat
        f["Geolocation/Longitude"] = lon
        f["Observations/NumObs"] = numobs.astype(np.int16)


@pytest.fixture
def ged_env(tmp_path, monkeypatch):
    """Point the mask at a tmp granule dir and a tmp (absent) artifact."""
    ged_dir = tmp_path / "aster_ged"
    monkeypatch.setattr(settings, "ged_dir", ged_dir)
    monkeypatch.setattr(settings, "ged_artifact", tmp_path / "ged_gap_mask.npz")
    return ged_dir


def fixture_granules(ged_dir, *, core_numobs: np.ndarray | None = None) -> None:
    write_granule(ged_dir, *CORE, core_numobs)
    for lat_top, lon_west in MARGIN:
        write_granule(ged_dir, lat_top, lon_west)


class TestGranuleName:
    def test_matches_the_archive_spelling(self):
        assert granule_name(-30, -65) == "AG1km.v003.-30.-065.0010.h5"
        assert granule_name(0, 8) == "AG1km.v003.00.008.0010.h5"
        assert granule_name(5, -2) == "AG1km.v003.05.-002.0010.h5"
        assert granule_name(40, -75) == "AG1km.v003.40.-075.0010.h5"
        assert granule_name(71, 25) == "AG1km.v003.71.025.0010.h5"


class TestDilation:
    def test_one_cell_buffer_is_eight_connected(self):
        cells = np.zeros((5, 5), dtype=bool)
        cells[2, 2] = True
        out = dilate_cells(cells, 1)
        assert out[1:4, 1:4].all()
        assert int(out.sum()) == 9

    def test_dilation_clips_at_the_array_edge(self):
        cells = np.zeros((3, 3), dtype=bool)
        cells[0, 0] = True
        out = dilate_cells(cells, 1)
        assert int(out.sum()) == 4  # the 2x2 corner

    def test_zero_buffer_is_identity(self):
        cells = np.zeros((3, 3), dtype=bool)
        cells[1, 2] = True
        assert dilate_cells(cells, 0) is cells

    def test_radius_two_is_a_five_by_five_footprint(self):
        cells = np.zeros((7, 7), dtype=bool)
        cells[3, 3] = True
        assert int(dilate_cells(cells, 2).sum()) == 25


class TestGapMaskFootprint:
    """A known gap cell lands on exactly its 36 x 36 native pixels, buffered."""

    def gap_at(self, cell_row: int, cell_col: int) -> np.ndarray:
        numobs = np.full((100, 100), 7, dtype=np.int16)
        numobs[cell_row, cell_col] = 0
        return numobs

    def test_unbuffered_gap_cell_footprint_is_exact(self, ged_env):
        # Cell (2, 3) of granule (40, -75): within the 10 x 10 cells BBOX
        # covers, local cell (2, 3) -> native rows [72, 108), cols [108, 144).
        fixture_granules(ged_env, core_numobs=self.gap_at(2, 3))
        mask = gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=0)
        assert mask.shape == (360, 360)
        assert mask[72:108, 108:144].all()
        assert int(mask.sum()) == 36 * 36

    def test_buffered_footprint_adds_the_ring(self, ged_env):
        fixture_granules(ged_env, core_numobs=self.gap_at(2, 3))
        mask = gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)
        assert mask[36:144, 72:180].all()
        assert int(mask.sum()) == 108 * 108

    def test_buffer_defaults_to_the_setting(self, ged_env, monkeypatch):
        fixture_granules(ged_env, core_numobs=self.gap_at(2, 3))
        monkeypatch.setattr(settings, "ged_gap_buffer_cells", 0)
        assert int(gap_mask_for_geobox(geobox_for_bbox(BBOX)).sum()) == 36 * 36
        monkeypatch.setattr(settings, "ged_gap_buffer_cells", 1)
        assert int(gap_mask_for_geobox(geobox_for_bbox(BBOX)).sum()) == 108 * 108

    def test_gap_across_the_margin_buffers_into_the_geobox(self, ged_env):
        # A gap cell in the *margin* granule north of BBOX, directly above its
        # first cell row: the buffer must reach one cell row into the geobox.
        fixture_granules(ged_env)
        north = np.full((100, 100), 7, dtype=np.int16)
        north[99, 2] = 0  # bottom row of granule (41, -75), cell col 2
        write_granule(ged_env, 41, -75, north)
        mask = gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)
        # Buffer spills into geobox cell row 0, cell cols 1..3.
        assert mask[0:36, 36:144].all()
        assert int(mask.sum()) == 36 * 108

    def test_a_flipped_granule_is_reoriented(self, ged_env):
        fixture_granules(ged_env, core_numobs=self.gap_at(2, 3))
        expected = gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)
        (ged_env / granule_name(*CORE)).unlink()
        write_granule(ged_env, *CORE, self.gap_at(2, 3), flip=True)
        flipped = gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)
        np.testing.assert_array_equal(flipped, expected)

    def test_a_band_slice_mask_is_the_slice_of_the_tile_mask(self, ged_env):
        fixture_granules(ged_env, core_numobs=self.gap_at(2, 3))
        tile = geobox_for_bbox(BBOX)
        whole = gap_mask_for_geobox(tile, buffer_cells=1)
        band = gap_mask_for_geobox(tile[100:200, :], buffer_cells=1)
        np.testing.assert_array_equal(band, whole[100:200, :])


class TestGranulePathErrors:
    def test_missing_core_granule_raises_naming_it(self, ged_env):
        for lat_top, lon_west in MARGIN:
            write_granule(ged_env, lat_top, lon_west)
        with pytest.raises(MissingGranuleError) as err:
            gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)
        assert granule_name(*CORE) in str(err.value)
        assert err.value.granules == [granule_name(*CORE)]

    def test_missing_margin_granule_is_tolerated(self, ged_env):
        # Only the corner margin granule is absent; the mask must still build
        # and carry no contribution from it -- matching what the artifact path
        # would produce from the same archive.
        write_granule(ged_env, *CORE)
        write_granule(ged_env, 40, -76)
        write_granule(ged_env, 41, -75)
        mask = gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)
        assert not mask.any()

    def test_no_source_at_all_refuses_loudly(self, ged_env):
        with pytest.raises(FileNotFoundError, match="build_ged_gap_mask"):
            gap_mask_for_geobox(geobox_for_bbox(BBOX))


class TestArtifact:
    def test_artifact_and_granules_build_identical_masks(self, ged_env):
        numobs = np.full((100, 100), 7, dtype=np.int16)
        numobs[0, 0] = 0  # NW corner: exercises the margin arithmetic
        numobs[5, 7] = 0
        fixture_granules(ged_env, core_numobs=numobs)
        north = np.full((100, 100), 7, dtype=np.int16)
        north[99, 4] = 0
        write_granule(ged_env, 41, -75, north)

        from_granules = gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)
        assert from_granules.any()

        build_artifact(ged_env, settings.ged_artifact)
        assert settings.ged_artifact.exists()
        from_artifact = gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1)
        np.testing.assert_array_equal(from_artifact, from_granules)

    def test_artifact_is_preferred_over_granules(self, ged_env):
        fixture_granules(ged_env)
        build_artifact(ged_env, settings.ged_artifact)
        # Poison the granule dir after the build: an artifact-first resolve
        # never opens it.
        for path in ged_env.glob("*.h5"):
            path.unlink()
        assert not gap_mask_for_geobox(geobox_for_bbox(BBOX), buffer_cells=1).any()

    def test_wrong_format_version_is_refused(self, ged_env, tmp_path):
        fixture_granules(ged_env)
        np.savez_compressed(
            settings.ged_artifact,
            format_version=np.int32(ged.ARTIFACT_FORMAT_VERSION + 1),
            gap_rows=np.empty(0, np.int32),
            gap_cols=np.empty(0, np.int32),
            coverage=np.zeros((180, 360), np.uint8),
            granule_count=np.int32(0),
        )
        with pytest.raises(ValueError, match="format"):
            gap_mask_for_geobox(geobox_for_bbox(BBOX))

    def test_build_reports_counts(self, ged_env):
        numobs = np.full((100, 100), 7, dtype=np.int16)
        numobs[3, 3] = 0
        fixture_granules(ged_env, core_numobs=numobs)
        report = build_artifact(ged_env, settings.ged_artifact)
        assert report["granules"] == 4
        assert report["gap_cells"] == 1
        # v2 also records provenance; tests/unit/test_ged_artifact.py owns it.
        assert len(report["content_sha256"]) == 64
        assert report["complete"] is False

    def test_build_refuses_an_empty_dir(self, ged_env):
        ged_env.mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="granules"):
            build_artifact(ged_env, settings.ged_artifact)
