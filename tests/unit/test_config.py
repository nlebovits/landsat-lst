"""Unit tests for configuration settings."""

import pytest
from pydantic import ValidationError

from landsat_lst.config import Settings


class TestSettings:
    def test_max_cloud_cover_default_is_100(self):
        """Verify scene-level filtering is disabled by default.

        Scene-level cloud cover filtering is redundant because pixel-level
        QA masking (qa_pixel band) already handles cloud/shadow filtering
        per-pixel. Setting the default to 100 disables scene-level filtering
        while maintaining data quality through pixel-level masking.

        See: https://github.com/nlebovits/landsat-lst/issues/34
        """
        settings = Settings()
        assert settings.max_cloud_cover == 100

    def test_max_cloud_cover_can_be_overridden(self):
        """Verify max_cloud_cover can be set to a custom value."""
        settings = Settings(max_cloud_cover=20)
        assert settings.max_cloud_cover == 20

    def test_resolution_is_exactly_one_over_pixels_per_degree(self):
        """Resolution derives from an integer pixel density, not a rounded float."""
        settings = Settings()
        assert settings.pixels_per_degree == 3600
        assert settings.source_resolution == 1.0 / 3600

    def test_grid_is_integral(self):
        """The globe and a tile both land on whole pixels (ADR-008)."""
        settings = Settings()
        ppd = settings.pixels_per_degree
        assert 360.0 * ppd == 1_296_000
        assert (settings.max_latitude - settings.min_latitude) * ppd == 432_000
        assert settings.tile_size_degrees * ppd == 18_000

    def test_fractional_grid_is_rejected(self):
        """A tile size that lands mid-pixel fails at construction, not at write time."""
        with pytest.raises(ValidationError, match="not a whole number"):
            Settings(tile_size_degrees=0.00015)

    def test_the_three_grids_are_separate_and_named(self):
        """Source, delivered, and offset. Conflating them is the ADR-017 trap."""
        settings = Settings()
        assert settings.source_resolution == 1.0 / 3600
        assert settings.output_resolution == 1.0 / 1200
        assert (
            settings.offset_resolution == (1.0 / 3600) * settings.destripe_offset_resolution_factor
        )

    def test_the_delivered_grid_is_an_exact_third_of_the_source(self):
        settings = Settings()
        assert settings.output_pixels_per_degree == 1200
        assert settings.spatial_aggregation_factor == 3
        assert settings.pixels_per_degree == 3 * settings.output_pixels_per_degree

    def test_the_delivered_grid_is_integral_too(self):
        """A five-degree tile is 6,000 delivered pixels, and the globe divides."""
        settings = Settings()
        oppd = settings.output_pixels_per_degree
        assert 360.0 * oppd == 432_000
        assert (settings.max_latitude - settings.min_latitude) * oppd == 144_000
        assert settings.tile_size_degrees * oppd == 6_000

    def test_an_output_grid_that_does_not_divide_the_source_is_rejected(self):
        """A partial block cannot sit on a shared global grid."""
        with pytest.raises(ValidationError, match="does not divide"):
            Settings(output_pixels_per_degree=700)

    def test_the_valid_area_default_is_five_of_nine(self):
        assert Settings().min_valid_source_cells == 5

    @pytest.mark.parametrize("cells", [0, 10])
    def test_a_valid_area_rule_outside_the_block_is_rejected(self, cells):
        """0 would emit a temperature with no observation; 10 is unreachable."""
        with pytest.raises(ValidationError, match="outside"):
            Settings(min_valid_source_cells=cells)

    def test_collection_default(self):
        """Verify default collection is landsat-c2-l2."""
        settings = Settings()
        assert settings.collection == "landsat-c2-l2"

    def test_destripe_defaults(self):
        """De-striping is on by default (issue #46, ADR-007)."""
        settings = Settings()
        assert settings.destripe is True
        assert settings.destripe_max_offset_c == 15.0
        assert settings.destripe_min_scene_pixels == 500

    def test_destripe_can_be_disabled(self):
        """The flag exists so raw composites stay reachable for benchmarking."""
        settings = Settings(destripe=False)
        assert settings.destripe is False

    def test_destripe_cap_from_env(self, monkeypatch):
        """The cap is climate-dependent, so it must be tunable without a code change."""
        monkeypatch.setenv("LST_DESTRIPE_MAX_OFFSET_C", "8.5")
        settings = Settings()
        assert settings.destripe_max_offset_c == 8.5
