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
        assert settings.resolution == 1.0 / 3600

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

    def test_legacy_credit_quota_does_not_break_an_existing_dotenv(self, tmp_path):
        """The removed setting is tolerated for migration, but never trusted."""
        env_file = tmp_path / ".env"
        env_file.write_text("LST_COILED_CREDIT_QUOTA=1000\n")

        settings = Settings(_env_file=env_file)

        assert settings.coiled_credit_quota == 1000.0
        assert "coiled_credit_quota" not in settings.model_dump()
