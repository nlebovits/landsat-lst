"""Unit tests for configuration settings."""

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

    def test_resolution_default(self):
        """Verify default resolution is ~30m in degrees (0.00027778°)."""
        settings = Settings()
        assert settings.resolution == 0.00027778

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
        """The cap is provisional, so it must be tunable without a code change."""
        monkeypatch.setenv("LST_DESTRIPE_MAX_OFFSET_C", "8.5")
        settings = Settings()
        assert settings.destripe_max_offset_c == 8.5
