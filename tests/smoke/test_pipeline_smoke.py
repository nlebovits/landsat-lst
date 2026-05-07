"""Smoke tests for the pipeline - minimal end-to-end validation."""

import pytest


@pytest.mark.smoke
class TestPipelineSmoke:
    def test_imports_work(self):
        """Verify all modules can be imported."""
        from landsat_lst import __version__

        assert __version__ is not None

    def test_settings_load(self):
        """Verify settings can be loaded."""
        from landsat_lst.config import settings

        assert settings.stac_url is not None
        assert settings.collection == "landsat-c2-l2"
        assert settings.tile_size_degrees == 5.0

    def test_can_generate_tiles(self):
        """Verify tile generation works."""
        from landsat_lst.tiling import generate_global_tiles

        tiles = list(generate_global_tiles())
        assert len(tiles) > 100

    def test_cli_help(self):
        """Verify CLI can show help."""
        from click.testing import CliRunner

        from landsat_lst.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Landsat" in result.output
