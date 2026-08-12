"""Unit tests for public-HTTPS verification of published COGs.

No network: rasterio.open is stubbed. What is under test is the decision logic
and the URL construction, not GDAL.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from landsat_lst.cli import main
from landsat_lst.storage import PRODUCTS, LocalStorage, collection_prefix
from landsat_lst.verify import public_url, verify_tile


def _finish_tile(root, window: str, tile: str) -> None:
    for product in PRODUCTS:
        path = root / collection_prefix(window) / tile / f"{product}_{window}_{tile}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tif")


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(output_dir=tmp_path / "cogs")


def _fake_dataset(**overrides):
    """A stand-in for the rasterio dataset a public COG open would return."""
    dataset = MagicMock()
    dataset.dtypes = ["uint16"]
    dataset.height = 18000
    dataset.width = 18000
    dataset.nodata = 0.0
    dataset.scales = [0.01]
    dataset.offsets = [-50.0]
    dataset.overviews.return_value = [2, 4, 8, 16]
    for key, value in overrides.items():
        setattr(dataset, key, value)
    dataset.__enter__.return_value = dataset
    dataset.__exit__.return_value = False
    return dataset


class TestPublicUrl:
    def test_uses_the_source_coop_read_base(self):
        url = public_url("2021-2025", "N40W075", "lst_p95")

        assert url == (
            "https://data.source.coop/nlebovits/landsat-lst/"
            "lst-p95-2021-2025/N40W075/lst_p95_2021-2025_N40W075.tif"
        )

    def test_matches_the_storage_key_layout(self, storage):
        """A URL built from a different layout than storage writes is a 404."""
        url = public_url("2021-2025", "N40W075", "qa_count", storage=storage)

        assert url.endswith(storage.cog_key("2021-2025", "N40W075", "qa_count"))


class TestVerifyTile:
    def test_reports_encoding_of_a_readable_tile(self, storage):
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")

        with patch("rasterio.open", return_value=_fake_dataset()):
            check = verify_tile("N40W075", "2021-2025", storage=storage)

        assert check.ok
        lst = next(a for a in check.assets if a.product == "lst_p95")
        assert lst.dtype == "uint16"
        assert lst.shape == (18000, 18000)
        assert lst.scale == 0.01
        assert lst.offset == -50.0
        assert lst.overviews == [2, 4, 8, 16]

    def test_missing_tile_fails_without_opening_anything(self, storage):
        with patch("rasterio.open") as mock_open:
            check = verify_tile("N40W075", "2021-2025", storage=storage)

        assert not check.ok
        mock_open.assert_not_called()
        assert all("incomplete in storage" in a.error for a in check.assets)

    def test_half_written_tile_fails(self, storage):
        """One asset present is a tile to rebuild, not a tile to publish."""
        path = (
            storage.output_dir
            / collection_prefix("2021-2025")
            / "N40W075"
            / "lst_p95_2021-2025_N40W075.tif"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tif")

        check = verify_tile("N40W075", "2021-2025", storage=storage)

        assert not check.ok

    def test_unreadable_asset_fails_with_its_error(self, storage):
        """Present in the bucket but unreadable in public is still unpublished."""
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")

        with patch("rasterio.open", side_effect=OSError("HTTP 403")):
            check = verify_tile("N40W075", "2021-2025", storage=storage)

        assert not check.ok
        assert all("HTTP 403" in a.error for a in check.assets)

    def test_checks_both_assets(self, storage):
        _finish_tile(storage.output_dir, "2021-2025", "N40W075")

        with patch("rasterio.open", return_value=_fake_dataset()):
            check = verify_tile("N40W075", "2021-2025", storage=storage)

        assert {a.product for a in check.assets} == set(PRODUCTS)


class TestVerifyCommand:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_passing_tile_reports_encoding(self, runner):
        from landsat_lst.verify import AssetCheck, TileCheck

        check = TileCheck(
            tile="N40W075",
            window="2021-2025",
            assets=[
                AssetCheck(
                    product=product,
                    key="k",
                    url="https://example/k.tif",
                    exists=True,
                    dtype="uint16",
                    shape=(18000, 18000),
                    nodata=0.0,
                    scale=0.01,
                    offset=-50.0,
                    overviews=[2, 4, 8, 16],
                )
                for product in PRODUCTS
            ],
        )
        with patch("landsat_lst.verify.verify_tile", return_value=check):
            result = runner.invoke(main, ["verify", "-t", "N40W075"])

        assert result.exit_code == 0
        assert "OK N40W075" in result.output
        assert "uint16" in result.output
        assert "Verified: 1" in result.output

    def test_failing_tile_exits_non_zero(self, runner):
        from landsat_lst.verify import AssetCheck, TileCheck

        check = TileCheck(
            tile="N40W075",
            window="2021-2025",
            assets=[
                AssetCheck(
                    product=p, key="k", url="u", exists=False, error="tile incomplete in storage"
                )
                for p in PRODUCTS
            ],
        )
        with patch("landsat_lst.verify.verify_tile", return_value=check):
            result = runner.invoke(main, ["verify", "-t", "N40W075"])

        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_defaults_to_the_production_window(self, runner):
        with patch("landsat_lst.verify.verify_tile") as mock_verify:
            mock_verify.return_value.ok = False
            mock_verify.return_value.assets = []
            runner.invoke(main, ["verify", "-t", "N40W075"])

        assert mock_verify.call_args.args[1] == "2021-2025"

    def test_single_year_window(self, runner):
        with patch("landsat_lst.verify.verify_tile") as mock_verify:
            mock_verify.return_value.ok = False
            mock_verify.return_value.assets = []
            runner.invoke(main, ["verify", "-t", "N40W075", "--year", "2024"])

        assert mock_verify.call_args.args[1] == "2024"

    def test_urls_flag_prints_access_urls(self, runner):
        from landsat_lst.verify import AssetCheck, TileCheck

        check = TileCheck(
            tile="N40W075",
            window="2021-2025",
            assets=[
                AssetCheck(
                    product=p,
                    key="k",
                    url=f"https://data.source.coop/{p}.tif",
                    exists=True,
                    dtype="uint16",
                    shape=(1, 1),
                    nodata=0.0,
                    scale=0.01,
                    offset=-50.0,
                )
                for p in PRODUCTS
            ],
        )
        with patch("landsat_lst.verify.verify_tile", return_value=check):
            result = runner.invoke(main, ["verify", "-t", "N40W075", "--urls"])

        assert "https://data.source.coop/lst_p95.tif" in result.output
