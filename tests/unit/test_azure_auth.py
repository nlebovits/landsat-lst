"""Unit tests for refreshable Planetary Computer Azure SAS auth.

Covers the URL rewriting / storage-detection helpers (pure functions) and the
token-refresh plumbing (mocked, so no network or GDAL is touched). See
landsat_lst.azure_auth and issue #31.
"""

from types import SimpleNamespace

import pytest

from landsat_lst import azure_auth

LANDSAT_HREF = (
    "https://landsateuwest.blob.core.windows.net/landsat-c2/level-2/standard/"
    "oli-tirs/2024/042/037/LC09_L2SP_042037_20240220_02_T1/"
    "LC09_L2SP_042037_20240220_02_T1_ST_B10.TIF"
)


class TestParseBlobUrl:
    def test_parses_account_container_blob(self):
        account, container, blob = azure_auth.parse_blob_url(LANDSAT_HREF)
        assert account == "landsateuwest"
        assert container == "landsat-c2"
        assert blob.startswith("level-2/standard/")
        assert blob.endswith("_ST_B10.TIF")

    def test_ignores_existing_sas_query(self):
        account, _container, blob = azure_auth.parse_blob_url(LANDSAT_HREF + "?st=2026&sig=abc")
        assert account == "landsateuwest"
        assert "?" not in blob

    @pytest.mark.parametrize(
        "href",
        [
            "https://example.com/foo/bar.tif",  # not a blob host
            "s3://usgs-landsat/collection02/foo.TIF",  # AWS path
            "/vsiaz/landsat-c2/already/vsiaz.tif",  # already rewritten
            "https://landsateuwest.blob.core.windows.net/landsat-c2",  # no blob path
        ],
    )
    def test_returns_none_for_non_blob(self, href):
        assert azure_auth.parse_blob_url(href) is None


class TestToVsiaz:
    def test_rewrites_blob_to_vsiaz_and_drops_token(self):
        result = azure_auth.to_vsiaz(LANDSAT_HREF + "?st=2026&se=2026&sig=secret")
        assert result == (
            "/vsiaz/landsat-c2/level-2/standard/oli-tirs/2024/042/037/"
            "LC09_L2SP_042037_20240220_02_T1/LC09_L2SP_042037_20240220_02_T1_ST_B10.TIF"
        )
        assert "sig=" not in result

    def test_passes_through_non_blob_urls(self):
        s3 = "s3://usgs-landsat/foo.TIF"
        assert azure_auth.to_vsiaz(s3) == s3


def _item_with_hrefs(*hrefs):
    assets = {f"b{i}": SimpleNamespace(href=h) for i, h in enumerate(hrefs)}
    return SimpleNamespace(assets=assets)


class TestPrimaryStorage:
    def test_picks_single_landsat_storage(self):
        items = [_item_with_hrefs(LANDSAT_HREF, LANDSAT_HREF) for _ in range(3)]
        assert azure_auth.primary_storage(items) == ("landsateuwest", "landsat-c2")

    def test_none_when_no_azure_assets(self):
        items = [_item_with_hrefs("s3://usgs-landsat/foo.TIF")]
        assert azure_auth.primary_storage(items) is None

    def test_picks_dominant_when_mixed(self):
        other = "https://otheracct.blob.core.windows.net/other-c/path/x.tif"
        items = [
            _item_with_hrefs(LANDSAT_HREF, LANDSAT_HREF),
            _item_with_hrefs(LANDSAT_HREF, other),
        ]
        assert azure_auth.primary_storage(items) == ("landsateuwest", "landsat-c2")


class TestRefreshSasToken:
    @staticmethod
    def _install_fake_pc(monkeypatch, *, token_cache):
        """Install fake planetary_computer + planetary_computer.sas modules."""
        import sys

        fake_token = SimpleNamespace(token="st=2026&sig=fresh", ttl=lambda: 2700.0)
        fake_sas = SimpleNamespace(get_token=lambda *_: fake_token, TOKEN_CACHE=token_cache)
        fake_pc = SimpleNamespace(sas=fake_sas)
        monkeypatch.setitem(sys.modules, "planetary_computer", fake_pc)
        monkeypatch.setitem(sys.modules, "planetary_computer.sas", fake_sas)
        return fake_token

    def test_publishes_token_via_os_environ_not_configure_rio(self, monkeypatch):
        # The token MUST go through os.environ (read fresh per-open), NOT through
        # configure_rio (frozen into the odc build env -> can't refresh). #31
        self._install_fake_pc(monkeypatch, token_cache={})
        monkeypatch.delenv("AZURE_STORAGE_SAS_TOKEN", raising=False)
        monkeypatch.delenv("AZURE_STORAGE_ACCOUNT", raising=False)

        ttl = azure_auth.refresh_sas_token("landsateuwest", "landsat-c2")

        import os

        assert ttl == 2700.0
        assert os.environ["AZURE_STORAGE_ACCOUNT"] == "landsateuwest"
        assert os.environ["AZURE_STORAGE_SAS_TOKEN"] == "st=2026&sig=fresh"

    def test_clears_token_cache_to_force_fresh_mint(self, monkeypatch):
        # A populated cache would otherwise make get_token serve a stale token
        # until it nearly expires (the ~45min race). refresh must clear it.
        cache = {"https://.../landsateuwest/landsat-c2": object()}
        self._install_fake_pc(monkeypatch, token_cache=cache)

        azure_auth.refresh_sas_token("landsateuwest", "landsat-c2")

        assert cache == {}  # cache cleared -> a fresh, full-lifetime token is minted


class TestEnablePcAzureRefresh:
    def test_no_azure_assets_returns_passthrough_without_refresher(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            azure_auth, "refresh_sas_token", lambda a, c: calls.append((a, c)) or 2700.0
        )
        patch_url = azure_auth.enable_pc_azure_refresh([_item_with_hrefs("s3://x/y.TIF")])
        assert patch_url is azure_auth.to_vsiaz
        assert calls == []  # no Azure storage -> nothing refreshed

    def test_starts_local_refresher_for_azure_storage(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            azure_auth, "refresh_sas_token", lambda a, c: calls.append((a, c)) or 2700.0
        )
        # Pretend distributed has no active client.
        azure_auth._local_refreshers.clear()
        items = [_item_with_hrefs(LANDSAT_HREF)]

        patch_url = azure_auth.enable_pc_azure_refresh(items, interval=9999)

        assert patch_url is azure_auth.to_vsiaz
        assert ("landsateuwest", "landsat-c2") in azure_auth._local_refreshers
        assert calls == [("landsateuwest", "landsat-c2")]  # initial sync refresh
        # idempotent: second call does not start a second refresher
        azure_auth.enable_pc_azure_refresh(items, interval=9999)
        assert calls == [("landsateuwest", "landsat-c2")]

        # cleanup the daemon thread started by the refresher
        azure_auth._local_refreshers[("landsateuwest", "landsat-c2")].stop()
        azure_auth._local_refreshers.clear()
