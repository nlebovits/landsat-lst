"""Refreshable Planetary Computer Azure SAS auth for long-running reads.

Planetary Computer signs blob URLs with SAS tokens that are valid for only
~45 minutes. odc-stac bakes those signed URLs into the lazy Dask task graph at
load time, so a composite that takes longer than the token lifetime fails
mid-compute with::

    RasterioIOError("'/vsicurl/https://...QA_PIXEL.TIF?...' not recognized as
    being in a supported file format.")

which is an authentication failure masquerading as a format error. Heavier
tiles (800+ scenes) reliably exceed the token window and never complete locally.

This module removes the dependency on URL-embedded tokens entirely:

1. Asset hrefs are rewritten to GDAL ``/vsiaz/`` virtual paths (token-free,
   static — safe to bake into the Dask graph). See :func:`to_vsiaz`.
2. The SAS token is published via the ``AZURE_STORAGE_SAS_TOKEN`` process
   environment variable, which GDAL reads fresh on every dataset open. It is
   deliberately NOT passed through ``configure_rio``: odc-stac snapshots the
   rasterio config at graph-build time and freezes it into every Dask task, so
   a config-delivered token cannot be refreshed mid-compute. An env-var token
   sidesteps that snapshot. See :func:`refresh_sas_token`.
3. A background timer mints a fresh full-lifetime token and updates the env var
   — in the local process and on every Dask worker — well inside the token
   lifetime, so reads never go stale.

This is Planetary-Computer-only plumbing. The AWS / Earth Search path (which
uses requester-pays S3 with its own credentials) is untouched.
"""

import re
import threading
from collections import Counter
from collections.abc import Callable
from urllib.parse import urlparse

import structlog
from odc.stac import configure_rio

log = structlog.get_logger()

# PC tokens last ~45 min; refresh comfortably inside that so a token is always
# valid even if one refresh is delayed or fails and the next one covers it.
REFRESH_INTERVAL_SECONDS = 15 * 60

# Matches "<account>.blob.core.windows.net"
_BLOB_HOST_RE = re.compile(r"^([a-z0-9]+)\.blob\.core\.windows\.net$")


def parse_blob_url(href: str) -> tuple[str, str, str] | None:
    """Parse an Azure blob URL into ``(account, container, blob_path)``.

    Returns ``None`` for anything that is not an Azure blob https URL (already
    a ``/vsiaz/`` path, an S3 URL, a local path, etc.).
    """
    parsed = urlparse(href)
    match = _BLOB_HOST_RE.match(parsed.netloc)
    if not match:
        return None
    account = match.group(1)
    container, _, blob = parsed.path.lstrip("/").partition("/")
    if not container or not blob:
        return None
    return account, container, blob


def to_vsiaz(href: str) -> str:
    """Rewrite an Azure blob https URL to a GDAL ``/vsiaz/`` path.

    Any existing SAS query string is dropped — authentication is supplied
    separately via the ``AZURE_STORAGE_SAS_TOKEN`` GDAL config option. Non-blob
    URLs are returned unchanged, so this is safe as a blanket ``patch_url``.
    """
    parsed = parse_blob_url(href)
    if parsed is None:
        return href
    _account, container, blob = parsed
    return f"/vsiaz/{container}/{blob}"


def primary_storage(items: list) -> tuple[str, str] | None:
    """Return the most common ``(account, container)`` across all item assets.

    GDAL's ``/vsiaz/`` driver authenticates against a single account+token, so
    we pick the dominant storage location. For Landsat C2 L2 every asset lives
    in ``landsateuwest/landsat-c2``, so there is only ever one.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    for item in items:
        for asset in item.assets.values():
            parsed = parse_blob_url(asset.href)
            if parsed is not None:
                account, container, _blob = parsed
                pairs[(account, container)] += 1
    if not pairs:
        return None
    if len(pairs) > 1:
        log.warning("azure_multiple_storage_accounts", accounts=dict(pairs))
    (account, container), _count = pairs.most_common(1)[0]
    return account, container


def refresh_sas_token(account: str, container: str) -> float:
    """Mint a fresh container SAS token and publish it via process env vars.

    Returns the seconds remaining until the new token expires.

    The token MUST be delivered through ``os.environ`` (``AZURE_STORAGE_*``),
    NOT through ``configure_rio``. odc-stac captures the rasterio/GDAL config
    once at graph-build time and serialises that snapshot into every Dask task
    (``capture_env`` -> ``restore_env``), so a token passed to ``configure_rio``
    is frozen at build time and worker-side refreshes never reach the reads —
    the read fails the moment that frozen token expires (~45min; issue #31).
    GDAL instead reads ``AZURE_STORAGE_SAS_TOKEN`` from the process environment
    fresh on every dataset open, so updating ``os.environ`` on each worker does
    reach subsequent reads. (Verified: an env-var token is not captured into the
    frozen build env.)

    ``planetary_computer.sas.get_token`` serves a cached token until it is
    within 60s of expiry; we clear that cache so every refresh mints a
    brand-new, full-lifetime token and the active token always has tens of
    minutes of headroom — no read ever races expiry.
    """
    import os  # noqa: PLC0415

    import planetary_computer as pc  # noqa: PLC0415

    try:
        from planetary_computer.sas import TOKEN_CACHE  # noqa: PLC0415

        TOKEN_CACHE.clear()
    except ImportError:  # pragma: no cover - defensive against PC internals moving
        log.warning("azure_sas_cache_clear_unavailable")

    token = pc.sas.get_token(account, container)
    os.environ["AZURE_STORAGE_ACCOUNT"] = account
    os.environ["AZURE_STORAGE_SAS_TOKEN"] = token.token
    return token.ttl()


class _SasRefresher:
    """Apply a SAS token now, then keep refreshing it on a daemon thread."""

    def __init__(self, account: str, container: str, interval: float):
        self._account = account
        self._container = container
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # Enable GDAL cloud defaults (HTTP retries etc.) once. These are static
        # and safe to freeze into the odc build env; the retries also smooth
        # over any transient 403 right at a token-refresh boundary.
        configure_rio(cloud_defaults=True)
        # Initial refresh is synchronous so the very first read is authenticated.
        ttl = refresh_sas_token(self._account, self._container)
        log.info(
            "azure_sas_configured",
            account=self._account,
            container=self._container,
            ttl_seconds=round(ttl),
            refresh_interval=self._interval,
        )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                ttl = refresh_sas_token(self._account, self._container)
                log.info("azure_sas_refreshed", ttl_seconds=round(ttl))
            except Exception as exc:
                log.warning("azure_sas_refresh_failed", error=str(exc))

    def stop(self) -> None:
        self._stop.set()


# distributed is optional at import time; the worker plugin is only built when a
# Dask client is present.
def _make_worker_plugin(account: str, container: str, interval: float):
    from distributed import WorkerPlugin  # noqa: PLC0415

    class AzureSasRefreshPlugin(WorkerPlugin):
        """Refresh the Azure SAS token inside each Dask worker process."""

        name = "azure-sas-refresh"

        def __init__(self) -> None:
            self._refresher: _SasRefresher | None = None

        def setup(self, worker) -> None:  # noqa: ARG002 - required by WorkerPlugin API
            self._refresher = _SasRefresher(account, container, interval)
            self._refresher.start()

        def teardown(self, worker) -> None:  # noqa: ARG002 - required by WorkerPlugin API
            if self._refresher is not None:
                self._refresher.stop()

    return AzureSasRefreshPlugin()


# Track local refreshers so we don't spawn duplicates on repeated calls.
_local_refreshers: dict[tuple[str, str], _SasRefresher] = {}


def enable_pc_azure_refresh(
    items: list,
    *,
    interval: float = REFRESH_INTERVAL_SECONDS,
) -> Callable[[str], str]:
    """Set up refreshable Azure SAS auth for the storage backing ``items``.

    Configures the local process immediately and, if a Dask client is active,
    registers a worker plugin so every worker refreshes its own token. Returns
    the :func:`to_vsiaz` ``patch_url`` callable to pass to ``stac_load``; pass
    it through even when there is no Azure storage (it is a no-op on non-blob
    URLs).

    Idempotent per ``(account, container)`` for the local process.
    """
    storage = primary_storage(items)
    if storage is None:
        # No Azure-backed assets (e.g. AWS path). Nothing to refresh.
        return to_vsiaz
    account, container = storage

    if (account, container) not in _local_refreshers:
        refresher = _SasRefresher(account, container, interval)
        refresher.start()
        _local_refreshers[(account, container)] = refresher

    try:
        from distributed import get_client  # noqa: PLC0415

        client = get_client()
    except (ImportError, ValueError):
        client = None

    if client is not None:
        plugin = _make_worker_plugin(account, container, interval)
        client.register_plugin(plugin)
        log.info("azure_sas_worker_plugin_registered", account=account, container=container)

    return to_vsiaz
