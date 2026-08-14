"""The STAC client's retry policy, exercised against a server that fails.

One transient HTTP 500 from Earth Search ended a tile at second 10 of a
five-hour budget on 2026-08-14. ``pystac_client`` did mount a retry adapter,
but it built one from a plain integer, and ``Retry.from_int`` leaves
``status_forcelist`` empty and restricts ``allowed_methods`` to idempotent
verbs. So neither the 500 nor the POST search was ever retried.

These tests run against a real socket rather than a mocked session, because the
bug lived in the adapter's configuration and a mock would have reproduced
whatever configuration the test assumed.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from pystac_client.exceptions import APIError

from landsat_lst.config import settings
from landsat_lst.pipeline import _stac_retry, open_catalog

pytestmark = pytest.mark.unit

CONFORMANCE = [
    "https://api.stacspec.org/v1.0.0/core",
    "https://api.stacspec.org/v1.0.0/item-search",
]


def _catalog_body(root: str) -> dict[str, Any]:
    """The smallest root document ``Client.open`` will accept."""
    return {
        "type": "Catalog",
        "id": "test-catalog",
        "stac_version": "1.0.0",
        "description": "Catalog served by the retry tests.",
        "conformsTo": CONFORMANCE,
        "links": [{"rel": "self", "href": root}],
    }


class FlakyServer:
    """An HTTP server that fails its first ``failures`` requests.

    Counts every request it receives, so a test can assert how many attempts
    the client actually made rather than only whether it succeeded.
    """

    def __init__(self, *, failures: int, status: int = 500) -> None:
        self.failures = failures
        self.status = status
        self.requests: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def _record(self, method: str, path: str) -> bool:
        """Log one request and say whether it should fail."""
        with self._lock:
            self.requests.append((method, path))
            return len(self.requests) <= self.failures

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _reply(self, method: str) -> None:
                if server._record(method, self.path):
                    body = b'{"code":"InternalServerError","description":"Response Error"}'
                    self.send_response(server.status)
                else:
                    body = json.dumps(_catalog_body(server.url)).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # BaseHTTPRequestHandler names it
                self._reply("GET")

            def do_POST(self) -> None:  # BaseHTTPRequestHandler names it
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                self._reply("POST")

            def log_message(self, *args: object) -> None:
                """Keep the test output clean."""

        return Handler

    def __enter__(self) -> FlakyServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> bool:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        return False


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry without waiting.

    The policy's real backoff spans about 30 seconds, and unit tests run under
    a 30-second timeout. Timing is not what these tests are about.
    """
    monkeypatch.setattr(settings, "stac_retry_backoff_s", 0.0)


class TestRetryPolicy:
    """The policy object itself, independent of any server."""

    def test_covers_throttling_and_the_5xx_family(self) -> None:
        assert set(_stac_retry().status_forcelist) == {429, 500, 502, 503, 504}

    def test_allows_every_method_so_a_post_search_retries(self) -> None:
        # urllib3's default excludes POST, which is why the search never retried.
        assert _stac_retry().allowed_methods is None

    def test_backs_off_between_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "stac_retry_backoff_s", 1.5)
        assert _stac_retry().backoff_factor == 1.5

    def test_attempt_count_is_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "stac_retries", 2)
        assert _stac_retry().total == 2


class TestTransientFailures:
    """What the client does when the server is briefly unwell."""

    def test_a_transient_500_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with FlakyServer(failures=2) as server:
            monkeypatch.setattr(settings, "stac_url", server.url)
            catalog = open_catalog()

            assert catalog.id == "test-catalog"
            assert len(server.requests) == 3

    def test_a_post_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A STAC search is a POST. urllib3 retries only idempotent verbs by
        # default, so this is the half of the bug a GET test cannot see.
        with FlakyServer(failures=1) as server:
            monkeypatch.setattr(settings, "stac_url", server.url)
            catalog = open_catalog()
            server.requests.clear()

            response = catalog._stac_io.session.post(  # the adapter under test
                f"{server.url}/search", json={"limit": 1}, timeout=10
            )

            assert response.status_code == 200
            assert [method for method, _ in server.requests] == ["POST", "POST"]

    def test_every_5xx_in_the_list_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for status in (500, 502, 503, 504):
            with FlakyServer(failures=1, status=status) as server:
                monkeypatch.setattr(settings, "stac_url", server.url)

                assert open_catalog().id == "test-catalog"
                assert len(server.requests) == 2

    def test_throttling_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with FlakyServer(failures=1, status=429) as server:
            monkeypatch.setattr(settings, "stac_url", server.url)

            assert open_catalog().id == "test-catalog"
            assert len(server.requests) == 2


class TestRealOutages:
    """A server that stays down must fail the tile, not hold the VM."""

    def test_exhaustion_raises_rather_than_hanging(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "stac_retries", 2)
        with FlakyServer(failures=99) as server:
            monkeypatch.setattr(settings, "stac_url", server.url)

            with pytest.raises(APIError):
                open_catalog()

            # The first attempt plus two retries. A tile budgeted for five
            # hours must not spend coiled_job_timeout discovering an outage.
            assert len(server.requests) == 3

    def test_a_client_error_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A 404 says the request is wrong. Repeating it only spends the budget.
        with FlakyServer(failures=99, status=404) as server:
            monkeypatch.setattr(settings, "stac_url", server.url)

            with pytest.raises(APIError):
                open_catalog()

            assert len(server.requests) == 1
