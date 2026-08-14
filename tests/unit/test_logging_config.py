"""Tests for the structlog configuration that keeps tracebacks small.

The load-bearing test here is :func:`test_large_frame_local_is_not_rendered`.
Asserting only that the exception formatter is
:func:`structlog.dev.plain_traceback` would also pass against a formatter that
printed frame locals anyway, and printing frame locals is what turned one
``tile_failed`` line into 3.8 MB and evicted a tile's phase history from its
uploaded log.
"""

from __future__ import annotations

import io
from types import FunctionType
from typing import TYPE_CHECKING

import pytest
import structlog

from landsat_lst.logging_config import configure_logging, default_processor_chain

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

SENTINEL = "landsat-lst-frame-local-sentinel"
ERROR_MESSAGE = "synthetic tile failure"


class BulkyLocal:
    """Stand-in for the deserialized Landsat collection that blew up the log.

    Rich pretty-prints a frame local by parsing its ``repr``, and it expands a
    nested structure in full rather than truncating it. An object holding a few
    thousand entries therefore renders hundreds of kilobytes, which is the shape
    of the real failure. A long flat string would be truncated at
    ``locals_max_string`` and prove nothing.
    """

    def __init__(self) -> None:
        self.entries = {f"{SENTINEL}-{i}": f"value-{i}" for i in range(3000)}

    def __repr__(self) -> str:
        return f"BulkyLocal({self.entries!r})"


def raise_holding_a_bulky_local() -> None:
    """Raise from a frame that holds the bulky object as a local variable."""
    payload = BulkyLocal()
    raise ValueError(f"{ERROR_MESSAGE} with {len(payload.entries)} entries loaded")


@pytest.fixture(autouse=True)
def restore_structlog_defaults() -> Iterator[None]:
    """Keep configuration from leaking between tests sharing an xdist worker."""
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def render_tile_failure() -> str:
    """Log an exception through the configured chain and return what was written.

    Only the logger factory is overridden, so the processors under test render
    the record exactly as they would in a running tile.
    """
    stream = io.StringIO()
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=stream))
    log = structlog.get_logger()
    try:
        raise_holding_a_bulky_local()
    except ValueError:
        log.exception("tile_failed")
    return stream.getvalue()


def test_exception_formatter_is_plain_traceback() -> None:
    """The installed renderer formats exceptions without frame locals."""
    configure_logging()

    renderer = structlog.get_config()["processors"][-1]

    assert isinstance(renderer, structlog.dev.ConsoleRenderer)
    assert renderer._exception_formatter is structlog.dev.plain_traceback


def test_second_call_is_a_no_op_and_force_reconfigures() -> None:
    """The guard skips a repeat call, and ``force`` overrides the guard."""
    configure_logging()
    first = structlog.get_config()["processors"][-1]

    configure_logging()
    assert structlog.get_config()["processors"][-1] is first

    configure_logging(force=True)
    forced = structlog.get_config()["processors"][-1]
    assert forced is not first
    assert forced._exception_formatter is structlog.dev.plain_traceback


def test_large_frame_local_is_not_rendered() -> None:
    """A bulky frame local stays out of the rendered exception.

    Under structlog's default chain this same record renders around 400 KB and
    contains the sentinel, because ``RichTracebackFormatter`` runs with
    ``show_locals=True``.
    """
    configure_logging()

    output = render_tile_failure()

    assert SENTINEL not in output
    assert "tile_failed" in output
    assert "ValueError" in output
    assert ERROR_MESSAGE in output
    assert len(output) < 4000


def processor_identity(processor: object) -> object:
    """Identify a processor by the function itself, or by the class of an instance.

    Processor instances such as ``TimeStamper`` do not compare equal, so two
    chains built the same way can only be matched on the classes involved.
    """
    return processor if isinstance(processor, FunctionType) else type(processor)


def test_chain_matches_structlog_defaults_apart_from_the_renderer() -> None:
    """Every processor is still the one structlog installs by default.

    A structlog release that adds or reorders a default processor would
    otherwise leave this package quietly logging less than it used to.
    """
    structlog.reset_defaults()
    default_chain = structlog.get_config()["processors"]

    chain = default_processor_chain()

    assert [processor_identity(p) for p in chain] == [processor_identity(p) for p in default_chain]
