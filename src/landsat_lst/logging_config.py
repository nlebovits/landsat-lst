"""Structlog configuration that stops one traceback from evicting a run's history.

On 2026-08-14 a single ``logger.exception("tile_failed", ...)`` call in
:mod:`landsat_lst.job` rendered 3.8 MB. The uploaded task log keeps the tail of
``settings.task_log_max_bytes``, so that one exception pushed the tile's entire
phase history out of the file. What arrived opened with
``3829375 earlier bytes dropped`` and said nothing about what the tile had been
doing when it died.

Nothing in this package called ``structlog.configure()``, so structlog fell back
to its default processor chain, which ends in a ``ConsoleRenderer`` whose
exception formatter is ``RichTracebackFormatter(show_locals=True)``. Rich then
printed every local in every frame, and one of those locals was the deserialized
Landsat collection. :func:`configure_logging` replaces that formatter with
:func:`structlog.dev.plain_traceback` and leaves the rest of the chain alone.
The traceback is worth keeping. The STAC collection sitting in one of its frames
is not.

Raising ``task_log_max_bytes`` is not the fix. A larger cap buys more of one
traceback at the price of the phase history that makes a run legible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from structlog.typing import Processor

__all__ = ["configure_logging", "default_processor_chain"]


def default_processor_chain() -> list[Processor]:
    """Build structlog's default processor chain with plain exception rendering.

    Every entry matches structlog's own default except the final renderer, which
    carries :func:`structlog.dev.plain_traceback` instead of the rich formatter
    that prints frame locals. Level filtering lives in the wrapper class rather
    than the chain, so it survives untouched.

    A fresh list comes back on every call, which lets
    :func:`configure_logging` with ``force=True`` install a renderer that shares
    no state with the previous one.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S", utc=False),
        structlog.dev.ConsoleRenderer(exception_formatter=structlog.dev.plain_traceback),
    ]


def configure_logging(force: bool = False) -> None:
    """Install the project's structlog configuration once.

    The CLI group and the Python API entry point both call this, and either can
    run first, so a call made after structlog is already configured returns
    without disturbing the existing setup.

    Args:
        force: Reconfigure even when structlog is already configured, replacing
            the processor chain with a freshly built one. A test that has to
            reinstall the chain over a configuration some earlier test left
            behind needs this, because the guard would otherwise skip it.
    """
    if structlog.is_configured() and not force:
        return

    structlog.configure(processors=default_processor_chain())
