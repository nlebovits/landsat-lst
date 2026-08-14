"""Where a printed figure came from, and how the files that hold them are read.

A number a command prints reads as fact. Some of ours are measured, some are
copied from a vendor's price list, some are arithmetic over the first two, and
some are guesses that a reader would never distinguish from the rest unless the
label travelled with the number. So every figure the planner and the cost
estimator emit carries a :class:`Provenance`, printed next to the figure rather
than buried in a docstring, because a reader deciding whether to trust a number
is looking at the number.

The committed data behind those figures lives in JSON files beside the modules
that read them, so a new measurement or a price change is an edit to data rather
than to code. :func:`load_committed_json` is the one reader for all of them.
"""

from __future__ import annotations

import json
from enum import StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger()


class Provenance(StrEnum):
    """How much weight a figure carries.

    Ordered from strongest to weakest. ``PUBLISHED`` sits between measurement
    and arithmetic. A vendor list rate was not measured here and was not
    computed here, and calling it assumed would understate a number the vendor
    commits to.
    """

    MEASURED = "measured"
    PUBLISHED = "published"
    DERIVED = "derived"
    ASSUMED = "assumed"


@lru_cache(maxsize=8)
def load_committed_json(path: Path) -> dict[str, Any]:
    """Read one committed data file, or an empty record if it cannot be read.

    A missing or malformed file must not stop the caller. The geometry half of
    a plan is still exact without a calibration record, and a reconcile is still
    worth having without a price. The caller says which half is missing rather
    than refusing to run.
    """
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:  # pragma: no cover - shipped with the package
        log.warning("committed_json_unreadable", path=str(path), error=str(e))
        return {}
