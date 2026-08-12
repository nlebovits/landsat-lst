"""Run the Portolan validator over a built catalog, in process.

The validator is `rashid`. It is called through its documented entry point,
``rashid.validate``, rather than by shelling out, so the findings arrive as
objects and the accepted-warning baseline can be applied to them directly.

Every pass used here is offline: the structural and profile schemas ship inside
the rashid wheel, and the data pass reads asset bytes through relative hrefs
that resolve inside the catalog tree. The live pass, the only one that reaches
a network, stays off.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rashid import validate

if TYPE_CHECKING:
    from pathlib import Path

    from rashid import Report

#: Warnings this catalog accepts, each for a stated reason. A warning outside
#: this set is a regression: the conformance gate fails on it rather than
#: letting it accumulate.
ACCEPTED_WARNING_RULE_IDS = frozenset(
    {
        # Assets carry file:size but no file:checksum. A checksum would have to
        # be recomputed on every republish of a multi-terabyte dataset, and
        # Portolan makes it a SHOULD for exactly that reason.
        "PTL-AST-003",
        # qa_count declares no nodata, because 0 ("no valid observation this
        # month") is a real value that must stay visible. A band without nodata
        # only SHOULD declare STATISTICS_VALID_PERCENT, so its absence warns.
        "PTL-DAT-010",
    }
)


def validate_catalog(path: Path | str) -> Report:
    """Validate a built catalog with every offline pass rashid offers."""
    return validate(path, structural=True, schema=True, data=True, live=False)


def unaccepted_warnings(report: Report) -> set[str]:
    """Rule ids of warnings that the frozen baseline does not cover."""
    return {finding.rule_id for finding in report.warnings} - ACCEPTED_WARNING_RULE_IDS
