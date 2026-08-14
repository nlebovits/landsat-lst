"""Measured residuals between what the planner predicts and what a run did.

A planning tool that is confidently wrong is worse than no planning tool. It is
the same failure ``scripts/measure_memory_scaling.py`` produced by assuming
ratios transfer from a 0.25 degree AOI, and it costs more, because a number
printed by a command reads as measured rather than assumed.

So every figure the planner emits carries a :class:`Provenance` saying where it
came from, and the two figures that cannot be derived from geometry alone -- how
far a real peak runs above the floor, and how fast tasks actually retire -- are
read from ``calibration.json`` rather than guessed. That file is data, so the
next tile run appends a record instead of reopening the argument. See issue #77
item 3.

Both constants are recorded per VM type and per thread count, because neither
transfers. The ~350 tasks/s below is an ``r6i.4xlarge`` at 4 threads and says
nothing about an ``m6i.4xlarge`` or about 16 threads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from landsat_lst.provenance import Provenance, load_committed_json

__all__ = [
    "CALIBRATION_PATH",
    "PeakResidual",
    "Provenance",
    "Throughput",
    "peak_residuals",
    "throughput_for",
    "wall_time_minutes",
]

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")


@dataclass(frozen=True)
class PeakResidual:
    """How far one real run's peak ran above the floor predicted for it."""

    vm_type: str
    threads: int
    measured_peak_gib: float
    scenes: int
    chunk_size: int
    run_id: str
    measured_on: str


@dataclass(frozen=True)
class Throughput:
    """Fused tasks retired per second, for one VM, thread count, and phase."""

    vm_type: str
    threads: int
    phase: str
    tasks_per_second: float
    measured_on: str
    source: str


def throughput_for(*, vm_type: str, threads: int, phase: str) -> Throughput | None:
    """The measured task rate for this VM, thread count, and phase, if recorded.

    Deliberately exact rather than nearest-match on all three. Interpolating
    between instance types, or reusing the offset pass's rate for the composite,
    would manufacture a measurement -- which is the failure this module exists
    to prevent. No record means the caller prints nothing, not a guess.
    """
    for row in load_committed_json(CALIBRATION_PATH).get("throughput", []):
        if (
            row.get("vm_type") == vm_type
            and row.get("threads") == threads
            and row.get("phase") == phase
        ):
            return Throughput(
                vm_type=row["vm_type"],
                threads=row["threads"],
                phase=row["phase"],
                tasks_per_second=float(row["tasks_per_second"]),
                measured_on=row.get("measured_on", "unknown"),
                source=row.get("source", ""),
            )
    return None


def peak_residuals(
    *, phase: str | None = None, exclude_vm: str | None = None
) -> list[PeakResidual]:
    """Recorded peaks, optionally for one phase and excluding one VM type.

    ``exclude_vm`` keeps a laptop's synthetic numbers out of a statement about
    production hardware, without deleting them: the gap between the two is
    itself the open question.
    """
    out = []
    for row in load_committed_json(CALIBRATION_PATH).get("peak_residuals", []):
        if phase is not None and row.get("phase") != phase:
            continue
        if exclude_vm is not None and row.get("vm_type") == exclude_vm:
            continue
        out.append(
            PeakResidual(
                vm_type=row["vm_type"],
                threads=row["threads"],
                measured_peak_gib=float(row["measured_peak_gib"]),
                scenes=int(row["scenes"]),
                chunk_size=int(row["chunk_size"]),
                run_id=row.get("run_id", "unknown"),
                measured_on=row.get("measured_on", "unknown"),
            )
        )
    return out


def wall_time_minutes(tasks: int, rate: Throughput) -> float:
    """Minutes to retire ``tasks`` at a measured rate."""
    return tasks / rate.tasks_per_second / 60.0
