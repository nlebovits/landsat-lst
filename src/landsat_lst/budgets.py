"""How long each stage of a sharded tile is allowed to take, and why.

Every barrier deadline in :mod:`landsat_lst.shard_driver` comes from here. None
of them is a hand-entered number any more, and that is the point: a stage
timeout typed into settings is a guess that ages badly and that nobody
recomputes when the geometry changes. A tile whose window doubles, whose fleet
halves, or whose composite rate is re-measured should move its own deadlines.

The model is the one ``projection.py`` already uses for cost -- bytes divided by
a measured rate -- applied per *shard* rather than per tile, plus the named
fixed costs a shard pays before it reads anything. Two rules keep it honest:

- **A budget is the work, and a deadline is the budget times a safety factor.**
  ``settings.shard_budget_safety`` is the only slack, it is named, and it is one
  number rather than one per stage. A deadline that has to be widened widens
  everywhere, which is a conversation rather than a silent edit.
- **Every constant carries its provenance.** A rate with no measurement behind
  it is a hand constant wearing a formula.

What this model does *not* claim: that a stage which fits its budget will
finish. It bounds how long the driver waits before concluding something is
wrong, which is a different question from how long the work takes. The S30W065
night made the distinction expensive -- a fleet Coiled had already killed sat
inside a 45-minute barrier because nothing was checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from landsat_lst.config import settings
from landsat_lst.projection import R_COMPOSITE_MB_S, R_OFFSETS_MB_S

if TYPE_CHECKING:
    from landsat_lst import shards

#: Cold boot to first line of Python on a Coiled Batch VM. Observed 4-5 minutes
#: across the S30W065 runs of 2026-08-21/22 (image pull dominates); 300 s is the
#: top of that range. Paid once per shard per round, concurrently.
VM_BOOT_S = 300.0

#: Shard 0's resolve: one STAC query over the window plus two lazy graph
#: builds. On S30W065 (4,403 items) ``plan.json`` landed within 3.5 minutes of
#: the fleet starting, boot included; 300 s is that with the boot taken out and
#: room for a slow catalog.
RESOLVE_S = 300.0

#: Assembling ~600 floats from the phase-B partials, in the driver. Kilobytes of
#: JSON in and out; the ceiling is a slow listing, not the work.
MERGE_OFFSETS_S = 60.0

#: Download every band slab, merge each into a full-tile intermediate, and
#: translate both to COGs. Three passes over the output rasters on one VM's
#: disk, so a disk-and-network rate rather than a decode rate.
R_EXPORT_MB_S = 100.0

#: Bytes per pixel per scene in the source pair: two ``uint16`` bands.
_SOURCE_BYTES_PER_PIXEL = 2 * 2

#: Output bytes per pixel: ``lst_p95`` as ``uint16`` plus 12 ``uint8`` months.
_OUTPUT_BYTES_PER_PIXEL = 2 + 12


@dataclass(frozen=True)
class StageBudget:
    """One stage's named phases, and the deadline they add up to."""

    stage: str
    #: Phase name to seconds, in the order the stage runs them. Named rather
    #: than summed on the spot so a barrier that expires can say *which* phase
    #: it was budgeted for, which is the first question anyone asks.
    phases: tuple[tuple[str, float], ...]

    @property
    def work_s(self) -> float:
        """What the stage is projected to take, with no slack."""
        return sum(seconds for _, seconds in self.phases)

    @property
    def deadline_s(self) -> float:
        """How long a round of this stage may run before the driver acts.

        An explicit ``settings.shard_barrier_timeout_s`` overrides it entirely,
        which is what a person reaching for a stopwatch during an incident
        needs. ``None`` -- the default -- means derived.
        """
        if settings.shard_barrier_timeout_s is not None:
            return float(settings.shard_barrier_timeout_s)
        return self.work_s * settings.shard_budget_safety

    def summary(self) -> str:
        """One line naming every phase, for the log a barrier writes on open."""
        parts = ", ".join(f"{name} {seconds / 60:.0f}m" for name, seconds in self.phases)
        return f"{self.stage}: {parts} = {self.work_s / 60:.0f}m x {self._factor():.1f}"

    def _factor(self) -> float:
        if settings.shard_barrier_timeout_s is not None:
            return self.deadline_s / self.work_s if self.work_s else 1.0
        return settings.shard_budget_safety


def _scene_count(plan: shards.TilePlan) -> int:
    """Time steps, not items: the axis is what every phase iterates."""
    return max(1, len(plan.scene_times))


def _coarse_bytes(plan: shards.TilePlan) -> float:
    """One full pass over the offset-resolution stack."""
    height, width = plan.coarse_shape
    return float(height) * float(width) * _scene_count(plan) * _SOURCE_BYTES_PER_PIXEL


def _native_bytes(plan: shards.TilePlan) -> float:
    """One full pass over the native stack (ADR-013: the tile reads it once)."""
    height, width = plan.native_shape
    return float(height) * float(width) * _scene_count(plan) * _SOURCE_BYTES_PER_PIXEL


def _widest_block_share(plan: shards.TilePlan) -> float:
    """The largest phase-A shard's share of the blocks.

    The *largest*, because a barrier waits for the slowest shard. Taken from
    the same ``balance_by_land`` split the shards themselves use rather than
    from ``blocks / ref_shards``: the groups are balanced on land, so an even
    division would understate the widest one on a coastal tile.
    """
    from landsat_lst.shards import balance_by_land  # noqa: PLC0415

    total = len(plan.blocks)
    if total == 0:
        return 1.0
    groups = balance_by_land(plan.blocks, plan.block_has_land, plan.ref_shards)
    return max(len(group) for group in groups) / total


def _widest_scene_share(plan: shards.TilePlan) -> float:
    """The largest phase-B shard's share of the scenes, for the same reason."""
    from landsat_lst.shards import partition  # noqa: PLC0415

    total = _scene_count(plan)
    groups = partition(plan.scene_batches, plan.scene_shards)
    widest = max(sum(stop - start for start, stop in group) for group in groups)
    return max(widest, 1) / total


def _widest_band_share(plan: shards.TilePlan) -> float:
    """The tallest row band's share of the output rows."""
    rows = plan.native_shape[0]
    if not plan.bands or rows == 0:
        return 1.0
    return max(stop - start for start, stop in plan.bands) / rows


def offsets_stage_budget(plan: shards.TilePlan) -> StageBudget:
    """The fused offsets stage: boot, resolve, climatology, phase B.

    All four are serial *within* a shard, and the shards run concurrently, so
    the stage's budget is the slowest shard's -- which is why every share above
    is the widest one rather than the mean.

    The phase-A barrier wait is not a phase of its own. A shard waits exactly
    as long as the slowest peer takes, and that peer's climatology is already
    counted here; adding a wait term would budget the same seconds twice.
    """
    coarse_mb = _coarse_bytes(plan) / 1e6
    climatology_s = coarse_mb * _widest_block_share(plan) / R_OFFSETS_MB_S
    phase_b_s = coarse_mb * _widest_scene_share(plan) / R_OFFSETS_MB_S
    return StageBudget(
        stage="offsets",
        phases=(
            ("boot", VM_BOOT_S),
            ("resolve", RESOLVE_S),
            ("climatology", climatology_s),
            ("phase_b", phase_b_s),
        ),
    )


def composite_stage_budget(plan: shards.TilePlan) -> StageBudget:
    """One row band, plus the tail of phase B it was started during.

    The composite fleet is started from inside the offsets barrier, so its
    round opens while phase B is still running and its VMs poll for a record
    the driver has not written yet. That wait is real time inside this stage's
    deadline, and budgeting only the band would give a fleet that booted early
    less time than one that booted late.
    """
    native_mb = _native_bytes(plan) / 1e6
    band_s = native_mb * _widest_band_share(plan) / R_COMPOSITE_MB_S
    offsets_tail_s = dict(offsets_stage_budget(plan).phases)["phase_b"]
    return StageBudget(
        stage="composite",
        phases=(
            ("boot", VM_BOOT_S),
            ("offsets_tail", offsets_tail_s),
            ("band", band_s),
        ),
    )


def export_stage_budget(plan: shards.TilePlan) -> StageBudget:
    """The merge and both COG translations, on one VM.

    Budgeted with a boot even though the usual path has none -- a composite
    worker claims it and is already running. This deadline governs the
    driver's *fallback* submission, which does boot.
    """
    height, width = plan.native_shape
    output_mb = float(height) * float(width) * _OUTPUT_BYTES_PER_PIXEL / 1e6
    # Downloaded, merged, then translated: three passes over the output.
    export_s = 3.0 * output_mb / R_EXPORT_MB_S
    return StageBudget(
        stage="export",
        phases=(("boot", VM_BOOT_S), ("merge_and_translate", export_s)),
    )


def merge_budget() -> StageBudget:
    """The driver's own offset merge. No boot: it runs in this process."""
    return StageBudget(stage="merge_offsets", phases=(("merge", MERGE_OFFSETS_S),))


#: Every stage the driver takes a barrier on, and what builds its budget.
_BUILDERS = {
    "offsets": offsets_stage_budget,
    "composite": composite_stage_budget,
    "export": export_stage_budget,
}


def stage_budget(stage: str, plan: shards.TilePlan) -> StageBudget:
    """The budget for one stage of one tile.

    Raises:
        ValueError: If ``stage`` is not a stage the driver runs a barrier on.
    """
    builder = _BUILDERS.get(stage)
    if builder is None:
        msg = f"no budget for stage {stage!r}; expected one of {sorted(_BUILDERS)}"
        raise ValueError(msg)
    return builder(plan)


def tile_budget_lines(plan: shards.TilePlan) -> list[str]:
    """Every stage's budget, for a driver that has just read the plan.

    Printed once per run so the numbers a barrier will act on are visible
    before they matter, rather than reconstructed afterwards from a timeout.
    """
    return [stage_budget(stage, plan).summary() for stage in _BUILDERS]
