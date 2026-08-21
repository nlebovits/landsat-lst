"""Production-tile wall-time and cost projection from measured rates.

Turns the Stage-2 probe's measured per-VM throughputs into the numbers the
60-minute-tile requirement is judged against: minutes per tile on one VM,
VM-hours per tile, and the fleet sizes that fit the phase budgets. Every
number here is a projection from a measured rate, tagged with its source;
none is a measurement of a full tile. The acceptance run
(``landsat-lst process --tile ... --distributed``) is what turns the
projection into a fact.

Rates come from ``scripts/probe_io_ladder.py`` (results/probe/, 2026-08-21):

- Offset pass (coarse, chunk 1024, 8 I/O threads): 158.4 MB/s measured;
  155.0 used, shaving the single-arm optimism.
- Composite (native, chunk 512, 16 threads): 70-85 MB/s warm across three
  arms; 75.0 used.

The byte model:

- Coarse pass bytes = (native_edge / factor)^2 * scenes * 2 bands * 2 B.
  Phase A pays it scaled by the tile's land fraction (blocks with no land
  are skipped and never read); phase B reads the full footprint.
- Native pass bytes = native_edge^2 * scenes * 2 bands * 2 B, read once
  (ADR-013), full footprint regardless of land.
"""

from __future__ import annotations

from dataclasses import dataclass

from landsat_lst.config import settings

#: Measured per-VM rates and their provenance. Update only from a probe run.
PROBE_AS_OF = "2026-08-21"
R_OFFSETS_MB_S = 155.0  # chunk 1024, 8 io threads; probe v2/v3
R_COMPOSITE_MB_S = 75.0  # chunk 512, 16 threads; probe v2/v3

#: Phase budgets inside the 60-minute tile (the rest is setup + export).
OFFSET_BUDGET_MIN = 15.0
COMPOSITE_BUDGET_MIN = 38.0

#: r6i.2xlarge on-demand; spot spans 0.30-0.75 of it (pricing.json discipline:
#: a range, never a scalar).
VM_HOURLY_ON_DEMAND = 0.504
SPOT_FACTOR_RANGE = (0.30, 0.75)


@dataclass(frozen=True)
class TileProjection:
    scenes: int
    land_fraction: float
    coarse_pass_gb: float
    native_pass_gb: float
    offsets_hours_1vm: float
    composite_hours_1vm: float
    minutes_per_tile_1vm: float
    vm_hours_per_tile: float
    n_vms_offsets: float
    n_vms_composite: float
    meets_60min_single_vm: bool
    cost_on_demand_usd: float
    cost_spot_usd_range: tuple[float, float]

    def summary_lines(self) -> list[str]:
        return [
            f"projected from probe rates of {PROBE_AS_OF} "
            f"(offsets {R_OFFSETS_MB_S:.0f} MB/s, composite "
            f"{R_COMPOSITE_MB_S:.0f} MB/s) -- projection, not measurement",
            f"single VM: {self.minutes_per_tile_1vm:.0f} min/tile "
            f"({self.offsets_hours_1vm:.1f} h offsets + "
            f"{self.composite_hours_1vm:.1f} h composite)",
            f"work: {self.vm_hours_per_tile:.1f} VM-hours/tile "
            f"(${self.cost_on_demand_usd:.2f} on-demand, "
            f"${self.cost_spot_usd_range[0]:.2f}-"
            f"${self.cost_spot_usd_range[1]:.2f} spot)",
            f"60-min fleet: {self.n_vms_offsets:.0f} offset VMs "
            f"({OFFSET_BUDGET_MIN:.0f} min) + {self.n_vms_composite:.0f} "
            f"composite VMs ({COMPOSITE_BUDGET_MIN:.0f} min)",
            "60 min on one VM: "
            + ("YES" if self.meets_60min_single_vm else "NO -- sharding required"),
        ]


def tile_projection(
    scenes: int = 2930,
    land_fraction: float = 1.0,
    *,
    r_offsets_mb_s: float = R_OFFSETS_MB_S,
    r_composite_mb_s: float = R_COMPOSITE_MB_S,
) -> TileProjection:
    """Project one tile's wall time, work, and cost from measured rates.

    Args:
        scenes: Scenes in the window. N40W075 measured 2,912 for 2021-2025;
            scene density grows toward the poles with WRS-2 path overlap.
        land_fraction: Fraction of the tile footprint that is land. Discounts
            only the phase-A coarse read (skipped blocks); phase B and the
            native pass read the full footprint.
        r_offsets_mb_s: Decoded per-VM rate for the coarse pass.
        r_composite_mb_s: Decoded per-VM rate for the native pass.
    """
    factor = settings.destripe_offset_resolution_factor
    native_edge = round(5 * settings.pixels_per_degree)  # 18,000 on a 5-deg tile
    coarse = (native_edge // factor) ** 2 * scenes * 2 * 2
    native = native_edge**2 * scenes * 2 * 2

    offsets_bytes = coarse * (land_fraction + 1.0)  # phase A (land only) + phase B
    off_h = offsets_bytes / (r_offsets_mb_s * 1e6) / 3600
    comp_h = native / (r_composite_mb_s * 1e6) / 3600
    vm_hours = off_h + comp_h

    n_off = off_h * 60 / OFFSET_BUDGET_MIN
    n_comp = comp_h * 60 / COMPOSITE_BUDGET_MIN

    return TileProjection(
        scenes=scenes,
        land_fraction=round(land_fraction, 3),
        coarse_pass_gb=round(coarse / 1e9, 1),
        native_pass_gb=round(native / 1e9, 1),
        offsets_hours_1vm=round(off_h, 2),
        composite_hours_1vm=round(comp_h, 2),
        minutes_per_tile_1vm=round((off_h + comp_h) * 60, 0),
        vm_hours_per_tile=round(vm_hours, 2),
        n_vms_offsets=round(n_off, 1),
        n_vms_composite=round(n_comp, 1),
        meets_60min_single_vm=(off_h + comp_h) <= 1.0,
        cost_on_demand_usd=round(vm_hours * VM_HOURLY_ON_DEMAND, 2),
        cost_spot_usd_range=(
            round(vm_hours * VM_HOURLY_ON_DEMAND * SPOT_FACTOR_RANGE[0], 2),
            round(vm_hours * VM_HOURLY_ON_DEMAND * SPOT_FACTOR_RANGE[1], 2),
        ),
    )
