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

**Aggregating to nominal ~100 m does not reduce the read.** Every delivered
cell is reduced from nine source cells, and those nine still have to be fetched
and decoded, so ``native_pass_gb`` is unchanged by ADR-017. What falls is the
output: nine times fewer pixels to percentile, to hold, to encode, and to
publish.

**``R_COMPOSITE_MB_S`` must be re-measured, and these projections are stale
until it is.** It is not an I/O rate. It is an *end-to-end composite* rate from
the packing probe, so the percentile is inside the number, and ADR-017 changed
the percentile. Dividing unchanged bytes by it and reading off a wall time
therefore assumes exactly what is in question. The direction is unknown here:
a microbenchmark puts the percentile kernel alone at 8.5x cheaper on the
delivered grid, but what share of this rate that kernel holds is unmeasured,
and the probe arm behind 45.5 MB/s ran on a 4.4%-land tile whose ocean nodata
deflated ~8x on the wire, which confounds any attempt to back it out.

**So do not quote an end-to-end speedup from this module.** Do not scale these
projections by the pixel-count ratio either. Only an acceptance run settles it.
"""

from __future__ import annotations

from dataclasses import dataclass

from landsat_lst.config import settings

#: Measured per-VM rates and their provenance. Update only from a probe run.
#: The frozen Stage-3 planning numbers, deliberately conservative:
#: - Offsets: v3 measured 158.4 at (8, 1024) warm, 120-134 at (16, 1024);
#:   140 planned.
#: - Composite: the m6i.4xlarge native probe (composite_rate_m6i4xl.json)
#:   measured 210-386 MB/s decoded at chunk 1024 -- but on a 4.4%-land tile
#:   whose ocean nodata deflates ~8x on the wire. A full-land tile
#:   compresses ~1.64x (U4), so 150 is the planning rate until the first
#:   sharded land-tile run calibrates it.
PROBE_AS_OF = "2026-08-21"
R_OFFSETS_MB_S = 140.0  # r6i.2xlarge, chunk 1024, 8 io threads; ladder v3
R_COMPOSITE_MB_S = 45.5  # m6i.4xlarge, native chunk 512; packing probe, real composite

#: Phase budgets inside the 60-minute tile (the rest is setup + export).
OFFSET_BUDGET_MIN = 15.0
COMPOSITE_BUDGET_MIN = 38.0

#: On-demand hourly per phase VM type; spot spans 0.30-0.75 of it
#: (pricing.json discipline: a range, never a scalar).
VM_HOURLY_ON_DEMAND = 0.504  # r6i.2xlarge -- offsets stage
VM_HOURLY_COMPOSITE = 0.768  # m6i.4xlarge -- composite stage
SPOT_FACTOR_RANGE = (0.30, 0.75)

#: vCPUs per configured VM type. Beside the hourly prices because they are the
#: same kind of fact about the same machines, and because Coiled bills credits
#: per *vCPU-hour* rather than per VM-hour -- so a fleet of 16-vCPU composite
#: VMs costs twice what the same count of 8-vCPU offsets VMs does for the same
#: wall clock. See :mod:`landsat_lst.quota`.
VM_VCPUS: dict[str, int] = {
    "r6i.xlarge": 4,
    "r6i.2xlarge": 8,
    "m6i.2xlarge": 8,
    "m6i.4xlarge": 16,
}


def vcpus(vm_type: str) -> int:
    """vCPUs for an EC2 instance type, from the table or from its name.

    The table covers what this project configures. The fallback parses AWS's
    own size grammar (``large`` = 2, ``xlarge`` = 4, ``Nxlarge`` = 4N) so a VM
    type someone sets tomorrow is priced rather than silently treated as one
    core -- which would under-estimate a fleet's credits by 8x and let an
    unaffordable run start.

    Returns:
        vCPU count, defaulting to the 8 of the primary type when the name
        parses as nothing recognizable.
    """
    known = VM_VCPUS.get(vm_type)
    if known is not None:
        return known

    size = vm_type.rsplit(".", 1)[-1]
    if size == "large":
        return 2
    if size == "xlarge":
        return 4
    if size.endswith("xlarge"):
        multiple = size[: -len("xlarge")]
        if multiple.isdigit():
            return 4 * int(multiple)
    return VM_VCPUS["r6i.2xlarge"]


@dataclass(frozen=True)
class TileProjection:
    scenes: int
    land_fraction: float
    coarse_pass_gb: float
    native_pass_gb: float
    #: Uncompressed delivered bytes for both products, before COG compression
    #: and file overhead. The one term ADR-017 moves.
    output_gb: float
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
            f"read: {self.native_pass_gb:.0f} GB source (unchanged by "
            f"aggregation) -> {self.output_gb:.2f} GB delivered, uncompressed",
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
    output_edge = round(5 * settings.output_pixels_per_degree)  # 6,000 delivered
    coarse = (native_edge // factor) ** 2 * scenes * 2 * 2
    native = native_edge**2 * scenes * 2 * 2
    # 2 B of uint16 LST plus 12 B of monthly uint8 counts, per delivered cell.
    output = output_edge**2 * 14

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
        output_gb=round(output / 1e9, 3),
        offsets_hours_1vm=round(off_h, 2),
        composite_hours_1vm=round(comp_h, 2),
        minutes_per_tile_1vm=round((off_h + comp_h) * 60, 0),
        vm_hours_per_tile=round(vm_hours, 2),
        n_vms_offsets=round(n_off, 1),
        n_vms_composite=round(n_comp, 1),
        meets_60min_single_vm=(off_h + comp_h) <= 1.0,
        # Each phase at its own VM price: offsets on r6i.2xlarge, composite
        # on m6i.4xlarge (the frozen Stage-3 fleet).
        cost_on_demand_usd=round(off_h * VM_HOURLY_ON_DEMAND + comp_h * VM_HOURLY_COMPOSITE, 2),
        cost_spot_usd_range=(
            round(
                (off_h * VM_HOURLY_ON_DEMAND + comp_h * VM_HOURLY_COMPOSITE) * SPOT_FACTOR_RANGE[0],
                2,
            ),
            round(
                (off_h * VM_HOURLY_ON_DEMAND + comp_h * VM_HOURLY_COMPOSITE) * SPOT_FACTOR_RANGE[1],
                2,
            ),
        ),
    )
