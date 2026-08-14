"""What a tile run cost, priced from published rates and a measured duration.

``reconcile`` knows how long every tile ran and which VM ran it. That is one
multiplication away from a dollar figure, and the dollar figure is the number a
reader wants when deciding whether the fleet of :data:`FLEET_TILES` tiles is
affordable. This module does the multiplication and, more importantly, says how
much of the answer it made up.

Rates live in ``pricing.json`` rather than in code, so a price change is an edit
to data. Three things make a cost an interval rather than a number, and each one
is labelled rather than hidden.

**Spot is a band.** Quotes sampled on one day in one region across one instance
generation ran from 0.35 of on-demand for an ``m6i.4xlarge`` to 0.71 for an
``r6i.2xlarge``. A single factor would be wrong by more than 2x for one of the
two VM types the fleet actually schedules. Spot is priced per instance type, per
availability zone, per minute, and no artifact this pipeline writes records what
a finished run paid. So a spot cost spans 0.30 to 0.75 of on-demand and carries
:attr:`~landsat_lst.provenance.Provenance.ASSUMED`.

**An unreported lifecycle is a wider band.** Under the default
``spot_with_fallback`` policy the tile ran on one of the two and nothing says
which, so its cost spans 0.30 to 1.0 of on-demand. That interval is the complete
honest answer and any point inside it is invented. It is also the argument for
reading the instance metadata service during a run. Measuring the lifecycle
narrows a 3.3x interval to 2.5x for spot, or to a point for on-demand.

**Short runs are not cheap runs.** EC2 Linux bills per second after a 60 second
minimum, so :func:`billed_seconds` floors every duration. The failure that
motivated this module ran 10.375 seconds and still billed a full minute.

Nothing here interpolates. An instance type the table lacks falls back to the
first entry of ``settings.coiled_vm_types`` and says so, and a region the table
lacks returns ``None``. Borrowing ``us-west-2`` rates for another region would
manufacture a price, which is the refusal
:func:`landsat_lst.calibration.throughput_for` already makes for measurements.
See issue #92 item B1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from landsat_lst.config import settings
from landsat_lst.provenance import Provenance, load_committed_json
from landsat_lst.tiling import LAND_TILES

if TYPE_CHECKING:
    from collections.abc import Iterable

log = structlog.get_logger()

PRICING_PATH = Path(__file__).with_name("pricing.json")

SECONDS_PER_HOUR = 3600.0

#: Tiles in a full production build. Read from the land tile set rather than
#: written down, so a grid change moves the fleet estimate with it.
FLEET_TILES = len(LAND_TILES)

#: What a printed cost leaves out. Print it under any figure this module
#: produces, because the omissions are structural rather than rounding.
DISCLAIMER = (
    "Estimate covers VM wall clock at a list rate only. It omits Coiled's own "
    "fee, S3 request and storage charges, and the provisioning time a VM bills "
    "before its tile starts."
)

#: Used when the table cannot be read at all. See ``pricing.json`` for the
#: samples behind the band and the vendor rule behind the minimum.
_FALLBACK_SPOT_BAND = (0.30, 0.75)
_FALLBACK_MINIMUM_S = 60.0

#: Weakest wins when a figure combines two provenances.
_STRENGTH = {
    Provenance.MEASURED: 0,
    Provenance.PUBLISHED: 1,
    Provenance.DERIVED: 2,
    Provenance.ASSUMED: 3,
}


class Lifecycle(StrEnum):
    """How a VM was purchased, as far as anything published can tell.

    ``UNKNOWN`` is a real answer rather than a missing one. It is what a
    ``spot_with_fallback`` policy leaves behind when the VM itself never
    reported, and it prices to the interval between the other two.
    """

    SPOT = "spot"
    ON_DEMAND = "on-demand"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CostRange:
    """A cost or a rate, as an interval. ``low == high`` when the price is a point."""

    low: float
    high: float
    provenance: Provenance

    @property
    def is_point(self) -> bool:
        """Whether the interval collapsed to a single price."""
        return self.low == self.high


@dataclass(frozen=True)
class InstancePrice:
    """One row of ``pricing.json``, for one instance type in one region."""

    instance_type: str
    region: str
    on_demand_usd_hour: float
    memory_gib: float
    vcpu: int


@dataclass(frozen=True)
class CostEstimate:
    """What one tile cost, and how much of that figure is an assumption."""

    usd: CostRange
    usd_hour: CostRange
    duration_s: float
    billed_s: float
    instance_type: str
    region: str
    lifecycle: Lifecycle
    provenance: Provenance


@dataclass(frozen=True)
class FleetCost:
    """A tile count priced from the tiles that have already run."""

    usd: CostRange
    mean_usd_per_tile: CostRange
    tiles: int
    observed_tiles: int
    provenance: Provenance


def _row(instance_type: str, region: str) -> InstancePrice | None:
    """The table row for one instance type in one region, if it carries one."""
    row = load_committed_json(PRICING_PATH).get("regions", {}).get(region, {}).get(instance_type)
    if row is None:
        return None
    return InstancePrice(
        instance_type=instance_type,
        region=region,
        on_demand_usd_hour=float(row["on_demand_usd_hour"]),
        memory_gib=float(row["memory_gib"]),
        vcpu=int(row["vcpu"]),
    )


def _spot_band() -> tuple[float, float]:
    """Spot rate as a fraction of on-demand, low and high."""
    entry = load_committed_json(PRICING_PATH).get("spot_band", {})
    low, high = _FALLBACK_SPOT_BAND
    return float(entry.get("low_fraction", low)), float(entry.get("high_fraction", high))


def _weakest(provenances: Iterable[Provenance]) -> Provenance:
    """The least trustworthy of several provenances."""
    return max(provenances, key=lambda p: _STRENGTH[p])


def billed_seconds(duration_s: float) -> float:
    """Seconds a run of this length is billed for.

    EC2 Linux bills per second after a per-launch minimum, so a tile that dies
    in 10.375 seconds still costs a full minute. Rounding that down to the
    wall clock would make a crashloop look free, which is the opposite of what
    a cost report is for.
    """
    minimum = float(
        load_committed_json(PRICING_PATH)
        .get("billing", {})
        .get("minimum_seconds", _FALLBACK_MINIMUM_S)
    )
    return max(float(duration_s), minimum)


def lifecycle_for_policy(policy: str) -> Lifecycle:
    """The lifecycle a purchase policy pins down, or ``UNKNOWN`` if it does not.

    ``spot_with_fallback`` names two outcomes and picks between them at launch
    time, so it maps to ``UNKNOWN`` rather than to ``SPOT``. Also used to read
    what a VM reported, since a reported lifecycle uses the same two names.
    """
    try:
        return Lifecycle(policy)
    except ValueError:
        return Lifecycle.UNKNOWN


def _resolve_lifecycle(lifecycle: str | None) -> Lifecycle:
    """What the VM said, or what the fleet policy implies when it said nothing."""
    reported = lifecycle_for_policy(lifecycle or "")
    if reported is not Lifecycle.UNKNOWN:
        return reported
    return lifecycle_for_policy(settings.coiled_spot_policy)


def _resolve_price(instance_type: str, region: str) -> tuple[InstancePrice, Provenance] | None:
    """Find a rate for this instance type, substitute one, or refuse.

    No interpolation between instance sizes and none between regions. A rate
    two thirds of the way between an ``r6i.2xlarge`` and an ``r6i.4xlarge``
    would read as a price while being an invention. Within a priced region an
    unknown type falls back to the VM type the fleet asks for first, which is
    the one an unrecorded tile most likely ran on. An unpriced region returns
    nothing at all.
    """
    if region not in load_committed_json(PRICING_PATH).get("regions", {}):
        log.warning("pricing_region_unknown", region=region)
        return None

    exact = _row(instance_type, region)
    if exact is not None:
        return exact, Provenance.PUBLISHED

    fallback_type = settings.coiled_vm_types[0]
    fallback = _row(fallback_type, region)
    if fallback is None:  # pragma: no cover - the shipped table carries this type
        return None
    log.warning(
        "pricing_instance_unknown",
        instance_type=instance_type,
        region=region,
        substituted=fallback_type,
    )
    return fallback, Provenance.ASSUMED


def _rate_range(on_demand_usd_hour: float, lifecycle: Lifecycle) -> CostRange:
    """The hourly rate this lifecycle implies, as a point or as a band."""
    if lifecycle is Lifecycle.ON_DEMAND:
        return CostRange(on_demand_usd_hour, on_demand_usd_hour, Provenance.PUBLISHED)

    low, high = _spot_band()
    if lifecycle is Lifecycle.SPOT:
        return CostRange(on_demand_usd_hour * low, on_demand_usd_hour * high, Provenance.ASSUMED)
    return CostRange(on_demand_usd_hour * low, on_demand_usd_hour, Provenance.ASSUMED)


def tile_cost(
    *,
    duration_s: float,
    instance_type: str,
    lifecycle: str | None,
    region: str | None = None,
) -> CostEstimate | None:
    """Price one tile's billed time, or ``None`` if the region has no rates.

    ``lifecycle`` is what the VM reported, or ``None`` when it reported
    nothing. ``region`` defaults to ``settings.coiled_region``. The result is
    a point only for a known instance type on a known on-demand VM. Everything
    else is an interval, and reading a single number out of it puts back the
    precision the inputs never had.
    """
    resolved_region = region or settings.coiled_region
    found = _resolve_price(instance_type, resolved_region)
    if found is None:
        return None

    price, price_provenance = found
    resolved_lifecycle = _resolve_lifecycle(lifecycle)
    rate = _rate_range(price.on_demand_usd_hour, resolved_lifecycle)
    billed = billed_seconds(duration_s)
    hours = billed / SECONDS_PER_HOUR

    exact = price_provenance is Provenance.PUBLISHED and rate.is_point
    provenance = Provenance.DERIVED if exact else Provenance.ASSUMED
    return CostEstimate(
        usd=CostRange(rate.low * hours, rate.high * hours, provenance),
        usd_hour=rate,
        duration_s=duration_s,
        billed_s=billed,
        instance_type=price.instance_type,
        region=price.region,
        lifecycle=resolved_lifecycle,
        provenance=provenance,
    )


def fleet_cost(estimates: Iterable[CostEstimate], *, tiles: int = FLEET_TILES) -> FleetCost | None:
    """Extrapolate observed tile costs to a fleet, or ``None`` with nothing to go on.

    The mean of what ran, multiplied out. Tiles differ by scene count, so a
    handful of observations carries a wide spread on top of the spot band, and
    any extrapolation past the observed count is assumed. Zero observations
    returns nothing rather than zero dollars, because a total built from no
    tiles is a guess wearing a total's clothes.
    """
    observed = list(estimates)
    if not observed:
        return None

    count = len(observed)
    low = sum(e.usd.low for e in observed) / count
    high = sum(e.usd.high for e in observed) / count
    provenance = _weakest(e.usd.provenance for e in observed)
    if count < tiles:
        provenance = Provenance.ASSUMED

    return FleetCost(
        usd=CostRange(low * tiles, high * tiles, provenance),
        mean_usd_per_tile=CostRange(low, high, provenance),
        tiles=tiles,
        observed_tiles=count,
        provenance=provenance,
    )


def instance_memory_gib(instance_type: str, region: str | None = None) -> float | None:
    """Memory this instance type carries, or ``None`` if the table lacks it.

    For a caller comparing a tile's peak RSS against the VM it ran on. Returns
    ``None`` rather than a substitute, because headroom measured against the
    wrong machine is worse than no headroom figure. Both VM types the fleet
    schedules today carry 64 GiB, which is why ``profiling.DEFAULT_VM_GIB`` can
    be a single number.
    """
    price = _row(instance_type, region or settings.coiled_region)
    return None if price is None else price.memory_gib
