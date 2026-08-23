"""Whether the workspace can afford this run, checked before anything starts.

On 2026-08-22 a healthy fleet was killed mid-stage with ``Scheduler Stopped ->
Instance Stopped: You have reached the workspace quota of 400 Coiled credits``,
and the same gate had already produced an empty ``ServerError`` on an earlier
cluster create. Both are the same fact -- the workspace was out of credits --
and in both cases the driver learned it the expensive way: once as a silent
kill misread as slow shards, once as an error with no message.

A quota is knowable *before* a submission, and this module makes the driver ask.
Three sources, best first, because the answer's quality varies and pretending
otherwise is how a preflight becomes a rubber stamp:

1. **The workspace usage endpoint.** ``/api/v2/user/account/{workspace}/usage``
   is what ``coiled login`` itself reads to print "You have reached your quota
   of Coiled credits for this account"; ``coiled.utils.has_program_quota``
   checks its ``has_quota`` flag. That flag is authoritative for *exhausted or
   not*. If the payload also carries a remaining-credit figure, it is used.
2. **Quota minus billing activity.** ``coiled.get_billing_activity`` pages
   per-event debits (``amount_credits``, e.g. ``"-2.7823"``) but exposes no
   balance, so a remaining figure has to be reconstructed:
   ``coiled_credit_quota`` minus the debits since the period began.
3. **A human.** If neither source answers, the driver refuses and prints the
   team page rather than guessing.

**The approximation in source 2, stated plainly.** Nothing in the billing data
says when the quota period resets, so "since the period began" is taken as the
last N days (``coiled_credit_period_days``). If the real period is a calendar
month and the run happens on the 3rd, this over-counts debits and refuses a run
that would have fit -- conservative, and the direction to be wrong in. If the
real period is longer than the window, it under-counts and the preflight passes
a run that will be killed. Source 1's ``has_quota`` flag is what catches that
case, which is why source 2 is never used alone when source 1 answered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from landsat_lst.config import settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from landsat_lst.shards import TilePlan

log = structlog.get_logger()

#: Where a person checks a workspace's credit balance by eye.
TEAM_URL = "https://cloud.coiled.io/team?organization=nissim-lebovits"

#: Credits per VM-hour. Calibrated from the observed billing events, whose
#: ``amount_credits`` cluster around ``-2.7823`` for the instance types this
#: project runs. **The uncertainty is real and it is in the denominator**: the
#: event stream does not say what interval an event covers, so this assumes one
#: event is roughly one VM-hour. If an event is really a whole cluster
#: lifetime, this over-estimates a run's cost and the preflight is
#: conservative; if an event covers less than an hour, it under-estimates.
#: Treat a preflight pass as "not obviously unaffordable", never as a promise.
CREDITS_PER_VM_HOUR = 2.7823


class QuotaRefused(RuntimeError):
    """The run was not started, because it looks unaffordable or unknowable."""

    def __init__(self, reason: str, *, estimate: float, remaining: float | None) -> None:
        self.reason = reason
        self.estimate = estimate
        self.remaining = remaining
        super().__init__(reason)


@dataclass(frozen=True)
class CreditBalance:
    """What is known about the workspace's remaining credits, and how well."""

    #: Credits believed to remain, or ``None`` when nothing could be read.
    remaining: float | None
    #: Which of the three sources answered.
    source: str
    #: The endpoint's own exhausted-or-not flag, when it answered at all. This
    #: is the only field that is authoritative rather than reconstructed.
    has_quota: bool | None = None
    detail: str = ""

    @property
    def known(self) -> bool:
        return self.remaining is not None or self.has_quota is not None


def _workspace() -> str:
    """The Coiled workspace this run bills to."""
    import dask.config  # noqa: PLC0415

    return (
        dask.config.get("coiled.workspace", None) or dask.config.get("coiled.account", None) or ""
    )


def _usage_endpoint_balance() -> CreditBalance | None:
    """Source 1: the endpoint ``coiled login`` reads to warn about the quota.

    Returns ``None`` -- not a zero balance -- when anything at all goes wrong.
    A preflight that cannot read the endpoint must fall through to the next
    source, never invent an answer from a failure.
    """
    try:
        import coiled  # noqa: PLC0415

        workspace = _workspace()
        if not workspace:
            return None
        with coiled.Cloud() as cloud:
            payload = cloud._sync(
                cloud._do_request_idempotent,
                "GET",
                f"{cloud.server}/api/v2/user/account/{workspace}/usage",
            )
            data: Any = payload
            if hasattr(payload, "json"):
                data = cloud._sync(payload.json)
    except Exception as e:
        log.info("quota_usage_endpoint_unavailable", error=str(e))
        return None

    if not isinstance(data, dict):
        return None
    return CreditBalance(
        remaining=_first_number(data, _REMAINING_KEYS),
        source="usage_endpoint",
        has_quota=data.get("has_quota") if isinstance(data.get("has_quota"), bool) else None,
        detail=f"workspace {_workspace()}",
    )


#: Candidate names for a remaining-credit figure in the usage payload. Several,
#: because the endpoint is undocumented and the shape is not pinned by anything
#: this project controls; a name that is absent simply does not answer.
_REMAINING_KEYS = (
    "credits_remaining",
    "remaining_credits",
    "coiled_credits_remaining",
    "credits_left",
)


def _first_number(payload: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _billing_balance() -> CreditBalance | None:
    """Source 2: the configured quota minus the debits in the recent window.

    See the module docstring for what "recent window" approximates and which
    way it is wrong.
    """
    try:
        import datetime as dt  # noqa: PLC0415

        import coiled  # noqa: PLC0415

        start = dt.datetime.now(tz=dt.UTC) - dt.timedelta(days=settings.coiled_credit_period_days)
        spent = 0.0
        page = 1
        while True:
            activity = coiled.get_billing_activity(start_time=start.isoformat(), page=page)
            events = activity.get("events") or activity.get("results") or []
            if not events:
                break
            for event in events:
                amount = event.get("amount_credits")
                if amount is None:
                    continue
                spent += abs(float(amount))
            if not activity.get("next") or page >= settings.coiled_billing_max_pages:
                break
            page += 1
    except Exception as e:
        log.info("quota_billing_unavailable", error=str(e))
        return None

    remaining = settings.coiled_credit_quota - spent
    return CreditBalance(
        remaining=remaining,
        source="billing_activity",
        detail=(
            f"{spent:.1f} of {settings.coiled_credit_quota:.0f} credits spent in the "
            f"last {settings.coiled_credit_period_days} days (period reset not observable)"
        ),
    )


def read_balance() -> CreditBalance:
    """Ask each source in turn, and say which one answered."""
    for source in (_usage_endpoint_balance, _billing_balance):
        balance = source()
        if balance is not None and balance.known:
            return balance
    return CreditBalance(remaining=None, source="unavailable")


def estimate_run_credits(plan: TilePlan | None = None, *, units: int | None = None) -> float:
    """What this tile is expected to cost, in credits.

    Built from the same budget model the deadlines come from, so a run whose
    geometry grows costs more here without anyone editing a number. Before the
    plan exists -- which is when the preflight runs -- it falls back to
    ``projection.tile_projection``'s VM-hours, the pre-plan estimator this
    project already trusts for cost.
    """
    if plan is None:
        from landsat_lst.projection import tile_projection  # noqa: PLC0415

        vm_hours = tile_projection().vm_hours_per_tile
    else:
        from landsat_lst import budgets  # noqa: PLC0415
        from landsat_lst.shards import offsets_fleet_units  # noqa: PLC0415

        offsets_vms = units or offsets_fleet_units()
        vm_hours = (
            budgets.stage_budget("offsets", plan).work_s * offsets_vms
            + budgets.stage_budget("composite", plan).work_s * len(plan.bands)
            + budgets.stage_budget("export", plan).work_s
        ) / 3600.0
    return vm_hours * CREDITS_PER_VM_HOUR


def preflight_credits(
    estimated_credits: float,
    *,
    balance_source: Callable[[], CreditBalance] | None = None,
    acknowledged: bool | None = None,
) -> CreditBalance:
    """Refuse to start a run the workspace cannot pay for. Scenario zero.

    Called before any submission, and before a run id is printed: a run id for
    a run that never started is a resume hint that leads nowhere.

    Args:
        estimated_credits: What the run is expected to cost, from
            :func:`estimate_run_credits`.
        balance_source: Where the balance comes from. Injectable so the whole
            decision is testable without a control plane -- and resolved at
            *call* time rather than bound as a default, so patching
            :func:`read_balance` actually takes effect. A default bound at
            definition let a unit test reach the real billing API.
        acknowledged: Whether a human has checked the balance by eye.
            ``None`` reads ``settings.ack_quota``.

    Returns:
        The balance that was read, for the caller to log.

    Raises:
        QuotaRefused: If the workspace is out of credits, if the estimate does
            not fit the remaining balance with its safety factor, or if nothing
            could be read and nobody acknowledged.
    """
    acked = settings.ack_quota if acknowledged is None else acknowledged
    balance = (balance_source or read_balance)()
    needed = estimated_credits * settings.coiled_credit_safety

    if balance.has_quota is False:
        raise QuotaRefused(
            "the Coiled workspace is out of credits (its usage endpoint reports "
            f"has_quota=false). This run needs about {estimated_credits:.0f}. "
            f"Check {TEAM_URL}.",
            estimate=estimated_credits,
            remaining=balance.remaining,
        )

    if balance.remaining is not None:
        if balance.remaining < needed:
            raise QuotaRefused(
                f"about {balance.remaining:.0f} Coiled credits remain "
                f"({balance.source}: {balance.detail}) and this run needs about "
                f"{estimated_credits:.0f} x {settings.coiled_credit_safety:.1f} = "
                f"{needed:.0f} -- short by {needed - balance.remaining:.0f}. "
                f"Check {TEAM_URL}.",
                estimate=estimated_credits,
                remaining=balance.remaining,
            )
        log.info(
            "quota_preflight_ok",
            source=balance.source,
            remaining=round(balance.remaining, 1),
            needed=round(needed, 1),
        )
        return balance

    if not acked:
        raise QuotaRefused(
            "could not read the Coiled credit balance, and a run that hits the "
            f"quota is killed mid-stage. This run needs about "
            f"{estimated_credits:.0f} credits. Check {TEAM_URL} and re-run with "
            "--ack-quota (or LST_ACK_QUOTA=1) to proceed on your own check.",
            estimate=estimated_credits,
            remaining=None,
        )

    log.warning(
        "quota_preflight_acknowledged",
        note="balance unreadable; proceeding on an operator's manual check",
        needed=round(needed, 1),
    )
    return balance
