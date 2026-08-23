"""Whether the workspace can afford this run, checked before anything starts.

On 2026-08-22 a healthy fleet was killed mid-stage with ``Scheduler Stopped ->
Instance Stopped: You have reached the workspace quota of 400 Coiled credits``,
and the same gate had already produced an empty ``ServerError`` on an earlier
cluster create. Both are the same fact -- the workspace was out of credits --
and in both cases the driver learned it the expensive way: once as a silent
kill misread as slow shards, once as an error with no message.

A quota is knowable *before* a submission, and this module makes the driver ask.
Two gates run here, in this order. **Identity first**: an AWS SSO session
expires within hours, which is less than a tile takes, and a driver whose
credentials are stale cannot write an artifact or read a barrier -- it used to
spend its whole startup before finding out. **Then credits**, from three
sources, best first, because the answer's quality varies and pretending
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
    from collections.abc import Callable, Iterable

    from landsat_lst.shards import TilePlan

log = structlog.get_logger()

#: Where a person checks a workspace's credit balance by eye.
TEAM_URL = "https://cloud.coiled.io/team?organization=nissim-lebovits"

#: Credits per **vCPU-hour**, which is Coiled's documented billing unit and
#: not the per-VM-hour guess this replaces.
#:
#: Calibrated against the S30W065 acceptance run of 2026-08-23, which billed
#: **268.11 credits** where the old model estimated 75 -- wrong by 3.6x, and
#: wrong in the direction that lets an unaffordable run start. Per cluster:
#:
#: =============  =======  ==========================  ==================
#: cluster        credits  fleet                       credits/vCPU-hour
#: =============  =======  ==========================  ==================
#: offse-r1        67.81   ~15 x r6i.2xlarge, ~31 min   ~1.09
#: offse-r2        16.15    14 x r6i.2xlarge, ~7 min    ~1.24
#: compo-r1       184.15    35 x m6i.4xlarge, 20-32 min ~0.62-0.99
#: =============  =======  ==========================  ==================
#:
#: The observed band is **0.6 to 1.25**, and the spread is staggered VM
#: lifetimes rather than a different rate: a fleet's VMs do not all boot or
#: finish together, so dividing one cluster's credits by its *nominal* wall
#: clock understates the rate for a fleet that finished early and overstates it
#: for one that straggled. 1.0 sits inside the band and prices the run's own
#: shape at ~318 credits against the 268.11 billed -- about 19% **high**, which
#: is the direction to be wrong in: over-estimating refuses a run that would
#: have fit, where under-estimating starts one that is killed mid-stage and
#: loses the whole tile. Both bounds are pinned by a regression test.
#:
#: Still an estimate. ``settings.coiled_credit_safety`` carries the band's
#: width, which is why its default is 2.0 rather than 1.5.
CREDITS_PER_VCPU_HOUR = 1.0


#: Botocore exception names that mean "your AWS session is gone", matched by
#: name rather than by class so this module never imports botocore to decide.
#: The SSO session expires within hours, which is shorter than a tile.
_EXPIRED_IDENTITY = (
    "UnauthorizedSSOTokenError",
    "NoCredentialsError",
    "TokenRetrievalError",
    "SSOTokenLoadError",
    "CredentialRetrievalError",
    "ProfileNotFound",
)

#: Error codes an STS call returns when the credentials exist but are stale.
_EXPIRED_CODES = ("ExpiredToken", "ExpiredTokenException", "InvalidClientTokenId")


class IdentityRefused(RuntimeError):
    """AWS credentials are missing or expired, so the run cannot start."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _sso_login_hint() -> str:
    """The exact command that fixes it, with the profile actually in play."""
    import os  # noqa: PLC0415

    profile = os.environ.get("AWS_PROFILE") or settings.aws_profile
    return f"aws sso login --profile {profile}" if profile else "aws sso login"


def _caller_identity() -> dict:
    """One STS call, with a short timeout and no retries.

    Short because this runs before anything else and a hung control plane must
    not become the reason a run did not start; no retries because an expired
    token does not un-expire.
    """
    import boto3  # noqa: PLC0415
    from botocore.config import Config  # noqa: PLC0415

    session = boto3.Session()
    client = session.client(
        "sts",
        config=Config(
            connect_timeout=5, read_timeout=5, retries={"max_attempts": 1, "mode": "standard"}
        ),
    )
    return client.get_caller_identity()


def preflight_identity(*, caller: Callable[[], dict] | None = None) -> str:
    """Verify AWS credentials before anything is submitted. Terminal on failure.

    This has bitten three times. The SSO session expires within hours -- less
    than a tile takes -- and the driver used to spend its whole startup (a STAC
    query, a plan, a fleet's boot) before discovering that nothing it wrote
    could reach S3. An expired token is not a transient failure and no backoff
    fixes it, so it fails now and names the command.

    Runs *before* the credit preflight: a session that cannot call STS cannot
    read a Coiled balance either, and "log in again" is a better message than
    "the balance could not be read".

    Args:
        caller: Where the identity comes from. Injectable so every refusal path
            is testable without a control plane.

    Returns:
        The caller ARN, for the driver to log.

    Raises:
        IdentityRefused: If credentials are missing, expired, or rejected.
    """
    probe = caller or _caller_identity
    try:
        identity = probe()
    except Exception as e:
        name = type(e).__name__
        code = ""
        response = getattr(e, "response", None)
        if isinstance(response, dict):
            code = str(response.get("Error", {}).get("Code", ""))

        text = f"{name}: {e}".strip().rstrip(":")
        expired = (
            name in _EXPIRED_IDENTITY
            or code in _EXPIRED_CODES
            or "expired" in text.lower()
            or "token has expired" in text.lower()
        )
        if expired or name.endswith(("Error", "Exception")):
            raise IdentityRefused(
                f"AWS credentials are not usable ({text}). The shard driver writes "
                f"every artifact to S3 and reads every barrier from it, so it cannot "
                f"start. Run: {_sso_login_hint()}"
            ) from e
        raise

    arn = str(identity.get("Arn", "")) if isinstance(identity, dict) else ""
    log.info("identity_preflight_ok", arn=arn or "unknown")
    return arn


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


def credits_for_fleets(fleets: Iterable[tuple[float, int, float]]) -> float:
    """Credits for ``(fleet_size, vcpus_each, wall_hours)`` triples.

    The whole model, in one line, so a regression test can hand it the shape of
    a run that has actually been billed and check the arithmetic against the
    invoice rather than against itself.
    """
    return sum(size * cpus * hours for size, cpus, hours in fleets) * CREDITS_PER_VCPU_HOUR


def run_fleets(
    plan: TilePlan | None = None, *, units: int | None = None
) -> list[tuple[float, int, float]]:
    """Each stage as ``(fleet_size, vcpus_each, wall_hours)``.

    Wall hours are *per VM*, because that is what a fleet bills: every VM in a
    stage pays its own boot, and then its share of the stage's work. Reading
    the stage's total wall clock instead would price a fleet of thirty-five as
    if it were one machine.
    """
    from landsat_lst.projection import vcpus  # noqa: PLC0415

    offsets_cpus = vcpus(settings.coiled_vm_types[0])
    composite_cpus = vcpus(settings.shard_composite_vm_type)

    if plan is None:
        # The preflight runs before shard 0 has written a plan, so the shape
        # comes from the pre-plan estimator this project already trusts.
        from landsat_lst import budgets  # noqa: PLC0415
        from landsat_lst.projection import tile_projection  # noqa: PLC0415

        projected = tile_projection()
        offsets_vms = max(1.0, round(projected.n_vms_offsets))
        composite_vms = max(1.0, round(projected.n_vms_composite))
        boot_h = budgets.VM_BOOT_S / 3600.0
        return [
            (
                offsets_vms,
                offsets_cpus,
                boot_h + budgets.RESOLVE_S / 3600.0 + projected.offsets_hours_1vm / offsets_vms,
            ),
            (
                composite_vms,
                composite_cpus,
                boot_h + projected.composite_hours_1vm / composite_vms,
            ),
        ]

    from landsat_lst import budgets  # noqa: PLC0415
    from landsat_lst.shards import offsets_fleet_units  # noqa: PLC0415

    offsets_vms = float(units or offsets_fleet_units())
    return [
        (offsets_vms, offsets_cpus, budgets.stage_budget("offsets", plan).work_s / 3600.0),
        (
            float(len(plan.bands)),
            composite_cpus,
            budgets.stage_budget("composite", plan).work_s / 3600.0,
        ),
        (1.0, composite_cpus, budgets.stage_budget("export", plan).work_s / 3600.0),
    ]


def estimate_run_credits(plan: TilePlan | None = None, *, units: int | None = None) -> float:
    """What this tile is expected to cost, in credits.

    Built from the same budget model the deadlines come from, so a run whose
    geometry grows costs more here without anyone editing a number -- and
    priced per **vCPU-hour**, which is what Coiled bills. The per-VM-hour
    model this replaces was wrong by 3.6x on the first run that could check it,
    because it could not see that a 16-vCPU composite VM costs twice an 8-vCPU
    offsets VM for the same wall clock.

    Raw: the safety factor lives in :func:`preflight_credits`, so this number
    stays comparable to an invoice.
    """
    return credits_for_fleets(run_fleets(plan, units=units))


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
