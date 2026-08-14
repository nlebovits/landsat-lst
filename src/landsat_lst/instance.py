"""What machine a tile actually ran on, asked of the machine itself.

Pricing a run needs the instance type, and ``settings.coiled_vm_types`` cannot
supply it. That list is a preference, not a record: Coiled falls back to the
second entry when the first is unavailable, and ``m6i.4xlarge`` costs 1.52x
``r6i.2xlarge`` for the same 64 GiB. ``settings.coiled_spot_policy`` is worse
still, because ``spot_with_fallback`` is a policy whose whole point is that it
does not decide in advance.

EC2 answers both questions about itself through the instance metadata service,
so a tile reads them once and publishes them in its own state object. Measuring
the lifecycle narrows an estimate that would otherwise span 0.30 to 1.00 of the
on-demand rate down to a band, or to a point.

Every read here is best-effort and bounded. Off EC2 the first request fails
immediately and the caller gets the configured assumption instead, labelled as
one.
"""

from __future__ import annotations

import http.client
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache

import structlog

from landsat_lst.config import settings
from landsat_lst.pricing import Lifecycle, lifecycle_for_policy
from landsat_lst.provenance import Provenance

log = structlog.get_logger()

# Named for what they are rather than for the credential they carry, so
# bandit's hardcoded-credential heuristic does not read a URL path as a secret.
_IMDS_HOST = "169.254.169.254"
_SESSION_PATH = "/latest/api/token"
_TYPE_PATH = "/latest/meta-data/instance-type"
_LIFECYCLE_PATH = "/latest/meta-data/instance-life-cycle"
_SESSION_TTL_S = "21600"

#: Sub-second on purpose. This runs once, on the tile's critical path, and a
#: machine that is not an EC2 instance has to pay almost nothing to establish
#: that. Instrumentation never fails a tile, and it must not stall one either.
IMDS_TIMEOUT_S = 0.3


@dataclass(frozen=True)
class InstanceIdentity:
    """The machine a tile ran on, and how much to trust that answer."""

    instance_type: str
    lifecycle: Lifecycle
    provenance: Provenance
    #: ``"imds"`` when the machine answered, ``"settings"`` when it did not.
    source: str


def _request(path: str, *, token: str | None = None, method: str = "GET") -> str | None:
    """One metadata request, or ``None`` if anything at all goes wrong.

    Uses ``http.client`` rather than ``urllib.request`` because the metadata
    service is plain HTTP on a link-local address by design. Passing a
    hardcoded ``http://`` URL to ``urlopen`` is a bandit finding for a scheme
    check that cannot apply here, and the connection needs its own timeout.
    """
    conn = None
    try:
        conn = http.client.HTTPConnection(_IMDS_HOST, timeout=IMDS_TIMEOUT_S)
        headers = {}
        if method == "PUT":
            headers["X-aws-ec2-metadata-token-ttl-seconds"] = _SESSION_TTL_S
        elif token:
            headers["X-aws-ec2-metadata-token"] = token
        conn.request(method, path, headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            return None
        return response.read().decode().strip()
    except Exception:
        return None
    finally:
        if conn is not None:
            with suppress(Exception):
                conn.close()


def _probe() -> InstanceIdentity | None:
    """Ask IMDSv2 for the instance type and purchase mode.

    IMDSv2 takes a token first, then presents it on each read. A token failure
    returns ``None`` rather than falling back to IMDSv1, because a host that
    refuses the token is either not EC2 or has IMDSv1 disabled, and guessing
    would cost another timeout for the same answer.
    """
    token = _request(_SESSION_PATH, method="PUT")
    if not token:
        return None

    instance_type = _request(_TYPE_PATH, token=token)
    if not instance_type:
        return None

    raw = _request(_LIFECYCLE_PATH, token=token)
    lifecycle = Lifecycle.UNKNOWN
    if raw in {Lifecycle.SPOT.value, Lifecycle.ON_DEMAND.value}:
        lifecycle = Lifecycle(raw)

    return InstanceIdentity(
        instance_type=instance_type,
        lifecycle=lifecycle,
        provenance=Provenance.MEASURED,
        source="imds",
    )


def _assumed() -> InstanceIdentity:
    """What the run was configured to ask for, when the machine will not say.

    Cannot raise. This is reached from the heartbeat payload, and a tile must
    not die because it could not name the machine it is running on.
    """
    configured = settings.coiled_vm_types
    return InstanceIdentity(
        instance_type=configured[0] if configured else "unknown",
        lifecycle=lifecycle_for_policy(settings.coiled_spot_policy),
        provenance=Provenance.ASSUMED,
        source="settings",
    )


@lru_cache(maxsize=1)
def instance_identity() -> InstanceIdentity:
    """This machine's instance type and purchase mode. Never raises.

    Cached for the life of the process. Neither answer changes under a running
    tile, and a tile beating every 60 seconds must not probe every 60 seconds.
    """
    probed = _probe()
    if probed is not None:
        log.info(
            "instance_identified",
            instance_type=probed.instance_type,
            lifecycle=probed.lifecycle.value,
        )
        return probed
    return _assumed()
