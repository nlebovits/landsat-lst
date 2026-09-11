"""Wait on a run without asking the process table what is running.

``pgrep -f`` matched the watching shell's own command line twice in one
session: once on ``sleep 150``, once on ``landsat-lst shard process``. Both
times the loop watched itself and reported a finished run as still going, and
the second time a failed run went unnoticed for seventeen minutes.

A pattern over the process table is the wrong instrument. It matches on text
that includes the watcher, it cannot tell a driver from a shell that mentions
one, and it says nothing about whether the work landed. So this module offers
two signals and no pattern matching at all:

* an explicitly captured PID, checked with :func:`os.kill` signal 0;
* a durable artifact count under an S3 or local prefix.

Nothing here shells out, and nothing accepts a command-line pattern.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class SelfWatchError(ValueError):
    """Raised when a watcher is asked to watch the process doing the watching."""


def pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a live process this user can signal.

    Signal 0 performs the permission and existence checks without delivering
    anything. ``PermissionError`` means the process exists and belongs to
    somebody else, which is still alive.

    One caveat, and it is why this is documented rather than discovered: a
    process this one spawned stays a zombie until it is reaped, and signal 0
    succeeds against a zombie. Watching your own child therefore needs the
    child reaped concurrently. Watching a driver you did not spawn -- the case
    this exists for -- has no such problem.
    """
    if pid == os.getpid():
        msg = (
            f"refusing to watch pid {pid}: that is this process. A watcher that "
            "can match itself never terminates, which is the defect this module "
            "exists to prevent."
        )
        raise SelfWatchError(msg)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid(
    pid: int,
    *,
    timeout_s: float | None = None,
    poll_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Block until ``pid`` exits. True if it exited, False on timeout."""
    start = clock()
    while pid_alive(pid):
        if timeout_s is not None and clock() - start >= timeout_s:
            return False
        sleep(poll_s)
    return True


def wait_for_artifacts(
    count: Callable[[], int],
    *,
    expected: int,
    timeout_s: float | None = None,
    poll_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Block until ``count()`` reaches ``expected``. Returns the last count.

    Completion is bytes in the bucket, never an exit code -- the same rule tile
    completion already follows. ``count`` is injected so a test needs no
    network and no clock.
    """
    start = clock()
    seen = count()
    while seen < expected:
        if timeout_s is not None and clock() - start >= timeout_s:
            return seen
        sleep(poll_s)
        seen = count()
    return seen
