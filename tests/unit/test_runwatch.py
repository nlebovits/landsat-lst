"""The watcher must stop when the work stops, and must refuse to watch itself."""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest

from landsat_lst.runwatch import (
    SelfWatchError,
    pid_alive,
    wait_for_artifacts,
    wait_for_pid,
)

pytestmark = pytest.mark.unit


class TestPid:
    def test_it_terminates_when_the_real_process_exits(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.4)"])
        # Reap concurrently: a dead child is a zombie until waited on, and
        # signal 0 succeeds against a zombie.
        reaper = threading.Thread(target=proc.wait, daemon=True)
        reaper.start()
        try:
            assert wait_for_pid(proc.pid, timeout_s=30.0, poll_s=0.05) is True
            assert pid_alive(proc.pid) is False
        finally:
            reaper.join(timeout=10)

    def test_it_refuses_to_watch_itself(self):
        """The exact defect: a watcher matching its own command line never ends."""
        with pytest.raises(SelfWatchError, match="that is this process"):
            pid_alive(os.getpid())
        with pytest.raises(SelfWatchError):
            wait_for_pid(os.getpid(), timeout_s=0.1, poll_s=0.01)

    def test_a_pid_that_never_existed_reads_as_finished(self):
        # A watcher must not hang because it was given a stale identifier.
        assert wait_for_pid(2**22 - 1, timeout_s=1.0, poll_s=0.01) is True

    def test_it_gives_up_at_the_timeout_rather_than_hanging(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert wait_for_pid(proc.pid, timeout_s=0.2, poll_s=0.05) is False
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_it_takes_no_command_pattern(self):
        """There is no pattern argument, so it cannot match a shell command."""
        import inspect

        for fn in (pid_alive, wait_for_pid):
            names = set(inspect.signature(fn).parameters)
            assert not names & {"pattern", "cmdline", "name", "match"}


class TestArtifacts:
    def test_it_returns_when_the_artifacts_land(self):
        counts = iter([0, 3, 7, 12])
        seen = wait_for_artifacts(
            lambda: next(counts), expected=12, poll_s=0.0, sleep=lambda _s: None
        )
        assert seen == 12

    def test_it_gives_up_at_the_timeout(self):
        ticks = iter([0.0, 0.0, 99.0])
        seen = wait_for_artifacts(
            lambda: 1,
            expected=35,
            timeout_s=10.0,
            poll_s=0.0,
            sleep=lambda _s: None,
            clock=lambda: next(ticks),
        )
        assert seen == 1
