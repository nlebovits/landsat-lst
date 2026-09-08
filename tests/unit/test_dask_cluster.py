"""The cluster session, without a cluster.

``cluster_kwargs`` is pure and every argument a launch depends on is pinned
here: one VM type, one thread per worker, the shard spot policy, the
scheduler settings in the environment, no local STAC URL, shutdown on close.
The preflight order is identity, write access, credits, then the operator's
cap. The credit stop is best-effort and says so. Cleanup confirmation polls
a probe and records what it saw. Nothing here touches Coiled.
"""

from __future__ import annotations

import pytest

from landsat_lst import dask_cluster, quota
from landsat_lst.config import settings
from landsat_lst.dask_cluster import (
    _confirm_stopped,
    cluster_kwargs,
    credit_stop,
    futures_cluster_name,
    preflight,
)

pytestmark = pytest.mark.unit

ENV = {
    "AWS_ACCESS_KEY_ID": "AKIA",
    "AWS_SECRET_ACCESS_KEY": "s",
    "LST_STORAGE_BACKEND": "s3",
    "LST_DESTRIPE": "true",
}


class TestClusterKwargs:
    def test_one_vm_type_one_thread_and_the_shard_spot_policy(self):
        kwargs = cluster_kwargs(run_id="run", tile="S30W065", n_workers=3, environ=ENV)
        assert kwargs["worker_vm_types"] == [settings.shard_composite_vm_type]
        assert kwargs["worker_options"] == {"nthreads": 1}
        assert kwargs["spot_policy"] == settings.shard_spot_policy
        assert kwargs["n_workers"] == 3
        assert kwargs["shutdown_on_close"] is True
        assert kwargs["wait_for_workers"] is False
        assert kwargs["idle_timeout"] == settings.futures_idle_timeout
        assert kwargs["tags"]["executor"] == "futures"
        assert "software" not in kwargs, "package sync ships the local environment"

    def test_scheduler_settings_ride_in_the_environment(self):
        env = cluster_kwargs(run_id="run", tile="S30W065", n_workers=1, environ=ENV)["environ"]
        assert env["DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES"] == str(
            settings.futures_worker_losses
        )
        assert env["DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL"] == settings.futures_worker_ttl
        assert env["FRISKY_TRACING_CAPACITY"] == str(settings.frisky_tracing_capacity)
        assert env["LST_LOAD_CHUNK_SIZE"] == str(settings.shard_composite_chunk)
        assert env["LST_STORAGE_BACKEND"] == "s3"
        assert "LST_STAC_URL" not in env
        assert env["AWS_ACCESS_KEY_ID"] == "AKIA"

    def test_the_name_survives_truncation_with_its_marker(self):
        name = futures_cluster_name("shard-S30W065-2021-2025-20260908T120000Z", "S30W065")
        assert name.startswith("lst-") and name.endswith("-fut")
        assert len(name) <= 60
        assert futures_cluster_name("a", "T") != futures_cluster_name("b", "T")

    def test_a_vm_type_override_is_honoured(self):
        kwargs = cluster_kwargs(
            run_id="run", tile="S30W065", n_workers=1, environ=ENV, vm_type="r6i.4xlarge"
        )
        assert kwargs["worker_vm_types"] == ["r6i.4xlarge"]


class TestPreflight:
    def test_identity_then_write_access_then_credits_then_the_cap(self, monkeypatch):
        order: list[str] = []
        monkeypatch.setattr(quota, "preflight_identity", lambda **_k: order.append("identity"))
        monkeypatch.setattr(
            quota, "preflight_write_access", lambda **_k: order.append("write") or []
        )
        monkeypatch.setattr(quota, "estimate_run_credits", lambda *_a, **_k: 4.0)

        def credits(estimate, *, balance_source=None, **_k):
            order.append("credits")
            return quota.CreditBalance(remaining=100.0, source="test")

        monkeypatch.setattr(quota, "preflight_credits", credits)
        estimate, balance = preflight(credit_cap=100.0, scheduler="dask")
        assert order == ["identity", "write", "credits"]
        assert estimate == 4.0 and balance.remaining == 100.0

    def test_an_estimate_above_the_cap_is_refused_before_credits_are_read(self, monkeypatch):
        monkeypatch.setattr(quota, "preflight_identity", lambda **_k: "arn")
        monkeypatch.setattr(quota, "preflight_write_access", lambda **_k: [])
        monkeypatch.setattr(quota, "estimate_run_credits", lambda *_a, **_k: 40.0)
        monkeypatch.setattr(settings, "coiled_credit_safety", 2.0)

        def never(*_a, **_k):
            raise AssertionError("credits must not be read after the cap refused")

        monkeypatch.setattr(quota, "preflight_credits", never)
        with pytest.raises(quota.QuotaRefused, match="exceeds the run's cap"):
            preflight(credit_cap=15.0, scheduler="dask")

    def test_frisky_must_be_importable_locally_for_a_frisky_run(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "frisky", None)
        with pytest.raises(RuntimeError, match="frisky extra"):
            preflight(scheduler="frisky")


class TestCreditStop:
    def test_it_reports_a_drawdown_over_the_cap_and_nothing_else(self):
        spent = [10.0]

        def source():
            return quota.CreditBalance(remaining=None, source="test", spent=spent[0])

        stop = credit_stop(cap=5.0, balance_source=source)
        assert stop() is None
        spent[0] = 14.0
        assert stop() is None
        spent[0] = 16.0
        reason = stop()
        assert reason and "best-effort" in reason

    def test_an_unreadable_balance_never_stops_the_run(self):
        def broken():
            raise RuntimeError("billing down")

        stop = credit_stop(cap=5.0, balance_source=broken)
        assert stop() is None


class TestCleanupConfirmation:
    def test_a_stopped_report_confirms(self):
        answers = iter([("running", ""), ("stopping", ""), ("stopped", "")])
        confirmed, state = _confirm_stopped(
            42, timeout_s=5.0, probe=lambda _id: next(answers, ("stopped", "")), poll_s=0.0
        )
        assert confirmed is True and state == "stopped"

    def test_no_answer_within_the_timeout_is_unconfirmed(self):
        confirmed, state = _confirm_stopped(42, timeout_s=0.0, probe=lambda _id: None)
        assert confirmed is False and state is None

    def test_no_cluster_id_is_unconfirmed(self):
        assert _confirm_stopped(None, timeout_s=0.0) == (False, None)


def test_the_module_is_the_only_backend_importer():
    import ast
    from pathlib import Path

    source = Path(dask_cluster.__file__).read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert {"coiled", "frisky", "distributed"} <= imported
