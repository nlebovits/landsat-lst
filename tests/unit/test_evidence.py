from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from landsat_lst.cli import main

if TYPE_CHECKING:
    from pathlib import Path

from landsat_lst.evidence import _coiled_cluster, _safe, capture_frisky, collect_evidence


class StubStorage:
    def __init__(self, revision: str | None = "83957932f3e1a72484246c421cbab1d91d4ba234"):
        self.revision = revision

    def run_prefix(self, run_id: str) -> str:
        return f"_runs/{run_id}/"

    def list_prefix(self, prefix: str) -> dict[str, datetime]:
        modified = datetime(2026, 1, 1, tzinfo=UTC)
        if prefix == "_runs/run-1/":
            return {"_runs/run-1/N40W075.1.json": modified}
        if prefix == "_shards/run-1/":
            return {
                "_shards/run-1/N40W075/state/composite.0003.1.composite.profile.json": modified,
                "_shards/run-1/N40W075/composite/lst_p95/band003.tif": modified,
            }
        if prefix == "_shards/timings/run-1/":
            return {}
        raise AssertionError(prefix)

    def read_text(self, key: str) -> str:
        assert key.endswith(".json")
        return json.dumps(
            {
                "status": "completed",
                "access_token": "do-not-retain",
                "code_identity": {"revision": self.revision, "package_version": "0.1-test"},
            }
        )


def contract(tmp_path: Path) -> Path:
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"wall_s": 840}\n')
    equivalence = tmp_path / "equivalence.json"
    equivalence.write_text('{"passed": true}\n')
    payload = {
        "schema_version": 1,
        "claim": "Treatment reduces wall time.",
        "inputs": {
            "workload": "N40W075 composite band 1",
            "launch_command": "landsat-lst shard process --tile N40W075",
            "baseline_revision": "c84448bbac2c95af408ba523521b712b43ba58e8",
            "treatment_revision": "83957932f3e1a72484246c421cbab1d91d4ba234",
        },
        "baseline": {
            "metric": "wall_s",
            "value": 840,
            "unit": "seconds",
            "classification": "measured",
            "artifact": str(baseline),
        },
        "target_metric": "wall_s",
        "minimum_effect": {"direction": "decrease", "fraction": 0.1},
        "production_discriminator": "One production shard.",
        "stop_rule": "Stop below ten percent.",
        "max_cloud_cost_usd": 1,
        "max_coiled_credits": 20,
        "code_identity_required": True,
        "output_equivalence_required": True,
        "output_equivalence": {
            "method": "SHA-256",
            "acceptance_criterion": "both output checksums match",
            "result_artifact": str(equivalence),
        },
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload))
    return path


def test_collect_evidence_with_injected_storage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("landsat_lst.evidence.importlib.metadata.version", lambda _: "0.1-test")
    destination = collect_evidence(
        output_dir=tmp_path / "bundle",
        contract_path=contract(tmp_path),
        run_id="run-1",
        storage=StubStorage(),
    )
    bundle = json.loads(destination.read_text())
    assert bundle["run_id"] == "run-1"
    assert bundle["run_artifacts"][0]["content"]["access_token"] == "<redacted>"
    assert bundle["run_artifacts"][0]["sha256"]
    assert bundle["code"]["package_version"] == "0.1-test"
    keys = {artifact["key"] for artifact in bundle["run_artifacts"]}
    assert "_shards/run-1/N40W075/state/composite.0003.1.composite.profile.json" in keys
    assert not any(key.endswith(".tif") for key in keys)
    assert bundle["worker_code_verification"]["status"] == "verified"
    for artifact in bundle["contract_artifacts"].values():
        retained = destination.parent / artifact["path"]
        assert retained.is_file()
        assert artifact["sha256"]


def test_collect_evidence_rejects_duplicate_attachment_basenames(tmp_path: Path) -> None:
    left = tmp_path / "left" / "same.txt"
    right = tmp_path / "right" / "same.txt"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_text("left")
    right.write_text("right")

    with pytest.raises(ValueError, match="basenames must be unique"):
        collect_evidence(
            output_dir=tmp_path / "bundle",
            contract_path=contract(tmp_path),
            attachments=(left, right),
        )


def test_collect_evidence_rejects_worker_revision_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="worker code revision"):
        collect_evidence(
            output_dir=tmp_path / "bundle",
            contract_path=contract(tmp_path),
            run_id="run-1",
            storage=StubStorage(revision="c" * 40),
        )


def test_collect_evidence_rejects_missing_worker_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identity is unavailable"):
        collect_evidence(
            output_dir=tmp_path / "bundle",
            contract_path=contract(tmp_path),
            run_id="run-1",
            storage=StubStorage(revision=None),
        )


def test_collect_evidence_refuses_to_overwrite_bundle(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "evidence.json").write_text("original")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        collect_evidence(output_dir=output, contract_path=contract(tmp_path))


def test_collect_evidence_rejects_failed_output_equivalence(tmp_path: Path) -> None:
    contract_path = contract(tmp_path)
    (tmp_path / "equivalence.json").write_text('{"passed": false}\n')

    with pytest.raises(ValueError, match="passed=true"):
        collect_evidence(output_dir=tmp_path / "bundle", contract_path=contract_path)


def test_safe_redacts_secrets_embedded_in_log_text(monkeypatch) -> None:
    monkeypatch.setenv("UNRELATED_API_TOKEN", "literal-environment-secret")
    raw = (
        "AWS_SECRET_ACCESS_KEY=hunter2 TOKEN:abcde "
        "Authorization: Bearer bearer-value literal-environment-secret"
    )
    safe = _safe(raw)
    assert "hunter2" not in safe
    assert "abcde" not in safe
    assert "bearer-value" not in safe
    assert "literal-environment-secret" not in safe
    assert safe.count("<redacted>") >= 4


class Response:
    status = 200

    async def json(self):
        return {"series": [1, 2]}

    async def text(self):
        return ""


class FakeCloud:
    server = "https://cloud.example"

    def __init__(self, *, workspace: str):
        self.workspace = workspace

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def cluster_details(self, cluster_id: int, *, workspace: str):
        return {"name": "cluster-name", "cluster_id": cluster_id, "workspace": workspace}

    def cluster_logs(self, _cluster_id: int, *, workspace: str):
        return {"worker": f"workspace={workspace} AWS_SESSION_TOKEN=session-value"}

    async def _do_request(self, *_args, **_kwargs):
        return Response()

    def _sync(self, function, *args):
        return asyncio.run(function(*args))


def test_coiled_135_private_api_contract_and_log_redaction(monkeypatch) -> None:
    billing_calls = []

    def billing(**kwargs):
        billing_calls.append(kwargs)
        return {"next": None, "api_token": "billing-secret"}

    fake = SimpleNamespace(
        Cloud=FakeCloud,
        __version__="1.135.2",
        get_billing_activity=billing,
    )
    monkeypatch.setitem(sys.modules, "coiled", fake)
    result = _coiled_cluster(123, "workspace", ("cpu",))
    assert result["details"]["name"] == "cluster-name"
    assert result["metrics"]["cpu"] == {"series": [1, 2]}
    assert "session-value" not in result["logs"]["worker"]
    assert result["billing_pages"][0]["api_token"] == "<redacted>"
    assert billing_calls == [{"account": "workspace", "cluster": "cluster-name", "page": 1}]


def test_coiled_details_shape_has_an_operator_error(monkeypatch) -> None:
    class NamelessCloud(FakeCloud):
        def cluster_details(self, cluster_id: int, *, workspace: str):
            return {"id": cluster_id, "workspace": workspace}

    monkeypatch.setitem(
        sys.modules,
        "coiled",
        SimpleNamespace(Cloud=NamelessCloud, __version__="1.135.2"),
    )
    with pytest.raises(RuntimeError, match="did not contain a string name"):
        _coiled_cluster(123, "workspace", ())


def test_capture_frisky_turns_process_failure_into_cli_message(tmp_path: Path, monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(2, ["frisky"], stderr=b"dashboard unavailable")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="dashboard unavailable"):
        capture_frisky("https://frisky.example", tmp_path / "direct")

    result = CliRunner().invoke(
        main,
        ["evidence", "capture-frisky", "https://frisky.example", "--out", str(tmp_path / "cli")],
    )
    assert result.exit_code == 1
    assert "dashboard unavailable" in result.output
    assert "Traceback" not in result.output
