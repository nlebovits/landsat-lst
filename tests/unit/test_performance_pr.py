from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from scripts import check_performance_pr


def payload(body: str, *, title: str = "Ordinary change", login: str = "human") -> dict[str, Any]:
    return {
        "pull_request": {
            "title": title,
            "body": body,
            "user": {"login": login},
        }
    }


def declaration(stage: str, contract: str = "n/a", evidence: str = "n/a") -> str:
    return (
        "<!-- performance-evidence\n"
        f"stage: {stage}\n"
        f"contract: {contract}\n"
        f"evidence: {evidence}\n"
        "-->"
    )


def test_human_pr_requires_exactly_one_declaration() -> None:
    assert check_performance_pr.validate_pr(payload("")) == [
        "PR body must contain exactly one performance-evidence declaration"
    ]
    doubled = declaration("none") + "\n" + declaration("none")
    assert "exactly one" in check_performance_pr.validate_pr(payload(doubled))[0]


def test_bot_pr_is_exempt() -> None:
    assert check_performance_pr.validate_pr(payload("", login="dependabot[bot]")) == []


def test_performance_language_cannot_be_declared_none() -> None:
    errors = check_performance_pr.validate_pr(
        payload(declaration("none"), title="Optimize composite throughput")
    )
    assert errors == ["performance language requires a non-none performance-evidence stage"]


def test_governance_stage_requires_no_claim_artifacts() -> None:
    assert check_performance_pr.validate_pr(payload(declaration("governance"))) == []
    errors = check_performance_pr.validate_pr(
        payload(declaration("governance", contract="contract.json"))
    )
    assert errors == ["governance PR must declare contract: n/a"]


def test_instrumentation_requires_a_valid_contract(tmp_path: Path, monkeypatch) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text("{}")
    monkeypatch.setattr(check_performance_pr, "load_contract", lambda path: {"path": str(path)})

    assert (
        check_performance_pr.validate_pr(
            payload(declaration("instrumentation", contract="contract.json")),
            root=tmp_path,
        )
        == []
    )
    assert check_performance_pr.validate_pr(
        payload(declaration("instrumentation", contract="../outside.json")),
        root=tmp_path,
    )


def test_optimization_requires_a_proceed_bundle(tmp_path: Path, monkeypatch) -> None:
    contract_path = tmp_path / "contract.json"
    evidence_path = tmp_path / "evidence.json"
    contract_path.write_text("{}")
    evidence_path.write_text("{}")
    contract = {"claim": "measured", "result_artifact": "result.json"}
    result = {"decision": "proceed"}
    calls: list[bool] = []
    monkeypatch.setattr(check_performance_pr, "load_contract", lambda _path: contract)
    monkeypatch.setattr(check_performance_pr, "load_result", lambda _path, _contract: result)

    def validate(_path: Path, *, require_proceed: bool) -> dict[str, Any]:
        calls.append(require_proceed)
        return {"contract": contract, "decision": result}

    monkeypatch.setattr(check_performance_pr, "validate_evidence_bundle", validate)
    body = declaration(
        "optimization",
        contract="contract.json",
        evidence="evidence.json",
    )
    assert check_performance_pr.validate_pr(payload(body), root=tmp_path) == []
    assert calls == [True]


def test_measurement_accepts_a_stop_bundle(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "contract.json").write_text("{}")
    (tmp_path / "evidence.json").write_text("{}")
    contract = {"claim": "measured", "result_artifact": "result.json"}
    result = {"decision": "stop"}
    calls: list[bool] = []
    monkeypatch.setattr(check_performance_pr, "load_contract", lambda _path: contract)
    monkeypatch.setattr(check_performance_pr, "load_result", lambda _path, _contract: result)

    def validate(_path: Path, *, require_proceed: bool) -> dict[str, Any]:
        calls.append(require_proceed)
        return {"contract": contract, "decision": result}

    monkeypatch.setattr(check_performance_pr, "validate_evidence_bundle", validate)
    body = declaration("measurement", contract="contract.json", evidence="evidence.json")
    assert check_performance_pr.validate_pr(payload(body), root=tmp_path) == []
    assert calls == [False]


def test_cli_reads_the_pull_request_event(tmp_path: Path, monkeypatch, capsys) -> None:
    event = tmp_path / "event.json"
    event.write_text(json.dumps(payload(declaration("governance"))))
    monkeypatch.setattr("sys.argv", ["check_performance_pr.py", "--event", str(event)])

    assert check_performance_pr.main() == 0
    assert "ERROR" not in capsys.readouterr().out
