from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

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
    assert errors[0].startswith(
        "performance language requires a non-none performance-evidence stage"
    )


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


@pytest.mark.parametrize(
    "prose",
    [
        "Cuts wall clock from 39h to 60min, halves peak RSS, cheaper credits.",
        "Reduces the runtime by 40%. Improves overall memory use.",
        "\u043eptimization of the composite pass",  # Cyrillic small o
        "f\u0430ster composite",  # Cyrillic small a
        "opti\u200bmization",  # zero-width space
        "Lower cost per tile.",
        "This is a perf change.",
        "One third the task count.",
    ],
)
def test_performance_language_is_caught_across_synonyms_and_homoglyphs(prose: str) -> None:
    errors = check_performance_pr.validate_pr(payload(declaration("none") + "\n" + prose))
    assert errors and errors[0].startswith("performance language requires"), prose


@pytest.mark.parametrize(
    "prose",
    [
        "Fix the QA COG header so qa_count carries no nodata.",
        "Rename the tile parser and update the docs.",
    ],
)
def test_ordinary_language_may_declare_none(prose: str) -> None:
    assert check_performance_pr.validate_pr(payload(declaration("none") + "\n" + prose)) == []


def test_a_crashing_validator_is_a_review_error_not_a_traceback(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "contract.json").write_text("{}")
    (tmp_path / "evidence.json").write_text("{}")
    monkeypatch.setattr(
        check_performance_pr, "load_contract", lambda _p: {"result_artifact": "r.json"}
    )
    monkeypatch.setattr(
        check_performance_pr, "load_result", lambda _p, _c: (_ for _ in ()).throw(KeyError("x"))
    )

    def explode(_path: Path, *, require_proceed: bool) -> dict[str, Any]:
        raise TypeError("float() argument must be a string or a real number, not 'dict'")

    monkeypatch.setattr(check_performance_pr, "validate_evidence_bundle", explode)
    body = declaration("optimization", contract="contract.json", evidence="evidence.json")
    errors = check_performance_pr.validate_pr(payload(body), root=tmp_path)
    assert any(e.startswith("result is invalid: KeyError") for e in errors), errors
    assert any(e.startswith("evidence bundle is invalid: TypeError") for e in errors), errors


def test_main_reads_the_ci_event_and_fails_on_a_bad_declaration(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The path CI runs: no --event flag, the event comes from the environment.

    A one-line edit that drops this branch would print "skipped" and exit 0
    on every pull request; this is the test that notices.
    """
    event = tmp_path / "event.json"
    event.write_text(json.dumps(payload("", title="Optimize everything")))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request_target")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))

    assert check_performance_pr.main([]) == 1
    assert "ERROR: PR body must contain exactly one" in capsys.readouterr().out

    event.write_text(json.dumps(payload(declaration("governance"))))
    assert check_performance_pr.main([]) == 0


def test_main_skips_outside_a_pull_request_event(monkeypatch, capsys) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    assert check_performance_pr.main([]) == 0
    assert "skipped" in capsys.readouterr().out


def test_main_resolves_declared_paths_under_root(tmp_path: Path, monkeypatch, capsys) -> None:
    """Under pull_request_target the PR tree is a sibling directory, not cwd."""
    pr_tree = tmp_path / "pr-head"
    pr_tree.mkdir()
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(payload(declaration("instrumentation", contract="docs/contract.json")))
    )
    monkeypatch.setattr(check_performance_pr, "load_contract", lambda path: {"path": str(path)})

    assert check_performance_pr.main(["--event", str(event), "--root", str(pr_tree)]) == 1
    assert "does not resolve to a committed file" in capsys.readouterr().out

    (pr_tree / "docs").mkdir()
    (pr_tree / "docs" / "contract.json").write_text("{}")
    assert check_performance_pr.main(["--event", str(event), "--root", str(pr_tree)]) == 0
