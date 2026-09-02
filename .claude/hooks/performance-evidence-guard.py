#!/usr/bin/env python3
"""Deny cloud experiments without a valid, launch-bound evidence contract."""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.DOTALL)
_SHELLS = {"bash", "dash", "sh", "zsh"}
_OPERATORS = set(";&|()")
_ENV_VALUE_OPTIONS = {"-C", "--chdir", "-u", "--unset"}
_UV_VALUE_OPTIONS = {
    "--directory",
    "--env-file",
    "--index",
    "--project",
    "--python",
    "--with",
    "--with-editable",
    "--with-requirements",
}


def _segments(command: str) -> list[list[str]]:
    """Tokenize shell segments while keeping quoted nested commands intact."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    lexer.commenters = ""
    segments: list[list[str]] = [[]]
    for token in lexer:
        if token and set(token) <= _OPERATORS:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(token)
    return [segment for segment in segments if segment]


def _after_wrappers(tokens: list[str]) -> list[str]:  # noqa: PLR0912 - wrapper grammar
    """Remove assignments and ordinary command-launch wrappers."""
    remaining = list(tokens)
    while remaining and _ASSIGNMENT.fullmatch(remaining[0]):
        remaining.pop(0)
    while remaining:
        executable = Path(remaining[0]).name
        if executable == "command":
            remaining.pop(0)
            if remaining and remaining[0] in {"-v", "-V"}:
                return []
            while remaining and remaining[0] == "-p":
                remaining.pop(0)
            continue
        if executable == "env":
            remaining.pop(0)
            while remaining:
                option = remaining[0]
                if _ASSIGNMENT.fullmatch(option) or option in {"-i", "--ignore-environment"}:
                    remaining.pop(0)
                elif option in _ENV_VALUE_OPTIONS:
                    remaining.pop(0)
                    if remaining:
                        remaining.pop(0)
                elif option.startswith(("--chdir=", "--unset=")):
                    remaining.pop(0)
                else:
                    break
            continue
        if executable in {"uv", "poetry"} and len(remaining) > 1 and remaining[1] == "run":
            remaining = remaining[2:]
            while remaining and remaining[0].startswith("-"):
                option = remaining.pop(0)
                if option == "--":
                    break
                if option in _UV_VALUE_OPTIONS and remaining:
                    remaining.pop(0)
            continue
        break
    return remaining


def _cli_launch(args: list[str]) -> bool:
    if not args:
        return False
    if args[0] == "process":
        return "--distributed" in args[1:]
    if args[0] == "benchmark":
        return True
    return len(args) > 1 and args[0] == "shard" and args[1] in {"process", "resume"}


def _tokens_launch(tokens: list[str]) -> bool:  # noqa: PLR0911 - command grammar branches
    tokens = _after_wrappers(tokens)
    if not tokens:
        return False
    executable = Path(tokens[0]).name
    args = tokens[1:]
    if executable in _SHELLS:
        for index, option in enumerate(args[:-1]):
            if option.startswith("-") and "c" in option[1:]:
                return command_launches(args[index + 1])
        return False
    if executable == "landsat-lst":
        return _cli_launch(args)
    if executable == "coiled":
        return args[:2] == ["batch", "run"]
    if not re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable):
        return False
    if "-m" in args:
        index = args.index("-m")
        return (
            len(args) > index + 1
            and args[index + 1] == "landsat_lst.cli"
            and _cli_launch(args[index + 2 :])
        )
    script_index = next((i for i, arg in enumerate(args) if not arg.startswith("-")), None)
    if script_index is None:
        return False
    script = Path(args[script_index]).name
    return script.startswith("probe_") and script.endswith(".py") and "--launch" in args


def command_launches(command: str) -> bool:
    """Recognize supported cloud launch forms without matching grep arguments."""
    try:
        return any(_tokens_launch(segment) for segment in _segments(command))
    except ValueError:
        return bool(re.search(r"(?:landsat-lst|coiled\s+batch|landsat_lst\.cli)", command))


def deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def contract_path(command: str) -> str | None:
    inline = re.search(r"\bLST_EVIDENCE_CONTRACT=(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))", command)
    if inline:
        return next(value for value in inline.groups() if value is not None)
    return os.environ.get("LST_EVIDENCE_CONTRACT")


def main() -> None:
    payload = json.load(sys.stdin)
    if payload.get("tool_name") != "Bash":
        print("{}")
        return
    command = payload.get("tool_input", {}).get("command", "")
    if not command_launches(command):
        print("{}")
        return
    contract = contract_path(command)
    if not contract:
        print(
            json.dumps(
                deny(
                    "Cloud experiment blocked: set LST_EVIDENCE_CONTRACT to a valid pre-registered JSON contract. The contract does not replace explicit operator approval."
                )
            )
        )
        return
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from landsat_lst.evidence_contract import (  # noqa: PLC0415
            bind_contract_to_repository,
            load_contract,
        )

        path = Path(contract)
        data = load_contract(path if path.is_absolute() else ROOT / path, launch_command=command)
        bind_contract_to_repository(data, ROOT)
    except (OSError, ValueError) as exc:
        print(json.dumps(deny(f"Cloud experiment blocked: invalid evidence contract: {exc}")))
        return
    print("{}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Hook failures are advisory in Claude Code. Emit a deny rather than a
        # traceback so malformed input or an import regression cannot fail open.
        print(json.dumps(deny(f"Cloud experiment blocked: evidence guard failed: {exc}")))
