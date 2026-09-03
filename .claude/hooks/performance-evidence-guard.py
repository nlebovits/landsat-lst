#!/usr/bin/env python3
"""Deny cloud experiments without a valid, launch-bound evidence contract.

The guard recognises a cloud launch through the shell forms an agent can
reach: direct, wrapped (``env``, ``uv run``, ``exec``, ``nohup``, ``timeout``,
``nice``, ...), nested (``bash -c``, ``eval``, ``ssh``, ``xargs``), the Python
API (``python -c``, a heredoc, a script file naming ``submit_batch``), and a
recipe runner (``make``, ``just``, ``tox``) whose file names one. Read-only
commands that merely mention these words (``grep``, ``git commit -m``) pass.

The contract is bound to the checkout the command runs *from* (the hook
payload's ``cwd``), never to the checkout that holds this file: a launch from
a worktree ships that worktree's code.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

#: The checkout holding this hook. It supplies the validator module and
#: nothing else: the launch checkout is resolved from the command's cwd.
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
#: Wrappers that take no options of their own before the command.
_BARE_WRAPPERS = {"exec", "nohup", "setsid", "time", "caffeinate", "unbuffer", "builtin"}
#: Wrappers whose options may carry a value, then the command follows. The
#: value sets name options that consume the next token.
_OPTION_WRAPPERS: dict[str, set[str]] = {
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "stdbuf": {"-i", "-o", "-e", "--input", "--output", "--error"},
    "ionice": {"-c", "-n", "--class", "--classdata"},
    "sudo": {"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-T", "-U"},
    "doas": {"-u", "-C"},
    "flock": {"-w", "-E", "--timeout", "--conflict-exit-code"},
    "chronic": set(),
}
#: ``xargs`` options that consume the next token, before the command.
_XARGS_VALUE_OPTIONS = {"-I", "-L", "-n", "-P", "-a", "-d", "-E", "-s", "-i", "-l"}
#: ``ssh`` options that consume the next token, before the host.
_SSH_VALUE_OPTIONS = set("bcDEeFIiJLlmOopQRSWw")
_PYTHON = re.compile(r"python(?:3(?:\.\d+)?)?")
#: The Python API surface that submits cloud work, whatever file spells it.
_PYTHON_API = re.compile(
    r"\b(?:submit_batch|submit_shard_stage|submit_fleet_stage|submit_sweep|"
    r"drive_tile|resume_tile|batch_run|coiled\.Cloud)\b"
)
_RECIPE_FILES = {
    "make": ("Makefile", "makefile", "GNUmakefile"),
    "just": ("justfile", "Justfile", ".justfile"),
    "tox": ("tox.ini",),
}
#: When the command cannot be tokenised (an unbalanced quote), fall back to
#: launch shapes only, so ``grep "landsat-lst process src/`` is not a launch.
_LAUNCH_SHAPE = re.compile(
    r"landsat-lst\s+(?:process\b[^|;&]*--distributed|shard\s+(?:process|resume)\b"
    r"|benchmark\b[^|;&]*--distributed)"
    r"|coiled\s+(?:batch\s+run|run|notebook)\b"
    r"|landsat_lst\.cli\s+(?:process\b[^|;&]*--distributed|shard\s+(?:process|resume)\b"
    r"|benchmark\b[^|;&]*--distributed)"
)


def _segments(command: str) -> list[list[str]]:
    """Tokenize shell segments while keeping quoted nested commands intact."""
    # ANSI-C quoting: ``$'process'`` is the word ``process`` to bash.
    command = re.sub(r"\$(?=')", "", command)
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


def _skip_wrapper_options(remaining: list[str], value_options: set[str]) -> list[str]:
    """Drop a wrapper's own options, including a positional duration or level."""
    while remaining and remaining[0].startswith("-"):
        option = remaining.pop(0)
        if option == "--":
            break
        if option in value_options and remaining:
            remaining.pop(0)
    return remaining


def _after_wrappers(tokens: list[str]) -> list[str]:  # noqa: PLR0912 - wrapper grammar
    """Remove assignments and command-launch wrappers until the command shows."""
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
        if executable in _BARE_WRAPPERS:
            remaining.pop(0)
            continue
        if executable in _OPTION_WRAPPERS:
            remaining.pop(0)
            remaining = _skip_wrapper_options(remaining, _OPTION_WRAPPERS[executable])
            if executable == "timeout" and remaining:
                remaining.pop(0)  # the duration
            continue
        break
    return remaining


def _cli_launch(args: list[str]) -> bool:
    if not args:
        return False
    if args[0] in {"process", "benchmark"}:
        # ``benchmark`` without ``--distributed`` is the local CI tier.
        return "--distributed" in args[1:]
    return len(args) > 1 and args[0] == "shard" and args[1] in {"process", "resume"}


def _python_launch(args: list[str], command: str, cwd: Path) -> bool:
    if "-m" in args:
        index = args.index("-m")
        return (
            len(args) > index + 1
            and args[index + 1] == "landsat_lst.cli"
            and _cli_launch(args[index + 2 :])
        )
    if "-c" in args:
        index = args.index("-c")
        return len(args) > index + 1 and bool(_PYTHON_API.search(args[index + 1]))
    if "<<" in command and _PYTHON_API.search(command):
        # ``python - <<'PY'`` or ``python <<PY``: the program is the heredoc,
        # which the lexer has already spread across the token list.
        return True
    script_index = next((i for i, arg in enumerate(args) if not arg.startswith(("-", "<"))), None)
    if script_index is None:
        return False
    script = args[script_index]
    if Path(script).name.startswith("probe_") and script.endswith(".py"):
        return "--launch" in args
    return _file_launches(cwd / script)


def _file_launches(path: Path) -> bool:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return False
    return bool(_PYTHON_API.search(text))


#: Recipe files being scanned, so a recursive ``make`` inside a Makefile
#: cannot recurse the scan.
_SCANNING: set[str] = set()


def _recipe_launches(executable: str, cwd: Path) -> bool:
    if executable in _SCANNING:
        return False
    _SCANNING.add(executable)
    try:
        for name in _RECIPE_FILES[executable]:
            path = cwd / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            if _PYTHON_API.search(text) or any(
                command_launches(line, cwd) for line in text.splitlines()
            ):
                return True
        return False
    finally:
        _SCANNING.discard(executable)


def _tokens_launch(  # noqa: PLR0911, PLR0912 - command grammar branches
    tokens: list[str], command: str, cwd: Path, flat: list[str]
) -> bool:
    tokens = _after_wrappers(tokens)
    if not tokens:
        return False
    executable = Path(tokens[0]).name
    args = tokens[1:]
    if executable in _SHELLS:
        for index, option in enumerate(args[:-1]):
            if option.startswith("-") and "c" in option[1:]:
                return command_launches(args[index + 1], cwd)
        return False
    if executable == "eval":
        return command_launches(" ".join(args), cwd)
    if executable == "ssh":
        rest = list(args)
        while rest and rest[0].startswith("-"):
            option = rest.pop(0)
            if len(option) == 2 and option[1] in _SSH_VALUE_OPTIONS and rest:
                rest.pop(0)
        return len(rest) > 1 and command_launches(" ".join(rest[1:]), cwd)
    if executable == "xargs":
        # The launch arguments arrive on stdin: ``echo --distributed | xargs
        # landsat-lst process``. Judge the command with every flag in the pipe.
        rest = _skip_wrapper_options(list(args), _XARGS_VALUE_OPTIONS)
        piped = [token for token in flat if token.startswith("--") and token not in rest]
        return _tokens_launch(rest + piped, command, cwd, [])
    if executable == "landsat-lst":
        return _cli_launch(args)
    if executable == "coiled":
        return args[:2] == ["batch", "run"] or args[:1] in (["run"], ["notebook"])
    if executable in _RECIPE_FILES:
        return _recipe_launches(executable, cwd)
    if _PYTHON.fullmatch(executable):
        return _python_launch(args, command, cwd)
    return False


def command_launches(command: str, cwd: Path | None = None) -> bool:
    """Recognize supported cloud launch forms without matching grep arguments."""
    cwd = cwd or Path.cwd()
    try:
        segments = _segments(command)
    except ValueError:
        return bool(_LAUNCH_SHAPE.search(command) or _PYTHON_API.search(command))
    flat = [token for segment in segments for token in segment]
    return any(_tokens_launch(segment, command, cwd, flat) for segment in segments)


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
    cwd = Path(payload["cwd"]) if payload.get("cwd") else Path.cwd()
    if not command_launches(command, cwd):
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
            launch_root,
            load_contract,
        )

        root = launch_root(cwd)
        path = Path(contract)
        data = load_contract(path if path.is_absolute() else root / path, launch_command=command)
        bind_contract_to_repository(data, root)
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
