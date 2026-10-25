from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from io import StringIO
from pathlib import Path


def run_pyflakes(path: Path) -> tuple[str, str]:
    """Run pyflakes and return ``(output, error)``.

    The API path is preferred because the OpenHands environment often has the
    library installed without a shell-visible ``pyflakes`` executable.
    """

    output, error = _run_pyflakes_api(path)
    if not error:
        return output, ""

    errors = [error]
    seen: set[tuple[str, ...]] = set()
    for command in _candidate_commands(path):
        key = tuple(command)
        if key in seen:
            continue
        seen.add(key)
        try:
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except OSError as exc:
            errors.append(f"{command[0]} failed: {type(exc).__name__}: {exc}")
            continue
        output = completed.stdout or ""
        if _valid_pyflakes_exit(completed.returncode, output):
            return output, ""
        errors.append(
            f"{' '.join(shlex.quote(part) for part in command)} exited "
            f"{completed.returncode}: {output.strip()}"
        )
    return "", "; ".join(errors)


def parse_undefined_names(output: str) -> list[str]:
    names: set[str] = set()
    for line in output.splitlines():
        match = re.search(r"undefined name ['\"]([^'\"]+)['\"]", line)
        if match:
            names.add(match.group(1))
    return sorted(names)


def _candidate_commands(path: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    configured_python = os.environ.get("ARCH_AGENT_PYFLAKES_PYTHON")
    if configured_python:
        commands.append([configured_python, "-m", "pyflakes", str(path)])
    configured_bin = os.environ.get("ARCH_AGENT_PYFLAKES_BIN")
    if configured_bin:
        commands.append([configured_bin, str(path)])
    commands.extend(
        [
            [sys.executable, "-m", "pyflakes", str(path)],
            ["pyflakes", str(path)],
        ]
    )
    return commands


def _valid_pyflakes_exit(returncode: int, output: str) -> bool:
    if "No module named pyflakes" in output:
        return False
    if returncode == 0:
        return True
    if returncode == 1:
        return "undefined name" in output or bool(re.search(r":\d+:\d+:", output))
    return False


def _run_pyflakes_api(path: Path) -> tuple[str, str]:
    try:
        from pyflakes.api import checkPath
        from pyflakes.reporter import Reporter
    except Exception as exc:
        return "", f"pyflakes API unavailable: {type(exc).__name__}: {exc}"

    warnings = StringIO()
    errors = StringIO()
    try:
        checkPath(str(path), reporter=Reporter(warnings, errors))
    except Exception as exc:
        return "", f"pyflakes API failed: {type(exc).__name__}: {exc}"
    return warnings.getvalue() + errors.getvalue(), ""
