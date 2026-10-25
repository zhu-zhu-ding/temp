from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Any

from .models import EntityRef
from .pyflakes_runner import parse_undefined_names, run_pyflakes
from .static_index import normalize_rel_path

STATIC_ISSUES_MD = "static_issues.md"


@dataclass
class StaticAnalysisResult:
    passed: bool
    markdown: str
    data: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def analyze_static_issues(
    repo_path: Path,
    state_dir: Path,
    *,
    task_id: str = "",
    interfaces: list[EntityRef] | None = None,
) -> StaticAnalysisResult:
    repo_path = repo_path.resolve()
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    if interfaces is None:
        interfaces = []

    if not interfaces:
        markdown = _default_static_markdown()
        (state_dir / STATIC_ISSUES_MD).write_text(markdown, encoding="utf-8")
        return StaticAnalysisResult(
            passed=True,
            markdown=markdown,
            data={
                "repo_path": str(repo_path),
                "task_id": task_id,
                "files": [],
                "file_issues": {},
                "skipped": True,
                "reason": "no interface paths found in task description",
            },
        )

    files = _static_analysis_scope(repo_path, interfaces)
    file_issues = _check_static_files(repo_path, files)
    markdown = _static_issues_markdown(file_issues)
    (state_dir / STATIC_ISSUES_MD).write_text(markdown, encoding="utf-8")
    return StaticAnalysisResult(
        passed=not file_issues,
        markdown=markdown,
        data={
            "repo_path": str(repo_path),
            "task_id": task_id,
            "files": files,
            "file_issues": file_issues,
        },
    )


def _default_static_markdown() -> str:
    return (
        "## static issue\n\n"
        "*No interface paths were found in the task description. This appears to be "
        "a standalone task, so static_analyze has no repo-local interface files to "
        "grep or check with pyflakes.*\n"
    )


def _static_analysis_scope(repo_path: Path, interfaces: list[EntityRef]) -> list[str]:
    files: set[str] = {
        normalize_rel_path(item.path) for item in interfaces if item.path
    }
    keywords: set[str] = set()
    for item in interfaces:
        if item.name:
            keywords.add(item.name)
        if item.parent:
            keywords.add(item.parent)

    search_dirs = _interface_search_dirs(files, repo_path)
    for keyword in sorted(keywords):
        if not keyword:
            continue
        for search_dir in search_dirs:
            try:
                result = subprocess.run(
                    [
                        "grep",
                        "-rlE",
                        "--include=*.py",
                        "-e",
                        rf"\bdef {keyword}\b",
                        "-e",
                        rf"\bclass {keyword}\b",
                        "-e",
                        rf"\bfrom .* import .*{keyword}\b",
                        "-e",
                        rf"\b{keyword}\(",
                        str(search_dir),
                    ],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            for line in result.stdout.splitlines():
                rel_path = _repo_relative_path(repo_path, line.strip())
                if rel_path:
                    files.add(rel_path)
    return sorted(files)


def _interface_search_dirs(files: set[str], repo_path: Path) -> list[Path]:
    dirs: set[Path] = set()
    for rel_path in files:
        parts = Path(rel_path).parts
        if not parts:
            continue
        top_level = repo_path / parts[0]
        if top_level.exists() and top_level.is_dir():
            dirs.add(top_level)
    return sorted(dirs)


def _repo_relative_path(repo_path: Path, raw_path: str) -> str:
    if not raw_path:
        return ""
    path = Path(raw_path)
    try:
        if path.is_absolute():
            raw_path = path.relative_to(repo_path).as_posix()
    except ValueError:
        return ""
    return normalize_rel_path(raw_path)


def _check_static_files(repo_path: Path, files: list[str]) -> dict[str, list[dict[str, str]]]:
    issues: dict[str, list[dict[str, str]]] = {}
    for rel_path in files:
        full_path = repo_path / rel_path
        if not full_path.exists():
            issues[rel_path] = [
                {
                    "type": "missing_file",
                    "detail": f"`{rel_path}` does not exist in the repository",
                }
            ]
            continue

        file_issues: list[dict[str, str]] = []
        output, pyflakes_error = run_pyflakes(full_path)
        if pyflakes_error and "No module named pyflakes" not in pyflakes_error:
            file_issues.append(
                {"type": "syntax_error", "detail": pyflakes_error[:200]}
            )
        for name in parse_undefined_names(output):
            file_issues.append(
                {
                    "type": "undefined_name",
                    "symbol": name,
                    "detail": f"'{name}' is used but not defined or imported in this file",
                }
            )
        if file_issues:
            issues[rel_path] = file_issues
    return issues


def _static_issues_markdown(file_issues: dict[str, list[dict[str, str]]]) -> str:
    lines = ["## static issue", ""]
    if not file_issues:
        lines.append("*No issues found.*")
        lines.append("")
        return "\n".join(lines)

    for rel_path in sorted(file_issues):
        issues = file_issues[rel_path]
        lines.append(f"- **`{rel_path}`** ({len(issues)}):")
        for issue in issues:
            marker = {
                "syntax_error": "X",
                "undefined_name": "?",
                "missing_file": "!",
            }.get(issue["type"], "!")
            lines.append(f"  - {marker} {issue['detail']}")
    lines.append("")
    return "\n".join(lines)


def _parse_interface_blocks(problem_statement: str) -> list[EntityRef]:
    import re

    interfaces: list[EntityRef] = []
    pattern = re.compile(
        r"Path:\s*`/testbed/([^`]+)`\s*\n```python\s*\n(.*?)\n```",
        re.S,
    )
    seen: set[tuple[str, str, str, str]] = set()
    for match in pattern.finditer(problem_statement):
        filepath = match.group(1)
        code = match.group(2)
        current_class = ""
        for raw_line in code.splitlines():
            line = raw_line.strip()
            class_match = re.match(r"class (\w+)", line)
            if class_match:
                current_class = class_match.group(1)
                key = (filepath, current_class, "class", "")
                if key not in seen:
                    seen.add(key)
                    interfaces.append(
                        EntityRef(
                            path=filepath,
                            name=current_class,
                            kind="class",
                            parent="",
                        )
                    )
                continue
            func_match = re.match(r"def (\w+)", line)
            if func_match:
                name = func_match.group(1)
                kind = "method" if current_class else "function"
                parent = current_class if current_class else ""
                key = (filepath, name, kind, parent)
                if key not in seen:
                    seen.add(key)
                    interfaces.append(
                        EntityRef(
                            path=filepath,
                            name=name,
                            kind=kind,
                            parent=parent,
                        )
                    )
    return interfaces
