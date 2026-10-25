from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import SimpleArtifacts
from .models import EntityRef
from .repo_understand import STATIC_ISSUES_MD, analyze_static_issues


COMMON_WORDS = {
    "above",
    "according",
    "after",
    "also",
    "below",
    "class",
    "code",
    "description",
    "function",
    "implementation",
    "important",
    "input",
    "interface",
    "must",
    "output",
    "path",
    "python",
    "returns",
    "should",
    "task",
    "test",
    "tests",
    "this",
    "under",
    "value",
}


@dataclass(frozen=True)
class InterfaceSpec:
    path: str
    symbols: list[str]
    snippet: str


@dataclass(frozen=True)
class StaticDesignResult:
    interfaces: list[InterfaceSpec]
    related_files: list[str]
    test_files: list[str]
    static_issues: str
    log: str


def prepare_static_analysis(task: str, repo_path: Path, artifacts: SimpleArtifacts) -> StaticDesignResult:
    interfaces = _extract_interfaces(task)
    static_result = analyze_static_issues(
        repo_path,
        artifacts.root,
        interfaces=_to_entity_refs(interfaces),
    )
    log_lines = [
        "static analysis prepared",
        f"interfaces={len(interfaces)}",
        "related_files=0",
        "test_files=0",
        f"static_analysis_passed={static_result.passed}",
        f"static_issues={artifacts.root / STATIC_ISSUES_MD}",
    ]
    return StaticDesignResult(interfaces, [], [], static_result.markdown, "\n".join(log_lines) + "\n")


def generate_static_design(task: str, repo_path: Path, artifacts: SimpleArtifacts) -> StaticDesignResult:
    interfaces = _extract_interfaces(task)
    keywords = _extract_keywords(task, interfaces)
    related_files = _find_related_files(repo_path, interfaces, keywords)
    test_files = _find_test_files(related_files)
    static_result = analyze_static_issues(
        repo_path,
        artifacts.root,
        interfaces=_to_entity_refs(interfaces),
    )
    design_items = _build_contracts(repo_path, interfaces, related_files, test_files)
    static_items = _build_static_issue_contracts(static_result.data.get("file_issues", {}))
    design_items = [*static_items, *design_items][:8]
    (artifacts.root / "static_design.md").write_text(
        _render_design_doc(task, interfaces, related_files, test_files, design_items, static_result.markdown),
        encoding="utf-8",
    )
    log_lines = [
        "static design generated",
        f"interfaces={len(interfaces)}",
        f"related_files={len(related_files)}",
        f"test_files={len(test_files)}",
        f"static_analysis_passed={static_result.passed}",
        f"static_issues={artifacts.root / STATIC_ISSUES_MD}",
    ]
    return StaticDesignResult(interfaces, related_files, test_files, static_result.markdown, "\n".join(log_lines) + "\n")


def _extract_interfaces(task: str) -> list[InterfaceSpec]:
    interfaces: list[InterfaceSpec] = []
    pattern = re.compile(
        r"Path:\s*`?(?P<path>/testbed/[^`\n]+)`?.*?```python\n(?P<snippet>.*?)```",
        re.DOTALL,
    )
    for match in pattern.finditer(task):
        path = _normalize_repo_path(match.group("path"))
        snippet = match.group("snippet").strip()
        symbols = _extract_symbols(snippet)
        interfaces.append(InterfaceSpec(path, symbols, snippet))

    if interfaces:
        return _dedupe_interfaces(interfaces)

    paths = [_normalize_repo_path(path) for path in re.findall(r"`(/testbed/[^`\n]+)`", task)]
    symbols = _extract_symbols(task)
    for path in paths[:12]:
        interfaces.append(InterfaceSpec(path, symbols[:8], ""))
    if interfaces:
        return _dedupe_interfaces(interfaces)

    fallback_symbols = symbols[:8] or _extract_keywords(task, [])[:5]
    return [InterfaceSpec("UNKNOWN_TARGET.py", fallback_symbols, task[:2000])]


def _extract_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    for match in re.finditer(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE):
        symbols.append(match.group(1))
    return _dedupe(symbols)[:20]


def _extract_keywords(task: str, interfaces: list[InterfaceSpec]) -> list[str]:
    values: list[str] = []
    for spec in interfaces:
        values.extend(spec.symbols)
        values.extend(part for part in Path(spec.path).parts if part.endswith(".py") or "_" in part)
    values.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", task))
    keywords = []
    for value in values:
        value = value.strip("_")
        if not value or value.lower() in COMMON_WORDS:
            continue
        if value.startswith("__") and value.endswith("__"):
            continue
        keywords.append(value)
    return _dedupe(keywords)[:40]


def _find_related_files(repo_path: Path, interfaces: list[InterfaceSpec], keywords: list[str]) -> list[str]:
    scores: dict[str, int] = {}
    for spec in interfaces:
        if spec.path != "UNKNOWN_TARGET.py":
            scores[spec.path] = scores.get(spec.path, 0) + 100
            sibling = Path(spec.path).parent
            for symbol in spec.symbols[:8]:
                for file in _rg_files(repo_path, symbol)[:30]:
                    scores[file] = scores.get(file, 0) + (30 if Path(file).parent == sibling else 12)

    for keyword in keywords[:18]:
        for file in _rg_files(repo_path, keyword)[:25]:
            scores[file] = scores.get(file, 0) + (8 if file.endswith(".py") else 2)

    ranked = [
        file
        for file, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if _is_useful_file(file)
    ]
    return ranked[:40]


def _find_test_files(files: list[str]) -> list[str]:
    return [file for file in files if _is_test_file(file)][:12]


def _build_contracts(
    repo_path: Path,
    interfaces: list[InterfaceSpec],
    related_files: list[str],
    test_files: list[str],
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    core_files = [spec.path for spec in interfaces if spec.path != "UNKNOWN_TARGET.py"]
    core_symbols = _dedupe(symbol for spec in interfaces for symbol in spec.symbols)
    if not core_files and related_files:
        core_files = related_files[:2]
    for index, file in enumerate(core_files[:8], start=1):
        file_symbols = []
        for spec in interfaces:
            if spec.path == file:
                file_symbols = spec.symbols[:12]
                break
        contracts.append(
            {
                "id": f"core_behavior_{index}",
                "role": "core_behavior",
                "required": True,
                "files": [file],
                "symbols": file_symbols or core_symbols[:8],
                "constraint": "Implement the requested behavior in this declared interface file without deleting existing compatible behavior.",
                "failure_if_omitted": "The benchmark imports or calls the required interface directly and will fail if the symbol remains missing or stubbed.",
                "evidence": [_evidence(repo_path, file, "declared task interface")],
            }
        )

    callers = [
        file
        for file in related_files
        if file not in core_files and not _is_test_file(file) and not file.endswith("__init__.py")
    ][:10]
    if callers:
        contracts.append(
            {
                "id": "caller_propagation",
                "role": "caller_compat",
                "required": True,
                "files": callers,
                "symbols": [],
                "constraint": "Propagate signature, return-value, error-handling, or data-shape changes through nearby callers and wrappers.",
                "failure_if_omitted": "Local interface tests may pass while integration paths still call the old contract.",
                "evidence": [_evidence(repo_path, file, "references requested symbols or sibling behavior") for file in callers],
            }
        )

    inferred_public_files = _infer_public_files(repo_path, core_files)
    public_files = _dedupe([
        *inferred_public_files,
        *[
        file
        for file in related_files
        if file.endswith("__init__.py") or "/api" in file or "public" in file or "entities" in file
        ],
    ])[:8]
    if public_files:
        contracts.append(
            {
                "id": "public_api_exports",
                "role": "public_api",
                "required": True,
                "files": public_files,
                "symbols": core_symbols[:12],
                "constraint": "Keep public imports, aliases, and documented entry points consistent with the implemented interfaces.",
                "failure_if_omitted": "Users or tests importing through the package-level API will not see the new behavior.",
                "evidence": [_evidence(repo_path, file, "public/export surface near the requested interface") for file in public_files],
            }
        )

    boundary_files = [
        file
        for file in related_files
        if re.search(r"(serial|proto|schema|json|to_dict|from_dict|config|registry|validation|validator)", file)
    ][:8]
    if boundary_files:
        contracts.append(
            {
                "id": "boundary_integration",
                "role": "serialization",
                "required": True,
                "files": boundary_files,
                "symbols": core_symbols[:12],
                "constraint": "Update serialization, validation, registry, and configuration boundaries when the requested interface crosses them.",
                "failure_if_omitted": "The feature works only in memory and fails across persistence, validation, registration, or config-driven paths.",
                "evidence": [_evidence(repo_path, file, "boundary module related to requested behavior") for file in boundary_files],
            }
        )

    if test_files:
        contracts.append(
            {
                "id": "integration_validation",
                "role": "validation",
                "required": False,
                "files": test_files[:8],
                "symbols": [],
                "constraint": "Use nearby existing tests as behavioral examples and run focused checks when practical.",
                "failure_if_omitted": "Implementation may satisfy the interface in isolation but miss repository-specific integration behavior.",
                "evidence": [_evidence(repo_path, file, "nearby test coverage") for file in test_files[:8]],
            }
        )

    return contracts[:8]


def _build_static_issue_contracts(file_issues: Any) -> list[dict[str, Any]]:
    if not isinstance(file_issues, dict) or not file_issues:
        return []

    files = [str(file) for file in file_issues if isinstance(file, str)][:8]
    if not files:
        return []

    issue_summaries: list[str] = []
    for file in files:
        issues = file_issues.get(file, [])
        if not isinstance(issues, list):
            continue
        details = []
        for issue in issues[:4]:
            if isinstance(issue, dict):
                details.append(str(issue.get("detail", "")).strip())
        detail = "; ".join(item for item in details if item)
        issue_summaries.append(f"{file}: {detail}" if detail else file)

    return [
        {
            "id": "static_issue_closure",
            "role": "validation",
            "required": True,
            "files": files,
            "symbols": [],
            "constraint": "Resolve the static issues reported by migrated static_analyze for task-related interface files before submitting.",
            "failure_if_omitted": "Undefined names, missing files, or syntax errors in the task interface path will fail direct imports or focused tests.",
            "evidence": issue_summaries,
        }
    ]


def _infer_public_files(repo_path: Path, core_files: list[str]) -> list[str]:
    files: list[str] = []
    for file in core_files:
        path = Path(file)
        for parent in [path.parent, *path.parents]:
            if not str(parent) or str(parent) == ".":
                continue
            candidate = parent / "__init__.py"
            if (repo_path / candidate).exists():
                files.append(str(candidate))
            if len(files) >= 8:
                return _dedupe(files)
    return _dedupe(files)


def _build_donor_features(related_files: list[str], test_files: list[str]) -> list[dict[str, Any]]:
    donors: list[dict[str, Any]] = []
    source_files = [file for file in related_files if not _is_test_file(file)][:8]
    if source_files:
        donors.append(
            {
                "name": "nearby repository implementation pattern",
                "source_files": source_files,
                "test_files": test_files[:6],
                "reused_pattern": "Reuse local naming, caller propagation, export, validation, and boundary-handling conventions instead of adding isolated code.",
            }
        )
    return donors


def _build_validation_plan(test_files: list[str], related_files: list[str]) -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []
    for file in test_files[:3]:
        plan.append({"command": f"python -m pytest {file} -q", "reason": "Focused nearby test for requested behavior."})
    if related_files:
        plan.append({"command": "python -m compileall -q .", "reason": "Catch syntax/import breakage after propagating the patch."})
    return plan[:4]


def _render_design_contract_py(
    interfaces: list[InterfaceSpec],
    related_files: list[str],
    test_files: list[str],
    contracts: list[dict[str, Any]],
    file_issues: Any,
) -> str:
    interface_rows = [
        {"path": spec.path, "symbols": spec.symbols, "snippet": _summarize_snippet(spec.snippet)}
        for spec in interfaces
        if spec.path != "UNKNOWN_TARGET.py"
    ]
    review_files = [
        file
        for file in related_files[:16]
        if not _is_low_signal_file(file)
    ]
    data = {
        "interfaces": interface_rows,
        "review_files": review_files,
        "test_files": test_files[:8],
        "contracts": contracts,
        "static_issues": file_issues if isinstance(file_issues, dict) else {},
    }
    return f'''#!/usr/bin/env python3
"""
Executable design contract generated by arch_swe_agent_simple.

This file follows the original arch_agent mechanism: it is a lightweight
architecture constraint script, not a JSON role-coverage contract and not a
generated benchmark test suite.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


DATA = {repr(data)}


def _repo_root() -> Path:
    return Path.cwd()


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _parse(path: Path) -> ast.AST:
    return ast.parse(_read(path), filename=str(path))


def _defined_symbols(path: Path) -> set[str]:
    tree = _parse(path)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _assert(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        print("PASS:", message)
    else:
        print("FAIL:", message)
        failures.append(message)


def test_static_issues(repo: Path, failures: list[str]) -> None:
    for rel_path, issues in DATA["static_issues"].items():
        path = repo / rel_path
        _assert(path.exists(), f"static issue file exists: {{rel_path}}", failures)
        if not path.exists():
            continue
        try:
            _parse(path)
            syntax_ok = True
        except SyntaxError as exc:
            syntax_ok = False
            print(f"FAIL: syntax error in {{rel_path}}: {{exc}}")
        _assert(syntax_ok, f"static issue file parses: {{rel_path}}", failures)
        text = _read(path)
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            symbol = issue.get("symbol")
            if issue.get("type") == "undefined_name" and symbol:
                defined_or_imported = (
                    f"def {{symbol}}" in text
                    or f"class {{symbol}}" in text
                    or f"import {{symbol}}" in text
                    or f"import {{symbol}} as " in text
                    or f" import {{symbol}}" in text
                )
                _assert(defined_or_imported, f"static undefined name resolved or imported: {{symbol}} in {{rel_path}}", failures)


def test_core_interfaces(repo: Path, failures: list[str]) -> None:
    for item in DATA["interfaces"]:
        rel_path = item["path"]
        path = repo / rel_path
        _assert(path.exists(), f"core interface file exists: {{rel_path}}", failures)
        if not path.exists() or path.suffix != ".py":
            continue
        try:
            symbols = _defined_symbols(path)
        except SyntaxError as exc:
            print(f"FAIL: cannot parse core interface {{rel_path}}: {{exc}}")
            failures.append(f"cannot parse {{rel_path}}")
            continue
        for symbol in item.get("symbols", []):
            _assert(symbol in symbols, f"core symbol exists: {{rel_path}}::{{symbol}}", failures)


def test_design_review_scope(repo: Path, failures: list[str]) -> None:
    for rel_path in DATA["review_files"]:
        path = repo / rel_path
        _assert(path.exists(), f"design review candidate exists: {{rel_path}}", failures)
    print("INFO: similar/test files for implementation review:")
    for rel_path in DATA["test_files"]:
        print(f"INFO: test/example candidate: {{rel_path}}")
    for contract in DATA["contracts"]:
        print(f"INFO: design constraint {{contract.get('id')}}/{{contract.get('role')}}: {{contract.get('constraint')}}")


def main() -> int:
    repo = _repo_root()
    failures: list[str] = []
    test_static_issues(repo, failures)
    test_core_interfaces(repo, failures)
    test_design_review_scope(repo, failures)
    if failures:
        print("\\nDESIGN CONTRACT FAILED")
        print(json.dumps(failures, indent=2))
        return 1
    print("\\nDESIGN CONTRACT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _render_design_doc(
    task: str,
    interfaces: list[InterfaceSpec],
    related_files: list[str],
    test_files: list[str],
    contracts: list[dict[str, Any]],
    static_issues: str,
) -> str:
    interface_lines = "\n".join(
        f"- `{spec.path}`: {', '.join(spec.symbols) if spec.symbols else '(symbols inferred during implementation)'}"
        for spec in interfaces
    )
    related_lines = "\n".join(f"- `{file}`" for file in related_files[:20]) or "- No related files found by static search."
    test_lines = "\n".join(f"- `{file}`" for file in test_files[:12]) or "- No nearby tests found by static search."
    contract_lines = "\n".join(
        f"- `{item['id']}` / `{item['role']}`: {item['constraint']}" for item in contracts
    )
    return f"""# Architecture Design

## Task Understanding
{_summarize_task(task)}

## Core Interfaces
{interface_lines}

## Similar Features And Reusable Patterns
The design pass used deterministic repository search around the requested paths and symbols. Reuse nearby implementation style, import/export conventions, validation behavior, and integration boundaries instead of adding isolated task-only code.

## Static Issues
{static_issues.strip()}

## Functional Architecture
The patch should start at the declared core interface, then propagate through callers, package-level exports, registries/configuration/serialization boundaries, and nearby integration tests when those files are present.

## Module Responsibilities
{related_lines}

## Design Constraints
{contract_lines}

## Validation Plan
{test_lines}
"""


def _summarize_task(task: str) -> str:
    text = re.sub(r"\s+", " ", task).strip()
    if len(text) <= 700:
        return text
    return text[:700].rsplit(" ", 1)[0] + " ..."


def _summarize_snippet(snippet: str) -> str:
    text = re.sub(r"\s+", " ", snippet).strip()
    if not text:
        return "Implement the behavior required by the task for this interface."
    return text[:500].rsplit(" ", 1)[0] + (" ..." if len(text) > 500 else "")


def _to_entity_refs(interfaces: list[InterfaceSpec]) -> list[EntityRef]:
    refs: list[EntityRef] = []
    for spec in interfaces:
        if spec.path == "UNKNOWN_TARGET.py":
            continue
        symbols = spec.symbols or [Path(spec.path).stem]
        for symbol in symbols:
            refs.append(
                EntityRef(
                    path=spec.path,
                    name=symbol,
                    kind="class_or_function",
                    parent="",
                    source="static_design",
                )
            )
    return refs


def _evidence(repo_path: Path, file: str, reason: str) -> str:
    path = repo_path / file
    if not path.exists():
        return f"{file} - {reason}"
    return f"{file}:1 - {reason}"


def _rg_files(repo_path: Path, pattern: str) -> list[str]:
    if not pattern or len(pattern) < 3:
        return []
    try:
        result = subprocess.run(
            ["rg", "-l", "--glob", "*.py", "--glob", "*.pyi", "--glob", "*.toml", "--glob", "*.yaml", pattern],
            cwd=repo_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=8,
            check=False,
        )
    except Exception:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _normalize_repo_path(path: str) -> str:
    return path.strip().removeprefix("/testbed/").lstrip("/")


def _is_test_file(path: str) -> bool:
    parts = Path(path).parts
    return "tests" in parts or Path(path).name.startswith("test_") or "/test_" in path


def _is_useful_file(path: str) -> bool:
    if not path:
        return False
    if ".git/" in path or "__pycache__" in path:
        return False
    return path.endswith((".py", ".pyi", ".toml", ".yaml", ".yml"))


def _is_low_signal_file(path: str) -> bool:
    parts = set(Path(path).parts)
    if parts & {"docs", "doc", "examples", "asv_bench", "benchmarks", "web"}:
        return True
    name = Path(path).name
    if name in {"conftest.py", "conf.py", "pyproject.toml"}:
        return True
    return path.endswith((".yaml", ".yml", ".toml"))


def _dedupe_interfaces(items: list[InterfaceSpec]) -> list[InterfaceSpec]:
    by_path: dict[str, InterfaceSpec] = {}
    for item in items:
        existing = by_path.get(item.path)
        if existing is None:
            by_path[item.path] = item
        else:
            by_path[item.path] = InterfaceSpec(
                item.path,
                _dedupe([*existing.symbols, *item.symbols]),
                existing.snippet or item.snippet,
            )
    return list(by_path.values())[:20]


def _dedupe(values: list[str] | Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
