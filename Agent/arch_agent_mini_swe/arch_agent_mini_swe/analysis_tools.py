from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import read_json, write_json

EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    ".venv",
    "venv",
    "libs",
    "docs",
}


@dataclass(frozen=True)
class InterfaceSpec:
    path: str
    symbols: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analysis helpers for arch_agent_mini_swe.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace_parser = subparsers.add_parser("trace", help="Collect related tests for design-pattern analysis.")
    trace_parser.add_argument("--repo", type=Path, required=True)
    trace_parser.add_argument("--task-file", type=Path, required=True)
    trace_parser.add_argument("--output", type=Path, required=True)
    trace_parser.add_argument("--timeout", type=int, default=180)
    trace_parser.add_argument("--max-tests", type=int, default=8)

    static_parser = subparsers.add_parser("static-goals", help="Build static goal candidates.")
    static_parser.add_argument("--repo", type=Path, required=True)
    static_parser.add_argument("--task-file", type=Path, required=True)
    static_parser.add_argument("--design-patterns", type=Path, required=True)
    static_parser.add_argument("--trace", type=Path, required=True)
    static_parser.add_argument("--output", type=Path, required=True)
    static_parser.add_argument("--max-files", type=int, default=300)

    args = parser.parse_args()
    if args.command == "trace":
        collect_trace(args.repo, args.task_file, args.output, timeout=args.timeout, max_tests=args.max_tests)
        return 0
    if args.command == "static-goals":
        build_static_goals(
            args.repo,
            args.task_file,
            args.design_patterns,
            args.trace,
            args.output,
            max_files=args.max_files,
        )
        return 0
    raise AssertionError(args.command)


def collect_trace(repo: Path, task_file: Path, output: Path, *, timeout: int, max_tests: int) -> None:
    repo = repo.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    task = task_file.read_text(encoding="utf-8")
    interfaces = parse_interfaces(task)
    related_tests = find_related_tests(repo, interfaces, max_tests=max_tests)
    commands: list[dict[str, Any]] = []

    if related_tests:
        command = build_related_test_command(related_tests)
        result = run_command(command, repo, timeout=timeout)
        commands.append({"command": command, **result})
    else:
        commands.append(
            {
                "command": "",
                "returncode": 2,
                "output": "No related tests found from interface paths or symbols.",
            }
        )

    data = {
        "interfaces": [asdict(item) for item in interfaces],
        "related_tests": related_tests,
        "related_test_surfaces": [
            {
                "file": path,
                "reason": "test file selected by task interface path/symbol keyword search",
            }
            for path in related_tests
        ],
        "commands": commands,
        "traced_files": [],
        "dynamic_surfaces": [],
        "trace_mode": "related_tests_only",
    }
    write_json(output, data)


def build_static_goals(
    repo: Path,
    task_file: Path,
    design_patterns: Path,
    trace: Path,
    output: Path,
    *,
    max_files: int,
) -> None:
    repo = repo.resolve()
    task = task_file.read_text(encoding="utf-8")
    interfaces = parse_interfaces(task)
    patterns = read_json(design_patterns, {})
    trace_data = read_json(trace, {})

    target_files = {item.path for item in interfaces if item.path.endswith(".py")}
    target_symbols = {symbol for item in interfaces for symbol in item.symbols}
    related_tests = set(trace_data.get("related_tests", []))
    keyword_terms = build_search_terms(interfaces)
    python_files = find_relevant_python_files(
        repo,
        interfaces,
        related_tests=related_tests,
        terms=keyword_terms,
        max_files=max_files,
    )
    symbols = collect_symbols(repo, python_files)

    related_files: dict[str, set[str]] = {path: set() for path in target_files}
    symbol_hits: dict[str, set[str]] = {symbol: set() for symbol in target_symbols}
    for path in python_files:
        rel = path.relative_to(repo).as_posix()
        text = safe_read(path)
        for target_file in target_files:
            module = module_name_for_path(target_file)
            if module in text or target_file in text or Path(target_file).name in text:
                related_files.setdefault(target_file, set()).add(rel)
        for symbol in target_symbols:
            if re.search(rf"\b{re.escape(symbol)}\b", text):
                symbol_hits.setdefault(symbol, set()).add(rel)

    static_surfaces = []
    for target_file, refs in sorted(related_files.items()):
        static_surfaces.append(
            {
                "file": target_file,
                "kind": "target_interface_file",
                "reason": "declared by task interface path",
            }
        )
        for ref in sorted(refs):
            if ref != target_file:
                static_surfaces.append(
                    {
                        "file": ref,
                        "kind": "import_or_reference",
                        "reason": f"imports or references {target_file}",
                    }
                )

    for symbol, refs in sorted(symbol_hits.items()):
        for ref in sorted(refs):
            static_surfaces.append(
                {
                    "file": ref,
                    "kind": "symbol_reference",
                    "reason": f"contains symbol {symbol}",
                }
            )

    related_test_files = {
        item.get("file") for item in trace_data.get("related_test_surfaces", []) if item.get("file")
    }
    static_files = {item["file"] for item in static_surfaces}
    data = {
        "interfaces": [asdict(item) for item in interfaces],
        "symbols": sorted(target_symbols),
        "analysis_scope": {
            "mode": "keyword_relevant_files",
            "terms": sorted(keyword_terms),
            "files_analyzed": sorted(path.relative_to(repo).as_posix() for path in python_files),
            "max_files": max_files,
            "note": "Static goals are based on task interface paths, related tests, and keyword grep hits, not a full-repository scan.",
        },
        "static_surfaces": static_surfaces,
        "related_test_surfaces": trace_data.get("related_test_surfaces", []),
        "dynamic_surfaces": [],
        "high_confidence_surfaces": [
            {
                "file": path,
                "reason": "found by both static reference search and related-test selection",
            }
            for path in sorted(static_files & related_test_files)
        ],
        "static_only_surfaces": [
            {"file": path, "reason": "static reference without related-test overlap"}
            for path in sorted(static_files - related_test_files)
        ],
        "related_test_only_surfaces": [
            {"file": path, "reason": "related test selected but no direct static reference found"}
            for path in sorted(related_test_files - static_files)
        ],
        "static_issue_goals": build_static_issue_goals(interfaces, symbols),
        "architecture_goal_inputs": {
            "design_patterns_path": str(design_patterns),
            "has_design_patterns": bool(patterns),
            "trace_path": str(trace),
            "task_goal_source": (
                "Do not derive task goals from this helper. The design-goal agent must read "
                "the task text/code signatures/docstrings/examples directly and convert key "
                "interface behavior into functional task_goals."
            ),
        },
    }
    write_json(output, data)


def parse_interfaces(task: str) -> list[InterfaceSpec]:
    paths = [normalize_task_path(path) for path in re.findall(r"Path:\s*`([^`]+)`", task)]
    code_blocks = re.findall(r"```python\n(.*?)```", task, flags=re.S)
    block_symbols = [extract_symbols_from_code(block) for block in code_blocks]
    interfaces: list[InterfaceSpec] = []
    for i, path in enumerate(paths):
        symbols = block_symbols[i] if i < len(block_symbols) else []
        interfaces.append(InterfaceSpec(path=path, symbols=symbols))
    if not interfaces:
        interfaces.append(InterfaceSpec(path="", symbols=sorted({s for b in block_symbols for s in b})))
    return interfaces


def extract_symbols_from_code(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return sorted(set(re.findall(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_]\w*)", code, flags=re.M)))
    symbols = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
    return sorted(set(symbols))


def normalize_task_path(path: str) -> str:
    path = path.strip()
    if path.startswith("/testbed/"):
        return path[len("/testbed/") :]
    return path.lstrip("/")


def find_related_tests(repo: Path, interfaces: list[InterfaceSpec], *, max_tests: int) -> list[str]:
    tests = [
        path
        for path in iter_python_files(repo)
        if is_test_file(path, repo) and not is_excluded_rel(path.relative_to(repo).as_posix())
    ]
    terms = build_search_terms(interfaces)
    scored: list[tuple[int, str]] = []
    for path in tests:
        rel = path.relative_to(repo).as_posix()
        text = safe_read(path)
        score = sum(1 for term in terms if term and (term in rel or re.search(rf"\b{re.escape(term)}\b", text)))
        if score:
            scored.append((score, rel))
    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    return [path for _, path in scored[:max_tests]]


def build_search_terms(interfaces: list[InterfaceSpec]) -> set[str]:
    terms: set[str] = set()
    for interface in interfaces:
        if interface.path:
            path = Path(interface.path)
            parts = [part for part in path.parts if part not in {"src", "lib", "python"}]
            terms.update(part for part in parts[-4:] if part and part != "__init__.py")
            terms.add(path.stem)
            terms.add(module_name_for_path(interface.path))
        terms.update(interface.symbols)
    return {term for term in terms if term and len(term) > 1}


def find_relevant_python_files(
    repo: Path,
    interfaces: list[InterfaceSpec],
    *,
    related_tests: set[str],
    terms: set[str],
    max_files: int,
) -> list[Path]:
    selected: dict[str, Path] = {}

    def add_rel(rel: str) -> None:
        if not rel or is_excluded_rel(rel):
            return
        path = repo / rel
        if path.exists() and path.suffix == ".py":
            selected[rel] = path

    for interface in interfaces:
        add_rel(interface.path)
        if interface.path:
            path = Path(interface.path)
            for sibling in (repo / path.parent).glob("*.py"):
                rel = sibling.relative_to(repo).as_posix()
                add_rel(rel)

    for rel in related_tests:
        add_rel(rel)
        path = Path(rel)
        for sibling in (repo / path.parent).glob("*.py"):
            add_rel(sibling.relative_to(repo).as_posix())

    scored: list[tuple[int, str, Path]] = []
    for path in iter_python_files(repo):
        rel = path.relative_to(repo).as_posix()
        if rel in selected or is_excluded_rel(rel):
            continue
        text = safe_read(path)
        score = 0
        for term in terms:
            if term in rel:
                score += 3
            if re.search(rf"\b{re.escape(term)}\b", text):
                score += 1
        if score:
            scored.append((score, rel, path))
    scored.sort(key=lambda item: (-item[0], len(item[1]), item[1]))

    for _, rel, path in scored:
        if len(selected) >= max_files:
            break
        selected[rel] = path

    return [selected[rel] for rel in sorted(selected)]


def build_related_test_command(test_files: list[str]) -> str:
    pytest_args = " ".join(sh_quote(path) for path in test_files)
    return f"{sh_quote(sys.executable)} -m pytest --collect-only -q {pytest_args}"


def run_command(command: str, cwd: Path, *, timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {"returncode": completed.returncode, "output": tail(completed.stdout)}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return {"returncode": 124, "output": tail(output), "timeout": timeout}


def collect_symbols(repo: Path, python_files: list[Path]) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for path in python_files:
        rel = path.relative_to(repo).as_posix()
        try:
            tree = ast.parse(safe_read(path))
        except SyntaxError:
            results[rel] = []
            continue
        entries = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                entries.append(
                    {
                        "name": node.name,
                        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                        "line": node.lineno,
                    }
                )
        results[rel] = entries
    return results


def build_static_issue_goals(
    interfaces: list[InterfaceSpec], symbols_by_file: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    goals = []
    for interface in interfaces:
        if not interface.path:
            continue
        existing = {item["name"] for item in symbols_by_file.get(interface.path, [])}
        for symbol in interface.symbols:
            if symbol not in existing:
                goals.append(
                    {
                        "goal": f"Static issue: missing symbol {symbol} in {interface.path}",
                        "check": "missing_symbol",
                        "path": interface.path,
                        "symbol": symbol,
                        "guidance": "Implement the symbol at the task-specified path or explain the repository-specific equivalent.",
                    }
                )
    return goals


def iter_python_files(repo: Path) -> list[Path]:
    result: list[Path] = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for filename in files:
            if filename.endswith(".py"):
                result.append(Path(root) / filename)
    return sorted(result)


def is_excluded_rel(rel: str) -> bool:
    parts = set(Path(rel).parts)
    return bool(parts & EXCLUDE_DIRS)


def is_test_file(path: Path, repo: Path) -> bool:
    rel = path.relative_to(repo).as_posix()
    name = path.name
    return rel.startswith(("test/", "tests/")) or name.startswith("test_") or name.endswith("_test.py")


def module_name_for_path(path: str) -> str:
    return Path(path).with_suffix("").as_posix().replace("/", ".")


def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")
    except OSError:
        return ""


def tail(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...<truncated>...\n" + text[-limit // 2 :]


def sh_quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


if __name__ == "__main__":
    raise SystemExit(main())
