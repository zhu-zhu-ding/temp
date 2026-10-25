from __future__ import annotations
import ast
import builtins
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from .models import EntityRef

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
}
BUILTIN_NAMES = set(dir(builtins)) | {
    "self",
    "cls",
    "super",
    "pytest",
    "np",
    "pd",
}

@dataclass(frozen=True)
class ImportAlias:
    alias: str
    module: str
    name: str = ""
    level: int = 0

@dataclass
class Symbol:
    path: str
    name: str
    kind: str
    parent: str = ""
    start_line: int = 0
    end_line: int = 0
    signature: str = ""
    decorators: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    status: str = "present"
    @property
    def qualname(self) -> str:
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name
    @property
    def id(self) -> str:
        if self.parent:
            return f"{self.path}::{self.parent}::{self.name}"
        return f"{self.path}::{self.name}"
    def to_entity(self, *, source: str = "static_index", status: str | None = None) -> EntityRef:
        return EntityRef(
            path=self.path,
            name=self.name,
            kind=self.kind,
            parent=self.parent,
            start_line=self.start_line,
            end_line=self.end_line,
            signature=self.signature,
            source=source,
            status=status or self.status,
        )

@dataclass
class FileModel:
    path: str
    imports: list[ImportAlias] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    module_level_calls: list[str] = field(default_factory=list)
    syntax_error: str = ""

@dataclass
class RepoIndex:
    repo_path: Path
    files: dict[str, FileModel]
    symbols_by_id: dict[str, Symbol]
    symbols_by_name: dict[str, list[Symbol]]
    symbols_by_path_name: dict[tuple[str, str], list[Symbol]]
    module_to_path: dict[str, str]
    def find_symbol(
        self,
        path: str,
        name: str,
        parent: str = "",
        *,
        prefer_exact_parent: bool = True,
    ) -> Symbol | None:
        path = normalize_rel_path(path)
        candidates = self.symbols_by_path_name.get((path, name), [])
        if prefer_exact_parent:
            for symbol in candidates:
                if symbol.parent == parent:
                    return symbol
        if parent:
            for symbol in candidates:
                if symbol.parent.endswith(parent) or symbol.qualname == f"{parent}.{name}":
                    return symbol
        return candidates[0] if candidates else None
    def symbols_in_file(self, path: str) -> list[Symbol]:
        model = self.files.get(normalize_rel_path(path))
        return model.symbols if model else []
    def files_for_roots(self, roots: Iterable[str]) -> set[str]:
        result: set[str] = set()
        for root in roots:
            rel = normalize_rel_path(root)
            if not rel:
                continue
            if rel in self.files:
                result.add(rel)
                continue
            if rel.endswith(".py"):
                result.add(rel)
                continue
            module_path = self.module_to_path.get(rel)
            if module_path:
                result.add(module_path)
        return result
    def resolve_call(self, call: str, source: Symbol | None = None, source_path: str = "") -> Symbol | None:
        source_path = normalize_rel_path(source_path or (source.path if source else ""))
        if not call:
            return None
        leaf = call.split(".")[-1]
        if source is not None:
            model = self.files.get(source.path)
            if model:
                imports = resolve_imports(model, self.module_to_path, source.path)
                imported = _resolve_imported_call(call, imports, self)
                if imported is not None:
                    return imported
                if "." in call:
                    owner, method = call.rsplit(".", 1)
                    if owner in {"self", "cls"} and source.parent:
                        found = self.find_symbol(source.path, method, source.parent)
                        if found:
                            return found
                found = self.find_symbol(source.path, leaf, source.parent)
                if found:
                    return found
                found = self.find_symbol(source.path, leaf, "")
                if found:
                    return found
        model = self.files.get(source_path)
        if model:
            imports = resolve_imports(model, self.module_to_path, source_path)
            imported = _resolve_imported_call(call, imports, self)
            if imported is not None:
                return imported
        same_file = self.symbols_by_path_name.get((source_path, leaf), [])
        if same_file:
            return same_file[0]
        all_named = self.symbols_by_name.get(leaf, [])
        if len(all_named) == 1:
            return all_named[0]
        return None

def build_repo_index(
    repo_path: Path,
    *,
    max_files: int = 5000,
    max_depth: int = 5,
    only_files: Iterable[str] | None = None,
) -> RepoIndex:
    repo_path = repo_path.resolve()
    py_files = iter_python_files(repo_path, max_files=max_files)
    all_module_to_path: dict[str, str] = {}
    for path in py_files:
        rel = path.relative_to(repo_path).as_posix()
        all_module_to_path[module_name_for_path(Path(rel))] = rel
    if only_files is not None:
        wanted = {normalize_rel_path(path) for path in only_files if str(path).strip()}
        py_files = [path for path in py_files if path.relative_to(repo_path).as_posix() in wanted]
    files: dict[str, FileModel] = {}
    for path in py_files:
        rel = path.relative_to(repo_path).as_posix()
        files[rel] = parse_file(repo_path, path, max_depth=max_depth)
    symbols_by_id: dict[str, Symbol] = {}
    symbols_by_name: dict[str, list[Symbol]] = defaultdict(list)
    symbols_by_path_name: dict[tuple[str, str], list[Symbol]] = defaultdict(list)
    for model in files.values():
        for symbol in model.symbols:
            symbols_by_id[symbol.id] = symbol
            symbols_by_name[symbol.name].append(symbol)
            symbols_by_path_name[(symbol.path, symbol.name)].append(symbol)
    return RepoIndex(
        repo_path=repo_path,
        files=files,
        symbols_by_id=symbols_by_id,
        symbols_by_name=dict(symbols_by_name),
        symbols_by_path_name=dict(symbols_by_path_name),
        module_to_path=all_module_to_path,
    )

def iter_python_files(repo_path: Path, *, max_files: int = 5000) -> list[Path]:
    result: list[Path] = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for filename in files:
            if not filename.endswith(".py"):
                continue
            path = Path(root) / filename
            result.append(path)
            if len(result) >= max_files:
                return sorted(result)
    return sorted(result)

def parse_file(repo_path: Path, path: Path, *, max_depth: int = 5) -> FileModel:
    rel = path.relative_to(repo_path).as_posix()
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    model = FileModel(path=rel)
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        model.syntax_error = f"{exc.msg} at line {exc.lineno}"
        return model
    try:
        model.imports = _collect_imports(tree, max_depth=max_depth)
    except RecursionError:
        model.syntax_error = "AST traversal exceeded recursion depth while collecting imports"
        return model
    lines = text.splitlines()
    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: list[str] = []
            self.current_symbol: Symbol | None = None
            self.depth = 0
        def _too_deep(self) -> bool:
            return self.depth >= max_depth
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if self._too_deep():
                return
            parent = ".".join(self.parents)
            symbol = Symbol(
                path=rel,
                name=node.name,
                kind="class",
                parent=parent,
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                signature=f"class {node.name}",
                decorators=[],
                status=_body_status(node.body),
            )
            model.symbols.append(symbol)
            self.parents.append(node.name)
            self.depth += 1
            previous = self.current_symbol
            self.current_symbol = symbol
            for child in node.body:
                self.visit(child)
            self.current_symbol = previous
            self.depth -= 1
            self.parents.pop()
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node, "method" if self.parents else "function")
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node, "method" if self.parents else "function")
        def _visit_function(self, node: ast.AST, default_kind: str) -> None:
            if self._too_deep():
                return
            assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            decorators = [_expr_name(item) for item in node.decorator_list]
            kind = "property" if any(d.endswith("property") for d in decorators) else default_kind
            parent = ".".join(self.parents)
            symbol = Symbol(
                path=rel,
                name=node.name,
                kind=kind,
                parent=parent,
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                signature=_signature_from_lines(lines, node),
                decorators=decorators,
                status=_body_status(node.body),
            )
            collector = _CallCollector(max_depth=max_depth)
            for child in node.body:
                collector.visit(child)
            symbol.calls = sorted(collector.calls)
            model.symbols.append(symbol)
            previous = self.current_symbol
            self.current_symbol = symbol
            self.parents.append(node.name)
            self.depth += 1
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self.visit(child)
            self.parents.pop()
            self.depth -= 1
            self.current_symbol = previous
        def visit_Call(self, node: ast.Call) -> None:
            if self.current_symbol is None:
                name = _expr_name(node.func)
                if name:
                    model.module_level_calls.append(name)
            self.generic_visit(node)
    try:
        Visitor().visit(tree)
    except RecursionError:
        model.syntax_error = "AST traversal exceeded recursion depth while collecting symbols"
    model.module_level_calls = sorted(set(model.module_level_calls))
    return model

def resolve_imports(
    model: FileModel,
    module_to_path: dict[str, str],
    current_path: str,
) -> dict[str, str]:
    imports: dict[str, str] = {}
    current_module = module_name_for_path(Path(current_path))
    package = current_module.rsplit(".", 1)[0] if "." in current_module else ""
    for item in model.imports:
        if item.level:
            base = _resolve_relative_module(package, item.module, item.level)
        else:
            base = item.module
        if not base:
            continue
        target_path = module_to_path.get(base)
        if item.name:
            symbol_path = module_to_path.get(f"{base}.{item.name}", target_path)
            if symbol_path:
                imports[item.alias] = f"{symbol_path}::{item.name}"
        elif target_path:
            imports[item.alias] = target_path
        else:
            parts = base.split(".")
            for i in range(len(parts), 0, -1):
                prefix = ".".join(parts[:i])
                prefix_path = module_to_path.get(prefix)
                if prefix_path:
                    imports[item.alias] = prefix_path
                    break
    return imports

def module_name_for_path(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)

def normalize_rel_path(path: str) -> str:
    path = str(path).replace("\\", "/")
    if path.startswith("/testbed/"):
        path = path[len("/testbed/") :]
    return path.lstrip("/")

def entity_from_missing_name(path: str, name: str, *, source: str = "pyflakes") -> EntityRef:
    kind = "class_or_function"
    if name and name[0].isupper():
        kind = "class"
    return EntityRef(
        path=normalize_rel_path(path),
        name=name,
        kind=kind,
        source=source,
        status="missing",
    )

class _CallCollector(ast.NodeVisitor):
    def __init__(self, *, max_depth: int = 5) -> None:
        self.calls: set[str] = set()
        self.max_depth = max_depth
        self.depth = 0
    def visit_Call(self, node: ast.Call) -> None:
        if self.depth >= self.max_depth:
            return
        name = _expr_name(node.func)
        if name and (("." in name) or name not in BUILTIN_NAMES):
            self.calls.add(name)
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

def _collect_imports(tree: ast.AST, *, max_depth: int = 5) -> list[ImportAlias]:
    imports: list[ImportAlias] = []
    for node in _iter_nodes_limited(tree, max_depth=max_depth):
        if isinstance(node, ast.Import):
            for alias in node.names:
                alias_name = alias.asname or alias.name.split(".", 1)[0]
                imports.append(ImportAlias(alias=alias_name, module=alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                alias_name = alias.asname or alias.name
                imports.append(
                    ImportAlias(
                        alias=alias_name,
                        module=module,
                        name=alias.name,
                        level=node.level,
                    )
                )
    return imports

def _iter_nodes_limited(tree: ast.AST, *, max_depth: int) -> Iterable[ast.AST]:
    stack: list[tuple[ast.AST, int]] = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        yield node
        if depth >= max_depth:
            continue
        children = list(ast.iter_child_nodes(node))
        for child in reversed(children):
            stack.append((child, depth + 1))

def _expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_name(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Call):
        return _expr_name(node.func)
    if isinstance(node, ast.Subscript):
        return _expr_name(node.value)
    return ""

def _signature_from_lines(lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    start = max(node.lineno - 1, 0)
    end = max(getattr(node.args, "end_lineno", node.lineno) - 1, start)
    collected: list[str] = []
    for line in lines[start : end + 1]:
        collected.append(line.strip())
        if line.rstrip().endswith(":"):
            break
    return " ".join(part for part in collected if part)

def _body_status(body: Iterable[ast.stmt]) -> str:
    body = list(body)
    meaningful = [
        stmt
        for stmt in body
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(getattr(stmt, "value", None), ast.Constant)
            and isinstance(stmt.value.value, str)
        )
    ]
    if not meaningful:
        return "missing"
    if all(isinstance(stmt, (ast.Pass, ast.Expr)) for stmt in meaningful):
        for stmt in meaningful:
            if isinstance(stmt, ast.Expr):
                value = getattr(stmt, "value", None)
                if isinstance(value, ast.Constant) and value.value is Ellipsis:
                    continue
                return "present"
        return "missing"
    if all(isinstance(stmt, ast.Raise) for stmt in meaningful):
        for stmt in meaningful:
            exc = getattr(stmt, "exc", None)
            if isinstance(exc, ast.Call):
                name = _expr_name(exc.func)
            else:
                name = _expr_name(exc) if exc is not None else ""
            if name.endswith(("NotImplementedError", "NotImplemented")):
                continue
            return "present"
        return "missing"
    return "present"

def _resolve_imported_call(
    call: str,
    imports: dict[str, str],
    index: RepoIndex,
) -> Symbol | None:
    head, _, tail = call.partition(".")
    target = imports.get(head)
    if not target:
        return None
    if "::" in target:
        path, imported_name = target.split("::", 1)
        if tail:
            found = index.find_symbol(path, tail, imported_name, prefer_exact_parent=False)
            if found is not None:
                return found
            # Symbol not found in the target file (may have been deleted by mask).
            # Return a synthetic MISSING symbol so resolve_call doesn't lose the
            # dependency edge.
            if (index.repo_path / path).exists():
                return Symbol(
                    path=normalize_rel_path(path), name=tail, kind="method",
                    parent=imported_name, status="missing",
                )
        else:
            found = index.find_symbol(path, imported_name)
            if found is not None:
                return found
            if (index.repo_path / path).exists():
                return Symbol(
                    path=normalize_rel_path(path), name=imported_name,
                    kind="class_or_function", status="missing",
                )
    if tail:
        found = index.find_symbol(target, tail)
        if found is not None:
            return found
        if (index.repo_path / target).exists():
            return Symbol(
                path=normalize_rel_path(target), name=tail, kind="function",
                status="missing",
            )
    return None

def _resolve_relative_module(package: str, module: str, level: int) -> str:
    parts = package.split(".") if package else []
    if level > 1:
        parts = parts[: -(level - 1)] if level - 1 <= len(parts) else []
    if module:
        parts.extend(module.split("."))
    return ".".join(part for part in parts if part)
