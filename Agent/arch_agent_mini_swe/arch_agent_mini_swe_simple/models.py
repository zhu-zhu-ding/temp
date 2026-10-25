from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    instruction: str
    repo_path: Path
    task_id: str = "task"
    dataset_path: Path | None = None


@dataclass(frozen=True)
class EntityRef:
    path: str
    name: str
    kind: str
    parent: str = ""
    start_line: int = 0
    end_line: int = 0
    signature: str = ""
    source: str = "problem_statement"
    status: str = "unknown"

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "name": self.name,
            "qualname": self.qualname,
            "kind": self.kind,
            "parent": self.parent,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "source": self.source,
            "status": self.status,
        }


@dataclass(frozen=True)
class CallChainEdge:
    source: str
    target: str
    relation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
        }


@dataclass
class ArchitectureMemory:
    task_id: str
    repo_path: str
    interfaces: list[EntityRef] = field(default_factory=list)
    required_implementations: list[EntityRef] = field(default_factory=list)
    call_chain: list[CallChainEdge] = field(default_factory=list)
    pyflakes_undefined: list[EntityRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "repo_path": self.repo_path,
            "interfaces": [item.to_dict() for item in self.interfaces],
            "required_implementations": [
                item.to_dict() for item in self.required_implementations
            ],
            "call_chain": [edge.to_dict() for edge in self.call_chain],
            "pyflakes_undefined": [item.to_dict() for item in self.pyflakes_undefined],
        }


@dataclass(frozen=True)
class CheckIssue:
    item_id: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "item_id": self.item_id,
            "path": self.path,
            "message": self.message,
        }


@dataclass
class CheckResult:
    passed: bool
    checked: list[str] = field(default_factory=list)
    issues: list[CheckIssue] = field(default_factory=list)
    report: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checked": self.checked,
            "issues": [issue.to_dict() for issue in self.issues],
            "report": self.report,
        }


@dataclass
class TestMemory:
    task_id: str
    repo_path: str
    test_targets: list[EntityRef] = field(default_factory=list)
    tests: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "repo_path": self.repo_path,
            "test_targets": [item.to_dict() for item in self.test_targets],
            "tests": self.tests,
        }


@dataclass
class WorkflowResult:
    passed: bool
    diff: str
    log_dir: Path
    final_report: str
