from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageArtifacts:
    root: Path
    task_problem_statement: Path
    spec_goal: Path
    design_spec_goal: Path
    test_spec_goal: Path
    impl_status: Path
    goal_decision: Path
    goal_report: Path
    stage_summary: Path


def make_artifacts(log_dir: Path) -> StageArtifacts:
    artifacts = log_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return StageArtifacts(
        root=artifacts,
        task_problem_statement=artifacts / "task_problem_statement.md",
        spec_goal=artifacts / "spec_goal.py",
        design_spec_goal=artifacts / "design_spec_goal.py",
        test_spec_goal=artifacts / "test_spec_goal.py",
        impl_status=artifacts / "impl_status.py",
        goal_decision=artifacts / "goal_decision.json",
        goal_report=artifacts / "goal_report.txt",
        stage_summary=artifacts / "summary.json",
    )


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
