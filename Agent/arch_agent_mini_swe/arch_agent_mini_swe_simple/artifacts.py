from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimpleArtifacts:
    root: Path
    architecture_spec: Path
    architecture_design: Path
    behavior_design: Path
    p2p_tests: Path
    f2p_tests: Path
    static_issues: Path
    stage_summary: Path


def make_artifacts(log_dir: Path) -> SimpleArtifacts:
    artifacts = log_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return SimpleArtifacts(
        root=artifacts,
        architecture_spec=artifacts / "architecture_spec.py",
        architecture_design=artifacts / "architecture_design.md",
        behavior_design=artifacts / "behavior_design.md",
        p2p_tests=artifacts / "p2p_tests.py",
        f2p_tests=artifacts / "f2p_tests.py",
        static_issues=artifacts / "static_issues.md",
        stage_summary=artifacts / "summary.json",
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
