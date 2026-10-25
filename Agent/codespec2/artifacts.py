from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifacts:
    root: Path
    spec_experience_avoid: Path
    spec_experience_detect: Path
    spec: Path
    spec_test: Path
    debug_report: Path
    summary: Path


def make_artifacts(log_dir: Path) -> Artifacts:
    root = log_dir / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return Artifacts(
        root=root,
        spec_experience_avoid=root / "spec_experience_avoid.md",
        spec_experience_detect=root / "spec_experience_detect.md",
        spec=root / "spec.md",
        spec_test=root / "spec_test.py",
        debug_report=root / "debug_report.md",
        summary=root / "summary.md",
    )


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
