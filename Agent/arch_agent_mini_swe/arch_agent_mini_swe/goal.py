from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoalRun:
    returncode: int
    output: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def run_goal_script(goal_script: Path, repo_path: Path, timeout: int = 120) -> GoalRun:
    if not goal_script.exists():
        return GoalRun(2, f"Goal script does not exist: {goal_script}")
    result = subprocess.run(
        [sys.executable, str(goal_script), "--repo", str(repo_path)],
        cwd=str(repo_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return GoalRun(result.returncode, result.stdout)
