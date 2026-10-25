"""Implementation-only CodeSpec2 replay for controlled model comparisons."""

from __future__ import annotations

import copy
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REFERENCE_AGENT_ROOT = Path(__file__).resolve().parents[1] / "arch_agent_mini_swe"
if str(REFERENCE_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_AGENT_ROOT))

from arch_agent_mini_swe.mini_swe import _safe_model_kwargs_for_log, build_default_model_config, run_mini_stage

from .artifacts import Artifacts, make_artifacts, write_markdown
from .prompts import get_stage_prompt


@dataclass(frozen=True)
class ReplayConfig:
    repo_path: Path
    task: str
    log_root: Path
    reference_artifacts: Path
    task_id: str = "task"
    model_name: str = ""
    model_class: str = "litellm"
    implementation_step_limit: int = 300
    cost_limit: float = 3.0
    timeout: int = 60
    wall_time_limit_seconds: int = 0


@dataclass(frozen=True)
class ReplayResult:
    passed: bool
    log_dir: Path
    artifacts: Artifacts
    trajectory: Path
    summary: dict[str, Any]


class CodeSpec2ImplementationReplayAgent:
    """Run only the implementation stage over a fixed refined spec and test."""

    def __init__(self, config: ReplayConfig) -> None:
        self.config = config
        self.log_dir = self._make_log_dir()
        self.artifacts = make_artifacts(self.log_dir)
        self._copy_reference_artifacts()
        model_name = config.model_name or os.getenv("MSWEA_MODEL_NAME", "")
        if not model_name:
            raise ValueError("Pass --model or set MSWEA_MODEL_NAME for mini-swe-agent.")
        self.model_config = build_default_model_config(model_name, config.model_class)

    def run(self) -> ReplayResult:
        trajectory = self.log_dir / "implement.traj.json"
        config = copy.deepcopy(self.model_config)
        kwargs = dict(config.get("model_kwargs") or {})
        kwargs.update({"temperature": 0.2, "reasoning_effort": "max"})
        config["model_kwargs"] = kwargs
        result = run_mini_stage(
            task=self.config.task, repo_path=self.config.repo_path, output_path=trajectory,
            instance_template=get_stage_prompt("implement").instance_template,
            template_vars={"problem_statement": self.config.task, "spec_path": str(self.artifacts.spec), "spec_test_path": str(self.artifacts.spec_test)},
            model_config=config, step_limit=self.config.implementation_step_limit,
            cost_limit=self.config.cost_limit, timeout=self.config.timeout,
            wall_time_limit_seconds=self.config.wall_time_limit_seconds,
        )
        status = str(result.get("exit_status", ""))
        write_markdown(self.artifacts.summary, f"# CodeSpec2 Implementation Replay Summary\n\n- Implement status: {status}\n- Trajectory: {trajectory}\n")
        return ReplayResult(status == "Submitted", self.log_dir, self.artifacts, trajectory, {"passed": status == "Submitted"})

    def _copy_reference_artifacts(self) -> None:
        for name, destination in (("spec.md", self.artifacts.spec), ("spec_test.py", self.artifacts.spec_test)):
            source = self.config.reference_artifacts / name
            if not source.is_file():
                raise FileNotFoundError(f"Reference artifact is missing: {source}")
            destination.write_bytes(source.read_bytes())

    def _make_log_dir(self) -> Path:
        safe_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in self.config.task_id)
        path = self.config.log_root / f"{safe_id}_{time.strftime('%Y%m%d_%H%M%S')}"
        path.mkdir(parents=True, exist_ok=True)
        return path
