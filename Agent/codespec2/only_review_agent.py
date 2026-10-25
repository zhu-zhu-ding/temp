"""Review and repair a prior CodeSpec2 attempt from an explicit patch."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REFERENCE_AGENT_ROOT = Path(__file__).resolve().parents[1] / "arch_agent_mini_swe"
if str(REFERENCE_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_AGENT_ROOT))

from arch_agent_mini_swe.mini_swe import _safe_model_kwargs_for_log, build_default_model_config, run_mini_stage

from .artifacts import Artifacts, make_artifacts, write_markdown
from .orchestrator import StageResult
from .prompts import get_stage_prompt


@dataclass(frozen=True)
class OnlyReviewConfig:
    repo_path: Path
    task: str
    log_root: Path
    initial_patch: Path
    reference_artifacts: Path
    task_id: str = "task"
    model_name: str = ""
    model_class: str = "litellm"
    review_step_limit: int = 300
    implementation_step_limit: int = 300
    cost_limit: float = 3.0
    timeout: int = 60
    wall_time_limit_seconds: int = 0


@dataclass(frozen=True)
class OnlyReviewResult:
    passed: bool
    log_dir: Path
    artifacts: Artifacts
    stages: list[StageResult]


class CodeSpec2OnlyReviewAgent:
    """Apply a previous model patch, then review its spec and implementation."""

    def __init__(self, config: OnlyReviewConfig) -> None:
        self.config = config
        self.log_dir = self._make_log_dir()
        self.artifacts = make_artifacts(self.log_dir)
        self._stage_counts: dict[str, int] = {}
        self._initialize_reference_artifacts()
        model_name = config.model_name or os.getenv("MSWEA_MODEL_NAME", "")
        if not model_name:
            raise ValueError("Pass --model or set MSWEA_MODEL_NAME for mini-swe-agent.")
        self.model_config = build_default_model_config(model_name, config.model_class)
        self._log(f"initialized task_id={config.task_id} repo={config.repo_path}")

    def run(self) -> OnlyReviewResult:
        self._apply_initial_patch()
        stages: list[StageResult] = []
        for stage, limit in (("only_review_spec", self.config.review_step_limit), ("implement", self.config.implementation_step_limit)):
            stages.append(self._run_stage(stage, limit))
            if stages[-1].exit_status != "Submitted":
                return self._finish(False, stages)
        return self._finish(True, stages)

    def _apply_initial_patch(self) -> None:
        patch = self.config.initial_patch
        if not patch.is_file() or not patch.read_text(encoding="utf-8", errors="replace").strip():
            raise FileNotFoundError(f"Initial model patch is missing or empty: {patch}")
        for command in (("git", "apply", "--check", str(patch)), ("git", "apply", str(patch))):
            completed = subprocess.run(command, cwd=self.config.repo_path, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise RuntimeError(f"Could not initialize repository from {patch}:\n{completed.stdout.strip()}")
        self._log(f"initialized repository from patch={patch}")

    def _initialize_reference_artifacts(self) -> None:
        source = self.config.reference_artifacts
        for name, destination in (("spec.md", self.artifacts.spec), ("spec_test.py", self.artifacts.spec_test)):
            candidate = source / name
            if not candidate.is_file():
                raise FileNotFoundError(f"Reference artifact is missing: {candidate}")
            destination.write_bytes(candidate.read_bytes())

    def _run_stage(self, stage: Literal["only_review_spec", "implement"], step_limit: int) -> StageResult:
        count = self._stage_counts.get(stage, 0) + 1
        self._stage_counts[stage] = count
        trajectory = self.log_dir / (f"{stage}.traj.json" if count == 1 else f"{stage}.{count}.traj.json")
        before = {path: path.read_text(encoding="utf-8", errors="replace") if path.exists() else None for path in self._outputs_for(stage)}
        self._log(f"stage={stage} start")
        result = run_mini_stage(
            task=self.config.task, repo_path=self.config.repo_path, output_path=trajectory,
            instance_template=get_stage_prompt(stage).instance_template,
            template_vars={"problem_statement": self.config.task, "spec_path": str(self.artifacts.spec), "spec_test_path": str(self.artifacts.spec_test)},
            model_config=self._model_config_for_stage(stage), step_limit=step_limit,
            cost_limit=self.config.cost_limit, timeout=self.config.timeout,
            wall_time_limit_seconds=self.config.wall_time_limit_seconds,
        )
        status = str(result.get("exit_status", ""))
        if status == "Submitted":
            invalid = self._invalid_artifacts(stage, before)
            if invalid:
                status = "InvalidArtifacts"
                self._log(f"stage={stage} missing, invalid, or unchanged: {', '.join(invalid)}")
            elif stage == "implement" and self._validate_spec_test():
                status = "SpecTestsFailed"
        self._log(f"stage={stage} done status={status}")
        return StageResult(stage, status, str(result.get("submission", "")), trajectory)

    def _outputs_for(self, stage: str) -> tuple[Path, ...]:
        return (self.artifacts.spec, self.artifacts.spec_test) if stage == "only_review_spec" else ()

    def _invalid_artifacts(self, stage: str, before: dict[Path, str | None]) -> list[str]:
        return [str(path) for path in self._outputs_for(stage) if not path.exists() or not path.read_text(encoding="utf-8", errors="replace").strip() or (before.get(path) is not None and path.read_text(encoding="utf-8", errors="replace") == before[path])]

    def _model_config_for_stage(self, stage: str) -> dict[str, Any]:
        config = copy.deepcopy(self.model_config)
        kwargs = dict(config.get("model_kwargs") or {})
        kwargs.update({"temperature": 0.2, "reasoning_effort": "max"})
        config["model_kwargs"] = kwargs
        self._log(f"stage={stage} model_kwargs={_safe_model_kwargs_for_log(kwargs)}")
        return config

    def _validate_spec_test(self) -> list[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.config.repo_path) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            completed = subprocess.run([sys.executable, str(self.artifacts.spec_test)], cwd=self.config.repo_path, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=max(1, self.config.timeout), check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._log(f"spec_test could not complete: {exc}")
            return [str(self.artifacts.spec_test)]
        if completed.returncode:
            self._log(f"spec_test exit={completed.returncode}\n{completed.stdout[-4000:].strip()}")
            return [str(self.artifacts.spec_test)]
        self._log("spec_test passed")
        return []

    def _finish(self, passed: bool, stages: list[StageResult]) -> OnlyReviewResult:
        content = ["# CodeSpec2 Only-Review Run Summary", "", f"- Task ID: {self.config.task_id}", f"- Repository: {self.config.repo_path}", f"- Initial patch: {self.config.initial_patch}", f"- Passed: {'yes' if passed else 'no'}", "", "## Artifacts"]
        content.extend(f"- {key}: {value}" for key, value in self.artifacts.__dict__.items())
        content.extend(["", "## Stages"])
        content.extend(f"- {item.stage}: {item.exit_status} ({item.trajectory})" for item in stages)
        write_markdown(self.artifacts.summary, "\n".join(content) + "\n")
        return OnlyReviewResult(passed, self.log_dir, self.artifacts, stages)

    def _make_log_dir(self) -> Path:
        safe_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in self.config.task_id)
        path = self.config.log_root / f"{safe_id}_{time.strftime('%Y%m%d_%H%M%S')}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _log(message: str) -> None:
        print(f"[codespec2_only_review] {message}", flush=True)
