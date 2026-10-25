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
from .prompts import get_stage_prompt


@dataclass(frozen=True)
class RunConfig:
    repo_path: Path
    task: str
    log_root: Path
    task_id: str = "task"
    model_name: str = ""
    model_class: str = "litellm"
    spec_experience_step_limit: int = 200
    spec_agent_step_limit: int = 300
    implementation_step_limit: int = 300
    debug_step_limit: int = 300
    cost_limit: float = 3.0
    timeout: int = 60
    wall_time_limit_seconds: int = 0
    stop_after_stage: str = ""


@dataclass(frozen=True)
class StageResult:
    stage: str
    exit_status: str
    submission: str
    trajectory: Path


@dataclass(frozen=True)
class RunResult:
    passed: bool
    log_dir: Path
    artifacts: Artifacts
    stages: list[StageResult]
    summary: dict[str, Any]


class CodeSpec2Agent:
    """Run repository-grounded specification followed by implementation."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.log_dir = self._make_log_dir()
        self.artifacts = make_artifacts(self.log_dir)
        self._stage_counts: dict[str, int] = {}
        self._initialize_artifacts()
        model_name = config.model_name or os.getenv("MSWEA_MODEL_NAME", "")
        if not model_name:
            raise ValueError("Pass --model or set MSWEA_MODEL_NAME for mini-swe-agent.")
        self.model_config = build_default_model_config(model_name, config.model_class)
        self._log(f"initialized task_id={config.task_id} repo={config.repo_path}")
        self._log(f"artifacts={self.artifacts.root} model={model_name}")

    def run(self) -> RunResult:
        stages: list[StageResult] = []
        for stage, limit in (
            ("spec_experience", self.config.spec_experience_step_limit),
            ("spec_agent", self.config.spec_agent_step_limit),
            ("implement", self.config.implementation_step_limit),
            ("debug_agent", self.config.debug_step_limit),
        ):
            stages.append(self._run_stage(stage, limit))
            if stages[-1].exit_status != "Submitted" or self._should_stop(stage):
                return self._finish(stages[-1].exit_status == "Submitted", stages)
        return self._finish(True, stages)

    def _run_stage(
        self,
        stage: Literal["spec_experience", "spec_agent", "implement", "debug_agent"],
        step_limit: int,
    ) -> StageResult:
        count = self._stage_counts.get(stage, 0) + 1
        self._stage_counts[stage] = count
        trajectory = self.log_dir / (f"{stage}.traj.json" if count == 1 else f"{stage}.{count}.traj.json")
        before = {
            path: path.read_text(encoding="utf-8", errors="replace") if path.exists() else None
            for path in self._outputs_for(stage)
        }
        self._log(f"stage={stage} start")
        result = run_mini_stage(
            task=self.config.task,
            repo_path=self.config.repo_path,
            output_path=trajectory,
            instance_template=get_stage_prompt(stage).instance_template,
            template_vars=self._template_vars(),
            model_config=self._model_config_for_stage(stage),
            step_limit=step_limit,
            cost_limit=self.config.cost_limit,
            timeout=self.config.timeout,
            wall_time_limit_seconds=self.config.wall_time_limit_seconds,
        )
        status = str(result.get("exit_status", ""))
        if status == "Submitted":
            invalid = self._invalid_artifacts(stage, before)
            if invalid:
                status = "InvalidArtifacts"
                self._log(f"stage={stage} missing, invalid, or unchanged: {', '.join(invalid)}")
            elif stage in ("implement", "debug_agent"):
                failed_tests = self._validate_spec_test()
                if failed_tests:
                    status = "SpecTestsFailed"
                    self._log(f"stage={stage} spec tests failed: {', '.join(failed_tests)}")
        self._log(f"stage={stage} done status={status}")
        return StageResult(stage, status, str(result.get("submission", "")), trajectory)

    def _outputs_for(self, stage: str) -> tuple[Path, ...]:
        if stage == "spec_experience":
            return (self.artifacts.spec_experience_avoid, self.artifacts.spec_experience_detect)
        if stage == "spec_agent":
            return (self.artifacts.spec, self.artifacts.spec_test)
        if stage == "debug_agent":
            return (self.artifacts.debug_report,)
        return ()

    def _template_vars(self) -> dict[str, str]:
        return {
            "problem_statement": self.config.task,
            "spec_experience_avoid_path": str(self.artifacts.spec_experience_avoid),
            "spec_experience_detect_path": str(self.artifacts.spec_experience_detect),
            "spec_path": str(self.artifacts.spec),
            "spec_test_path": str(self.artifacts.spec_test),
            "debug_report_path": str(self.artifacts.debug_report),
        }

    def _model_config_for_stage(self, stage: str) -> dict[str, Any]:
        config = copy.deepcopy(self.model_config)
        kwargs = dict(config.get("model_kwargs") or {})
        kwargs["temperature"] = 0.2
        kwargs["reasoning_effort"] = "max"
        config["model_kwargs"] = kwargs
        self._log(f"stage={stage} model_kwargs={_safe_model_kwargs_for_log(kwargs)}")
        return config

    def _invalid_artifacts(self, stage: str, before: dict[Path, str | None] | None = None) -> list[str]:
        paths = self._outputs_for(stage)
        invalid = [str(path) for path in paths if not path.exists() or not path.read_text(encoding="utf-8", errors="replace").strip()]
        if before:
            invalid.extend(
                str(path)
                for path, content in before.items()
                if content is not None and path.exists() and path.read_text(encoding="utf-8", errors="replace") == content
            )
        return invalid

    def _validate_spec_test(self) -> list[str]:
        failed: list[str] = []
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.config.repo_path) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        path = self.artifacts.spec_test
        try:
            completed = subprocess.run(
                [sys.executable, str(path)], cwd=self.config.repo_path, env=env,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=max(1, self.config.timeout), check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failed.append(str(path))
            self._log(f"spec_test={path} could not complete: {exc}")
            return failed
        if completed.returncode != 0:
            failed.append(str(path))
            self._log(f"spec_test={path} exit={completed.returncode}\n{completed.stdout[-4000:].strip()}")
        else:
            self._log(f"spec_test={path} passed")
        return failed

    def _initialize_artifacts(self) -> None:
        self.artifacts.spec_experience_avoid.write_text(
            "# Spec Experience - Avoid\n\nNot generated yet.\n", encoding="utf-8"
        )
        self.artifacts.spec_experience_detect.write_text(
            "# Spec Experience - Detect\n\nNot generated yet.\n", encoding="utf-8"
        )
        self.artifacts.spec.write_text("# Requirement Obligation Graph\n\nNot generated yet.\n", encoding="utf-8")
        self.artifacts.spec_test.write_text("print('FAIL: spec_test.py has not been generated.')\nraise SystemExit(1)\n", encoding="utf-8")
        self.artifacts.debug_report.write_text(
            "# Debug Report\n\nNot generated yet.\n", encoding="utf-8"
        )

    def _finish(self, passed: bool, stages: list[StageResult]) -> RunResult:
        summary = {"task_id": self.config.task_id, "repo_path": str(self.config.repo_path), "passed": passed, "log_dir": str(self.log_dir)}
        content = ["# CodeSpec2 Run Summary", "", f"- Task ID: {self.config.task_id}", f"- Repository: {self.config.repo_path}", f"- Passed: {'yes' if passed else 'no'}", f"- Log Directory: {self.log_dir}", "", "## Artifacts"]
        content.extend(f"- {key}: {value}" for key, value in self.artifacts.__dict__.items())
        content.extend(["", "## Stages"])
        content.extend(f"- {item.stage}: {item.exit_status} ({item.trajectory})" for item in stages)
        write_markdown(self.artifacts.summary, "\n".join(content) + "\n")
        return RunResult(passed, self.log_dir, self.artifacts, stages, summary)

    def _make_log_dir(self) -> Path:
        safe_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in self.config.task_id)
        path = self.config.log_root / f"{safe_id}_{time.strftime('%Y%m%d_%H%M%S')}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _should_stop(self, stage: str) -> bool:
        return bool(self.config.stop_after_stage and self.config.stop_after_stage == stage)

    def _log(self, message: str) -> None:
        print(f"[codespec2] {message}", flush=True)
