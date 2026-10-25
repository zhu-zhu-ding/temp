from __future__ import annotations

import os
import time
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from arch_agent_mini_swe.mini_swe import _safe_model_kwargs_for_log, build_default_model_config, run_mini_stage

from .artifacts import SimpleArtifacts, make_artifacts, write_json
from .prompts import get_stage_prompt
from .static_design import prepare_static_analysis

StageName = Literal["architecture", "behavior", "design", "test", "implement"]


@dataclass(frozen=True)
class RunConfig:
    repo_path: Path
    task: str
    log_root: Path
    task_id: str = "task"
    model_name: str = ""
    model_class: str = "litellm"
    step_limit: int = 200
    implementation_step_limit: int = 300
    cost_limit: float = 3.0
    timeout: int = 60
    wall_time_limit_seconds: int = 0
    stop_after_stage: str = ""
    ablation: str = ""
    mode: str = "codespec"


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
    artifacts: SimpleArtifacts
    stages: list[StageResult]
    summary: dict[str, Any]


class ArchMiniSweSimpleAgent:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.log_dir = self._make_log_dir()
        self.artifacts = make_artifacts(self.log_dir)
        self._stage_counts: dict[str, int] = {}
        self._initialize_artifacts()
        self._log(f"initialized task_id={config.task_id}")
        self._log(f"repo_path={config.repo_path}")
        self._log(f"log_dir={self.log_dir}")
        self._log(f"mode={config.mode} ablation={config.ablation or 'baseline'}")
        self._log(f"architecture_spec={self.artifacts.architecture_spec}")
        self._log(f"architecture_design={self.artifacts.architecture_design}")
        self._log(f"behavior_design={self.artifacts.behavior_design}")
        self._log(f"p2p_tests={self.artifacts.p2p_tests}")
        self._log(f"f2p_tests={self.artifacts.f2p_tests}")
        self._log(f"static_issues={self.artifacts.static_issues}")
        model_name = config.model_name or os.getenv("MSWEA_MODEL_NAME", "")
        if not model_name:
            raise ValueError(
                "No model configured. Pass --model or set MSWEA_MODEL_NAME "
                "inside the environment running mini-swe-agent."
            )
        self.model_config = build_default_model_config(model_name, config.model_class)
        self._log(f"model={model_name} model_class={config.model_class}")

    def run(self) -> RunResult:
        stages: list[StageResult] = []

        if self.config.mode == "design_document":
            stages_to_run = ["architecture", "behavior", "implement"]
        else:
            stages_to_run = ["static_analysis", "design"]
            if self.config.ablation != "without_generated_tests":
                stages_to_run.append("test")
            stages_to_run.append("implement")
        self._log(f"pipeline started: {' -> '.join(stages_to_run)}")
        if self.config.mode != "design_document":
            stages.append(self._prepare_static_analysis())
        first_stage: StageName = "architecture" if self.config.mode == "design_document" else "design"
        stages.append(self._run_stage(first_stage, self.config.step_limit))

        if stages[-1].exit_status != "Submitted":
            return self._finish(False, stages)

        if self._should_stop(first_stage):
            return self._finish(stages[-1].exit_status == "Submitted", stages)

        if self.config.mode == "design_document":
            stages.append(self._run_stage("behavior", self.config.step_limit))

            if stages[-1].exit_status != "Submitted":
                return self._finish(False, stages)

            if self._should_stop("behavior"):
                return self._finish(True, stages)

        if self.config.mode != "design_document" and self.config.ablation != "without_generated_tests":
            stages.append(self._run_stage("test", self.config.step_limit))

            if stages[-1].exit_status != "Submitted":
                return self._finish(False, stages)

            if self._should_stop("test"):
                return self._finish(stages[-1].exit_status == "Submitted", stages)

        stages.append(self._run_stage("implement", self.config.implementation_step_limit))

        if self._should_stop("implement"):
            return self._finish(stages[-1].exit_status == "Submitted", stages)

        passed = stages[-1].exit_status == "Submitted"
        return self._finish(passed, stages)

    def _run_stage(self, stage: StageName, step_limit: int) -> StageResult:
        prompt = get_stage_prompt(stage, self.config.ablation, self.config.mode)
        count = self._stage_counts.get(stage, 0) + 1
        self._stage_counts[stage] = count
        trajectory_name = f"{stage}.traj.json" if count == 1 else f"{stage}.{count}.traj.json"
        trajectory = self.log_dir / trajectory_name
        self._log(f"stage={stage} start step_limit={step_limit} trajectory={trajectory}")
        template_vars = self._template_vars(stage)
        model_config = self._model_config_for_stage(stage)
        result = run_mini_stage(
            task=self.config.task,
            repo_path=self.config.repo_path,
            output_path=trajectory,
            instance_template=prompt.instance_template,
            template_vars=template_vars,
            model_config=model_config,
            step_limit=step_limit,
            cost_limit=self.config.cost_limit,
            timeout=self.config.timeout,
            wall_time_limit_seconds=self.config.wall_time_limit_seconds,
        )
        exit_status = str(result.get("exit_status", ""))
        if exit_status == "Submitted":
            missing = self._invalid_stage_artifacts(stage)
            if missing:
                exit_status = "InvalidArtifacts"
                self._log(
                    f"stage={stage} rejected: required artifacts were not written "
                    f"to their supplied paths: {', '.join(missing)}"
                )
        self._log(f"stage={stage} done exit_status={exit_status}")
        return StageResult(
            stage=stage,
            exit_status=exit_status,
            submission=str(result.get("submission", "")),
            trajectory=trajectory,
        )

    def _invalid_stage_artifacts(self, stage: StageName) -> list[str]:
        """Reject stages that submitted while their bootstrap artifacts remain."""
        required = {
            "architecture": [self.artifacts.architecture_design],
            "behavior": [self.artifacts.behavior_design],
            "design": [
                self.artifacts.architecture_design
                if self.config.mode == "design_document"
                else self.artifacts.architecture_spec
            ],
            "test": [self.artifacts.p2p_tests, self.artifacts.f2p_tests],
            "implement": [],
        }[stage]
        placeholders = (
            "has not been generated by the Design Agent yet",
            "Not generated yet.",
            "No P2P task reproduction tests generated yet.",
            "No F2P critical-path tests generated yet.",
        )
        invalid: list[str] = []
        for path in required:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                invalid.append(str(path))
                continue
            if not content.strip() or any(marker in content for marker in placeholders):
                invalid.append(str(path))
        return invalid

    def _template_vars(self, stage: StageName) -> dict[str, str]:
        if stage == "architecture":
            return {
                "problem_statement": self.config.task,
                "architecture_design_path": str(self.artifacts.architecture_design),
            }
        if stage == "behavior":
            return {
                "problem_statement": self.config.task,
                "architecture_design_path": str(self.artifacts.architecture_design),
                "behavior_design_path": str(self.artifacts.behavior_design),
            }
        if stage == "design":
            vars = {
                "problem_statement": self.config.task,
                "architecture_design_path": str(self.artifacts.architecture_design),
            }
            if self.config.mode != "design_document":
                vars.update(
                    static_issues=self.artifacts.static_issues.read_text(encoding="utf-8", errors="replace"),
                    architecture_spec_path=str(self.artifacts.architecture_spec),
                )
            return vars
        if stage == "implement":
            if self.config.mode == "design_document":
                return {
                    "problem_statement": self.config.task,
                    "architecture_design_path": str(self.artifacts.architecture_design),
                    "behavior_design_path": str(self.artifacts.behavior_design),
                }
            return {
                "problem_statement": self.config.task,
                "architecture_spec_path": str(self.artifacts.architecture_spec),
                "architecture_design_path": str(self.artifacts.architecture_design),
                "p2p_tests_path": str(self.artifacts.p2p_tests),
                "f2p_tests_path": str(self.artifacts.f2p_tests),
            }
        if stage == "test":
            return {
                "problem_statement": self.config.task,
                "architecture_spec_path": str(self.artifacts.architecture_spec),
                "architecture_design_path": str(self.artifacts.architecture_design),
                "p2p_tests_path": str(self.artifacts.p2p_tests),
                "f2p_tests_path": str(self.artifacts.f2p_tests),
            }
        raise ValueError(f"Unknown stage: {stage}")

    def _model_config_for_stage(self, stage: StageName) -> dict[str, Any]:
        model_config = copy.deepcopy(self.model_config)
        model_kwargs = dict(model_config.get("model_kwargs") or {})
        # The document workflow relies on deliberate architecture and behavior analysis.
        model_kwargs["reasoning_effort"] = "max" if self.config.mode == "design_document" else "medium"
        model_kwargs["temperature"] = 0.2
        model_config["model_kwargs"] = model_kwargs
        self._log(f"stage={stage} model_kwargs={_safe_model_kwargs_for_log(model_kwargs)}")
        return model_config

    def _prepare_static_analysis(self) -> StageResult:
        trajectory = self.log_dir / "design.static.log"
        self._log(f"stage=static_analysis start trajectory={trajectory}")
        result = prepare_static_analysis(self.config.task, self.config.repo_path, self.artifacts)
        trajectory.write_text(result.log, encoding="utf-8")
        self._log(result.log.rstrip())
        self._log("stage=static_analysis done exit_status=Submitted")
        return StageResult("static_analysis", "Submitted", "", trajectory)

    def _finish(self, passed: bool, stages: list[StageResult]) -> RunResult:
        summary = {
            "task_id": self.config.task_id,
            "repo_path": str(self.config.repo_path),
            "passed": passed,
            "log_dir": str(self.log_dir),
            "artifacts": {
                "architecture_spec": str(self.artifacts.architecture_spec),
                "architecture_design": str(self.artifacts.architecture_design),
                "behavior_design": str(self.artifacts.behavior_design),
                "p2p_tests": str(self.artifacts.p2p_tests),
                "f2p_tests": str(self.artifacts.f2p_tests),
                "static_issues": str(self.artifacts.static_issues),
            },
            "stages": [
                {
                    "stage": stage.stage,
                    "exit_status": stage.exit_status,
                    "trajectory": str(stage.trajectory),
                }
                for stage in stages
            ],
        }
        write_json(self.artifacts.stage_summary, summary)
        self._log(f"summary={self.artifacts.stage_summary}")
        self._log(f"run finished passed={passed}")
        return RunResult(passed, self.log_dir, self.artifacts, stages, summary)

    def _should_stop(self, stage: str) -> bool:
        return bool(self.config.stop_after_stage and self.config.stop_after_stage == stage)

    def _make_log_dir(self) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        task_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in self.config.task_id)
        return self.config.log_root / f"{task_id}_{stamp}"

    def _initialize_artifacts(self) -> None:
        if not self.artifacts.architecture_spec.exists():
            self.artifacts.architecture_spec.write_text(_DEFAULT_ARCHITECTURE_SPEC, encoding="utf-8")
        if not self.artifacts.architecture_design.exists():
            self.artifacts.architecture_design.write_text(_DEFAULT_ARCHITECTURE_DESIGN, encoding="utf-8")
        if not self.artifacts.behavior_design.exists():
            self.artifacts.behavior_design.write_text(_DEFAULT_BEHAVIOR_DESIGN, encoding="utf-8")
        if not self.artifacts.p2p_tests.exists():
            self.artifacts.p2p_tests.write_text(_DEFAULT_P2P_TESTS, encoding="utf-8")
        if not self.artifacts.f2p_tests.exists():
            self.artifacts.f2p_tests.write_text(_DEFAULT_F2P_TESTS, encoding="utf-8")
        self.artifacts.static_issues.write_text("", encoding="utf-8")

    def _log(self, message: str) -> None:
        print(f"[arch_swe_agent_simple] {message}", flush=True)


_DEFAULT_ARCHITECTURE_SPEC = """from __future__ import annotations

print("FAIL: architecture_spec.py has not been generated by the Design Agent yet.")
raise SystemExit(1)
"""


_DEFAULT_ARCHITECTURE_DESIGN = """# Architecture Design

Not generated yet.
"""


_DEFAULT_BEHAVIOR_DESIGN = """# Behavior Design

Not generated yet.
"""


_DEFAULT_P2P_TESTS = """from __future__ import annotations

print("No P2P task reproduction tests generated yet.")
"""


_DEFAULT_F2P_TESTS = """from __future__ import annotations

print("No F2P critical-path tests generated yet.")
"""
