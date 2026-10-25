from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .artifacts import StageArtifacts, make_artifacts, read_json, write_json
from .goal import GoalRun, run_goal_script
from .mini_swe import build_default_model_config, run_mini_stage
from .prompts import STAGE_PROMPTS

StageName = Literal["goal", "design", "test", "implement"]


@dataclass(frozen=True)
class RunConfig:
    repo_path: Path
    task: str
    log_root: Path
    task_id: str = "task"
    model_name: str = ""
    model_class: str = "litellm"
    step_limit: int = 80
    implementation_step_limit: int = 200
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
    loop_index: int


@dataclass(frozen=True)
class RunResult:
    passed: bool
    log_dir: Path
    artifacts: StageArtifacts
    stages: list[StageResult]
    goal_run: GoalRun | None
    summary: dict[str, Any]


class ArchMiniSweAgent:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.log_dir = self._make_log_dir()
        self.artifacts = make_artifacts(self.log_dir)
        self.artifacts.task_problem_statement.write_text(config.task, encoding="utf-8")
        self.max_loops = max(1, int(os.getenv("ARCH_SWE_AGENT_MAX_LOOPS", "4")))
        self._initialize_spec_scripts()
        self._log(f"initialized task_id={config.task_id}")
        self._log(f"repo_path={config.repo_path}")
        self._log(f"log_dir={self.log_dir}")
        self._log(f"spec_goal={self.artifacts.spec_goal}")
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
        goal_run: GoalRun | None = None
        passed = False

        self._log("goal-driven loop started")
        for loop_index in range(1, self.max_loops + 1):
            self._log(f"loop={loop_index}/{self.max_loops} running central spec")
            goal_run = self._run_central_spec()
            self.artifacts.goal_report.write_text(goal_run.output, encoding="utf-8")
            self._log(f"loop={loop_index} central spec returncode={goal_run.returncode}")

            stages.append(self._run_goal_agent(loop_index, goal_run))
            if self._should_stop("goal"):
                return self._finish(False, stages, goal_run)

            decision = self._read_decision()
            if goal_run.passed and bool(decision.get("task_satisfied")):
                self._log("goal agent marked task satisfied and central spec passed")
                passed = True
                break

            ran_agent = False
            if bool(decision.get("call_design_agent")):
                stages.append(self._run_child_agent("design", loop_index, str(decision.get("design_objective", ""))))
                ran_agent = True
                if self._should_stop("design"):
                    return self._finish(False, stages, goal_run)
            if bool(decision.get("call_test_agent")):
                stages.append(self._run_child_agent("test", loop_index, str(decision.get("test_objective", ""))))
                ran_agent = True
                if self._should_stop("test"):
                    return self._finish(False, stages, goal_run)
            if bool(decision.get("call_implement_agent")):
                stages.append(
                    self._run_child_agent(
                        "implement",
                        loop_index,
                        str(decision.get("implement_objective", "")),
                    )
                )
                ran_agent = True
                if self._should_stop("implement"):
                    goal_run = self._run_central_spec()
                    return self._finish(goal_run.passed, stages, goal_run)

            if not ran_agent:
                self._log("goal agent selected no child agent; stopping to avoid idle loop")
                break

        goal_run = self._run_central_spec()
        self.artifacts.goal_report.write_text(goal_run.output, encoding="utf-8")
        passed = passed or goal_run.passed
        self._log(f"final central spec returncode={goal_run.returncode}")
        return self._finish(passed, stages, goal_run)

    def _run_goal_agent(self, loop_index: int, goal_run: GoalRun) -> StageResult:
        self._log(f"loop={loop_index} goal agent start")
        return self._run_stage(
            "goal",
            loop_index,
            {
                "loop_index": loop_index,
                "max_loops": self.max_loops,
                "spec_goal_path": str(self.artifacts.spec_goal),
                "design_spec_goal_path": str(self.artifacts.design_spec_goal),
                "test_spec_goal_path": str(self.artifacts.test_spec_goal),
                "impl_status_path": str(self.artifacts.impl_status),
                "goal_decision_path": str(self.artifacts.goal_decision),
                "goal_report_path": str(self.artifacts.goal_report),
                "goal_report": goal_run.output or "<no report>",
            },
            self.config.step_limit,
        )

    def _run_child_agent(self, stage: StageName, loop_index: int, objective: str) -> StageResult:
        self._log(f"loop={loop_index} {stage} agent start")
        step_limit = self.config.implementation_step_limit if stage == "implement" else self.config.step_limit
        return self._run_stage(
            stage,
            loop_index,
            {
                "agent_objective": objective or "Use the task and current executable spec goals to make progress.",
                "spec_goal_path": str(self.artifacts.spec_goal),
                "design_spec_goal_path": str(self.artifacts.design_spec_goal),
                "test_spec_goal_path": str(self.artifacts.test_spec_goal),
                "impl_status_path": str(self.artifacts.impl_status),
            },
            step_limit,
        )

    def _run_stage(
        self,
        stage: StageName,
        loop_index: int,
        template_vars: dict[str, Any],
        step_limit: int,
    ) -> StageResult:
        prompt = STAGE_PROMPTS[stage]
        trajectory = self.log_dir / f"loop_{loop_index:02d}_{stage}.traj.json"
        self._log(f"stage={stage} loop={loop_index} step_limit={step_limit} trajectory={trajectory}")
        result = run_mini_stage(
            task=self.config.task,
            repo_path=self.config.repo_path,
            output_path=trajectory,
            instance_template=prompt.instance_template,
            template_vars=template_vars,
            model_config=self.model_config,
            step_limit=step_limit,
            cost_limit=self.config.cost_limit,
            timeout=self.config.timeout,
            wall_time_limit_seconds=self.config.wall_time_limit_seconds,
        )
        self._log(f"stage={stage} loop={loop_index} done exit_status={result.get('exit_status', '')}")
        return StageResult(
            stage=stage,
            exit_status=str(result.get("exit_status", "")),
            submission=str(result.get("submission", "")),
            trajectory=trajectory,
            loop_index=loop_index,
        )

    def _run_central_spec(self) -> GoalRun:
        return run_goal_script(self.artifacts.spec_goal, self.config.repo_path, self.config.timeout)

    def _read_decision(self) -> dict[str, Any]:
        decision = read_json(self.artifacts.goal_decision, {})
        if isinstance(decision, dict) and decision:
            return decision
        self._log("goal decision missing/empty; using safe default decision")
        return {
            "task_satisfied": False,
            "reason": "Goal Agent did not produce a decision.",
            "call_design_agent": not self.artifacts.design_spec_goal.exists(),
            "call_test_agent": not self.artifacts.test_spec_goal.exists(),
            "call_implement_agent": self.artifacts.design_spec_goal.exists() and self.artifacts.test_spec_goal.exists(),
            "design_objective": "Create or strengthen executable design_spec_goal.py from the task.",
            "test_objective": "Create or strengthen executable test_spec_goal.py from the task.",
            "implement_objective": "Implement the task so the central spec goal passes.",
            "review_notes": ["Fallback decision generated by orchestrator."],
        }

    def _finish(self, passed: bool, stages: list[StageResult], goal_run: GoalRun | None) -> RunResult:
        summary = {
            "task_id": self.config.task_id,
            "repo_path": str(self.config.repo_path),
            "passed": passed,
            "log_dir": str(self.log_dir),
            "artifacts": {
                "task_problem_statement": str(self.artifacts.task_problem_statement),
                "spec_goal": str(self.artifacts.spec_goal),
                "design_spec_goal": str(self.artifacts.design_spec_goal),
                "test_spec_goal": str(self.artifacts.test_spec_goal),
                "impl_status": str(self.artifacts.impl_status),
                "goal_decision": str(self.artifacts.goal_decision),
                "goal_report": str(self.artifacts.goal_report),
            },
            "stages": [
                {
                    "stage": stage.stage,
                    "loop_index": stage.loop_index,
                    "exit_status": stage.exit_status,
                    "trajectory": str(stage.trajectory),
                }
                for stage in stages
            ],
            "goal": None
            if goal_run is None
            else {"returncode": goal_run.returncode, "passed": goal_run.passed, "output": goal_run.output},
        }
        write_json(self.artifacts.stage_summary, summary)
        self._log(f"summary={self.artifacts.stage_summary}")
        self._log(f"run finished passed={passed}")
        return RunResult(passed, self.log_dir, self.artifacts, stages, goal_run, summary)

    def _should_stop(self, stage: str) -> bool:
        requested = self.config.stop_after_stage
        aliases = {
            "design_pattern": "design",
            "design_goal": "test",
            "implementation": "implement",
        }
        requested = aliases.get(requested, requested)
        return bool(requested and requested == stage)

    def _make_log_dir(self) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        task_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in self.config.task_id)
        return self.config.log_root / f"{task_id}_{stamp}"

    def _initialize_spec_scripts(self) -> None:
        if not self.artifacts.design_spec_goal.exists():
            self.artifacts.design_spec_goal.write_text(_child_placeholder("design"), encoding="utf-8")
        if not self.artifacts.test_spec_goal.exists():
            self.artifacts.test_spec_goal.write_text(_child_placeholder("test"), encoding="utf-8")
        if not self.artifacts.impl_status.exists():
            self.artifacts.impl_status.write_text(_child_placeholder("implementation"), encoding="utf-8")
        if not self.artifacts.spec_goal.exists():
            self.artifacts.spec_goal.write_text(
                _central_spec_template(
                    self.artifacts.design_spec_goal,
                    self.artifacts.test_spec_goal,
                    self.artifacts.impl_status,
                ),
                encoding="utf-8",
            )

    def _log(self, message: str) -> None:
        print(f"[arch_swe_agent] {message}", flush=True)


def _child_placeholder(kind: str) -> str:
    return f'''from __future__ import annotations

import argparse
import sys
from pathlib import Path

{kind.upper().replace(" ", "_")}_SPEC_GOALS = [
    "Placeholder {kind} spec goal. The corresponding agent must replace this script."
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.parse_args()
    print("[{kind}_spec_goal] missing executable {kind} spec: agent must replace placeholder")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _central_spec_template(design_path: Path, test_path: Path, impl_path: Path) -> str:
    return f'''from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SPEC_GOAL = {{
    "design_spec_goal": {str(design_path)!r},
    "test_spec_goal": {str(test_path)!r},
    "impl_status": {str(impl_path)!r},
}}


def run_child(script: str, repo: Path) -> int:
    path = Path(script)
    if not path.exists():
        print(f"[spec_goal] missing child script: {{path}}")
        return 1
    result = subprocess.run(
        [sys.executable, str(path), "--repo", str(repo)],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(f"\\n[spec_goal] child={{path.name}} returncode={{result.returncode}}")
    print(result.stdout)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    for name, script in SPEC_GOAL.items():
        rc = run_child(script, args.repo)
        if rc != 0:
            failures.append(name)
    if failures:
        print(f"[spec_goal] failing components: {{', '.join(failures)}}")
        return 1
    print("[spec_goal] design_spec_goal + test_spec_goal + impl_status all pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
