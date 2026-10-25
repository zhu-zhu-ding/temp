from __future__ import annotations

import argparse
import json
from pathlib import Path

from arch_agent_mini_swe_simple import ArchMiniSweSimpleAgent, RunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the simplified architecture mini-swe agent.")
    parser.add_argument("--workspace", type=Path, required=True, help="Target repository.")
    parser.add_argument("--task", default="", help="Natural-language task.")
    parser.add_argument("--task-file", type=Path, help="JSON/JSONL/plain text task file.")
    parser.add_argument("--task-id", default="task", help="Run identifier.")
    parser.add_argument("--model", default="", help="mini-swe-agent model name.")
    parser.add_argument("--model-class", default="litellm", help="mini-swe-agent model class.")
    parser.add_argument("--log-root", type=Path, default=Path("mini_swe_simple_logs"))
    parser.add_argument("--step-limit", type=int, default=200)
    parser.add_argument("--implementation-step-limit", type=int, default=300)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--wall-time-limit-seconds", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["codespec", "design_document"],
        default="codespec",
        help="Workflow mode: CodeSpec pipeline or architecture-behavior-implementation pipeline.",
    )
    parser.add_argument(
        "--ablation",
        choices=[
            "",
            "without_generated_tests",
            "without_multi_view_evidence",
            "without_executable_architecture",
        ],
        default="",
        help="Run one isolated ablation workflow.",
    )
    parser.add_argument(
        "--stop-after-stage",
        choices=["", "architecture", "behavior", "design", "test", "implement"],
        default="",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task, task_id = load_task(args)
    runner = ArchMiniSweSimpleAgent(
        RunConfig(
            repo_path=args.workspace.resolve(),
            task=task,
            log_root=args.log_root.resolve(),
            task_id=task_id or args.task_id,
            model_name=args.model,
            model_class=args.model_class,
            step_limit=args.step_limit,
            implementation_step_limit=args.implementation_step_limit,
            cost_limit=args.cost_limit,
            timeout=args.timeout,
            wall_time_limit_seconds=args.wall_time_limit_seconds,
            stop_after_stage=args.stop_after_stage,
            ablation=args.ablation,
            mode=args.mode,
        )
    )
    result = runner.run()
    print(json.dumps(result.summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if result.passed else 1


def load_task(args: argparse.Namespace) -> tuple[str, str]:
    if args.task_file:
        raw = args.task_file.read_text(encoding="utf-8")
        suffix = args.task_file.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            line = raw.splitlines()[0] if suffix == ".jsonl" else raw
            data = json.loads(line)
            problem_statement = data.get("problem_statement")
            if not isinstance(problem_statement, str) or not problem_statement.strip():
                raise ValueError("Task JSON must contain a non-empty problem_statement field.")
            return (problem_statement, data.get("instance_id", args.task_id))
        return raw.strip(), args.task_id
    return args.task, args.task_id


if __name__ == "__main__":
    raise SystemExit(main())
