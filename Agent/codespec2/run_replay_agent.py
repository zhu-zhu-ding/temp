from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from codespec2.replay_agent import CodeSpec2ImplementationReplayAgent, ReplayConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay only CodeSpec2's implementation stage over fixed artifacts.")
    parser.add_argument("--workspace", type=Path, required=True, help="Target repository.")
    parser.add_argument("--task", default="", help="Natural-language task document.")
    parser.add_argument("--task-file", type=Path, help="Plain text, JSON, or JSONL task file.")
    parser.add_argument("--task-id", default="task")
    parser.add_argument("--reference-artifacts", type=Path, required=True)
    parser.add_argument("--model", default="", help="mini-swe-agent model name.")
    parser.add_argument("--model-class", default="litellm")
    parser.add_argument("--log-root", type=Path, default=Path("codespec2_replay_logs"))
    parser.add_argument("--implementation-step-limit", type=int, default=300)
    parser.add_argument("--cost-limit", type=float, default=3.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--wall-time-limit-seconds", type=int, default=0)
    return parser.parse_args()


def load_task(args: argparse.Namespace) -> tuple[str, str]:
    if not args.task_file:
        if not args.task.strip():
            raise ValueError("Provide --task or --task-file.")
        return args.task, args.task_id
    raw = args.task_file.read_text(encoding="utf-8")
    if args.task_file.suffix.lower() not in {".json", ".jsonl"}:
        return raw.strip(), args.task_id
    payload = json.loads(raw.splitlines()[0] if args.task_file.suffix.lower() == ".jsonl" else raw)
    task = payload.get("problem_statement")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("JSON task must contain non-empty problem_statement.")
    task_id = payload.get("instance_id", args.task_id)
    return task, task_id if isinstance(task_id, str) else args.task_id


def main() -> int:
    args = parse_args()
    task, task_id = load_task(args)
    result = CodeSpec2ImplementationReplayAgent(
        ReplayConfig(
            repo_path=args.workspace.resolve(),
            task=task,
            log_root=args.log_root.resolve(),
            reference_artifacts=args.reference_artifacts.resolve(),
            task_id=task_id,
            model_name=args.model,
            model_class=args.model_class,
            implementation_step_limit=args.implementation_step_limit,
            cost_limit=args.cost_limit,
            timeout=args.timeout,
            wall_time_limit_seconds=args.wall_time_limit_seconds,
        )
    ).run()
    print(result.artifacts.summary.read_text(encoding="utf-8"), flush=True)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
