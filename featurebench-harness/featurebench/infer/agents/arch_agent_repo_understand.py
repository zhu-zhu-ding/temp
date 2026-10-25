"""
Repo-understand wrapper for design-pattern extraction. Runs through the
design_pattern stage and stops.
Success check: trajectory has a "finish" action.
"""

from __future__ import annotations

import json
from pathlib import Path

from featurebench.infer.agents.arch_agent import ArchAgent


class ArchAgentRepoUnderstand(ArchAgent):
    @property
    def name(self) -> str:
        return "arch_agent_repo_understand"

    def get_run_command(self, instruction: str) -> str:  # noqa: ARG002
        max_iterations = str(self.env_vars.get("OPENHANDS_MAX_ITERATIONS") or "500")
        reference_test_flags = self._reference_test_path_flags()
        return (
            "set -o pipefail; "
            "PYTHONPATH=/installed-agent:$PYTHONPATH; "
            "if [ -d /testbed/pydantic ]; then "
            "  mv /testbed/pydantic /testbed/__pydantic_hidden__; "
            "  _restore_pydantic='mv /testbed/__pydantic_hidden__ /testbed/pydantic'; "
            "else _restore_pydantic=true; fi; "
            "ARCH_AGENT_LOG_ROOT=/agent-logs/arch_agent "
            "PYTHONUNBUFFERED=1 "
            "/opt/openhands-venv/bin/python -u /installed-agent/run_agent.py "
            "--workspace /testbed "
            "--task-file /agent-logs/arch_agent/task.json "
            "--dataset /installed-agent/data/archmon_three_cases.jsonl "
            "--stop-after-stage implementation "
            f"{reference_test_flags} "
            f"--max-iterations {max_iterations}; "
            "status=$?; "
            "$_restore_pydantic; "
            "[ -f /agent-logs/trajectory.json ] || "
            "printf '%s\\n' '[{\"action\":\"finish\",\"message\":\"arch_agent_repo_understand completed\"}]' "
            "> /agent-logs/trajectory.json; "
            "if [ \"$status\" -eq 0 ] || [ \"$status\" -eq 2 ]; then exit 0; fi; "
            "exit \"$status\""
        )

    def post_run_hook(self, container, log_file: Path) -> bool:
        self._copy_arch_agent_logs(container, log_file)
        log_dir = Path(log_file).parent

        # Copy trajectory from container (same as standard OpenHands hook)
        self.cm.copy_from_container(
            container,
            "/agent-logs/trajectory.json",
            log_dir / "trajectory.json",
        )

        traj_path = log_dir / "trajectory.json"
        if not traj_path.exists():
            self.logger.warning("No trajectory.json found")
            return False

        try:
            trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning("Failed to parse trajectory: %s", exc)
            return False

        # Scan backwards for the last non-empty action (skip observation events)
        for event in reversed(trajectory):
            action = (event.get("action") or "").lower()
            if action == "finish":
                self.logger.info("Agent finished successfully (found finish action)")
                return True
            if action and action != "finish":
                self.logger.warning("Last action is '%s', not 'finish'", action)
                return False

        self.logger.warning("No action found in trajectory")
        return False
