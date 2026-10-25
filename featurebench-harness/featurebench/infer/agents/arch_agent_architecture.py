"""
Architecture-only variant of the custom ArchAgent OpenHands wrapper.
"""

from __future__ import annotations

from featurebench.infer.agents.arch_agent import ArchAgent


class ArchAgentArchitecture(ArchAgent):
    @property
    def name(self) -> str:
        return "arch_agent_architecture"

    def get_run_command(self, instruction: str) -> str:  # noqa: ARG002
        max_iterations = str(self.env_vars.get("OPENHANDS_MAX_ITERATIONS") or "500")
        reference_test_flags = self._reference_test_path_flags()
        return (
            "set -o pipefail; "
            "ARCH_AGENT_LOG_ROOT=/agent-logs/arch_agent "
            "PYTHONUNBUFFERED=1 "
            "PYTHONPATH=/installed-agent:$PYTHONPATH; "
            "if [ -d /testbed/pydantic ]; then "
            "  mv /testbed/pydantic /testbed/__pydantic_hidden__; "
            "  _restore_pydantic='mv /testbed/__pydantic_hidden__ /testbed/pydantic'; "
            "else _restore_pydantic=true; fi; "
            "/opt/openhands-venv/bin/python -u /installed-agent/run_agent.py "
            "--workspace /testbed "
            "--task-file /agent-logs/arch_agent/task.json "
            "--dataset /installed-agent/data/archmon_three_cases.jsonl "
            "--stop-after-stage architecture "
            f"{reference_test_flags} "
            f"--max-iterations {max_iterations}; "
            "status=$?; "
            "$_restore_pydantic; "
            "[ -f /agent-logs/trajectory.json ] || "
            "printf '%s\\n' '[{\"action\":\"finish\",\"message\":\"arch_agent_architecture completed\"}]' "
            "> /agent-logs/trajectory.json; "
            "if [ \"$status\" -eq 0 ] || [ \"$status\" -eq 2 ]; then exit 0; fi; "
            "exit \"$status\""
        )
