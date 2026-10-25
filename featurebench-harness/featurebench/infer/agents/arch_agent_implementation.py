"""
Architecture+implementation variant of the custom ArchAgent OpenHands wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path

from featurebench.infer.agents.arch_agent import ArchAgent


class ArchAgentImplementation(ArchAgent):
    @property
    def name(self) -> str:
        return "arch_agent_implementation"

    def get_run_command(self, instruction: str) -> str:  # noqa: ARG002
        max_iterations = str(self.env_vars.get("OPENHANDS_MAX_ITERATIONS") or "500")
        skip_test_analysis_flag = "--skip-test-analysis "
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
            "--stop-after-stage implementation "
            f"{skip_test_analysis_flag}"
            f"{reference_test_flags} "
            f"--max-iterations {max_iterations}; "
            "status=$?; "
            "$_restore_pydantic; "
            "[ -f /agent-logs/trajectory.json ] || "
            "printf '%s\\n' '[{\"action\":\"finish\",\"message\":\"arch_agent_implementation completed\"}]' "
            "> /agent-logs/trajectory.json; "
            "if [ \"$status\" -eq 0 ] || [ \"$status\" -eq 2 ]; then exit 0; fi; "
            "exit \"$status\""
        )

    def post_run_hook(self, container, log_file: Path) -> bool:
        self._copy_arch_agent_logs(container, log_file)
        log_path = Path(log_file)
        if self._implementation_stage_succeeded(log_path):
            return True
        if self._repo_has_changes(container, log_path):
            self.logger.warning(
                "feature_probe.py did not pass, but repository changes were produced; "
                "accepting patch under fallback policy"
            )
            return True
        return False

    def failure_hook(self, container, log_file: Path) -> None:
        self._copy_arch_agent_logs(container, log_file)
        return super().failure_hook(container, log_file)

    def _implementation_stage_succeeded(self, log_file: Path) -> bool:
        log_dir = log_file.parent
        arch_root = log_dir / "arch_agent"
        if not arch_root.exists():
            self.logger.error("arch_agent logs not found; treating implementation run as failed")
            return False

        final_checks = list(arch_root.rglob("state/final_check.json"))
        if not final_checks:
            self.logger.warning("final_check.json not found under arch_agent logs; using fallback policy")
            return False

        latest_final_check = max(final_checks, key=lambda path: path.stat().st_mtime)
        try:
            data = json.loads(latest_final_check.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning("Failed to parse %s: %s", latest_final_check, exc)
            return False

        if isinstance(data, dict) and data.get("implementation_passed"):
            self.logger.info("Implementation stage passed according to %s", latest_final_check)
            return True

        self.logger.warning("No passing implementation-stage final_check.json found")
        return False

    def _repo_has_changes(self, container, log_file: Path) -> bool:
        exit_code, output = self.cm.exec_command(
            container,
            "cd /testbed && git status --porcelain",
            log_file=log_file,
        )
        if exit_code != 0:
            self.logger.warning("Failed to inspect repository status for fallback acceptance")
            return False
        if output.strip():
            self.logger.info("Repository has changes; fallback acceptance enabled")
            return True
        self.logger.info("Repository is clean; fallback acceptance not enabled")
        return False
