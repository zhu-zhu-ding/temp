"""
Architecture-aware mini-swe-agent implementation.

This adapter runs Agent/arch_agent_mini_swe inside the task container.
"""

from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path

from featurebench.infer.agents.mini_swe_agent import MiniSweAgent


class ArchSweAgent(MiniSweAgent):
    """FeatureBench adapter for the arch_agent_mini_swe workflow."""

    @property
    def name(self) -> str:
        return "arch_swe_agent"

    @property
    def install_script(self) -> str:
        base_script = super().install_script
        return (
            base_script
            + """

echo "Installing arch_swe_agent wrapper..."
apt-get update
apt-get install -y git ripgrep

mkdir -p /installed-agent
if [ -d /host_arch_swe_agent ]; then
  rm -rf /installed-agent/arch_agent_mini_swe /installed-agent/run_agent.py /installed-agent/README.md
  cp -a /host_arch_swe_agent/arch_agent_mini_swe /installed-agent/
  cp -a /host_arch_swe_agent/run_agent.py /installed-agent/
  if [ -f /host_arch_swe_agent/README.md ]; then
    cp -a /host_arch_swe_agent/README.md /installed-agent/
  fi
else
  echo "arch_agent_mini_swe source not found: /host_arch_swe_agent" >&2
  exit 1
fi

"$VENV_DIR/bin/python" -m py_compile /installed-agent/run_agent.py /installed-agent/arch_agent_mini_swe/*.py
echo "arch_swe_agent wrapper installed"
"""
        )

    def pre_run_setup(self, container, instance, log_file: Path) -> bool:
        ok = super().pre_run_setup(container, instance, log_file)
        task_payload = {
            "instance_id": instance.instance_id,
            "problem_statement": instance.problem_statement,
        }
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                json.dump(task_payload, tmp, ensure_ascii=False)
                temp_path = Path(tmp.name)
            exit_code, output = self.cm.exec_command(
                container,
                "mkdir -p /agent-logs/arch_swe_agent",
                log_file=log_file,
            )
            if exit_code != 0:
                self.logger.warning("Failed to create arch_swe_agent task directory: %s", output)
                return False
            self.cm.copy_to_container(
                container,
                temp_path,
                "/agent-logs/arch_swe_agent/task.json",
            )
        except Exception as exc:
            self.logger.warning("Failed to write arch_swe_agent task payload: %s", exc)
            return False
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return ok

    def pre_run_hook(self, container, log_file: Path) -> bool:
        self.cm.exec_command(container, "mkdir -p /agent-logs/arch_swe_agent", log_file=log_file)
        return True

    def post_run_hook(self, container, log_file: Path) -> bool:
        self._copy_arch_swe_logs(container, log_file)
        return True

    def failure_hook(self, container, log_file: Path) -> None:
        self._copy_arch_swe_logs(container, log_file)
        return super().failure_hook(container, log_file)

    def get_run_command(self, instruction: str) -> str:  # noqa: ARG002
        model_name = self._get_model_name()
        max_iters = str(self._kwargs.get("max_iters") or self.env_vars.get("ARCH_SWE_MAX_ITERS") or "200")
        return (
            "set -o pipefail; "
            "PYTHONPATH=/installed-agent:$PYTHONPATH "
            "PYTHONUNBUFFERED=1 "
            '"$MINI_SWE_AGENT_PYTHON" -u /installed-agent/run_agent.py '
            "--workspace /testbed "
            "--task-file /agent-logs/arch_swe_agent/task.json "
            f"--task-id task "
            f"--model {shlex.quote(model_name)} "
            "--log-root /agent-logs/arch_swe_agent "
            f"--implementation-step-limit {shlex.quote(max_iters)} "
            f"--step-limit {shlex.quote(max_iters)} "
            "2>&1 | tee /agent-logs/arch_swe_agent_output.log; "
            "status=${PIPESTATUS[0]}; "
            "if [ \"$status\" -eq 0 ]; then exit 0; fi; "
            "if [ -n \"$(git -C /testbed status --porcelain)\" ]; then "
            "  echo \"arch_swe_agent produced repository changes; returning success so FeatureBench can collect the patch\"; "
            "  exit 0; "
            "fi; "
            "exit \"$status\""
        )

    def _copy_arch_swe_logs(self, container, log_file: Path) -> None:
        log_dir = Path(log_file).parent
        try:
            self.cm.copy_from_container(
                container,
                "/agent-logs/arch_swe_agent",
                log_dir / "arch_swe_agent",
            )
        except Exception:
            pass
        try:
            self.cm.copy_from_container(
                container,
                "/agent-logs/arch_swe_agent_output.log",
                log_dir / "arch_swe_agent_output.log",
            )
        except Exception:
            pass
