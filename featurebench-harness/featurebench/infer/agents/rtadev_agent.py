"""FeatureBench adapters for RTADev planning with interchangeable programmers."""

from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path

from featurebench.infer.agents.mini_swe_agent import MiniSweAgent
from featurebench.infer.agents.openhands import OpenHandsAgent


class _RTADevBase:
    """Shared task payload and artifact handling for RTADev adapters."""

    programmer = ""

    def _install_runner(self, python: str) -> str:
        return rf'''
echo "Installing RTADev FeatureBench adapter..."
test -f /host_rtadev/featurebench_runner.py
cp /host_rtadev/featurebench_runner.py /installed-agent/rtadev_runner.py
{python} -m py_compile /installed-agent/rtadev_runner.py
echo "RTADev FeatureBench adapter installed"
'''

    def pre_run_setup(self, container, instance, log_file: Path) -> bool:
        if not super().pre_run_setup(container, instance, log_file):
            return False
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
                json.dump({"instance_id": instance.instance_id, "problem_statement": instance.problem_statement}, tmp)
                temp_path = Path(tmp.name)
            self.cm.exec_command(container, "mkdir -p /agent-logs/rtadev_agent", log_file=log_file)
            self.cm.copy_to_container(container, temp_path, "/agent-logs/rtadev_agent/task.json")
            return True
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def pre_run_hook(self, container, log_file: Path) -> bool:
        if not super().pre_run_hook(container, log_file):
            return False
        self.cm.exec_command(container, "mkdir -p /agent-logs/rtadev_agent", log_file=log_file)
        return True

    def _copy_rtadev_logs(self, container, log_file: Path) -> None:
        try:
            self.cm.copy_from_container(container, "/agent-logs/rtadev_agent", Path(log_file).parent / "rtadev_agent")
        except Exception:
            pass

    def post_run_hook(self, container, log_file: Path) -> bool:
        self._copy_rtadev_logs(container, log_file)
        return True

    def failure_hook(self, container, log_file: Path) -> None:
        self._copy_rtadev_logs(container, log_file)
        super().failure_hook(container, log_file)


class RTADevOpenHandsAgent(_RTADevBase, OpenHandsAgent):
    """RTADev planning stages followed by an OpenHands Programmer."""

    programmer = "openhands"

    @property
    def name(self) -> str:
        return "rtadev_agent_openhands"

    @property
    def install_script(self) -> str:
        return OpenHandsAgent.install_script.fget(self) + self._install_runner("/opt/openhands-venv/bin/python")

    def get_run_command(self, instruction: str) -> str:  # noqa: ARG002
        max_iters = str(self._kwargs.get("max_iters") or self.env_vars.get("OPENHANDS_MAX_ITERATIONS") or "200")
        return self._run_command("/opt/openhands-venv/bin/python", max_iters)

    def _run_command(self, python: str, max_iters: str) -> str:
        return (
            "set -o pipefail; cd /testbed; PYTHONUNBUFFERED=1 "
            f"{python} -u /installed-agent/rtadev_runner.py "
            "--task-file /agent-logs/rtadev_agent/task.json "
            f"--model {shlex.quote(str(self._kwargs.get('model') or ''))} "
            f"--programmer {self.programmer} --max-iters {shlex.quote(max_iters)} "
            "--log-dir /agent-logs/rtadev_agent 2>&1 | tee /agent-logs/rtadev_agent_output.log; "
            "status=${PIPESTATUS[0]}; "
            "if [ \"$status\" -eq 0 ] || [ -n \"$(git -C /testbed status --porcelain)\" ]; then exit 0; fi; exit \"$status\""
        )


class RTADevMiniSweAgent(_RTADevBase, MiniSweAgent):
    """RTADev planning artifacts followed by the unmodified MiniSweAgent workflow."""

    programmer = "planning"

    @property
    def name(self) -> str:
        return "rtadev_agent_mini_swe"

    @property
    def install_script(self) -> str:
        return MiniSweAgent.install_script.fget(self) + self._install_runner('"$VENV_DIR/bin/python"')

    def get_run_command(self, instruction: str) -> str:  # noqa: ARG002
        # Keep MiniSweAgent's official invocation, prompt, model configuration,
        # reasoning handling, and CLI compatibility intact. RTADev planning is
        # logged as an auxiliary artifact and does not replace the coder prompt.
        planning = RTADevOpenHandsAgent._run_command(
            self,
            '"$MINI_SWE_AGENT_PYTHON"',
            str(self._kwargs.get("max_iters") or self.env_vars.get("RTADEV_MAX_ITERS") or "200"),
        )
        official_mini_command = MiniSweAgent.get_run_command(self, instruction)
        planning = planning.replace("--programmer planning", "--programmer planning")
        return (
            "set -o pipefail; "
            f"{planning}; planning_status=$?; "
            "if [ \"$planning_status\" -ne 0 ]; then exit \"$planning_status\"; fi; "
            f"{official_mini_command}"
        )
