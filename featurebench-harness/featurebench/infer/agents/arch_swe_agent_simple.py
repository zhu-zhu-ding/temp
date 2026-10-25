"""
Simplified architecture-aware mini-swe-agent implementation.

This adapter runs Agent/arch_agent_mini_swe/run_agent_simple.py inside the task
container.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from featurebench.infer.agents.arch_swe_agent import ArchSweAgent
from featurebench.infer.agents.mini_swe_agent import MiniSweAgent


class ArchSweAgentSimple(ArchSweAgent):
    """FeatureBench adapter for the simplified arch_agent_mini_swe workflow."""

    _WRAPPER_SETUP_COMMAND = r'''
set -e
VENV_DIR="${VENV_DIR:-/opt/mini-swe-agent-venv}"
test -x "$VENV_DIR/bin/python"
test -d /host_arch_swe_agent
test -f /host_arch_swe_agent/run_agent_simple.py
test -d /host_arch_swe_agent/arch_agent_mini_swe
test -d /host_arch_swe_agent/arch_agent_mini_swe_simple
rm -rf /installed-agent/arch_agent_mini_swe /installed-agent/arch_agent_mini_swe_simple /installed-agent/run_agent.py /installed-agent/run_agent_simple.py /installed-agent/README.md
cp -a /host_arch_swe_agent/arch_agent_mini_swe /installed-agent/
cp -a /host_arch_swe_agent/arch_agent_mini_swe_simple /installed-agent/
cp -a /host_arch_swe_agent/run_agent_simple.py /installed-agent/
if [ -f /host_arch_swe_agent/README.md ]; then
  cp -a /host_arch_swe_agent/README.md /installed-agent/
fi
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check -q pyflakes "litellm[proxy]"
"$VENV_DIR/bin/python" -c '
import sys
sys.path = [p for p in sys.path if p not in ("", "/testbed")]
import fastapi  # noqa: F401
import litellm  # noqa: F401
import orjson  # noqa: F401
from litellm.responses.mcp.chat_completions_handler import acompletion_with_mcp  # noqa: F401

print("arch_swe_agent_simple LiteLLM completion dependencies available")
'
"$VENV_DIR/bin/python" -m py_compile /installed-agent/run_agent_simple.py /installed-agent/arch_agent_mini_swe/*.py /installed-agent/arch_agent_mini_swe_simple/*.py
test -f /installed-agent/run_agent_simple.py
'''

    @property
    def name(self) -> str:
        return "arch_swe_agent_simple"

    @property
    def install_script(self) -> str:
        base_script = MiniSweAgent.install_script.fget(self)
        return (
            base_script
            + f"""

echo "Installing arch_swe_agent_simple wrapper..."
if ! command -v git >/dev/null 2>&1 || ! command -v rg >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git ripgrep
fi

echo "Installing arch_swe_agent_simple Python analysis dependencies..."
{self._WRAPPER_SETUP_COMMAND}
echo "arch_swe_agent_simple wrapper installed"
"""
        )

    def pre_run_setup(self, container, instance, log_file: Path) -> bool:
        if not super().pre_run_setup(container, instance, log_file):
            raise RuntimeError("Failed to prepare arch_swe_agent_simple task payload")
        exit_code, output = self.cm.exec_command(
            container,
            self._WRAPPER_SETUP_COMMAND,
            log_file=log_file,
        )
        if exit_code != 0:
            self.logger.error("Failed to prepare arch_swe_agent_simple wrapper: %s", output)
            raise RuntimeError("Failed to prepare arch_swe_agent_simple wrapper")
        return True

    def pre_run_hook(self, container, log_file: Path) -> bool:
        self.cm.exec_command(
            container,
            "mkdir -p /agent-logs/arch_swe_agent_simple && "
            "rm -rf /testbed/agent-logs && ln -s /agent-logs /testbed/agent-logs",
            log_file=log_file,
        )
        return True

    def get_run_command(self, instruction: str) -> str:  # noqa: ARG002
        model_name = self._get_model_name()
        max_iters = str(self._kwargs.get("max_iters") or self.env_vars.get("ARCH_SWE_MAX_ITERS") or "200")
        design_iters = str(self.env_vars.get("ARCH_SWE_DESIGN_ITERS") or "200")
        return (
            "set -o pipefail; cd /testbed; "
            "test -f /installed-agent/run_agent_simple.py || { "
            "  echo 'Missing /installed-agent/run_agent_simple.py' >&2; exit 1; }; "
            "pre_state=$( { git diff --binary HEAD --; "
            "git ls-files --others --exclude-standard -z | xargs -0 -r sha256sum; } | sha256sum); "
            "PYTHONPATH=/installed-agent:$PYTHONPATH "
            "PYTHONUNBUFFERED=1 "
            '"$MINI_SWE_AGENT_PYTHON" -u /installed-agent/run_agent_simple.py '
            "--workspace /testbed "
            "--task-file /agent-logs/arch_swe_agent/task.json "
            f"--task-id task "
            f"--model {shlex.quote(model_name)} "
            "--model-class litellm "
            "--log-root /agent-logs/arch_swe_agent_simple "
            f"--implementation-step-limit {shlex.quote(max_iters)} "
            f"--step-limit {shlex.quote(design_iters)} "
            "2>&1 | tee /agent-logs/arch_swe_agent_simple_output.log; "
            "status=${PIPESTATUS[0]}; "
            "if [ \"$status\" -eq 0 ]; then exit 0; fi; "
            "post_state=$( { git diff --binary HEAD --; "
            "git ls-files --others --exclude-standard -z | xargs -0 -r sha256sum; } | sha256sum); "
            "if [ \"$post_state\" != \"$pre_state\" ]; then "
            "  echo \"arch_swe_agent_simple produced repository changes; returning success so FeatureBench can collect the patch\"; "
            "  exit 0; "
            "fi; "
            "exit \"$status\""
        )

    def _copy_arch_swe_logs(self, container, log_file: Path) -> None:
        log_dir = Path(log_file).parent
        for src, dst in (
            ("/agent-logs/arch_swe_agent", log_dir / "arch_swe_agent"),
            ("/agent-logs/arch_swe_agent_simple", log_dir / "arch_swe_agent_simple"),
            ("/agent-logs/arch_swe_agent_simple_output.log", log_dir / "arch_swe_agent_simple_output.log"),
        ):
            try:
                self.cm.copy_from_container(container, src, dst)
            except Exception:
                pass
