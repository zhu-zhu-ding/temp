"""FeatureBench adapter for the CodeSpec2 obligation-spec workflow."""

from __future__ import annotations

import shlex
from pathlib import Path

from featurebench.infer.agents.arch_swe_agent import ArchSweAgent
from featurebench.infer.agents.mini_swe_agent import MiniSweAgent


class CodeSpec2Agent(ArchSweAgent):
    """Install CodeSpec2 into each task container."""

    _WRAPPER_SETUP_COMMAND = r'''
set -e
VENV_DIR="${VENV_DIR:-/opt/mini-swe-agent-venv}"
test -x "$VENV_DIR/bin/python"
test -d /host_arch_swe_agent
# Security: the answer-leaking reference data must NOT be mounted into the
# container. arch_agent/repodata holds pristine *unmasked* repos extracted
# from the image (arch_agent/batch_test.py::extract_repo does `docker cp
# <image>:/testbed/.`), and arch_agent/test_output holds prior run artifacts;
# both let the model copy the reference solution instead of implementing it.
for _leak in arch_agent repodata test_output; do
  if [ -e "/host_arch_swe_agent/$_leak" ]; then
    echo "SECURITY VIOLATION: answer-leaking path /host_arch_swe_agent/$_leak is mounted" >&2
    exit 1
  fi
done
test -d /host_arch_swe_agent/codespec2
test -d /host_arch_swe_agent/arch_agent_mini_swe/arch_agent_mini_swe
test -d /host_arch_swe_agent/arch_agent_mini_swe/arch_agent_mini_swe_simple
rm -rf /installed-agent/codespec2 /installed-agent/arch_agent_mini_swe /installed-agent/arch_agent_mini_swe_simple
cp -a /host_arch_swe_agent/codespec2 /installed-agent/
cp -a /host_arch_swe_agent/arch_agent_mini_swe/arch_agent_mini_swe /installed-agent/
cp -a /host_arch_swe_agent/arch_agent_mini_swe/arch_agent_mini_swe_simple /installed-agent/
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check -q pyflakes "litellm[proxy]"
"$VENV_DIR/bin/python" -m py_compile /installed-agent/codespec2/*.py /installed-agent/arch_agent_mini_swe/*.py /installed-agent/arch_agent_mini_swe_simple/*.py
'''

    @property
    def name(self) -> str:
        return "codespec2"

    @property
    def install_script(self) -> str:
        base_script = MiniSweAgent.install_script.fget(self)
        return (
            base_script
            + f'''

echo "Installing CodeSpec2 wrapper..."
if ! command -v git >/dev/null 2>&1 || ! command -v rg >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git ripgrep
fi
{self._WRAPPER_SETUP_COMMAND}
echo "CodeSpec2 wrapper installed"
'''
        )

    def pre_run_setup(self, container, instance, log_file: Path) -> bool:
        if not super().pre_run_setup(container, instance, log_file):
            raise RuntimeError("Failed to prepare CodeSpec2 task payload")
        exit_code, output = self.cm.exec_command(container, self._WRAPPER_SETUP_COMMAND, log_file=log_file)
        if exit_code != 0:
            self.logger.error("Failed to prepare CodeSpec2 wrapper: %s", output)
            raise RuntimeError("Failed to prepare CodeSpec2 wrapper")
        return True

    def pre_run_hook(self, container, log_file: Path) -> bool:
        self.cm.exec_command(
            container,
            "mkdir -p /agent-logs/codespec2 && rm -rf /testbed/agent-logs && ln -s /agent-logs /testbed/agent-logs",
            log_file=log_file,
        )
        return True

    def get_run_command(self, instruction: str) -> str:  # noqa: ARG002
        model_name = self._get_model_name()
        implementation_steps = str(self._kwargs.get("max_iters") or self.env_vars.get("CODESPEC2_MAX_ITERS") or "300")
        spec_agent_steps = str(self.env_vars.get("CODESPEC2_SPEC_AGENT_ITERS") or "300")
        return (
            "set -o pipefail; cd /testbed; "
            "test -f /installed-agent/codespec2/run_agent.py || { echo 'Missing CodeSpec2 agent' >&2; exit 1; }; "
            "PYTHONPATH=/installed-agent:$PYTHONPATH PYTHONUNBUFFERED=1 "
            '"$MINI_SWE_AGENT_PYTHON" -u /installed-agent/codespec2/run_agent.py '
            "--workspace /testbed --task-file /agent-logs/arch_swe_agent/task.json --task-id task "
            f"--model {shlex.quote(model_name)} --model-class litellm "
            "--log-root /agent-logs/codespec2 "
            f"--spec-agent-step-limit {shlex.quote(spec_agent_steps)} "
            f"--implementation-step-limit {shlex.quote(implementation_steps)} "
            "2>&1 | tee /agent-logs/codespec2_output.log; "
            "exit ${PIPESTATUS[0]}"
        )

    def post_run_hook(self, container, log_file: Path) -> bool:
        self._copy_codespec2_logs(container, log_file)
        return True

    def failure_hook(self, container, log_file: Path) -> None:
        self._copy_codespec2_logs(container, log_file)
        return super().failure_hook(container, log_file)

    def _copy_codespec2_logs(self, container, log_file: Path) -> None:
        log_dir = Path(log_file).parent
        for src, dst in (
            ("/agent-logs/codespec2", log_dir / "codespec2"),
            ("/agent-logs/codespec2_output.log", log_dir / "codespec2_output.log"),
        ):
            try:
                self.cm.copy_from_container(container, src, dst)
            except Exception:
                pass
