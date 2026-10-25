"""
OpenHands agent wrapped by the Architecture Monitor workflow.
"""

import json
import shlex
import tempfile
from pathlib import Path

from featurebench.infer.agents.openhands import OpenHandsAgent


class OpenHandsArchMonAgent(OpenHandsAgent):
    """OpenHands FeatureBench agent with mandatory architecture monitoring."""

    @property
    def name(self) -> str:
        return "openhands_archmon"

    @property
    def install_script(self) -> str:
        base_script = super().install_script
        archmon_src = self.env_vars.get("ARCHMON_SOURCE_DIR")
        if not archmon_src:
            archmon_src = "/host_archmon"

        return (
            base_script
            + f"""

echo "Installing Architecture Monitor wrapper..."
uv_pip_install_with_fallback openhands-sdk==1.16.1 openhands-tools==1.16.1 tree-sitter==0.25.2 tree-sitter-python==0.25.0 pyflakes==3.4.0 pytest
mkdir -p /installed-agent/architecture_monitor
if [ -d {shlex.quote(archmon_src)} ]; then
  cp -r {shlex.quote(archmon_src)}/architecture_monitor /installed-agent/
  cp {shlex.quote(archmon_src)}/architecture_monitor_cli.py /installed-agent/
else
  echo "Architecture monitor source not found: {archmon_src}" >&2
  exit 1
fi
$VENV_DIR/bin/python - <<'PY'
import openhands.sdk
import openhands.tools.file_editor
import tree_sitter_python
import pyflakes
import pytest
PY
$VENV_DIR/bin/python - <<'PY'
import openhands.sdk.llm.utils.model_features as model_features

for token in (
    "deepseek/deepseek-v4-flash",
    "deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "deepseek-v4-pro",
):
    if token not in model_features.SEND_REASONING_CONTENT_MODELS:
        model_features.SEND_REASONING_CONTENT_MODELS.append(token)
clear_cache = getattr(model_features.get_features, "cache_clear", None)
if callable(clear_cache):
    clear_cache()
PY
echo "Architecture Monitor wrapper installed"
"""
        )

    def pre_run_setup(self, container, instance, log_file: Path) -> bool:
        ok = super().pre_run_setup(container, instance, log_file)
        task_payload = {"problem_statement": instance.problem_statement}
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
                "mkdir -p /agent-logs/architecture_monitor",
                log_file=log_file,
            )
            if exit_code != 0:
                self.logger.warning(
                    "Failed to create Architecture Monitor task directory: %s",
                    output,
                )
                return False
            self.cm.copy_to_container(
                container,
                temp_path,
                "/agent-logs/architecture_monitor/task.json",
            )
        except Exception as exc:
            self.logger.warning("Failed to write Architecture Monitor task payload: %s", exc)
            return False
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return ok

    def pre_run_hook(self, container, log_file: Path) -> bool:
        self.cm.exec_command(container, "mkdir -p /agent-logs/architecture_monitor", log_file=log_file)
        exit_code, output = self.cm.exec_command(
            container,
            (
                "source /opt/miniconda3/etc/profile.d/conda.sh && "
                "conda activate testbed && "
                "python - <<'PY'\n"
                "try:\n"
                "    import pytest\n"
                "except ImportError:\n"
                "    raise SystemExit(1)\n"
                "PY\n"
                "if [ \"$?\" -ne 0 ]; then "
                "python -m pip install pytest -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/ "
                "|| python -m pip install pytest; "
                "fi"
            ),
            log_file=log_file,
        )
        if exit_code != 0:
            self.logger.warning("Failed to install pytest in testbed env: %s", output)
            return False
        return True

    def post_run_hook(self, container, log_file: Path) -> bool:
        self._copy_architecture_monitor_logs(container, log_file)
        return super().post_run_hook(container, log_file)

    def failure_hook(self, container, log_file: Path) -> None:
        self._copy_architecture_monitor_logs(container, log_file)
        return super().failure_hook(container, log_file)

    def _copy_architecture_monitor_logs(self, container, log_file: Path) -> None:
        log_dir = Path(log_file).parent
        copied = self.cm.copy_from_container(
            container,
            "/agent-logs/architecture_monitor",
            log_dir / "architecture_monitor",
        )
        if not copied:
            self.logger.warning("Failed to copy Architecture Monitor logs from container")

    def get_run_command(self, instruction: str) -> str:  # noqa: ARG002
        return (
            "set -o pipefail; "
            "ARCHMON_LOG_ROOT=/agent-logs/architecture_monitor "
            "PYTHONPATH=/installed-agent:$PYTHONPATH "
            "/opt/openhands-venv/bin/python /installed-agent/architecture_monitor_cli.py "
            "--workspace /testbed "
            "--task-file /agent-logs/architecture_monitor/task.json; "
            "status=$?; "
            "[ -f /agent-logs/trajectory.json ] || "
            "printf '%s\\n' '[{\"action\":\"finish\",\"message\":\"architecture_monitor completed\"}]' "
            "> /agent-logs/trajectory.json; "
            "if [ \"$status\" -eq 0 ] || [ \"$status\" -eq 2 ]; then exit 0; fi; "
            "exit \"$status\""
        )
