"""
OpenHands agent wrapped by the custom ArchAgent workflow.
"""

import json
import os
import shlex
import tempfile
from pathlib import Path

from featurebench.infer.agents.openhands import OpenHandsAgent


class ArchAgent(OpenHandsAgent):
    """OpenHands FeatureBench agent with arch_agent MCP tools."""

    @property
    def name(self) -> str:
        return "arch_agent"

    @property
    def install_script(self) -> str:
        base_script = super().install_script
        arch_agent_src = self.env_vars.get("ARCH_AGENT_SOURCE_DIR")
        if not arch_agent_src:
            arch_agent_src = "/host_arch_agent"

        return (
            base_script
            + f"""

echo "Installing ArchAgent wrapper..."
apt-get update
apt-get install -y ripgrep
uv_pip_install_with_fallback openhands-sdk==1.16.1 pyflakes==3.4.0 pytest fastmcp
uv_pip_install_with_fallback openhands-tools==1.16.1
/opt/openhands-venv/bin/python -c "
import json, os
p = os.path.join(os.path.dirname(__import__('litellm').__file__), 'model_prices_and_context_window_backup.json')
with open(p) as f:
    data = json.load(f)
for model, info in {{
    'deepseek/deepseek-v4-pro': {{'cache_creation_input_token_cost':0.0,'input_cost_per_token':4.35e-07,'output_cost_per_token':8.7e-07,'cache_read_input_token_cost':3.625e-09,'input_cost_per_token_cache_hit':3.625e-09,'litellm_provider':'deepseek','mode':'chat','max_input_tokens':1048576,'max_output_tokens':393216,'max_tokens':393216,'supports_function_calling':True,'supports_native_streaming':True,'supports_reasoning':True,'supports_prompt_caching':True,'supports_response_schema':True,'supports_system_messages':True,'supports_tool_choice':True,'supported_endpoints':['/v1/chat/completions'],'source':'https://api-docs.deepseek.com/quick_start/pricing'}},
    'deepseek-v4-pro': {{'cache_creation_input_token_cost':0.0,'input_cost_per_token':4.35e-07,'output_cost_per_token':8.7e-07,'cache_read_input_token_cost':3.625e-09,'input_cost_per_token_cache_hit':3.625e-09,'litellm_provider':'deepseek','mode':'chat','max_input_tokens':1048576,'max_output_tokens':393216,'max_tokens':393216,'supports_function_calling':True,'supports_native_streaming':True,'supports_reasoning':True,'supports_prompt_caching':True,'supports_response_schema':True,'supports_system_messages':True,'supports_tool_choice':True,'supported_endpoints':['/v1/chat/completions'],'source':'https://api-docs.deepseek.com/quick_start/pricing'}},
    'openai/deepseek-v4-pro': {{'cache_creation_input_token_cost':0.0,'input_cost_per_token':4.35e-07,'output_cost_per_token':8.7e-07,'cache_read_input_token_cost':3.625e-09,'input_cost_per_token_cache_hit':3.625e-09,'litellm_provider':'openai','mode':'chat','max_input_tokens':1048576,'max_output_tokens':393216,'max_tokens':393216,'supports_function_calling':True,'supports_native_streaming':True,'supports_reasoning':True,'supports_prompt_caching':True,'supports_response_schema':True,'supports_system_messages':True,'supports_tool_choice':True,'supported_endpoints':['/v1/chat/completions'],'source':'https://api-docs.deepseek.com/quick_start/pricing'}},
    'deepseek/deepseek-v4-flash': {{'cache_creation_input_token_cost':0.0,'input_cost_per_token':1.4e-07,'output_cost_per_token':2.8e-07,'cache_read_input_token_cost':2.8e-09,'input_cost_per_token_cache_hit':2.8e-09,'litellm_provider':'deepseek','mode':'chat','max_input_tokens':1048576,'max_output_tokens':393216,'max_tokens':393216,'supports_function_calling':True,'supports_native_streaming':True,'supports_prompt_caching':True,'supports_reasoning':True,'supports_response_schema':True,'supports_system_messages':True,'supports_tool_choice':True,'supported_endpoints':['/v1/chat/completions'],'source':'https://api-docs.deepseek.com/quick_start/pricing'}},
    'deepseek-v4-flash': {{'cache_creation_input_token_cost':0.0,'input_cost_per_token':1.4e-07,'output_cost_per_token':2.8e-07,'cache_read_input_token_cost':2.8e-09,'input_cost_per_token_cache_hit':2.8e-09,'litellm_provider':'deepseek','mode':'chat','max_input_tokens':1048576,'max_output_tokens':393216,'max_tokens':393216,'supports_function_calling':True,'supports_native_streaming':True,'supports_reasoning':True,'supports_prompt_caching':True,'supports_response_schema':True,'supports_system_messages':True,'supports_tool_choice':True,'supported_endpoints':['/v1/chat/completions'],'source':'https://api-docs.deepseek.com/quick_start/pricing'}},
    'openai/deepseek-v4-flash': {{'cache_creation_input_token_cost':0.0,'input_cost_per_token':1.4e-07,'output_cost_per_token':2.8e-07,'cache_read_input_token_cost':2.8e-09,'input_cost_per_token_cache_hit':2.8e-09,'litellm_provider':'openai','mode':'chat','max_input_tokens':1048576,'max_output_tokens':393216,'max_tokens':393216,'supports_function_calling':True,'supports_native_streaming':True,'supports_prompt_caching':True,'supports_reasoning':True,'supports_response_schema':True,'supports_system_messages':True,'supports_tool_choice':True,'supported_endpoints':['/v1/chat/completions'],'source':'https://api-docs.deepseek.com/quick_start/pricing'}},
}}.items():
    data[model] = info
with open(p, 'w') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
print('pricing patched')
"
mkdir -p /installed-agent
if [ -d /host_arch_agent ]; then
  rm -rf /installed-agent/arch_agent /installed-agent/data /installed-agent/run_agent.py
  cp -a /host_arch_agent/arch_agent /installed-agent/
  cp -a /host_arch_agent/run_agent.py /installed-agent/
  if [ -d /host_arch_agent/data ]; then
    cp -a /host_arch_agent/data /installed-agent/
  fi
else
  echo "ArchAgent source not found: /host_arch_agent" >&2
  exit 1
fi
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
echo "ArchAgent wrapper installed"
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
                "mkdir -p /agent-logs/arch_agent",
                log_file=log_file,
            )
            if exit_code != 0:
                self.logger.warning("Failed to create ArchAgent task directory: %s", output)
                return False
            self.cm.copy_to_container(
                container,
                temp_path,
                "/agent-logs/arch_agent/task.json",
            )
        except Exception as exc:
            self.logger.warning("Failed to write ArchAgent task payload: %s", exc)
            return False
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return ok

    def pre_run_hook(self, container, log_file: Path) -> bool:
        self.cm.exec_command(container, "mkdir -p /agent-logs/arch_agent", log_file=log_file)
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
        self._copy_arch_agent_logs(container, log_file)
        return super().post_run_hook(container, log_file)

    def failure_hook(self, container, log_file: Path) -> None:
        self._copy_arch_agent_logs(container, log_file)
        return super().failure_hook(container, log_file)

    def _copy_arch_agent_logs(self, container, log_file: Path) -> None:
        log_dir = Path(log_file).parent
        copied = self.cm.copy_from_container(
            container,
            "/agent-logs/arch_agent",
            log_dir / "arch_agent",
        )
        if not copied:
            self.logger.warning("Failed to copy ArchAgent logs from container")

    def _reference_test_path_flags(self) -> str:
        raw = (
            self.env_vars.get("ARCH_AGENT_REFERENCE_TEST_PATHS")
            or self.env_vars.get("FEATUREBENCH_REFERENCE_TEST_PATHS")
            or os.environ.get("ARCH_AGENT_REFERENCE_TEST_PATHS")
            or os.environ.get("FEATUREBENCH_REFERENCE_TEST_PATHS")
            or ""
        )
        if not str(raw).strip():
            return ""
        normalized = str(raw).replace(",", "\n")
        flags: list[str] = []
        for line in normalized.splitlines():
            path = line.strip()
            if not path:
                continue
            flags.append(f"--reference-test-path {shlex.quote(path)}")
        return " ".join(flags)

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
            f"{reference_test_flags} "
            f"--max-iterations {shlex.quote(max_iterations)}; "
            "status=$?; "
            "$_restore_pydantic; "
            "[ -f /agent-logs/trajectory.json ] || "
            "printf '%s\\n' '[{\"action\":\"finish\",\"message\":\"arch_agent completed\"}]' "
            "> /agent-logs/trajectory.json; "
            "if [ \"$status\" -eq 0 ] || [ \"$status\" -eq 2 ]; then exit 0; fi; "
            "exit \"$status\""
        )

    def get_env_setup_script(self) -> str:
        script = super().get_env_setup_script()
        lines = [script, "export LITELLM_LOCAL_MODEL_COST_MAP=True"]
        passthrough = {
            "LLM_TIMEOUT",
            "LLM_NUM_RETRIES",
            "LLM_RETRY_MIN_WAIT",
            "LLM_RETRY_MAX_WAIT",
            "LLM_DISABLE_THINKING",
            "LLM_REASONING_EFFORT",
        }
        for key in sorted(passthrough):
            value = self.env_vars.get(key)
            if value is None or not str(value).strip():
                continue
            escaped = str(value).replace("'", "'\\''")
            lines.append(f"export {key}='{escaped}'")
        return "\n".join(lines)
