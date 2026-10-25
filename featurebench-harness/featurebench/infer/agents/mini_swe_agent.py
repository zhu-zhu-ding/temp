"""
mini-swe-agent implementation.

This agent runs the external `mini` CLI in the container.
"""

import shlex
from pathlib import Path
from typing import Dict, List, Optional

from featurebench.infer.agents.base import BaseAgent
from featurebench.infer.container import DOCKER_HOST_GATEWAY


class MiniSweAgent(BaseAgent):
    """mini-swe-agent for FeatureBench inference."""

    _PROVIDER_API_KEY_ENV: Dict[str, str] = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GEMINI_API_KEY",
        "xai": "XAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "together": "TOGETHERAI_API_KEY",
        "togetherai": "TOGETHERAI_API_KEY",
        "together_ai": "TOGETHERAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "azure": "AZURE_API_KEY",
    }
    _PROVIDER_BASE_URL_ENV: Dict[str, List[str]] = {
        "anthropic": ["ANTHROPIC_BASE_URL"],
        "openai": ["OPENAI_BASE_URL"],
        "azure": ["AZURE_API_BASE"],
        "google": ["GOOGLE_GEMINI_BASE_URL"],
    }

    @property
    def name(self) -> str:
        return "mini_swe_agent"

    def _get_model_name(self) -> str:
        model = self._kwargs.get("model")
        model_name = str(model).strip() if model is not None else ""
        if not model_name:
            raise RuntimeError("Model is required for mini_swe_agent")
        return model_name

    def _get_model_provider(self) -> str:
        model_name = self._get_model_name()
        if "/" not in model_name:
            raise RuntimeError(
                "mini_swe_agent expects model in 'provider/model' format, "
                f"got: {model_name!r}"
            )
        provider, _ = model_name.split("/", 1)
        return provider.strip().lower()

    def _get_provider_api_key_env(self, provider: str) -> Optional[str]:
        return self._PROVIDER_API_KEY_ENV.get(provider)

    def _get_provider_base_url_envs(self, provider: str) -> List[str]:
        return self._PROVIDER_BASE_URL_ENV.get(provider, [])

    def _get_reasoning_effort(self) -> str:
        """Return a LiteLLM-compatible reasoning_effort value for mini-swe-agent."""
        raw = (
            self.env_vars.get("MSWEA_REASONING_EFFORT")
            or self.env_vars.get("LLM_REASONING_EFFORT")
            or self.env_vars.get("MODEL_REASONING_EFFORT")
            or "max"
        )
        value = str(raw).strip().lower()
        aliases = {
            "": "",
            "off": "none",
            "false": "none",
            "0": "none",
            "true": "high",
            "1": "high",
        }
        value = aliases.get(value, value)
        allowed = {"none", "minimal", "low", "medium", "high", "max", "default"}
        if value and value not in allowed:
            raise RuntimeError(
                "mini_swe_agent reasoning effort must be one of "
                f"{sorted(allowed)} or alias 'max', got: {raw!r}"
            )
        return value

    def _get_litellm_price_patch_script(self) -> str:
        """Patch LiteLLM's local backup price map for DeepSeek V4 models."""
        return r'''
echo "Patching LiteLLM DeepSeek V4 pricing..."
"$VENV_DIR/bin/python" - <<'PY'
import json
import os

import litellm

prices = {
    # Prices are per token. Source prices are per 1M tokens:
    # https://api-docs.deepseek.com/quick_start/pricing
    "deepseek/deepseek-v4-pro": {
        "cache_creation_input_token_cost": 0.0,
        "cache_read_input_token_cost": 3.625e-09,
        "input_cost_per_token": 4.35e-07,
        "input_cost_per_token_cache_hit": 3.625e-09,
        "output_cost_per_token": 8.7e-07,
        "litellm_provider": "deepseek",
        "mode": "chat",
        "max_input_tokens": 1048576,
        "max_output_tokens": 393216,
        "max_tokens": 393216,
        "supports_function_calling": True,
        "supports_native_streaming": True,
        "supports_prompt_caching": True,
        "supports_reasoning": True,
        "supports_response_schema": True,
        "supports_system_messages": True,
        "supports_tool_choice": True,
        "supported_endpoints": ["/v1/chat/completions"],
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
    },
    "deepseek-v4-pro": {
        "cache_creation_input_token_cost": 0.0,
        "cache_read_input_token_cost": 3.625e-09,
        "input_cost_per_token": 4.35e-07,
        "input_cost_per_token_cache_hit": 3.625e-09,
        "output_cost_per_token": 8.7e-07,
        "litellm_provider": "deepseek",
        "mode": "chat",
        "max_input_tokens": 1048576,
        "max_output_tokens": 393216,
        "max_tokens": 393216,
        "supports_function_calling": True,
        "supports_native_streaming": True,
        "supports_prompt_caching": True,
        "supports_reasoning": True,
        "supports_response_schema": True,
        "supports_system_messages": True,
        "supports_tool_choice": True,
        "supported_endpoints": ["/v1/chat/completions"],
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
    },
    "openai/deepseek-v4-pro": {
        "cache_creation_input_token_cost": 0.0,
        "cache_read_input_token_cost": 3.625e-09,
        "input_cost_per_token": 4.35e-07,
        "input_cost_per_token_cache_hit": 3.625e-09,
        "output_cost_per_token": 8.7e-07,
        "litellm_provider": "openai",
        "mode": "chat",
        "max_input_tokens": 1048576,
        "max_output_tokens": 393216,
        "max_tokens": 393216,
        "supports_function_calling": True,
        "supports_native_streaming": True,
        "supports_prompt_caching": True,
        "supports_reasoning": True,
        "supports_response_schema": True,
        "supports_system_messages": True,
        "supports_tool_choice": True,
        "supported_endpoints": ["/v1/chat/completions"],
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
    },
    "deepseek/deepseek-v4-flash": {
        "cache_creation_input_token_cost": 0.0,
        "cache_read_input_token_cost": 2.8e-09,
        "input_cost_per_token": 1.4e-07,
        "input_cost_per_token_cache_hit": 2.8e-09,
        "output_cost_per_token": 2.8e-07,
        "litellm_provider": "deepseek",
        "mode": "chat",
        "max_input_tokens": 1048576,
        "max_output_tokens": 393216,
        "max_tokens": 393216,
        "supports_function_calling": True,
        "supports_native_streaming": True,
        "supports_prompt_caching": True,
        "supports_reasoning": True,
        "supports_response_schema": True,
        "supports_system_messages": True,
        "supports_tool_choice": True,
        "supported_endpoints": ["/v1/chat/completions"],
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
    },
    "deepseek-v4-flash": {
        "cache_creation_input_token_cost": 0.0,
        "cache_read_input_token_cost": 2.8e-09,
        "input_cost_per_token": 1.4e-07,
        "input_cost_per_token_cache_hit": 2.8e-09,
        "output_cost_per_token": 2.8e-07,
        "litellm_provider": "deepseek",
        "mode": "chat",
        "max_input_tokens": 1048576,
        "max_output_tokens": 393216,
        "max_tokens": 393216,
        "supports_function_calling": True,
        "supports_native_streaming": True,
        "supports_prompt_caching": True,
        "supports_reasoning": True,
        "supports_response_schema": True,
        "supports_system_messages": True,
        "supports_tool_choice": True,
        "supported_endpoints": ["/v1/chat/completions"],
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
    },
    "openai/deepseek-v4-flash": {
        "cache_creation_input_token_cost": 0.0,
        "cache_read_input_token_cost": 2.8e-09,
        "input_cost_per_token": 1.4e-07,
        "input_cost_per_token_cache_hit": 2.8e-09,
        "output_cost_per_token": 2.8e-07,
        "litellm_provider": "openai",
        "mode": "chat",
        "max_input_tokens": 1048576,
        "max_output_tokens": 393216,
        "max_tokens": 393216,
        "supports_function_calling": True,
        "supports_native_streaming": True,
        "supports_prompt_caching": True,
        "supports_reasoning": True,
        "supports_response_schema": True,
        "supports_system_messages": True,
        "supports_tool_choice": True,
        "supported_endpoints": ["/v1/chat/completions"],
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
    },
}

price_path = os.path.join(
    os.path.dirname(litellm.__file__),
    "model_prices_and_context_window_backup.json",
)
with open(price_path, encoding="utf-8") as f:
    data = json.load(f)
data.update(prices)
with open(price_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

litellm.model_cost.update(prices)

print("patched LiteLLM DeepSeek V4 pricing:", ", ".join(sorted(prices)))
try:
    from litellm import ModelResponse
    from litellm.cost_calculator import completion_cost

    response = ModelResponse(
        model="deepseek/deepseek-v4-pro",
        choices=[{"message": {"role": "assistant", "content": "ok"}}],
        usage={"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000},
    )
    cost = completion_cost(response, model="deepseek/deepseek-v4-pro")
    print(f"DeepSeek V4 Pro pricing verification cost_1k_in_1k_out=${cost:.6f}")
except Exception as exc:
    raise SystemExit(f"DeepSeek V4 pricing verification failed: {exc}") from exc
PY
'''

    def _get_miniswe_reasoning_patch_script(self) -> str:
        """Patch mini-swe-agent reasoning history and OpenAI tool-call enforcement."""
        return r'''
echo "Patching mini-swe-agent reasoning preservation and tool-call enforcement..."
"$VENV_DIR/bin/python" - <<'PY'
from pathlib import Path
import textwrap

import minisweagent.models.litellm_model as litellm_model

path = Path(litellm_model.__file__)
text = path.read_text(encoding="utf-8")

helper = textwrap.dedent(
    """
def _fb_extract_reasoning_content_v2(message: dict) -> str:
    direct = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(direct, str) and direct.strip():
        return direct
    provider_specific = message.get("provider_specific_fields")
    if isinstance(provider_specific, dict):
        provider_reasoning = provider_specific.get("reasoning_content") or provider_specific.get("reasoning")
        if isinstance(provider_reasoning, str) and provider_reasoning.strip():
            return provider_reasoning
    extra = message.get("extra")
    if isinstance(extra, dict):
        response = extra.get("response")
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices and isinstance(choices[0], dict):
                response_message = choices[0].get("message") or {}
                if isinstance(response_message, dict):
                    nested = response_message.get("reasoning_content") or response_message.get("reasoning")
                    if isinstance(nested, str) and nested.strip():
                        return nested
                    nested_provider = response_message.get("provider_specific_fields")
                    if isinstance(nested_provider, dict):
                        nested_provider_reasoning = nested_provider.get("reasoning_content") or nested_provider.get("reasoning")
                        if isinstance(nested_provider_reasoning, str) and nested_provider_reasoning.strip():
                            return nested_provider_reasoning
    return ""


def _fb_preserve_reasoning_content_v2(message: dict) -> None:
    reasoning = _fb_extract_reasoning_content_v2(message)
    if not reasoning:
        if message.get("role") != "assistant":
            return
        reasoning = ""
    message["reasoning_content"] = reasoning
    provider_specific = message.get("provider_specific_fields")
    if not isinstance(provider_specific, dict):
        provider_specific = {}
    message["provider_specific_fields"] = provider_specific
    provider_specific["reasoning_content"] = reasoning


def _fb_required_tool_choice(model_name: str) -> str | None:
    # Some OpenAI-compatible GPT gateways otherwise emit a prose-only turn.
    setting = os.getenv("MSWEA_REQUIRE_TOOL_CALLS", "auto").strip().lower()
    if setting in {"1", "true", "yes", "required"}:
        return "required"
    if setting in {"0", "false", "no", "off", "none"}:
        return None
    normalized = model_name.lower()
    if normalized.startswith("openai/gpt-"):
        return "required"
    return None

"""
)

if "_fb_preserve_reasoning_content_v2" not in text:
    marker = "\n\nclass LitellmModelConfig(BaseModel):"
    if marker not in text:
        raise SystemExit("Could not patch mini-swe-agent: class marker not found")
    text = text.replace(marker, "\n\n" + helper + "class LitellmModelConfig(BaseModel):", 1)

old = "        message = response.choices[0].message.model_dump()\n        message[\"extra\"] = {"
new = (
    "        message = response.choices[0].message.model_dump()\n"
    "        _fb_preserve_reasoning_content_v2(message)\n"
    "        message[\"extra\"] = {"
)
if new not in text:
    if old not in text:
        old_v1 = "        message = response.choices[0].message.model_dump()\n        _fb_preserve_reasoning_content(message)\n        message[\"extra\"] = {"
        if old_v1 not in text:
            raise SystemExit("Could not patch mini-swe-agent: query message marker not found")
        text = text.replace(old_v1, new, 1)
    else:
        text = text.replace(old, new, 1)

old_query = """    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **(self.config.model_kwargs | kwargs),
            )
"""
new_query = """    def _query(self, messages: list[dict[str, str]], **kwargs):
        request_kwargs = self.config.model_kwargs | kwargs
        if "tool_choice" not in request_kwargs:
            tool_choice = _fb_required_tool_choice(self.config.model_name)
            if tool_choice:
                request_kwargs["tool_choice"] = tool_choice
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **request_kwargs,
            )
"""
if new_query not in text:
    if old_query not in text:
        raise SystemExit("Could not patch mini-swe-agent: query marker not found")
    text = text.replace(old_query, new_query, 1)

old_prepare = "        prepared = [{k: v for k, v in msg.items() if k != \"extra\"} for msg in messages]"
new_prepare = "\n".join(
    [
        "        prepared = []",
        "        for msg in messages:",
        "            if isinstance(msg, dict) and msg.get(\"role\") == \"assistant\":",
        "                _fb_preserve_reasoning_content_v2(msg)",
        "            entry = {k: v for k, v in msg.items() if k != \"extra\"}",
        "            if isinstance(entry, dict) and entry.get(\"role\") == \"assistant\":",
        "                _fb_preserve_reasoning_content_v2(entry)",
        "            prepared.append(entry)",
    ]
)
if new_prepare not in text:
    if old_prepare not in text:
        old_prepare_v1 = "\n".join(
            [
                "        prepared = []",
                "        for msg in messages:",
                "            if isinstance(msg, dict) and msg.get(\"role\") == \"assistant\":",
                "                _fb_preserve_reasoning_content(msg)",
                "            entry = {k: v for k, v in msg.items() if k != \"extra\"}",
                "            if isinstance(entry, dict) and entry.get(\"role\") == \"assistant\":",
                "                _fb_preserve_reasoning_content(entry)",
                "            prepared.append(entry)",
            ]
        )
        if old_prepare_v1 not in text:
            raise SystemExit("Could not patch mini-swe-agent: prepare messages marker not found")
        text = text.replace(old_prepare_v1, new_prepare, 1)
    else:
        text = text.replace(old_prepare, new_prepare, 1)

path.write_text(text, encoding="utf-8")
print(f"patched mini-swe-agent reasoning preservation and tool-call enforcement: {path}")
PY
'''

    def _get_litellm_deepseek_transform_patch_script(self) -> str:
        """Patch LiteLLM DeepSeek V4 multi-turn thinking behavior."""
        return r'''
echo "Patching LiteLLM DeepSeek V4 multi-turn thinking transform..."
"$VENV_DIR/bin/python" - <<'PY'
from pathlib import Path

import litellm.llms.deepseek.chat.transformation as deepseek_transformation

path = Path(deepseek_transformation.__file__)
text = path.read_text(encoding="utf-8")

patch = "\n".join(
    [
        "",
        "# FeatureBench patch for BerriAI/litellm#26660:",
        "# DeepSeek V4 thinking mode requires reasoning_content on every assistant",
        "# history message. Match upstream behavior: inject an empty string when",
        "# missing, while preserving real provider_specific_fields reasoning.",
        "if not getattr(DeepSeekChatConfig, \"_fb_deepseek_v4_reasoning_patch_v3\", False):",
        "    _fb_original_transform_messages_v3 = DeepSeekChatConfig._transform_messages",
        "",
        "    def _fb_ensure_reasoning_content_on_assistant_messages(messages):",
        "        if not isinstance(messages, list):",
        "            return messages",
        "        for message in messages:",
        "            if not isinstance(message, dict) or message.get(\"role\") != \"assistant\":",
        "                continue",
        "            if \"reasoning_content\" in message:",
        "                continue",
        "            provider_specific = message.get(\"provider_specific_fields\")",
        "            if isinstance(provider_specific, dict) and \"reasoning_content\" in provider_specific:",
        "                message[\"reasoning_content\"] = provider_specific[\"reasoning_content\"]",
        "            else:",
        "                message[\"reasoning_content\"] = \"\"",
        "        return messages",
        "",
        "    def _fb_transform_messages(self, messages, model, is_async=False):",
        "        messages = _fb_ensure_reasoning_content_on_assistant_messages(messages)",
        "        return _fb_original_transform_messages_v3(self, messages, model, is_async)",
        "",
        "    DeepSeekChatConfig._transform_messages = _fb_transform_messages",
        "    DeepSeekChatConfig._fb_deepseek_v4_reasoning_patch_v3 = True",
        "",
    ]
)

if "_fb_deepseek_v4_reasoning_patch_v3" not in text:
    text = text.rstrip() + "\n" + patch + "\n"
    path.write_text(text, encoding="utf-8")

print(f"patched LiteLLM DeepSeek transform: {path}")
PY
'''

    @property
    def install_script(self) -> str:
        """Installation script for mini-swe-agent CLI."""
        version = self._kwargs.get("version") or self.env_vars.get("MINI_SWE_AGENT_VERSION")
        version_str = str(version).strip() if version is not None else ""
        pkg = "mini-swe-agent" if not version_str else f"mini-swe-agent=={version_str}"
        pkg_escaped = shlex.quote(pkg)

        return f"""#!/bin/bash
set -e

echo "Installing mini-swe-agent..."

if ! command -v python3 >/dev/null 2>&1 || ! python3 -m venv --help >/dev/null 2>&1; then
  apt-get update
  apt-get install -y python3 python3-pip python3-venv
fi

CACHE_ROOT="${{AGENT_DOWNLOAD_CACHE:-/download}}"
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/pip"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"

VENV_DIR="/opt/mini-swe-agent-venv"
if [ ! -x "$VENV_DIR/bin/python" ] || ! "$VENV_DIR/bin/python" -c "import minisweagent" >/dev/null 2>&1; then
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/pip" install {pkg_escaped}
else
  echo "mini-swe-agent already installed in $VENV_DIR"
fi

# Keep the shared agent venv complete for both the official mini-swe-agent and
# wrappers that use LiteLLM's MCP completion path.  These extras are not
# guaranteed to be installed by every mini-swe-agent release or cached venv.
echo "Installing mini-swe-agent runtime dependencies..."
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check -q pyflakes "litellm[proxy]"
"$VENV_DIR/bin/python" -c '
import sys
sys.path = [p for p in sys.path if p not in ("", "/testbed")]
import fastapi  # noqa: F401
import litellm  # noqa: F401
import orjson  # noqa: F401
from litellm.responses.mcp.chat_completions_handler import acompletion_with_mcp  # noqa: F401

print("mini-swe-agent LiteLLM completion dependencies available")
'

"$VENV_DIR/bin/python" -m minisweagent.run.mini --help >/dev/null 2>&1 || true

{self._get_litellm_price_patch_script()}

{self._get_miniswe_reasoning_patch_script()}

{self._get_litellm_deepseek_transform_patch_script()}

echo "mini-swe-agent installation complete"
"""

    def pre_run_hook(self, container, log_file) -> bool:
        """Create agent logs directory before running."""
        self.cm.exec_command(container, "mkdir -p /agent-logs", log_file=log_file)
        return True

    def post_run_hook(self, container, log_file) -> bool:
        """Persist mini-swe-agent CLI output (best-effort)."""
        log_dir = Path(log_file).parent
        try:
            self.cm.copy_from_container(
                container,
                "/agent-logs/mini_swe_agent_output.log",
                log_dir / "mini_swe_agent_output.log",
            )
        except Exception:
            pass
        return True

    def failure_hook(self, container, log_file: Path) -> None:
        """Collect mini-swe-agent output on failures (best-effort)."""
        log_dir = Path(log_file).parent
        try:
            self.cm.copy_from_container(
                container,
                "/agent-logs/mini_swe_agent_output.log",
                log_dir / "mini_swe_agent_output.log",
            )
        except Exception:
            pass

    def get_run_command(self, instruction: str) -> str:
        """Get the command to run mini-swe-agent."""
        instruction = (
            f"{instruction}\n\n"
            "## Mandatory Native Tool Calling\n"
            "Every assistant turn MUST contain at least one native `bash` tool call. "
            "Never substitute a Markdown code block, a textual command, or an explanation "
            "for a tool call. If you need to think, make a small bash call first. "
            "A response without a native tool call is invalid.\n"
        )
        escaped_instruction = shlex.quote(instruction)
        model_name = self._get_model_name()
        is_openai_gpt = model_name.lower().startswith("openai/gpt-")
        reasoning_effort = self._get_reasoning_effort()
        config_args = ""
        format_error_limit = self.env_vars.get("MSWEA_MAX_CONSECUTIVE_FORMAT_ERRORS")
        if format_error_limit is None and is_openai_gpt:
            format_error_limit = "10"
        if reasoning_effort or self._kwargs.get("max_iters") or format_error_limit:
            # Passing any -c overrides mini-swe-agent's default config list, so
            # explicitly include mini.yaml before adding the LiteLLM parameter.
            config_parts = [" -c mini.yaml"]
            if reasoning_effort:
                config_parts.append(
                    f" -c model.model_kwargs.reasoning_effort={shlex.quote(reasoning_effort)}"
                )
            if self._kwargs.get("max_iters"):
                config_parts.append(f" -c agent.step_limit={shlex.quote(str(self._kwargs['max_iters']))}")
            if format_error_limit:
                config_parts.append(
                    " -c agent.max_consecutive_format_errors="
                    f"{shlex.quote(str(format_error_limit))}"
                )
            config_args = "".join(config_parts)
        return (
            "set -o pipefail; "
            f'"$MINI_SWE_AGENT_PYTHON" -I - -m {shlex.quote(model_name)} -t {escaped_instruction} '
            f"-y --exit-immediately{config_args} "
            "<<'PY' | tee /agent-logs/mini_swe_agent_output.log\n"
            "import runpy\n"
            "import sys\n"
            "\n"
            "sys.path = [p for p in sys.path if p and not p.startswith('/testbed')]\n"
            "\n"
            "def _arch_patch_deepseek_reasoning_transform():\n"
            "    try:\n"
            "        from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig\n"
            "    except Exception as exc:\n"
            "        print(f'[mini_swe_agent][DeepSeekTransformPatchWarning] {exc}', flush=True)\n"
            "        return\n"
            "    if getattr(DeepSeekChatConfig, '_fb_runtime_deepseek_v4_reasoning_patch', False):\n"
            "        return\n"
            "    original_transform_messages = DeepSeekChatConfig._transform_messages\n"
            "\n"
            "    def ensure_reasoning_content(messages):\n"
            "        if not isinstance(messages, list):\n"
            "            return messages\n"
            "        for message in messages:\n"
            "            if not isinstance(message, dict) or message.get('role') != 'assistant':\n"
            "                continue\n"
            "            if 'reasoning_content' in message:\n"
            "                continue\n"
            "            provider_specific = message.get('provider_specific_fields')\n"
            "            if isinstance(provider_specific, dict) and 'reasoning_content' in provider_specific:\n"
            "                message['reasoning_content'] = provider_specific['reasoning_content']\n"
            "            else:\n"
            "                message['reasoning_content'] = ''\n"
            "        return messages\n"
            "\n"
            "    def transform_messages_with_reasoning(self, messages, model, is_async=False):\n"
            "        messages = ensure_reasoning_content(messages)\n"
            "        return original_transform_messages(self, messages, model, is_async)\n"
            "\n"
            "    DeepSeekChatConfig._transform_messages = transform_messages_with_reasoning\n"
            "    DeepSeekChatConfig._fb_runtime_deepseek_v4_reasoning_patch = True\n"
            "    print('[mini_swe_agent][DeepSeekTransformPatch] enabled', flush=True)\n"
            "\n"
            "_arch_patch_deepseek_reasoning_transform()\n"
            "\n"
            "def _arch_extract_reasoning_content(message):\n"
            "    if not isinstance(message, dict):\n"
            "        return ''\n"
            "    direct = message.get('reasoning_content') or message.get('reasoning')\n"
            "    if direct:\n"
            "        return str(direct)\n"
            "    provider_specific = message.get('provider_specific_fields') or {}\n"
            "    if isinstance(provider_specific, dict):\n"
            "        provider_reasoning = provider_specific.get('reasoning_content') or provider_specific.get('reasoning')\n"
            "        if provider_reasoning:\n"
            "            return str(provider_reasoning)\n"
            "    extra = message.get('extra') or {}\n"
            "    response = extra.get('response')\n"
            "    try:\n"
            "        if hasattr(response, 'model_dump'):\n"
            "            response = response.model_dump()\n"
            "    except Exception:\n"
            "        pass\n"
            "    if not isinstance(response, dict):\n"
            "        return ''\n"
            "    choices = response.get('choices') or []\n"
            "    if not choices:\n"
            "        return ''\n"
            "    choice = choices[0]\n"
            "    try:\n"
            "        if hasattr(choice, 'model_dump'):\n"
            "            choice = choice.model_dump()\n"
            "    except Exception:\n"
            "        pass\n"
            "    if not isinstance(choice, dict):\n"
            "        return ''\n"
            "    choice_message = choice.get('message') or {}\n"
            "    try:\n"
            "        if hasattr(choice_message, 'model_dump'):\n"
            "            choice_message = choice_message.model_dump()\n"
            "    except Exception:\n"
            "        pass\n"
            "    if isinstance(choice_message, dict):\n"
            "        nested = choice_message.get('reasoning_content') or choice_message.get('reasoning')\n"
            "        if nested:\n"
            "            return str(nested)\n"
            "        provider_specific = choice_message.get('provider_specific_fields') or {}\n"
            "        if isinstance(provider_specific, dict):\n"
            "            nested_provider = provider_specific.get('reasoning_content') or provider_specific.get('reasoning')\n"
            "            if nested_provider:\n"
            "                return str(nested_provider)\n"
            "    return ''\n"
            "\n"
            "def _arch_preserve_reasoning_content(message):\n"
            "    reasoning = _arch_extract_reasoning_content(message)\n"
            "    if not reasoning or not isinstance(message, dict):\n"
            "        if isinstance(message, dict) and message.get('role') == 'assistant':\n"
            "            reasoning = ''\n"
            "        else:\n"
            "            return reasoning\n"
            "    message['reasoning_content'] = reasoning\n"
            "    provider_specific = message.get('provider_specific_fields')\n"
            "    if not isinstance(provider_specific, dict):\n"
            "        provider_specific = {}\n"
            "        message['provider_specific_fields'] = provider_specific\n"
            "    provider_specific['reasoning_content'] = reasoning\n"
            "    return reasoning\n"
            "\n"
            "try:\n"
            "    from minisweagent.agents.default import DefaultAgent\n"
            "    from minisweagent.models.litellm_model import LitellmModel\n"
            "    _arch_original_add_messages = DefaultAgent.add_messages\n"
            "    _arch_original_step = DefaultAgent.step\n"
            "    _arch_original_prepare_messages = LitellmModel._prepare_messages_for_api\n"
            "\n"
            "    def _arch_step_with_progress(self):\n"
            "        print(f'[mini_swe_agent] step {self.n_calls + 1} start', flush=True)\n"
            "        result = _arch_original_step(self)\n"
            "        print(f'[mini_swe_agent] step {self.n_calls} end', flush=True)\n"
            "        return result\n"
            "\n"
            "    def _arch_add_messages_with_reasoning(self, *messages):\n"
            "        for message in messages:\n"
            "            if isinstance(message, dict) and (message.get('role') == 'assistant' or message.get('object') == 'response'):\n"
            "                _arch_preserve_reasoning_content(message)\n"
            "        result = _arch_original_add_messages(self, *messages)\n"
            "        for message in messages:\n"
            "            if not isinstance(message, dict):\n"
            "                continue\n"
            "            is_response = message.get('object') == 'response'\n"
            "            is_assistant = message.get('role') == 'assistant' or is_response\n"
            "            if is_assistant:\n"
            "                header = 'Responses' if is_response else 'Assistant'\n"
            "                print(f'\\n[mini_swe_agent][{header}]', flush=True)\n"
            "                from minisweagent.models.utils.content_string import get_content_string\n"
            "                print(get_content_string(message), flush=True)\n"
            "                reasoning = _arch_extract_reasoning_content(message)\n"
            "                if reasoning:\n"
            "                    print('\\n[mini_swe_agent][Reasoning]\\n' + reasoning + '\\n', flush=True)\n"
            "        return result\n"
            "\n"
            "    def _arch_prepare_messages_with_reasoning(self, messages):\n"
            "        for message in messages:\n"
            "            if isinstance(message, dict) and message.get('role') == 'assistant':\n"
            "                _arch_preserve_reasoning_content(message)\n"
            "        prepared = _arch_original_prepare_messages(self, messages)\n"
            "        for message in prepared:\n"
            "            if isinstance(message, dict) and message.get('role') == 'assistant':\n"
            "                _arch_preserve_reasoning_content(message)\n"
            "        return prepared\n"
            "\n"
            "    DefaultAgent.step = _arch_step_with_progress\n"
            "    DefaultAgent.add_messages = _arch_add_messages_with_reasoning\n"
            "    LitellmModel._prepare_messages_for_api = _arch_prepare_messages_with_reasoning\n"
            "except Exception as exc:\n"
            "    print(f'[mini_swe_agent][ReasoningPatchWarning] {exc}', flush=True)\n"
            "\n"
            "sys.argv = ['minisweagent.run.mini', *sys.argv[1:]]\n"
            "runpy.run_module('minisweagent.run.mini', run_name='__main__')\n"
            "PY"
        )

    def get_env_setup_script(self) -> str:
        """Get environment setup script for mini-swe-agent."""
        lines = ["#!/bin/bash", ""]

        provider = self._get_model_provider()
        env_settings: Dict[str, str] = {"MSWEA_CONFIGURED": "true"}
        env_settings["MINI_SWE_AGENT_PYTHON"] = "/opt/mini-swe-agent-venv/bin/python"
        env_settings["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

        # api key setup
        mswea_api_key = self.env_vars.get("MSWEA_API_KEY")
        if not mswea_api_key:
            raise RuntimeError("MSWEA_API_KEY is required for mini_swe_agent")
        env_settings["MSWEA_API_KEY"] = mswea_api_key
        provider_api_key_env = self._get_provider_api_key_env(provider)
        if provider_api_key_env:
            env_settings[provider_api_key_env] = mswea_api_key

        # base url setup
        mswea_base_url = self.env_vars.get("MSWEA_BASE_URL")
        if mswea_base_url:
            env_settings["MSWEA_BASE_URL"] = mswea_base_url
            for provider_base_key in self._get_provider_base_url_envs(provider):
                env_settings[provider_base_key] = mswea_base_url

        # cost tracking setup
        mswea_cost_tracking = self.env_vars.get("MSWEA_COST_TRACKING")
        if mswea_cost_tracking is not None and str(mswea_cost_tracking).strip():
            env_settings["MSWEA_COST_TRACKING"] = str(mswea_cost_tracking).strip()
        else:
            # Avoid hard failures when LiteLLM has no price mapping for the selected model.
            env_settings["MSWEA_COST_TRACKING"] = "ignore_errors"

        # Add any additional env vars
        for key, value in self.env_vars.items():
            if key not in env_settings and value:
                env_settings[key] = value

        # proxy setup
        _no_rewrite_keys = {
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
            "NO_PROXY",
            "no_proxy",
        }
        for key, value in env_settings.items():
            if value:
                value_str = str(value)
                if key not in _no_rewrite_keys and (
                    "localhost" in value_str or "127.0.0.1" in value_str
                ):
                    value_str = value_str.replace("localhost", DOCKER_HOST_GATEWAY)
                    value_str = value_str.replace("127.0.0.1", DOCKER_HOST_GATEWAY)
                escaped_value = value_str.replace("'", "'\\''")
                lines.append(f"export {key}='{escaped_value}'")

        # Handle --runtime-proxy off mode.
        lines.extend(self._get_proxy_unset_lines())

        return "\n".join(lines)
