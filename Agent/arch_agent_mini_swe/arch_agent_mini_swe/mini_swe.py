from __future__ import annotations

import copy
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .prompts import SYSTEM_TEMPLATE


def _import_minisweagent() -> ModuleType:
    try:
        import minisweagent  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mini-swe-agent is not importable. Run this agent with the conda "
            "environment that has mini-swe-agent installed, for example "
            "`/home/wpd/miniconda3/envs/openhands/bin/python run_agent.py ...`."
        ) from exc
    return minisweagent


@contextmanager
def _without_repo_on_import_path(repo_path: Path):
    """Prevent target repo packages from shadowing agent runtime dependencies."""
    original_path = list(sys.path)
    repo_resolved = repo_path.resolve()
    cwd_resolved = Path.cwd().resolve()

    def should_keep(entry: str) -> bool:
        path = Path(entry or os.getcwd()).resolve()
        return path not in {repo_resolved, cwd_resolved}

    sys.path = [entry for entry in sys.path if should_keep(entry)]
    try:
        yield
    finally:
        sys.path = original_path


def build_default_model_config(model_name: str, model_class: str = "litellm") -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "drop_params": True,
        "temperature": 0.2,
    }

    return {
        "model_class": model_class,
        "model_name": model_name,
        "cost_tracking": os.getenv("MSWEA_COST_TRACKING", "ignore_errors"),
        "model_kwargs": model_kwargs,
        "observation_template": (
            "{%- if output.output | length < 12000 -%}\n"
            "{\n"
            '  "returncode": {{ output.returncode }},\n'
            '  "output": {{ output.output | tojson }}'
            "{%- if output.exception_info %}, "
            '"exception_info": {{ output.exception_info | tojson }}{% endif %}\n'
            "}\n"
            "{%- else -%}\n"
            "{\n"
            '  "returncode": {{ output.returncode }},\n'
            '  "output_head": {{ output.output[:6000] | tojson }},\n'
            '  "output_tail": {{ output.output[-6000:] | tojson }},\n'
            '  "elided_chars": {{ output.output | length - 12000 }}'
            "{%- if output.exception_info %}, "
            '"exception_info": {{ output.exception_info | tojson }}{% endif %}\n'
            "}\n"
            "{%- endif -%}"
        ),
        "format_error_template": (
            "Every response must use at least one bash tool call. "
            "If finishing, call bash with only: "
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        ),
    }


def run_mini_stage(
    *,
    task: str,
    repo_path: Path,
    output_path: Path,
    instance_template: str,
    template_vars: dict[str, Any],
    model_config: dict[str, Any],
    step_limit: int,
    cost_limit: float,
    timeout: int,
    wall_time_limit_seconds: int,
    environment_factory: Callable[[type], Any] | None = None,
) -> dict[str, Any]:
    print(
        f"[arch_swe_agent] mini-swe import/start output={output_path} step_limit={step_limit}",
        flush=True,
    )
    with _without_repo_on_import_path(repo_path):
        _import_minisweagent()

        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models import get_model
        from minisweagent.models import litellm_model
        from minisweagent.models.litellm_model import LitellmModel
        from minisweagent.models.utils.content_string import get_content_string

    if not hasattr(litellm_model, "_fb_preserve_reasoning_content"):
        original_query = LitellmModel.query
        original_prepare_messages = LitellmModel._prepare_messages_for_api

        def query_with_preserved_reasoning(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict:
            message = original_query(self, messages, **kwargs)
            if isinstance(message, dict) and message.get("role") == "assistant":
                _preserve_reasoning_content(message)
            return message

        def prepare_messages_with_preserved_reasoning(
            self, messages: list[dict[str, Any]]
        ) -> list[dict[str, Any]]:
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "assistant":
                    _preserve_reasoning_content(message)
            prepared = original_prepare_messages(self, messages)
            for message in prepared:
                if isinstance(message, dict) and message.get("role") == "assistant":
                    _preserve_reasoning_content(message)
            return prepared

        LitellmModel.query = query_with_preserved_reasoning
        LitellmModel._prepare_messages_for_api = prepare_messages_with_preserved_reasoning

    class StreamingAgent(DefaultAgent):
        def add_messages(self, *messages: dict) -> list[dict]:
            for msg in messages:
                if msg.get("role") == "assistant":
                    _preserve_reasoning_content(msg)
                role = msg.get("role") or msg.get("type", "unknown")
                content = get_content_string(msg)
                if role == "assistant":
                    header = f"mini-swe-agent step={self.n_calls} cost=${self.cost:.2f}"
                else:
                    header = str(role).capitalize()
                print(f"\n[arch_swe_agent][{header}]", flush=True)
                reasoning_content = _get_reasoning_content(msg)
                if reasoning_content:
                    print("[arch_swe_agent][Reasoning]", flush=True)
                    print(_clip_for_log(reasoning_content), flush=True)
                print(_clip_for_log(content), flush=True)
            return super().add_messages(*messages)

        def step(self) -> list[dict]:
            print(f"[arch_swe_agent] mini-swe step {self.n_calls + 1} start", flush=True)
            result = super().step()
            print(f"[arch_swe_agent] mini-swe step {self.n_calls} end", flush=True)
            return result

    print("[arch_swe_agent] mini-swe building model/environment", flush=True)
    print(
        "[arch_swe_agent] model_kwargs="
        f"{_safe_model_kwargs_for_log(model_config.get('model_kwargs', {}))}",
        flush=True,
    )
    model = get_model(config=copy.deepcopy(model_config))
    if environment_factory is None:
        env = LocalEnvironment(
            cwd=str(repo_path),
            timeout=timeout,
            env={
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
        )
    else:
        env = environment_factory(LocalEnvironment)
    format_error_limit = os.getenv("MSWEA_MAX_CONSECUTIVE_FORMAT_ERRORS")
    if format_error_limit is None:
        model_name = str(model_config.get("model_name", "")).lower()
        format_error_limit = "10" if model_name.startswith("openai/gpt-") else "3"

    agent = StreamingAgent(
        model,
        env,
        system_template=SYSTEM_TEMPLATE,
        instance_template=instance_template,
        step_limit=step_limit,
        cost_limit=cost_limit,
        wall_time_limit_seconds=wall_time_limit_seconds,
        max_consecutive_format_errors=int(format_error_limit),
        output_path=output_path,
    )
    print("[arch_swe_agent] mini-swe agent.run start", flush=True)
    result = agent.run(task, repo_path=str(repo_path), **template_vars)
    print(
        f"[arch_swe_agent] mini-swe agent.run done exit_status={result.get('exit_status', '')}",
        flush=True,
    )
    return result


def _clip_for_log(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n...<arch_swe_agent log clipped {len(text) - limit} chars>...\n{tail}"


def _get_reasoning_content(message: dict[str, Any]) -> str:
    direct = message.get("reasoning_content")
    if isinstance(direct, str) and direct.strip():
        return direct
    provider_specific = message.get("provider_specific_fields")
    if isinstance(provider_specific, dict):
        provider_reasoning = provider_specific.get("reasoning_content") or provider_specific.get("reasoning")
        if isinstance(provider_reasoning, str) and provider_reasoning.strip():
            return provider_reasoning
    extra = message.get("extra")
    if not isinstance(extra, dict):
        return ""
    response = extra.get("response")
    response = _as_dict(response)
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = _as_dict(choices[0])
    if not isinstance(choice, dict):
        return ""
    response_message = _as_dict(choice.get("message"))
    if not isinstance(response_message, dict):
        return ""
    nested = response_message.get("reasoning_content") or response_message.get("reasoning")
    if isinstance(nested, str) and nested.strip():
        return nested
    provider_specific = response_message.get("provider_specific_fields")
    if isinstance(provider_specific, dict):
        nested_provider = provider_specific.get("reasoning_content") or provider_specific.get("reasoning")
        if isinstance(nested_provider, str) and nested_provider.strip():
            return nested_provider
    return ""


def _preserve_reasoning_content(message: dict[str, Any]) -> str:
    reasoning_content = _get_reasoning_content(message)
    if not reasoning_content and message.get("role") != "assistant":
        return ""
    message["reasoning_content"] = reasoning_content
    provider_specific = message.get("provider_specific_fields")
    if not isinstance(provider_specific, dict):
        provider_specific = {}
        message["provider_specific_fields"] = provider_specific
    provider_specific["reasoning_content"] = reasoning_content
    return reasoning_content


def _as_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            return value
    return value


def _safe_model_kwargs_for_log(model_kwargs: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(model_kwargs)
    for key in list(redacted):
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            redacted[key] = "***"
    return redacted
