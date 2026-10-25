"""
Claude Code agent implementation.
"""

import json
import shlex
from pathlib import Path
from typing import Dict, List, Optional

from featurebench.infer.agents.base import BaseAgent
from featurebench.infer.container import DOCKER_HOST_GATEWAY



class ClaudeCodeAgent(BaseAgent):
    """Claude Code agent for FeatureBench inference."""
    _MAX_TURNS_ERROR_MARKERS = (
        "max turns",
        "max-turns",
        "max_turns",
        "maximum turns",
        "turn limit",
        "reached the maximum",
        "maximum number of turns",
    )
    
    # Allowed tools for Claude Code
    ALLOWED_TOOLS = [
        "Bash",
        "Edit",
        "Write",
        "Read",
        "Glob",
        "Grep",
        "LS",
        "WebFetch",
        "NotebookEdit", 
        "NotebookRead",
        "TodoRead",
        "TodoWrite",
        "Agent",
    ]
    
    @property
    def name(self) -> str:
        return "claude_code"
    
    @property
    def install_script(self) -> str:
        """Installation script for Claude Code."""
        version = self._kwargs.get("version") or self.env_vars.get("CLAUDE_CODE_VERSION") or "latest"
        
        return f"""#!/bin/bash
set -e

echo "Installing Claude Code agent..."

# Update package manager
apt-get update
apt-get install -y curl ca-certificates tar xz-utils

CACHE_ROOT="${{AGENT_DOWNLOAD_CACHE:-/download}}"
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT/npm" "$CACHE_ROOT/node"

export npm_config_cache="$CACHE_ROOT/npm"
export NPM_CONFIG_CACHE="$CACHE_ROOT/npm"

NODE_VERSION="22.16.0"
NODE_DIR="/opt/featurebench/node"
NODE_TARBALL="$CACHE_ROOT/node/node-v$NODE_VERSION-linux-x64.tar.xz"
mkdir -p "$(dirname "$NODE_TARBALL")" "$(dirname "$NODE_DIR")"

if [ ! -x "$NODE_DIR/bin/node" ]; then
    if [ ! -s "$NODE_TARBALL" ]; then
        curl -L --fail --retry 3 --retry-delay 2 \
            -o "$NODE_TARBALL" \
            "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz"
    fi
    rm -rf "$NODE_DIR"
    mkdir -p "$NODE_DIR"
    tar -xJf "$NODE_TARBALL" --strip-components=1 -C "$NODE_DIR"
fi

export PATH="$NODE_DIR/bin:$PATH"

# Verify npm
node -v
npm -v

# Install Claude Code (downloads cached via NPM_CONFIG_CACHE)
npm install -g @anthropic-ai/claude-code@{version}

# Verify installation
claude --version || echo "Claude Code installed"

echo "Claude Code installation complete"
"""
    
    def get_run_command(self, instruction: str) -> str:
        """Get the command to run Claude Code."""
        full_instruction = instruction.rstrip()

        escaped_instruction = shlex.quote(full_instruction)
        allowed_tools = " ".join(self.ALLOWED_TOOLS)
        max_turns = str(
            self._kwargs.get("max_turns")
            or self._kwargs.get("max_iters")
            or self.env_vars.get("CLAUDE_CODE_MAX_TURNS")
            or "200"
        ).strip()
        max_turns_arg = f" --max-turns {shlex.quote(max_turns)}" if max_turns else ""
        
        return (
            f"export PATH=/opt/featurebench/node/bin:$PATH; "
            f"claude --verbose "
            f"-p {escaped_instruction} --allowedTools {allowed_tools} "
            f"{max_turns_arg} "
            # f"-p \"please create a hello-world.txt containing your self-introduction under testbed\" --allowedTools {allowed_tools} "
            f"--output-format stream-json | tee /agent-logs/claude_code_stream_output.jsonl"
        )
    
    def pre_run_hook(self, container, log_file) -> bool:
        """
        Create agent logs directory before running.
        """
        self.cm.exec_command(container, "mkdir -p /agent-logs", log_file=log_file)

        return True

    def post_run_hook(self, container, log_file) -> bool:
        """
        Save claude_code_stream_output.jsonl
        Check if the last line of claude_code_stream_output.jsonl is a success
        If not, return False
        If success, return True
        """
        log_dir = Path(log_file).parent

        claude_code_stream_output_copied = self.cm.copy_from_container(
            container,
            "/agent-logs/claude_code_stream_output.jsonl",
            log_dir / "claude_code_stream_output.jsonl"
        )

        if not claude_code_stream_output_copied:
            self.logger.error("Failed to copy claude_code_stream_output.jsonl from container")
            return False

        # check the last line of claude_code_stream_output.jsonl as a json object
        # if key "type" is "result" and key "subtype" is "success", then return True
        # otherwise return False
        with open(log_dir / "claude_code_stream_output.jsonl", "r", encoding="utf-8") as f:
            last_line = f.readlines()[-1]
        last_line_json = json.loads(last_line)

        is_result_success = (
            last_line_json.get("type") == "result" and last_line_json.get("subtype") == "success"
        )
        if not is_result_success:
            self.logger.error(
                "Last line of claude_code_stream_output.jsonl is not result/success; agent may not have finished properly"
            )
            return False

        # Strengthened success criteria: even if the CLI reports a result/success,
        # treat runs with is_error=true as failures. This avoids counting API
        # errors (e.g., token exhaustion) as successful attempts.
        if "is_error" not in last_line_json:
            self.logger.error(
                "claude_code_stream_output.jsonl last result is missing is_error; treating as failure"
            )
            return False

        is_error = last_line_json.get("is_error")
        if is_error is False:
            return True

        if is_error is True:
            result = last_line_json.get("result")
            if isinstance(result, str):
                result_preview = result.replace("\n", " ")[:300]
            else:
                result_preview = str(result)[:300]
            self.logger.error(
                "Claude Code reported is_error=true; treating as failure. "
                f"Result preview: {result_preview}"
            )
            return False

        self.logger.error(f"Claude Code reported unexpected is_error={is_error!r}; treating as failure")
        return False

    def _copy_stream_output(self, container, log_file: Path) -> Optional[Path]:
        log_dir = Path(log_file).parent
        output_path = log_dir / "claude_code_stream_output.jsonl"
        try:
            copied = self.cm.copy_from_container(
                container,
                "/agent-logs/claude_code_stream_output.jsonl",
                output_path,
            )
        except Exception:
            copied = False
        return output_path if copied else None

    def _read_last_stream_event(self, output_path: Path) -> Optional[dict]:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                return None
            return json.loads(lines[-1])
        except Exception:
            return None

    def _is_max_turns_result(self, event: Optional[dict]) -> bool:
        if not isinstance(event, dict):
            return False
        if event.get("type") != "result":
            return False
        text_parts = [
            str(event.get(key) or "")
            for key in ("result", "subtype", "error", "message", "reason", "terminal_reason", "stop_reason")
        ]
        errors = event.get("errors")
        if isinstance(errors, list):
            text_parts.extend(str(error) for error in errors)
        text = " ".join(text_parts).lower()
        return any(marker in text for marker in self._MAX_TURNS_ERROR_MARKERS)

    def _has_repository_diff(self, container, log_file: Path) -> bool:
        exit_code, output = self.cm.exec_command(
            container,
            "cd /testbed && git status --porcelain --untracked-files=all",
            log_file=log_file,
        )
        return exit_code == 0 and bool(output.strip())

    def should_accept_failed_run(self, container, log_file: Path, exit_code: int) -> bool:
        """Accept max-turns/timeout termination if Claude Code already produced a patch."""
        stream_output = self._copy_stream_output(container, log_file)
        event = self._read_last_stream_event(stream_output) if stream_output else None
        hit_max_turns = self._is_max_turns_result(event)
        hit_timeout = self._infer_log_has_timeout_marker(log_file)
        if not hit_max_turns and not hit_timeout:
            return False
        if not self._has_repository_diff(container, log_file):
            reason = "timeout" if hit_timeout else "max turns"
            self.logger.error(f"Claude Code reached {reason}, but no repository diff exists")
            return False
        reason = "timeout" if hit_timeout else "max turns"
        self.logger.warning(
            f"Claude Code reached {reason}; accepting existing repository diff as the final patch"
        )
        return True

    def failure_hook(self, container, log_file) -> None:
        log_dir = Path(log_file).parent

        # Copy stream output if present (can be partial on failures/timeouts).
        try:
            self.cm.copy_from_container(
                container,
                "/agent-logs/claude_code_stream_output.jsonl",
                log_dir / "claude_code_stream_output.jsonl",
            )
        except Exception:
            pass


    def get_env_setup_script(self) -> str:
        """Get environment setup script for Claude Code."""
        lines = ["#!/bin/bash", ""]
        
        # Required environment variables
        required_vars = {
            "ANTHROPIC_API_KEY": self.env_vars.get("ANTHROPIC_API_KEY", ""),
            "FORCE_AUTO_BACKGROUND_TASKS": "1",
            "ENABLE_BACKGROUND_TASKS": "1",
            "DISABLE_TELEMETRY": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_EFFORT_LEVEL": self.env_vars.get("CLAUDE_CODE_EFFORT_LEVEL", "max"),
        }
        
        # Add model if specified (CLI only)
        model = self._kwargs.get("model")
        if model:
            # Remove provider prefix if present
            if "/" in model:
                model = model.split("/")[-1]
            required_vars["ANTHROPIC_MODEL"] = model

            # Force Claude Code internal routing defaults to the selected model.
            # This prevents automatic fallback to smaller families (e.g. haiku/sonnet)
            # when you want everything to run on the specified model.
            required_vars["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
            required_vars["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
            required_vars["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
            capabilities = self.env_vars.get(
                "CLAUDE_CODE_MODEL_SUPPORTED_CAPABILITIES",
                "effort,xhigh_effort,max_effort,thinking,adaptive_thinking,interleaved_thinking",
            )
            required_vars["ANTHROPIC_DEFAULT_HAIKU_MODEL_SUPPORTED_CAPABILITIES"] = capabilities
            required_vars["ANTHROPIC_DEFAULT_SONNET_MODEL_SUPPORTED_CAPABILITIES"] = capabilities
            required_vars["ANTHROPIC_DEFAULT_OPUS_MODEL_SUPPORTED_CAPABILITIES"] = capabilities
        
        # Add any additional env vars
        for key, value in self.env_vars.items():
            if key not in required_vars and value:
                required_vars[key] = value
        
        for key, value in required_vars.items():
            if value:
                # Replace localhost/127.0.0.1 with Docker host gateway for bridge mode
                value_str = str(value)
                if 'localhost' in value_str or '127.0.0.1' in value_str:
                    value_str = value_str.replace('localhost', DOCKER_HOST_GATEWAY)
                    value_str = value_str.replace('127.0.0.1', DOCKER_HOST_GATEWAY)
                escaped_value = value_str.replace("'", "'\\''")
                lines.append(f"export {key}='{escaped_value}'")

        lines.extend(self._get_proxy_unset_lines())
        
        # Add Node setup
        lines.extend([
            "",
            "# Load Node",
            'export PATH="/opt/featurebench/node/bin:$PATH"',
        ])
        
        return "\n".join(lines)
