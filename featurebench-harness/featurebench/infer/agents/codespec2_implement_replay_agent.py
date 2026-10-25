"""FeatureBench adapter for implementation-only CodeSpec2 replay experiments."""

from __future__ import annotations

import shlex
from pathlib import Path

from featurebench.infer.agents.codespec2_agent import CodeSpec2Agent


class CodeSpec2ImplementReplayAgent(CodeSpec2Agent):
    """Reuse fixed CodeSpec2 artifacts and run only the implementation model."""

    DEFAULT_REFERENCE_RUN = Path(
        "/home/wpd/workspace/FeatureBench/FeatureBench/runs/codespec2_sample/2026-08-12__22-31-43"
    )

    @property
    def name(self) -> str:
        return "codespec2_implement_replay"

    def pre_run_setup(self, container, instance, log_file: Path) -> bool:
        if not super().pre_run_setup(container, instance, log_file):
            return False
        source_run = Path(
            self.env_vars.get("CODESPEC2_REPLAY_SOURCE_RUN") or self.DEFAULT_REFERENCE_RUN
        ).expanduser().resolve()
        candidates = sorted(
            (source_run / "run_outputs" / instance.instance_id / "attempt-1" / "codespec2").glob(
                "*/artifacts"
            )
        )
        candidates = [path for path in candidates if path.is_dir()]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one reference artifact directory for {instance.instance_id} "
                f"under {source_run}, found {len(candidates)}: {candidates}"
            )
        required = {
            "spec.md",
            "spec_test.py",
        }
        missing = sorted(name for name in required if not (candidates[0] / name).is_file())
        if missing:
            raise RuntimeError(f"Reference artifacts are incomplete for {instance.instance_id}: {missing}")
        self.cm.copy_to_container(
            container,
            candidates[0],
            "/agent-logs/codespec2_reference/artifacts",
        )
        return True

    def pre_run_hook(self, container, log_file: Path) -> bool:
        self.cm.exec_command(
            container,
            "mkdir -p /agent-logs/codespec2_replay && rm -rf /testbed/agent-logs && ln -s /agent-logs /testbed/agent-logs",
            log_file=log_file,
        )
        return True

    def get_run_command(self, instruction: str) -> str:  # noqa: ARG002
        model_name = self._get_model_name()
        implementation_steps = str(
            self._kwargs.get("max_iters")
            or self.env_vars.get("CODESPEC2_REPLAY_MAX_ITERS")
            or self.env_vars.get("CODESPEC2_MAX_ITERS")
            or "300"
        )
        return (
            "set -o pipefail; cd /testbed; "
            "test -f /installed-agent/codespec2/run_replay_agent.py || { echo 'Missing CodeSpec2 replay agent' >&2; exit 1; }; "
            "test -d /agent-logs/codespec2_reference/artifacts || { echo 'Missing fixed reference artifacts' >&2; exit 1; }; "
            "PYTHONPATH=/installed-agent:$PYTHONPATH PYTHONUNBUFFERED=1 "
            '"$MINI_SWE_AGENT_PYTHON" -u /installed-agent/codespec2/run_replay_agent.py '
            "--workspace /testbed --task-file /agent-logs/arch_swe_agent/task.json --task-id task "
            "--reference-artifacts /agent-logs/codespec2_reference/artifacts "
            f"--model {shlex.quote(model_name)} --model-class litellm "
            "--log-root /agent-logs/codespec2_replay "
            f"--implementation-step-limit {shlex.quote(implementation_steps)} "
            "2>&1 | tee /agent-logs/codespec2_replay_output.log; "
            "exit ${PIPESTATUS[0]}"
        )

    def post_run_hook(self, container, log_file: Path) -> bool:
        self._copy_replay_logs(container, log_file)
        return True

    def failure_hook(self, container, log_file: Path) -> None:
        self._copy_replay_logs(container, log_file)
        return super().failure_hook(container, log_file)

    def _copy_replay_logs(self, container, log_file: Path) -> None:
        log_dir = Path(log_file).parent
        for src, dst in (
            ("/agent-logs/codespec2_replay", log_dir / "codespec2_replay"),
            ("/agent-logs/codespec2_replay_output.log", log_dir / "codespec2_replay_output.log"),
            ("/agent-logs/codespec2_reference", log_dir / "codespec2_reference"),
        ):
            try:
                self.cm.copy_from_container(container, src, dst)
            except Exception:
                pass
