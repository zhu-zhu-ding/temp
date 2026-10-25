"""FeatureBench adapters for isolated arch_swe_agent_simple ablations."""

from __future__ import annotations

from featurebench.infer.agents.arch_swe_agent_simple import ArchSweAgentSimple


class _ArchSweAgentSimpleAblation(ArchSweAgentSimple):
    """Reuse the baseline wrapper while selecting one runner ablation mode."""

    ablation = ""
    agent_name = ""

    @property
    def name(self) -> str:
        return self.agent_name

    def get_run_command(self, instruction: str) -> str:
        command = super().get_run_command(instruction)
        marker = " --log-root /agent-logs/arch_swe_agent_simple "
        replacement = f" --ablation {self.ablation}{marker}"
        if marker not in command:
            raise RuntimeError("Could not inject arch_swe_agent_simple ablation mode")
        return command.replace(marker, replacement, 1)


class ArchSweAgentWithoutGeneratedTests(_ArchSweAgentSimpleAblation):
    """Ablation removing the generated P2P/F2P test stage."""

    ablation = "without_generated_tests"
    agent_name = "arch_swe_agent_wo_generated_tests"


class ArchSweAgentWithoutMultiViewEvidence(_ArchSweAgentSimpleAblation):
    """Ablation removing the design example and repository-evidence requirements."""

    ablation = "without_multi_view_evidence"
    agent_name = "arch_swe_agent_wo_multi_view_evidence"


class ArchSweAgentWithoutExecutableArchitecture(_ArchSweAgentSimpleAblation):
    """Ablation replacing the executable spec with a text design document."""

    ablation = "without_executable_architecture"
    agent_name = "arch_swe_agent_wo_executable_architecture"
