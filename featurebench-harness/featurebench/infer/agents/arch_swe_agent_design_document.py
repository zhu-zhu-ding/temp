"""Architecture-behavior-implementation variant of the mini-swe agent."""

from __future__ import annotations

from featurebench.infer.agents.arch_swe_agent_simple import ArchSweAgentSimple


class ArchSweAgentDesignDocument(ArchSweAgentSimple):
    """Generate architecture and behavior documents, then implement the task."""

    @property
    def name(self) -> str:
        return "arch_swe_agent_design_document"

    def get_run_command(self, instruction: str) -> str:
        command = super().get_run_command(instruction)
        marker = " --log-root /agent-logs/arch_swe_agent_simple "
        replacement = " --mode design_document" + marker
        if marker not in command:
            raise RuntimeError("Could not inject design-document workflow mode")
        return command.replace(marker, replacement, 1)
