"""
Agent implementations for FeatureBench inference.
"""

from featurebench.infer.agents.base import BaseAgent
from featurebench.infer.agents.arch_swe_agent import ArchSweAgent
from featurebench.infer.agents.arch_swe_agent_simple import ArchSweAgentSimple
from featurebench.infer.agents.codespec2_agent import CodeSpec2Agent
from featurebench.infer.agents.codespec2_implement_replay_agent import CodeSpec2ImplementReplayAgent
from featurebench.infer.agents.codespec_skill_bootstrap_agent import CodeSpecSkillBootstrapAgent
from featurebench.infer.agents.arch_swe_agent_design_document import ArchSweAgentDesignDocument
from featurebench.infer.agents.arch_swe_agent_ablations import (
    ArchSweAgentWithoutExecutableArchitecture,
    ArchSweAgentWithoutGeneratedTests,
    ArchSweAgentWithoutMultiViewEvidence,
)
from featurebench.infer.agents.codex import CodexAgent
from featurebench.infer.agents.claude_code import ClaudeCodeAgent
from featurebench.infer.agents.gemini_cli import GeminiCliAgent
from featurebench.infer.agents.mini_swe_agent import MiniSweAgent
from featurebench.infer.agents.openhands import OpenHandsAgent
from featurebench.infer.agents.rtadev_agent import RTADevMiniSweAgent, RTADevOpenHandsAgent
from featurebench.infer.agents.rtadev_agent_mini_swe_only import RTADevMiniSweOnlyAgent

__all__ = [
    "BaseAgent",
    "ArchSweAgent",
    "ArchSweAgentSimple",
    "CodeSpec2Agent",
    "CodeSpec2ImplementReplayAgent",
    "CodeSpecSkillBootstrapAgent",
    "ArchSweAgentDesignDocument",
    "ArchSweAgentWithoutExecutableArchitecture",
    "ArchSweAgentWithoutGeneratedTests",
    "ArchSweAgentWithoutMultiViewEvidence",
    "CodexAgent",
    "ClaudeCodeAgent",
    "GeminiCliAgent",
    "MiniSweAgent",
    "OpenHandsAgent",
    "RTADevMiniSweAgent",
    "RTADevOpenHandsAgent",
    "RTADevMiniSweOnlyAgent",
]


def get_agent(agent_name: str, **kwargs) -> BaseAgent:
    """
    Get an agent by name.
    
    Args:
        agent_name: Name of the agent (claude_code, openhands)
        **kwargs: Additional arguments for the agent
        
    Returns:
        Agent instance
    """
    agents = {
        "codex": CodexAgent,
        "arch_swe_agent": ArchSweAgent,
        "arch_swe_agent_simple": ArchSweAgentSimple,
        "codespec2": CodeSpec2Agent,
        "codespec2_implement_replay": CodeSpec2ImplementReplayAgent,
        "codespec_skill_bootstrap": CodeSpecSkillBootstrapAgent,
        "arch_swe_agent_design_document": ArchSweAgentDesignDocument,
        "arch_swe_agent_wo_generated_tests": ArchSweAgentWithoutGeneratedTests,
        "arch_swe_agent_wo_multi_view_evidence": ArchSweAgentWithoutMultiViewEvidence,
        "arch_swe_agent_wo_executable_architecture": ArchSweAgentWithoutExecutableArchitecture,
        "claude_code": ClaudeCodeAgent,
        "gemini_cli": GeminiCliAgent,
        "mini_swe_agent": MiniSweAgent,
        "openhands": OpenHandsAgent,
        "rtadev_agent_mini_swe": RTADevMiniSweAgent,
        "rtadev_agent_openhands": RTADevOpenHandsAgent,
        "rtadev_agent_mini_swe_only": RTADevMiniSweOnlyAgent,
    }
    
    agent_class = agents.get(agent_name.lower())
    if agent_class is None:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(agents.keys())}")
    
    return agent_class(**kwargs)
