"""
AgentDS - Autonomous Data Science Pipeline with Multi-Agent Orchestration.

This package provides a complete data science automation system with:
- 10 specialized agents covering Build, Deploy, and Learn phases
- 100+ LLM providers via LiteLLM universal gateway
- Human-in-the-loop controls at every step
- Pydantic-AI for structured LLM outputs
- Logfire for full observability
- Agent Lightning APO for self-optimization

Author: Malav Patel
Email: malav.patel203@gmail.com
License: MIT
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentds")
except PackageNotFoundError:
    __version__ = "1.0.0"

__author__ = "Malav Patel"
__email__ = "malav.patel203@gmail.com"
__license__ = "MIT"

# Core exports
# Agent exports
from agentds.agents.base import AgentResult, AgentStatus, BaseAgent
from agentds.core.config import Settings, get_settings
from agentds.core.llm_gateway import LLMGateway
from agentds.core.logger import configure_logfire, get_logger, setup_logging

# Workflow exports
from agentds.workflows.pipeline import AgentDSPipeline

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Core
    "Settings",
    "get_settings",
    "LLMGateway",
    "get_logger",
    "setup_logging",
    "configure_logfire",
    # Agents
    "BaseAgent",
    "AgentResult",
    "AgentStatus",
    # Workflows
    "AgentDSPipeline",
]
