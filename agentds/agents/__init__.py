"""
AgentDS Agents Module.

This module provides 10 specialized agents covering the complete data science lifecycle:

Phase 1: Build (Execution Squad)
- DataLoaderAgent: Load data from various sources
- DataCleaningAgent: Clean and validate data
- EDACopilotAgent: Generate exploratory analysis
- FeatureEngineerAgent: Create preprocessing pipelines
- AutoMLAgent: Train and select models

Phase 2: Deploy (MLOps Squad)
- APIWrapperAgent: Generate API server code
- DevOpsAgent: Create Docker configuration
- CloudDeployAgent: Deploy to cloud platforms

Phase 3: Learn (Optimization Squad)
- DriftMonitorAgent: Monitor model performance
- OptimizationAgent: Self-improve agent prompts

Author: Malav Patel
"""

from agentds.agents.api_wrapper import APIWrapperAgent
from agentds.agents.automl import AutoMLAgent
from agentds.agents.base import (
    AgentAction,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from agentds.agents.cloud_deploy import CloudDeployAgent
from agentds.agents.data_cleaning import DataCleaningAgent
from agentds.agents.data_loader import DataLoaderAgent
from agentds.agents.devops import DevOpsAgent
from agentds.agents.drift_monitor import DriftMonitorAgent
from agentds.agents.eda_copilot import EDACopilotAgent
from agentds.agents.feature_engineer import FeatureEngineerAgent
from agentds.agents.optimization import OptimizationAgent

__all__ = [
    # Base
    "BaseAgent",
    "AgentResult",
    "AgentStatus",
    "AgentAction",
    # Phase 1: Build
    "DataLoaderAgent",
    "DataCleaningAgent",
    "EDACopilotAgent",
    "FeatureEngineerAgent",
    "AutoMLAgent",
    # Phase 2: Deploy
    "APIWrapperAgent",
    "DevOpsAgent",
    "CloudDeployAgent",
    # Phase 3: Learn
    "DriftMonitorAgent",
    "OptimizationAgent",
]

# Agent registry for dynamic lookup
AGENT_REGISTRY = {
    "DataLoaderAgent": DataLoaderAgent,
    "DataCleaningAgent": DataCleaningAgent,
    "EDACopilotAgent": EDACopilotAgent,
    "FeatureEngineerAgent": FeatureEngineerAgent,
    "AutoMLAgent": AutoMLAgent,
    "APIWrapperAgent": APIWrapperAgent,
    "DevOpsAgent": DevOpsAgent,
    "CloudDeployAgent": CloudDeployAgent,
    "DriftMonitorAgent": DriftMonitorAgent,
    "OptimizationAgent": OptimizationAgent,
}


def get_agent(name: str) -> type[BaseAgent]:
    """Get agent class by name."""
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {name}")
    return AGENT_REGISTRY[name]
