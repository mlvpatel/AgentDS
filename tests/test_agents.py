"""
AgentDS Agent Tests.

Tests for all 10 specialized agents.

Author: Malav Patel
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentds.agents import (
    AGENT_REGISTRY,
    AgentAction,
    AgentResult,
    AgentStatus,
    APIWrapperAgent,
    AutoMLAgent,
    CloudDeployAgent,
    DataCleaningAgent,
    DataLoaderAgent,
    DevOpsAgent,
    DriftMonitorAgent,
    EDACopilotAgent,
    FeatureEngineerAgent,
    OptimizationAgent,
)
from agentds.agents.base import AgentContext
from agentds.core.artifact_store import ArtifactStore
from agentds.core.config import Settings

# =============================================================================
# Base Agent Tests
# =============================================================================

class TestBaseAgent:
    """Tests for BaseAgent class."""

    def test_agent_registry(self):
        """Test that all agents are registered."""
        expected_agents = [
            "DataLoaderAgent",
            "DataCleaningAgent",
            "EDACopilotAgent",
            "FeatureEngineerAgent",
            "AutoMLAgent",
            "APIWrapperAgent",
            "DevOpsAgent",
            "CloudDeployAgent",
            "DriftMonitorAgent",
            "OptimizationAgent",
        ]
        for agent_name in expected_agents:
            assert agent_name in AGENT_REGISTRY

    def test_agent_result_model(self):
        """Test AgentResult model."""
        result = AgentResult(
            agent_name="TestAgent",
            status=AgentStatus.RUNNING,
        )
        assert result.agent_name == "TestAgent"
        assert result.status == AgentStatus.RUNNING
        assert result.llm_calls == 0

    def test_agent_result_completed(self):
        """Test marking result as completed."""
        result = AgentResult(
            agent_name="TestAgent",
            status=AgentStatus.RUNNING,
        )
        result.mark_completed()
        assert result.status == AgentStatus.COMPLETED
        assert result.completed_at is not None

    def test_agent_result_failed(self):
        """Test marking result as failed."""
        result = AgentResult(
            agent_name="TestAgent",
            status=AgentStatus.RUNNING,
        )
        result.mark_failed("Test error")
        assert result.status == AgentStatus.FAILED
        assert result.error == "Test error"


# =============================================================================
# DataLoaderAgent Tests
# =============================================================================

class TestDataLoaderAgent:
    """Tests for DataLoaderAgent."""

    def test_init(self, mock_settings: Settings):
        """Test agent initialization."""
        agent = DataLoaderAgent(settings=mock_settings)
        assert agent.name == "DataLoaderAgent"
        assert agent.phase == "build"
        assert agent.complexity == "LOW"

    def test_system_prompt(self, mock_settings: Settings):
        """Test system prompt generation."""
        agent = DataLoaderAgent(settings=mock_settings)
        prompt = agent.get_system_prompt()
        assert "DataLoaderAgent" in prompt
        assert "load" in prompt.lower()

    @pytest.mark.parametrize(
        "file_ext,expected_format",
        [
            (".csv", "csv"),
            (".parquet", "parquet"),
            (".json", "json"),
            (".xlsx", "excel"),
        ],
    )
    def test_format_detection(
        self,
        mock_settings: Settings,
        file_ext: str,
        expected_format: str,
    ):
        """Test file format detection."""
        agent = DataLoaderAgent(settings=mock_settings)
        # Format detection is internal but we can test via analysis
        analysis = agent._analyze_source(f"/path/to/data{file_ext}", MagicMock())
        assert analysis["format"] == expected_format


# =============================================================================
# DataCleaningAgent Tests
# =============================================================================

class TestDataCleaningAgent:
    """Tests for DataCleaningAgent."""

    def test_init(self, mock_settings: Settings):
        """Test agent initialization."""
        agent = DataCleaningAgent(settings=mock_settings)
        assert agent.name == "DataCleaningAgent"
        assert agent.phase == "build"


# =============================================================================
# EDACopilotAgent Tests
# =============================================================================

class TestEDACopilotAgent:
    """Tests for EDACopilotAgent."""

    def test_init(self, mock_settings: Settings):
        """Test agent initialization."""
        agent = EDACopilotAgent(settings=mock_settings)
        assert agent.name == "EDACopilotAgent"
        assert agent.phase == "build"


class TestFeatureEngineerAgent:
    """Tests for FeatureEngineerAgent."""

    def test_init(self, mock_settings: Settings):
        """Test agent initialization."""
        agent = FeatureEngineerAgent(settings=mock_settings)
        assert agent.name == "FeatureEngineerAgent"
        assert agent.phase == "build"


# =============================================================================
# AutoMLAgent Tests
# =============================================================================

class TestAutoMLAgent:
    """Tests for AutoMLAgent."""

    def test_init(self, mock_settings: Settings):
        """Test agent initialization."""
        agent = AutoMLAgent(settings=mock_settings)
        assert agent.name == "AutoMLAgent"
        assert agent.phase == "build"
        assert agent.complexity == "HIGH"


# =============================================================================
# APIWrapperAgent Tests
# =============================================================================

class TestAPIWrapperAgent:
    """Tests for APIWrapperAgent."""

    def test_init(self, mock_settings: Settings):
        """Test agent initialization."""
        agent = APIWrapperAgent(settings=mock_settings)
        assert agent.name == "APIWrapperAgent"
        assert agent.phase == "deploy"

    def test_system_prompt(self, mock_settings: Settings):
        """Test system prompt."""
        agent = APIWrapperAgent(settings=mock_settings)
        prompt = agent.get_system_prompt()
        assert "APIWrapperAgent" in prompt


# =============================================================================
# DevOpsAgent Tests
# =============================================================================

class TestDevOpsAgent:
    """Tests for DevOpsAgent."""

    def test_init(self, mock_settings: Settings):
        """Test agent initialization."""
        agent = DevOpsAgent(settings=mock_settings)
        assert agent.name == "DevOpsAgent"
        assert agent.phase == "deploy"

    def test_dockerfile_generation(self, mock_settings: Settings, mock_llm_gateway: MagicMock, agent_context: AgentContext):
        """Test Dockerfile template generation."""
        agent = DevOpsAgent(settings=mock_settings, llm_gateway=mock_llm_gateway)
        dockerfile = agent._generate_dockerfile(agent_context)
        assert "FROM python" in dockerfile
        assert "EXPOSE" in dockerfile
        assert "HEALTHCHECK" in dockerfile

    def test_docker_compose_generation(self, mock_settings: Settings, mock_llm_gateway: MagicMock, agent_context: AgentContext):
        """Test docker-compose generation."""
        agent = DevOpsAgent(settings=mock_settings, llm_gateway=mock_llm_gateway)
        compose = agent._generate_docker_compose(agent_context)
        assert "version:" in compose
        assert "services:" in compose

    def test_dockerignore_generation(self, mock_settings: Settings):
        """Test .dockerignore generation."""
        agent = DevOpsAgent(settings=mock_settings)
        ignore = agent._generate_dockerignore()
        assert "__pycache__" in ignore
        assert ".git" in ignore


# =============================================================================
# CloudDeployAgent Tests
# =============================================================================

class TestCloudDeployAgent:
    """Tests for CloudDeployAgent."""

    def test_init(self, mock_settings: Settings):
        """Test agent initialization."""
        agent = CloudDeployAgent(settings=mock_settings)
        assert agent.name == "CloudDeployAgent"
        assert agent.phase == "deploy"


class TestDriftMonitorAgent:
    """Tests for DriftMonitorAgent."""

    def test_init(self, mock_settings: Settings):
        """Test agent initialization."""
        agent = DriftMonitorAgent(settings=mock_settings)
        assert agent.name == "DriftMonitorAgent"
        assert agent.phase == "learn"


# =============================================================================
# OptimizationAgent Tests
# =============================================================================

class TestOptimizationAgent:
    """Tests for OptimizationAgent."""

    def test_init(self, mock_settings: Settings):
        """Test agent initialization."""
        agent = OptimizationAgent(settings=mock_settings)
        assert agent.name == "OptimizationAgent"
        assert agent.phase == "learn"
        assert agent.complexity == "CRITICAL"

    def test_identify_optimization_targets(
        self,
        mock_settings: Settings,
        mock_llm_gateway: MagicMock,
    ):
        """Test optimization target identification."""
        agent = OptimizationAgent(
            settings=mock_settings,
            llm_gateway=mock_llm_gateway,
        )

        # Test with performance metrics
        targets = agent._identify_optimization_targets(
            drift_result=None,
            user_feedback="The feature engineering could be improved",
            performance_metrics={
                "DataLoaderAgent": {"success_rate": 0.95},
                "FeatureEngineerAgent": {"success_rate": 0.6},  # Below threshold
            },
        )

        assert "FeatureEngineerAgent" in targets


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgentIntegration:
    """Integration tests for agent pipeline."""

    def test_agent_chain(
        self,
        mock_settings: Settings,
        mock_llm_gateway: MagicMock,
        mock_artifact_store: ArtifactStore,
        sample_csv_path: Path,
    ):
        """Test running agents in sequence."""
        # This is a simplified integration test
        # In production, use the full pipeline

        _context = AgentContext(
            job_id="integration-test-001",
            settings=mock_settings,
            llm_gateway=mock_llm_gateway,
            artifact_store=mock_artifact_store,
            task_description="Test classification task",
            extra={"data_source": str(sample_csv_path)},
        )

        # Test that DataLoaderAgent can be instantiated and has required methods
        loader = DataLoaderAgent(
            settings=mock_settings,
            llm_gateway=mock_llm_gateway,
            artifact_store=mock_artifact_store,
        )

        assert hasattr(loader, "execute")
        assert hasattr(loader, "run")
        assert hasattr(loader, "get_system_prompt")


# =============================================================================
# Agent Action Tests
# =============================================================================

class TestAgentActions:
    """Tests for human-in-the-loop actions."""

    def test_action_enum(self):
        """Test AgentAction enum values."""
        assert AgentAction.APPROVE_AND_CONTINUE.value == "approve_and_continue"
        assert AgentAction.RERUN.value == "rerun"
        assert AgentAction.SKIP.value == "skip"
        assert AgentAction.STOP_PIPELINE.value == "stop_pipeline"

    def test_all_actions_defined(self):
        """Test all expected actions are defined."""
        expected_actions = [
            "approve_and_continue",
            "rerun",
            "rerun_with_feedback",
            "skip",
            "stop_pipeline",
            "download_output",
            "rollback",
        ]
        actual_actions = [a.value for a in AgentAction]
        for action in expected_actions:
            assert action in actual_actions


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
