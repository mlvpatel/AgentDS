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
    @pytest.mark.llm
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
# Data handoff between build agents
# =============================================================================

def _completed(name: str, **outputs: object) -> AgentResult:
    return AgentResult(agent_name=name, status=AgentStatus.COMPLETED, outputs=dict(outputs))


class TestDataHandoff:
    """Agents read their input dataset from the nearest upstream agent that produced one."""

    @pytest.fixture
    def build_context(
        self, agent_context: AgentContext, mock_settings: Settings, tmp_path: Path
    ) -> AgentContext:
        mock_settings.temp_dir = tmp_path / "temp"
        mock_settings.temp_dir.mkdir()
        return agent_context

    def test_eda_passes_through_its_input_data_path(
        self,
        build_context: AgentContext,
        mock_settings: Settings,
        mock_llm_gateway: MagicMock,
        mock_artifact_store: ArtifactStore,
        sample_parquet_path: Path,
    ):
        build_context.previous_results = {
            "DataCleaningAgent": _completed("DataCleaningAgent", data_path=str(sample_parquet_path)),
        }
        agent = EDACopilotAgent(
            settings=mock_settings, llm_gateway=mock_llm_gateway, artifact_store=mock_artifact_store
        )

        result = agent.execute(build_context)

        assert result.status == AgentStatus.COMPLETED, result.error
        assert result.outputs["data_path"] == str(sample_parquet_path)

    def test_feature_engineer_runs_after_eda_without_data_path(
        self,
        build_context: AgentContext,
        mock_settings: Settings,
        mock_llm_gateway: MagicMock,
        mock_artifact_store: ArtifactStore,
        sample_parquet_path: Path,
    ):
        # Results as an EDA agent from before the fix would leave them: no data_path.
        build_context.previous_results = {
            "DataLoaderAgent": _completed("DataLoaderAgent", data_path="/missing/raw.parquet"),
            "DataCleaningAgent": _completed("DataCleaningAgent", data_path=str(sample_parquet_path)),
            "EDACopilotAgent": _completed("EDACopilotAgent", statistics={}),
        }
        agent = FeatureEngineerAgent(
            settings=mock_settings, llm_gateway=mock_llm_gateway, artifact_store=mock_artifact_store
        )

        result = agent.execute(build_context)

        assert result.status == AgentStatus.COMPLETED, result.error

    def test_resolve_data_path_prefers_nearest_upstream(self, agent_context: AgentContext):
        agent_context.previous_results = {
            "DataCleaningAgent": _completed("DataCleaningAgent", data_path="clean.parquet"),
            "EDACopilotAgent": _completed("EDACopilotAgent", data_path="eda.parquet"),
        }

        path = FeatureEngineerAgent.resolve_data_path(
            agent_context, "EDACopilotAgent", "DataCleaningAgent"
        )

        assert path == "eda.parquet"

    def test_resolve_data_path_skips_failed_results(self, agent_context: AgentContext):
        failed = _completed("DataCleaningAgent", data_path="partial.parquet")
        failed.status = AgentStatus.FAILED
        agent_context.previous_results = {
            "DataLoaderAgent": _completed("DataLoaderAgent", data_path="raw.parquet"),
            "DataCleaningAgent": failed,
        }

        path = FeatureEngineerAgent.resolve_data_path(
            agent_context, "DataCleaningAgent", "DataLoaderAgent"
        )

        assert path == "raw.parquet"

    def test_resolve_data_path_names_agents_when_none_found(self, agent_context: AgentContext):
        agent_context.previous_results = {
            "EDACopilotAgent": _completed("EDACopilotAgent", statistics={}),
        }

        with pytest.raises(ValueError, match="EDACopilotAgent, DataLoaderAgent"):
            FeatureEngineerAgent.resolve_data_path(
                agent_context, "EDACopilotAgent", "DataLoaderAgent"
            )


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

    @staticmethod
    def _serve_generated_api(
        mock_settings: Settings,
        tmp_path: Path,
        features: list[str],
        task_type: str,
        prediction: float,
        requests: list[tuple[str, dict]],
        categorical: tuple[str, ...] = (),
        pipeline: object | None = None,
    ) -> tuple[list, list[list[str]], list[dict]]:
        """Generate app.py, import it, and send requests to it with a stub model."""
        import importlib.util
        import sys

        from litestar.testing import TestClient

        agent = APIWrapperAgent(settings=mock_settings)
        code = agent._generate_api_code({}, features, task_type, categorical_features=categorical)

        app_file = tmp_path / "generated_app.py"
        app_file.write_text(code)
        spec = importlib.util.spec_from_file_location("generated_app", app_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["generated_app"] = module

        class StubModel:
            def __init__(self) -> None:
                self.columns: list[list[str]] = []
                self.rows: list[dict] = []

            def predict(self, frame):
                self.columns.append(list(frame.columns))
                self.rows.append(frame.iloc[0].to_dict())
                return [prediction]

        stub = StubModel()
        try:
            spec.loader.exec_module(module)
            module._model = stub
            module._pipeline = pipeline
            with TestClient(app=module.app) as client:
                responses = [client.post(path, json=body) for path, body in requests]
        finally:
            sys.modules.pop("generated_app", None)
        return responses, stub.columns, stub.rows

    @pytest.mark.parametrize(
        ("task_type", "expected"),
        [("classification", 1), ("regression", 1.5)],
    )
    def test_generated_api_serves_predictions(
        self,
        mock_settings: Settings,
        tmp_path: Path,
        task_type: str,
        expected: float,
    ):
        """The generated app.py imports, and its endpoints build input from the request."""
        (single, batch), columns, _rows = self._serve_generated_api(
            mock_settings,
            tmp_path,
            ["tenure", "monthly_charges"],
            task_type,
            expected,
            [
                ("/predict", {"tenure": 3, "monthly_charges": 20.5}),
                ("/predict/batch", {"instances": [{"tenure": 1, "monthly_charges": 9.0}]}),
            ],
        )

        assert single.status_code == 201, single.text
        assert single.json()["prediction"] == expected
        assert batch.status_code == 201, batch.text
        assert batch.json()["predictions"][0]["prediction"] == expected
        assert columns == [["tenure", "monthly_charges"]] * 2

    def test_generated_api_handles_awkward_column_names(
        self, mock_settings: Settings, tmp_path: Path
    ):
        """Any column name yields valid code, and the model sees the original names."""
        features = [
            "Annual Income (k$)",
            "2019_sales",
            "class",
            'say "hi" \\ there',
            "a b",
            "a-b",
            "_id",
            "json",
            "model_type",
            "naïve score",
        ]
        body = {name: float(i) for i, name in enumerate(features)}

        (single, batch), columns, _rows = self._serve_generated_api(
            mock_settings,
            tmp_path,
            features,
            "regression",
            2.0,
            [("/predict", body), ("/predict/batch", {"instances": [body]})],
        )

        assert single.status_code == 201, single.text
        assert batch.status_code == 201, batch.text
        # Every feature reached the model under its original name, in order:
        # nothing collided, was dropped, or was renamed.
        assert columns == [features, features]

    def test_generated_field_names_are_unique_identifiers(self, mock_settings: Settings):
        """Generated field names are valid, distinct, and safe as Pydantic fields."""
        import keyword

        from pydantic import BaseModel

        names = APIWrapperAgent._field_identifiers(
            ["a b", "a-b", "a_b", "", "123", "for", "_x", "dict", "model_x", "x" * 3]
        )

        assert len(set(names)) == len(names)
        for name in names:
            assert name.isidentifier() and not keyword.iskeyword(name), name
            assert not name.startswith("_"), name
            assert not name.startswith("model_"), name
            assert not hasattr(BaseModel, name), name

    def test_generated_api_accepts_categorical_values(
        self, mock_settings: Settings, tmp_path: Path
    ):
        """Categorical columns take string values; numeric columns stay floats."""
        body = {"tenure": 3, "plan type": "gold"}

        (single, batch), _columns, rows = self._serve_generated_api(
            mock_settings,
            tmp_path,
            ["tenure", "plan type"],
            "classification",
            1,
            [("/predict", body), ("/predict/batch", {"instances": [body]})],
            categorical=("plan type",),
        )

        assert single.status_code == 201, single.text
        assert batch.status_code == 201, batch.text
        assert rows == [{"tenure": 3.0, "plan type": "gold"}] * 2

    def test_generated_api_reports_preprocessing_errors(
        self, mock_settings: Settings, tmp_path: Path
    ):
        """A failing pipeline returns 422; the model never sees unprocessed input."""

        class StubPipeline:
            def transform(self, frame):
                if "platinum" in frame["plan"].tolist():
                    raise ValueError("Found unknown categories ['platinum'] in column 1")
                return frame.assign(transformed=True)

        gold = {"tenure": 1, "plan": "gold"}
        platinum = {"tenure": 2, "plan": "platinum"}

        (ok, bad, bad_batch), columns, _rows = self._serve_generated_api(
            mock_settings,
            tmp_path,
            ["tenure", "plan"],
            "classification",
            1,
            [
                ("/predict", gold),
                ("/predict", platinum),
                ("/predict/batch", {"instances": [gold, platinum]}),
            ],
            categorical=("plan",),
            pipeline=StubPipeline(),
        )

        assert ok.status_code == 201, ok.text
        assert bad.status_code == 422, bad.text
        assert "Preprocessing failed for input" in bad.json()["detail"]
        assert "platinum" in bad.json()["detail"]
        assert bad_batch.status_code == 422, bad_batch.text
        assert "Preprocessing failed for instance 1" in bad_batch.json()["detail"]
        # The model only ever received preprocessed rows: the good single
        # request and the good first batch instance.
        assert columns == [["tenure", "plan", "transformed"]] * 2

    def test_execute_types_columns_from_feature_plan(
        self,
        agent_context: AgentContext,
        mock_settings: Settings,
        mock_llm_gateway: MagicMock,
        mock_artifact_store: ArtifactStore,
    ):
        """The agent reads column types from FeatureEngineerAgent's plan."""
        agent_context.previous_results = {
            "AutoMLAgent": AgentResult(
                agent_name="AutoMLAgent",
                status=AgentStatus.COMPLETED,
                outputs={"task_type": "classification"},
            ),
            "FeatureEngineerAgent": AgentResult(
                agent_name="FeatureEngineerAgent",
                status=AgentStatus.COMPLETED,
                outputs={
                    "target_column": "churned",
                    "feature_plan": {
                        # Plans can list the target; it must not become an API input
                        "numeric_features": {"columns": ["tenure", "churned"]},
                        "categorical_features": {"columns": ["plan"]},
                    },
                },
            ),
        }
        agent = APIWrapperAgent(
            settings=mock_settings, llm_gateway=mock_llm_gateway, artifact_store=mock_artifact_store
        )

        result = agent.execute(agent_context)

        assert result.status == AgentStatus.COMPLETED, result.error
        code = mock_artifact_store.load_text(result.outputs["api_code_artifact_id"])
        assert "    tenure: float = Field(..., alias='tenure'" in code
        assert "    plan: str = Field(..., alias='plan'" in code
        assert "alias='churned'" not in code
        schema = result.outputs["api_config"]["input_schema"]
        assert schema["tenure"]["type"] == "float"
        assert schema["plan"]["type"] == "string"
        assert "churned" not in schema


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
