"""
Tests for AgentDS pipeline orchestration.

Agents and the LLM connection are stubbed, so these tests exercise only the
LangGraph routing, human-in-the-loop pause/resume and job bookkeeping.

Author: Malav Patel
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from agentds.agents.base import AgentAction, AgentResult, AgentStatus, BaseAgent
from agentds.core.config import Settings, get_settings
from agentds.core.job_queue import JobQueue, JobStatus
from agentds.workflows import pipeline as pipeline_module
from agentds.workflows.pipeline import AgentDSPipeline, PipelineConfig, PipelinePhase

BUILD_AGENTS = [
    "DataLoaderAgent",
    "DataCleaningAgent",
    "EDACopilotAgent",
    "FeatureEngineerAgent",
    "AutoMLAgent",
]
LEARN_AGENTS = ["DriftMonitorAgent", "OptimizationAgent"]


class FakeAgents:
    """Replaces BaseAgent.run and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.behaviour: dict[str, dict[str, Any]] = {}

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def run(self, agent: BaseAgent, context: Any) -> AgentResult:
        name = agent.__class__.__name__
        self.calls.append((name, context.user_feedback))
        spec = self.behaviour.get(name, {})
        return AgentResult(
            agent_name=name,
            status=spec.get("status", AgentStatus.COMPLETED),
            outputs=spec.get("outputs", {"data_path": f"/tmp/{name}.parquet"}),
            error=spec.get("error"),
            requires_approval=spec.get("requires_approval", True),
        )


@pytest.fixture
def fake_agents(monkeypatch: pytest.MonkeyPatch) -> FakeAgents:
    """Stub agent execution, the LLM gateway, the artifact store and Redis."""
    fake = FakeAgents()

    def fake_run(self: BaseAgent, context: Any) -> AgentResult:
        return fake.run(self, context)

    gateway = MagicMock()
    gateway.validate_connection.return_value = True
    monkeypatch.setattr(BaseAgent, "run", fake_run)
    monkeypatch.setattr(pipeline_module, "LLMGateway", MagicMock(return_value=gateway))
    monkeypatch.setattr(pipeline_module, "ArtifactStore", MagicMock())
    # Keep the job queue in memory so the tests never touch Redis.
    monkeypatch.setattr(Settings, "is_feature_enabled", lambda _self, _name: False)
    return fake


@pytest.fixture
def job_queue() -> JobQueue:
    return JobQueue(get_settings())


def make_pipeline(
    job_queue: JobQueue,
    phases: list[PipelinePhase] | None = None,
    human_in_loop: bool = False,
    checkpoint_enabled: bool = True,
) -> AgentDSPipeline:
    config = PipelineConfig(
        phases=phases or [PipelinePhase.BUILD],
        human_in_loop=human_in_loop,
        checkpoint_enabled=checkpoint_enabled,
    )
    return AgentDSPipeline(config=config, job_queue=job_queue)


def start(pipeline: AgentDSPipeline, data: Path, tmp_path: Path) -> dict[str, Any]:
    return pipeline.run(
        data_source=str(data),
        task_description="Predict the target column",
        output_destination=str(tmp_path / "out"),
    )


class TestAutomaticMode:
    """human_in_loop=False runs straight through."""

    def test_runs_each_agent_once_in_order(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        result = start(make_pipeline(job_queue), sample_csv_path, tmp_path)

        assert fake_agents.names == BUILD_AGENTS
        assert result["status"] == "completed"
        assert result["state"]["completed"] is True
        assert not result["state"].get("error")
        assert set(result["state"]["final_outputs"]) == set(BUILD_AGENTS)

    def test_job_is_stored_under_returned_id(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        result = start(make_pipeline(job_queue), sample_csv_path, tmp_path)

        job = job_queue.get_job(result["job_id"])
        assert job is not None
        assert job.status == JobStatus.COMPLETED
        assert job.progress_percent == 100.0

    def test_failed_agent_stops_pipeline(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        fake_agents.behaviour["DataCleaningAgent"] = {
            "status": AgentStatus.FAILED,
            "error": "bad column",
        }

        result = start(make_pipeline(job_queue), sample_csv_path, tmp_path)

        assert fake_agents.names == ["DataLoaderAgent", "DataCleaningAgent"]
        assert result["status"] == "failed"
        assert "DataCleaningAgent" in result["state"]["error"]
        assert "bad column" in result["state"]["error"]
        assert job_queue.get_job(result["job_id"]).status == JobStatus.FAILED

    def test_runs_without_checkpointer(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        pipeline = make_pipeline(job_queue, checkpoint_enabled=False)

        result = start(pipeline, sample_csv_path, tmp_path)

        assert fake_agents.names == BUILD_AGENTS
        assert result["status"] == "completed"


class TestHumanInTheLoop:
    """human_in_loop=True pauses for review after each agent."""

    def test_pauses_after_first_agent(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        result = start(make_pipeline(job_queue, human_in_loop=True), sample_csv_path, tmp_path)

        assert fake_agents.names == ["DataLoaderAgent"]
        assert result["status"] == "awaiting_approval"
        assert result["current_agent"] == "DataLoaderAgent"
        job = job_queue.get_job(result["job_id"])
        assert job.status == JobStatus.PAUSED
        assert job.current_agent == "DataLoaderAgent"

    def test_pauses_even_when_checkpointing_disabled(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        pipeline = make_pipeline(job_queue, human_in_loop=True, checkpoint_enabled=False)

        result = start(pipeline, sample_csv_path, tmp_path)
        resumed = pipeline.resume(result["job_id"], AgentAction.APPROVE_AND_CONTINUE.value)

        assert resumed["status"] == "awaiting_approval"
        assert resumed["current_agent"] == "DataCleaningAgent"

    def test_approving_every_agent_completes(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        pipeline = make_pipeline(job_queue, human_in_loop=True)
        result = start(pipeline, sample_csv_path, tmp_path)

        reviewed = []
        while result["status"] == "awaiting_approval":
            reviewed.append(result["current_agent"])
            result = pipeline.resume(result["job_id"], AgentAction.APPROVE_AND_CONTINUE.value)

        assert reviewed == BUILD_AGENTS
        assert fake_agents.names == BUILD_AGENTS
        assert result["status"] == "completed"
        assert job_queue.get_job(result["job_id"]).status == JobStatus.COMPLETED

    def test_rerun_with_feedback_reruns_same_agent(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        pipeline = make_pipeline(job_queue, human_in_loop=True)
        result = start(pipeline, sample_csv_path, tmp_path)

        result = pipeline.resume(
            result["job_id"],
            AgentAction.RERUN_WITH_FEEDBACK.value,
            user_feedback="impute with the median",
        )

        assert fake_agents.calls == [
            ("DataLoaderAgent", None),
            ("DataLoaderAgent", "impute with the median"),
        ]
        assert result["status"] == "awaiting_approval"
        assert result["current_agent"] == "DataLoaderAgent"

    def test_skip_on_last_agent_finalizes(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        pipeline = make_pipeline(job_queue, phases=[PipelinePhase.LEARN], human_in_loop=True)
        result = start(pipeline, sample_csv_path, tmp_path)
        result = pipeline.resume(result["job_id"], AgentAction.APPROVE_AND_CONTINUE.value)
        assert result["current_agent"] == "OptimizationAgent"

        result = pipeline.resume(result["job_id"], AgentAction.SKIP.value)

        assert fake_agents.names == LEARN_AGENTS
        assert result["status"] == "completed"
        assert not result["state"].get("error")

    def test_stop_cancels_job(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        pipeline = make_pipeline(job_queue, human_in_loop=True)
        result = start(pipeline, sample_csv_path, tmp_path)

        result = pipeline.resume(result["job_id"], AgentAction.STOP_PIPELINE.value)

        assert fake_agents.names == ["DataLoaderAgent"]
        assert result["status"] == "cancelled"
        assert job_queue.get_job(result["job_id"]).status == JobStatus.CANCELLED

    def test_failed_agent_pauses_for_review(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        fake_agents.behaviour["DataLoaderAgent"] = {
            "status": AgentStatus.FAILED,
            "error": "file is empty",
            "requires_approval": False,
        }

        result = start(make_pipeline(job_queue, human_in_loop=True), sample_csv_path, tmp_path)

        assert result["status"] == "awaiting_approval"
        assert result["current_agent"] == "DataLoaderAgent"
        agent_result = result["state"]["agent_results"]["DataLoaderAgent"]
        assert agent_result["status"] == AgentStatus.FAILED.value

    def test_resume_rejects_job_that_is_not_waiting(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        pipeline = make_pipeline(job_queue, phases=[PipelinePhase.LEARN], human_in_loop=True)
        result = start(pipeline, sample_csv_path, tmp_path)
        while result["status"] == "awaiting_approval":
            result = pipeline.resume(result["job_id"], AgentAction.APPROVE_AND_CONTINUE.value)

        with pytest.raises(ValueError, match="not awaiting review"):
            pipeline.resume(result["job_id"], AgentAction.APPROVE_AND_CONTINUE.value)
        with pytest.raises(ValueError, match="not awaiting review"):
            pipeline.resume("no-such-job", AgentAction.APPROVE_AND_CONTINUE.value)

    @pytest.mark.parametrize("action", ["approve", AgentAction.DOWNLOAD_OUTPUT.value])
    def test_resume_rejects_unsupported_actions(
        self,
        fake_agents: FakeAgents,
        job_queue: JobQueue,
        sample_csv_path: Path,
        tmp_path: Path,
        action: str,
    ) -> None:
        pipeline = make_pipeline(job_queue, human_in_loop=True)
        result = start(pipeline, sample_csv_path, tmp_path)

        with pytest.raises(ValueError, match="action"):
            pipeline.resume(result["job_id"], action)

    def test_resume_rejected_in_automatic_mode(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        pipeline = make_pipeline(job_queue)
        result = start(pipeline, sample_csv_path, tmp_path)

        with pytest.raises(ValueError, match="human-in-the-loop"):
            pipeline.resume(result["job_id"], AgentAction.APPROVE_AND_CONTINUE.value)


class TestCLI:
    """`agentds run` drives the review loop and reports the real outcome."""

    def run_cli(self, data: Path, tmp_path: Path, *args: str, input: str = "") -> Any:
        from click.testing import CliRunner

        from agentds.cli import cli

        return CliRunner().invoke(
            cli,
            ["run", str(data), "-t", "churn", "-o", str(tmp_path / "out"), *args],
            input=input,
        )

    def test_review_prompts_then_completes(
        self, fake_agents: FakeAgents, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        result = self.run_cli(
            sample_csv_path, tmp_path, "-p", "learn",
            input="feedback\nuse median\napprove\napprove\n",
        )

        assert result.exit_code == 0, result.output
        assert fake_agents.calls == [
            ("DriftMonitorAgent", None),
            ("DriftMonitorAgent", "use median"),
            ("OptimizationAgent", None),
        ]
        assert "[OK] Pipeline completed successfully!" in result.output

    def test_failed_run_exits_non_zero(
        self, fake_agents: FakeAgents, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        fake_agents.behaviour["DriftMonitorAgent"] = {
            "status": AgentStatus.FAILED,
            "error": "no reference data",
        }

        result = self.run_cli(sample_csv_path, tmp_path, "-p", "learn", "--no-hitl")

        assert result.exit_code == 1
        assert "[OK]" not in result.output
        assert "no reference data" in result.output

    def test_missing_review_input_exits_non_zero(
        self, fake_agents: FakeAgents, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        result = self.run_cli(sample_csv_path, tmp_path, "-p", "learn", input="")

        assert result.exit_code == 1
        assert "--no-hitl" in result.output


class TestStateSerialization:
    """Agent outputs must survive checkpointing and JSON responses."""

    def test_non_serializable_outputs_are_made_safe(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        fake_agents.behaviour["DataLoaderAgent"] = {
            "outputs": {
                "dataframe": pl.DataFrame({"a": [1, 2]}),
                "data_path": tmp_path / "loaded.parquet",
                "status": AgentStatus.COMPLETED,
            }
        }

        result = start(make_pipeline(job_queue, human_in_loop=True), sample_csv_path, tmp_path)

        outputs = result["state"]["agent_results"]["DataLoaderAgent"]["outputs"]
        assert outputs["data_path"] == str(tmp_path / "loaded.parquet")
        assert outputs["status"] == "completed"
        assert isinstance(outputs["dataframe"], str)
        assert "DataFrame" in outputs["dataframe"]
