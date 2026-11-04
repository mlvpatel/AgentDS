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
from agentds.core.job_queue import JobQueue, JobStatus, RedisQueue
from agentds.workflows import pipeline as pipeline_module
from agentds.workflows.pipeline import (
    AgentDSPipeline,
    PipelineConfig,
    PipelinePhase,
    create_pipeline_job,
)
from agentds.workflows.tasks import resume_pipeline_job, run_pipeline_job

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


class TestResumeInAnotherProcess:
    """A paused run can be resumed by a different process, e.g. a background worker."""

    def test_paused_state_is_saved_on_job_and_cleared_when_done(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        pipeline = make_pipeline(job_queue, phases=[PipelinePhase.LEARN], human_in_loop=True)
        result = start(pipeline, sample_csv_path, tmp_path)

        saved = job_queue.get_job(result["job_id"]).pipeline_state
        assert saved is not None
        assert saved["current_agent"] == "DriftMonitorAgent"

        while result["status"] == "awaiting_approval":
            result = pipeline.resume(result["job_id"], AgentAction.APPROVE_AND_CONTINUE.value)
        assert job_queue.get_job(result["job_id"]).pipeline_state is None

    def test_fresh_pipeline_instance_resumes_from_saved_state(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        first = make_pipeline(job_queue, human_in_loop=True)
        result = start(first, sample_csv_path, tmp_path)

        # A new instance has an empty in-memory checkpointer, like a worker
        second = make_pipeline(job_queue, human_in_loop=True)
        resumed = second.resume(result["job_id"], AgentAction.APPROVE_AND_CONTINUE.value)

        assert fake_agents.names == ["DataLoaderAgent", "DataCleaningAgent"]
        assert resumed["status"] == "awaiting_approval"
        assert resumed["current_agent"] == "DataCleaningAgent"
        # Earlier agents' results came across with the saved state
        assert set(resumed["state"]["agent_results"]) == {"DataLoaderAgent", "DataCleaningAgent"}


class TestBackgroundTasks:
    """Task functions the RQ worker runs, and handing tasks to it."""

    def test_run_and_resume_tasks_load_everything_from_the_job(
        self, fake_agents: FakeAgents, job_queue: JobQueue, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        config = PipelineConfig(phases=[PipelinePhase.LEARN], human_in_loop=True)
        job = create_pipeline_job(
            job_queue,
            config,
            data_source=str(sample_csv_path),
            task_description="churn",
            output_destination=str(tmp_path / "out"),
        )

        paused = run_pipeline_job(job.id, job_queue=job_queue)
        assert paused == {
            "job_id": job.id,
            "status": "awaiting_approval",
            "current_agent": "DriftMonitorAgent",
            "progress_percent": 50.0,
            "error": None,
        }

        resumed = resume_pipeline_job(job.id, "approve_and_continue", job_queue=job_queue)
        assert resumed["current_agent"] == "OptimizationAgent"
        done = resume_pipeline_job(job.id, "approve_and_continue", job_queue=job_queue)
        assert done["status"] == "completed"
        assert fake_agents.names == LEARN_AGENTS

    def test_run_task_rejects_unknown_job(self, job_queue: JobQueue) -> None:
        with pytest.raises(ValueError, match="No job"):
            run_pipeline_job("missing", job_queue=job_queue)

    def test_submit_without_redis_returns_false(self, job_queue: JobQueue) -> None:
        assert job_queue.submit(run_pipeline_job, "J1", job_timeout=60) is False

    def test_submit_hands_task_to_redis_backend(self, job_queue: JobQueue) -> None:
        backend = MagicMock(spec=RedisQueue)
        backend.submit.return_value = "rq-job-1"
        job_queue._backend = backend

        assert job_queue.submit(run_pipeline_job, "J1", job_timeout=60) is True
        backend.submit.assert_called_once_with(run_pipeline_job, "J1", job_timeout=60)

        backend.submit.return_value = None  # no worker listening
        assert job_queue.submit(run_pipeline_job, "J1", job_timeout=60) is False

    def test_redis_queue_enqueues_only_when_a_worker_listens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import rq

        redis_queue = RedisQueue("redis://localhost:6399/0")  # no connection is made here
        enqueue = MagicMock(return_value=MagicMock(id="rq-job-1"))
        monkeypatch.setattr(redis_queue._queue, "enqueue", enqueue)

        monkeypatch.setattr(rq.Worker, "count", MagicMock(return_value=0))
        assert redis_queue.submit(run_pipeline_job, "J1", job_timeout=60) is None
        enqueue.assert_not_called()

        monkeypatch.setattr(rq.Worker, "count", MagicMock(return_value=1))
        assert redis_queue.submit(run_pipeline_job, "J1", job_timeout=60) == "rq-job-1"
        enqueue.assert_called_once_with(run_pipeline_job, "J1", job_timeout=60)


class TestPipelineAPI:
    """The REST API hands pipeline runs to a worker, or runs them off the event loop."""

    HEADERS = {"X-API-Key": "test-key"}

    @pytest.fixture
    def client(self, fake_agents: FakeAgents, job_queue: JobQueue, monkeypatch: pytest.MonkeyPatch):
        from litestar.testing import TestClient

        from agentds.web.api import webhooks

        monkeypatch.setattr(webhooks, "_job_queue", job_queue)
        monkeypatch.setattr(webhooks, "_settings", get_settings())
        with TestClient(app=webhooks.create_api()) as client:
            yield client

    def start_body(self, data: Path, tmp_path: Path) -> dict[str, Any]:
        return {
            "data_source": str(data),
            "task_description": "churn",
            "output_destination": str(tmp_path / "out"),
            "phases": ["learn"],
            "human_in_loop": True,
        }

    def test_without_worker_runs_in_thread_and_resumes_from_saved_state(
        self, client, fake_agents: FakeAgents, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        start_response = client.post(
            "/api/pipeline/start", json=self.start_body(sample_csv_path, tmp_path), headers=self.HEADERS
        )
        assert start_response.status_code == 201, start_response.text
        started = start_response.json()
        assert started["status"] == "awaiting_approval"
        job_id = started["job_id"]

        status = client.get(f"/api/pipeline/status/{job_id}", headers=self.HEADERS).json()
        assert status["status"] == "paused"
        assert status["current_agent"] == "DriftMonitorAgent"

        approve = {"action": "approve_and_continue"}
        first = client.post(f"/api/pipeline/action/{job_id}", json=approve, headers=self.HEADERS)
        assert first.status_code == 201, first.text
        assert first.json()["current_agent"] == "OptimizationAgent"
        second = client.post(f"/api/pipeline/action/{job_id}", json=approve, headers=self.HEADERS)
        assert second.json()["pipeline_status"] == "completed"

        again = client.post(f"/api/pipeline/action/{job_id}", json=approve, headers=self.HEADERS)
        assert again.status_code == 409
        assert fake_agents.names == LEARN_AGENTS

    def test_with_worker_queues_runs_and_review_actions(
        self,
        client,
        fake_agents: FakeAgents,
        job_queue: JobQueue,
        sample_csv_path: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        submitted: list[tuple[Any, tuple, dict]] = []

        def submit(func, *args, **kwargs) -> bool:
            submitted.append((func, args, kwargs))
            return True

        monkeypatch.setattr(job_queue, "submit", submit)

        start_response = client.post(
            "/api/pipeline/start", json=self.start_body(sample_csv_path, tmp_path), headers=self.HEADERS
        )
        assert start_response.status_code == 202, start_response.text
        job_id = start_response.json()["job_id"]
        assert start_response.json()["status"] == "queued"
        func, args, kwargs = submitted.pop()
        assert func is run_pipeline_job and args == (job_id,)
        assert kwargs == {"job_timeout": get_settings().pipeline_job_timeout}
        assert fake_agents.calls == []  # nothing ran in the API process
        assert job_queue.get_job(job_id).status == JobStatus.QUEUED

        # Play the worker's part
        func(*args, job_queue=job_queue)
        assert job_queue.get_job(job_id).status == JobStatus.PAUSED

        action = client.post(
            f"/api/pipeline/action/{job_id}",
            json={"action": "rerun_with_feedback", "feedback": "drop id"},
            headers=self.HEADERS,
        )
        assert action.status_code == 202, action.text
        func, args, _ = submitted.pop()
        assert func is resume_pipeline_job
        assert args == (job_id, "rerun_with_feedback", "drop id")

        # The review is claimed until the worker picks it up
        assert job_queue.get_job(job_id).status == JobStatus.RUNNING
        duplicate = client.post(
            f"/api/pipeline/action/{job_id}",
            json={"action": "approve_and_continue"},
            headers=self.HEADERS,
        )
        assert duplicate.status_code == 409

        func(*args, job_queue=job_queue)
        assert fake_agents.calls[-1] == ("DriftMonitorAgent", "drop id")
        assert job_queue.get_job(job_id).status == JobStatus.PAUSED

    def test_action_rejects_non_review_actions(
        self, client, sample_csv_path: Path, tmp_path: Path
    ) -> None:
        response = client.post(
            "/api/pipeline/action/anything",
            json={"action": "download_output"},
            headers=self.HEADERS,
        )
        assert response.status_code == 400


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
