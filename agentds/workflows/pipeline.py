"""
AgentDS Pipeline Orchestration.

LangGraph-based workflow orchestration for multi-agent pipelines.

Author: Malav Patel
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agentds.agents import (
    AGENT_REGISTRY,
    AgentAction,
    AgentResult,
    BaseAgent,
)
from agentds.agents.base import AgentContext, AgentStatus
from agentds.core.artifact_store import ArtifactStore
from agentds.core.config import Settings, get_settings
from agentds.core.job_queue import Job, JobQueue
from agentds.core.llm_gateway import LLMGateway
from agentds.core.logger import LogContext, get_logger

logger = get_logger(__name__)


class PipelinePhase(str, Enum):
    """Pipeline execution phases."""

    BUILD = "build"
    DEPLOY = "deploy"
    LEARN = "learn"


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    name: str = Field(default="AgentDS Pipeline")
    phases: list[PipelinePhase] = Field(
        default=[PipelinePhase.BUILD, PipelinePhase.DEPLOY]
    )
    human_in_loop: bool = Field(default=True)
    auto_approve_low_risk: bool = Field(default=False)
    checkpoint_enabled: bool = Field(default=True)
    max_retries: int = Field(default=3)


class PipelineState(TypedDict, total=False):
    """State passed through the pipeline graph."""

    job_id: str
    task_description: str
    data_source: str
    output_destination: str

    # Execution state
    current_phase: str | None
    current_agent: str | None  # Next agent to run; None once every agent has run
    agent_results: dict[str, dict[str, Any]]

    # Human-in-the-loop
    awaiting_approval: bool
    user_action: str | None
    user_feedback: str | None

    # Error handling
    error: str | None
    retry_count: int

    # Completion
    completed: bool
    stopped: bool
    final_outputs: dict[str, Any]


# Node the graph pauses before when human review is required.
REVIEW_NODE = "human_review"

# Actions that can resume a paused pipeline.
REVIEW_ACTIONS = frozenset(
    {
        AgentAction.APPROVE_AND_CONTINUE,
        AgentAction.RERUN,
        AgentAction.RERUN_WITH_FEEDBACK,
        AgentAction.SKIP,
        AgentAction.STOP_PIPELINE,
        AgentAction.ROLLBACK,
    }
)


def _state_safe(value: Any) -> Any:
    """
    Convert agent output into plain data that checkpoints and JSON can hold.

    Agents may return objects such as DataFrames or Paths. LangGraph's
    checkpointer cannot serialize arbitrary objects, so those are replaced with
    a short placeholder; downstream agents use file paths, not in-memory frames.
    """
    # Enum and numpy checks come first: str-enums and numpy floats subclass
    # the builtin types below.
    if isinstance(value, Enum):
        return _state_safe(value.value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return _state_safe(value.tolist())
    if isinstance(value, BaseModel):
        return _state_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _state_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_state_safe(item) for item in value]
    return f"<{type(value).__name__} not stored in pipeline state>"


@dataclass
class PreflightCheckResult:
    """Result of pre-flight validation."""

    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class AgentDSPipeline:
    """
    Main pipeline orchestrator using LangGraph.

    Orchestrates the flow through:
    1. Pre-flight checks
    2. Build phase (agents 1-5)
    3. Deploy phase (agents 6-8)
    4. Learn phase (agents 9-10)

    With human-in-the-loop controls at every step.
    """

    # Agent execution order by phase
    PHASE_AGENTS = {
        PipelinePhase.BUILD: [
            "DataLoaderAgent",
            "DataCleaningAgent",
            "EDACopilotAgent",
            "FeatureEngineerAgent",
            "AutoMLAgent",
        ],
        PipelinePhase.DEPLOY: [
            "APIWrapperAgent",
            "DevOpsAgent",
            "CloudDeployAgent",
        ],
        PipelinePhase.LEARN: [
            "DriftMonitorAgent",
            "OptimizationAgent",
        ],
    }

    def __init__(
        self,
        config: PipelineConfig | None = None,
        settings: Settings | None = None,
        job_queue: JobQueue | None = None,
    ) -> None:
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
            settings: Application settings
            job_queue: Job queue to record jobs in. Pass the caller's queue so
                job status can be looked up by the returned job_id.
        """
        self.config = config or PipelineConfig()
        self.settings = settings or get_settings()
        self.llm_gateway = LLMGateway(self.settings)
        self.artifact_store = ArtifactStore(self.settings)
        self.job_queue = job_queue or JobQueue(self.settings)

        # Initialize agents
        self._agents: dict[str, BaseAgent] = {}
        self._init_agents()

        # Build graph. Pausing for review needs a checkpointer to resume from,
        # so human-in-the-loop always gets one.
        self._graph = self._build_graph()
        needs_checkpointer = self.config.checkpoint_enabled or self.config.human_in_loop
        self._checkpointer = MemorySaver() if needs_checkpointer else None
        self._compiled = self._graph.compile(
            checkpointer=self._checkpointer,
            interrupt_before=[REVIEW_NODE] if self.config.human_in_loop else None,
        )

    def agent_sequence(self) -> list[str]:
        """Agents in execution order for the configured phases."""
        sequence: list[str] = []
        for phase in self.config.phases:
            sequence.extend(self.PHASE_AGENTS.get(phase, []))
        return sequence

    def _init_agents(self) -> None:
        """Initialize all agent instances."""
        for phase in self.config.phases:
            for agent_name in self.PHASE_AGENTS.get(phase, []):
                if agent_name in AGENT_REGISTRY:
                    agent_class = AGENT_REGISTRY[agent_name]
                    self._agents[agent_name] = agent_class(  # type: ignore[abstract]
                        settings=self.settings,
                        llm_gateway=self.llm_gateway,
                        artifact_store=self.artifact_store,
                    )

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine."""
        graph = StateGraph(PipelineState)

        # Add nodes
        graph.add_node("preflight", self._preflight_node)
        graph.add_node("run_agent", self._run_agent_node)
        graph.add_node(REVIEW_NODE, self._human_review_node)
        graph.add_node("handle_action", self._handle_action_node)
        graph.add_node("finalize", self._finalize_node)
        graph.add_node("error", self._error_node)

        # Set entry point
        graph.set_entry_point("preflight")

        # Add edges
        graph.add_conditional_edges(
            "preflight",
            self._route_after_preflight,
            {
                "run_agent": "run_agent",
                "finalize": "finalize",
                "error": "error",
            },
        )

        graph.add_conditional_edges(
            "run_agent",
            self._route_after_agent,
            {
                REVIEW_NODE: REVIEW_NODE,
                "run_agent": "run_agent",
                "finalize": "finalize",
                "error": "error",
            },
        )

        graph.add_edge(REVIEW_NODE, "handle_action")

        graph.add_conditional_edges(
            "handle_action",
            self._route_after_action,
            {
                "run_agent": "run_agent",
                "finalize": "finalize",
                "error": "error",
            },
        )

        graph.add_edge("finalize", END)
        graph.add_edge("error", END)

        return graph

    def _preflight_node(self, state: PipelineState) -> PipelineState:
        """Run pre-flight checks."""
        logger.info("Running pre-flight checks", job_id=state.get("job_id"))

        result = self._run_preflight_checks(state)

        if not result.passed:
            state["error"] = f"Pre-flight checks failed: {', '.join(result.errors)}"
            return state

        # Set initial execution state
        agents_to_run = self.agent_sequence()
        first_agent = agents_to_run[0] if agents_to_run else None
        state["current_agent"] = first_agent
        state["current_phase"] = self._get_agent_phase(first_agent) if first_agent else None

        state["agent_results"] = {}
        state["retry_count"] = 0

        return state

    def _run_preflight_checks(self, state: PipelineState) -> PreflightCheckResult:
        """Execute pre-flight validation checks."""
        result = PreflightCheckResult(passed=True)

        # Check LLM connection
        try:
            if self.llm_gateway.validate_connection():
                result.checks["llm_connection"] = True
            else:
                result.checks["llm_connection"] = False
                result.errors.append("LLM connection failed")
                result.passed = False
        except Exception as e:
            result.checks["llm_connection"] = False
            result.errors.append(f"LLM connection error: {e}")
            result.passed = False

        # Check data source
        data_source = state.get("data_source")
        if data_source:
            if Path(data_source).exists() or data_source.startswith(("http", "s3://", "gs://")):
                result.checks["data_source"] = True
            else:
                result.checks["data_source"] = False
                result.errors.append(f"Data source not found: {data_source}")
                result.passed = False
        else:
            result.checks["data_source"] = False
            result.errors.append("No data source provided")
            result.passed = False

        # Check output destination
        output_dest = state.get("output_destination")
        if output_dest:
            try:
                Path(output_dest).mkdir(parents=True, exist_ok=True)
                result.checks["output_destination"] = True
            except Exception as e:
                result.checks["output_destination"] = False
                result.warnings.append(f"Output destination issue: {e}")
        else:
            result.checks["output_destination"] = True
            result.warnings.append("Using default output destination")

        logger.info(
            "Pre-flight checks completed",
            passed=result.passed,
            checks=result.checks,
        )

        return result

    def _run_agent_node(self, state: PipelineState) -> PipelineState:
        """Run the current agent."""
        agent_name = state.get("current_agent")
        if not agent_name or agent_name not in self._agents:
            state["error"] = f"Unknown agent: {agent_name}"
            return state

        agent = self._agents[agent_name]
        job_id = state.get("job_id", "unknown")

        logger.info("Running agent", agent=agent_name, job_id=job_id)

        # Build agent context
        context = AgentContext(
            job_id=job_id,
            settings=self.settings,
            llm_gateway=self.llm_gateway,
            artifact_store=self.artifact_store,
            previous_results={
                name: AgentResult(**data)
                for name, data in state.get("agent_results", {}).items()
            },
            user_feedback=state.get("user_feedback"),
            task_description=state.get("task_description"),
            extra={
                "data_source": state.get("data_source"),
                "output_destination": state.get("output_destination"),
            },
        )

        # Run agent
        try:
            result = agent.run(context)
        except Exception as e:
            logger.error("Agent execution failed", agent=agent_name, error=str(e))
            state["error"] = f"Agent {agent_name} failed: {e}"
            return state

        # Store a checkpoint-safe copy of the result (copy the dict rather than
        # mutating the one held in the previous checkpoint).
        agent_results = dict(state.get("agent_results") or {})
        agent_results[agent_name] = _state_safe(result.model_dump())
        state["agent_results"] = agent_results
        state["user_feedback"] = None  # Feedback applies to one run only

        failed = result.status == AgentStatus.FAILED
        if self.config.human_in_loop and (failed or result.requires_approval):
            # Pause for review. current_agent stays on this agent so the
            # reviewer can rerun it, skip it or approve and move on.
            state["awaiting_approval"] = True
            return state

        state["awaiting_approval"] = False
        if failed:
            state["error"] = f"Agent {agent_name} failed: {result.error or 'unknown error'}"
            return state

        # Advance here, inside a node: LangGraph discards state changes made
        # in routing functions, so the router must only read current_agent.
        self._advance_to_next_agent(state, agent_name)
        return state

    def _advance_to_next_agent(self, state: PipelineState, from_agent: str | None) -> None:
        """Point current_agent at the agent after from_agent (None when done)."""
        next_agent = self._get_next_agent(from_agent)
        state["current_agent"] = next_agent
        if next_agent:
            state["current_phase"] = self._get_agent_phase(next_agent)

    def _human_review_node(self, state: PipelineState) -> PipelineState:
        """Placeholder for the review step.

        The graph is compiled with interrupt_before=[REVIEW_NODE], so execution
        pauses before this node. resume() records the reviewer's action as this
        node's output, and execution continues at handle_action.
        """
        logger.info(
            "Awaiting human review",
            agent=state.get("current_agent"),
            job_id=state.get("job_id"),
        )
        return state

    def _handle_action_node(self, state: PipelineState) -> PipelineState:
        """Handle user action from human review."""
        action = state.get("user_action")
        current_agent = state.get("current_agent")

        logger.info(
            "Handling user action",
            action=action,
            agent=current_agent,
        )

        state["awaiting_approval"] = False

        if action in (AgentAction.APPROVE_AND_CONTINUE.value, AgentAction.SKIP.value):
            self._advance_to_next_agent(state, current_agent)

        elif action in (AgentAction.RERUN.value, AgentAction.RERUN_WITH_FEEDBACK.value):
            # current_agent is unchanged; user_feedback was set by resume()
            state["retry_count"] = state.get("retry_count", 0) + 1

        elif action == AgentAction.STOP_PIPELINE.value:
            state["completed"] = True
            state["stopped"] = True

        elif action == AgentAction.ROLLBACK.value:
            prev_agent = self._get_previous_agent(current_agent)
            if prev_agent:
                state["current_agent"] = prev_agent
                state["current_phase"] = self._get_agent_phase(prev_agent)

        else:
            state["error"] = f"Unsupported review action: {action!r}"

        # Clear action
        state["user_action"] = None

        return state

    def _finalize_node(self, state: PipelineState) -> PipelineState:
        """Finalize pipeline execution."""
        logger.info("Finalizing pipeline", job_id=state.get("job_id"))

        # Collect final outputs
        final_outputs = {}
        for agent_name, result_data in state.get("agent_results", {}).items():
            if result_data.get("outputs"):
                final_outputs[agent_name] = result_data["outputs"]

        state["final_outputs"] = final_outputs
        state["completed"] = True

        return state

    def _error_node(self, state: PipelineState) -> PipelineState:
        """Handle pipeline errors."""
        error = state.get("error", "Unknown error")
        logger.error("Pipeline error", error=error, job_id=state.get("job_id"))
        state["completed"] = True
        return state

    # Routers only read state. LangGraph does not persist changes made here.

    def _route_after_preflight(
        self, state: PipelineState
    ) -> Literal["run_agent", "finalize", "error"]:
        """Route after pre-flight checks."""
        if state.get("error"):
            return "error"
        if not state.get("current_agent"):
            return "finalize"
        return "run_agent"

    def _route_after_agent(
        self, state: PipelineState
    ) -> Literal["human_review", "run_agent", "finalize", "error"]:
        """Route after agent execution."""
        if state.get("error"):
            return "error"
        if state.get("awaiting_approval"):
            return REVIEW_NODE
        if state.get("current_agent"):
            return "run_agent"
        return "finalize"

    def _route_after_action(
        self, state: PipelineState
    ) -> Literal["run_agent", "finalize", "error"]:
        """Route after handling user action."""
        if state.get("error"):
            return "error"
        if state.get("completed") or not state.get("current_agent"):
            return "finalize"
        return "run_agent"

    def _get_agent_phase(self, agent_name: str) -> str:
        """Get phase for an agent."""
        for phase, agents in self.PHASE_AGENTS.items():
            if agent_name in agents:
                return phase.value
        return "unknown"

    def _get_next_agent(self, current_agent: str | None) -> str | None:
        """Get next agent in execution order."""
        sequence = self.agent_sequence()
        if current_agent not in sequence:
            return None
        idx = sequence.index(current_agent)
        return sequence[idx + 1] if idx + 1 < len(sequence) else None

    def _get_previous_agent(self, current_agent: str | None) -> str | None:
        """Get previous agent in execution order."""
        sequence = self.agent_sequence()
        if current_agent not in sequence:
            return None
        idx = sequence.index(current_agent)
        return sequence[idx - 1] if idx > 0 else None

    def run(
        self,
        data_source: str,
        task_description: str,
        output_destination: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the complete pipeline.

        Args:
            data_source: Path or URL to input data
            task_description: Description of the ML task
            output_destination: Where to save outputs
            job_id: Optional job identifier

        Returns:
            Pipeline results: ``job_id``, ``status`` (``completed``, ``failed``,
            ``cancelled`` or ``awaiting_approval``), ``current_agent``,
            ``progress_percent``, the graph ``state`` and the ``job`` record.
            When ``status`` is ``awaiting_approval``, call :meth:`resume`.
        """
        job_id = job_id or str(uuid.uuid4())
        output_destination = output_destination or str(self.settings.output_dir / job_id)

        # Create the job under the same id the caller gets back
        job = self.job_queue.create_job(
            name=f"Pipeline: {task_description[:50]}",
            task="full_pipeline",
            kwargs={
                "data_source": data_source,
                "task_description": task_description,
                "output_destination": output_destination,
            },
            job_id=job_id,
        )
        job.mark_running()
        self.job_queue.update_job(job)

        with LogContext(job_id=job_id):
            logger.info(
                "Starting pipeline",
                job_id=job_id,
                data_source=data_source,
                phases=[p.value for p in self.config.phases],
            )

            # Initial state
            initial_state: PipelineState = {
                "job_id": job_id,
                "task_description": task_description,
                "data_source": data_source,
                "output_destination": output_destination,
                "agent_results": {},
                "awaiting_approval": False,
                "retry_count": 0,
                "completed": False,
                "stopped": False,
            }

            # Runs until the pipeline finishes, fails, or (human-in-the-loop)
            # pauses before REVIEW_NODE.
            final_state = self._invoke(initial_state, job_id, job)
            return self._sync_job(job_id, job, final_state)

    def get_compiled_graph(self):
        """Get the compiled graph (shares this pipeline's checkpointer)."""
        return self._compiled

    def resume(
        self,
        job_id: str,
        user_action: str,
        user_feedback: str | None = None,
    ) -> dict[str, Any]:
        """
        Resume a pipeline that is paused for human review.

        Args:
            job_id: Job identifier returned by :meth:`run`
            user_action: An :class:`AgentAction` value: approve_and_continue,
                rerun, rerun_with_feedback, skip, stop_pipeline or rollback
            user_feedback: Optional feedback passed to the agent on rerun

        Returns:
            Pipeline results, in the same shape as :meth:`run`

        Raises:
            ValueError: If the action is not a review action, the pipeline is
                not in human-in-the-loop mode, or the job is not awaiting review.
        """
        try:
            action = AgentAction(user_action)
        except ValueError:
            action = None
        if action not in REVIEW_ACTIONS:
            allowed = sorted(a.value for a in REVIEW_ACTIONS)
            raise ValueError(f"Unsupported review action {user_action!r}; expected one of {allowed}")

        if not self.config.human_in_loop:
            raise ValueError("Pipeline is not running in human-in-the-loop mode")

        config = self._thread_config(job_id)
        snapshot = self._compiled.get_state(config)
        if REVIEW_NODE not in snapshot.next:
            raise ValueError(f"Job {job_id} is not awaiting review")

        # Record the decision as the review node's output; execution then
        # continues at handle_action.
        self._compiled.update_state(
            config,
            {"user_action": action.value, "user_feedback": user_feedback},
            as_node=REVIEW_NODE,
        )

        job = self.job_queue.get_job(job_id)
        if job:
            job.resume(user_feedback)
            self.job_queue.update_job(job)

        with LogContext(job_id=job_id):
            logger.info("Resuming pipeline", job_id=job_id, action=action.value)
            final_state = self._invoke(None, job_id, job)
            return self._sync_job(job_id, job, final_state)

    @staticmethod
    def _thread_config(job_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": job_id}}

    def _invoke(
        self, graph_input: PipelineState | None, job_id: str, job: Job | None
    ) -> PipelineState:
        """Invoke the graph, marking the job failed if execution raises."""
        try:
            return self._compiled.invoke(graph_input, config=self._thread_config(job_id))
        except Exception as e:
            logger.error("Pipeline execution failed", job_id=job_id, error=str(e))
            if job:
                job.mark_failed(f"Pipeline execution failed: {e}")
                self.job_queue.update_job(job)
            raise

    def _sync_job(
        self, job_id: str, job: Job | None, state: PipelineState
    ) -> dict[str, Any]:
        """Record the outcome of a run or resume on the job and build the result."""
        sequence = self.agent_sequence()
        finished = sum(
            1
            for name, data in (state.get("agent_results") or {}).items()
            if name in sequence and data.get("status") == AgentStatus.COMPLETED.value
        )
        current_agent = state.get("current_agent")

        if state.get("stopped"):
            status = "cancelled"
        elif state.get("completed"):
            status = "failed" if state.get("error") else "completed"
        elif state.get("awaiting_approval"):
            status = "awaiting_approval"
        else:
            status = "running"

        if job:
            job.update_progress(finished, len(sequence), agent=current_agent)
            if status == "cancelled":
                job.mark_cancelled()
            elif status == "failed":
                job.mark_failed(state.get("error") or "Pipeline failed")
            elif status == "completed":
                job.mark_completed(state.get("final_outputs"))
            elif status == "awaiting_approval":
                job.mark_paused()
            self.job_queue.update_job(job)

        if status == "completed":
            progress = 100.0
        else:
            progress = finished / len(sequence) * 100 if sequence else 0.0
        return {
            "job_id": job_id,
            "status": status,
            "current_agent": current_agent,
            "progress_percent": progress,
            "error": state.get("error"),
            "state": state,
            "job": job.model_dump(mode="json") if job else None,
        }
