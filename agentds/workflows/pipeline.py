"""
AgentDS Pipeline Orchestration.

LangGraph-based workflow orchestration for multi-agent pipelines.

Author: Malav Patel
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agentds.agents import (
    AGENT_REGISTRY,
    AgentAction,
    AgentResult,
    BaseAgent,
)
from agentds.agents.base import AgentContext
from agentds.core.artifact_store import ArtifactStore
from agentds.core.config import Settings, get_settings
from agentds.core.job_queue import JobQueue
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
    current_phase: str
    current_agent: str
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
    final_outputs: dict[str, Any]


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
    ) -> None:
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
            settings: Application settings
        """
        self.config = config or PipelineConfig()
        self.settings = settings or get_settings()
        self.llm_gateway = LLMGateway(self.settings)
        self.artifact_store = ArtifactStore(self.settings)
        self.job_queue = JobQueue(self.settings)

        # Initialize agents
        self._agents: dict[str, BaseAgent] = {}
        self._init_agents()

        # Build graph
        self._graph = self._build_graph()
        self._checkpointer = MemorySaver() if self.config.checkpoint_enabled else None

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
        graph.add_node("human_review", self._human_review_node)
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
                "error": "error",
            },
        )

        graph.add_conditional_edges(
            "run_agent",
            self._route_after_agent,
            {
                "human_review": "human_review",
                "run_agent": "run_agent",
                "finalize": "finalize",
                "error": "error",
            },
        )

        graph.add_edge("human_review", "handle_action")

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
        agents_to_run = []
        for phase in self.config.phases:
            agents_to_run.extend(self.PHASE_AGENTS.get(phase, []))

        if agents_to_run:
            state["current_agent"] = agents_to_run[0]
            state["current_phase"] = self._get_agent_phase(agents_to_run[0])

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

            # Store result
            state["agent_results"] = state.get("agent_results", {})
            state["agent_results"][agent_name] = result.model_dump()

            # Set approval state if human-in-loop enabled
            if self.config.human_in_loop and result.requires_approval:
                state["awaiting_approval"] = True
            else:
                state["awaiting_approval"] = False

            # Clear user feedback after use
            state["user_feedback"] = None

        except Exception as e:
            logger.error("Agent execution failed", agent=agent_name, error=str(e))
            state["error"] = f"Agent {agent_name} failed: {e}"

        return state

    def _human_review_node(self, state: PipelineState) -> PipelineState:
        """Wait for human review and action."""
        # This node is typically handled externally via the web interface
        # The graph will pause here until user_action is set
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

        if action == AgentAction.APPROVE_AND_CONTINUE.value:
            state["awaiting_approval"] = False
            state["current_agent"] = self._get_next_agent(current_agent)

        elif action == AgentAction.RERUN.value or action == AgentAction.RERUN_WITH_FEEDBACK.value:
            state["awaiting_approval"] = False
            state["retry_count"] = state.get("retry_count", 0) + 1
            # user_feedback should already be set

        elif action == AgentAction.SKIP.value:
            state["awaiting_approval"] = False
            state["current_agent"] = self._get_next_agent(current_agent)

        elif action == AgentAction.STOP_PIPELINE.value:
            state["completed"] = True
            state["error"] = "Pipeline stopped by user"

        elif action == AgentAction.ROLLBACK.value:
            # Rollback to previous checkpoint
            prev_agent = self._get_previous_agent(current_agent)
            if prev_agent:
                state["current_agent"] = prev_agent
            state["awaiting_approval"] = False

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

    def _route_after_preflight(
        self, state: PipelineState
    ) -> Literal["run_agent", "error"]:
        """Route after pre-flight checks."""
        if state.get("error"):
            return "error"
        return "run_agent"

    def _route_after_agent(
        self, state: PipelineState
    ) -> Literal["human_review", "run_agent", "finalize", "error"]:
        """Route after agent execution."""
        if state.get("error"):
            return "error"

        if state.get("awaiting_approval"):
            return "human_review"

        # Check if there are more agents
        next_agent = self._get_next_agent(state.get("current_agent"))
        if next_agent:
            state["current_agent"] = next_agent
            state["current_phase"] = self._get_agent_phase(next_agent)
            return "run_agent"

        return "finalize"

    def _route_after_action(
        self, state: PipelineState
    ) -> Literal["run_agent", "finalize", "error"]:
        """Route after handling user action."""
        if state.get("error"):
            return "error"

        if state.get("completed"):
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
        if not current_agent:
            return None

        all_agents = []
        for phase in self.config.phases:
            all_agents.extend(self.PHASE_AGENTS.get(phase, []))

        try:
            idx = all_agents.index(current_agent)
            if idx + 1 < len(all_agents):
                return all_agents[idx + 1]
        except ValueError:
            pass

        return None

    def _get_previous_agent(self, current_agent: str | None) -> str | None:
        """Get previous agent in execution order."""
        if not current_agent:
            return None

        all_agents = []
        for phase in self.config.phases:
            all_agents.extend(self.PHASE_AGENTS.get(phase, []))

        try:
            idx = all_agents.index(current_agent)
            if idx > 0:
                return all_agents[idx - 1]
        except ValueError:
            pass

        return None

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
            Pipeline results
        """
        job_id = job_id or str(uuid.uuid4())
        output_destination = output_destination or str(self.settings.output_dir / job_id)

        # Create job
        job = self.job_queue.create_job(
            name=f"Pipeline: {task_description[:50]}",
            task="full_pipeline",
            kwargs={
                "data_source": data_source,
                "task_description": task_description,
                "output_destination": output_destination,
            },
        )

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
            }

            # Compile and run graph
            compiled = self._graph.compile(checkpointer=self._checkpointer)

            # For synchronous execution without human-in-loop
            if not self.config.human_in_loop:
                final_state = compiled.invoke(
                    initial_state,
                    config={"configurable": {"thread_id": job_id}},
                )
            else:
                # Return compiled graph for external control
                final_state = initial_state

            # Update job
            if final_state.get("completed"):
                if final_state.get("error"):
                    job.mark_failed(final_state["error"])
                else:
                    job.mark_completed(final_state.get("final_outputs"))
            self.job_queue.update_job(job)

            return {
                "job_id": job_id,
                "state": final_state,
                "job": job.model_dump(),
            }

    def get_compiled_graph(self):
        """Get compiled graph for external control."""
        return self._graph.compile(checkpointer=self._checkpointer)

    def resume(
        self,
        job_id: str,
        user_action: str,
        user_feedback: str | None = None,
    ) -> dict[str, Any]:
        """
        Resume pipeline after human review.

        Args:
            job_id: Job identifier
            user_action: User action (approve, rerun, skip, etc.)
            user_feedback: Optional feedback for rerun

        Returns:
            Updated pipeline state
        """
        compiled = self._graph.compile(checkpointer=self._checkpointer)

        # Get current state
        config = {"configurable": {"thread_id": job_id}}
        state = compiled.get_state(config)

        if not state:
            raise ValueError(f"No state found for job: {job_id}")

        # Update state with user action
        update = {
            "user_action": user_action,
            "user_feedback": user_feedback,
            "awaiting_approval": False,
        }

        # Continue execution
        final_state = compiled.invoke(
            update,
            config=config,
        )

        return {
            "job_id": job_id,
            "state": final_state,
        }
