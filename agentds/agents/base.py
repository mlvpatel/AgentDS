"""
AgentDS - Base Agent.

Provides the abstract base class for all agents with common functionality
including Pydantic-AI integration for structured LLM outputs.

Author: Malav Patel
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from agentds.core.artifact_store import ArtifactStore, ArtifactType
from agentds.core.config import Settings, get_settings
from agentds.core.llm_gateway import LLMGateway, LLMResponse
from agentds.core.logger import LogContext, get_logger, get_logfire_span

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentStatus(str, Enum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentAction(str, Enum):
    """User actions for human-in-the-loop."""

    APPROVE_AND_CONTINUE = "approve_and_continue"
    RERUN = "rerun"
    RERUN_WITH_FEEDBACK = "rerun_with_feedback"
    SKIP = "skip"
    STOP_PIPELINE = "stop_pipeline"
    DOWNLOAD_OUTPUT = "download_output"
    ROLLBACK = "rollback"


class AgentResult(BaseModel):
    """Result from agent execution."""

    agent_name: str = Field(..., description="Name of the agent")
    status: AgentStatus = Field(..., description="Execution status")

    # Timing
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_seconds: float = Field(default=0.0)

    # Outputs
    outputs: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)

    # Metadata
    llm_calls: int = Field(default=0)
    llm_cost: float = Field(default=0.0)
    tokens_used: int = Field(default=0)

    # Error handling
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Human-in-the-loop
    requires_approval: bool = Field(default=True)
    approval_message: Optional[str] = None
    user_feedback: Optional[str] = None

    def mark_completed(self) -> None:
        """Mark result as completed."""
        self.status = AgentStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def mark_failed(self, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Mark result as failed."""
        self.status = AgentStatus.FAILED
        self.error = error
        self.error_details = details
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()


@dataclass
class AgentContext:
    """Context passed to agent during execution."""

    job_id: str
    settings: Settings
    llm_gateway: LLMGateway
    artifact_store: ArtifactStore
    previous_results: Dict[str, AgentResult] = field(default_factory=dict)
    user_feedback: Optional[str] = None
    task_description: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Provides common functionality for:
    - LLM interaction with Pydantic-AI structured outputs
    - Artifact management
    - Human-in-the-loop controls
    - Progress tracking with Logfire
    - Error handling
    """

    # Class attributes to be overridden by subclasses
    name: str = "BaseAgent"
    description: str = "Base agent class"
    phase: str = "unknown"
    complexity: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL
    input_types: List[str] = []
    output_types: List[str] = []

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm_gateway: Optional[LLMGateway] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> None:
        """
        Initialize agent.

        Args:
            settings: Application settings
            llm_gateway: LLM gateway instance
            artifact_store: Artifact store instance
        """
        self.settings = settings or get_settings()
        self.llm_gateway = llm_gateway or LLMGateway(self.settings)
        self.artifact_store = artifact_store or ArtifactStore(self.settings)
        self._llm_calls = 0
        self._tokens_used = 0

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.

        Override in subclasses for agent-specific prompts.

        Returns:
            System prompt string
        """
        return f"""You are {self.name}, a specialized AI agent in the AgentDS pipeline.

Your role: {self.description}

Guidelines:
- Be precise and deterministic in your outputs
- Follow best practices for data science and ML engineering
- Provide clear explanations for your decisions
- Format outputs consistently for downstream processing
- Report any issues or anomalies found

Current task phase: {self.phase}
"""

    def call_llm(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Call the LLM with agent context.

        Args:
            user_message: User message content
            system_prompt: Optional custom system prompt
            context_messages: Optional context messages
            **kwargs: Additional LLM parameters

        Returns:
            LLM response
        """
        messages: List[Dict[str, str]] = []

        # Add system prompt
        system = system_prompt or self.get_system_prompt()
        messages.append({"role": "system", "content": system})

        # Add context messages if provided
        if context_messages:
            messages.extend(context_messages)

        # Add user message
        messages.append({"role": "user", "content": user_message})

        # Call LLM with agent name for config lookup
        with get_logfire_span("agent_llm_call", agent=self.name):
            response = self.llm_gateway.complete(
                messages=messages,
                agent_name=self.name,
                **kwargs,
            )

        # Track usage
        self._llm_calls += 1
        self._tokens_used += response.usage.get("total_tokens", 0)

        return response

    def call_llm_structured(
        self,
        user_message: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        context_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> T:
        """
        Call the LLM and parse response into a Pydantic model.

        Uses Pydantic-AI style structured outputs for reliable parsing.

        Args:
            user_message: User message content
            response_model: Pydantic model class for response
            system_prompt: Optional custom system prompt
            context_messages: Optional context messages
            **kwargs: Additional LLM parameters

        Returns:
            Parsed response as Pydantic model
        """
        import json
        import re

        # Enhance prompt to request JSON output
        schema = response_model.model_json_schema()
        enhanced_prompt = f"""{user_message}

Respond ONLY with a valid JSON object matching this schema:
{json.dumps(schema, indent=2)}

Do not include any other text, markdown formatting, or code blocks. Only output the JSON object."""

        response = self.call_llm(
            user_message=enhanced_prompt,
            system_prompt=system_prompt,
            context_messages=context_messages,
            **kwargs,
        )

        # Parse response
        content = response.content.strip()
        
        # Remove potential markdown code blocks
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        
        # Extract JSON from response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        try:
            data = json.loads(content)
            return response_model.model_validate(data)
        except Exception as e:
            logger.error(
                "Failed to parse structured response",
                agent=self.name,
                error=str(e),
                content=content[:500],
            )
            raise ValueError(f"Failed to parse LLM response: {e}")

    async def acall_llm(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async version of call_llm."""
        messages: List[Dict[str, str]] = []
        system = system_prompt or self.get_system_prompt()
        messages.append({"role": "system", "content": system})

        if context_messages:
            messages.extend(context_messages)

        messages.append({"role": "user", "content": user_message})

        with get_logfire_span("agent_llm_call_async", agent=self.name):
            response = await self.llm_gateway.acomplete(
                messages=messages,
                agent_name=self.name,
                **kwargs,
            )

        self._llm_calls += 1
        self._tokens_used += response.usage.get("total_tokens", 0)

        return response

    async def acall_llm_structured(
        self,
        user_message: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        context_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> T:
        """Async version of call_llm_structured."""
        import json
        import re

        schema = response_model.model_json_schema()
        enhanced_prompt = f"""{user_message}

Respond ONLY with a valid JSON object matching this schema:
{json.dumps(schema, indent=2)}

Do not include any other text, markdown formatting, or code blocks. Only output the JSON object."""

        response = await self.acall_llm(
            user_message=enhanced_prompt,
            system_prompt=system_prompt,
            context_messages=context_messages,
            **kwargs,
        )

        content = response.content.strip()
        
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        try:
            data = json.loads(content)
            return response_model.model_validate(data)
        except Exception as e:
            logger.error(
                "Failed to parse structured response",
                agent=self.name,
                error=str(e),
            )
            raise ValueError(f"Failed to parse LLM response: {e}")

    def save_artifact(
        self,
        job_id: str,
        name: str,
        data: Any,
        artifact_type: ArtifactType = ArtifactType.OTHER,
        **kwargs: Any,
    ) -> str:
        """
        Save an artifact.

        Args:
            job_id: Job identifier
            name: Artifact name
            data: Artifact data
            artifact_type: Type of artifact
            **kwargs: Additional metadata

        Returns:
            Artifact ID
        """
        artifact = self.artifact_store.save(
            job_id=job_id,
            agent=self.name,
            name=name,
            data=data,
            artifact_type=artifact_type,
            **kwargs,
        )
        return artifact.id

    @abstractmethod
    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute the agent's task.

        Args:
            context: Execution context with inputs and configuration

        Returns:
            AgentResult with outputs and status
        """
        pass

    async def aexecute(self, context: AgentContext) -> AgentResult:
        """
        Async version of execute.

        Default implementation calls sync execute.
        Override in subclasses for true async execution.
        """
        return self.execute(context)

    def run(
        self,
        context: AgentContext,
    ) -> AgentResult:
        """
        Run the agent with logging and error handling.

        Args:
            context: Execution context

        Returns:
            AgentResult
        """
        result = AgentResult(
            agent_name=self.name,
            status=AgentStatus.RUNNING,
        )

        with LogContext(job_id=context.job_id, agent=self.name):
            with get_logfire_span("agent_run", agent=self.name, phase=self.phase):
                logger.info(
                    "Agent starting",
                    agent=self.name,
                    phase=self.phase,
                )

                try:
                    # Execute agent logic
                    result = self.execute(context)

                    # Add LLM usage stats
                    result.llm_calls = self._llm_calls
                    result.tokens_used = self._tokens_used
                    result.llm_cost = self.llm_gateway.get_total_cost()

                    # Mark completed if not already set
                    if result.status == AgentStatus.RUNNING:
                        result.mark_completed()

                    logger.info(
                        "Agent completed",
                        agent=self.name,
                        status=result.status,
                        duration=result.duration_seconds,
                        llm_calls=result.llm_calls,
                    )

                except Exception as e:
                    logger.error(
                        "Agent failed",
                        agent=self.name,
                        error=str(e),
                        exc_info=True,
                    )
                    result.mark_failed(str(e))

                # Reset counters for next run
                self._llm_calls = 0
                self._tokens_used = 0

        return result

    def validate_inputs(self, context: AgentContext) -> List[str]:
        """
        Validate inputs before execution.

        Args:
            context: Execution context

        Returns:
            List of validation errors (empty if valid)
        """
        return []

    def get_approval_message(self, result: AgentResult) -> str:
        """
        Generate message for human-in-the-loop approval.

        Args:
            result: Current agent result

        Returns:
            Approval message
        """
        return f"""
Agent: {self.name}
Status: {result.status}
Duration: {result.duration_seconds:.2f}s
LLM Calls: {result.llm_calls}

Outputs:
{self._format_outputs(result.outputs)}

Please review and choose an action:
- APPROVE_AND_CONTINUE: Accept output and continue pipeline
- RERUN: Run this agent again
- RERUN_WITH_FEEDBACK: Provide feedback and rerun
- SKIP: Skip this agent
- STOP_PIPELINE: Stop the entire pipeline
- DOWNLOAD_OUTPUT: Download current outputs
- ROLLBACK: Rollback to previous checkpoint
"""

    def _format_outputs(self, outputs: Dict[str, Any]) -> str:
        """Format outputs for display."""
        lines = []
        for key, value in outputs.items():
            if isinstance(value, (dict, list)):
                lines.append(f"  {key}: [complex data]")
            elif isinstance(value, Path):
                lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines) or "  (no outputs)"
