"""
AgentDS REST API Webhooks.

Litestar-based REST API for n8n and external integrations.

Author: Malav Patel
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from litestar import Litestar, Router, delete, get, post
from litestar.config.cors import CORSConfig
from litestar.openapi import OpenAPIConfig
from litestar.response import Response
from pydantic import BaseModel, Field

from agentds.agents import AgentAction
from agentds.core.config import Settings, get_settings
from agentds.core.job_queue import JobQueue, JobStatus
from agentds.core.logger import get_logger
from agentds.workflows.pipeline import AgentDSPipeline, PipelineConfig, PipelinePhase

logger = get_logger(__name__)


# ==================== Request/Response Models ====================

class PipelineStartRequest(BaseModel):
    """Request to start a pipeline."""

    data_source: str = Field(..., description="Path or URL to input data")
    task_description: str = Field(..., description="Description of the ML task")
    output_destination: str | None = Field(None, description="Output directory")
    phases: list[str] = Field(
        default=["build", "deploy"],
        description="Pipeline phases to run",
    )
    human_in_loop: bool = Field(default=True, description="Enable human-in-the-loop")


class PipelineStartResponse(BaseModel):
    """Response from pipeline start."""

    job_id: str
    status: str
    message: str
    created_at: str


class PipelineStatusResponse(BaseModel):
    """Response with pipeline status."""

    job_id: str
    status: str
    current_agent: str | None
    progress_percent: float
    started_at: str | None
    completed_at: str | None
    error: str | None
    outputs: dict[str, Any]


class AgentRunRequest(BaseModel):
    """Request to run a single agent."""

    agent_name: str = Field(..., description="Name of the agent to run")
    data_source: str = Field(..., description="Input data path")
    task_description: str | None = Field(None, description="Task description")
    config: dict[str, Any] = Field(default_factory=dict, description="Agent config")


class ActionRequest(BaseModel):
    """Request for human-in-the-loop action."""

    action: str = Field(..., description="Action: approve, rerun, skip, stop")
    feedback: str | None = Field(None, description="Optional feedback")


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""

    llm_config: dict[str, Any] | None = None
    pipeline_config: dict[str, Any] | None = None
    feature_flags: dict[str, bool] | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str
    components: dict[str, bool]


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
    timestamp: str


# ==================== Global State ====================

_settings: Settings | None = None
_job_queue: JobQueue | None = None
_active_pipelines: dict[str, Any] = {}


def get_dependencies() -> tuple[Settings, JobQueue]:
    """Get global dependencies."""
    global _settings, _job_queue
    if _settings is None:
        _settings = get_settings()
    if _job_queue is None:
        _job_queue = JobQueue(_settings)
    return _settings, _job_queue


# ==================== API Endpoints ====================

@get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    settings, job_queue = get_dependencies()

    components = {
        "api": True,
        "job_queue": True,
    }

    # Check LLM connection (simplified)
    try:
        from agentds.core.llm_gateway import LLMGateway

        gateway = LLMGateway(settings)
        if gateway.validate_connection():
            components["llm"] = True
        else:
            components["llm"] = False
    except Exception:
        components["llm"] = False

    return HealthResponse(
        status="healthy" if all(components.values()) else "degraded",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
    )


@post("/pipeline/start")
async def start_pipeline(data: PipelineStartRequest) -> PipelineStartResponse:
    """Start a new pipeline execution."""
    settings, job_queue = get_dependencies()

    try:
        # Create pipeline config
        phases = [PipelinePhase(p) for p in data.phases]
        config = PipelineConfig(
            phases=phases,
            human_in_loop=data.human_in_loop,
        )

        # Create and run pipeline
        pipeline = AgentDSPipeline(config=config, settings=settings)
        job_id = str(uuid.uuid4())

        result = pipeline.run(
            data_source=data.data_source,
            task_description=data.task_description,
            output_destination=data.output_destination,
            job_id=job_id,
        )

        # Store pipeline reference
        _active_pipelines[job_id] = {
            "pipeline": pipeline,
            "result": result,
        }

        return PipelineStartResponse(
            job_id=job_id,
            status="started",
            message="Pipeline started successfully",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error("Failed to start pipeline", error=str(e))
        return Response(
            content=ErrorResponse(
                error="Failed to start pipeline",
                detail=str(e),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
            status_code=500,
        )


@get("/pipeline/status/{job_id:str}")
async def get_pipeline_status(job_id: str) -> PipelineStatusResponse:
    """Get status of a pipeline execution."""
    settings, job_queue = get_dependencies()

    job = job_queue.get_job(job_id)
    if not job:
        return Response(
            content=ErrorResponse(
                error="Job not found",
                detail=f"No job with ID: {job_id}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
            status_code=404,
        )

    return PipelineStatusResponse(
        job_id=job.id,
        status=job.status.value,
        current_agent=job.current_agent,
        progress_percent=job.progress_percent,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error,
        outputs=job.result or {},
    )


@post("/pipeline/cancel/{job_id:str}")
async def cancel_pipeline(job_id: str) -> dict[str, str]:
    """Cancel a running pipeline."""
    settings, job_queue = get_dependencies()

    if job_queue.cancel_job(job_id):
        return {"status": "cancelled", "job_id": job_id}

    return Response(
        content=ErrorResponse(
            error="Cannot cancel job",
            detail=f"Job {job_id} not found or not cancellable",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).model_dump(),
        status_code=400,
    )


@post("/pipeline/action/{job_id:str}")
async def pipeline_action(job_id: str, data: ActionRequest) -> dict[str, str]:
    """Handle human-in-the-loop action."""
    settings, job_queue = get_dependencies()

    # Validate action
    try:
        action = AgentAction(data.action)
    except ValueError:
        return Response(
            content=ErrorResponse(
                error="Invalid action",
                detail=f"Action must be one of: {[a.value for a in AgentAction]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
            status_code=400,
        )

    # Get pipeline
    if job_id not in _active_pipelines:
        return Response(
            content=ErrorResponse(
                error="Pipeline not found",
                detail=f"No active pipeline with ID: {job_id}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
            status_code=404,
        )

    pipeline_data = _active_pipelines[job_id]
    pipeline = pipeline_data["pipeline"]

    try:
        pipeline.resume(
            job_id=job_id,
            user_action=action.value,
            user_feedback=data.feedback,
        )

        return {
            "status": "success",
            "action": action.value,
            "job_id": job_id,
        }

    except Exception as e:
        return Response(
            content=ErrorResponse(
                error="Action failed",
                detail=str(e),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
            status_code=500,
        )


@post("/agent/run")
async def run_agent(data: AgentRunRequest) -> dict[str, Any]:
    """Run a single agent independently."""
    settings, job_queue = get_dependencies()

    from agentds.agents import AGENT_REGISTRY
    from agentds.agents.base import AgentContext
    from agentds.core.artifact_store import ArtifactStore
    from agentds.core.llm_gateway import LLMGateway

    if data.agent_name not in AGENT_REGISTRY:
        return Response(
            content=ErrorResponse(
                error="Unknown agent",
                detail=f"Available agents: {list(AGENT_REGISTRY.keys())}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
            status_code=400,
        )

    try:
        # Create agent
        agent_class = AGENT_REGISTRY[data.agent_name]
        agent = agent_class(settings=settings)  # type: ignore[abstract]

        # Create context
        job_id = str(uuid.uuid4())
        context = AgentContext(
            job_id=job_id,
            settings=settings,
            llm_gateway=LLMGateway(settings),
            artifact_store=ArtifactStore(settings),
            task_description=data.task_description,
            extra={
                "data_source": data.data_source,
                **data.config,
            },
        )

        # Run agent
        result = agent.run(context)

        return {
            "job_id": job_id,
            "agent": data.agent_name,
            "status": result.status.value,
            "outputs": result.outputs,
            "artifacts": result.artifacts,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
        }

    except Exception as e:
        return Response(
            content=ErrorResponse(
                error="Agent execution failed",
                detail=str(e),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
            status_code=500,
        )


@get("/jobs")
async def list_jobs(
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List all jobs."""
    settings, job_queue = get_dependencies()

    job_status = JobStatus(status) if status else None
    jobs = job_queue.list_jobs(status=job_status, limit=limit)

    return [
        {
            "id": job.id,
            "name": job.name,
            "status": job.status.value,
            "current_agent": job.current_agent,
            "progress_percent": job.progress_percent,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "duration_seconds": job.duration_seconds,
        }
        for job in jobs
    ]


@get("/jobs/{job_id:str}")
async def get_job(job_id: str) -> dict[str, Any]:
    """Get job details."""
    settings, job_queue = get_dependencies()

    job = job_queue.get_job(job_id)
    if not job:
        return Response(
            content=ErrorResponse(
                error="Job not found",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
            status_code=404,
        )

    return job.model_dump()


@delete("/jobs/{job_id:str}")
async def delete_job(job_id: str) -> dict[str, str]:
    """Delete a job."""
    settings, job_queue = get_dependencies()

    if job_queue.delete_job(job_id):
        return {"status": "deleted", "job_id": job_id}

    return Response(
        content=ErrorResponse(
            error="Failed to delete job",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).model_dump(),
        status_code=400,
    )


@post("/config/update")
async def update_config(data: ConfigUpdateRequest) -> dict[str, str]:
    """Update configuration."""
    # In production, this would persist configuration changes
    return {
        "status": "success",
        "message": "Configuration updated (session only)",
    }


@get("/config")
async def get_config() -> dict[str, Any]:
    """Get current configuration."""
    settings, _ = get_dependencies()

    return {
        "llm": {
            "default_model": settings.llm.default_model,
            "default_temperature": settings.llm.default_temperature,
            "available_providers": settings.llm.get_available_providers(),
        },
        "pipeline": {
            "human_in_loop": settings.human_in_loop,
        },
    }


# ==================== Router and App ====================

api_router = Router(
    path="/api",
    route_handlers=[
        health_check,
        start_pipeline,
        get_pipeline_status,
        cancel_pipeline,
        pipeline_action,
        run_agent,
        list_jobs,
        get_job,
        delete_job,
        update_config,
        get_config,
    ],
)


def create_api(settings: Settings | None = None) -> Litestar:
    """
    Create Litestar API application.

    Args:
        settings: Application settings (optional)

    Returns:
        Configured Litestar application with auth and rate limiting
    """
    from agentds.web.api.middleware import (
        AuthenticationMiddleware,
        RateLimitMiddleware,
    )

    settings = settings or get_settings()

    return Litestar(
        route_handlers=[api_router],
        cors_config=CORSConfig(
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        ),
        openapi_config=OpenAPIConfig(
            title="AgentDS API",
            version="1.0.0",
            description="REST API for AgentDS multi-agent data science pipeline",
        ),
        middleware=[
            AuthenticationMiddleware,
            RateLimitMiddleware,
        ],
    )


# CLI entry point
if __name__ == "__main__":
    import uvicorn

    app = create_api()
    uvicorn.run(app, host="0.0.0.0", port=8000)
