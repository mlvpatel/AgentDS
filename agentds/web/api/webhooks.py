"""
AgentDS REST API Webhooks.

Litestar-based REST API for n8n and external integrations.

Author: Malav Patel
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from anyio import to_thread
from litestar import Litestar, Router, delete, get, post
from litestar.config.cors import CORSConfig
from litestar.middleware import DefineMiddleware
from litestar.openapi import OpenAPIConfig
from litestar.response import Response
from pydantic import BaseModel, Field

from agentds.agents import AGENT_REGISTRY, AgentAction
from agentds.core.config import Settings, get_settings
from agentds.core.job_queue import JobQueue, JobStatus
from agentds.core.logger import get_logger
from agentds.workflows.pipeline import (
    REVIEW_ACTIONS,
    PipelineConfig,
    PipelinePhase,
    create_pipeline_job,
)
from agentds.workflows.tasks import resume_pipeline_job, run_pipeline_job

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
    status: str  # completed, failed, cancelled or awaiting_approval
    message: str
    created_at: str
    current_agent: str | None = None
    error: str | None = None


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


class LivenessResponse(BaseModel):
    """Liveness probe response."""

    status: str
    timestamp: str


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    status: str
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


def get_dependencies() -> tuple[Settings, JobQueue]:
    """Get global dependencies."""
    global _settings, _job_queue
    if _settings is None:
        _settings = get_settings()
    if _job_queue is None:
        _job_queue = JobQueue(_settings)
    return _settings, _job_queue


async def get_dependencies_async() -> tuple[Settings, JobQueue]:
    """
    Get global dependencies without blocking the event loop.

    The first call builds the JobQueue, which connects to Redis and can wait
    for the full socket timeout when Redis is unreachable; that happens in a
    worker thread so other requests (liveness probes included) keep flowing.
    """
    if _settings is None or _job_queue is None:
        return await to_thread.run_sync(get_dependencies)
    return _settings, _job_queue


# ==================== API Endpoints ====================

# Health checks of dependencies run in the background and are cached, so a
# slow or hanging dependency never delays a probe. The LLM check is a real
# (billed) completion that can take minutes while a provider is slow; the
# job queue check is a Redis PING.
LLM_HEALTH_TTL_SECONDS = 300.0
JOB_QUEUE_HEALTH_TTL_SECONDS = 10.0
# How long the readiness probe waits for the very first job queue check;
# below the 1 s default probe timeout of Kubernetes
READINESS_FIRST_CHECK_WAIT_SECONDS = 0.5


class BackgroundCheck:
    """
    A dependency check run in a worker thread, at most one at a time.

    ``status()`` returns the last result immediately (None before the first
    check finishes) and starts a new check when the result is older than
    ``ttl_seconds``.
    """

    def __init__(self, name: str, check: Callable[[], bool], ttl_seconds: float) -> None:
        self.name = name
        self.check = check
        self.ttl_seconds = ttl_seconds
        self.ok: bool | None = None
        self.checked_at: float | None = None
        self._task: asyncio.Task[None] | None = None

    def status(self) -> bool | None:
        stale = self.checked_at is None or time.monotonic() - self.checked_at > self.ttl_seconds
        if stale and (self._task is None or self._task.done()):
            # Keep a reference so the task isn't garbage-collected mid-check
            self._task = asyncio.create_task(self._refresh())
        return self.ok

    async def first_status(self, wait_seconds: float) -> bool | None:
        """
        Like ``status()``, but before the first check has finished, wait up to
        ``wait_seconds`` for it, so a fast dependency is reported on the first
        call instead of None.
        """
        ok = self.status()
        if self.checked_at is None and self._task is not None:
            await asyncio.wait({self._task}, timeout=wait_seconds)  # doesn't cancel the check
            ok = self.ok
        return ok

    async def _refresh(self) -> None:
        try:
            ok = await to_thread.run_sync(self.check)
        except Exception as e:
            logger.warning("Health check failed", check=self.name, error=str(e))
            ok = False
        self.ok = bool(ok)
        self.checked_at = time.monotonic()


_health_checks: dict[str, BackgroundCheck] = {}


def _check_llm(settings: Settings) -> bool:
    """Blocking check that the default LLM answers."""
    try:
        from agentds.core.llm_gateway import LLMGateway

        return bool(LLMGateway(settings).validate_connection())
    except Exception as e:
        logger.warning("LLM health check failed", error=str(e))
        return False


def _check_job_queue(job_queue: JobQueue) -> bool:
    """Blocking check that the job queue backend answers (a Redis PING)."""
    return job_queue.ping()


def _get_health_checks(settings: Settings, job_queue: JobQueue) -> dict[str, BackgroundCheck]:
    if not _health_checks:
        _health_checks["llm"] = BackgroundCheck(
            "llm", lambda: _check_llm(settings), LLM_HEALTH_TTL_SECONDS
        )
        _health_checks["job_queue"] = BackgroundCheck(
            "job_queue", lambda: _check_job_queue(job_queue), JOB_QUEUE_HEALTH_TTL_SECONDS
        )
    return _health_checks


@get("/health/live")
async def liveness_check() -> LivenessResponse:
    """Liveness probe: the process is up and its event loop is answering.

    Checks no dependencies, so a Redis or LLM outage never gets a healthy
    process restarted (restarting would not fix either).
    """
    return LivenessResponse(status="alive", timestamp=datetime.now(timezone.utc).isoformat())


@get("/health/ready")
async def readiness_check() -> Response[ReadinessResponse]:
    """Readiness probe: this instance can accept work.

    Returns 200 when the job queue backend answered its last background check
    (a Redis PING, refreshed every JOB_QUEUE_HEALTH_TTL_SECONDS), and 503 while
    it is failing or before its first check has passed. The LLM is left out on
    purpose: a provider outage affects every instance alike, and taking them
    all out of service would turn it into a full API outage.
    """
    settings, job_queue = await get_dependencies_async()
    job_queue_check = _get_health_checks(settings, job_queue)["job_queue"]
    components = {
        "job_queue": bool(await job_queue_check.first_status(READINESS_FIRST_CHECK_WAIT_SECONDS)),
    }
    ready = all(components.values())

    return Response(
        content=ReadinessResponse(
            status="ready" if ready else "not_ready",
            timestamp=datetime.now(timezone.utc).isoformat(),
            components=components,
        ),
        status_code=200 if ready else 503,
    )


@get("/health")
async def health_check() -> HealthResponse:
    """Health report of every component, for dashboards and people.

    Always returns 200; use /health/live and /health/ready for probes.
    ``job_queue`` and ``llm`` report the last background check of each
    dependency (refreshed every JOB_QUEUE_HEALTH_TTL_SECONDS and
    LLM_HEALTH_TTL_SECONDS) and are false until a first check has passed.
    """
    settings, job_queue = await get_dependencies_async()
    checks = _get_health_checks(settings, job_queue)

    components = {
        "api": True,
        "job_queue": bool(checks["job_queue"].status()),
        "llm": bool(checks["llm"].status()),
    }

    return HealthResponse(
        status="healthy" if all(components.values()) else "degraded",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
    )


@post("/pipeline/start")
async def start_pipeline(data: PipelineStartRequest) -> PipelineStartResponse:
    """Start a new pipeline execution.

    With the Redis queue and an RQ worker running, the run is handed to the
    worker and this returns 202 with status "queued"; poll
    /api/pipeline/status/{job_id}. Otherwise the pipeline runs in a thread of
    this process, so the event loop stays free, and the response reports the
    outcome.
    """
    settings, job_queue = get_dependencies()

    try:
        # Create pipeline config
        phases = [PipelinePhase(p) for p in data.phases]
        config = PipelineConfig(
            phases=phases,
            human_in_loop=data.human_in_loop,
        )

        # Record the job first: the worker (or thread) loads it by id, and
        # /pipeline/status can report on it straight away.
        job = create_pipeline_job(
            job_queue,
            config,
            data_source=data.data_source,
            task_description=data.task_description,
            output_destination=data.output_destination,
        )
        created_at = datetime.now(timezone.utc).isoformat()

        if job_queue.submit(run_pipeline_job, job.id, job_timeout=settings.pipeline_job_timeout):
            return Response(
                content=PipelineStartResponse(
                    job_id=job.id,
                    status="queued",
                    message=f"Pipeline queued for a worker; poll /api/pipeline/status/{job.id}",
                    created_at=created_at,
                ).model_dump(),
                status_code=202,
            )

        # No worker available: run here, off the event loop
        result = await to_thread.run_sync(run_pipeline_job, job.id, job_queue)

        return PipelineStartResponse(
            job_id=job.id,
            status=result["status"],
            message=_status_message(result),
            created_at=created_at,
            current_agent=result["current_agent"],
            error=result["error"],
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


def _status_message(result: dict[str, Any]) -> str:
    """Human-readable summary of a pipeline run or resume result."""
    status = result["status"]
    if status == "awaiting_approval":
        return f"Waiting for review of {result['current_agent']}"
    if status == "completed":
        return "Pipeline completed"
    if status == "cancelled":
        return "Pipeline stopped by user"
    return f"Pipeline failed: {result.get('error') or 'unknown error'}"


@post("/pipeline/action/{job_id:str}")
async def pipeline_action(job_id: str, data: ActionRequest) -> dict[str, Any]:
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

    if action not in REVIEW_ACTIONS:
        return Response(
            content=ErrorResponse(
                error="Invalid action",
                detail=f"Action must be one of: {sorted(a.value for a in REVIEW_ACTIONS)}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
            status_code=400,
        )

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

    if job.status != JobStatus.PAUSED or not job.pipeline_state:
        return Response(
            content=ErrorResponse(
                error="Action not allowed",
                detail=f"Job {job_id} is not awaiting review (status: {job.status.value})",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
            status_code=409,
        )

    # Claim the review before handing it off, so a second action gets 409 and
    # this request's copy of the job can't overwrite a worker's later updates.
    job.resume(data.feedback)
    job_queue.update_job(job)

    if job_queue.submit(
        resume_pipeline_job,
        job_id,
        action.value,
        data.feedback,
        job_timeout=settings.pipeline_job_timeout,
    ):
        return Response(
            content={
                "status": "queued",
                "action": action.value,
                "job_id": job_id,
                "message": f"Review action queued for a worker; poll /api/pipeline/status/{job_id}",
            },
            status_code=202,
        )

    # No worker available: resume here, off the event loop
    try:
        result = await to_thread.run_sync(
            resume_pipeline_job, job_id, action.value, data.feedback, job_queue
        )
    except Exception as e:
        return Response(
            content=ErrorResponse(
                error="Action failed",
                detail=str(e),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ).model_dump(),
            status_code=500,
        )

    return {
        "status": "success",
        "action": action.value,
        "job_id": job_id,
        "pipeline_status": result["status"],
        "current_agent": result["current_agent"],
        "message": _status_message(result),
        "error": result["error"],
    }


def _run_single_agent(settings: Settings, data: AgentRunRequest, job_id: str) -> dict[str, Any]:
    """Build and run one agent to completion. Blocking: call it from a worker thread."""
    from agentds.agents.base import AgentContext
    from agentds.core.artifact_store import ArtifactStore
    from agentds.core.llm_gateway import LLMGateway

    # One gateway and store, shared by the agent and its context
    llm_gateway = LLMGateway(settings)
    artifact_store = ArtifactStore(settings)
    agent = AGENT_REGISTRY[data.agent_name](  # type: ignore[abstract]
        settings=settings,
        llm_gateway=llm_gateway,
        artifact_store=artifact_store,
    )
    context = AgentContext(
        job_id=job_id,
        settings=settings,
        llm_gateway=llm_gateway,
        artifact_store=artifact_store,
        task_description=data.task_description,
        extra={
            "data_source": data.data_source,
            **data.config,
        },
    )

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


@post("/agent/run")
async def run_agent(data: AgentRunRequest) -> dict[str, Any]:
    """Run a single agent independently.

    The agent runs in a worker thread, so the API keeps serving other
    requests (including health probes) while it works.
    """
    settings, job_queue = get_dependencies()

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
        return await to_thread.run_sync(_run_single_agent, settings, data, str(uuid.uuid4()))

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


@delete("/jobs/{job_id:str}", status_code=200)
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
        liveness_check,
        readiness_check,
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
        # Pass the app's settings on, so API keys and rate limits come from them
        middleware=[
            DefineMiddleware(AuthenticationMiddleware, settings=settings),
            DefineMiddleware(RateLimitMiddleware, settings=settings),
        ],
    )


# CLI entry point
if __name__ == "__main__":
    import uvicorn

    app = create_api()
    uvicorn.run(app, host="0.0.0.0", port=8000)
