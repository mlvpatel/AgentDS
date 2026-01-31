"""
AgentDS Job Queue.

Provides job queue management with Redis Queue (RQ) backend.

Author: Malav Patel
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentds.core.config import Settings, get_settings
from agentds.core.logger import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Job(BaseModel):
    """Job model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Job name")
    status: JobStatus = Field(default=JobStatus.PENDING)
    priority: JobPriority = Field(default=JobPriority.NORMAL)

    # Job data
    task: str = Field(..., description="Task identifier")
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Progress tracking
    current_agent: str | None = None
    current_step: int = Field(default=0)
    total_steps: int = Field(default=0)
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)

    # Results
    result: Any | None = None
    error: str | None = None

    # User interaction
    requires_approval: bool = Field(default=False)
    user_feedback: str | None = None

    def update_progress(
        self,
        step: int,
        total: int,
        agent: str | None = None,
    ) -> None:
        """Update job progress."""
        self.current_step = step
        self.total_steps = total
        self.progress_percent = (step / total * 100) if total > 0 else 0.0
        if agent:
            self.current_agent = agent
        self.updated_at = datetime.now(timezone.utc)

    def mark_running(self) -> None:
        """Mark job as running."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self, result: Any = None) -> None:
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.progress_percent = 100.0
        if result is not None:
            self.result = result

    def mark_failed(self, error: str) -> None:
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.error = error

    def mark_cancelled(self) -> None:
        """Mark job as cancelled."""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def mark_paused(self) -> None:
        """Mark job as paused (awaiting approval)."""
        self.status = JobStatus.PAUSED
        self.requires_approval = True
        self.updated_at = datetime.now(timezone.utc)

    def resume(self, feedback: str | None = None) -> None:
        """Resume paused job."""
        self.status = JobStatus.RUNNING
        self.requires_approval = False
        if feedback:
            self.user_feedback = feedback
        self.updated_at = datetime.now(timezone.utc)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate job duration in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.now(timezone.utc)
        return (end_time - self.started_at).total_seconds()


class InMemoryQueue:
    """Simple in-memory queue for development."""

    def __init__(self) -> None:
        """Initialize in-memory queue."""
        self._jobs: dict[str, Job] = {}
        self._queue: list[str] = []

    def enqueue(self, job: Job) -> str:
        """Add job to queue."""
        self._jobs[job.id] = job
        self._queue.append(job.id)
        job.status = JobStatus.QUEUED
        return job.id

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def update_job(self, job: Job) -> bool:
        """Update job in queue."""
        if job.id in self._jobs:
            self._jobs[job.id] = job
            return True
        return False

    def delete_job(self, job_id: str) -> bool:
        """Delete job from queue."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            if job_id in self._queue:
                self._queue.remove(job_id)
            return True
        return False

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]


class RedisQueue:
    """Redis Queue (RQ) based job queue."""

    def __init__(self, url: str) -> None:
        """
        Initialize Redis queue.

        Args:
            url: Redis connection URL
        """
        import redis
        from rq import Queue

        self._connection = redis.from_url(url)
        self._queue = Queue(connection=self._connection)
        self._jobs_key = "agentds:jobs"

    def enqueue(self, job: Job) -> str:
        """Add job to queue."""
        import pickle

        # Store job data in Redis hash
        self._connection.hset(
            self._jobs_key,
            job.id,
            pickle.dumps(job),
        )
        job.status = JobStatus.QUEUED
        return job.id

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        import pickle

        data = self._connection.hget(self._jobs_key, job_id)
        if data:
            return pickle.loads(data)
        return None

    def update_job(self, job: Job) -> bool:
        """Update job in queue."""
        import pickle

        if self._connection.hexists(self._jobs_key, job.id):
            self._connection.hset(self._jobs_key, job.id, pickle.dumps(job))
            return True
        return False

    def delete_job(self, job_id: str) -> bool:
        """Delete job from queue."""
        return bool(self._connection.hdel(self._jobs_key, job_id))

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs, optionally filtered by status."""
        import pickle

        jobs = []
        for data in self._connection.hvals(self._jobs_key):
            job = pickle.loads(data)
            if status is None or job.status == status:
                jobs.append(job)

        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]

    def ping(self) -> bool:
        """Check Redis connection."""
        try:
            return self._connection.ping()
        except Exception:
            return False


class JobQueue:
    """
    Job queue with automatic backend selection.

    Tries Redis first, falls back to in-memory queue if unavailable.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize job queue.

        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self._backend: InMemoryQueue | RedisQueue
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize appropriate queue backend."""
        if self.settings.is_feature_enabled("redis_queue"):
            try:
                redis_queue = RedisQueue(self.settings.redis.url)
                if redis_queue.ping():
                    self._backend = redis_queue
                    logger.info("Using Redis job queue backend")
                    return
            except Exception as e:
                logger.warning(
                    "Redis unavailable, falling back to memory queue", error=str(e)
                )

        self._backend = InMemoryQueue()
        logger.info("Using in-memory job queue backend")

    def create_job(
        self,
        name: str,
        task: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        priority: JobPriority = JobPriority.NORMAL,
    ) -> Job:
        """
        Create and enqueue a new job.

        Args:
            name: Job name
            task: Task identifier
            args: Positional arguments
            kwargs: Keyword arguments
            priority: Job priority

        Returns:
            Created Job instance
        """
        job = Job(
            name=name,
            task=task,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
        )
        self._backend.enqueue(job)
        logger.info("Job created", job_id=job.id, name=name, task=task)
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._backend.get_job(job_id)

    def update_job(self, job: Job) -> bool:
        """Update job."""
        return self._backend.update_job(job)

    def delete_job(self, job_id: str) -> bool:
        """Delete job."""
        return self._backend.delete_job(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs."""
        return self._backend.list_jobs(status, limit)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.get_job(job_id)
        if job and job.status in [JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED]:
            job.mark_cancelled()
            self.update_job(job)
            logger.info("Job cancelled", job_id=job_id)
            return True
        return False

    def pause_job(self, job_id: str) -> bool:
        """Pause a running job."""
        job = self.get_job(job_id)
        if job and job.status == JobStatus.RUNNING:
            job.mark_paused()
            self.update_job(job)
            logger.info("Job paused", job_id=job_id)
            return True
        return False

    def resume_job(self, job_id: str, feedback: str | None = None) -> bool:
        """Resume a paused job."""
        job = self.get_job(job_id)
        if job and job.status == JobStatus.PAUSED:
            job.resume(feedback)
            self.update_job(job)
            logger.info("Job resumed", job_id=job_id)
            return True
        return False
