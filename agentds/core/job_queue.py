"""
AgentDS Job Queue.

Provides job queue management with Redis Queue (RQ) backend.

Author: Malav Patel
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
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

    # Pipeline graph state saved while paused for review, so the run can be
    # resumed by a different process (for example a background worker)
    pipeline_state: dict[str, Any] | None = None

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

    def __init__(self, url: str, socket_timeout: float | None = None) -> None:
        """
        Initialize Redis queue.

        Args:
            url: Redis connection URL
            socket_timeout: Seconds to wait when connecting to or reading from
                Redis; None waits indefinitely
        """
        import redis
        from rq import Queue

        self._connection = redis.from_url(
            url,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
        )
        self._queue = Queue(connection=self._connection)
        self._jobs_key = "agentds:jobs"

    def enqueue(self, job: Job) -> str:
        """Add job to queue."""
        import pickle

        # Set the status before storing, so readers see "queued"
        job.status = JobStatus.QUEUED
        self._connection.hset(
            self._jobs_key,
            job.id,
            pickle.dumps(job),
        )
        return job.id

    def store(self, job: Job) -> None:
        """Write a job as it is, creating or replacing its record."""
        import pickle

        self._connection.hset(self._jobs_key, job.id, pickle.dumps(job))

    def submit(self, func: Callable[..., Any], *args: Any, job_timeout: int) -> str | None:
        """
        Hand a task to an RQ worker.

        Returns the RQ job id, or None when no worker is listening on the
        queue (the task would otherwise wait in Redis indefinitely).
        """
        from rq import Worker

        if Worker.count(queue=self._queue) == 0:
            return None
        return self._queue.enqueue(func, *args, job_timeout=job_timeout).id

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

    Uses Redis when the ``redis_queue`` feature is enabled. If Redis can't be
    reached, it falls back to an in-memory queue so the process still works,
    reports itself unhealthy (``ping()`` is False) and reconnects on a later
    ``ping()`` once Redis answers. With the feature disabled, the in-memory
    queue is the intended backend.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize job queue.

        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self._redis_wanted = self.settings.is_feature_enabled("redis_queue")
        self._backend_lock = threading.Lock()
        self._backend: InMemoryQueue | RedisQueue
        self._init_backend()

    def _connect_redis(self) -> RedisQueue | None:
        """Return a connected Redis backend, or None if Redis can't be reached."""
        try:
            redis_queue = RedisQueue(
                self.settings.redis.url,
                socket_timeout=self.settings.redis.socket_timeout,
            )
            if redis_queue.ping():
                return redis_queue
        except Exception as e:
            logger.warning("Redis unavailable", error=str(e))
        return None

    def _init_backend(self) -> None:
        """Initialize appropriate queue backend."""
        if self._redis_wanted:
            redis_queue = self._connect_redis()
            if redis_queue is not None:
                self._backend = redis_queue
                logger.info("Using Redis job queue backend")
                return
            logger.warning(
                "Redis unavailable, falling back to memory queue until it can be reached"
            )

        self._backend = InMemoryQueue()
        logger.info("Using in-memory job queue backend")

    @property
    def using_fallback(self) -> bool:
        """True while the Redis queue is wanted but the in-memory queue is in use."""
        return self._redis_wanted and isinstance(self._backend, InMemoryQueue)

    def _reconnect_redis(self) -> bool:
        """
        Switch from the in-memory fallback to Redis if Redis now answers.

        Jobs created while on the fallback are copied into Redis, so their ids
        keep working.
        """
        with self._backend_lock:
            fallback = self._backend
            if not isinstance(fallback, InMemoryQueue):
                return True  # another caller already switched
            redis_queue = self._connect_redis()
            if redis_queue is None:
                return False
            jobs = list(fallback._jobs.values())
            for job in jobs:
                redis_queue.store(job)
            self._backend = redis_queue
        logger.info(
            "Reconnected to Redis; job queue switched from the in-memory fallback",
            jobs_moved=len(jobs),
        )
        return True

    def create_job(
        self,
        name: str,
        task: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        priority: JobPriority = JobPriority.NORMAL,
        job_id: str | None = None,
    ) -> Job:
        """
        Create and enqueue a new job.

        Args:
            name: Job name
            task: Task identifier
            args: Positional arguments
            kwargs: Keyword arguments
            priority: Job priority
            job_id: Use this id instead of generating one

        Returns:
            Created Job instance
        """
        job = Job(
            name=name,
            task=task,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            **({"id": job_id} if job_id else {}),
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

    def ping(self) -> bool:
        """
        Check the backend can be reached.

        Pings Redis for the Redis backend. On the in-memory fallback (Redis
        wanted but unreachable), tries to reconnect and is False until that
        works. When the Redis queue is disabled, the in-memory backend is the
        intended one and always reachable.
        """
        if isinstance(self._backend, RedisQueue):
            return self._backend.ping()
        if self.using_fallback:
            return self._reconnect_redis()
        return True

    def submit(self, func: Callable[..., Any], *args: Any, job_timeout: int) -> bool:
        """
        Run a task on an RQ background worker, if one is available.

        Args:
            func: Importable task function (for example from agentds.workflows.tasks)
            *args: Task arguments; they must be picklable
            job_timeout: Seconds the worker may spend on the task

        Returns:
            True if a worker will run the task; False if the caller must run it
            (in-memory backend, or no worker listening on the Redis queue).
        """
        if not isinstance(self._backend, RedisQueue):
            return False
        rq_job_id = self._backend.submit(func, *args, job_timeout=job_timeout)
        if rq_job_id is None:
            logger.warning("No RQ worker is listening; running the task in this process")
            return False
        logger.info("Task submitted to RQ worker", task=getattr(func, "__name__", str(func)), rq_job_id=rq_job_id)
        return True

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
