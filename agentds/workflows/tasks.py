"""
AgentDS Background Tasks.

Pipeline runs executed by an RQ worker (``python -m rq.cli worker``) instead
of inside an API request. The API creates the job record with
:func:`~agentds.workflows.pipeline.create_pipeline_job` and submits one of
these functions; the worker loads the job from Redis, rebuilds the pipeline
from the configuration stored on it, and records progress back on the job.

The same functions run in-process when no worker is available.

Author: Malav Patel
"""

from __future__ import annotations

from typing import Any

from agentds.core.config import get_settings
from agentds.core.job_queue import Job, JobQueue
from agentds.workflows.pipeline import AgentDSPipeline, PipelineConfig


def run_pipeline_job(job_id: str, job_queue: JobQueue | None = None) -> dict[str, Any]:
    """
    Run a pipeline that was recorded with ``create_pipeline_job``.

    Runs until the pipeline completes, fails, or pauses for human review.

    Args:
        job_id: Job to run
        job_queue: Queue holding the job; workers leave this out and connect
            to the configured Redis queue

    Returns:
        Summary of the outcome: ``job_id``, ``status``, ``current_agent``,
        ``progress_percent`` and ``error``
    """
    queue = job_queue or JobQueue(get_settings())
    job = _load_job(queue, job_id)
    result = _pipeline_for(job, queue).run(
        data_source=job.kwargs["data_source"],
        task_description=job.kwargs["task_description"],
        output_destination=job.kwargs.get("output_destination"),
        job_id=job_id,
    )
    return _summary(result)


def resume_pipeline_job(
    job_id: str,
    user_action: str,
    user_feedback: str | None = None,
    job_queue: JobQueue | None = None,
) -> dict[str, Any]:
    """
    Resume a pipeline paused for human review.

    Works in any process: the pipeline state saved on the job when it paused
    is restored before the review action is applied.

    Args:
        job_id: Job to resume
        user_action: Review action, for example ``approve_and_continue``
        user_feedback: Optional feedback for a rerun
        job_queue: Queue holding the job; workers leave this out

    Returns:
        Summary of the outcome, as for :func:`run_pipeline_job`
    """
    queue = job_queue or JobQueue(get_settings())
    job = _load_job(queue, job_id)
    result = _pipeline_for(job, queue).resume(job_id, user_action, user_feedback)
    return _summary(result)


def _load_job(queue: JobQueue, job_id: str) -> Job:
    job = queue.get_job(job_id)
    if job is None:
        raise ValueError(f"No job with ID: {job_id}")
    return job


def _pipeline_for(job: Job, queue: JobQueue) -> AgentDSPipeline:
    config = PipelineConfig(**job.kwargs.get("pipeline_config", {}))
    return AgentDSPipeline(config=config, job_queue=queue)


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    """The parts of a pipeline result worth returning from a task (not the full state)."""
    return {
        key: result[key]
        for key in ("job_id", "status", "current_agent", "progress_percent", "error")
    }
