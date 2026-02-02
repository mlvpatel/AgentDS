"""
AgentDS Custom Exceptions.

Provides a comprehensive exception hierarchy for structured error handling
across the entire application.

Author: Malav Patel
"""

from __future__ import annotations

from typing import Any


class AgentDSError(Exception):
    """
    Base exception for all AgentDS errors.

    All custom exceptions should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize AgentDS error.

        Args:
            message: Human-readable error message
            code: Optional error code for programmatic handling
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.code = code or "AGENTDS_ERROR"
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(AgentDSError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)


class MissingAPIKeyError(ConfigurationError):
    """Raised when a required API key is not configured."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            message=f"API key not configured for provider: {provider}",
            config_key=f"{provider.upper()}_API_KEY",
            details={"provider": provider},
        )
        self.code = "MISSING_API_KEY"


# =============================================================================
# LLM Errors
# =============================================================================


class LLMError(AgentDSError):
    """Base exception for LLM-related errors."""

    def __init__(
        self,
        message: str,
        model: str | None = None,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if model:
            details["model"] = model
        if provider:
            details["provider"] = provider
        super().__init__(message, code="LLM_ERROR", details=details)


class LLMRateLimitError(LLMError):
    """Raised when LLM provider rate limit is exceeded."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(
            message="LLM provider rate limit exceeded",
            model=model,
            provider=provider,
            details=details,
        )
        self.code = "LLM_RATE_LIMIT"
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        super().__init__(
            message="LLM request timed out",
            model=model,
            provider=provider,
            details=details,
        )
        self.code = "LLM_TIMEOUT"


class LLMConnectionError(LLMError):
    """Raised when connection to LLM provider fails."""

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        original_error: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if original_error:
            details["original_error"] = original_error
        super().__init__(
            message="Failed to connect to LLM provider",
            model=model,
            provider=provider,
            details=details,
        )
        self.code = "LLM_CONNECTION_ERROR"


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid or cannot be parsed."""

    def __init__(
        self,
        message: str = "Invalid LLM response",
        model: str | None = None,
        response_content: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if response_content:
            # Truncate for safety
            details["response_preview"] = response_content[:500]
        super().__init__(message=message, model=model, details=details)
        self.code = "LLM_RESPONSE_ERROR"


class AllProvidersFailedError(LLMError):
    """Raised when all providers in fallback chain fail."""

    def __init__(self, attempted_models: list[str]) -> None:
        super().__init__(
            message="All LLM providers in fallback chain failed",
            details={"attempted_models": attempted_models},
        )
        self.code = "ALL_PROVIDERS_FAILED"


# =============================================================================
# Pipeline Errors
# =============================================================================


class PipelineError(AgentDSError):
    """Base exception for pipeline execution errors."""

    def __init__(
        self,
        message: str,
        job_id: str | None = None,
        phase: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if job_id:
            details["job_id"] = job_id
        if phase:
            details["phase"] = phase
        super().__init__(message, code="PIPELINE_ERROR", details=details)


class PipelineCancelledError(PipelineError):
    """Raised when pipeline is cancelled by user."""

    def __init__(self, job_id: str) -> None:
        super().__init__(
            message="Pipeline execution was cancelled",
            job_id=job_id,
        )
        self.code = "PIPELINE_CANCELLED"


class PipelineTimeoutError(PipelineError):
    """Raised when pipeline execution times out."""

    def __init__(self, job_id: str, timeout_minutes: int) -> None:
        super().__init__(
            message=f"Pipeline execution timed out after {timeout_minutes} minutes",
            job_id=job_id,
            details={"timeout_minutes": timeout_minutes},
        )
        self.code = "PIPELINE_TIMEOUT"


# =============================================================================
# Agent Errors
# =============================================================================


class AgentError(AgentDSError):
    """Base exception for agent execution errors."""

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        job_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if agent_name:
            details["agent_name"] = agent_name
        if job_id:
            details["job_id"] = job_id
        super().__init__(message, code="AGENT_ERROR", details=details)


class AgentExecutionError(AgentError):
    """Raised when agent execution fails."""

    def __init__(
        self,
        agent_name: str,
        original_error: str,
        job_id: str | None = None,
    ) -> None:
        super().__init__(
            message=f"Agent {agent_name} execution failed: {original_error}",
            agent_name=agent_name,
            job_id=job_id,
            details={"original_error": original_error},
        )
        self.code = "AGENT_EXECUTION_ERROR"


class AgentNotFoundError(AgentError):
    """Raised when requested agent does not exist."""

    def __init__(self, agent_name: str) -> None:
        super().__init__(
            message=f"Agent not found: {agent_name}",
            agent_name=agent_name,
        )
        self.code = "AGENT_NOT_FOUND"


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(AgentDSError):
    """Base exception for input validation errors."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if field:
            details["field"] = field
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class PathTraversalError(ValidationError):
    """Raised when path traversal attack is detected."""

    def __init__(self, path: str) -> None:
        # Don't include the actual path in the error for security
        super().__init__(
            message="Invalid path: directory traversal detected",
            field="path",
        )
        self.code = "PATH_TRAVERSAL"


class FileSizeLimitError(ValidationError):
    """Raised when file exceeds size limit."""

    def __init__(self, size_bytes: int, limit_bytes: int) -> None:
        super().__init__(
            message=f"File size ({size_bytes} bytes) exceeds limit ({limit_bytes} bytes)",
            field="file",
            details={
                "size_bytes": size_bytes,
                "limit_bytes": limit_bytes,
            },
        )
        self.code = "FILE_SIZE_LIMIT"


class InvalidContentTypeError(ValidationError):
    """Raised when content type is not allowed."""

    def __init__(self, content_type: str, allowed_types: list[str]) -> None:
        super().__init__(
            message=f"Content type '{content_type}' is not allowed",
            field="content_type",
            details={
                "content_type": content_type,
                "allowed_types": allowed_types,
            },
        )
        self.code = "INVALID_CONTENT_TYPE"


class InvalidURLError(ValidationError):
    """Raised when URL is invalid or not allowed."""

    def __init__(self, message: str = "Invalid URL") -> None:
        super().__init__(message=message, field="url")
        self.code = "INVALID_URL"


# =============================================================================
# Authentication & Authorization Errors
# =============================================================================


class AuthenticationError(AgentDSError):
    """Base exception for authentication errors."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="AUTHENTICATION_ERROR", details=details)


class InvalidAPIKeyError(AuthenticationError):
    """Raised when API key is invalid or missing."""

    def __init__(self) -> None:
        super().__init__(message="Invalid or missing API key")
        self.code = "INVALID_API_KEY"


class ExpiredTokenError(AuthenticationError):
    """Raised when authentication token has expired."""

    def __init__(self) -> None:
        super().__init__(message="Authentication token has expired")
        self.code = "EXPIRED_TOKEN"


# =============================================================================
# Rate Limiting Errors
# =============================================================================


class RateLimitError(AgentDSError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        limit: int,
        window_seconds: int,
        retry_after: int | None = None,
    ) -> None:
        details = {
            "limit": limit,
            "window_seconds": window_seconds,
        }
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            code="RATE_LIMIT_EXCEEDED",
            details=details,
        )
        self.retry_after = retry_after


# =============================================================================
# Data Errors
# =============================================================================


class DataError(AgentDSError):
    """Base exception for data-related errors."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if source:
            details["source"] = source
        super().__init__(message, code="DATA_ERROR", details=details)


class DataLoadError(DataError):
    """Raised when data cannot be loaded."""

    def __init__(self, source: str, reason: str) -> None:
        super().__init__(
            message=f"Failed to load data from {source}: {reason}",
            source=source,
            details={"reason": reason},
        )
        self.code = "DATA_LOAD_ERROR"


class DataFormatError(DataError):
    """Raised when data format is unsupported."""

    def __init__(self, format: str, supported_formats: list[str]) -> None:
        super().__init__(
            message=f"Unsupported data format: {format}",
            details={
                "format": format,
                "supported_formats": supported_formats,
            },
        )
        self.code = "DATA_FORMAT_ERROR"


# =============================================================================
# Job Queue Errors
# =============================================================================


class JobError(AgentDSError):
    """Base exception for job-related errors."""

    def __init__(
        self,
        message: str,
        job_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if job_id:
            details["job_id"] = job_id
        super().__init__(message, code="JOB_ERROR", details=details)


class JobNotFoundError(JobError):
    """Raised when job is not found."""

    def __init__(self, job_id: str) -> None:
        super().__init__(
            message=f"Job not found: {job_id}",
            job_id=job_id,
        )
        self.code = "JOB_NOT_FOUND"


class JobAlreadyExistsError(JobError):
    """Raised when job with same ID already exists."""

    def __init__(self, job_id: str) -> None:
        super().__init__(
            message=f"Job already exists: {job_id}",
            job_id=job_id,
        )
        self.code = "JOB_ALREADY_EXISTS"
