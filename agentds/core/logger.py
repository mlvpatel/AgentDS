"""
Personal Data Scientist - Structured Logging with Logfire.

Provides JSON-structured logging with context propagation and Logfire observability.

Author: Malav Patel
"""

from __future__ import annotations

import contextlib
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog
from structlog.types import Processor

from agentds.core.config import get_settings

# Context variables for request tracking
job_id_ctx: ContextVar[str | None] = ContextVar("job_id", default=None)
agent_ctx: ContextVar[str | None] = ContextVar("agent", default=None)

# Logfire instance (initialized lazily)
_logfire_configured = False


def configure_logfire() -> None:
    """Configure Logfire for observability."""
    global _logfire_configured
    if _logfire_configured:
        return

    try:
        import logfire

        settings = get_settings()

        logfire.configure(
            service_name="agentds",
            environment=settings.environment,
            send_to_logfire=settings.environment != "test",
        )

        # Instrument Pydantic-AI if available
        with contextlib.suppress(Exception):
            logfire.instrument_pydantic_ai()

        # Instrument HTTPX for API calls
        with contextlib.suppress(Exception):
            logfire.instrument_httpx()

        _logfire_configured = True

    except ImportError:
        pass  # Logfire not installed


def add_timestamp(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add ISO timestamp to log event."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_context(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add context variables to log event."""
    job_id = job_id_ctx.get()
    agent = agent_ctx.get()
    if job_id:
        event_dict["job_id"] = job_id
    if agent:
        event_dict["agent"] = agent
    return event_dict


def add_app_info(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add application info to log event."""
    settings = get_settings()
    event_dict["app"] = settings.app_name
    event_dict["version"] = settings.app_version
    event_dict["environment"] = settings.environment
    return event_dict


def setup_logging(
    level: str | None = None,
    log_format: str | None = None,
    log_file: Path | None = None,
    enable_logfire: bool = True,
) -> None:
    """
    Setup structured logging with optional Logfire.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_format: Output format (json, text)
        log_file: Optional file path for logging
        enable_logfire: Enable Logfire observability
    """
    settings = get_settings()
    level = level or settings.log_level
    log_format = log_format or settings.log_format

    # Configure Logfire if enabled
    if enable_logfire:
        configure_logfire()

    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Define processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_timestamp,
        add_context,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Configure structlog
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(numeric_level)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    for lib in ["httpx", "httpcore", "urllib3", "asyncio", "litellm"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


@lru_cache(maxsize=100)
def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Bound logger instance
    """
    return structlog.get_logger(name)


def get_logfire_span(name: str, **attributes: Any):
    """
    Get a Logfire span for tracing.

    Args:
        name: Span name
        **attributes: Span attributes

    Returns:
        Logfire span context manager or no-op
    """
    try:
        import logfire
        return logfire.span(name, **attributes)
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()


def log_llm_call(
    model: str,
    tokens: int,
    cost: float,
    latency_ms: float,
    agent: str | None = None,
) -> None:
    """
    Log an LLM call for observability.

    Args:
        model: Model name
        tokens: Total tokens used
        cost: Cost in USD
        latency_ms: Latency in milliseconds
        agent: Agent name (optional)
    """
    try:
        import logfire
        logfire.info(
            "LLM call",
            model=model,
            tokens=tokens,
            cost=cost,
            latency_ms=latency_ms,
            agent=agent,
        )
    except ImportError:
        logger = get_logger(__name__)
        logger.info(
            "LLM call",
            model=model,
            tokens=tokens,
            cost=cost,
            latency_ms=latency_ms,
            agent=agent,
        )


class LogContext:
    """Context manager for setting log context."""

    def __init__(
        self,
        job_id: str | None = None,
        agent: str | None = None,
    ) -> None:
        """
        Initialize log context.

        Args:
            job_id: Job identifier
            agent: Agent name
        """
        self.job_id = job_id
        self.agent = agent
        self._job_token: Any | None = None
        self._agent_token: Any | None = None

    def __enter__(self) -> "LogContext":
        """Enter context and set variables."""
        if self.job_id:
            self._job_token = job_id_ctx.set(self.job_id)
        if self.agent:
            self._agent_token = agent_ctx.set(self.agent)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and reset variables."""
        if self._job_token:
            job_id_ctx.reset(self._job_token)
        if self._agent_token:
            agent_ctx.reset(self._agent_token)


def set_job_context(job_id: str) -> None:
    """Set job ID in logging context."""
    job_id_ctx.set(job_id)


def set_agent_context(agent: str) -> None:
    """Set agent name in logging context."""
    agent_ctx.set(agent)


def clear_context() -> None:
    """Clear all context variables."""
    job_id_ctx.set(None)
    agent_ctx.set(None)
