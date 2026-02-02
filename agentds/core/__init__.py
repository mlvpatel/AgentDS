"""
AgentDS Core Module.

This module provides core functionality including:
- Configuration management
- LLM gateway for multi-provider support
- Caching layer
- Job queue management
- Artifact storage
- Structured logging
- APO (Automatic Prompt Optimization)

Author: Malav Patel
"""

from agentds.core.apo import (
    APOOptimizer,
    BeamSearch,
    OptimizationResult,
    PromptCandidate,
    PromptHistory,
    RewardAggregator,
    create_apo_optimizer,
)
from agentds.core.artifact_store import ArtifactStore
from agentds.core.cache_layer import CacheLayer, get_cache
from agentds.core.config import APOSettings, Settings, get_settings
from agentds.core.exceptions import (
    AgentDSError,
    AgentError,
    AuthenticationError,
    ConfigurationError,
    DataError,
    JobError,
    LLMError,
    PipelineError,
    RateLimitError,
    ValidationError,
)
from agentds.core.job_queue import Job, JobQueue, JobStatus
from agentds.core.llm_gateway import LLMGateway
from agentds.core.logger import get_logger, setup_logging
from agentds.core.validation import (
    InputValidator,
    sanitize_path,
    validate_content_type,
    validate_file_size,
    validate_url,
)

__all__ = [
    # Config
    "Settings",
    "APOSettings",
    "get_settings",
    # LLM
    "LLMGateway",
    # Cache
    "CacheLayer",
    "get_cache",
    # Jobs
    "JobQueue",
    "Job",
    "JobStatus",
    # Artifacts
    "ArtifactStore",
    # Logging
    "get_logger",
    "setup_logging",
    # Exceptions
    "AgentDSError",
    "ConfigurationError",
    "LLMError",
    "PipelineError",
    "AgentError",
    "ValidationError",
    "AuthenticationError",
    "RateLimitError",
    "DataError",
    "JobError",
    # Validation
    "InputValidator",
    "sanitize_path",
    "validate_file_size",
    "validate_content_type",
    "validate_url",
    # APO
    "APOOptimizer",
    "BeamSearch",
    "PromptCandidate",
    "PromptHistory",
    "RewardAggregator",
    "OptimizationResult",
    "create_apo_optimizer",
]

