"""
Tests for AgentDS Custom Exceptions.

Author: Malav Patel
"""

import pytest

from agentds.core.exceptions import (
    AgentDSError,
    AgentError,
    AgentExecutionError,
    AgentNotFoundError,
    AllProvidersFailedError,
    AuthenticationError,
    ConfigurationError,
    DataError,
    DataFormatError,
    DataLoadError,
    ExpiredTokenError,
    FileSizeLimitError,
    InvalidAPIKeyError,
    InvalidContentTypeError,
    InvalidURLError,
    JobAlreadyExistsError,
    JobError,
    JobNotFoundError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    MissingAPIKeyError,
    PathTraversalError,
    PipelineCancelledError,
    PipelineError,
    PipelineTimeoutError,
    RateLimitError,
    ValidationError,
)


class TestAgentDSError:
    """Tests for base AgentDSError."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = AgentDSError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.code == "AGENTDS_ERROR"
        assert error.details == {}

    def test_error_with_code_and_details(self) -> None:
        """Test error with code and details."""
        error = AgentDSError(
            "Custom error",
            code="CUSTOM_CODE",
            details={"key": "value"},
        )
        assert error.code == "CUSTOM_CODE"
        assert error.details == {"key": "value"}

    def test_to_dict(self) -> None:
        """Test to_dict method for API responses."""
        error = AgentDSError("Test", code="TEST", details={"foo": "bar"})
        result = error.to_dict()
        assert result == {
            "error": "TEST",
            "message": "Test",
            "details": {"foo": "bar"},
        }


class TestConfigurationErrors:
    """Tests for configuration errors."""

    def test_configuration_error(self) -> None:
        """Test ConfigurationError."""
        error = ConfigurationError("Missing config", config_key="api_key")
        assert error.code == "CONFIGURATION_ERROR"
        assert error.details["config_key"] == "api_key"

    def test_missing_api_key_error(self) -> None:
        """Test MissingAPIKeyError."""
        error = MissingAPIKeyError("openai")
        assert "openai" in error.message
        assert error.code == "MISSING_API_KEY"
        assert error.details["provider"] == "openai"


class TestLLMErrors:
    """Tests for LLM errors."""

    def test_llm_error(self) -> None:
        """Test base LLMError."""
        error = LLMError("Provider failed", model="gpt-4", provider="openai")
        assert error.code == "LLM_ERROR"
        assert error.details["model"] == "gpt-4"
        assert error.details["provider"] == "openai"

    def test_llm_rate_limit_error(self) -> None:
        """Test LLMRateLimitError."""
        error = LLMRateLimitError(model="gpt-4", provider="openai", retry_after=30)
        assert error.code == "LLM_RATE_LIMIT"
        assert error.retry_after == 30
        assert error.details["retry_after_seconds"] == 30

    def test_llm_timeout_error(self) -> None:
        """Test LLMTimeoutError."""
        error = LLMTimeoutError(model="claude-3", timeout_seconds=120)
        assert error.code == "LLM_TIMEOUT"
        assert error.details["timeout_seconds"] == 120

    def test_llm_connection_error(self) -> None:
        """Test LLMConnectionError."""
        error = LLMConnectionError(provider="anthropic", original_error="Connection refused")
        assert error.code == "LLM_CONNECTION_ERROR"
        assert error.details["original_error"] == "Connection refused"

    def test_llm_response_error(self) -> None:
        """Test LLMResponseError."""
        error = LLMResponseError("Parse failed", response_content="invalid json")
        assert error.code == "LLM_RESPONSE_ERROR"
        assert "invalid json" in error.details["response_preview"]

    def test_all_providers_failed_error(self) -> None:
        """Test AllProvidersFailedError."""
        error = AllProvidersFailedError(["gpt-4", "claude-3", "gemini-pro"])
        assert error.code == "ALL_PROVIDERS_FAILED"
        assert len(error.details["attempted_models"]) == 3


class TestPipelineErrors:
    """Tests for pipeline errors."""

    def test_pipeline_error(self) -> None:
        """Test base PipelineError."""
        error = PipelineError("Execution failed", job_id="123", phase="build")
        assert error.code == "PIPELINE_ERROR"
        assert error.details["job_id"] == "123"
        assert error.details["phase"] == "build"

    def test_pipeline_cancelled_error(self) -> None:
        """Test PipelineCancelledError."""
        error = PipelineCancelledError(job_id="123")
        assert error.code == "PIPELINE_CANCELLED"
        assert "cancelled" in error.message.lower()

    def test_pipeline_timeout_error(self) -> None:
        """Test PipelineTimeoutError."""
        error = PipelineTimeoutError(job_id="123", timeout_minutes=60)
        assert error.code == "PIPELINE_TIMEOUT"
        assert error.details["timeout_minutes"] == 60


class TestAgentErrors:
    """Tests for agent errors."""

    def test_agent_error(self) -> None:
        """Test base AgentError."""
        error = AgentError("Agent failed", agent_name="DataLoader", job_id="123")
        assert error.code == "AGENT_ERROR"
        assert error.details["agent_name"] == "DataLoader"

    def test_agent_execution_error(self) -> None:
        """Test AgentExecutionError."""
        error = AgentExecutionError(
            agent_name="EDA",
            original_error="Memory limit exceeded",
            job_id="123",
        )
        assert error.code == "AGENT_EXECUTION_ERROR"
        assert "EDA" in error.message
        assert "Memory limit" in error.message

    def test_agent_not_found_error(self) -> None:
        """Test AgentNotFoundError."""
        error = AgentNotFoundError("UnknownAgent")
        assert error.code == "AGENT_NOT_FOUND"
        assert "UnknownAgent" in error.message


class TestValidationErrors:
    """Tests for validation errors."""

    def test_validation_error(self) -> None:
        """Test base ValidationError."""
        error = ValidationError("Invalid input", field="data_source")
        assert error.code == "VALIDATION_ERROR"
        assert error.details["field"] == "data_source"

    def test_path_traversal_error(self) -> None:
        """Test PathTraversalError."""
        error = PathTraversalError("../../../etc/passwd")
        assert error.code == "PATH_TRAVERSAL"
        # Path should NOT be in the error message for security
        assert "etc/passwd" not in error.message

    def test_file_size_limit_error(self) -> None:
        """Test FileSizeLimitError."""
        error = FileSizeLimitError(size_bytes=200_000_000, limit_bytes=100_000_000)
        assert error.code == "FILE_SIZE_LIMIT"
        assert error.details["size_bytes"] == 200_000_000

    def test_invalid_content_type_error(self) -> None:
        """Test InvalidContentTypeError."""
        error = InvalidContentTypeError("text/html", ["text/csv", "application/json"])
        assert error.code == "INVALID_CONTENT_TYPE"
        assert "text/html" in error.message

    def test_invalid_url_error(self) -> None:
        """Test InvalidURLError."""
        error = InvalidURLError("Localhost not allowed")
        assert error.code == "INVALID_URL"


class TestAuthenticationErrors:
    """Tests for authentication errors."""

    def test_authentication_error(self) -> None:
        """Test base AuthenticationError."""
        error = AuthenticationError("Auth failed")
        assert error.code == "AUTHENTICATION_ERROR"

    def test_invalid_api_key_error(self) -> None:
        """Test InvalidAPIKeyError."""
        error = InvalidAPIKeyError()
        assert error.code == "INVALID_API_KEY"
        assert "API key" in error.message

    def test_expired_token_error(self) -> None:
        """Test ExpiredTokenError."""
        error = ExpiredTokenError()
        assert error.code == "EXPIRED_TOKEN"
        assert "expired" in error.message.lower()


class TestRateLimitError:
    """Tests for rate limit error."""

    def test_rate_limit_error(self) -> None:
        """Test RateLimitError."""
        error = RateLimitError(limit=60, window_seconds=60, retry_after=30)
        assert error.code == "RATE_LIMIT_EXCEEDED"
        assert error.retry_after == 30
        assert error.details["limit"] == 60


class TestDataErrors:
    """Tests for data errors."""

    def test_data_error(self) -> None:
        """Test base DataError."""
        error = DataError("Data issue", source="file.csv")
        assert error.code == "DATA_ERROR"
        assert error.details["source"] == "file.csv"

    def test_data_load_error(self) -> None:
        """Test DataLoadError."""
        error = DataLoadError(source="s3://bucket/data.parquet", reason="Access denied")
        assert error.code == "DATA_LOAD_ERROR"
        assert "Access denied" in error.message

    def test_data_format_error(self) -> None:
        """Test DataFormatError."""
        error = DataFormatError(format="xml", supported_formats=["csv", "parquet", "json"])
        assert error.code == "DATA_FORMAT_ERROR"
        assert "xml" in error.message


class TestJobErrors:
    """Tests for job errors."""

    def test_job_error(self) -> None:
        """Test base JobError."""
        error = JobError("Job issue", job_id="abc-123")
        assert error.code == "JOB_ERROR"
        assert error.details["job_id"] == "abc-123"

    def test_job_not_found_error(self) -> None:
        """Test JobNotFoundError."""
        error = JobNotFoundError(job_id="abc-123")
        assert error.code == "JOB_NOT_FOUND"

    def test_job_already_exists_error(self) -> None:
        """Test JobAlreadyExistsError."""
        error = JobAlreadyExistsError(job_id="abc-123")
        assert error.code == "JOB_ALREADY_EXISTS"


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_exceptions_inherit_from_base(self) -> None:
        """Test all custom exceptions inherit from AgentDSError."""
        exceptions = [
            ConfigurationError("test"),
            LLMError("test"),
            PipelineError("test"),
            AgentError("test"),
            ValidationError("test"),
            AuthenticationError("test"),
            RateLimitError(60, 60),
            DataError("test"),
            JobError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, AgentDSError)
            assert hasattr(exc, "to_dict")
            assert hasattr(exc, "code")
            assert hasattr(exc, "details")

    def test_exception_catching(self) -> None:
        """Test exception hierarchy for catching."""
        # LLM errors should be catchable as AgentDSError
        with pytest.raises(AgentDSError):
            raise LLMRateLimitError()

        # More specific catch should work
        with pytest.raises(LLMError):
            raise LLMTimeoutError()
