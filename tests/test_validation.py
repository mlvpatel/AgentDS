"""
Tests for AgentDS Input Validation.

Author: Malav Patel
"""

from pathlib import Path

import pytest

from agentds.core.exceptions import (
    FileSizeLimitError,
    InvalidContentTypeError,
    InvalidURLError,
    PathTraversalError,
    ValidationError,
)
from agentds.core.validation import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    InputValidator,
    sanitize_path,
    validate_agent_name,
    validate_content_type,
    validate_file_extension,
    validate_file_size,
    validate_job_id,
    validate_sql_query,
    validate_url,
)


class TestSanitizePath:
    """Tests for path sanitization."""

    def test_valid_relative_path(self) -> None:
        """Test valid relative path."""
        result = sanitize_path("data/input.csv")
        assert isinstance(result, Path)

    def test_path_traversal_double_dot(self) -> None:
        """Test path traversal with ../."""
        with pytest.raises(PathTraversalError):
            sanitize_path("../../../etc/passwd")

    def test_path_traversal_windows(self) -> None:
        """Test Windows path traversal."""
        with pytest.raises(PathTraversalError):
            sanitize_path("..\\..\\Windows\\System32")

    def test_path_traversal_etc(self) -> None:
        """Test /etc/ access attempt."""
        with pytest.raises(PathTraversalError):
            sanitize_path("/etc/passwd")

    def test_path_traversal_proc(self) -> None:
        """Test /proc/ access attempt."""
        with pytest.raises(PathTraversalError):
            sanitize_path("/proc/self/environ")

    def test_base_dir_restriction(self, tmp_path: Path) -> None:
        """Test base directory restriction."""
        # Create a file in temp dir
        safe_file = tmp_path / "data.csv"
        safe_file.touch()

        # Should work within base dir
        result = sanitize_path(safe_file, base_dir=tmp_path, allow_absolute=True)
        assert result == safe_file.resolve()

        # Should fail outside base dir
        with pytest.raises(PathTraversalError):
            sanitize_path("/tmp/other/file.csv", base_dir=tmp_path, allow_absolute=True)

    def test_absolute_path_not_allowed(self) -> None:
        """Test absolute path rejection when not allowed."""
        with pytest.raises(ValidationError):
            sanitize_path("/home/user/data.csv", allow_absolute=False)

    def test_absolute_path_allowed(self, tmp_path: Path) -> None:
        """Test absolute path allowed when explicitly enabled."""
        test_file = tmp_path / "test.csv"
        test_file.touch()
        result = sanitize_path(str(test_file), allow_absolute=True)
        assert result.is_absolute()


class TestValidateFileSize:
    """Tests for file size validation."""

    def test_valid_file_size(self, tmp_path: Path) -> None:
        """Test valid file size."""
        test_file = tmp_path / "small.csv"
        test_file.write_text("a,b,c\n1,2,3")

        size = validate_file_size(test_file)
        assert size > 0
        assert size < 100  # Small file

    def test_file_exceeds_limit(self, tmp_path: Path) -> None:
        """Test file exceeding size limit."""
        test_file = tmp_path / "large.csv"
        test_file.write_bytes(b"x" * 1000)  # 1KB file

        with pytest.raises(FileSizeLimitError):
            validate_file_size(test_file, max_size_bytes=500)

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test non-existent file."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_size(tmp_path / "nonexistent.csv")
        assert "does not exist" in str(exc_info.value)

    def test_path_is_directory(self, tmp_path: Path) -> None:
        """Test validation fails for directories."""
        with pytest.raises(ValidationError) as exc_info:
            validate_file_size(tmp_path)
        assert "not a file" in str(exc_info.value)


class TestValidateContentType:
    """Tests for content type validation."""

    def test_valid_csv(self) -> None:
        """Test valid CSV content type."""
        result = validate_content_type("text/csv")
        assert result == "text/csv"

    def test_valid_json(self) -> None:
        """Test valid JSON content type."""
        result = validate_content_type("application/json")
        assert result == "application/json"

    def test_content_type_with_charset(self) -> None:
        """Test content type with charset parameter."""
        result = validate_content_type("text/csv; charset=utf-8")
        assert result == "text/csv"

    def test_invalid_content_type(self) -> None:
        """Test invalid content type."""
        with pytest.raises(InvalidContentTypeError):
            validate_content_type("text/html")

    def test_custom_allowed_types(self) -> None:
        """Test custom allowed types."""
        result = validate_content_type("image/png", allowed_types=["image/png", "image/jpeg"])
        assert result == "image/png"


class TestValidateFileExtension:
    """Tests for file extension validation."""

    def test_valid_csv_extension(self) -> None:
        """Test valid CSV extension."""
        result = validate_file_extension("data.csv")
        assert result == ".csv"

    def test_valid_parquet_extension(self) -> None:
        """Test valid Parquet extension."""
        result = validate_file_extension("data.parquet")
        assert result == ".parquet"

    def test_case_insensitive(self) -> None:
        """Test case insensitive extension check."""
        result = validate_file_extension("data.CSV")
        assert result == ".csv"

    def test_invalid_extension(self) -> None:
        """Test invalid extension."""
        with pytest.raises(ValidationError):
            validate_file_extension("script.exe")

    def test_custom_extensions(self) -> None:
        """Test custom allowed extensions."""
        result = validate_file_extension("doc.pdf", allowed_extensions=[".pdf", ".doc"])
        assert result == ".pdf"


class TestValidateURL:
    """Tests for URL validation."""

    def test_valid_https_url(self) -> None:
        """Test valid HTTPS URL."""
        result = validate_url("https://example.com/data.csv")
        assert result == "https://example.com/data.csv"

    def test_valid_s3_url(self) -> None:
        """Test valid S3 URL."""
        result = validate_url("s3://bucket/key/data.parquet")
        assert "s3://" in result

    def test_valid_gs_url(self) -> None:
        """Test valid GCS URL."""
        result = validate_url("gs://bucket/data.csv")
        assert "gs://" in result

    def test_invalid_scheme(self) -> None:
        """Test invalid URL scheme."""
        with pytest.raises(InvalidURLError):
            validate_url("ftp://example.com/data.csv")

    def test_missing_scheme(self) -> None:
        """Test URL without scheme."""
        with pytest.raises(InvalidURLError):
            validate_url("example.com/data.csv")

    def test_localhost_not_allowed(self) -> None:
        """Test localhost rejection."""
        with pytest.raises(InvalidURLError):
            validate_url("http://localhost:8000/data.csv")

    def test_localhost_allowed(self) -> None:
        """Test localhost when explicitly allowed."""
        result = validate_url("http://localhost:8000/data.csv", allow_localhost=True)
        assert "localhost" in result

    def test_loopback_ip_not_allowed(self) -> None:
        """Test 127.0.0.1 rejection."""
        with pytest.raises(InvalidURLError):
            validate_url("http://127.0.0.1:8000/data.csv")


class TestValidateSQLQuery:
    """Tests for SQL query validation."""

    def test_valid_select(self) -> None:
        """Test valid SELECT query."""
        result = validate_sql_query("SELECT * FROM users WHERE id = 1")
        assert "SELECT" in result

    def test_drop_rejected(self) -> None:
        """Test DROP statement rejection."""
        with pytest.raises(ValidationError):
            validate_sql_query("DROP TABLE users")

    def test_delete_rejected(self) -> None:
        """Test DELETE statement rejection."""
        with pytest.raises(ValidationError):
            validate_sql_query("DELETE FROM users WHERE id = 1")

    def test_insert_rejected(self) -> None:
        """Test INSERT statement rejection."""
        with pytest.raises(ValidationError):
            validate_sql_query("INSERT INTO users VALUES (1, 'test')")

    def test_update_rejected(self) -> None:
        """Test UPDATE statement rejection."""
        with pytest.raises(ValidationError):
            validate_sql_query("UPDATE users SET name = 'hacked'")

    def test_sql_comment_rejected(self) -> None:
        """Test SQL comment injection rejection."""
        with pytest.raises(ValidationError):
            validate_sql_query("SELECT * FROM users -- WHERE id = 1")

    def test_non_select_rejected(self) -> None:
        """Test non-SELECT queries rejected."""
        with pytest.raises(ValidationError):
            validate_sql_query("SHOW TABLES")


class TestValidateJobId:
    """Tests for job ID validation."""

    def test_valid_uuid(self) -> None:
        """Test valid UUID format."""
        result = validate_job_id("550e8400-e29b-41d4-a716-446655440000")
        assert result == "550e8400-e29b-41d4-a716-446655440000"

    def test_uppercase_uuid(self) -> None:
        """Test uppercase UUID is normalized."""
        result = validate_job_id("550E8400-E29B-41D4-A716-446655440000")
        assert result == "550e8400-e29b-41d4-a716-446655440000"

    def test_invalid_format(self) -> None:
        """Test invalid job ID format."""
        with pytest.raises(ValidationError):
            validate_job_id("not-a-uuid")

    def test_empty_string(self) -> None:
        """Test empty string."""
        with pytest.raises(ValidationError):
            validate_job_id("")


class TestValidateAgentName:
    """Tests for agent name validation."""

    def test_valid_name(self) -> None:
        """Test valid agent name."""
        result = validate_agent_name("DataLoaderAgent")
        assert result == "DataLoaderAgent"

    def test_valid_with_underscore(self) -> None:
        """Test valid name with underscore."""
        result = validate_agent_name("Data_Loader_Agent")
        assert result == "Data_Loader_Agent"

    def test_starts_with_number(self) -> None:
        """Test name starting with number is rejected."""
        with pytest.raises(ValidationError):
            validate_agent_name("123Agent")

    def test_special_characters(self) -> None:
        """Test special characters rejected."""
        with pytest.raises(ValidationError):
            validate_agent_name("Agent-Name")

    def test_too_long(self) -> None:
        """Test name exceeding max length."""
        with pytest.raises(ValidationError):
            validate_agent_name("A" * 101)


class TestInputValidator:
    """Tests for InputValidator class."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        validator = InputValidator()
        assert validator.max_file_size_bytes == DEFAULT_MAX_FILE_SIZE_BYTES
        assert len(validator.allowed_extensions) > 0
        assert len(validator.allowed_content_types) > 0

    def test_custom_configuration(self) -> None:
        """Test custom configuration."""
        validator = InputValidator(
            max_file_size_bytes=50 * 1024 * 1024,  # 50MB
            allowed_extensions=[".csv"],
            allow_localhost=True,
        )
        assert validator.max_file_size_bytes == 50 * 1024 * 1024
        assert validator.allowed_extensions == [".csv"]
        assert validator.allow_localhost is True

    def test_validate_file(self, tmp_path: Path) -> None:
        """Test file validation."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b,c\n1,2,3")

        validator = InputValidator(base_dir=tmp_path, allow_absolute_paths=True)
        result = validator.validate_file(test_file)

        assert "path" in result
        assert result["size_bytes"] > 0
        assert result["extension"] == ".csv"

    def test_validate_url(self) -> None:
        """Test URL validation through validator."""
        validator = InputValidator()
        result = validator.validate_url("https://example.com/data.csv")
        assert result == "https://example.com/data.csv"

    def test_validate_content_type(self) -> None:
        """Test content type validation through validator."""
        validator = InputValidator()
        result = validator.validate_content_type("text/csv")
        assert result == "text/csv"
