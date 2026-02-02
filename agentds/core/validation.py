"""
AgentDS Input Validation.

Provides validation utilities for security and data integrity.

Author: Malav Patel
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agentds.core.exceptions import (
    FileSizeLimitError,
    InvalidContentTypeError,
    InvalidURLError,
    PathTraversalError,
    ValidationError,
)
from agentds.core.logger import get_logger

logger = get_logger(__name__)

# Default limits
DEFAULT_MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
DEFAULT_MAX_PATH_LENGTH = 4096

# Allowed content types for data files
ALLOWED_DATA_CONTENT_TYPES = [
    "text/csv",
    "text/plain",
    "application/json",
    "application/octet-stream",  # For Parquet, Pickle, etc.
    "application/x-parquet",
    "application/vnd.apache.parquet",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]

# Allowed file extensions for data files
ALLOWED_DATA_EXTENSIONS = [
    ".csv",
    ".parquet",
    ".json",
    ".jsonl",
    ".xlsx",
    ".xls",
    ".feather",
    ".arrow",
    ".pkl",
    ".pickle",
    ".txt",
    ".tsv",
]

# Allowed URL schemes
ALLOWED_URL_SCHEMES = ["http", "https", "s3", "gs", "az", "file"]

# Dangerous path patterns
DANGEROUS_PATH_PATTERNS = [
    r"\.\./",  # Parent directory traversal
    r"\.\.\\",  # Windows parent directory traversal
    r"^/etc/",  # Linux system files
    r"^/var/",  # Linux var files
    r"^/proc/",  # Linux proc filesystem
    r"^/sys/",  # Linux sys filesystem
    r"^/dev/",  # Linux device files
    r"^C:\\Windows",  # Windows system directory
    r"^C:\\System",  # Windows system
    r"\\\\",  # UNC paths
]


def sanitize_path(
    path: str | Path,
    base_dir: Path | None = None,
    allow_absolute: bool = False,
) -> Path:
    """
    Sanitize and validate a file path to prevent directory traversal attacks.

    Args:
        path: Input path to sanitize
        base_dir: Optional base directory to restrict paths to
        allow_absolute: Whether to allow absolute paths

    Returns:
        Sanitized Path object

    Raises:
        PathTraversalError: If path traversal is detected
        ValidationError: If path is otherwise invalid
    """
    path_str = str(path)

    # Check for dangerous patterns
    for pattern in DANGEROUS_PATH_PATTERNS:
        if re.search(pattern, path_str, re.IGNORECASE):
            logger.warning("Path traversal attempt detected", path=path_str[:100])
            raise PathTraversalError(path_str)

    # Convert to Path object
    path_obj = Path(path_str)

    # Resolve the path (this handles .. and .)
    try:
        resolved = path_obj.resolve()
    except (OSError, ValueError) as e:
        raise ValidationError(f"Invalid path: {e}") from e

    # Check length
    if len(str(resolved)) > DEFAULT_MAX_PATH_LENGTH:
        raise ValidationError(
            f"Path exceeds maximum length of {DEFAULT_MAX_PATH_LENGTH} characters"
        )

    # If base_dir is specified, ensure path is within it
    if base_dir:
        base_resolved = base_dir.resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            logger.warning(
                "Path outside base directory",
                path=str(resolved)[:100],
                base_dir=str(base_resolved),
            )
            raise PathTraversalError(path_str) from None

    # Check if absolute paths are allowed
    if not allow_absolute and path_obj.is_absolute():
        raise ValidationError("Absolute paths are not allowed")

    return resolved


def validate_file_size(
    path: Path | str,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> int:
    """
    Validate that a file does not exceed the size limit.

    Args:
        path: Path to the file
        max_size_bytes: Maximum allowed file size in bytes

    Returns:
        Actual file size in bytes

    Raises:
        FileSizeLimitError: If file exceeds size limit
        ValidationError: If file cannot be accessed
    """
    path = Path(path)

    if not path.exists():
        raise ValidationError(f"File does not exist: {path.name}")

    if not path.is_file():
        raise ValidationError(f"Path is not a file: {path.name}")

    try:
        size = path.stat().st_size
    except OSError as e:
        raise ValidationError(f"Cannot access file: {e}") from e

    if size > max_size_bytes:
        raise FileSizeLimitError(size, max_size_bytes)

    return size


def validate_content_type(
    content_type: str,
    allowed_types: list[str] | None = None,
) -> str:
    """
    Validate that content type is allowed.

    Args:
        content_type: MIME type to validate
        allowed_types: List of allowed MIME types (uses default if None)

    Returns:
        Validated content type

    Raises:
        InvalidContentTypeError: If content type is not allowed
    """
    allowed = allowed_types or ALLOWED_DATA_CONTENT_TYPES

    # Normalize content type (remove charset, etc.)
    base_type = content_type.split(";")[0].strip().lower()

    if base_type not in [t.lower() for t in allowed]:
        raise InvalidContentTypeError(base_type, allowed)

    return base_type


def validate_file_extension(
    path: Path | str,
    allowed_extensions: list[str] | None = None,
) -> str:
    """
    Validate that file extension is allowed.

    Args:
        path: File path to validate
        allowed_extensions: List of allowed extensions (uses default if None)

    Returns:
        File extension (lowercase)

    Raises:
        ValidationError: If extension is not allowed
    """
    path = Path(path)
    allowed = allowed_extensions or ALLOWED_DATA_EXTENSIONS

    extension = path.suffix.lower()

    if extension not in [ext.lower() for ext in allowed]:
        raise ValidationError(
            f"File extension '{extension}' is not allowed. "
            f"Allowed extensions: {allowed}"
        )

    return extension


def validate_url(
    url: str,
    allowed_schemes: list[str] | None = None,
    allow_localhost: bool = False,
) -> str:
    """
    Validate a URL for safety.

    Args:
        url: URL to validate
        allowed_schemes: List of allowed URL schemes
        allow_localhost: Whether to allow localhost URLs

    Returns:
        Validated URL

    Raises:
        InvalidURLError: If URL is invalid or not allowed
    """
    allowed = allowed_schemes or ALLOWED_URL_SCHEMES

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise InvalidURLError(f"Cannot parse URL: {e}") from e

    # Check scheme
    if not parsed.scheme:
        raise InvalidURLError("URL must include a scheme (e.g., https://)")

    if parsed.scheme.lower() not in [s.lower() for s in allowed]:
        raise InvalidURLError(
            f"URL scheme '{parsed.scheme}' is not allowed. "
            f"Allowed schemes: {allowed}"
        )

    # Check for localhost/private IPs unless allowed
    if not allow_localhost:
        hostname = parsed.hostname or ""
        if hostname in ["localhost", "127.0.0.1", "0.0.0.0", "::1"]:
            raise InvalidURLError("Localhost URLs are not allowed")

        # Check for private IP ranges
        if _is_private_ip(hostname):
            raise InvalidURLError("Private IP addresses are not allowed")

    # Check for empty host
    if parsed.scheme in ["http", "https"] and not parsed.netloc:
        raise InvalidURLError("URL must include a host")

    return url


def _is_private_ip(hostname: str) -> bool:
    """Check if hostname is a private IP address."""
    import socket

    try:
        # Try to resolve and check IP
        ip = socket.gethostbyname(hostname)
        parts = [int(p) for p in ip.split(".")]

        # 10.0.0.0/8
        if parts[0] == 10:
            return True
        # 172.16.0.0/12
        if parts[0] == 172 and 16 <= parts[1] <= 31:
            return True
        # 192.168.0.0/16
        if parts[0] == 192 and parts[1] == 168:
            return True
        # 169.254.0.0/16 (link-local)
        return bool(parts[0] == 169 and parts[1] == 254)
    except (socket.gaierror, ValueError, IndexError):
        # Can't resolve or parse - not a private IP
        return False


def validate_sql_query(query: str) -> str:
    """
    Validate SQL query for safety (basic checks).

    Args:
        query: SQL query to validate

    Returns:
        Validated query

    Raises:
        ValidationError: If query contains dangerous patterns

    Note:
        This is a basic validation. For production, use parameterized queries
        and proper SQL injection prevention at the database layer.
    """
    query_upper = query.upper()

    # Block dangerous statements
    dangerous_keywords = [
        "DROP ",
        "DELETE ",
        "TRUNCATE ",
        "ALTER ",
        "CREATE ",
        "INSERT ",
        "UPDATE ",
        "GRANT ",
        "REVOKE ",
        "EXEC ",
        "EXECUTE ",
        "XP_",
        "SP_",
        "--",
        ";--",
        "/*",
        "*/",
    ]

    for keyword in dangerous_keywords:
        if keyword in query_upper:
            raise ValidationError(
                f"SQL query contains potentially dangerous keyword: {keyword.strip()}"
            )

    # Only allow SELECT queries
    if not query_upper.strip().startswith("SELECT"):
        raise ValidationError("Only SELECT queries are allowed")

    return query


def validate_job_id(job_id: str) -> str:
    """
    Validate job ID format.

    Args:
        job_id: Job ID to validate

    Returns:
        Validated job ID

    Raises:
        ValidationError: If job ID is invalid
    """
    # UUID format: 8-4-4-4-12 hex chars
    uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"

    if not re.match(uuid_pattern, job_id.lower()):
        raise ValidationError("Invalid job ID format. Expected UUID format.")

    return job_id.lower()


def validate_agent_name(agent_name: str) -> str:
    """
    Validate agent name.

    Args:
        agent_name: Agent name to validate

    Returns:
        Validated agent name

    Raises:
        ValidationError: If agent name is invalid
    """
    # Agent names should be alphanumeric with underscores
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", agent_name):
        raise ValidationError(
            "Invalid agent name. Must start with a letter and contain only "
            "letters, numbers, and underscores."
        )

    if len(agent_name) > 100:
        raise ValidationError("Agent name exceeds maximum length of 100 characters")

    return agent_name


class InputValidator:
    """
    Configurable input validator.

    Provides a unified interface for validating various inputs with
    customizable limits and allowed values.
    """

    def __init__(
        self,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        allowed_extensions: list[str] | None = None,
        allowed_content_types: list[str] | None = None,
        allowed_url_schemes: list[str] | None = None,
        base_dir: Path | None = None,
        allow_absolute_paths: bool = False,
        allow_localhost: bool = False,
    ) -> None:
        """
        Initialize input validator.

        Args:
            max_file_size_bytes: Maximum allowed file size
            allowed_extensions: Allowed file extensions
            allowed_content_types: Allowed MIME types
            allowed_url_schemes: Allowed URL schemes
            base_dir: Base directory for path validation
            allow_absolute_paths: Whether to allow absolute paths
            allow_localhost: Whether to allow localhost URLs
        """
        self.max_file_size_bytes = max_file_size_bytes
        self.allowed_extensions = allowed_extensions or ALLOWED_DATA_EXTENSIONS
        self.allowed_content_types = allowed_content_types or ALLOWED_DATA_CONTENT_TYPES
        self.allowed_url_schemes = allowed_url_schemes or ALLOWED_URL_SCHEMES
        self.base_dir = base_dir
        self.allow_absolute_paths = allow_absolute_paths
        self.allow_localhost = allow_localhost

    def validate_path(self, path: str | Path) -> Path:
        """Validate and sanitize a file path."""
        return sanitize_path(
            path,
            base_dir=self.base_dir,
            allow_absolute=self.allow_absolute_paths,
        )

    def validate_file(self, path: str | Path) -> dict[str, Any]:
        """
        Validate a file (path, size, extension).

        Returns:
            Dictionary with validated file info
        """
        path = self.validate_path(path)
        size = validate_file_size(path, self.max_file_size_bytes)
        extension = validate_file_extension(path, self.allowed_extensions)

        return {
            "path": path,
            "size_bytes": size,
            "extension": extension,
        }

    def validate_url(self, url: str) -> str:
        """Validate a URL."""
        return validate_url(
            url,
            allowed_schemes=self.allowed_url_schemes,
            allow_localhost=self.allow_localhost,
        )

    def validate_content_type(self, content_type: str) -> str:
        """Validate content type."""
        return validate_content_type(content_type, self.allowed_content_types)
