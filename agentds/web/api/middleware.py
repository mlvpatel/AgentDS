"""
AgentDS API Middleware.

Provides authentication and rate limiting middleware for the REST API.

Author: Malav Patel
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from litestar import Request, Response
from litestar.connection import ASGIConnection
from litestar.handlers import BaseRouteHandler
from litestar.middleware import AbstractMiddleware
from litestar.types import ASGIApp, Receive, Scope, Send

from agentds.core.config import Settings, get_settings
from agentds.core.exceptions import InvalidAPIKeyError
from agentds.core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Token Bucket Rate Limiter
# =============================================================================


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Initialize with full bucket."""
        self.tokens = float(self.capacity)

    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Returns:
            True if tokens were consumed, False if not enough tokens
        """
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until tokens are available."""
        self.refill()
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.refill_rate


class RateLimiter:
    """
    In-memory rate limiter using token bucket algorithm.

    For production, consider Redis-backed implementation for
    distributed rate limiting across multiple instances.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int | None = None,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size (defaults to 2x rate)
        """
        self.requests_per_minute = requests_per_minute
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        self.capacity = burst_size or (requests_per_minute * 2)
        self._buckets: dict[str, TokenBucket] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()

    def _get_bucket(self, key: str) -> TokenBucket:
        """Get or create token bucket for key."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                capacity=self.capacity,
                refill_rate=self.refill_rate,
            )
        return self._buckets[key]

    def _cleanup(self) -> None:
        """Remove old buckets to prevent memory leaks."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        # Remove buckets that haven't been used in 10 minutes
        stale_threshold = now - 600
        stale_keys = [
            key
            for key, bucket in self._buckets.items()
            if bucket.last_refill < stale_threshold
        ]
        for key in stale_keys:
            del self._buckets[key]

        self._last_cleanup = now
        if stale_keys:
            logger.debug("Cleaned up rate limit buckets", count=len(stale_keys))

    def check(self, key: str, tokens: int = 1) -> tuple[bool, float]:
        """
        Check if request is allowed.

        Args:
            key: Rate limit key (e.g., API key, IP address)
            tokens: Number of tokens to consume

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        self._cleanup()
        bucket = self._get_bucket(key)

        if bucket.consume(tokens):
            return True, 0.0
        else:
            retry_after = bucket.time_until_available(tokens)
            return False, retry_after

    def get_remaining(self, key: str) -> int:
        """Get remaining requests for key."""
        bucket = self._get_bucket(key)
        bucket.refill()
        return int(bucket.tokens)


# =============================================================================
# Authentication
# =============================================================================


class APIKeyAuthenticator:
    """API key authentication handler."""

    def __init__(
        self,
        api_keys: list[str] | None = None,
        header_name: str = "X-API-Key",
        query_param: str | None = "api_key",
    ) -> None:
        """
        Initialize API key authenticator.

        Args:
            api_keys: List of valid API keys
            header_name: Header name for API key
            query_param: Optional query parameter name for API key
        """
        self.api_keys = set(api_keys or [])
        self.header_name = header_name
        self.query_param = query_param

    def add_key(self, key: str) -> None:
        """Add an API key."""
        self.api_keys.add(key)

    def remove_key(self, key: str) -> None:
        """Remove an API key."""
        self.api_keys.discard(key)

    def extract_key(self, request: Request) -> str | None:
        """
        Extract API key from request.

        Checks header first, then query parameter.
        """
        # Check header
        key = request.headers.get(self.header_name)
        if key:
            return key

        # Check query parameter
        if self.query_param:
            key = request.query_params.get(self.query_param)
            if key:
                return key

        return None

    def validate(self, key: str | None) -> bool:
        """Validate an API key."""
        if not key:
            return False

        # If no keys configured, allow all (development mode)
        if not self.api_keys:
            logger.warning("No API keys configured - allowing all requests")
            return True

        return key in self.api_keys


# =============================================================================
# Litestar Middleware
# =============================================================================


class AuthenticationMiddleware(AbstractMiddleware):
    """
    Authentication middleware for Litestar.

    Validates API keys and adds authentication info to request state.
    """

    # Paths that don't require authentication
    EXEMPT_PATHS = {
        "/api/health",
        "/api/docs",
        "/api/openapi.json",
        "/api/redoc",
    }

    def __init__(
        self,
        app: ASGIApp,
        settings: Settings | None = None,
    ) -> None:
        """Initialize authentication middleware."""
        super().__init__(app)
        settings = settings or get_settings()
        self.authenticator = APIKeyAuthenticator(
            api_keys=settings.api_keys,
            header_name=settings.api_key_header,
        )

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Process request through middleware."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check if path is exempt
        path = scope.get("path", "")
        if path in self.EXEMPT_PATHS or path.startswith("/api/docs"):
            await self.app(scope, receive, send)
            return

        # Create request object for easier access
        request = Request(scope)

        # Extract and validate API key
        api_key = self.authenticator.extract_key(request)

        if not self.authenticator.validate(api_key):
            logger.warning(
                "Authentication failed",
                path=path,
                client_ip=request.client.host if request.client else "unknown",
            )

            response = Response(
                content={
                    "error": "INVALID_API_KEY",
                    "message": "Invalid or missing API key",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status_code=401,
                headers={"WWW-Authenticate": "ApiKey"},
            )
            await response(scope, receive, send)
            return

        # Add auth info to scope state
        scope.setdefault("state", {})
        scope["state"]["api_key"] = api_key
        scope["state"]["authenticated"] = True

        await self.app(scope, receive, send)


class RateLimitMiddleware(AbstractMiddleware):
    """
    Rate limiting middleware for Litestar.

    Uses token bucket algorithm for smooth rate limiting.
    """

    def __init__(
        self,
        app: ASGIApp,
        settings: Settings | None = None,
    ) -> None:
        """Initialize rate limit middleware."""
        super().__init__(app)
        settings = settings or get_settings()
        self.limiter = RateLimiter(
            requests_per_minute=settings.rate_limit_per_minute,
        )
        self.settings = settings

    def _get_rate_limit_key(self, scope: Scope) -> str:
        """
        Get rate limit key from request.

        Uses API key if authenticated, otherwise uses IP address.
        """
        state = scope.get("state", {})

        # Use API key if available
        api_key = state.get("api_key")
        if api_key:
            return f"apikey:{api_key}"

        # Fall back to IP address
        client = scope.get("client")
        if client:
            return f"ip:{client[0]}"

        return "anonymous"

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Process request through middleware."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Get rate limit key
        key = self._get_rate_limit_key(scope)

        # Check rate limit
        allowed, retry_after = self.limiter.check(key)

        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                key=key[:50],  # Truncate for logging
                retry_after=retry_after,
            )

            response = Response(
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Retry after {int(retry_after)} seconds.",
                    "retry_after_seconds": int(retry_after),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status_code=429,
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit": str(self.settings.rate_limit_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )
            await response(scope, receive, send)
            return

        # Add rate limit headers
        remaining = self.limiter.get_remaining(key)

        # Continue to next middleware/handler
        async def send_wrapper(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"X-RateLimit-Limit"] = str(
                    self.settings.rate_limit_per_minute
                ).encode()
                headers[b"X-RateLimit-Remaining"] = str(remaining).encode()
                message["headers"] = list(headers.items())
            await send(message)

        await self.app(scope, receive, send_wrapper)


# =============================================================================
# Guard Functions (for route-level auth)
# =============================================================================


def require_api_key(
    connection: ASGIConnection,
    _: BaseRouteHandler,
) -> None:
    """
    Litestar guard to require API key authentication.

    Usage:
        @get("/protected", guards=[require_api_key])
        async def protected_endpoint() -> dict:
            ...
    """
    state = connection.scope.get("state", {})
    if not state.get("authenticated"):
        raise InvalidAPIKeyError()


def require_admin_key(
    connection: ASGIConnection,
    _: BaseRouteHandler,
) -> None:
    """
    Litestar guard to require admin API key.

    Note: This is a placeholder. Implement admin key checking as needed.
    """
    state = connection.scope.get("state", {})
    api_key = state.get("api_key", "")

    # Check if this is an admin key (implement your logic)
    if not api_key.startswith("admin_"):
        raise InvalidAPIKeyError()
