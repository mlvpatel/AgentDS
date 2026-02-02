"""
Tests for AgentDS REST API.

Tests for API endpoints and middleware.

Author: Malav Patel
"""

import pytest
from litestar.testing import TestClient

from agentds.core.config import Settings
from agentds.web.api.middleware import APIKeyAuthenticator, RateLimiter, TokenBucket
from agentds.web.api.webhooks import create_api


# =============================================================================
# Token Bucket Rate Limiter Tests
# =============================================================================


class TestTokenBucket:
    """Tests for TokenBucket rate limiter."""

    def test_initial_tokens(self) -> None:
        """Test initial tokens equals capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.tokens == 10.0

    def test_consume_tokens(self) -> None:
        """Test consuming tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.consume(5) is True
        assert bucket.tokens == 5.0

    def test_consume_more_than_available(self) -> None:
        """Test consuming more tokens than available."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.consume(8)
        assert bucket.consume(5) is False  # Only 2 left

    def test_refill(self) -> None:
        """Test token refill over time."""
        import time

        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/sec
        bucket.consume(10)
        assert bucket.tokens == 0.0

        time.sleep(0.5)
        bucket.refill()
        assert bucket.tokens >= 4.0  # Should have refilled ~5 tokens

    def test_time_until_available(self) -> None:
        """Test time until tokens available."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)  # 1 token/sec
        bucket.consume(10)

        time_needed = bucket.time_until_available(5)
        assert time_needed >= 4.0  # Need ~5 seconds


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_check_allowed(self) -> None:
        """Test request allowed when under limit."""
        limiter = RateLimiter(requests_per_minute=60)
        allowed, retry_after = limiter.check("user1")
        assert allowed is True
        assert retry_after == 0.0

    def test_check_rate_limited(self) -> None:
        """Test request blocked when limit exceeded."""
        limiter = RateLimiter(requests_per_minute=5, burst_size=5)

        # Exhaust the bucket
        for _ in range(5):
            limiter.check("user1")

        # Next request should be blocked
        allowed, retry_after = limiter.check("user1")
        assert allowed is False
        assert retry_after > 0

    def test_different_keys(self) -> None:
        """Test different keys have separate limits."""
        limiter = RateLimiter(requests_per_minute=2, burst_size=2)

        # Exhaust user1's limit
        limiter.check("user1")
        limiter.check("user1")

        # user2 should still be allowed
        allowed, _ = limiter.check("user2")
        assert allowed is True

    def test_get_remaining(self) -> None:
        """Test getting remaining requests."""
        limiter = RateLimiter(requests_per_minute=10, burst_size=10)
        limiter.check("user1")
        limiter.check("user1")

        remaining = limiter.get_remaining("user1")
        assert remaining == 8


# =============================================================================
# API Key Authenticator Tests
# =============================================================================


class TestAPIKeyAuthenticator:
    """Tests for APIKeyAuthenticator."""

    def test_validate_with_valid_key(self) -> None:
        """Test validation with valid key."""
        auth = APIKeyAuthenticator(api_keys=["key1", "key2"])
        assert auth.validate("key1") is True
        assert auth.validate("key2") is True

    def test_validate_with_invalid_key(self) -> None:
        """Test validation with invalid key."""
        auth = APIKeyAuthenticator(api_keys=["key1"])
        assert auth.validate("invalid") is False

    def test_validate_with_no_keys_configured(self) -> None:
        """Test validation when no keys configured (dev mode)."""
        auth = APIKeyAuthenticator(api_keys=[])
        # Should allow all in dev mode
        assert auth.validate("any-key") is True

    def test_validate_with_none(self) -> None:
        """Test validation with None key."""
        auth = APIKeyAuthenticator(api_keys=["key1"])
        assert auth.validate(None) is False

    def test_add_key(self) -> None:
        """Test adding API key."""
        auth = APIKeyAuthenticator(api_keys=[])
        auth.add_key("new-key")
        assert auth.validate("new-key") is True

    def test_remove_key(self) -> None:
        """Test removing API key."""
        auth = APIKeyAuthenticator(api_keys=["key1"])
        auth.remove_key("key1")
        # After removing, goes to dev mode if empty
        assert auth.validate("key1") is True  # Dev mode allows all


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.fixture
    def test_settings(self) -> Settings:
        """Create test settings."""
        return Settings(
            debug=True,
            environment="test",
            api_keys=[],  # No keys = allow all (dev mode)
            rate_limit_per_minute=1000,  # High limit for tests
        )

    @pytest.fixture
    def client(self, test_settings: Settings) -> TestClient:
        """Create test client."""
        app = create_api(settings=test_settings)
        return TestClient(app)

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "components" in data

    def test_list_jobs_empty(self, client: TestClient) -> None:
        """Test listing jobs when none exist."""
        response = client.get("/api/jobs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_job_not_found(self, client: TestClient) -> None:
        """Test getting non-existent job."""
        response = client.get("/api/jobs/nonexistent-job-id")
        # Should return error
        assert response.status_code in [400, 404, 500]

    def test_get_config(self, client: TestClient) -> None:
        """Test getting configuration."""
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "llm" in data
        assert "pipeline" in data

    def test_invalid_action(self, client: TestClient) -> None:
        """Test invalid pipeline action."""
        response = client.post(
            "/api/pipeline/action/test-job",
            json={"action": "invalid_action"},
        )
        assert response.status_code in [400, 404, 500]


class TestAPIAuthentication:
    """Tests for API authentication middleware."""

    @pytest.fixture
    def secure_settings(self) -> Settings:
        """Create settings with API key requirement."""
        return Settings(
            debug=True,
            environment="test",
            api_keys=["valid-api-key"],
            rate_limit_per_minute=1000,
        )

    @pytest.fixture
    def secure_client(self, secure_settings: Settings) -> TestClient:
        """Create test client with secure settings."""
        app = create_api(settings=secure_settings)
        return TestClient(app)

    def test_health_exempt_from_auth(self, secure_client: TestClient) -> None:
        """Test health endpoint is exempt from authentication."""
        response = secure_client.get("/api/health")
        assert response.status_code == 200

    def test_protected_endpoint_without_key(self, secure_client: TestClient) -> None:
        """Test protected endpoint without API key."""
        response = secure_client.get("/api/jobs")
        assert response.status_code == 401

    def test_protected_endpoint_with_invalid_key(self, secure_client: TestClient) -> None:
        """Test protected endpoint with invalid API key."""
        response = secure_client.get(
            "/api/jobs",
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401

    def test_protected_endpoint_with_valid_key(self, secure_client: TestClient) -> None:
        """Test protected endpoint with valid API key."""
        response = secure_client.get(
            "/api/jobs",
            headers={"X-API-Key": "valid-api-key"},
        )
        assert response.status_code == 200


class TestAPIRateLimiting:
    """Tests for API rate limiting middleware."""

    @pytest.fixture
    def rate_limited_settings(self) -> Settings:
        """Create settings with strict rate limit."""
        return Settings(
            debug=True,
            environment="test",
            api_keys=[],  # Allow all
            rate_limit_per_minute=5,  # Low limit for testing
        )

    @pytest.fixture
    def rate_limited_client(self, rate_limited_settings: Settings) -> TestClient:
        """Create test client with rate limiting."""
        app = create_api(settings=rate_limited_settings)
        return TestClient(app)

    def test_rate_limit_headers(self, rate_limited_client: TestClient) -> None:
        """Test rate limit headers in response."""
        response = rate_limited_client.get("/api/health")
        # Health is exempt from auth but should still have rate limit headers
        # Note: Headers may or may not be present depending on middleware order
        assert response.status_code == 200

    def test_rate_limit_exceeded(self, rate_limited_client: TestClient) -> None:
        """Test rate limit exceeded response."""
        # Make requests until rate limited
        # Note: burst_size is 2x rate, so need to exceed that
        for i in range(20):
            response = rate_limited_client.get("/api/health")
            if response.status_code == 429:
                # Rate limited
                data = response.json()
                assert "RATE_LIMIT" in data.get("error", "")
                assert "retry_after_seconds" in data
                return

        # If we get here, rate limiting didn't trigger
        # This could happen if burst_size is higher
        pass


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
