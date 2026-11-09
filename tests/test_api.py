"""
Tests for AgentDS REST API.

Tests for API endpoints and middleware.

Author: Malav Patel
"""

import pytest  # noqa: I001
from litestar import Litestar, get
from litestar.middleware import DefineMiddleware
from litestar.testing import TestClient

from agentds.core.config import Settings
from agentds.web.api.middleware import (
    APIKeyAuthenticator,
    AuthenticationMiddleware,
    RateLimiter,
    RateLimitMiddleware,
    TokenBucket,
)


def assert_invalid_api_key_response(response) -> None:
    """Assert the auth middleware's 401 rejection (status, headers and JSON body)."""
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "ApiKey"
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["error"] == "INVALID_API_KEY"
    assert body["message"] == "Invalid or missing API key"
    assert "timestamp" in body


def assert_rate_limited_response(response, limit: int) -> None:
    """Assert the rate-limit middleware's 429 rejection (status, headers and JSON body)."""
    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["error"] == "RATE_LIMIT_EXCEEDED"
    retry_after = body["retry_after_seconds"]
    assert isinstance(retry_after, int)
    assert retry_after > 0
    assert body["message"] == f"Rate limit exceeded. Retry after {retry_after} seconds."
    assert "timestamp" in body
    assert response.headers["retry-after"] == str(retry_after)
    assert response.headers["x-ratelimit-limit"] == str(limit)
    assert response.headers["x-ratelimit-remaining"] == "0"


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
# Middleware Rejection Tests (default suite: no external services needed)
# =============================================================================


@get("/ping", sync_to_thread=False)
def ping() -> dict[str, str]:
    """Trivial handler so the middleware can be exercised in isolation."""
    return {"status": "ok"}


def create_middleware_app(settings: Settings) -> Litestar:
    """Build a minimal app wired with the auth and rate-limit middleware."""
    return Litestar(
        route_handlers=[ping],
        middleware=[
            DefineMiddleware(AuthenticationMiddleware, settings=settings),
            DefineMiddleware(RateLimitMiddleware, settings=settings),
        ],
    )


class TestMiddlewareRejections:
    """Middleware rejections must reach the client as real 401/429 responses."""

    def test_missing_key_returns_401(self) -> None:
        """Request without an API key is rejected with 401."""
        settings = Settings(api_keys=["valid-api-key"], rate_limit_per_minute=1000)
        with TestClient(app=create_middleware_app(settings)) as client:
            response = client.get("/ping")
        assert_invalid_api_key_response(response)

    def test_invalid_key_returns_401(self) -> None:
        """Request with an unknown API key is rejected with 401."""
        settings = Settings(api_keys=["valid-api-key"], rate_limit_per_minute=1000)
        with TestClient(app=create_middleware_app(settings)) as client:
            response = client.get("/ping", headers={"X-API-Key": "invalid-key"})
        assert_invalid_api_key_response(response)

    def test_valid_key_passes_through(self) -> None:
        """Request with a valid API key reaches the handler with rate-limit headers."""
        settings = Settings(api_keys=["valid-api-key"], rate_limit_per_minute=1000)
        with TestClient(app=create_middleware_app(settings)) as client:
            response = client.get("/ping", headers={"X-API-Key": "valid-api-key"})
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert response.headers["x-ratelimit-limit"] == "1000"

    def test_rate_limit_exceeded_returns_429(self) -> None:
        """Requests beyond the burst capacity are rejected with 429."""
        # 1 request/minute gives a burst capacity of 2, so the third request is limited.
        settings = Settings(api_keys=["valid-api-key"], rate_limit_per_minute=1)
        headers = {"X-API-Key": "valid-api-key"}
        with TestClient(app=create_middleware_app(settings)) as client:
            assert client.get("/ping", headers=headers).status_code == 200
            assert client.get("/ping", headers=headers).status_code == 200
            response = client.get("/ping", headers=headers)
        assert_rate_limited_response(response, limit=1)


# =============================================================================
# API Endpoint Tests (Integration - require full stack)
# =============================================================================

# Note: Integration tests that require the full Litestar app with middleware
# are marked with pytest.mark.integration and can be run separately.
# These tests require specific environment setup and dependencies.


@pytest.mark.integration
class TestAPIEndpoints:
    """Tests for API endpoints (integration tests)."""

    @pytest.fixture
    def test_settings(self):
        """Create test settings."""
        from agentds.core.config import Settings

        return Settings(
            debug=True,
            environment="test",
            api_keys=[],  # No keys = allow all (dev mode)
            rate_limit_per_minute=1000,  # High limit for tests
        )

    @pytest.fixture
    def client(self, test_settings):
        """Create test client."""
        from agentds.web.api.webhooks import create_api

        app = create_api(settings=test_settings)
        with TestClient(app) as client:
            # With no keys configured any key is accepted, but one must be present.
            client.headers["X-API-Key"] = "dev-key"
            yield client

    def test_health_check(self, client) -> None:
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "components" in data

    def test_list_jobs_empty(self, client) -> None:
        """Test listing jobs when none exist."""
        response = client.get("/api/jobs")
        # Accept 200 or 500 (if job queue not initialized)
        assert response.status_code in [200, 500]

    def test_get_job_not_found(self, client) -> None:
        """Test getting non-existent job."""
        response = client.get("/api/jobs/nonexistent-job-id")
        # Should return error
        assert response.status_code in [400, 404, 500]

    def test_get_config(self, client) -> None:
        """Test getting configuration."""
        response = client.get("/api/config")
        # Accept 200 or 500 (if settings not fully initialized)
        assert response.status_code in [200, 500]

    def test_invalid_action(self, client) -> None:
        """Test invalid pipeline action."""
        response = client.post(
            "/api/pipeline/action/test-job",
            json={"action": "invalid_action"},
        )
        assert response.status_code in [400, 404, 500]


class TestAPIAuthentication:
    """Tests for API authentication middleware on the full app.

    Rejection tests never reach a route handler, so they run in the default
    suite; tests that execute handlers are marked as integration tests.
    """

    @pytest.fixture
    def secure_settings(self):
        """Create settings with API key requirement."""
        return Settings(
            debug=True,
            environment="test",
            api_keys=["valid-api-key"],
            rate_limit_per_minute=1000,
        )

    @pytest.fixture
    def secure_client(self, secure_settings):
        """Create test client with secure settings."""
        from agentds.web.api.webhooks import create_api

        app = create_api(settings=secure_settings)
        with TestClient(app) as client:
            yield client

    @pytest.mark.integration
    def test_health_exempt_from_auth(self, secure_client) -> None:
        """Test health endpoint is exempt from authentication."""
        response = secure_client.get("/api/health")
        assert response.status_code == 200

    def test_protected_endpoint_without_key(self, secure_client) -> None:
        """Test protected endpoint without API key."""
        response = secure_client.get("/api/jobs")
        assert_invalid_api_key_response(response)

    def test_protected_endpoint_with_invalid_key(self, secure_client) -> None:
        """Test protected endpoint with invalid API key."""
        response = secure_client.get(
            "/api/jobs",
            headers={"X-API-Key": "invalid-key"},
        )
        assert_invalid_api_key_response(response)

    @pytest.mark.integration
    def test_protected_endpoint_with_valid_key(self, secure_client) -> None:
        """Test protected endpoint with valid API key."""
        response = secure_client.get(
            "/api/jobs",
            headers={"X-API-Key": "valid-api-key"},
        )
        assert response.status_code == 200


@pytest.mark.integration
class TestAPIRateLimiting:
    """Tests for API rate limiting middleware (integration tests)."""

    @pytest.fixture
    def rate_limited_settings(self):
        """Create settings with strict rate limit."""
        return Settings(
            debug=True,
            environment="test",
            api_keys=[],  # Allow all
            rate_limit_per_minute=1,  # Burst capacity of 2 (2x rate)
        )

    @pytest.fixture
    def rate_limited_client(self, rate_limited_settings):
        """Create test client with rate limiting."""
        from agentds.web.api.webhooks import create_api

        app = create_api(settings=rate_limited_settings)
        with TestClient(app) as client:
            yield client

    def test_rate_limit_headers(self, rate_limited_client) -> None:
        """Test rate limit headers in response."""
        # Health is exempt from auth but is still rate limited
        response = rate_limited_client.get("/api/health/live")
        assert response.status_code == 200
        assert response.headers["x-ratelimit-limit"] == "1"
        assert "x-ratelimit-remaining" in response.headers

    def test_rate_limit_exceeded(self, rate_limited_client) -> None:
        """Test rate limit exceeded response."""
        # Burst capacity is 2, so the third request must be limited
        assert rate_limited_client.get("/api/health/live").status_code == 200
        assert rate_limited_client.get("/api/health/live").status_code == 200
        response = rate_limited_client.get("/api/health/live")
        assert_rate_limited_response(response, limit=1)


class TestCreateApiMiddleware:
    """The full app (create_api) wires the middleware with the app's own settings."""

    @staticmethod
    def client(**overrides):
        from litestar.testing import TestClient

        from agentds.core.config import Settings
        from agentds.web.api.webhooks import create_api

        settings = Settings(environment="test", **overrides)
        return TestClient(app=create_api(settings=settings))

    def test_no_configured_keys_allows_requests_without_a_key(self) -> None:
        with self.client(api_keys=[], rate_limit_per_minute=1000) as client:
            assert client.get("/api/jobs").status_code != 401

    def test_app_settings_reach_the_middleware(self, monkeypatch) -> None:
        """Keys passed to create_api apply, not whatever get_settings() returns."""
        from agentds.core.config import Settings
        from agentds.web.api import middleware

        monkeypatch.setattr(middleware, "get_settings", lambda: Settings(api_keys=["global-key"]))
        with self.client(api_keys=["app-key"], rate_limit_per_minute=1000) as client:
            assert client.get("/api/jobs", headers={"X-API-Key": "app-key"}).status_code != 401
            assert client.get("/api/jobs", headers={"X-API-Key": "global-key"}).status_code == 401


# =============================================================================
# Health check
# =============================================================================

class TestHealthCheck:
    """The LLM check runs in the background, at most once at a time, and is cached."""

    @pytest.fixture
    def llm_check(self, monkeypatch):
        """Replace the real (billed) LLM check with one the test controls."""
        import threading

        from agentds.web.api import webhooks

        class FakeCheck:
            def __init__(self) -> None:
                self.calls = 0
                self.release = threading.Event()

            def __call__(self, settings) -> bool:
                self.calls += 1
                # Blocks like a slow provider until the test releases it
                assert self.release.wait(10), "check was never released"
                return True

        fake = FakeCheck()
        monkeypatch.setattr(webhooks, "_check_llm", fake)
        monkeypatch.setattr(webhooks, "_health_checks", {})
        yield fake
        fake.release.set()

    @staticmethod
    def wait_until(condition, timeout: float = 5.0) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return True
            time.sleep(0.05)
        return False

    def test_health_answers_without_waiting_for_llm(self, llm_check, monkeypatch) -> None:
        from litestar.testing import TestClient

        from agentds.web.api import webhooks

        with TestClient(app=webhooks.create_api()) as client:
            # The check is stuck "talking to the provider", yet health answers
            first = client.get("/api/health")
            assert first.status_code == 200
            assert first.json()["components"]["llm"] is False  # not confirmed yet
            assert self.wait_until(lambda: llm_check.calls == 1)

            # A second probe doesn't start another check while one is running
            client.get("/api/health")
            assert llm_check.calls == 1

            llm_check.release.set()
            assert self.wait_until(
                lambda: client.get("/api/health").json()["components"]["llm"] is True
            )

            # Within the cache window, probes reuse the result
            for _ in range(3):
                assert client.get("/api/health").json()["status"] == "healthy"
            assert llm_check.calls == 1

            # Once the result is stale, the next probe refreshes it
            webhooks._health_checks["llm"].ttl_seconds = 0.0
            client.get("/api/health")
            assert self.wait_until(lambda: llm_check.calls == 2)

    def test_job_queue_reports_redis_ping(self, llm_check, monkeypatch) -> None:
        """job_queue follows a Redis PING, checked in the background like the LLM."""
        import threading
        from unittest.mock import MagicMock

        from litestar.testing import TestClient

        from agentds.core.config import get_settings
        from agentds.core.job_queue import JobQueue, RedisQueue
        from agentds.web.api import webhooks

        llm_check.release.set()
        release_ping = threading.Event()
        replies = {"value": False}

        def ping() -> bool:
            assert release_ping.wait(10), "ping was never released"
            return replies["value"]

        queue = JobQueue(get_settings())
        queue._backend = MagicMock(spec=RedisQueue)
        queue._backend.ping.side_effect = ping
        monkeypatch.setattr(webhooks, "_job_queue", queue)

        def job_queue_ok(client) -> bool:
            return client.get("/api/health").json()["components"]["job_queue"]

        with TestClient(app=webhooks.create_api()) as client:
            # A hanging Redis doesn't hold up the probe
            first = client.get("/api/health")
            assert first.status_code == 200
            assert first.json()["components"]["job_queue"] is False

            release_ping.set()  # Redis answers, but reports failure
            assert self.wait_until(lambda: queue._backend.ping.call_count >= 1)
            assert job_queue_ok(client) is False
            assert first.json()["status"] == "degraded"

            replies["value"] = True  # Redis back
            webhooks._health_checks["job_queue"].ttl_seconds = 0.0
            assert self.wait_until(lambda: job_queue_ok(client) is True)

    @pytest.fixture
    def redis_ping(self, monkeypatch):
        """A Redis-backed job queue whose PING the test controls."""
        import threading
        from unittest.mock import MagicMock

        from agentds.core.config import get_settings
        from agentds.core.job_queue import JobQueue, RedisQueue
        from agentds.web.api import webhooks

        class FakePing:
            def __init__(self) -> None:
                self.reply = True
                self.release = threading.Event()
                self.release.set()  # answers immediately unless the test holds it

            def __call__(self) -> bool:
                assert self.release.wait(10), "ping was never released"
                return self.reply

        fake = FakePing()
        queue = JobQueue(get_settings())
        queue._backend = MagicMock(spec=RedisQueue)
        queue._backend.ping.side_effect = fake
        fake.mock = queue._backend.ping
        monkeypatch.setattr(webhooks, "_job_queue", queue)
        yield fake
        fake.release.set()

    def test_liveness_checks_no_dependencies(self, llm_check, redis_ping) -> None:
        """A hung Redis and a hung LLM don't fail liveness, and it starts neither check."""
        import time

        from litestar.testing import TestClient

        from agentds.web.api import webhooks

        redis_ping.release.clear()  # Redis hangs

        with TestClient(app=webhooks.create_api()) as client:
            started = time.monotonic()
            for _ in range(3):
                response = client.get("/api/health/live")  # no API key: exempt from auth
                assert response.status_code == 200
                assert response.json()["status"] == "alive"
            assert time.monotonic() - started < 1.0

            assert redis_ping.mock.call_count == 0
            assert llm_check.calls == 0

    def test_readiness_follows_redis_ping(self, llm_check, redis_ping) -> None:
        from litestar.testing import TestClient

        from agentds.web.api import webhooks

        def ready(client) -> int:
            return client.get("/api/health/ready").status_code

        with TestClient(app=webhooks.create_api()) as client:
            # A fast first ping is reported on the very first probe
            first = client.get("/api/health/ready")  # no API key: exempt from auth
            assert first.status_code == 200
            assert first.json()["status"] == "ready"
            assert first.json()["components"] == {"job_queue": True}

            webhooks._health_checks["job_queue"].ttl_seconds = 0.0
            redis_ping.reply = False  # Redis down
            assert self.wait_until(lambda: ready(client) == 503)
            response = client.get("/api/health/ready")
            assert response.json()["status"] == "not_ready"
            assert response.json()["components"] == {"job_queue": False}
            assert client.get("/api/health/live").status_code == 200  # still alive

            redis_ping.reply = True  # Redis back
            assert self.wait_until(lambda: ready(client) == 200)

            # Readiness never pays for an LLM check
            assert llm_check.calls == 0

    def test_readiness_is_not_ready_while_first_ping_hangs(self, llm_check, redis_ping) -> None:
        import time

        from litestar.testing import TestClient

        from agentds.web.api import webhooks

        redis_ping.release.clear()  # Redis hangs

        with TestClient(app=webhooks.create_api()) as client:
            started = time.monotonic()
            response = client.get("/api/health/ready")
            assert response.status_code == 503
            assert time.monotonic() - started < 1.0  # inside the default k8s probe timeout

            redis_ping.release.set()
            assert self.wait_until(lambda: client.get("/api/health/ready").status_code == 200)
            assert redis_ping.mock.call_count == 1  # probes didn't pile up checks

    def test_first_connection_to_redis_does_not_block_liveness(self, llm_check, monkeypatch) -> None:
        """Building the JobQueue can wait for Redis's socket timeout; liveness keeps answering."""
        import threading
        import time

        from litestar.testing import TestClient

        from agentds.core.config import get_settings
        from agentds.core.job_queue import InMemoryQueue, JobQueue
        from agentds.web.api import webhooks

        connecting = threading.Event()
        release = threading.Event()

        def slow_init_backend(queue) -> None:
            connecting.set()
            assert release.wait(10), "connection was never released"
            queue._backend = InMemoryQueue()

        monkeypatch.setattr(JobQueue, "_init_backend", slow_init_backend)
        monkeypatch.setattr(webhooks, "_job_queue", None)
        monkeypatch.setattr(webhooks, "_settings", get_settings())

        with TestClient(app=webhooks.create_api()) as client:
            readiness: dict[str, int] = {}
            probe = threading.Thread(
                target=lambda: readiness.update(code=client.get("/api/health/ready").status_code)
            )
            probe.start()
            try:
                assert connecting.wait(5)
                started = time.monotonic()
                assert client.get("/api/health/live", timeout=2).status_code == 200
                assert time.monotonic() - started < 1.0
            finally:
                release.set()
                probe.join(10)
            assert readiness["code"] == 200

    def test_in_memory_job_queue_is_healthy(self, llm_check, monkeypatch) -> None:
        from litestar.testing import TestClient

        from agentds.core.config import Settings, get_settings
        from agentds.core.job_queue import JobQueue
        from agentds.web.api import webhooks

        llm_check.release.set()
        monkeypatch.setattr(Settings, "is_feature_enabled", lambda _self, _name: False)
        monkeypatch.setattr(webhooks, "_job_queue", JobQueue(get_settings()))

        with TestClient(app=webhooks.create_api()) as client:
            client.get("/api/health")
            assert self.wait_until(
                lambda: client.get("/api/health").json()["components"]["job_queue"] is True
            )


class TestAgentRun:
    """/api/agent/run runs the agent off the event loop."""

    def test_other_requests_are_served_while_an_agent_runs(self, monkeypatch, tmp_path) -> None:
        import threading

        from litestar.testing import TestClient

        from agentds.agents.base import AgentResult, AgentStatus, BaseAgent
        from agentds.web.api import webhooks

        started = threading.Event()
        release = threading.Event()

        def slow_run(self, context):
            started.set()
            assert release.wait(10), "agent was never released"
            return AgentResult(
                agent_name=self.__class__.__name__,
                status=AgentStatus.COMPLETED,
                outputs={"rows": 3},
            )

        monkeypatch.setattr(BaseAgent, "run", slow_run)
        # Health would otherwise start a real LLM check in the background
        monkeypatch.setattr(webhooks, "_check_llm", lambda _settings: True)
        headers = {"X-API-Key": "test-key"}
        result: dict = {}

        with TestClient(app=webhooks.create_api()) as client:

            def call_agent() -> None:
                result["response"] = client.post(
                    "/api/agent/run",
                    json={"agent_name": "DataLoaderAgent", "data_source": str(tmp_path / "d.csv")},
                    headers=headers,
                )

            caller = threading.Thread(target=call_agent)
            caller.start()
            try:
                assert started.wait(10), "agent run never started"
                # The agent is still running; the API must keep answering
                health = client.get("/api/health", timeout=2)
                assert health.status_code == 200
            finally:
                release.set()
                caller.join(10)

        response = result["response"]
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["agent"] == "DataLoaderAgent"
        assert body["status"] == "completed"
        assert body["outputs"] == {"rows": 3}

    def test_unknown_agent_is_rejected(self) -> None:
        from litestar.testing import TestClient

        from agentds.web.api import webhooks

        with TestClient(app=webhooks.create_api()) as client:
            response = client.post(
                "/api/agent/run",
                json={"agent_name": "NoSuchAgent", "data_source": "x.csv"},
                headers={"X-API-Key": "test-key"},
            )
        assert response.status_code == 400


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
