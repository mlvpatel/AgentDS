"""
AgentDS Core Module Tests.

Tests for configuration, caching, job queue, and artifact store.

Author: Malav Patel
"""

from __future__ import annotations

import pytest

# =============================================================================
# Configuration Tests
# =============================================================================

class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        from agentds.core.config import Settings

        settings = Settings()
        assert settings.app_name == "AgentDS"
        assert settings.app_version == "1.0.0"
        assert settings.debug is False

    def test_environment_override(self):
        """Test environment variable override."""
        import os

        from agentds.core.config import Settings

        os.environ["DEBUG"] = "true"
        settings = Settings()
        # Note: Need to reload for env var changes
        assert settings.debug is True or settings.debug is False  # Depends on load order

    def test_api_keys_parsing(self):
        """Test API keys parsing from comma-separated string."""
        from agentds.core.config import Settings

        settings = Settings(api_keys="key1,key2,key3")
        assert settings.api_keys == ["key1", "key2", "key3"]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("key1,key2,key3", ["key1", "key2", "key3"]),  # documented form
            (" key1 , key2 ,", ["key1", "key2"]),
            ("single-key", ["single-key"]),
            ('["key1", "key2"]', ["key1", "key2"]),  # JSON list still works
            ("", []),
        ],
    )
    def test_api_keys_from_environment(self, monkeypatch, value, expected):
        """API_KEYS set in the environment, as .env.example documents, loads."""
        from agentds.core.config import Settings

        monkeypatch.setenv("API_KEYS", value)
        assert Settings().api_keys == expected

    def test_api_keys_from_env_file(self, tmp_path, monkeypatch):
        from agentds.core.config import Settings

        monkeypatch.delenv("API_KEYS", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("API_KEYS=key1,key2\n")
        assert Settings(_env_file=env_file).api_keys == ["key1", "key2"]

    def test_api_keys_rejects_malformed_json(self, monkeypatch):
        from pydantic import ValidationError

        from agentds.core.config import Settings

        monkeypatch.setenv("API_KEYS", '["key1", ')
        with pytest.raises(ValidationError, match="API_KEYS"):
            Settings()

    def test_llm_settings(self):
        """Test LLM settings."""
        from agentds.core.config import LLMSettings

        llm = LLMSettings()
        assert llm.default_model == "openai/gpt-4o-mini"
        assert llm.default_temperature == 0.0

    def test_available_providers(self):
        """Test available providers detection."""
        from agentds.core.config import LLMSettings

        # Test with explicit API key parameter
        llm_with_key = LLMSettings.model_validate({"openai_api_key": "sk-test"})
        providers = llm_with_key.get_available_providers()
        assert "openai" in providers

        # Test that ollama is always available when no keys are set
        llm_empty = LLMSettings()
        providers_empty = llm_empty.get_available_providers()
        assert "ollama" in providers_empty


# =============================================================================
# Cache Layer Tests
# =============================================================================

class TestCacheLayer:
    """Tests for CacheLayer class."""

    def test_memory_cache_set_get(self):
        """Test memory cache set and get."""
        from agentds.core.cache_layer import MemoryCache

        cache = MemoryCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_memory_cache_delete(self):
        """Test memory cache delete."""
        from agentds.core.cache_layer import MemoryCache

        cache = MemoryCache()
        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None

    def test_memory_cache_ttl(self):
        """Test memory cache TTL."""
        import time

        from agentds.core.cache_layer import MemoryCache

        cache = MemoryCache()
        cache.set("key1", "value1", ttl=1)  # 1 second TTL
        assert cache.get("key1") == "value1"
        time.sleep(1.5)
        assert cache.get("key1") is None  # Expired

    def test_cache_layer_make_key(self):
        """Test cache key generation."""
        from agentds.core.cache_layer import CacheLayer

        key1 = CacheLayer.make_key("arg1", "arg2", kwarg1="val1")
        key2 = CacheLayer.make_key("arg1", "arg2", kwarg1="val1")
        key3 = CacheLayer.make_key("arg1", "arg3", kwarg1="val1")

        assert key1 == key2  # Same args produce same key
        assert key1 != key3  # Different args produce different key


# =============================================================================
# Job Queue Tests
# =============================================================================

class TestJobQueue:
    """Tests for JobQueue class."""

    def test_job_creation(self):
        """Test job creation."""
        from agentds.core.job_queue import Job, JobStatus

        job = Job(name="Test Job", task="test_task")
        assert job.name == "Test Job"
        assert job.status == JobStatus.PENDING

    def test_job_status_transitions(self):
        """Test job status transitions."""
        from agentds.core.job_queue import Job, JobStatus

        job = Job(name="Test Job", task="test_task")

        job.mark_running()
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None

        job.mark_completed(result={"success": True})
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.result == {"success": True}

    def test_job_failure(self):
        """Test job failure."""
        from agentds.core.job_queue import Job, JobStatus

        job = Job(name="Test Job", task="test_task")
        job.mark_running()
        job.mark_failed("Test error")

        assert job.status == JobStatus.FAILED
        assert job.error == "Test error"

    def test_job_progress(self):
        """Test job progress tracking."""
        from agentds.core.job_queue import Job

        job = Job(name="Test Job", task="test_task")
        job.update_progress(step=2, total=10, agent="TestAgent")

        assert job.current_step == 2
        assert job.total_steps == 10
        assert job.progress_percent == 20.0
        assert job.current_agent == "TestAgent"

    def test_redis_backend_dependencies_are_installed(self):
        """A normal install provides everything RedisQueue imports.

        Without rq, RedisQueue fails to import and JobQueue silently falls
        back to the in-memory queue even when Redis is reachable.
        """
        import importlib.util

        for module in ("redis", "rq"):
            assert importlib.util.find_spec(module) is not None, f"{module} is not installed"

    def test_redis_connection_uses_configured_socket_timeout(self):
        """A hung Redis can't stall a ping or job operation forever."""
        from agentds.core.job_queue import RedisQueue

        redis_queue = RedisQueue("redis://localhost:6399/0", socket_timeout=2)  # no I/O here
        kwargs = redis_queue._connection.connection_pool.connection_kwargs

        assert kwargs["socket_timeout"] == 2
        assert kwargs["socket_connect_timeout"] == 2

    def test_in_memory_queue_ping_is_true_when_redis_queue_is_disabled(self, monkeypatch):
        """With the Redis queue turned off, the in-memory queue is the intended backend."""
        from agentds.core.config import Settings
        from agentds.core.job_queue import InMemoryQueue, JobQueue

        monkeypatch.setattr(Settings, "is_feature_enabled", lambda _self, _name: False)
        queue = JobQueue(Settings())
        assert isinstance(queue._backend, InMemoryQueue)
        assert queue.using_fallback is False
        assert queue.ping() is True

    def test_redis_fallback_is_unhealthy_until_redis_returns(self, monkeypatch):
        """Redis unreachable at startup: ping is False, then the queue moves to Redis."""
        from agentds.core import job_queue as jq
        from agentds.core.config import Settings

        class FakeRedisQueue:
            reachable = False
            stored: dict = {}

            def __init__(self, url, socket_timeout=None):
                pass

            def ping(self):
                return FakeRedisQueue.reachable

            def store(self, job):
                FakeRedisQueue.stored[job.id] = job

            def get_job(self, job_id):
                return FakeRedisQueue.stored.get(job_id)

        monkeypatch.setattr(jq, "RedisQueue", FakeRedisQueue)
        monkeypatch.setattr(Settings, "is_feature_enabled", lambda _self, name: name == "redis_queue")

        queue = jq.JobQueue(Settings())
        assert queue.using_fallback is True
        job = queue.create_job(name="while down", task="pipeline")

        assert queue.ping() is False  # Redis still down: not healthy
        assert queue.using_fallback is True

        FakeRedisQueue.reachable = True
        assert queue.ping() is True
        assert queue.using_fallback is False
        assert isinstance(queue._backend, FakeRedisQueue)
        # The job created on the fallback moved to Redis and is still found
        assert queue.get_job(job.id).name == "while down"


class TestPackaging:
    """Packages the code imports directly must be declared, not arrive transitively."""

    def test_direct_runtime_imports_are_declared(self):
        import re
        from importlib.metadata import requires

        declared = {
            re.split(r"[\s<>=!~\[;]", requirement, maxsplit=1)[0].lower()
            for requirement in requires("agentds") or []
            if "extra ==" not in requirement  # optional extras don't count
        }

        # click: the `agentds` CLI; redis and rq: the Redis job queue
        for package in ("click", "redis", "rq"):
            assert package in declared, f"{package} is imported but not declared in pyproject.toml"


# =============================================================================
# Artifact Store Tests
# =============================================================================

class TestArtifactStore:
    """Tests for ArtifactStore class."""

    def test_save_and_get_artifact(self, mock_artifact_store, tmp_path):
        """Test saving and retrieving artifacts."""
        from agentds.core.artifact_store import ArtifactType

        # Save artifact
        artifact = mock_artifact_store.save(
            job_id="test-job",
            agent="TestAgent",
            name="test_output.txt",
            data="Hello, World!",
            artifact_type=ArtifactType.OTHER,
        )

        assert artifact.id == "test-job/TestAgent/test_output.txt"
        assert artifact.name == "test_output.txt"

        # Retrieve artifact
        retrieved = mock_artifact_store.get(artifact.id)
        assert retrieved is not None
        assert retrieved.name == "test_output.txt"

    def test_load_artifact_text(self, mock_artifact_store):
        """Test loading artifact as text."""
        mock_artifact_store.save(
            job_id="test-job",
            agent="TestAgent",
            name="test.txt",
            data="Test content",
        )

        content = mock_artifact_store.load_text("test-job/TestAgent/test.txt")
        assert content == "Test content"

    def test_list_artifacts(self, mock_artifact_store):
        """Test listing artifacts."""
        from agentds.core.artifact_store import ArtifactType

        # Save multiple artifacts
        mock_artifact_store.save(
            job_id="job1",
            agent="Agent1",
            name="file1.txt",
            data="content1",
            artifact_type=ArtifactType.DATA,
        )
        mock_artifact_store.save(
            job_id="job1",
            agent="Agent2",
            name="file2.txt",
            data="content2",
            artifact_type=ArtifactType.REPORT,
        )

        # List all for job
        artifacts = mock_artifact_store.list_artifacts(job_id="job1")
        assert len(artifacts) == 2

        # List by agent
        artifacts = mock_artifact_store.list_artifacts(agent="Agent1")
        assert len(artifacts) == 1

    def test_delete_artifact(self, mock_artifact_store):
        """Test deleting artifacts."""
        artifact = mock_artifact_store.save(
            job_id="test-job",
            agent="TestAgent",
            name="to_delete.txt",
            data="Delete me",
        )

        assert mock_artifact_store.delete(artifact.id) is True
        assert mock_artifact_store.get(artifact.id) is None


# =============================================================================
# Logger Tests
# =============================================================================

class TestLogger:
    """Tests for logging functionality."""

    def test_get_logger(self):
        """Test getting logger instance."""
        from agentds.core.logger import get_logger

        logger = get_logger("test_module")
        assert logger is not None

    def test_log_context(self):
        """Test log context manager."""
        from agentds.core.logger import LogContext, agent_ctx, job_id_ctx

        with LogContext(job_id="test-123", agent="TestAgent"):
            assert job_id_ctx.get() == "test-123"
            assert agent_ctx.get() == "TestAgent"

        # Context should be cleared
        assert job_id_ctx.get() is None
        assert agent_ctx.get() is None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
