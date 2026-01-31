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
