"""
AgentDS Test Configuration.

Shared fixtures and configuration for pytest.

Author: Malav Patel
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def test_dir() -> Generator[Path, None, None]:
    """Create temporary test directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Create sample CSV file for testing."""
    csv_content = """id,feature1,feature2,feature3,target
1,0.5,a,10,1
2,0.3,b,20,0
3,0.8,a,15,1
4,0.2,c,25,0
5,0.9,b,30,1
6,0.1,a,12,0
7,0.7,c,18,1
8,0.4,b,22,0
9,0.6,a,28,1
10,0.35,c,16,0
"""
    csv_path = tmp_path / "test_data.csv"
    csv_path.write_text(csv_content)
    return csv_path


@pytest.fixture
def sample_parquet_path(tmp_path: Path, sample_csv_path: Path) -> Path:
    """Create sample Parquet file for testing."""
    import polars as pl

    df = pl.read_csv(sample_csv_path)
    parquet_path = tmp_path / "test_data.parquet"
    df.write_parquet(parquet_path)
    return parquet_path


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    from agentds.core.config import Settings

    return Settings(
        debug=True,
        environment="test",
        log_level="DEBUG",
        human_in_loop=False,
    )


@pytest.fixture
def mock_llm_response():
    """Create mock LLM response."""
    from agentds.core.llm_gateway import LLMResponse

    return LLMResponse(
        content="Test response content",
        model="test/model",
        provider="test",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        cost=0.001,
        latency_ms=100.0,
    )


@pytest.fixture
def mock_llm_gateway(mock_llm_response):
    """Create mock LLM gateway."""
    from agentds.core.llm_gateway import LLMGateway

    gateway = MagicMock(spec=LLMGateway)
    gateway.complete.return_value = mock_llm_response
    gateway.acomplete.return_value = mock_llm_response
    gateway.get_total_cost.return_value = 0.001
    gateway.validate_connection.return_value = True
    return gateway


@pytest.fixture
def mock_artifact_store(tmp_path: Path, mock_settings):
    """Create mock artifact store."""
    from agentds.core.artifact_store import ArtifactStore

    mock_settings.output_dir = tmp_path / "outputs"
    mock_settings.output_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactStore(mock_settings)


@pytest.fixture
def agent_context(mock_settings, mock_llm_gateway, mock_artifact_store):
    """Create test agent context."""
    from agentds.agents.base import AgentContext

    return AgentContext(
        job_id="test-job-001",
        settings=mock_settings,
        llm_gateway=mock_llm_gateway,
        artifact_store=mock_artifact_store,
        task_description="Test ML task for unit testing",
        extra={},
    )


# Markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "llm: marks tests that require LLM connection")
