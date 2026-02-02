"""
Tests for APO (Automatic Prompt Optimization) module.

Author: Malav Patel
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentds.core.apo import (
    APOOptimizer,
    BeamSearch,
    OptimizationResult,
    PromptCandidate,
    PromptHistory,
    RewardAggregator,
    create_apo_optimizer,
)
from agentds.core.config import APOSettings


class TestPromptCandidate:
    """Tests for PromptCandidate class."""

    def test_create_candidate(self) -> None:
        """Test creating a prompt candidate."""
        candidate = PromptCandidate(
            content="Test prompt",
            score=0.8,
            generation=1,
        )
        assert candidate.content == "Test prompt"
        assert candidate.score == 0.8
        assert candidate.generation == 1

    def test_candidate_id_generation(self) -> None:
        """Test that candidates get unique IDs."""
        c1 = PromptCandidate(content="Prompt A", generation=1)
        c2 = PromptCandidate(content="Prompt B", generation=1)

        assert c1.id != c2.id
        assert c1.id.startswith("prompt_1_")

    def test_candidate_with_metadata(self) -> None:
        """Test candidate with metadata."""
        candidate = PromptCandidate(
            content="Test",
            metadata={"agent": "DataLoader", "version": 1},
        )
        assert candidate.metadata["agent"] == "DataLoader"


class TestBeamSearch:
    """Tests for BeamSearch class."""

    def test_beam_initialization(self) -> None:
        """Test beam search initialization."""
        beam = BeamSearch(beam_width=3)
        assert beam.beam_width == 3
        assert len(beam.candidates) == 0

    def test_add_candidate(self) -> None:
        """Test adding candidates to beam."""
        beam = BeamSearch(beam_width=3)
        beam.add(PromptCandidate(content="A", score=0.5))
        beam.add(PromptCandidate(content="B", score=0.8))

        assert len(beam.candidates) == 2

    def test_beam_pruning(self) -> None:
        """Test that beam prunes to keep only top-k."""
        beam = BeamSearch(beam_width=2)

        beam.add(PromptCandidate(content="Low", score=0.3))
        beam.add(PromptCandidate(content="Mid", score=0.5))
        beam.add(PromptCandidate(content="High", score=0.9))

        assert len(beam.candidates) == 2
        assert beam.candidates[0].score == 0.9
        assert beam.candidates[1].score == 0.5

    def test_get_best(self) -> None:
        """Test getting best candidate."""
        beam = BeamSearch(beam_width=3)
        beam.add(PromptCandidate(content="A", score=0.3))
        beam.add(PromptCandidate(content="B", score=0.9))
        beam.add(PromptCandidate(content="C", score=0.5))

        best = beam.get_best()
        assert best is not None
        assert best.score == 0.9
        assert best.content == "B"

    def test_add_all(self) -> None:
        """Test adding multiple candidates at once."""
        beam = BeamSearch(beam_width=2)
        candidates = [
            PromptCandidate(content="A", score=0.1),
            PromptCandidate(content="B", score=0.5),
            PromptCandidate(content="C", score=0.9),
        ]
        beam.add_all(candidates)

        assert len(beam.candidates) == 2
        assert beam.get_best().score == 0.9


class TestRewardAggregator:
    """Tests for RewardAggregator class."""

    def test_default_weights(self) -> None:
        """Test aggregator with default weights."""
        aggregator = RewardAggregator()

        metrics = {
            "success_rate": 0.8,
            "user_satisfaction": 0.7,
            "efficiency": 0.9,
            "safety": 1.0,
        }

        score = aggregator.aggregate(metrics)
        # 0.8*0.4 + 0.7*0.3 + 0.9*0.2 + 1.0*0.1 = 0.32 + 0.21 + 0.18 + 0.1 = 0.81
        assert abs(score - 0.81) < 0.01

    def test_custom_weights(self) -> None:
        """Test aggregator with custom weights."""
        aggregator = RewardAggregator(weights={"success_rate": 1.0})

        metrics = {"success_rate": 0.95}
        score = aggregator.aggregate(metrics)

        assert abs(score - 0.95) < 0.01

    def test_missing_metrics(self) -> None:
        """Test aggregator with missing metrics."""
        aggregator = RewardAggregator()
        metrics = {"success_rate": 0.8}  # Only one metric

        score = aggregator.aggregate(metrics)
        assert 0.0 <= score <= 1.0

    def test_empty_metrics(self) -> None:
        """Test aggregator with no metrics."""
        aggregator = RewardAggregator()
        score = aggregator.aggregate({})

        assert score == 0.5  # Default neutral score


class TestPromptHistory:
    """Tests for PromptHistory class."""

    def test_add_and_get_latest(self) -> None:
        """Test adding and retrieving latest prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "history.json"
            history = PromptHistory(storage, max_versions=5)

            history.add(
                "TestAgent",
                PromptCandidate(content="v1", score=0.5, generation=1),
            )
            history.add(
                "TestAgent",
                PromptCandidate(content="v2", score=0.7, generation=2),
            )

            latest = history.get_latest("TestAgent")
            assert latest is not None
            assert latest.content == "v2"

    def test_get_best(self) -> None:
        """Test getting best performing prompt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "history.json"
            history = PromptHistory(storage, max_versions=5)

            history.add(
                "TestAgent",
                PromptCandidate(content="low", score=0.3, generation=1),
            )
            history.add(
                "TestAgent",
                PromptCandidate(content="high", score=0.9, generation=2),
            )
            history.add(
                "TestAgent",
                PromptCandidate(content="mid", score=0.5, generation=3),
            )

            best = history.get_best("TestAgent")
            assert best is not None
            assert best.content == "high"
            assert best.score == 0.9

    def test_rollback(self) -> None:
        """Test rolling back to previous version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "history.json"
            history = PromptHistory(storage, max_versions=5)

            for i in range(3):
                history.add(
                    "TestAgent",
                    PromptCandidate(content=f"v{i}", generation=i),
                )

            rollback = history.rollback("TestAgent", steps=1)
            assert rollback is not None
            assert rollback.content == "v1"

    def test_max_versions_pruning(self) -> None:
        """Test that old versions are pruned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "history.json"
            history = PromptHistory(storage, max_versions=3)

            for i in range(5):
                history.add(
                    "TestAgent",
                    PromptCandidate(content=f"v{i}", generation=i),
                )

            versions = history.get_all("TestAgent")
            assert len(versions) == 3
            assert versions[0].content == "v2"  # Oldest remaining

    def test_persistence(self) -> None:
        """Test that history persists to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir) / "history.json"

            # Create and add
            history1 = PromptHistory(storage, max_versions=5)
            history1.add(
                "TestAgent",
                PromptCandidate(content="persistent", score=0.8, generation=1),
            )

            # Load in new instance
            history2 = PromptHistory(storage, max_versions=5)
            latest = history2.get_latest("TestAgent")

            assert latest is not None
            assert latest.content == "persistent"


class TestAPOOptimizer:
    """Tests for APOOptimizer class."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create mock LLM provider."""
        llm = MagicMock()
        llm.complete.return_value = "Improved prompt based on critique"
        return llm

    def test_optimizer_initialization(self, mock_llm: MagicMock) -> None:
        """Test optimizer initialization."""
        optimizer = APOOptimizer(
            llm_provider=mock_llm,
            num_rounds=3,
            beam_width=2,
        )

        assert optimizer.num_rounds == 3
        assert optimizer.beam_width == 2

    def test_optimize_basic(self, mock_llm: MagicMock) -> None:
        """Test basic optimization run."""
        # Setup mock responses
        mock_llm.complete.side_effect = [
            "0.6",  # Initial eval
            "The prompt lacks specificity",  # Critique
            "Improved specific prompt",  # Rewrite
            "0.7",  # Eval improved
        ] * 5  # Multiple rounds

        optimizer = APOOptimizer(
            llm_provider=mock_llm,
            num_rounds=2,
            beam_width=1,
        )

        result = optimizer.optimize(
            initial_prompt="Original prompt",
            agent_name="TestAgent",
        )

        assert isinstance(result, OptimizationResult)
        assert result.original_prompt == "Original prompt"
        assert result.num_rounds == 2

    def test_optimization_result_to_dict(self, mock_llm: MagicMock) -> None:
        """Test converting optimization result to dict."""
        result = OptimizationResult(
            original_prompt="Original",
            original_score=0.5,
            best_prompt="Best",
            best_score=0.8,
            improvement=0.3,
            num_rounds=5,
            candidates_evaluated=10,
            history=[],
            duration_seconds=1.5,
        )

        d = result.to_dict()
        assert d["original_score"] == 0.5
        assert d["best_score"] == 0.8
        assert d["improvement"] == 0.3


class TestAPOSettings:
    """Tests for APOSettings class."""

    def test_default_settings(self) -> None:
        """Test default APO settings."""
        settings = APOSettings()

        assert settings.enabled is True
        assert settings.gradient_model == "gpt-4o"
        assert settings.edit_model == "gpt-4o-mini"
        assert settings.num_rounds == 5
        assert settings.beam_width == 3
        assert settings.temperature == 0.2

    def test_custom_settings(self) -> None:
        """Test custom APO settings."""
        settings = APOSettings(
            gradient_model="claude-3-5-sonnet",
            num_rounds=10,
            beam_width=5,
        )

        assert settings.gradient_model == "claude-3-5-sonnet"
        assert settings.num_rounds == 10
        assert settings.beam_width == 5

    def test_ab_testing_settings(self) -> None:
        """Test A/B testing settings."""
        settings = APOSettings(
            ab_testing_enabled=True,
            ab_test_sample_size=200,
            ab_test_confidence=0.99,
        )

        assert settings.ab_testing_enabled is True
        assert settings.ab_test_sample_size == 200
        assert settings.ab_test_confidence == 0.99


class TestCreateAPOOptimizer:
    """Tests for create_apo_optimizer factory function."""

    def test_disabled_apo(self) -> None:
        """Test that None is returned when APO is disabled."""
        settings = MagicMock()
        settings.apo = APOSettings(enabled=False)

        result = create_apo_optimizer(settings, None)
        assert result is None

    def test_no_llm_gateway(self) -> None:
        """Test handling of missing LLM gateway."""
        settings = MagicMock()
        settings.apo = APOSettings(enabled=True, use_agent_lightning=False)

        result = create_apo_optimizer(settings, None)
        assert result is None

    def test_create_native_optimizer(self) -> None:
        """Test creating native optimizer."""
        settings = MagicMock()
        settings.apo = APOSettings(
            enabled=True,
            use_agent_lightning=False,
            fallback_enabled=True,
        )

        llm = MagicMock()
        result = create_apo_optimizer(settings, llm)

        assert result is not None
        assert isinstance(result, APOOptimizer)

    @patch.dict("sys.modules", {"agentlightning": None})
    def test_fallback_when_agent_lightning_unavailable(self) -> None:
        """Test fallback to native when Agent Lightning not installed."""
        settings = MagicMock()
        settings.apo = APOSettings(
            enabled=True,
            use_agent_lightning=True,
            fallback_enabled=True,
        )

        llm = MagicMock()
        result = create_apo_optimizer(settings, llm)

        # Should fall back to native optimizer
        assert result is not None
