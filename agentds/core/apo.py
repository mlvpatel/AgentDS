"""
AgentDS Native APO (Automatic Prompt Optimization) Implementation.

This module provides a native APO algorithm that works independently of
the Agent Lightning library. It implements the Evaluate-Critique-Rewrite
cycle for prompt optimization.

Author: Malav Patel
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from agentds.core.logger import get_logger

logger = get_logger(__name__)


# Type aliases
RewardFunction = Callable[[str, dict[str, Any]], float]
LLMFunction = Callable[[str, float], str]


class LLMProvider(Protocol):
    """Protocol for LLM provider used in APO."""

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """Generate completion from LLM."""
        ...


@dataclass
class PromptCandidate:
    """A prompt candidate with its performance metrics."""

    content: str
    score: float = 0.0
    generation: int = 0
    parent_id: str | None = None
    critique: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate unique ID for this candidate."""
        return f"prompt_{self.generation}_{hash(self.content) % 10000:04d}"


@dataclass
class OptimizationResult:
    """Result of an APO optimization run."""

    original_prompt: str
    original_score: float
    best_prompt: str
    best_score: float
    improvement: float
    num_rounds: int
    candidates_evaluated: int
    history: list[PromptCandidate]
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_prompt": self.original_prompt[:500] + "..."
            if len(self.original_prompt) > 500
            else self.original_prompt,
            "original_score": self.original_score,
            "best_prompt": self.best_prompt[:500] + "..."
            if len(self.best_prompt) > 500
            else self.best_prompt,
            "best_score": self.best_score,
            "improvement": self.improvement,
            "num_rounds": self.num_rounds,
            "candidates_evaluated": self.candidates_evaluated,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


class PromptHistory:
    """
    Manages prompt version history with rollback support.

    Stores prompt versions per agent with metadata for tracking
    performance over time.
    """

    def __init__(self, storage_path: Path, max_versions: int = 10) -> None:
        self.storage_path = storage_path
        self.max_versions = max_versions
        self._history: dict[str, list[PromptCandidate]] = {}
        self._load()

    def _load(self) -> None:
        """Load history from storage."""
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                for agent_name, versions in data.items():
                    self._history[agent_name] = [
                        PromptCandidate(
                            content=v["content"],
                            score=v["score"],
                            generation=v["generation"],
                            critique=v.get("critique", ""),
                        )
                        for v in versions
                    ]
            except Exception as e:
                logger.warning(f"Failed to load prompt history: {e}")
                self._history = {}

    def _save(self) -> None:
        """Save history to storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            agent_name: [
                {
                    "content": c.content,
                    "score": c.score,
                    "generation": c.generation,
                    "critique": c.critique,
                    "created_at": c.created_at.isoformat(),
                }
                for c in versions
            ]
            for agent_name, versions in self._history.items()
        }
        self.storage_path.write_text(json.dumps(data, indent=2))

    def add(self, agent_name: str, candidate: PromptCandidate) -> None:
        """Add a new prompt version."""
        if agent_name not in self._history:
            self._history[agent_name] = []

        self._history[agent_name].append(candidate)

        # Prune old versions
        if len(self._history[agent_name]) > self.max_versions:
            self._history[agent_name] = self._history[agent_name][-self.max_versions :]

        self._save()

    def get_latest(self, agent_name: str) -> PromptCandidate | None:
        """Get the latest prompt version."""
        versions = self._history.get(agent_name, [])
        return versions[-1] if versions else None

    def get_best(self, agent_name: str) -> PromptCandidate | None:
        """Get the best-performing prompt version."""
        versions = self._history.get(agent_name, [])
        if not versions:
            return None
        return max(versions, key=lambda c: c.score)

    def rollback(self, agent_name: str, steps: int = 1) -> PromptCandidate | None:
        """Rollback to a previous version."""
        versions = self._history.get(agent_name, [])
        if len(versions) <= steps:
            return None
        return versions[-(steps + 1)]

    def get_all(self, agent_name: str) -> list[PromptCandidate]:
        """Get all versions for an agent."""
        return self._history.get(agent_name, [])


class RewardAggregator:
    """
    Aggregates multiple reward signals into a single score.

    Supports weighted combination of different metrics like
    success rate, user satisfaction, and efficiency.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.weights = weights or {
            "success_rate": 0.4,
            "user_satisfaction": 0.3,
            "efficiency": 0.2,
            "safety": 0.1,
        }

    def aggregate(self, metrics: dict[str, float]) -> float:
        """Aggregate metrics into single reward score."""
        total_weight = 0.0
        weighted_sum = 0.0

        for metric_name, weight in self.weights.items():
            if metric_name in metrics:
                value = max(0.0, min(1.0, metrics[metric_name]))
                weighted_sum += value * weight
                total_weight += weight

        if total_weight == 0:
            return 0.5  # Default neutral score

        return weighted_sum / total_weight


class BeamSearch:
    """
    Beam search for prompt optimization.

    Maintains top-k candidates and prunes low performers
    after each optimization round.
    """

    def __init__(self, beam_width: int = 3) -> None:
        self.beam_width = beam_width
        self.candidates: list[PromptCandidate] = []

    def add(self, candidate: PromptCandidate) -> None:
        """Add a candidate to the beam."""
        self.candidates.append(candidate)
        self._prune()

    def add_all(self, candidates: list[PromptCandidate]) -> None:
        """Add multiple candidates."""
        self.candidates.extend(candidates)
        self._prune()

    def _prune(self) -> None:
        """Keep only top-k candidates by score."""
        self.candidates.sort(key=lambda c: c.score, reverse=True)
        self.candidates = self.candidates[: self.beam_width]

    def get_best(self) -> PromptCandidate | None:
        """Get the best candidate."""
        return self.candidates[0] if self.candidates else None

    def get_all(self) -> list[PromptCandidate]:
        """Get all candidates in beam."""
        return self.candidates.copy()


class APOOptimizer:
    """
    Native APO (Automatic Prompt Optimization) implementation.

    Implements the Evaluate-Critique-Rewrite cycle:
    1. EVALUATE: Score current prompt using reward function
    2. CRITIQUE: Generate textual gradient describing weaknesses
    3. REWRITE: Apply critique to create improved prompt
    4. VALIDATE: Ensure improvement meets threshold

    This is a fallback implementation when Agent Lightning is unavailable.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        reward_function: RewardFunction | None = None,
        gradient_model: str = "gpt-4o",
        edit_model: str = "gpt-4o-mini",
        num_rounds: int = 5,
        beam_width: int = 3,
        temperature: float = 0.2,
        min_improvement: float = 0.05,
    ) -> None:
        self.llm = llm_provider
        self.reward_function = reward_function
        self.gradient_model = gradient_model
        self.edit_model = edit_model
        self.num_rounds = num_rounds
        self.beam_width = beam_width
        self.temperature = temperature
        self.min_improvement = min_improvement

        self.beam = BeamSearch(beam_width)
        self.reward_aggregator = RewardAggregator()

    def optimize(
        self,
        initial_prompt: str,
        agent_name: str,
        context: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """
        Run full APO optimization cycle.

        Args:
            initial_prompt: Starting prompt to optimize
            agent_name: Name of agent being optimized
            context: Additional context (user feedback, metrics, etc.)

        Returns:
            OptimizationResult with best prompt and metrics
        """
        start_time = time.time()
        context = context or {}

        # Initialize with original prompt
        original = PromptCandidate(
            content=initial_prompt,
            score=self._evaluate(initial_prompt, context),
            generation=0,
        )
        self.beam.add(original)
        history = [original]

        logger.info(
            f"Starting APO for {agent_name}",
            initial_score=original.score,
            rounds=self.num_rounds,
        )

        # Run optimization rounds
        for round_num in range(1, self.num_rounds + 1):
            logger.debug(f"APO round {round_num}/{self.num_rounds}")

            new_candidates = []
            for parent in self.beam.get_all():
                # CRITIQUE: Generate textual gradient
                critique = self._critique(parent.content, parent.score, context)

                # REWRITE: Apply critique
                improved = self._rewrite(parent.content, critique, agent_name)

                # EVALUATE: Score new candidate
                score = self._evaluate(improved, context)

                candidate = PromptCandidate(
                    content=improved,
                    score=score,
                    generation=round_num,
                    parent_id=parent.id,
                    critique=critique,
                )
                new_candidates.append(candidate)
                history.append(candidate)

            # Add new candidates to beam (automatically prunes)
            self.beam.add_all(new_candidates)

            # Early stopping if no improvement
            best = self.beam.get_best()
            if best and best.score <= original.score + self.min_improvement:
                if round_num > 2:  # Give at least 2 rounds
                    logger.info(f"Early stopping at round {round_num} - no improvement")
                    break

        # Get final result
        best = self.beam.get_best()
        duration = time.time() - start_time

        if best is None:
            best = original

        improvement = best.score - original.score

        result = OptimizationResult(
            original_prompt=initial_prompt,
            original_score=original.score,
            best_prompt=best.content,
            best_score=best.score,
            improvement=improvement,
            num_rounds=self.num_rounds,
            candidates_evaluated=len(history),
            history=history,
            duration_seconds=duration,
            metadata={
                "agent_name": agent_name,
                "beam_width": self.beam_width,
                "temperature": self.temperature,
            },
        )

        logger.info(
            "APO optimization complete",
            original_score=original.score,
            best_score=best.score,
            improvement=improvement,
            duration=f"{duration:.2f}s",
        )

        return result

    def _evaluate(self, prompt: str, context: dict[str, Any]) -> float:
        """Evaluate prompt performance."""
        if self.reward_function:
            return self.reward_function(prompt, context)

        # Default evaluation using heuristics
        metrics = context.get("metrics", {})
        if metrics:
            return self.reward_aggregator.aggregate(metrics)

        # Fallback: Use LLM to estimate prompt quality
        eval_prompt = f"""Rate this agent prompt on a scale of 0.0 to 1.0 based on:
- Clarity and specificity
- Actionable instructions
- Appropriate scope
- Error handling guidance

Prompt to evaluate:
{prompt[:1000]}

Respond with ONLY a number between 0.0 and 1.0."""

        try:
            response = self.llm.complete(eval_prompt, temperature=0.0)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, AttributeError):
            return 0.5  # Neutral fallback

    def _critique(
        self,
        prompt: str,
        current_score: float,
        context: dict[str, Any],
    ) -> str:
        """Generate textual gradient (critique) for prompt."""
        user_feedback = context.get("user_feedback", "")
        failures = context.get("failures", [])

        critique_prompt = f"""Analyze this agent prompt and identify specific improvements.

Current Performance Score: {current_score:.2f}

Prompt:
{prompt}

{f"User Feedback: {user_feedback}" if user_feedback else ""}
{f"Recent Failures: {failures}" if failures else ""}

Generate a concise critique that:
1. Identifies 2-3 specific weaknesses
2. Explains why they cause problems
3. Suggests concrete improvements

Keep the critique focused and actionable."""

        return self.llm.complete(critique_prompt, temperature=0.1)

    def _rewrite(self, prompt: str, critique: str, agent_name: str) -> str:
        """Rewrite prompt based on critique."""
        rewrite_prompt = f"""Improve this agent prompt based on the critique.

Agent: {agent_name}

Original Prompt:
{prompt}

Critique:
{critique}

Generate an improved prompt that:
1. Addresses all critique points
2. Maintains core responsibilities
3. Is clear and actionable
4. Is concise but comprehensive

Output ONLY the improved prompt, no explanations."""

        return self.llm.complete(rewrite_prompt, temperature=self.temperature)


def create_apo_optimizer(
    settings: Any,
    llm_gateway: Any | None = None,
) -> APOOptimizer | None:
    """
    Create APO optimizer with appropriate backend.

    Args:
        settings: Application settings with APO configuration
        llm_gateway: LLM gateway for completions

    Returns:
        APOOptimizer instance or None if APO is disabled
    """
    apo_settings = settings.apo

    if not apo_settings.enabled:
        logger.info("APO is disabled")
        return None

    # Try Agent Lightning first if enabled
    if apo_settings.use_agent_lightning:
        try:
            import agentlightning as agl

            logger.info("Using Agent Lightning for APO")
            # Return Agent Lightning wrapper (to be implemented)
            # For now, fall through to native implementation
        except ImportError:
            if not apo_settings.fallback_enabled:
                logger.warning("Agent Lightning not available and fallback disabled")
                return None
            logger.info("Agent Lightning not available, using native APO")

    # Create native APO optimizer
    if llm_gateway is None:
        logger.warning("No LLM gateway provided for APO")
        return None

    return APOOptimizer(
        llm_provider=llm_gateway,
        gradient_model=apo_settings.gradient_model,
        edit_model=apo_settings.edit_model,
        num_rounds=apo_settings.num_rounds,
        beam_width=apo_settings.beam_width,
        temperature=apo_settings.temperature,
        min_improvement=apo_settings.min_improvement,
    )
