# APO (Automatic Prompt Optimization) Guide

**Self-Improving Agent Prompts for AgentDS**

Author: Malav Patel

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Using Agent Lightning](#using-agent-lightning)
4. [Native Fallback Mode](#native-fallback-mode)
5. [A/B Testing](#ab-testing)
6. [Prompt History & Rollback](#prompt-history--rollback)
7. [Monitoring](#monitoring)

---

## Overview

APO (Automatic Prompt Optimization) is a methodology for self-improving agent prompts through an iterative **Evaluate-Critique-Rewrite** cycle.

### How APO Works

```
┌─────────────────────────────────────────────────────────────┐
│                    APO Optimization Cycle                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐               │
│   │EVALUATE │────▶│CRITIQUE │────▶│ REWRITE │               │
│   │         │     │         │     │         │               │
│   │Score 0-1│     │"Textual │     │Apply    │               │
│   │reward   │     │Gradient"│     │Changes  │               │
│   └────┬────┘     └─────────┘     └────┬────┘               │
│        │                               │                     │
│        └───────────◀───────────────────┘                     │
│                    Repeat N rounds                           │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              BEAM SEARCH                             │   │
│   │   Keep top-K candidates, prune low performers        │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Textual Gradient** | Natural language critique describing what's wrong |
| **Beam Search** | Maintains multiple prompt candidates, keeps best |
| **Reward Function** | Scores prompt quality (0-1) based on performance |
| **Prompt History** | Version control for prompts with rollback |

---

## Configuration

APO is configured via environment variables or the `Settings` class.

### Environment Variables

```bash
# Core APO settings
APO_ENABLED=true
APO_GRADIENT_MODEL=gpt-4o
APO_EDIT_MODEL=gpt-4o-mini
APO_NUM_ROUNDS=5
APO_BEAM_WIDTH=3
APO_TEMPERATURE=0.2

# Quality thresholds
APO_REWARD_THRESHOLD=0.7
APO_MIN_IMPROVEMENT=0.05

# Fallback behavior
APO_USE_AGENT_LIGHTNING=true
APO_FALLBACK_ENABLED=true

# History
APO_HISTORY_SIZE=10
APO_AUTO_ROLLBACK=true

# A/B Testing
APO_AB_TESTING_ENABLED=false
APO_AB_TEST_SAMPLE_SIZE=100
APO_AB_TEST_CONFIDENCE=0.95
```

### Python Configuration

```python
from agentds.core.config import get_settings

settings = get_settings()

# Access APO settings
print(settings.apo.gradient_model)  # gpt-4o
print(settings.apo.beam_width)      # 3
print(settings.apo.num_rounds)      # 5
```

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable APO optimization |
| `gradient_model` | `gpt-4o` | Model for generating critiques |
| `edit_model` | `gpt-4o-mini` | Model for rewriting prompts |
| `num_rounds` | `5` | Optimization iterations (1-20) |
| `beam_width` | `3` | Parallel candidates (1-10) |
| `temperature` | `0.2` | Creativity in rewrites (0-1) |
| `reward_threshold` | `0.7` | Min score to accept prompt |
| `min_improvement` | `0.05` | Min improvement to adopt |
| `history_size` | `10` | Versions to keep per agent |

---

## Using Agent Lightning

AgentDS integrates with [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning) for advanced APO capabilities.

### Installation

```bash
pip install agentlightning[apo]>=0.2.0
```

### How Integration Works

When Agent Lightning is installed, AgentDS uses it automatically:

```python
# AgentDS detects Agent Lightning and uses it
from agentds.core.apo import create_apo_optimizer

optimizer = create_apo_optimizer(settings, llm_gateway)
# Uses agentlightning.APO internally if available
```

### Agent Lightning Features

| Feature | Description |
|---------|-------------|
| Advanced beam search | More sophisticated candidate management |
| Training datasets | Use historical runs for optimization |
| Validation sets | Prevent prompt overfitting |
| Rich metrics | Detailed optimization analytics |

---

## Native Fallback Mode

If Agent Lightning is unavailable, AgentDS uses its native APO implementation.

### Enabling Fallback

```bash
APO_USE_AGENT_LIGHTNING=true   # Try Agent Lightning first
APO_FALLBACK_ENABLED=true      # Use native if unavailable
```

### Native Implementation

```python
from agentds.core.apo import APOOptimizer, create_apo_optimizer

# Create optimizer (uses native if Agent Lightning unavailable)
optimizer = create_apo_optimizer(settings, llm_gateway)

# Or create native directly
from agentds.core.apo import APOOptimizer

optimizer = APOOptimizer(
    llm_provider=llm_gateway,
    gradient_model="gpt-4o",
    edit_model="gpt-4o-mini",
    num_rounds=5,
    beam_width=3,
)

# Run optimization
result = optimizer.optimize(
    initial_prompt="Your agent prompt here...",
    agent_name="DataLoaderAgent",
    context={"user_feedback": "Too slow on large files"},
)

print(f"Original score: {result.original_score:.2f}")
print(f"Best score: {result.best_score:.2f}")
print(f"Improvement: {result.improvement:.2%}")
```

### Native Components

```python
from agentds.core.apo import (
    APOOptimizer,      # Main optimizer
    BeamSearch,        # Candidate management
    PromptHistory,     # Version control
    RewardAggregator,  # Combine metrics
    PromptCandidate,   # Prompt with score
)
```

---

## A/B Testing

Compare original and optimized prompts in production.

### Enable A/B Testing

```bash
APO_AB_TESTING_ENABLED=true
APO_AB_TEST_SAMPLE_SIZE=100
APO_AB_TEST_CONFIDENCE=0.95
```

### How A/B Testing Works

1. **Split traffic** between original and optimized prompt
2. **Collect metrics** from both variants
3. **Statistical analysis** determines winner
4. **Auto-promote** winning prompt if significant

### Monitoring A/B Tests

```python
# Check A/B test status (future API)
from agentds.core.apo import ABTestManager

ab_manager = ABTestManager(settings)
results = ab_manager.get_test_results("DataLoaderAgent")

print(f"Original: {results.control_score:.2f}")
print(f"Optimized: {results.treatment_score:.2f}")
print(f"P-value: {results.p_value:.4f}")
print(f"Significant: {results.is_significant}")
```

---

## Prompt History & Rollback

Every optimization creates a versioned prompt that can be rolled back.

### Viewing History

```python
from agentds.core.apo import PromptHistory
from pathlib import Path

history = PromptHistory(
    storage_path=Path("outputs/prompt_history.json"),
    max_versions=10,
)

# Get all versions for an agent
versions = history.get_all("DataLoaderAgent")
for v in versions:
    print(f"v{v.generation}: score={v.score:.2f}")

# Get best performing version
best = history.get_best("DataLoaderAgent")
print(f"Best prompt (score {best.score:.2f}):")
print(best.content)
```

### Rolling Back

```python
# Rollback to previous version
previous = history.rollback("DataLoaderAgent", steps=1)
if previous:
    print(f"Rolled back to v{previous.generation}")
```

### Auto-Rollback

When enabled, prompts automatically rollback if performance drops:

```bash
APO_AUTO_ROLLBACK=true
```

---

## Monitoring

### OptimizationAgent Outputs

The `OptimizationAgent` produces:

```json
{
  "agents_optimized": ["DataLoaderAgent", "AutoMLAgent"],
  "optimization_results": {
    "DataLoaderAgent": {
      "original_reward": 0.72,
      "improvement": 0.12,
      "critique": "...",
      "optimization_timestamp": "2026-02-02T09:30:00Z"
    }
  }
}
```

### Artifacts Generated

| Artifact | Description |
|----------|-------------|
| `optimization_report.json` | Full optimization details |
| `optimized_prompts.json` | Before/after prompts |
| `prompt_history.json` | Version history |

### Metrics to Track

- **Improvement rate**: % improvement per optimization cycle
- **Time to optimize**: Duration of optimization runs
- **Rollback frequency**: How often prompts are rolled back
- **A/B test win rate**: % of optimized prompts that win

---

## Best Practices

1. **Start conservative**: Use `num_rounds=3` initially
2. **Monitor closely**: Watch for prompt drift
3. **Use A/B testing**: Validate improvements in production
4. **Keep history**: Enable `auto_rollback` for safety
5. **Review critiques**: Textual gradients reveal insights

---

*Author: Malav Patel | malav.patel203@gmail.com*
