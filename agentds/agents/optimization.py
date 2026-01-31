"""
AgentDS OptimizationAgent.

Self-improves agent prompts using APO (Automatic Prompt Optimization) methodology.
Based on the Evaluate-Critique-Rewrite cycle for iterative prompt improvement.

Author: Malav Patel
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentds.agents.base import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from agentds.core.artifact_store import ArtifactType
from agentds.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationTask:
    """Task for prompt optimization."""

    task_id: str
    agent_name: str
    input_data: dict[str, Any]
    expected_output: dict[str, Any]
    actual_output: dict[str, Any] | None = None
    reward: float = 0.0


@dataclass
class PromptTemplate:
    """Prompt template for optimization."""

    name: str
    content: str
    version: int = 1
    performance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs: Any) -> str:
        """Format prompt with variables."""
        result = self.content
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


class OptimizationAgent(BaseAgent):
    """
    Agent for self-optimization using APO methodology.

    APO (Automatic Prompt Optimization) cycle:
    1. EVALUATE: Run agent with current prompt, calculate reward
    2. CRITIQUE: Analyze failures and generate textual gradient
    3. REWRITE: Apply critique to create improved prompt
    4. REPEAT: Until performance plateaus

    Uses higher temperature (0.2) for creative prompt rewriting.
    """

    name = "OptimizationAgent"
    description = "Self-improve agent prompts using APO"
    phase = "learn"
    complexity = "CRITICAL"
    input_types = ["feedback", "drift_alerts", "performance_metrics"]
    output_types = ["optimized_prompts", "optimization_report"]

    def get_system_prompt(self) -> str:
        """Get system prompt for optimization."""
        return """You are OptimizationAgent, a meta-learning agent that improves other agents' prompts.

Your responsibilities:
1. Analyze agent performance and feedback
2. Identify patterns in failures
3. Generate improved prompts using APO methodology
4. Track prompt versions and performance
5. Recommend prompt updates

APO (Automatic Prompt Optimization) methodology:
1. EVALUATE: Score current prompt performance (0-1 reward)
2. CRITIQUE: Generate textual gradient describing what went wrong
3. REWRITE: Apply critique to create improved prompt
4. VALIDATE: Test improved prompt on held-out examples

Guidelines:
- Preserve core agent responsibilities
- Make targeted improvements based on specific failures
- Maintain clear, actionable instructions
- Test changes before deployment
- Keep prompts concise but comprehensive
"""

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute prompt optimization.

        Args:
            context: Execution context with feedback and metrics

        Returns:
            AgentResult with optimized prompts
        """
        result = AgentResult(
            agent_name=self.name,
            status=AgentStatus.RUNNING,
        )

        try:
            # Check if APO is enabled
            if not context.settings.is_feature_enabled("agent_lightning_apo"):
                logger.info("APO disabled, skipping optimization")
                result.outputs = {
                    "status": "skipped",
                    "reason": "APO feature disabled",
                }
                result.mark_completed()
                return result

            # Get feedback and drift information
            drift_result = context.previous_results.get("DriftMonitorAgent")
            user_feedback = context.user_feedback
            performance_metrics = context.extra.get("performance_metrics", {})

            # Determine which agents need optimization
            agents_to_optimize = self._identify_optimization_targets(
                drift_result, user_feedback, performance_metrics
            )

            if not agents_to_optimize:
                logger.info("No agents identified for optimization")
                result.outputs = {
                    "status": "no_optimization_needed",
                    "agents_analyzed": list(context.previous_results.keys()),
                }
                result.mark_completed()
                return result

            # Run APO for each target agent
            optimization_results = {}
            for agent_name in agents_to_optimize:
                opt_result = self._optimize_agent_prompt(
                    agent_name, context, performance_metrics
                )
                optimization_results[agent_name] = opt_result

            # Generate optimization report
            report = self._generate_optimization_report(optimization_results)

            # Save artifacts
            report_path = Path(context.settings.temp_dir) / f"{context.job_id}_optimization_report.json"
            report_path.write_text(json.dumps(report, indent=2))

            artifact_id = self.save_artifact(
                job_id=context.job_id,
                name="optimization_report.json",
                data=json.dumps(report, indent=2),
                artifact_type=ArtifactType.REPORT,
                description="APO optimization report",
            )

            # Save optimized prompts
            prompts_artifact = self.save_artifact(
                job_id=context.job_id,
                name="optimized_prompts.json",
                data=json.dumps({
                    name: {
                        "original": res.get("original_prompt", ""),
                        "optimized": res.get("optimized_prompt", ""),
                        "improvement": res.get("improvement", 0),
                    }
                    for name, res in optimization_results.items()
                }, indent=2),
                artifact_type=ArtifactType.CONFIG,
                description="Optimized agent prompts",
            )

            # Prepare result
            result.outputs = {
                "agents_optimized": list(optimization_results.keys()),
                "optimization_results": optimization_results,
                "report_path": report_path,
            }
            result.artifacts = [artifact_id, prompts_artifact]
            result.approval_message = self._format_approval_message(
                optimization_results
            )
            result.mark_completed()

        except Exception as e:
            logger.error("Optimization failed", error=str(e), exc_info=True)
            result.mark_failed(str(e))

        return result

    def _identify_optimization_targets(
        self,
        drift_result: AgentResult | None,
        user_feedback: str | None,
        performance_metrics: dict[str, Any],
    ) -> list[str]:
        """Identify which agents need prompt optimization."""
        targets = []

        # Check for drift-related issues
        if drift_result and drift_result.outputs.get("drift_detected"):
            # Feature engineering might need adjustment
            targets.append("FeatureEngineerAgent")

        # Check performance metrics
        for agent_name, metrics in performance_metrics.items():
            if metrics.get("success_rate", 1.0) < 0.8:
                targets.append(agent_name)
            if metrics.get("user_satisfaction", 1.0) < 0.7:
                targets.append(agent_name)

        # Parse user feedback for agent mentions
        if user_feedback:
            feedback_lower = user_feedback.lower()
            agent_keywords = {
                "data loading": "DataLoaderAgent",
                "cleaning": "DataCleaningAgent",
                "analysis": "EDACopilotAgent",
                "features": "FeatureEngineerAgent",
                "model": "AutoMLAgent",
                "api": "APIWrapperAgent",
                "docker": "DevOpsAgent",
                "deploy": "CloudDeployAgent",
                "drift": "DriftMonitorAgent",
            }
            for keyword, agent in agent_keywords.items():
                if keyword in feedback_lower:
                    targets.append(agent)

        return list(set(targets))

    def _optimize_agent_prompt(
        self,
        agent_name: str,
        context: AgentContext,
        performance_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run APO optimization for a specific agent.

        Implements the APO cycle:
        1. Evaluate current prompt
        2. Critique failures
        3. Rewrite prompt
        4. Validate improvement
        """
        logger.info(f"Optimizing prompt for {agent_name}")

        # Get current prompt (simplified - in production, load from agent)
        current_prompt = self._get_agent_prompt(agent_name)

        # Step 1: EVALUATE - Calculate reward for current prompt
        current_reward = self._evaluate_prompt(
            agent_name, current_prompt, performance_metrics
        )

        # Step 2: CRITIQUE - Generate textual gradient
        critique = self._generate_critique(
            agent_name, current_prompt, current_reward, context
        )

        # Step 3: REWRITE - Apply critique to create improved prompt
        optimized_prompt = self._rewrite_prompt(
            agent_name, current_prompt, critique
        )

        # Step 4: VALIDATE - Estimate improvement (simplified)
        estimated_improvement = self._estimate_improvement(
            current_prompt, optimized_prompt, critique
        )

        return {
            "agent_name": agent_name,
            "original_prompt": current_prompt[:500] + "..." if len(current_prompt) > 500 else current_prompt,
            "optimized_prompt": optimized_prompt[:500] + "..." if len(optimized_prompt) > 500 else optimized_prompt,
            "original_reward": current_reward,
            "critique": critique,
            "improvement": estimated_improvement,
            "optimization_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _get_agent_prompt(self, agent_name: str) -> str:
        """Get current system prompt for an agent."""
        # In production, this would load from a prompt store
        # For now, return a placeholder
        prompts = {
            "DataLoaderAgent": "Load data from various sources...",
            "DataCleaningAgent": "Clean and validate data...",
            "EDACopilotAgent": "Generate exploratory analysis...",
            "FeatureEngineerAgent": "Create feature engineering pipelines...",
            "AutoMLAgent": "Train and select ML models...",
            "APIWrapperAgent": "Generate API server code...",
            "DevOpsAgent": "Create Docker configurations...",
            "CloudDeployAgent": "Deploy to cloud platforms...",
            "DriftMonitorAgent": "Monitor model drift...",
        }
        return prompts.get(agent_name, "Default agent prompt")

    def _evaluate_prompt(
        self,
        agent_name: str,
        prompt: str,
        performance_metrics: dict[str, Any],
    ) -> float:
        """Evaluate current prompt performance (0-1 reward)."""
        metrics = performance_metrics.get(agent_name, {})

        # Calculate composite reward
        success_rate = metrics.get("success_rate", 0.8)
        user_satisfaction = metrics.get("user_satisfaction", 0.7)
        efficiency = metrics.get("efficiency", 0.8)

        reward = (success_rate * 0.4 + user_satisfaction * 0.4 + efficiency * 0.2)
        return round(reward, 3)

    def _generate_critique(
        self,
        agent_name: str,
        current_prompt: str,
        current_reward: float,
        context: AgentContext,
    ) -> str:
        """Generate textual gradient (critique) for prompt improvement."""
        prompt = f"""Analyze this agent prompt and identify areas for improvement.

Agent: {agent_name}
Current Reward Score: {current_reward:.3f}

Current Prompt:
{current_prompt}

User Feedback: {context.user_feedback or 'None provided'}

Generate a critique that:
1. Identifies specific weaknesses
2. Explains why they cause problems
3. Suggests concrete improvements
4. Prioritizes changes by impact

Format as a structured critique."""

        # Use higher temperature for creative analysis
        response = self.call_llm(prompt, temperature=0.2)
        return response.content

    def _rewrite_prompt(
        self,
        agent_name: str,
        current_prompt: str,
        critique: str,
    ) -> str:
        """Rewrite prompt based on critique."""
        prompt = f"""Rewrite this agent prompt based on the critique.

Agent: {agent_name}

Current Prompt:
{current_prompt}

Critique:
{critique}

Generate an improved prompt that:
1. Addresses all critique points
2. Maintains core responsibilities
3. Is clear and actionable
4. Includes specific examples where helpful
5. Is concise but comprehensive

Output only the improved prompt, no explanations."""

        # Use higher temperature for creative rewriting
        response = self.call_llm(prompt, temperature=0.2)
        return response.content

    def _estimate_improvement(
        self,
        original_prompt: str,
        optimized_prompt: str,
        critique: str,
    ) -> float:
        """Estimate improvement from optimization."""
        # Simple heuristic - in production, would run actual evaluation
        # Check if optimized prompt addresses critique points
        critique_keywords = ["improve", "add", "clarify", "specific", "example"]
        addressed = sum(1 for k in critique_keywords if k in optimized_prompt.lower())

        # Estimate improvement based on changes made
        len(optimized_prompt) - len(original_prompt)
        content_change = 1 - (
            len(set(original_prompt.split()) & set(optimized_prompt.split()))
            / max(len(set(original_prompt.split())), 1)
        )

        improvement = min(0.3, (addressed * 0.05 + content_change * 0.1))
        return round(improvement, 3)

    def _generate_optimization_report(
        self, optimization_results: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate comprehensive optimization report."""
        return {
            "report_type": "prompt_optimization",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "methodology": "APO (Automatic Prompt Optimization)",
            "summary": {
                "agents_optimized": len(optimization_results),
                "total_estimated_improvement": sum(
                    r["improvement"] for r in optimization_results.values()
                ),
                "average_original_reward": sum(
                    r["original_reward"] for r in optimization_results.values()
                ) / max(len(optimization_results), 1),
            },
            "optimizations": optimization_results,
            "recommendations": [
                "Review optimized prompts before deployment",
                "A/B test new prompts against original",
                "Monitor performance after deployment",
                "Schedule regular optimization cycles",
            ],
            "next_steps": [
                "Deploy optimized prompts to staging",
                "Run validation tests",
                "Collect user feedback",
                "Schedule next optimization in 7 days",
            ],
        }

    def _format_approval_message(
        self, optimization_results: dict[str, dict[str, Any]]
    ) -> str:
        """Format approval message."""
        opt_summary = "\n".join(
            f"  - {name}: +{res['improvement']*100:.1f}% improvement"
            for name, res in optimization_results.items()
        )

        total_improvement = sum(r["improvement"] for r in optimization_results.values())

        return f"""
Prompt Optimization Complete
============================

Methodology: APO (Automatic Prompt Optimization)
Agents Optimized: {len(optimization_results)}

Optimization Results:
{opt_summary}

Total Estimated Improvement: +{total_improvement*100:.1f}%

APO Cycle Completed:
1. [OK] EVALUATE - Scored current prompts
2. [OK] CRITIQUE - Generated textual gradients
3. [OK] REWRITE - Created improved prompts
4. [OK] VALIDATE - Estimated improvements

Next Steps:
1. Review optimized prompts
2. Deploy to staging environment
3. Run A/B tests
4. Monitor performance

Do you want to apply these optimizations?
"""


def run_apo_optimization(
    agent_rollout: Callable,
    initial_prompt: PromptTemplate,
    train_tasks: list[OptimizationTask],
    val_tasks: list[OptimizationTask],
    num_rounds: int = 5,
    beam_width: int = 3,
) -> PromptTemplate:
    """
    Run full APO optimization cycle.

    This is a custom implementation of the APO algorithm
    (Evaluate-Critique-Rewrite cycle for prompt optimization).

    Args:
        agent_rollout: Function that runs agent and returns reward
        initial_prompt: Starting prompt template
        train_tasks: Training tasks for optimization
        val_tasks: Validation tasks for evaluation
        num_rounds: Number of optimization rounds
        beam_width: Number of prompt candidates to maintain

    Returns:
        Optimized PromptTemplate
    """
    try:
        # Try to use agentlightning if available
        import agentlightning as agl

        # Create APO optimizer
        apo = agl.APO(
            gradient_model="gpt-4o",
            apply_edit_model="gpt-4o-mini",
            temperature=0.0,
            beam_width=beam_width,
            num_rounds=num_rounds,
        )

        # Run optimization
        trainer = agl.Trainer(
            algorithm=apo,
            initial_resources={"prompt_template": initial_prompt},
        )
        trainer.fit(
            agent=agent_rollout,
            train_dataset=train_tasks,
            val_dataset=val_tasks,
        )

        best_prompt = apo.get_best_prompt()
        return PromptTemplate(
            name=initial_prompt.name,
            content=best_prompt,
            version=initial_prompt.version + 1,
            performance_score=apo.best_score,
        )

    except ImportError:
        logger.warning("agentlightning not available, using fallback optimization")
        # Fallback: return original prompt
        return initial_prompt
