"""
AgentDS DriftMonitorAgent.

Monitors model performance and detects data/concept drift.

Author: Malav Patel
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from agentds.agents.base import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from agentds.core.artifact_store import ArtifactType
from agentds.core.logger import get_logger

logger = get_logger(__name__)


class DriftMonitorAgent(BaseAgent):
    """
    Agent for monitoring model drift and data quality.

    Detects:
    - Data drift: Changes in input feature distributions
    - Prediction drift: Changes in model output distributions
    - Performance drift: Degradation in model metrics
    - Feature drift: Changes in individual feature statistics
    """

    name = "DriftMonitorAgent"
    description = "Monitor model performance and detect drift"
    phase = "learn"
    complexity = "LOW"
    input_types = ["reference_data", "current_data", "predictions"]
    output_types = ["drift_report", "alerts"]

    def get_system_prompt(self) -> str:
        """Get system prompt for drift monitoring."""
        return """You are DriftMonitorAgent, a specialized agent for monitoring ML models in production.

Your responsibilities:
1. Compare current data distribution against reference (training) data
2. Detect statistical drift in features
3. Monitor prediction distribution changes
4. Track model performance metrics over time
5. Generate alerts when drift exceeds thresholds

Drift detection methods:
- Kolmogorov-Smirnov test for continuous features
- Chi-squared test for categorical features
- Population Stability Index (PSI)
- Jensen-Shannon divergence

Alert thresholds:
- LOW: 0.1 < drift < 0.2
- MEDIUM: 0.2 < drift < 0.3
- HIGH: drift > 0.3

Output analysis in JSON format with:
{
    "overall_drift_score": float,
    "drift_detected": bool,
    "feature_drift": {...},
    "prediction_drift": {...},
    "recommendations": [...]
}
"""

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute drift monitoring.

        Args:
            context: Execution context with reference and current data

        Returns:
            AgentResult with drift report and alerts
        """
        result = AgentResult(
            agent_name=self.name,
            status=AgentStatus.RUNNING,
        )

        try:
            # Get data paths
            reference_path = context.extra.get("reference_data")
            current_path = context.extra.get("current_data")

            if not reference_path or not current_path:
                # Try to get from previous results
                data_result = context.previous_results.get("DataCleaningAgent")
                if data_result:
                    reference_path = data_result.outputs.get("data_path")

            if not reference_path:
                raise ValueError("Reference data not provided")

            # Load data
            reference_df = self._load_data(reference_path)
            current_df = self._load_data(current_path) if current_path else reference_df.sample(
                fraction=0.3, seed=42
            )  # Demo: use sample if no current data

            # Calculate drift metrics
            drift_results = self._calculate_drift(reference_df, current_df)

            # Analyze with LLM for insights
            insights = self._analyze_drift(drift_results, context)

            # Generate alerts
            alerts = self._generate_alerts(drift_results)

            # Create drift report
            report = self._create_drift_report(drift_results, insights, alerts)

            # Save artifacts
            report_path = Path(context.settings.temp_dir) / f"{context.job_id}_drift_report.json"
            report_path.write_text(json.dumps(report, indent=2))

            artifact_id = self.save_artifact(
                job_id=context.job_id,
                name="drift_report.json",
                data=json.dumps(report, indent=2),
                artifact_type=ArtifactType.REPORT,
                description="Model drift analysis report",
                metadata={
                    "drift_detected": drift_results["drift_detected"],
                    "overall_score": drift_results["overall_drift_score"],
                },
            )

            # Prepare result
            result.outputs = {
                "drift_detected": drift_results["drift_detected"],
                "overall_drift_score": drift_results["overall_drift_score"],
                "alerts": alerts,
                "report_path": report_path,
                "insights": insights,
            }
            result.artifacts.append(artifact_id)
            result.approval_message = self._format_approval_message(
                drift_results, alerts
            )
            result.mark_completed()

        except Exception as e:
            logger.error("Drift monitoring failed", error=str(e), exc_info=True)
            result.mark_failed(str(e))

        return result

    def _load_data(self, data_path: Any) -> pl.DataFrame:
        """Load data from path."""
        if isinstance(data_path, pl.DataFrame):
            return data_path

        path = Path(data_path)
        if path.suffix == ".parquet":
            return pl.read_parquet(path)
        elif path.suffix == ".csv":
            return pl.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _calculate_drift(
        self, reference_df: pl.DataFrame, current_df: pl.DataFrame
    ) -> dict[str, Any]:
        """Calculate drift metrics between reference and current data."""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_rows": reference_df.height,
            "current_rows": current_df.height,
            "feature_drift": {},
            "overall_drift_score": 0.0,
            "drift_detected": False,
        }

        drift_scores = []

        # Calculate drift for each common column
        common_cols = set(reference_df.columns) & set(current_df.columns)

        for col in common_cols:
            ref_col = reference_df[col]
            cur_col = current_df[col]

            # Skip if all nulls
            if ref_col.null_count() == ref_col.len() or cur_col.null_count() == cur_col.len():
                continue

            drift_score = self._calculate_column_drift(ref_col, cur_col)
            results["feature_drift"][col] = {
                "drift_score": drift_score,
                "dtype": str(ref_col.dtype),
                "alert_level": self._get_alert_level(drift_score),
            }
            drift_scores.append(drift_score)

        # Calculate overall drift score
        if drift_scores:
            results["overall_drift_score"] = sum(drift_scores) / len(drift_scores)
            results["drift_detected"] = results["overall_drift_score"] > 0.2

        return results

    def _calculate_column_drift(
        self, ref_col: pl.Series, cur_col: pl.Series
    ) -> float:
        """Calculate drift score for a single column."""
        # Use PSI (Population Stability Index) approximation
        try:
            if ref_col.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                return self._calculate_psi_numeric(ref_col, cur_col)
            else:
                return self._calculate_psi_categorical(ref_col, cur_col)
        except Exception as e:
            logger.warning(f"Drift calculation failed for column: {e}")
            return 0.0

    def _calculate_psi_numeric(
        self, ref_col: pl.Series, cur_col: pl.Series, n_bins: int = 10
    ) -> float:
        """Calculate PSI for numeric columns."""
        # Remove nulls
        ref_vals = ref_col.drop_nulls().to_numpy()
        cur_vals = cur_col.drop_nulls().to_numpy()

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            return 0.0

        import numpy as np

        # Create bins from reference data
        _, bin_edges = np.histogram(ref_vals, bins=n_bins)

        # Calculate proportions
        ref_counts, _ = np.histogram(ref_vals, bins=bin_edges)
        cur_counts, _ = np.histogram(cur_vals, bins=bin_edges)

        ref_props = ref_counts / len(ref_vals)
        cur_props = cur_counts / len(cur_vals)

        # Avoid division by zero
        ref_props = np.clip(ref_props, 0.0001, None)
        cur_props = np.clip(cur_props, 0.0001, None)

        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

        return float(min(psi, 1.0))  # Cap at 1.0

    def _calculate_psi_categorical(
        self, ref_col: pl.Series, cur_col: pl.Series
    ) -> float:
        """Calculate PSI for categorical columns."""
        # Get value counts
        ref_counts = ref_col.drop_nulls().value_counts()
        cur_counts = cur_col.drop_nulls().value_counts()

        # Convert to dictionaries
        ref_dict = dict(zip(
            ref_counts.get_column(ref_col.name).to_list(),
            ref_counts.get_column("count").to_list(), strict=False
        ))
        cur_dict = dict(zip(
            cur_counts.get_column(cur_col.name).to_list(),
            cur_counts.get_column("count").to_list(), strict=False
        ))

        # Get all categories
        all_cats = set(ref_dict.keys()) | set(cur_dict.keys())

        if not all_cats:
            return 0.0

        ref_total = sum(ref_dict.values())
        cur_total = sum(cur_dict.values())

        psi = 0.0
        for cat in all_cats:
            ref_prop = max(ref_dict.get(cat, 0) / ref_total, 0.0001)
            cur_prop = max(cur_dict.get(cat, 0) / cur_total, 0.0001)
            psi += (cur_prop - ref_prop) * (cur_prop / ref_prop)

        return float(min(abs(psi), 1.0))

    def _get_alert_level(self, drift_score: float) -> str:
        """Get alert level based on drift score."""
        if drift_score < 0.1:
            return "NONE"
        elif drift_score < 0.2:
            return "LOW"
        elif drift_score < 0.3:
            return "MEDIUM"
        else:
            return "HIGH"

    def _analyze_drift(
        self, drift_results: dict[str, Any], context: AgentContext
    ) -> str:
        """Use LLM to analyze drift and provide insights."""
        prompt = f"""Analyze this drift monitoring report and provide actionable insights:

Drift Results:
- Overall drift score: {drift_results['overall_drift_score']:.3f}
- Drift detected: {drift_results['drift_detected']}
- Reference rows: {drift_results['reference_rows']}
- Current rows: {drift_results['current_rows']}

Feature-level drift:
{json.dumps(drift_results['feature_drift'], indent=2)}

Provide:
1. Summary of drift findings
2. Most concerning features
3. Potential causes
4. Recommended actions
"""

        response = self.call_llm(prompt)
        return response.content

    def _generate_alerts(self, drift_results: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate alerts based on drift results."""
        alerts = []

        # Overall drift alert
        if drift_results["drift_detected"]:
            alerts.append({
                "level": "HIGH" if drift_results["overall_drift_score"] > 0.3 else "MEDIUM",
                "type": "overall_drift",
                "message": f"Overall data drift detected: {drift_results['overall_drift_score']:.3f}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # Feature-level alerts
        for feature, info in drift_results["feature_drift"].items():
            if info["alert_level"] in ["MEDIUM", "HIGH"]:
                alerts.append({
                    "level": info["alert_level"],
                    "type": "feature_drift",
                    "feature": feature,
                    "message": f"Feature '{feature}' drift: {info['drift_score']:.3f}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        return alerts

    def _create_drift_report(
        self,
        drift_results: dict[str, Any],
        insights: str,
        alerts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create comprehensive drift report."""
        return {
            "report_type": "drift_monitoring",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "overall_drift_score": drift_results["overall_drift_score"],
                "drift_detected": drift_results["drift_detected"],
                "reference_rows": drift_results["reference_rows"],
                "current_rows": drift_results["current_rows"],
                "features_monitored": len(drift_results["feature_drift"]),
                "features_with_drift": sum(
                    1 for f in drift_results["feature_drift"].values()
                    if f["alert_level"] != "NONE"
                ),
            },
            "feature_drift": drift_results["feature_drift"],
            "alerts": alerts,
            "insights": insights,
            "recommendations": [
                "Review high-drift features for data quality issues",
                "Consider retraining if drift persists",
                "Investigate data pipeline for potential issues",
                "Set up automated drift alerts",
            ],
        }

    def _format_approval_message(
        self, drift_results: dict[str, Any], alerts: list[dict[str, Any]]
    ) -> str:
        """Format approval message."""
        alert_summary = "\n".join(
            f"  [{a['level']}] {a['message']}"
            for a in alerts[:5]
        ) or "  No alerts"

        top_drift_features = sorted(
            drift_results["feature_drift"].items(),
            key=lambda x: x[1]["drift_score"],
            reverse=True,
        )[:5]

        feature_summary = "\n".join(
            f"  - {f[0]}: {f[1]['drift_score']:.3f} ({f[1]['alert_level']})"
            for f in top_drift_features
        )

        return f"""
Drift Monitoring Complete
=========================

Overall Drift Score: {drift_results['overall_drift_score']:.3f}
Drift Detected: {'YES' if drift_results['drift_detected'] else 'NO'}

Data Summary:
- Reference rows: {drift_results['reference_rows']:,}
- Current rows: {drift_results['current_rows']:,}

Top Drifting Features:
{feature_summary}

Alerts ({len(alerts)} total):
{alert_summary}

Recommended Actions:
1. Review features with HIGH drift
2. Check data pipeline for issues
3. Consider model retraining if drift persists

Do you want to proceed with optimization?
"""
