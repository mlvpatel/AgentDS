"""
AgentDS EDACopilotAgent.

Generates exploratory data analysis with AI-powered insights.

Author: Malav Patel
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class EDACopilotAgent(BaseAgent):
    """
    Agent for exploratory data analysis.

    Generates:
    - Statistical summaries
    - Distribution analysis
    - Correlation analysis
    - Missing value patterns
    - AI-powered insights
    - Interactive HTML dashboard
    """

    name = "EDACopilotAgent"
    description = "Generate exploratory data analysis with AI-powered insights"
    phase = "build"
    complexity = "MEDIUM"
    input_types = ["parquet", "dataframe"]
    output_types = ["html", "json"]

    def get_system_prompt(self) -> str:
        """Get system prompt for EDA."""
        return """You are EDACopilotAgent, a specialized agent for exploratory data analysis.

Your responsibilities:
1. Analyze data distributions and statistics
2. Identify patterns, correlations, and anomalies
3. Generate actionable insights for the data science task
4. Recommend features for modeling
5. Highlight potential issues or opportunities

When analyzing data, provide insights in JSON format:
{
    "summary": "Brief overview of the dataset",
    "key_findings": [
        {"finding": "description", "importance": "HIGH|MEDIUM|LOW", "recommendation": "action"}
    ],
    "feature_recommendations": [
        {"feature": "name", "reason": "why useful for modeling"}
    ],
    "potential_issues": [
        {"issue": "description", "impact": "description", "mitigation": "suggestion"}
    ],
    "target_analysis": {
        "recommended_target": "column_name",
        "task_type": "classification|regression",
        "reasoning": "why this target"
    }
}

Be specific and actionable in your insights. Focus on what matters for the ML task.
"""

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute exploratory data analysis.

        Args:
            context: Execution context with cleaned data

        Returns:
            AgentResult with EDA dashboard
        """
        result = AgentResult(
            agent_name=self.name,
            status=AgentStatus.RUNNING,
        )

        try:
            # Get input data from previous agent
            prev_result = context.previous_results.get("DataCleaningAgent")
            if not prev_result:
                # Try DataLoaderAgent if cleaning was skipped
                prev_result = context.previous_results.get("DataLoaderAgent")
            if not prev_result:
                raise ValueError("No input data from previous agent")

            data_path = prev_result.outputs.get("data_path")
            if not data_path:
                raise ValueError("No data_path in previous output")

            # Load data
            df = pl.read_parquet(data_path)

            # Generate statistical summary
            stats = self._generate_statistics(df)

            # Generate correlation analysis
            correlations = self._generate_correlations(df)

            # Get AI insights
            insights = self._generate_insights(df, stats, correlations, context)

            # Generate HTML dashboard
            html_content = self._generate_html_dashboard(df, stats, correlations, insights)

            # Save artifacts
            dashboard_artifact = self.save_artifact(
                job_id=context.job_id,
                name="eda_dashboard.html",
                data=html_content,
                artifact_type=ArtifactType.VISUALIZATION,
                description="Interactive EDA dashboard",
            )

            insights_artifact = self.save_artifact(
                job_id=context.job_id,
                name="eda_insights.json",
                data=json.dumps(insights, indent=2, default=str),
                artifact_type=ArtifactType.REPORT,
                description="AI-generated insights",
            )

            result.outputs = {
                "dashboard_path": Path(context.settings.output_dir) / context.job_id / self.name / "eda_dashboard.html",
                "dashboard_artifact_id": dashboard_artifact,
                "insights_artifact_id": insights_artifact,
                "statistics": stats,
                "insights": insights,
            }
            result.artifacts.extend([dashboard_artifact, insights_artifact])
            result.approval_message = self._format_approval_message(stats, insights)
            result.mark_completed()

        except Exception as e:
            logger.error("EDA failed", error=str(e), exc_info=True)
            result.mark_failed(str(e))

        return result

    def _generate_statistics(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary."""
        stats = {
            "shape": {"rows": df.height, "columns": df.width},
            "columns": {},
            "memory_mb": df.estimated_size() / 1024 / 1024,
        }

        for col in df.columns:
            col_data = df[col]
            col_stats: Dict[str, Any] = {
                "dtype": str(col_data.dtype),
                "null_count": col_data.null_count(),
                "null_percent": round(col_data.null_count() / df.height * 100, 2),
                "n_unique": col_data.n_unique(),
            }

            # Numeric statistics
            if col_data.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                col_stats.update({
                    "mean": float(col_data.mean()) if col_data.mean() is not None else None,
                    "std": float(col_data.std()) if col_data.std() is not None else None,
                    "min": float(col_data.min()) if col_data.min() is not None else None,
                    "max": float(col_data.max()) if col_data.max() is not None else None,
                    "median": float(col_data.median()) if col_data.median() is not None else None,
                    "q25": float(col_data.quantile(0.25)) if col_data.quantile(0.25) is not None else None,
                    "q75": float(col_data.quantile(0.75)) if col_data.quantile(0.75) is not None else None,
                })

            # Categorical statistics
            elif col_data.dtype in [pl.Utf8, pl.Categorical]:
                value_counts = col_data.value_counts().sort("count", descending=True)
                top_values = value_counts.head(5).to_dicts()
                col_stats["top_values"] = top_values

            stats["columns"][col] = col_stats

        return stats

    def _generate_correlations(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Generate correlation analysis for numeric columns."""
        numeric_cols = [
            col for col in df.columns
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]

        if len(numeric_cols) < 2:
            return {"message": "Not enough numeric columns for correlation analysis"}

        # Limit to first 20 numeric columns
        numeric_cols = numeric_cols[:20]

        correlations = {}
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                try:
                    corr = df.select(pl.corr(col1, col2)).item()
                    if corr is not None and abs(corr) > 0.3:  # Only store significant correlations
                        key = f"{col1}__{col2}"
                        correlations[key] = round(corr, 3)
                except Exception:
                    continue

        # Sort by absolute correlation
        sorted_corr = dict(
            sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return {
            "numeric_columns": numeric_cols,
            "significant_correlations": sorted_corr,
            "highest_correlation": list(sorted_corr.items())[0] if sorted_corr else None,
        }

    def _generate_insights(
        self,
        df: pl.DataFrame,
        stats: Dict[str, Any],
        correlations: Dict[str, Any],
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Generate AI-powered insights."""
        # Prepare summary for LLM
        summary = {
            "shape": stats["shape"],
            "column_summary": [
                {
                    "name": col,
                    "dtype": info["dtype"],
                    "null_percent": info["null_percent"],
                    "n_unique": info["n_unique"],
                }
                for col, info in list(stats["columns"].items())[:15]
            ],
            "correlations": correlations.get("significant_correlations", {}),
        }

        prompt = f"""Analyze this dataset and provide insights for machine learning.

Dataset Summary:
{json.dumps(summary, indent=2)}

Task Description: {context.task_description or 'General ML task'}

Provide comprehensive insights including:
1. Key findings about the data
2. Feature recommendations for modeling
3. Potential issues to address
4. Target variable analysis (if applicable)

Output as JSON.
"""

        response = self.call_llm(prompt)

        # Parse JSON from response
        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning("Failed to parse LLM insights", error=str(e))

        # Default insights
        return {
            "summary": f"Dataset with {stats['shape']['rows']} rows and {stats['shape']['columns']} columns",
            "key_findings": [],
            "feature_recommendations": [],
            "potential_issues": [],
        }

    def _generate_html_dashboard(
        self,
        df: pl.DataFrame,
        stats: Dict[str, Any],
        correlations: Dict[str, Any],
        insights: Dict[str, Any],
    ) -> str:
        """Generate HTML dashboard."""
        # Generate column cards
        column_cards = ""
        for col, info in list(stats["columns"].items())[:20]:
            dtype_badge = "numeric" if "mean" in info else "categorical"
            column_cards += f"""
            <div class="card">
                <h3>{col}</h3>
                <span class="badge badge-{dtype_badge}">{info['dtype']}</span>
                <p>Nulls: {info['null_percent']}% | Unique: {info['n_unique']}</p>
                {'<p>Mean: ' + str(round(info.get('mean', 0), 2)) + ' | Std: ' + str(round(info.get('std', 0), 2)) + '</p>' if 'mean' in info else ''}
            </div>
            """

        # Generate insights list
        findings_html = ""
        for finding in insights.get("key_findings", []):
            importance = finding.get("importance", "MEDIUM")
            findings_html += f"""
            <div class="insight insight-{importance.lower()}">
                <strong>[{importance}]</strong> {finding.get('finding', '')}
                <br><em>Recommendation: {finding.get('recommendation', 'N/A')}</em>
            </div>
            """

        # Generate correlations table
        corr_rows = ""
        for pair, corr in list(correlations.get("significant_correlations", {}).items())[:10]:
            cols = pair.split("__")
            corr_class = "positive" if corr > 0 else "negative"
            corr_rows += f"""
            <tr>
                <td>{cols[0]}</td>
                <td>{cols[1]}</td>
                <td class="corr-{corr_class}">{corr}</td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Dashboard - AgentDS</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        h1 {{ font-size: 2rem; margin-bottom: 10px; }}
        h2 {{ font-size: 1.5rem; margin: 20px 0 10px; color: #444; }}
        h3 {{ font-size: 1.1rem; margin-bottom: 10px; color: #555; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat-box {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
        .stat-value {{ font-size: 2rem; font-weight: bold; color: #667eea; }}
        .stat-label {{ color: #888; font-size: 0.9rem; }}
        .card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 10px; }}
        .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; }}
        .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; }}
        .badge-numeric {{ background: #e3f2fd; color: #1976d2; }}
        .badge-categorical {{ background: #f3e5f5; color: #7b1fa2; }}
        .insight {{ padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid; }}
        .insight-high {{ background: #ffebee; border-color: #f44336; }}
        .insight-medium {{ background: #fff3e0; border-color: #ff9800; }}
        .insight-low {{ background: #e8f5e9; border-color: #4caf50; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .corr-positive {{ color: #4caf50; font-weight: bold; }}
        .corr-negative {{ color: #f44336; font-weight: bold; }}
        .section {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        footer {{ text-align: center; padding: 20px; color: #888; font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Exploratory Data Analysis</h1>
            <p>Generated by AgentDS v2.0</p>
        </header>

        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{stats['shape']['rows']:,}</div>
                <div class="stat-label">Rows</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats['shape']['columns']}</div>
                <div class="stat-label">Columns</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats['memory_mb']:.1f} MB</div>
                <div class="stat-label">Memory</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len([c for c in stats['columns'].values() if 'mean' in c])}</div>
                <div class="stat-label">Numeric</div>
            </div>
        </div>

        <div class="section">
            <h2>AI Insights</h2>
            <p style="margin-bottom: 15px;">{insights.get('summary', 'No summary available')}</p>
            {findings_html if findings_html else '<p>No significant findings detected.</p>'}
        </div>

        <div class="section">
            <h2>Column Analysis</h2>
            <div class="cards-grid">
                {column_cards}
            </div>
        </div>

        <div class="section">
            <h2>Significant Correlations</h2>
            <table>
                <thead>
                    <tr><th>Column 1</th><th>Column 2</th><th>Correlation</th></tr>
                </thead>
                <tbody>
                    {corr_rows if corr_rows else '<tr><td colspan="3">No significant correlations found</td></tr>'}
                </tbody>
            </table>
        </div>

        <footer>
            <p>AgentDS v2.0 - Autonomous Data Science Pipeline</p>
            <p>Author: Malav Patel</p>
        </footer>
    </div>
</body>
</html>
"""
        return html

    def _format_approval_message(
        self, stats: Dict[str, Any], insights: Dict[str, Any]
    ) -> str:
        """Format approval message for human review."""
        findings_text = "\n".join(
            f"  - [{f.get('importance', 'MEDIUM')}] {f.get('finding', '')}"
            for f in insights.get("key_findings", [])[:5]
        ) or "  No significant findings"

        return f"""
Exploratory Data Analysis Complete
==================================

Dataset Overview:
- Rows: {stats['shape']['rows']:,}
- Columns: {stats['shape']['columns']}
- Memory: {stats['memory_mb']:.1f} MB

Summary:
{insights.get('summary', 'N/A')}

Key Findings:
{findings_text}

An interactive HTML dashboard has been generated.

Do you want to proceed to feature engineering?
"""
