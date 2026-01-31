"""
AgentDS DataCleaningAgent.

Cleans and validates data using AI-guided transformations.

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


class DataCleaningAgent(BaseAgent):
    """
    Agent for cleaning and validating data.

    Operations:
    - Remove duplicates
    - Handle missing values
    - Fix data types
    - Remove outliers
    - Validate schema
    - Data quality scoring
    """

    name = "DataCleaningAgent"
    description = "Clean and validate data using AI-guided transformations"
    phase = "build"
    complexity = "LOW"
    input_types = ["parquet", "dataframe"]
    output_types = ["parquet", "quality_report"]

    def get_system_prompt(self) -> str:
        """Get system prompt for data cleaning."""
        return """You are DataCleaningAgent, a specialized agent for data cleaning and validation.

Your responsibilities:
1. Identify data quality issues (nulls, duplicates, outliers, type mismatches)
2. Recommend cleaning operations with clear justifications
3. Generate cleaning code using Polars expressions
4. Validate the cleaned data meets quality thresholds

Output your cleaning plan in JSON format:
{
    "issues_found": [
        {"column": "col_name", "issue": "description", "severity": "HIGH|MEDIUM|LOW"}
    ],
    "cleaning_operations": [
        {
            "operation": "remove_duplicates|handle_missing|fix_types|remove_outliers",
            "target": "column_name or 'all'",
            "strategy": "specific strategy",
            "polars_expr": "Polars expression to apply"
        }
    ],
    "quality_score_before": 0.0,
    "expected_quality_score": 0.0
}

Be conservative: only remove or modify data when there is a clear quality issue.
Preserve as much original data as possible while ensuring quality.
"""

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute data cleaning.

        Args:
            context: Execution context with input data

        Returns:
            AgentResult with cleaned data
        """
        result = AgentResult(
            agent_name=self.name,
            status=AgentStatus.RUNNING,
        )

        try:
            # Get input data from previous agent
            prev_result = context.previous_results.get("DataLoaderAgent")
            if not prev_result:
                raise ValueError("No input data from DataLoaderAgent")

            data_path = prev_result.outputs.get("data_path")
            if not data_path:
                raise ValueError("No data_path in DataLoaderAgent output")

            # Load data
            df = pl.read_parquet(data_path)
            original_rows = df.height
            original_cols = df.width

            # Analyze data quality
            quality_before = self._calculate_quality_score(df)
            issues = self._identify_issues(df)

            # Get cleaning plan from LLM
            cleaning_plan = self._get_cleaning_plan(df, issues, context)

            # Apply cleaning operations
            df_cleaned, operations_applied = self._apply_cleaning(df, cleaning_plan)

            # Calculate final quality
            quality_after = self._calculate_quality_score(df_cleaned)

            # Generate quality report
            quality_report = {
                "original_rows": original_rows,
                "original_cols": original_cols,
                "cleaned_rows": df_cleaned.height,
                "cleaned_cols": df_cleaned.width,
                "rows_removed": original_rows - df_cleaned.height,
                "quality_before": quality_before,
                "quality_after": quality_after,
                "quality_improvement": quality_after - quality_before,
                "issues_found": issues,
                "operations_applied": operations_applied,
            }

            # Save cleaned data
            output_path = Path(context.settings.temp_dir) / f"{context.job_id}_clean_data.parquet"
            df_cleaned.write_parquet(output_path)

            data_artifact = self.save_artifact(
                job_id=context.job_id,
                name="clean_data.parquet",
                data=output_path,
                artifact_type=ArtifactType.DATA,
                metadata={
                    "rows": df_cleaned.height,
                    "columns": df_cleaned.width,
                    "quality_score": quality_after,
                },
                description="Cleaned and validated data",
            )

            # Save quality report
            report_artifact = self.save_artifact(
                job_id=context.job_id,
                name="quality_report.json",
                data=json.dumps(quality_report, indent=2, default=str),
                artifact_type=ArtifactType.REPORT,
                description="Data quality report",
            )

            result.outputs = {
                "data_path": output_path,
                "data_artifact_id": data_artifact,
                "report_artifact_id": report_artifact,
                "quality_report": quality_report,
                "dataframe": df_cleaned,
            }
            result.artifacts.extend([data_artifact, report_artifact])
            result.approval_message = self._format_approval_message(quality_report)
            result.mark_completed()

        except Exception as e:
            logger.error("Data cleaning failed", error=str(e), exc_info=True)
            result.mark_failed(str(e))

        return result

    def _calculate_quality_score(self, df: pl.DataFrame) -> float:
        """Calculate data quality score (0-1)."""
        if df.height == 0:
            return 0.0

        scores = []

        # Completeness: percentage of non-null values
        null_ratio = df.null_count().sum_horizontal()[0] / (df.height * df.width)
        completeness = 1 - null_ratio
        scores.append(completeness * 0.4)  # 40% weight

        # Uniqueness: percentage of unique rows
        unique_ratio = df.unique().height / df.height
        scores.append(unique_ratio * 0.3)  # 30% weight

        # Validity: basic type consistency (simplified)
        validity = 0.8  # Assume 80% valid by default
        scores.append(validity * 0.3)  # 30% weight

        return min(sum(scores), 1.0)

    def _identify_issues(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
        """Identify data quality issues."""
        issues = []

        for col in df.columns:
            col_data = df[col]

            # Check for high null percentage
            null_pct = col_data.null_count() / df.height
            if null_pct > 0.1:
                severity = "HIGH" if null_pct > 0.5 else "MEDIUM"
                issues.append({
                    "column": col,
                    "issue": f"High null percentage: {null_pct:.1%}",
                    "severity": severity,
                    "type": "missing_values",
                })

            # Check for potential duplicates in ID-like columns
            if "id" in col.lower() or col.lower().endswith("_id"):
                unique_ratio = col_data.n_unique() / df.height
                if unique_ratio < 0.99:
                    issues.append({
                        "column": col,
                        "issue": f"Potential duplicate IDs: {(1-unique_ratio):.1%} duplicates",
                        "severity": "HIGH",
                        "type": "duplicates",
                    })

            # Check for constant columns
            if col_data.n_unique() == 1:
                issues.append({
                    "column": col,
                    "issue": "Constant column (single value)",
                    "severity": "LOW",
                    "type": "constant",
                })

        # Check for duplicate rows
        dup_ratio = 1 - (df.unique().height / df.height)
        if dup_ratio > 0.01:
            issues.append({
                "column": "_all_",
                "issue": f"Duplicate rows: {dup_ratio:.1%}",
                "severity": "MEDIUM" if dup_ratio < 0.1 else "HIGH",
                "type": "duplicate_rows",
            })

        return issues

    def _get_cleaning_plan(
        self,
        df: pl.DataFrame,
        issues: List[Dict[str, Any]],
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Get cleaning plan from LLM."""
        # Prepare data summary for LLM
        summary = {
            "rows": df.height,
            "columns": df.width,
            "column_info": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "null_count": df[col].null_count(),
                    "n_unique": df[col].n_unique(),
                }
                for col in df.columns[:20]  # Limit to first 20 columns
            ],
            "issues": issues,
        }

        prompt = f"""Analyze this data and create a cleaning plan.

Data Summary:
{json.dumps(summary, indent=2)}

Task Description: {context.task_description or 'General data cleaning'}

User Feedback: {context.user_feedback or 'None'}

Provide a JSON cleaning plan with operations to address the identified issues.
Be conservative - only suggest operations that are clearly needed.
"""

        response = self.call_llm(prompt)

        # Parse JSON from response
        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning("Failed to parse LLM cleaning plan", error=str(e))

        # Default plan if LLM fails
        return {
            "issues_found": issues,
            "cleaning_operations": [
                {"operation": "remove_duplicates", "target": "all", "strategy": "keep_first"},
            ],
        }

    def _apply_cleaning(
        self, df: pl.DataFrame, plan: Dict[str, Any]
    ) -> tuple[pl.DataFrame, List[Dict[str, Any]]]:
        """Apply cleaning operations from plan."""
        operations_applied = []

        for op in plan.get("cleaning_operations", []):
            operation = op.get("operation", "")
            target = op.get("target", "all")

            try:
                if operation == "remove_duplicates":
                    before = df.height
                    df = df.unique()
                    operations_applied.append({
                        "operation": "remove_duplicates",
                        "rows_affected": before - df.height,
                    })

                elif operation == "handle_missing":
                    strategy = op.get("strategy", "drop")
                    if target == "all":
                        if strategy == "drop":
                            before = df.height
                            df = df.drop_nulls()
                            operations_applied.append({
                                "operation": "drop_null_rows",
                                "rows_affected": before - df.height,
                            })
                    else:
                        col = target
                        if col in df.columns:
                            if strategy == "drop":
                                before = df.height
                                df = df.drop_nulls(subset=[col])
                                operations_applied.append({
                                    "operation": f"drop_nulls_{col}",
                                    "rows_affected": before - df.height,
                                })
                            elif strategy == "mean":
                                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                                    mean_val = df[col].mean()
                                    df = df.with_columns(pl.col(col).fill_null(mean_val))
                                    operations_applied.append({
                                        "operation": f"fill_mean_{col}",
                                        "fill_value": mean_val,
                                    })
                            elif strategy == "mode":
                                mode_val = df[col].mode().first()
                                df = df.with_columns(pl.col(col).fill_null(mode_val))
                                operations_applied.append({
                                    "operation": f"fill_mode_{col}",
                                    "fill_value": str(mode_val),
                                })

                elif operation == "remove_outliers":
                    if target != "all" and target in df.columns:
                        col = target
                        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                            q1 = df[col].quantile(0.25)
                            q3 = df[col].quantile(0.75)
                            iqr = q3 - q1
                            lower = q1 - 1.5 * iqr
                            upper = q3 + 1.5 * iqr
                            before = df.height
                            df = df.filter(
                                (pl.col(col) >= lower) & (pl.col(col) <= upper)
                            )
                            operations_applied.append({
                                "operation": f"remove_outliers_{col}",
                                "rows_affected": before - df.height,
                                "bounds": [lower, upper],
                            })

            except Exception as e:
                logger.warning(
                    "Cleaning operation failed",
                    operation=operation,
                    error=str(e),
                )

        return df, operations_applied

    def _format_approval_message(self, report: Dict[str, Any]) -> str:
        """Format approval message for human review."""
        issues_text = "\n".join(
            f"  - [{i['severity']}] {i['column']}: {i['issue']}"
            for i in report.get("issues_found", [])[:10]
        ) or "  None detected"

        ops_text = "\n".join(
            f"  - {op['operation']}: {op.get('rows_affected', 'N/A')} rows affected"
            for op in report.get("operations_applied", [])
        ) or "  None applied"

        return f"""
Data Cleaning Complete
======================

Before:
- Rows: {report['original_rows']:,}
- Quality Score: {report['quality_before']:.2%}

After:
- Rows: {report['cleaned_rows']:,} ({report['rows_removed']:,} removed)
- Quality Score: {report['quality_after']:.2%}
- Improvement: {report['quality_improvement']:+.2%}

Issues Found:
{issues_text}

Operations Applied:
{ops_text}

Do you want to proceed with the cleaned data?
"""
