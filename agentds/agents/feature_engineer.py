"""
AgentDS FeatureEngineerAgent.

Creates preprocessing pipelines and engineered features.

Author: Malav Patel
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from agentds.agents.base import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from agentds.core.artifact_store import ArtifactType
from agentds.core.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineerAgent(BaseAgent):
    """
    Agent for feature engineering and preprocessing.

    Operations:
    - Encoding categorical variables
    - Scaling numerical features
    - Imputation strategies
    - Feature selection
    - Feature creation
    - Pipeline generation
    """

    name = "FeatureEngineerAgent"
    description = "Create preprocessing pipelines and engineered features"
    phase = "build"
    complexity = "MEDIUM"
    input_types = ["parquet", "dataframe"]
    output_types = ["pkl", "parquet"]

    def get_system_prompt(self) -> str:
        """Get system prompt for feature engineering."""
        return """You are FeatureEngineerAgent, a specialized agent for feature engineering.

Your responsibilities:
1. Analyze features and recommend preprocessing strategies
2. Create encoding schemes for categorical variables
3. Determine scaling methods for numerical features
4. Suggest feature interactions or transformations
5. Identify target variable and task type

Output your feature engineering plan in JSON format:
{
    "target_column": "column_name",
    "task_type": "classification|regression",
    "numeric_features": {
        "columns": ["col1", "col2"],
        "scaling": "standard|minmax|none",
        "imputation": "mean|median|none"
    },
    "categorical_features": {
        "columns": ["col3", "col4"],
        "encoding": "onehot|label|target",
        "handle_unknown": "ignore|error"
    },
    "drop_columns": ["col5"],
    "feature_interactions": [
        {"type": "multiply", "columns": ["col1", "col2"], "name": "col1_x_col2"}
    ]
}

Be thoughtful about feature engineering - it significantly impacts model performance.
"""

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute feature engineering.

        Args:
            context: Execution context with cleaned data

        Returns:
            AgentResult with preprocessing pipeline
        """
        result = AgentResult(
            agent_name=self.name,
            status=AgentStatus.RUNNING,
        )

        try:
            # Get input data
            prev_result = context.previous_results.get("EDACopilotAgent")
            if not prev_result:
                prev_result = context.previous_results.get("DataCleaningAgent")
            if not prev_result:
                prev_result = context.previous_results.get("DataLoaderAgent")
            if not prev_result:
                raise ValueError("No input data from previous agent")

            data_path = prev_result.outputs.get("data_path")
            if not data_path:
                raise ValueError("No data_path in previous output")

            # Load data
            df = pl.read_parquet(data_path)

            # Get feature engineering plan from LLM
            fe_plan = self._get_feature_plan(df, context)

            # Identify target and features
            target_col = fe_plan.get("target_column")
            if not target_col or target_col not in df.columns:
                # Try to infer target
                target_col = self._infer_target(df, context)

            # Create preprocessing pipeline
            pipeline, feature_names = self._create_pipeline(df, fe_plan, target_col)

            # Apply pipeline to data
            X, y, processed_df = self._apply_pipeline(df, pipeline, target_col, feature_names)

            # Save artifacts
            output_path = Path(context.settings.temp_dir) / f"{context.job_id}_processed_data.parquet"
            processed_df.write_parquet(output_path)

            pipeline_path = Path(context.settings.temp_dir) / f"{context.job_id}_pipeline.pkl"
            with open(pipeline_path, "wb") as f:
                pickle.dump(pipeline, f)

            data_artifact = self.save_artifact(
                job_id=context.job_id,
                name="processed_data.parquet",
                data=output_path,
                artifact_type=ArtifactType.DATA,
                metadata={
                    "rows": processed_df.height,
                    "features": len(feature_names),
                    "target": target_col,
                },
                description="Preprocessed data ready for modeling",
            )

            pipeline_artifact = self.save_artifact(
                job_id=context.job_id,
                name="preprocessing_pipeline.pkl",
                data=pipeline_path,
                artifact_type=ArtifactType.MODEL,
                description="Sklearn preprocessing pipeline",
            )

            result.outputs = {
                "data_path": output_path,
                "pipeline_path": pipeline_path,
                "data_artifact_id": data_artifact,
                "pipeline_artifact_id": pipeline_artifact,
                "target_column": target_col,
                "feature_names": feature_names,
                "feature_plan": fe_plan,
                "task_type": fe_plan.get("task_type", "classification"),
            }
            result.artifacts.extend([data_artifact, pipeline_artifact])
            result.approval_message = self._format_approval_message(fe_plan, feature_names, target_col)
            result.mark_completed()

        except Exception as e:
            logger.error("Feature engineering failed", error=str(e), exc_info=True)
            result.mark_failed(str(e))

        return result

    def _get_feature_plan(
        self, df: pl.DataFrame, context: AgentContext
    ) -> dict[str, Any]:
        """Get feature engineering plan from LLM."""
        # Prepare column summary
        columns_info = []
        for col in df.columns[:30]:  # Limit columns
            info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": df[col].n_unique(),
                "null_pct": round(df[col].null_count() / df.height * 100, 1),
            }
            columns_info.append(info)

        prompt = f"""Create a feature engineering plan for this dataset.

Columns:
{json.dumps(columns_info, indent=2)}

Task Description: {context.task_description or 'Predict target variable'}

User Feedback: {context.user_feedback or 'None'}

Provide a comprehensive feature engineering plan as JSON.
"""

        response = self.call_llm(prompt)

        # Parse JSON
        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning("Failed to parse feature plan", error=str(e))

        # Default plan
        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Int64]]
        categorical_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]

        return {
            "target_column": df.columns[-1],
            "task_type": "classification",
            "numeric_features": {
                "columns": numeric_cols[:20],
                "scaling": "standard",
                "imputation": "mean",
            },
            "categorical_features": {
                "columns": categorical_cols[:10],
                "encoding": "onehot",
                "handle_unknown": "ignore",
            },
            "drop_columns": [],
        }

    def _infer_target(self, df: pl.DataFrame, context: AgentContext) -> str:
        """Infer target column from data and context."""
        # Common target column names
        target_names = ["target", "label", "y", "class", "outcome", "result"]

        for name in target_names:
            if name in df.columns:
                return name
            if name.lower() in [c.lower() for c in df.columns]:
                for col in df.columns:
                    if col.lower() == name.lower():
                        return col

        # Use last column as default
        return df.columns[-1]

    def _create_pipeline(
        self,
        df: pl.DataFrame,
        plan: dict[str, Any],
        target_col: str,
    ) -> tuple[Pipeline, list[str]]:
        """Create sklearn preprocessing pipeline."""
        transformers = []
        feature_names = []

        # Numeric features
        numeric_config = plan.get("numeric_features", {})
        numeric_cols = [c for c in numeric_config.get("columns", []) if c in df.columns and c != target_col]

        if numeric_cols:
            numeric_steps = []

            # Imputation
            impute_strategy = numeric_config.get("imputation", "mean")
            if impute_strategy != "none":
                numeric_steps.append(("imputer", SimpleImputer(strategy=impute_strategy)))

            # Scaling
            scaling = numeric_config.get("scaling", "standard")
            if scaling == "standard":
                numeric_steps.append(("scaler", StandardScaler()))
            elif scaling == "minmax":
                numeric_steps.append(("scaler", MinMaxScaler()))

            if numeric_steps:
                numeric_pipeline = Pipeline(numeric_steps)
                transformers.append(("numeric", numeric_pipeline, numeric_cols))
                feature_names.extend(numeric_cols)

        # Categorical features
        categorical_config = plan.get("categorical_features", {})
        categorical_cols = [c for c in categorical_config.get("columns", []) if c in df.columns and c != target_col]

        if categorical_cols:
            encoding = categorical_config.get("encoding", "onehot")
            handle_unknown = categorical_config.get("handle_unknown", "ignore")

            if encoding == "onehot":
                cat_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoder", OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)),
                ])
            else:
                cat_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ])

            transformers.append(("categorical", cat_pipeline, categorical_cols))
            # Note: actual feature names from onehot will be different
            feature_names.extend([f"cat_{c}" for c in categorical_cols])

        # Create column transformer
        if transformers:
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
            )
        else:
            preprocessor = "passthrough"

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
        ])

        return pipeline, feature_names

    def _apply_pipeline(
        self,
        df: pl.DataFrame,
        pipeline: Pipeline,
        target_col: str,
        feature_names: list[str],
    ) -> tuple[Any, Any, pl.DataFrame]:
        """Apply pipeline to data."""
        # Convert to pandas for sklearn
        pdf = df.to_pandas()

        # Separate features and target
        if target_col in pdf.columns:
            y = pdf[target_col].values
            X_df = pdf.drop(columns=[target_col])
        else:
            y = None
            X_df = pdf

        # Fit and transform
        X = pipeline.fit_transform(X_df)

        # Create processed dataframe
        if hasattr(X, 'toarray'):
            X = X.toarray()

        # Get actual feature names from pipeline
        try:
            if hasattr(pipeline.named_steps.get("preprocessor", None), "get_feature_names_out"):
                feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        processed_df = pl.DataFrame(X, schema=feature_names[:X.shape[1]])

        if y is not None:
            processed_df = processed_df.with_columns(pl.Series(target_col, y))

        return X, y, processed_df

    def _format_approval_message(
        self,
        plan: dict[str, Any],
        feature_names: list[str],
        target_col: str,
    ) -> str:
        """Format approval message for human review."""
        numeric_cols = plan.get("numeric_features", {}).get("columns", [])
        categorical_cols = plan.get("categorical_features", {}).get("columns", [])

        return f"""
Feature Engineering Complete
============================

Target Variable: {target_col}
Task Type: {plan.get('task_type', 'unknown')}

Numeric Features ({len(numeric_cols)}):
- Scaling: {plan.get('numeric_features', {}).get('scaling', 'none')}
- Imputation: {plan.get('numeric_features', {}).get('imputation', 'none')}

Categorical Features ({len(categorical_cols)}):
- Encoding: {plan.get('categorical_features', {}).get('encoding', 'none')}

Total Features After Processing: {len(feature_names)}

A preprocessing pipeline has been saved for inference.

Do you want to proceed to model training?
"""
