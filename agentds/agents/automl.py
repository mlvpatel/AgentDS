"""
AgentDS AutoMLAgent.

Automated machine learning with hyperparameter optimization.

Author: Malav Patel
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from agentds.agents.base import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from agentds.core.artifact_store import ArtifactType
from agentds.core.logger import get_logger

logger = get_logger(__name__)


class AutoMLAgent(BaseAgent):
    """
    Agent for automated machine learning.

    Features:
    - Multiple model algorithms
    - Hyperparameter optimization with Optuna
    - Cross-validation
    - Model selection
    - Performance metrics
    - Feature importance
    """

    name = "AutoMLAgent"
    description = "Automated machine learning with hyperparameter optimization"
    phase = "build"
    complexity = "HIGH"
    input_types = ["parquet", "pkl"]
    output_types = ["pkl", "json"]

    def get_system_prompt(self) -> str:
        """Get system prompt for AutoML."""
        return """You are AutoMLAgent, a specialized agent for automated machine learning.

Your responsibilities:
1. Select appropriate algorithms based on task type and data characteristics
2. Configure hyperparameter search spaces
3. Evaluate models using cross-validation
4. Select the best model based on performance metrics
5. Generate feature importance analysis

Output your model configuration in JSON format:
{
    "task_type": "classification|regression",
    "algorithms": [
        {
            "name": "xgboost|lightgbm|random_forest|logistic_regression|linear_regression",
            "priority": 1,
            "hyperparameters": {
                "param_name": {"type": "int|float|categorical", "low": 0, "high": 100, "choices": []}
            }
        }
    ],
    "optimization": {
        "metric": "accuracy|f1|roc_auc|rmse|mae|r2",
        "direction": "maximize|minimize",
        "n_trials": 50,
        "cv_folds": 5
    },
    "reasoning": "explanation of choices"
}

Prioritize XGBoost and LightGBM for tabular data. Use simpler models for small datasets.
"""

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute AutoML.

        Args:
            context: Execution context with processed data

        Returns:
            AgentResult with trained model
        """
        result = AgentResult(
            agent_name=self.name,
            status=AgentStatus.RUNNING,
        )

        try:
            # Get input data
            prev_result = context.previous_results.get("FeatureEngineerAgent")
            if not prev_result:
                raise ValueError("No input from FeatureEngineerAgent")

            data_path = prev_result.outputs.get("data_path")
            target_col = prev_result.outputs.get("target_column")
            task_type = prev_result.outputs.get("task_type", "classification")

            if not data_path or not target_col:
                raise ValueError("Missing data_path or target_column")

            # Load data
            df = pl.read_parquet(data_path)

            # Prepare data
            X, y = self._prepare_data(df, target_col)

            # Get model configuration from LLM
            config = self._get_model_config(df, task_type, context)

            # Run optimization
            best_model, best_params, best_score, all_results = self._run_optimization(
                X, y, config, task_type
            )

            # Calculate final metrics
            metrics = self._calculate_metrics(best_model, X, y, task_type)

            # Get feature importance
            importance = self._get_feature_importance(best_model, df.columns)

            # Save model
            model_path = Path(context.settings.temp_dir) / f"{context.job_id}_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)

            model_artifact = self.save_artifact(
                job_id=context.job_id,
                name="best_model.pkl",
                data=model_path,
                artifact_type=ArtifactType.MODEL,
                metadata={
                    "algorithm": config["algorithms"][0]["name"],
                    "best_score": best_score,
                    "task_type": task_type,
                },
                description="Best trained model",
            )

            # Save metrics
            metrics_artifact = self.save_artifact(
                job_id=context.job_id,
                name="metrics.json",
                data=json.dumps({
                    "metrics": metrics,
                    "best_params": best_params,
                    "feature_importance": importance,
                    "optimization_history": all_results[:20],
                }, indent=2, default=str),
                artifact_type=ArtifactType.REPORT,
                description="Model metrics and optimization history",
            )

            result.outputs = {
                "model_path": model_path,
                "model_artifact_id": model_artifact,
                "metrics_artifact_id": metrics_artifact,
                "metrics": metrics,
                "best_params": best_params,
                "best_score": best_score,
                "feature_importance": importance,
                "task_type": task_type,
                "target_column": target_col,
            }
            result.artifacts.extend([model_artifact, metrics_artifact])
            result.approval_message = self._format_approval_message(metrics, best_params, importance)
            result.mark_completed()

        except Exception as e:
            logger.error("AutoML failed", error=str(e), exc_info=True)
            result.mark_failed(str(e))

        return result

    def _prepare_data(
        self, df: pl.DataFrame, target_col: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        pdf = df.to_pandas()

        if target_col not in pdf.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        y = pdf[target_col].values
        X = pdf.drop(columns=[target_col]).values

        return X, y

    def _get_model_config(
        self,
        df: pl.DataFrame,
        task_type: str,
        context: AgentContext,
    ) -> dict[str, Any]:
        """Get model configuration from LLM."""
        prompt = f"""Configure AutoML for this task.

Data shape: {df.height} rows, {df.width} columns
Task type: {task_type}
Task description: {context.task_description or 'N/A'}

Recommend algorithms and hyperparameter search spaces.
Output as JSON.
"""

        response = self.call_llm(prompt)

        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning("Failed to parse model config", error=str(e))

        # Default configuration
        if task_type == "classification":
            return {
                "task_type": "classification",
                "algorithms": [
                    {
                        "name": "xgboost",
                        "priority": 1,
                        "hyperparameters": {
                            "n_estimators": {"type": "int", "low": 50, "high": 300},
                            "max_depth": {"type": "int", "low": 3, "high": 10},
                            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                        },
                    }
                ],
                "optimization": {
                    "metric": "f1",
                    "direction": "maximize",
                    "n_trials": 30,
                    "cv_folds": 5,
                },
            }
        else:
            return {
                "task_type": "regression",
                "algorithms": [
                    {
                        "name": "xgboost",
                        "priority": 1,
                        "hyperparameters": {
                            "n_estimators": {"type": "int", "low": 50, "high": 300},
                            "max_depth": {"type": "int", "low": 3, "high": 10},
                            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                        },
                    }
                ],
                "optimization": {
                    "metric": "rmse",
                    "direction": "minimize",
                    "n_trials": 30,
                    "cv_folds": 5,
                },
            }

    def _run_optimization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: dict[str, Any],
        task_type: str,
    ) -> tuple[Any, dict[str, Any], float, list[dict[str, Any]]]:
        """Run hyperparameter optimization with Optuna."""
        import optuna
        from xgboost import XGBClassifier, XGBRegressor

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        optimization = config.get("optimization", {})
        n_trials = optimization.get("n_trials", 30)
        cv_folds = optimization.get("cv_folds", 5)
        metric = optimization.get("metric", "f1" if task_type == "classification" else "rmse")
        direction = optimization.get("direction", "maximize")

        all_results = []

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
            }

            if task_type == "classification":
                model = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
                scoring = "f1_weighted" if metric == "f1" else "accuracy"
            else:
                model = XGBRegressor(**params)
                scoring = "neg_mean_squared_error" if metric == "rmse" else "r2"

            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            score = scores.mean()

            all_results.append({
                "trial": trial.number,
                "params": params,
                "score": float(score),
            })

            return score

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_params["random_state"] = 42

        # Train final model
        if task_type == "classification":
            best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss")
        else:
            best_model = XGBRegressor(**best_params)

        best_model.fit(X, y)

        return best_model, best_params, study.best_value, all_results

    def _calculate_metrics(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
    ) -> dict[str, float]:
        """Calculate model metrics."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        y_pred = model.predict(X_test)

        if task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            }
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                    if y_proba.shape[1] == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
            except Exception:
                pass
        else:
            metrics = {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }

        return metrics

    def _get_feature_importance(
        self, model: Any, columns: list[str]
    ) -> dict[str, float]:
        """Get feature importance from model."""
        importance = {}

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = columns[:-1]  # Exclude target

            for i, imp in enumerate(importances):
                if i < len(feature_names):
                    importance[feature_names[i]] = float(imp)

            # Sort by importance
            importance = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )

        return importance

    def _format_approval_message(
        self,
        metrics: dict[str, float],
        params: dict[str, Any],
        importance: dict[str, float],
    ) -> str:
        """Format approval message for human review."""
        metrics_text = "\n".join(
            f"  - {k}: {v:.4f}" for k, v in metrics.items()
        )

        params_text = "\n".join(
            f"  - {k}: {v}" for k, v in list(params.items())[:5]
        )

        top_features = list(importance.items())[:5]
        features_text = "\n".join(
            f"  - {name}: {imp:.4f}" for name, imp in top_features
        ) or "  N/A"

        return f"""
AutoML Complete
===============

Best Model: XGBoost

Performance Metrics:
{metrics_text}

Best Hyperparameters:
{params_text}

Top 5 Features by Importance:
{features_text}

A trained model has been saved and is ready for deployment.

Do you want to proceed to API generation?
"""
