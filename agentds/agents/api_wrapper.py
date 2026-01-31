"""
AgentDS APIWrapperAgent.

Generates production-ready API server code for model serving.

Author: Malav Patel
"""

from __future__ import annotations

import json
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


class APIWrapperAgent(BaseAgent):
    """
    Agent for generating API server code.

    Generates:
    - Litestar/FastAPI server code
    - Health check endpoint
    - Prediction endpoint
    - Batch prediction endpoint
    - OpenAPI documentation
    - Requirements file
    """

    name = "APIWrapperAgent"
    description = "Generate production-ready API server code for model serving"
    phase = "deploy"
    complexity = "HIGH"
    input_types = ["pkl", "json"]
    output_types = ["py", "txt"]

    def get_system_prompt(self) -> str:
        """Get system prompt for API generation."""
        return """You are APIWrapperAgent, a specialized agent for generating API server code.

Your responsibilities:
1. Generate production-ready API code using Litestar framework
2. Create health check, prediction, and batch prediction endpoints
3. Include proper error handling and validation
4. Generate OpenAPI documentation
5. Create requirements.txt for dependencies

The API should follow best practices:
- Input validation with Pydantic
- Proper error handling
- Async endpoints where beneficial
- Health check endpoint at /health
- Prediction endpoint at /predict
- Batch endpoint at /predict/batch
- OpenAPI docs at /docs

Output your API configuration in JSON format:
{
    "framework": "litestar",
    "endpoints": [
        {"path": "/predict", "method": "POST", "description": "Single prediction"}
    ],
    "input_schema": {
        "field_name": {"type": "float|int|str", "description": "field description"}
    },
    "output_schema": {
        "prediction": {"type": "float|int|str", "description": "prediction output"}
    }
}
"""

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute API generation.

        Args:
            context: Execution context with trained model

        Returns:
            AgentResult with API code
        """
        result = AgentResult(
            agent_name=self.name,
            status=AgentStatus.RUNNING,
        )

        try:
            # Get input from AutoMLAgent
            prev_result = context.previous_results.get("AutoMLAgent")
            if not prev_result:
                raise ValueError("No input from AutoMLAgent")

            prev_result.outputs.get("model_path")
            prev_result.outputs.get("metrics", {})
            task_type = prev_result.outputs.get("task_type", "classification")
            prev_result.outputs.get("feature_importance", {})

            # Get feature names from FeatureEngineerAgent
            fe_result = context.previous_results.get("FeatureEngineerAgent")
            feature_names = fe_result.outputs.get("feature_names", []) if fe_result else []
            fe_plan = fe_result.outputs.get("feature_plan", {}) if fe_result else {}

            # Determine input features from plan (raw inputs) instead of transformed outputs
            input_features = []
            if fe_plan:
                input_features.extend(fe_plan.get("numeric_features", {}).get("columns", []))
                input_features.extend(fe_plan.get("categorical_features", {}).get("columns", []))

            # Fallback if plan is empty
            if not input_features:
                input_features = feature_names

            # Get API configuration from LLM
            api_config = self._get_api_config(input_features, task_type, context)

            # Generate API code
            api_code = self._generate_api_code(api_config, input_features, task_type)

            # Generate requirements
            requirements = self._generate_requirements()

            # Save artifacts
            api_artifact = self.save_artifact(
                job_id=context.job_id,
                name="app.py",
                data=api_code,
                artifact_type=ArtifactType.CODE,
                description="Litestar API server code",
            )

            req_artifact = self.save_artifact(
                job_id=context.job_id,
                name="requirements.txt",
                data=requirements,
                artifact_type=ArtifactType.CONFIG,
                description="Python dependencies for API",
            )

            result.outputs = {
                "api_code_artifact_id": api_artifact,
                "requirements_artifact_id": req_artifact,
                "api_config": api_config,
                "endpoints": ["/health", "/predict", "/predict/batch", "/docs"],
            }
            result.artifacts.extend([api_artifact, req_artifact])
            result.approval_message = self._format_approval_message(api_config)
            result.mark_completed()

        except Exception as e:
            logger.error("API generation failed", error=str(e), exc_info=True)
            result.mark_failed(str(e))

        return result

    def _get_api_config(
        self,
        feature_names: list[str],
        task_type: str,
        context: AgentContext,
    ) -> dict[str, Any]:
        """Get API configuration from LLM."""
        prompt = f"""Generate API configuration for model serving.

Features: {feature_names[:20]}
Task type: {task_type}
Task description: {context.task_description or 'ML prediction API'}

Provide input/output schema and endpoint configuration as JSON.
"""

        response = self.call_llm(prompt)

        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning("Failed to parse API config", error=str(e))

        # Default configuration
        input_schema = {name: {"type": "float", "description": f"Feature {name}"} for name in feature_names[:20]}

        return {
            "framework": "litestar",
            "endpoints": [
                {"path": "/health", "method": "GET", "description": "Health check"},
                {"path": "/predict", "method": "POST", "description": "Single prediction"},
                {"path": "/predict/batch", "method": "POST", "description": "Batch predictions"},
            ],
            "input_schema": input_schema,
            "output_schema": {
                "prediction": {"type": "float" if task_type == "regression" else "int", "description": "Model prediction"},
                "probability": {"type": "float", "description": "Prediction probability"},
            },
        }

    def _generate_api_code(
        self,
        config: dict[str, Any],
        feature_names: list[str],
        task_type: str,
    ) -> str:
        """Generate Litestar API code."""
        # Generate Pydantic model fields
        input_fields = []
        for name in feature_names[:50]:
            safe_name = name.replace(" ", "_").replace("-", "_")
            input_fields.append(f"    {safe_name}: float = Field(..., description=\"Feature {name}\")")

        input_fields_str = "\n".join(input_fields) if input_fields else "    value: float = Field(..., description=\"Input value\")"

        code = f'''"""
AgentDS Generated API Server.

Production-ready ML inference API using Litestar framework.

Author: Malav Patel
Generated by: AgentDS
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from litestar import Litestar, get, post
from litestar.config.cors import CORSConfig
from litestar.openapi import OpenAPIConfig
from litestar.status_codes import HTTP_200_OK, HTTP_500_INTERNAL_SERVER_ERROR
from pydantic import BaseModel, Field


# =============================================================================
# MODELS
# =============================================================================

class PredictionInput(BaseModel):
    """Input schema for prediction."""
{input_fields_str}


class PredictionOutput(BaseModel):
    """Output schema for prediction."""
    prediction: {"float" if task_type == "regression" else "int"} = Field(..., description="Model prediction")
    probability: Optional[float] = Field(None, description="Prediction probability")


class BatchInput(BaseModel):
    """Input schema for batch prediction."""
    instances: List[PredictionInput] = Field(..., description="List of instances")


class BatchOutput(BaseModel):
    """Output schema for batch prediction."""
    predictions: List[PredictionOutput] = Field(..., description="List of predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Model load status")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")


# =============================================================================
# MODEL LOADING
# =============================================================================

MODEL_PATH = Path("model/model.pkl")
PIPELINE_PATH = Path("model/preprocessing_pipeline.pkl")

_model = None
_pipeline = None


def load_model() -> None:
    """Load model and pipeline from disk."""
    global _model, _pipeline

    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)

    if PIPELINE_PATH.exists():
        _pipeline = joblib.load(PIPELINE_PATH)


def get_model():
    """Get loaded model."""
    if _model is None:
        load_model()
    return _model


def get_pipeline():
    """Get loaded pipeline."""
    if _pipeline is None:
        load_model()
    return _pipeline


# =============================================================================
# ENDPOINTS
# =============================================================================

@get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    model = get_model()
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0",
    )


@post("/predict")
async def predict(data: PredictionInput) -> PredictionOutput:
    """Single prediction endpoint."""
    model = get_model()
    if model is None:
        raise ValueError("Model not loaded")

    # Create DataFrame with correct column names
    input_data = {field: [getattr(data, field)] for field in data.model_fields.keys()}  # type: ignore[name-defined]
    features = pd.DataFrame(input_data)

    # Apply preprocessing if available
    pipeline = get_pipeline()
    if pipeline is not None:
        try:
            features = pipeline.transform(features)
        except Exception:
            pass  # Use raw features if pipeline fails

    # Make prediction
    prediction = model.predict(features)[0]

    # Get probability if available
    probability = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(features)[0]
            probability = float(max(proba))
        except Exception:
            pass

    return PredictionOutput(
        prediction={"float(prediction)" if task_type == "regression" else "int(prediction)"},
        probability=probability,
    )


@post("/predict/batch")
async def predict_batch(data: BatchInput) -> BatchOutput:
    """Batch prediction endpoint."""
    model = get_model()
    if model is None:
        raise ValueError("Model not loaded")

    predictions = []
    for instance in data.instances:
        # Create DataFrame for single instance
        input_data = {field: [getattr(instance, field)] for field in instance.model_fields.keys()}  # type: ignore[name-defined]
        features = pd.DataFrame(input_data)

        pipeline = get_pipeline()
        if pipeline is not None:
            try:
                features = pipeline.transform(features)
            except Exception:
                pass

        pred = model.predict(features)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(features)[0]
                probability = float(max(proba))
            except Exception:
                pass

        predictions.append(PredictionOutput(
            prediction={"float(pred)" if task_type == "regression" else "int(pred)"},
            probability=probability,
        ))

    return BatchOutput(predictions=predictions)


# =============================================================================
# APP CONFIGURATION
# =============================================================================

cors_config = CORSConfig(
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

openapi_config = OpenAPIConfig(
    title="AgentDS ML API",
    version="1.0.0",
    description="Machine Learning inference API generated by AgentDS",
)

app = Litestar(
    route_handlers=[health_check, predict, predict_batch],
    cors_config=cors_config,
    openapi_config=openapi_config,
    on_startup=[load_model],
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        return code

    def _generate_requirements(self) -> str:
        """Generate requirements.txt."""
        return """# AgentDS Generated API Requirements
# Author: Malav Patel

# Web framework
litestar>=2.14.0
uvicorn[standard]>=0.34.0

# Data processing
numpy>=1.26.0
pandas>=2.2.0
joblib>=1.3.0

# ML
scikit-learn>=1.4.0
xgboost>=2.1.0

# Validation
pydantic>=2.10.0

# Utilities
python-dotenv>=1.0.0
"""

    def _format_approval_message(self, config: dict[str, Any]) -> str:
        """Format approval message for human review."""
        endpoints = config.get("endpoints", [])
        endpoints_text = "\n".join(
            f"  - {e['method']} {e['path']}: {e['description']}"
            for e in endpoints
        )

        return f"""
API Generation Complete
=======================

Framework: Litestar

Endpoints Generated:
{endpoints_text}

Features:
- Health check endpoint
- Single prediction endpoint
- Batch prediction endpoint
- OpenAPI documentation at /docs
- CORS enabled
- Pydantic validation

Files generated:
- app.py (API server code)
- requirements.txt (dependencies)

Do you want to proceed to Docker configuration?
"""
