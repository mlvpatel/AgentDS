"""
AgentDS DevOpsAgent.

Generates Docker configuration, CI/CD pipelines, and infrastructure code.

Author: Malav Patel
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agentds.agents.base import (
    AgentContext,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from agentds.core.artifact_store import ArtifactType
from agentds.core.logger import get_logger

logger = get_logger(__name__)


class DevOpsAgent(BaseAgent):
    """
    Agent for generating DevOps configurations.

    Outputs:
    - Dockerfile (multi-stage build)
    - docker-compose.yml
    - .dockerignore
    - requirements.txt (production)
    - GitHub Actions workflow (optional)
    - Kubernetes manifests (optional)
    """

    name = "DevOpsAgent"
    description = "Generate Docker and infrastructure configuration"
    phase = "deploy"
    complexity = "MEDIUM"
    input_types = ["api_code", "model_path"]
    output_types = ["dockerfile", "docker-compose", "kubernetes"]

    def get_system_prompt(self) -> str:
        """Get system prompt for DevOps configuration."""
        return """You are DevOpsAgent, a specialized agent for generating DevOps configurations.

Your responsibilities:
1. Generate production-ready Dockerfile with multi-stage builds
2. Create docker-compose.yml for local development
3. Generate .dockerignore for efficient builds
4. Create CI/CD pipeline configurations
5. Optionally generate Kubernetes manifests

Best practices to follow:
- Use multi-stage Docker builds to minimize image size
- Pin dependency versions for reproducibility
- Use non-root user in containers
- Include health checks
- Set appropriate resource limits
- Use secrets management for sensitive data

Output configurations that are:
- Production-ready
- Security-conscious
- Resource-efficient
- Easy to maintain
"""

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Execute DevOps configuration generation.

        Args:
            context: Execution context with API code path

        Returns:
            AgentResult with generated configurations
        """
        result = AgentResult(
            agent_name=self.name,
            status=AgentStatus.RUNNING,
        )

        try:
            # Get inputs from previous agents
            api_result = context.previous_results.get("APIWrapperAgent")
            if not api_result:
                raise ValueError("APIWrapperAgent result not found")
            # Generate configurations
            dockerfile = self._generate_dockerfile(context)
            docker_compose = self._generate_docker_compose(context)
            dockerignore = self._generate_dockerignore()
            requirements = self._generate_requirements(context)

            # Generate optional configurations
            github_actions = None
            k8s_manifests = None

            if context.settings.is_feature_enabled("github_actions"):
                github_actions = self._generate_github_actions(context)

            if context.settings.is_feature_enabled("kubernetes_configs"):
                k8s_manifests = self._generate_kubernetes(context)

            # Save artifacts
            output_dir = Path(context.settings.temp_dir) / context.job_id / "devops"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Copy application code (app.py)
            api_artifact_id = api_result.outputs.get("api_code_artifact_id")
            if api_artifact_id:
                app_code = self.artifact_store.load_text(api_artifact_id)
                (output_dir / "app.py").write_text(app_code)
            else:
                logger.warning("No API code artifact ID found")

            # Copy model artifacts
            model_dir = output_dir / "model"
            model_dir.mkdir(exist_ok=True)

            automl_result = context.previous_results.get("AutoMLAgent")
            if automl_result:
                model_path = automl_result.outputs.get("model_path")
                if model_path and Path(model_path).exists():
                    shutil.copy2(model_path, model_dir / "model.pkl")

            fe_result = context.previous_results.get("FeatureEngineerAgent")
            if fe_result:
                pipeline_path = fe_result.outputs.get("pipeline_path")
                if pipeline_path and Path(pipeline_path).exists():
                    shutil.copy2(pipeline_path, model_dir / "preprocessing_pipeline.pkl")

            artifacts = []

            # Save Dockerfile
            dockerfile_path = output_dir / "Dockerfile"
            dockerfile_path.write_text(dockerfile)
            artifact_id = self.save_artifact(
                job_id=context.job_id,
                name="Dockerfile",
                data=dockerfile,
                artifact_type=ArtifactType.CONFIG,
                description="Production Dockerfile with multi-stage build",
            )
            artifacts.append(artifact_id)

            # Save docker-compose.yml
            compose_path = output_dir / "docker-compose.yml"
            compose_path.write_text(docker_compose)
            artifact_id = self.save_artifact(
                job_id=context.job_id,
                name="docker-compose.yml",
                data=docker_compose,
                artifact_type=ArtifactType.CONFIG,
                description="Docker Compose configuration",
            )
            artifacts.append(artifact_id)

            # Save .dockerignore
            ignore_path = output_dir / ".dockerignore"
            ignore_path.write_text(dockerignore)
            artifact_id = self.save_artifact(
                job_id=context.job_id,
                name=".dockerignore",
                data=dockerignore,
                artifact_type=ArtifactType.CONFIG,
            )
            artifacts.append(artifact_id)

            # Save requirements.txt
            req_path = output_dir / "requirements.txt"
            req_path.write_text(requirements)
            artifact_id = self.save_artifact(
                job_id=context.job_id,
                name="requirements.txt",
                data=requirements,
                artifact_type=ArtifactType.CONFIG,
            )
            artifacts.append(artifact_id)

            # Save optional configs
            if github_actions:
                ga_dir = output_dir / ".github" / "workflows"
                ga_dir.mkdir(parents=True, exist_ok=True)
                ga_path = ga_dir / "deploy.yml"
                ga_path.write_text(github_actions)
                artifact_id = self.save_artifact(
                    job_id=context.job_id,
                    name=".github/workflows/deploy.yml",
                    data=github_actions,
                    artifact_type=ArtifactType.CONFIG,
                )
                artifacts.append(artifact_id)

            if k8s_manifests:
                k8s_dir = output_dir / "k8s"
                k8s_dir.mkdir(parents=True, exist_ok=True)
                for name, content in k8s_manifests.items():
                    k8s_path = k8s_dir / name
                    k8s_path.write_text(content)
                    artifact_id = self.save_artifact(
                        job_id=context.job_id,
                        name=f"k8s/{name}",
                        data=content,
                        artifact_type=ArtifactType.CONFIG,
                    )
                    artifacts.append(artifact_id)

            # Prepare result
            result.outputs = {
                "output_dir": output_dir,
                "dockerfile_path": dockerfile_path,
                "compose_path": compose_path,
                "files_generated": len(artifacts),
            }
            result.artifacts = artifacts
            result.approval_message = self._format_approval_message(
                dockerfile, docker_compose, len(artifacts)
            )
            result.mark_completed()

        except Exception as e:
            logger.error("DevOps generation failed", error=str(e), exc_info=True)
            result.mark_failed(str(e))

        return result

    def _generate_dockerfile(self, context: AgentContext) -> str:
        """Generate multi-stage Dockerfile."""
        # Use LLM for intelligent Dockerfile generation
        prompt = f"""Generate a production-ready multi-stage Dockerfile for a Python ML API service.

Requirements:
- Python 3.11 slim base image
- Multi-stage build (builder + runtime)
- Non-root user for security
- Health check endpoint at /health
- Expose port 8000
- Include model file copying
- Optimize for small image size

Task context: {context.task_description or 'ML model serving API'}

Output only the Dockerfile content, no explanations."""

        response = self.call_llm(prompt)

        # If LLM response looks valid, use it; otherwise use template
        if "FROM" in response.content and "COPY" in response.content:
            return response.content.strip()

        # Fallback template
        return '''# =============================================================================
# AgentDS Generated Dockerfile
# Multi-stage build for production ML API
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser model/ ./model/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''

    def _generate_docker_compose(self, context: AgentContext) -> str:
        """Generate docker-compose.yml."""
        return '''# =============================================================================
# AgentDS Generated Docker Compose
# Local development and testing configuration
# =============================================================================

version: "3.9"

services:
  # ---------------------------------------------------------------------------
  # ML API Service
  # ---------------------------------------------------------------------------
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: agentds-api
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - WORKERS=2
    volumes:
      - ./model:/app/model:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 2G
        reservations:
          cpus: "0.5"
          memory: 512M

  # ---------------------------------------------------------------------------
  # Redis Cache (Optional)
  # ---------------------------------------------------------------------------
  redis:
    image: redis:7-alpine
    container_name: agentds-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  # ---------------------------------------------------------------------------
  # Monitoring (Optional)
  # ---------------------------------------------------------------------------
  prometheus:
    image: prom/prometheus:latest
    container_name: agentds-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
    restart: unless-stopped
    profiles:
      - monitoring

volumes:
  redis_data:
  prometheus_data:

networks:
  default:
    name: agentds-network
'''

    def _generate_dockerignore(self) -> str:
        """Generate .dockerignore file."""
        return '''# =============================================================================
# AgentDS Docker Ignore
# =============================================================================

# Git
.git
.gitignore

# Python
__pycache__
*.py[cod]
*$py.class
*.so
.Python
.env
.venv
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.mypy_cache/

# Documentation
docs/
*.md
!README.md

# Data and models (use volumes instead)
data/
*.csv
*.parquet
mlruns/

# Logs
logs/
*.log

# Docker
Dockerfile*
docker-compose*
.docker/

# Kubernetes
k8s/

# CI/CD
.github/
.gitlab-ci.yml

# Misc
*.bak
*.tmp
.DS_Store
Thumbs.db
'''

    def _generate_requirements(self, context: AgentContext) -> str:
        """Generate production requirements.txt."""
        return '''# =============================================================================
# AgentDS Production Requirements
# Generated for ML API deployment
# =============================================================================

# Web Framework
litestar>=2.14.0
uvicorn[standard]>=0.34.0

# ML/Data
numpy>=1.26.0
pandas>=2.2.0
polars>=1.37.0
scikit-learn>=1.4.0
xgboost>=2.1.0
joblib>=1.3.0

# Utilities
pydantic>=2.10.0
python-dotenv>=1.0.0
structlog>=24.1.0

# Monitoring
prometheus-client>=0.19.0
'''

    def _generate_github_actions(self, context: AgentContext) -> str:
        """Generate GitHub Actions workflow."""
        return '''# =============================================================================
# AgentDS CI/CD Pipeline
# GitHub Actions workflow for build and deploy
# =============================================================================

name: Build and Deploy

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ---------------------------------------------------------------------------
  # Test
  # ---------------------------------------------------------------------------
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: pytest tests/ -v --cov=app

  # ---------------------------------------------------------------------------
  # Build and Push
  # ---------------------------------------------------------------------------
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha
            type=raw,value=latest

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  # ---------------------------------------------------------------------------
  # Deploy
  # ---------------------------------------------------------------------------
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    environment: production

    steps:
      - name: Deploy to production
        run: |
          echo "Deploying to production..."
          # Add your deployment commands here
          # e.g., kubectl apply, docker-compose up, etc.
'''

    def _generate_kubernetes(self, context: AgentContext) -> dict[str, str]:
        """Generate Kubernetes manifests."""
        manifests = {}

        # Deployment
        manifests["deployment.yaml"] = '''# =============================================================================
# AgentDS Kubernetes Deployment
# =============================================================================

apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentds-api
  labels:
    app: agentds-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: agentds-api
  template:
    metadata:
      labels:
        app: agentds-api
    spec:
      containers:
        - name: api
          image: ghcr.io/mlvpatel/agentds-api:latest
          ports:
            - containerPort: 8000
          env:
            - name: LOG_LEVEL
              value: "INFO"
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
'''

        # Service
        manifests["service.yaml"] = '''# =============================================================================
# AgentDS Kubernetes Service
# =============================================================================

apiVersion: v1
kind: Service
metadata:
  name: agentds-api
  labels:
    app: agentds-api
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
  selector:
    app: agentds-api
'''

        # HPA
        manifests["hpa.yaml"] = '''# =============================================================================
# AgentDS Horizontal Pod Autoscaler
# =============================================================================

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agentds-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agentds-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
'''

        return manifests

    def _format_approval_message(
        self, dockerfile: str, docker_compose: str, file_count: int
    ) -> str:
        """Format approval message."""
        return f"""
DevOps Configuration Generated
==============================

Files Generated: {file_count}

Dockerfile Preview (first 20 lines):
{chr(10).join(dockerfile.split(chr(10))[:20])}
...

Docker Compose Services:
- api: ML API service on port 8000
- redis: Redis cache (optional)
- prometheus: Monitoring (optional profile)

Features:
- Multi-stage Docker build
- Non-root user for security
- Health checks configured
- Resource limits set

Ready to build and deploy?
"""
