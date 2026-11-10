# =============================================================================
# AgentDS Dockerfile
# Multi-stage build for production deployment
# Author: Malav Patel
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# uv installs exactly what uv.lock pins (same version as CI)
RUN pip install --no-cache-dir uv==0.11.32

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Install dependencies from the lockfile first, for better layer caching.
# --locked fails the build if uv.lock is out of date with pyproject.toml.
# Runtime dependencies plus the `all` and `apo` extras, no dev tools
# (the same set as requirements.txt).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra all --extra apo --no-install-project

# Then install AgentDS itself (non-editable, so the venv is self-contained)
COPY agentds/ ./agentds/
RUN uv sync --locked --no-dev --extra all --extra apo --no-editable

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Labels
LABEL maintainer="Malav Patel <malav.patel203@gmail.com>"
LABEL version="1.0.0"
LABEL description="AgentDS - Autonomous Data Science Pipeline"

# Create non-root user
RUN groupadd -r agentds && useradd -r -g agentds agentds

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the locked virtualenv (dependencies and AgentDS) from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Runtime configuration (read relative to the working directory)
COPY --chown=agentds:agentds config/ ./config/

# Create directories
RUN mkdir -p /app/outputs /app/logs /app/temp /app/checkpoints \
    && chown -R agentds:agentds /app

# Switch to non-root user
USER agentds

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=7860

# Expose ports
EXPOSE 7860 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command: Gradio web interface
CMD ["python", "-m", "agentds.web.app", "--host", "0.0.0.0", "--port", "7860"]
