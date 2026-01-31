# =============================================================================
# AgentDS Dockerfile
# Multi-stage build for production deployment
# Author: Malav Patel
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create wheels for all dependencies
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

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

# Copy wheels from builder and install
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY --chown=agentds:agentds agentds/ ./agentds/
COPY --chown=agentds:agentds config/ ./config/
COPY --chown=agentds:agentds pyproject.toml .
COPY --chown=agentds:agentds README.md .

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
