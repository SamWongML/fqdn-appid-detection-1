# FQDN Orphan Detection ML System
# Multi-stage build for optimized image size

FROM python:3.11-slim as builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock .

# Install dependencies
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
# Create venv and install dependencies
RUN uv sync --frozen --no-install-project

# Runtime stage
FROM python:3.11-slim as runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . .

RUN mkdir -p /app/data/raw /app/data/processed /app/models/trained /app/reports /app/logs

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MODEL_ARTIFACTS_PATH=/app/models/trained \
    LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

CMD ["python", "-m", "src"]

# API Server stage
FROM runtime as api

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Training stage
FROM runtime as training

CMD ["python", "scripts/train.py"]
