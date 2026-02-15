# =============================================================================
# StatmateAI – Production Dockerfile
# =============================================================================
# Builds a self-contained image that can run the FastAPI backend (default)
# or the React frontend via multi-stage build targets.
#
# Usage:
#   docker compose up              # preferred (see docker-compose.yml)
#   docker build -t statmate-ai .  # standalone backend image
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 – Python backend
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS backend

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # uv puts the venv here
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

# System deps required at runtime (weasyprint needs libcairo, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libcairo2 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libgdk-pixbuf2.0-0 \
        libffi-dev \
        libglib2.0-0 \
        shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python deps first (layer cache)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p data/uploads data/results data/logs database

EXPOSE 8000

# Default: run the FastAPI backend via uvicorn
CMD ["python", "-m", "uvicorn", "statmate.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# Stage 2 – Frontend build (Node / Vite)
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 3 – Lightweight frontend server (serves static build via nginx)
# ---------------------------------------------------------------------------
FROM nginx:alpine AS frontend

COPY --from=frontend-build /app/frontend/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000
