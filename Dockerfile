# SwasthyaSetu - Production Dockerfile
# Multi-stage build for optimized image size and security
# Python 3.13+ | FastAPI | FAISS-CPU | Torch
# =============================================================================

# =============================================================================
# STAGE 1: Builder Stage
# =============================================================================
FROM python:3.13-slim AS builder

# Set environment variables
ENV PYTHONDONTWREBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system build dependencies required for compiling Python packages
# - build-essential: Required for C/C++ extensions (torch, faiss, etc.)
# - cmake: Required for building some ML libraries
# - libomp-dev: Required for FAISS OpenMP support (parallel processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolated dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
# Copy only pyproject.toml first to leverage Docker layer caching
WORKDIR /build
COPY pyproject.toml .

# Install Python dependencies
# Use pip install with pyproject.toml (PEP 517/518 compliant)
RUN pip install --upgrade pip && \
    pip install -e . && \
    pip cache purge

# =============================================================================
# STAGE 2: Production Stage
# =============================================================================
FROM python:3.13-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    WORKERS=1

# Install runtime system dependencies only (no build tools)
# - libgomp1: Runtime library for OpenMP (required by FAISS)
# - curl: For health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
# UID 1000 is commonly used for container users
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
# Copy only necessary files to minimize attack surface
COPY --chown=appuser:appgroup main.py .
COPY --chown=appuser:appgroup static/ ./static/
COPY --chown=appuser:appgroup templates/ ./templates/

# Copy pubmed_data directory (kept in repo with .gitkeep placeholder)
# In production, prefer mounting real data via volume at /app/pubmed_data.
RUN mkdir -p pubmed_data
COPY --chown=appuser:appgroup pubmed_data/ ./pubmed_data/

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 8000

# Health check to ensure container is healthy
# FastAPI provides /health endpoint or we can use root
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || curl -f http://localhost:8000/ || exit 1

# Run the FastAPI application with Uvicorn
# Using uvicorn directly for simplicity; for production with multiple workers,
# consider using gunicorn with uvicorn workers:
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
