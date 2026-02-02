# Asset Shield V2 - Production Container
# Multi-stage build for optimized image size

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================
# Stage 2: Production
# ============================================
FROM python:3.11-slim as production

# Security: Run as non-root user
RUN groupadd -r assetshield && useradd -r -g assetshield assetshield

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/assetshield/.local

# Set environment
ENV PATH=/home/assetshield/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application code (Version-J only)
COPY modules/__init__.py modules/
COPY modules/jquants_client.py modules/
COPY modules/itayose_analyzer.py modules/
COPY modules/execution_core.py modules/
COPY modules/screener.py modules/
COPY modules/money_management.py modules/
COPY modules/tracker.py modules/
COPY modules/strategy_config.py modules/
COPY main_execution.py .
COPY config/ config/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/output \
    && chown -R assetshield:assetshield /app

# Switch to non-root user
USER assetshield

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import modules.jquants_client; print('OK')" || exit 1

# Default command
CMD ["python", "main_execution.py"]

# ============================================
# Stage 3: Development (optional)
# ============================================
FROM production as development

USER root

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

USER assetshield

CMD ["python", "-m", "pytest", "-v"]
