FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd -r botuser && useradd -r -g botuser -s /bin/false botuser

# Create necessary directories
RUN mkdir -p /app/logs /app/models \
    && chown -R botuser:botuser /app

# Copy application code
COPY --chown=botuser:botuser . .

# Remove unnecessary files from image
RUN rm -rf tests/ docs/ DEVELOPMENT_PLAN.md .github/ scripts/setup_digitalocean.sh \
    && find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Switch to non-root user
USER botuser

# Health check port
EXPOSE 8001

# Environment defaults (overridable)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STOCKS_ENV=production \
    STOCKS_LOG_LEVEL=INFO \
    STOCKS_HEALTH_PORT=8001

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# Graceful shutdown support (SIGTERM)
STOPSIGNAL SIGTERM

CMD ["python", "main.py"]
