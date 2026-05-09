FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip + setuptools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Python deps (non-editable for production)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# App code
COPY . .

# Expose
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
