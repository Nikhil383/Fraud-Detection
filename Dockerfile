FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project definition and source
COPY pyproject.toml .
RUN uv pip install --system -e .

COPY src/ src/
COPY models/ models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENV PORT=8000
CMD uvicorn fraud_capstone.api:app --host 0.0.0.0 --port ${PORT:-8000}
