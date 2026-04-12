FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Install dependencies
RUN uv sync --no-dev --no-editable

# Create data directory
RUN mkdir -p /app/data/vectors /app/data/exports

ENV JARVIS_DATA_DIR=/app/data
ENV OLLAMA_HOST=http://ollama:11434

# Expose web UI port
EXPOSE 7777

# Default: run web UI (override with "jarvis" for MCP server)
CMD ["uv", "run", "jarvis-ui"]
