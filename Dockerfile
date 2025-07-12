FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation and Python optimization
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONOPTIMIZE=1
ENV UV_LINK_MODE=copy

# Set Python path to include the src directory for imports
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Copy application code
COPY src/ ./src/

# Copy only the processed RAG documents (not the large raw files)
COPY data/processed/ ./data/processed/

# Copy entrypoint script
COPY docker-entrypoint.sh ./docker-entrypoint.sh

# Pre-compile Python files to bytecode
RUN python -m compileall ./src/

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user with proper home directory and set permissions
RUN addgroup --system app && \
    adduser --system --ingroup app --home /home/app app && \
    mkdir -p /home/app/.streamlit && \
    mkdir -p /app/data/weaviate_db && \
    mkdir -p /app/logs && \
    chmod +x /app/docker-entrypoint.sh && \
    chown -R app:app /app && \
    chown -R app:app /home/app

# Ensure non-root user can write to necessary directories
RUN chmod 755 /app/data/weaviate_db && \
    chmod 755 /app/logs && \
    chmod 755 /home/app/.streamlit

# Switch to non-root user (ensure this is the final user)
USER app

# Verify we're running as non-root user
RUN whoami && id

# Set HOME environment variable
ENV HOME=/home/app

# Configure Streamlit to disable usage stats and set proper config
RUN echo '[browser]\ngatherUsageStats = false\n[server]\nheadless = true\n' > /home/app/.streamlit/config.toml

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["./docker-entrypoint.sh"]