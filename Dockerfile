FROM python:3.12-slim

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

# Copy pyproject.toml from project root
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies with uv sync (auto-resolves psycopg[binary])
RUN uv sync --locked

# Copy application code including src directory
COPY ./app ./

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]