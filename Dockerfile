# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set work directory
WORKDIR /app

# Copy dependency definition files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies
# --frozen ensures we stick to the lockfile
# --no-install-project avoids needing the source code at this step
RUN uv sync --frozen --no-install-project

# Copy the rest of the project
COPY . .

# Place the virtual environment in the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port Streamlit runs on
EXPOSE 8501

# Healthcheck to ensure the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the app
CMD ["streamlit", "run", "demo/app.py"]
