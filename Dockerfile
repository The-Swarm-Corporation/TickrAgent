# --------------------------------------------
# Step 1: Build Stage - Install dependencies
# --------------------------------------------
FROM python:3.11-slim AS builder

# Set environment variables for better production use
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a virtual environment
RUN python -m venv /opt/venv

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

# --------------------------------------------
# Step 2: Final Stage - Copy code and run FastAPI
# --------------------------------------------
FROM python:3.11-slim AS final

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy the virtual environment from the build stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Run the application with Uvicorn, referencing api.py as the entry point
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
