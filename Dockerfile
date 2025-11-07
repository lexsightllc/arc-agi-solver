FROM python:3.9-slim-buster

WORKDIR /app

# Install system dependencies for Z3 and Numba
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libz3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables for determinism (example)
ENV PYTHONHASHSEED=0
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Expose any ports if necessary (e.g., for MLflow UI)
# EXPOSE 5000

# Entrypoint for the application (e.g., to run the CLI)
ENTRYPOINT ["python", "src/main.py"]
