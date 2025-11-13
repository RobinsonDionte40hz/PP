# Worker Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Copy ubf_protein module (needed by prediction_tasks) to parent of /app
COPY ubf_protein/ /ubf_protein/

# Add parent directory to PYTHONPATH so ubf_protein can be imported
ENV PYTHONPATH=/

# Run Celery worker
CMD ["celery", "-A", "celery_app", "worker", "--loglevel=info", "--pool=solo"]