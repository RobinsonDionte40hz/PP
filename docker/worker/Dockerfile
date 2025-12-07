# Worker Dockerfile
# Runs Celery workers for prediction tasks
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including PostgreSQL development files for psycopg2)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first for better caching
COPY backend/requirements/base.txt ./requirements_base.txt
COPY backend/requirements/prod.txt ./requirements_prod.txt

# Install backend dependencies
RUN pip install --no-cache-dir -r requirements_base.txt && \
    pip install --no-cache-dir -r requirements_prod.txt

# Copy ubf_protein package and install with worker extras (includes QCPP dependencies)
COPY ubf_protein/ /packages/ubf_protein/
COPY src/ /packages/src/

# Install ubf_protein as a proper Python package with worker dependencies
RUN pip install --no-cache-dir -e /packages/ubf_protein[worker]

# Set up QCPP module path (src/ contains quantum_coherence_proteins)
ENV PYTHONPATH=/packages

# Copy backend code
COPY backend/ .

# Run Celery worker
CMD ["celery", "-A", "celery_app", "worker", "--loglevel=info", "--pool=solo"]