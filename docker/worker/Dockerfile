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

# Copy ubf_protein package and QCPP module
COPY ubf_protein/ /packages/ubf_protein/
COPY src/ /packages/src/

# Install worker dependencies (QCPP needs numpy, scipy, biopython, etc.)
RUN pip install --no-cache-dir numpy scipy biopython pandas scikit-learn matplotlib

# Add /packages to PYTHONPATH so ubf_protein and src can be imported directly
ENV PYTHONPATH="/packages:${PYTHONPATH}"

# Copy backend code
COPY backend/ .

# Run Celery worker
CMD ["celery", "-A", "celery_app", "worker", "--loglevel=info", "--pool=solo"]