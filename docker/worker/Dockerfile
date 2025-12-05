# Worker Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including PostgreSQL development files for psycopg2)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt ./backend_requirements.txt
COPY requirements_qcpp.txt ./qcpp_requirements.txt
COPY ubf_protein/requirements.txt ./ubf_requirements.txt

# Install Python dependencies (backend + QCPP + UBF)
RUN pip install --no-cache-dir -r backend_requirements.txt && \
    pip install --no-cache-dir -r qcpp_requirements.txt && \
    pip install --no-cache-dir -r ubf_requirements.txt

# Copy backend code
COPY backend/ .

# Copy ubf_protein module (needed by prediction_tasks) to parent of /app
COPY ubf_protein/ /ubf_protein/

# Copy src directory (contains QCPP protein_predictor.py)
COPY src/ /src/

# Add parent directory to PYTHONPATH so ubf_protein and src can be imported
ENV PYTHONPATH=/

# Run Celery worker
CMD ["celery", "-A", "celery_app", "worker", "--loglevel=info", "--pool=solo"]