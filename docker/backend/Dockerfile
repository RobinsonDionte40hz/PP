# Backend Dockerfile
# Runs FastAPI server with Socket.IO support
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements first for better caching
COPY backend/requirements/base.txt ./requirements_base.txt
COPY backend/requirements/prod.txt ./requirements_prod.txt

# Install backend dependencies
RUN pip install --no-cache-dir -r requirements_base.txt && \
    pip install --no-cache-dir -r requirements_prod.txt

# Install numpy for ubf_protein QCPP integration
RUN pip install --no-cache-dir numpy

# Copy ubf_protein package (backend only needs core, no QCPP)
COPY ubf_protein/ /packages/ubf_protein/

# Add /packages to PYTHONPATH so ubf_protein can be imported directly
# This is simpler and more reliable than pip install -e for this structure
ENV PYTHONPATH="/packages:${PYTHONPATH}"

# Copy backend code
COPY backend/ .

# Expose port
EXPOSE 8000

# Run the application with Socket.IO support
CMD ["uvicorn", "wsgi:socket_app", "--host", "0.0.0.0", "--port", "8000", "--reload"]