#!/bin/bash

# Production deployment script for Protein Predictor
# Handles initial deployment and updates

set -e

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Banner
echo "========================================="
echo "  Protein Predictor - Production Deploy"
echo "========================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    log_warn "Running as root. Consider using a non-root user with sudo."
fi

# Check prerequisites
log_step "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

log_info "Docker version: $(docker --version)"
log_info "Docker Compose version: $(docker-compose --version)"

# Check environment file
if [ ! -f "$ENV_FILE" ]; then
    log_error "Environment file not found: $ENV_FILE"
    log_error "Please copy .env.production.example to .env.production and configure it."
    exit 1
fi

# Validate environment file
log_step "Validating environment configuration..."

# Check for required variables
REQUIRED_VARS=("POSTGRES_PASSWORD" "SECRET_KEY" "REDIS_PASSWORD")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if ! grep -q "^${var}=" "$ENV_FILE" || grep -q "^${var}=$" "$ENV_FILE"; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    log_error "Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    exit 1
fi

log_info "Environment configuration validated"

# Check SSL certificates
log_step "Checking SSL certificates..."

SSL_CERT="docker/nginx/ssl/cert.pem"
SSL_KEY="docker/nginx/ssl/key.pem"

if [ ! -f "$SSL_CERT" ] || [ ! -f "$SSL_KEY" ]; then
    log_warn "SSL certificates not found"
    log_info "Generating self-signed certificates for development..."
    bash docker/nginx/ssl/generate-self-signed-cert.sh
fi

log_info "SSL certificates found"

# Create required directories
log_step "Creating required directories..."

mkdir -p logs/app logs/nginx logs/celery
mkdir -p backups
mkdir -p pdb_cache
mkdir -p checkpoints
mkdir -p prediction_results

log_info "Directories created"

# Pull latest images
log_step "Pulling Docker images..."
docker-compose -f "$COMPOSE_FILE" pull

# Build custom images
log_step "Building application images..."
docker-compose -f "$COMPOSE_FILE" build --no-cache

# Stop existing containers (if any)
if docker-compose -f "$COMPOSE_FILE" ps -q | grep -q .; then
    log_step "Stopping existing containers..."
    docker-compose -f "$COMPOSE_FILE" down
fi

# Start services
log_step "Starting services..."
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for services to be healthy
log_step "Waiting for services to become healthy..."

MAX_WAIT=120
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q "unhealthy"; then
        log_info "Waiting for services... ($ELAPSED/$MAX_WAIT seconds)"
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    else
        log_info "All services are healthy"
        break
    fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    log_error "Services did not become healthy in time"
    log_error "Check logs: docker-compose -f $COMPOSE_FILE logs"
    exit 1
fi

# Run database migrations (if applicable)
log_step "Running database migrations..."
docker-compose -f "$COMPOSE_FILE" exec -T backend alembic upgrade head || log_warn "No migrations to run"

# Display service status
log_step "Service Status:"
docker-compose -f "$COMPOSE_FILE" ps

# Display service URLs
echo ""
log_info "Deployment completed successfully!"
echo ""
echo "Services are available at:"
echo "  - Frontend: https://localhost"
echo "  - Backend API: https://localhost/api"
echo "  - API Docs: https://localhost/api/docs"
echo ""
echo "To view logs:"
echo "  docker-compose -f $COMPOSE_FILE logs -f"
echo ""
echo "To stop services:"
echo "  docker-compose -f $COMPOSE_FILE down"
echo ""
echo "To create a backup:"
echo "  bash docker/scripts/backup.sh"
echo ""

# Create initial backup
read -p "Create initial backup? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_step "Creating initial backup..."
    bash docker/scripts/backup.sh
fi

log_info "Deployment script completed"

exit 0
