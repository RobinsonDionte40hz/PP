#!/bin/bash

# Update script for Protein Predictor production deployment
# Handles zero-downtime updates

set -e

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
BACKUP_BEFORE_UPDATE=true

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
echo "  Protein Predictor - Update Deployment"
echo "========================================="
echo ""

# Confirm update
log_warn "This will update the running production deployment"
read -p "Do you want to continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    log_info "Update cancelled"
    exit 0
fi

# Create backup before update
if [ "$BACKUP_BEFORE_UPDATE" = true ]; then
    log_step "Creating backup before update..."
    bash docker/scripts/backup.sh
    if [ $? -ne 0 ]; then
        log_error "Backup failed. Update aborted."
        exit 1
    fi
fi

# Pull latest code (if using git)
if [ -d ".git" ]; then
    log_step "Pulling latest code from git..."
    git pull origin main
fi

# Pull latest images
log_step "Pulling latest Docker images..."
docker-compose -f "$COMPOSE_FILE" pull

# Build custom images
log_step "Building updated application images..."
docker-compose -f "$COMPOSE_FILE" build

# Perform rolling update
log_step "Performing rolling update..."

# Update backend (one instance at a time if scaled)
log_info "Updating backend service..."
docker-compose -f "$COMPOSE_FILE" up -d --no-deps --scale backend=2 backend
sleep 10
docker-compose -f "$COMPOSE_FILE" up -d --no-deps --scale backend=1 backend

# Update worker
log_info "Updating worker service..."
docker-compose -f "$COMPOSE_FILE" up -d --no-deps worker

# Update nginx (last to minimize downtime)
log_info "Updating nginx service..."
docker-compose -f "$COMPOSE_FILE" up -d --no-deps nginx

# Run database migrations
log_step "Running database migrations..."
docker-compose -f "$COMPOSE_FILE" exec -T backend alembic upgrade head || log_warn "No migrations to run"

# Clean up old images
log_step "Cleaning up old Docker images..."
docker image prune -f

# Verify services
log_step "Verifying services..."
sleep 5
docker-compose -f "$COMPOSE_FILE" ps

# Check health
log_step "Checking service health..."
MAX_WAIT=60
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q "unhealthy"; then
        log_info "Waiting for services to become healthy... ($ELAPSED/$MAX_WAIT seconds)"
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    else
        log_info "All services are healthy"
        break
    fi
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    log_error "Services did not become healthy after update"
    log_error "Rolling back..."
    
    # Attempt rollback
    docker-compose -f "$COMPOSE_FILE" down
    # Restore from backup (manual step)
    log_error "Please restore from the backup created before update"
    log_error "Run: bash docker/scripts/restore.sh <timestamp>"
    exit 1
fi

# Display updated service status
log_step "Updated Service Status:"
docker-compose -f "$COMPOSE_FILE" ps

log_info "Update completed successfully!"
echo ""
echo "Services have been updated and are running"
echo ""
echo "To view logs:"
echo "  docker-compose -f $COMPOSE_FILE logs -f"
echo ""

exit 0
