#!/bin/bash

# Health check script for Protein Predictor services

set -e

COMPOSE_FILE="docker-compose.prod.yml"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_service() {
    local service=$1
    local url=$2
    
    if curl -sf "$url" > /dev/null; then
        echo -e "${GREEN}✓${NC} $service is healthy"
        return 0
    else
        echo -e "${RED}✗${NC} $service is unhealthy"
        return 1
    fi
}

echo "Checking service health..."
echo ""

# Check nginx
check_service "Nginx" "http://localhost/health"

# Check backend API
check_service "Backend API" "http://localhost/api/health"

# Check Redis
if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping > /dev/null; then
    echo -e "${GREEN}✓${NC} Redis is healthy"
else
    echo -e "${RED}✗${NC} Redis is unhealthy"
fi

# Check PostgreSQL
if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U postgres > /dev/null; then
    echo -e "${GREEN}✓${NC} PostgreSQL is healthy"
else
    echo -e "${RED}✗${NC} PostgreSQL is unhealthy"
fi

# Check Celery worker
CELERY_STATUS=$(docker-compose -f "$COMPOSE_FILE" exec -T worker celery -A celery_app inspect ping 2>&1)
if echo "$CELERY_STATUS" | grep -q "pong"; then
    echo -e "${GREEN}✓${NC} Celery worker is healthy"
else
    echo -e "${RED}✗${NC} Celery worker is unhealthy"
fi

echo ""
echo "Container status:"
docker-compose -f "$COMPOSE_FILE" ps

exit 0
