#!/bin/bash
# Dependency Verification Script for PP Production Deployment
# Verifies all required packages are installed in each container

set -e

echo "=================================="
echo "PP Dependency Verification"
echo "=================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_container() {
    local container=$1
    local package=$2
    
    if docker exec "$container" pip list 2>/dev/null | grep -q "$package"; then
        echo -e "${GREEN}✓${NC} $container: $package installed"
        return 0
    else
        echo -e "${RED}✗${NC} $container: $package NOT installed"
        return 1
    fi
}

echo "Checking Backend Container..."
echo "------------------------------"
check_container "pp-backend-1" "fastapi"
check_container "pp-backend-1" "uvicorn"
check_container "pp-backend-1" "celery"
check_container "pp-backend-1" "redis"
check_container "pp-backend-1" "sqlalchemy"
check_container "pp-backend-1" "psycopg2-binary"
check_container "pp-backend-1" "python-socketio"
check_container "pp-backend-1" "httpx"
echo ""

echo "Checking Worker Container..."
echo "-----------------------------"
check_container "pp-worker-1" "celery"
check_container "pp-worker-1" "redis"
check_container "pp-worker-1" "sqlalchemy"
check_container "pp-worker-1" "psycopg2-binary"
check_container "pp-worker-1" "httpx"
check_container "pp-worker-1" "numpy"
check_container "pp-worker-1" "scipy"
check_container "pp-worker-1" "biopython"
check_container "pp-worker-1" "matplotlib"
check_container "pp-worker-1" "pandas"
echo ""

echo "Checking Database Connectivity..."
echo "----------------------------------"
if docker exec pp-backend-1 python -c "from app.database import engine; engine.connect()" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Backend can connect to PostgreSQL"
else
    echo -e "${RED}✗${NC} Backend CANNOT connect to PostgreSQL"
fi

if docker exec pp-worker-1 python -c "from app.database import engine; engine.connect()" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Worker can connect to PostgreSQL"
else
    echo -e "${RED}✗${NC} Worker CANNOT connect to PostgreSQL"
fi
echo ""

echo "Checking Redis Connectivity..."
echo "-------------------------------"
if docker exec pp-backend-1 python -c "import redis; r=redis.from_url('redis://redis:6379/0'); r.ping()" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Backend can connect to Redis"
else
    echo -e "${RED}✗${NC} Backend CANNOT connect to Redis"
fi

if docker exec pp-worker-1 python -c "import redis; r=redis.from_url('redis://redis:6379/0'); r.ping()" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Worker can connect to Redis"
else
    echo -e "${RED}✗${NC} Worker CANNOT connect to Redis"
fi
echo ""

echo "Checking Celery Worker Status..."
echo "---------------------------------"
if docker exec pp-worker-1 celery -A celery_app inspect ping 2>/dev/null | grep -q "pong"; then
    echo -e "${GREEN}✓${NC} Celery worker is responsive"
else
    echo -e "${YELLOW}⚠${NC} Celery worker may not be running"
fi
echo ""

echo "Checking Container Health..."
echo "-----------------------------"
for container in pp-backend-1 pp-worker-1 pp-redis-1 pp-postgres-1; do
    if [ "$(docker inspect -f '{{.State.Running}}' $container 2>/dev/null)" == "true" ]; then
        echo -e "${GREEN}✓${NC} $container is running"
    else
        echo -e "${RED}✗${NC} $container is NOT running"
    fi
done
echo ""

echo "=================================="
echo "Verification Complete"
echo "=================================="
