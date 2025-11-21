#!/bin/bash

# Restore script for Protein Predictor production deployment
# Restores database and volumes from backup

set -e

# Configuration
BACKUP_DIR="/backups"
COMPOSE_FILE="docker-compose.prod.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if backup file is provided
if [ $# -eq 0 ]; then
    log_error "Usage: $0 <timestamp>"
    log_error "Available backups:"
    ls -1 "$BACKUP_DIR" | grep "_[0-9]\{8\}_[0-9]\{6\}" | sed 's/.*_\([0-9]\{8\}_[0-9]\{6\}\).*/\1/' | sort -u
    exit 1
fi

TIMESTAMP=$1

# Verify backup files exist
POSTGRES_BACKUP="$BACKUP_DIR/postgres_$TIMESTAMP.sql.gz"
REDIS_BACKUP="$BACKUP_DIR/redis_$TIMESTAMP.rdb"

if [ ! -f "$POSTGRES_BACKUP" ]; then
    log_error "PostgreSQL backup not found: $POSTGRES_BACKUP"
    exit 1
fi

log_warn "This will restore data from backup timestamp: $TIMESTAMP"
log_warn "Current data will be OVERWRITTEN!"
read -p "Are you sure you want to continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    log_info "Restore cancelled"
    exit 0
fi

# Stop services
log_info "Stopping services..."
docker-compose -f "$COMPOSE_FILE" stop

# Restore PostgreSQL database
log_info "Restoring PostgreSQL database..."
gunzip < "$POSTGRES_BACKUP" | docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U postgres

if [ $? -eq 0 ]; then
    log_info "PostgreSQL restore completed"
else
    log_error "PostgreSQL restore failed"
    exit 1
fi

# Restore Redis data
if [ -f "$REDIS_BACKUP" ]; then
    log_info "Restoring Redis data..."
    docker cp "$REDIS_BACKUP" $(docker-compose -f "$COMPOSE_FILE" ps -q redis):/data/dump.rdb
    log_info "Redis restore completed"
else
    log_warn "Redis backup not found, skipping"
fi

# Restore volumes (optional)
POSTGRES_VOLUME_BACKUP="$BACKUP_DIR/postgres_volume_$TIMESTAMP.tar.gz"
REDIS_VOLUME_BACKUP="$BACKUP_DIR/redis_volume_$TIMESTAMP.tar.gz"

if [ -f "$POSTGRES_VOLUME_BACKUP" ]; then
    log_info "Restoring PostgreSQL volume..."
    POSTGRES_VOLUME=$(docker-compose -f "$COMPOSE_FILE" config | grep -A 1 "postgres_data:" | tail -n 1 | awk '{print $2}')
    if [ -n "$POSTGRES_VOLUME" ]; then
        docker run --rm \
            -v "$POSTGRES_VOLUME:/data" \
            -v "$BACKUP_DIR:/backup" \
            alpine sh -c "rm -rf /data/* && tar xzf /backup/postgres_volume_$TIMESTAMP.tar.gz -C /data"
        log_info "PostgreSQL volume restore completed"
    fi
fi

if [ -f "$REDIS_VOLUME_BACKUP" ]; then
    log_info "Restoring Redis volume..."
    REDIS_VOLUME=$(docker-compose -f "$COMPOSE_FILE" config | grep -A 1 "redis_data:" | tail -n 1 | awk '{print $2}')
    if [ -n "$REDIS_VOLUME" ]; then
        docker run --rm \
            -v "$REDIS_VOLUME:/data" \
            -v "$BACKUP_DIR:/backup" \
            alpine sh -c "rm -rf /data/* && tar xzf /backup/redis_volume_$TIMESTAMP.tar.gz -C /data"
        log_info "Redis volume restore completed"
    fi
fi

# Restore PDB cache
PDB_CACHE_BACKUP="$BACKUP_DIR/pdb_cache_$TIMESTAMP.tar.gz"
if [ -f "$PDB_CACHE_BACKUP" ]; then
    log_info "Restoring PDB cache..."
    rm -rf ./pdb_cache
    tar xzf "$PDB_CACHE_BACKUP" -C .
    log_info "PDB cache restore completed"
fi

# Restore checkpoints
CHECKPOINTS_BACKUP="$BACKUP_DIR/checkpoints_$TIMESTAMP.tar.gz"
if [ -f "$CHECKPOINTS_BACKUP" ]; then
    log_info "Restoring checkpoints..."
    rm -rf ./checkpoints
    tar xzf "$CHECKPOINTS_BACKUP" -C .
    log_info "Checkpoints restore completed"
fi

# Start services
log_info "Starting services..."
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for services to be healthy
log_info "Waiting for services to become healthy..."
sleep 10

# Verify services
log_info "Verifying services..."
docker-compose -f "$COMPOSE_FILE" ps

log_info "Restore completed successfully at $(date)"
log_info "Please verify that all services are running correctly"

exit 0
