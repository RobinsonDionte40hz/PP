#!/bin/bash

# Backup script for Protein Predictor production deployment
# Creates backups of database and volumes

set -e

# Configuration
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=14

# Docker Compose file
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

# Create backup directory
mkdir -p "$BACKUP_DIR"

log_info "Starting backup at $TIMESTAMP"

# Backup PostgreSQL database
log_info "Backing up PostgreSQL database..."
docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dumpall -U postgres | gzip > "$BACKUP_DIR/postgres_$TIMESTAMP.sql.gz"

if [ $? -eq 0 ]; then
    log_info "PostgreSQL backup completed: postgres_$TIMESTAMP.sql.gz"
else
    log_error "PostgreSQL backup failed"
    exit 1
fi

# Backup Redis data
log_info "Backing up Redis data..."
docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli BGSAVE
sleep 5  # Wait for BGSAVE to complete
docker cp $(docker-compose -f "$COMPOSE_FILE" ps -q redis):/data/dump.rdb "$BACKUP_DIR/redis_$TIMESTAMP.rdb"

if [ $? -eq 0 ]; then
    log_info "Redis backup completed: redis_$TIMESTAMP.rdb"
else
    log_warn "Redis backup failed (non-critical)"
fi

# Backup Docker volumes
log_info "Backing up Docker volumes..."

# Get volume names
POSTGRES_VOLUME=$(docker-compose -f "$COMPOSE_FILE" config | grep -A 1 "postgres_data:" | tail -n 1 | awk '{print $2}')
REDIS_VOLUME=$(docker-compose -f "$COMPOSE_FILE" config | grep -A 1 "redis_data:" | tail -n 1 | awk '{print $2}')

# Backup postgres volume
if [ -n "$POSTGRES_VOLUME" ]; then
    docker run --rm \
        -v "$POSTGRES_VOLUME:/data:ro" \
        -v "$BACKUP_DIR:/backup" \
        alpine tar czf "/backup/postgres_volume_$TIMESTAMP.tar.gz" -C /data .
    log_info "PostgreSQL volume backup completed: postgres_volume_$TIMESTAMP.tar.gz"
fi

# Backup redis volume
if [ -n "$REDIS_VOLUME" ]; then
    docker run --rm \
        -v "$REDIS_VOLUME:/data:ro" \
        -v "$BACKUP_DIR:/backup" \
        alpine tar czf "/backup/redis_volume_$TIMESTAMP.tar.gz" -C /data .
    log_info "Redis volume backup completed: redis_volume_$TIMESTAMP.tar.gz"
fi

# Backup PDB cache (if exists)
if [ -d "./pdb_cache" ]; then
    log_info "Backing up PDB cache..."
    tar czf "$BACKUP_DIR/pdb_cache_$TIMESTAMP.tar.gz" -C . pdb_cache
    log_info "PDB cache backup completed: pdb_cache_$TIMESTAMP.tar.gz"
fi

# Backup checkpoints (if exists)
if [ -d "./checkpoints" ]; then
    log_info "Backing up checkpoints..."
    tar czf "$BACKUP_DIR/checkpoints_$TIMESTAMP.tar.gz" -C . checkpoints
    log_info "Checkpoints backup completed: checkpoints_$TIMESTAMP.tar.gz"
fi

# Backup configuration files
log_info "Backing up configuration files..."
tar czf "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" \
    .env.production \
    docker-compose.prod.yml \
    docker/nginx/nginx.conf \
    docker/nginx/conf.d/app.conf \
    docker/redis/redis.conf \
    docker/logging/logging.conf

log_info "Configuration backup completed: config_$TIMESTAMP.tar.gz"

# Calculate backup sizes
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
log_info "Total backup size: $TOTAL_SIZE"

# Clean up old backups
log_info "Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "*.rdb" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete

# List recent backups
log_info "Recent backups:"
ls -lh "$BACKUP_DIR" | tail -n 10

log_info "Backup completed successfully at $(date)"

# Optional: Upload to cloud storage (uncomment and configure)
# log_info "Uploading to cloud storage..."
# aws s3 sync "$BACKUP_DIR" s3://your-bucket/backups/ --exclude "*" --include "*$TIMESTAMP*"
# log_info "Cloud upload completed"

exit 0
