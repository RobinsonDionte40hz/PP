# Docker Configuration

Complete Docker configuration for the Protein Predictor platform, including development and production deployments.

## Overview

- **Development**: Standard Docker Compose setup (`docker-compose.yml`)
- **Production**: Enterprise-grade deployment (`docker-compose.prod.yml`) with:
  - Nginx reverse proxy with SSL/TLS
  - Redis persistence and optimization
  - PostgreSQL database
  - Logging and monitoring
  - Automated backups
  - Health checks and auto-restart

## Quick Start

### Development

1. **Prerequisites**: Docker 24.0+, Docker Compose 2.20+

2. **Setup**:
   ```bash
   cp .env.example .env
   docker-compose up --build
   ```

3. **Access**:
   - Frontend: http://localhost:3000
   - Backend: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Production

1. **Configure**:
   ```bash
   cp .env.production.example .env.production
   # Edit .env.production with production values
   ```

2. **Deploy** (Linux/Mac):
   ```bash
   bash docker/scripts/deploy.sh
   ```
   
   **Deploy** (Windows):
   ```cmd
   call docker\scripts\deploy.bat
   ```

3. **Access**:
   - Frontend: https://localhost
   - Backend: https://localhost/api
   - API Docs: https://localhost/api/docs

## Production Infrastructure

### Services

- **nginx**: Reverse proxy with SSL/TLS, rate limiting, compression
- **backend**: FastAPI application (Uvicorn)
- **worker**: Celery workers for async tasks
- **redis**: Cache and message broker (RDB + AOF persistence)
- **postgres**: PostgreSQL database with health checks

### Features

✅ **Security**
- SSL/TLS encryption
- Rate limiting (10 req/s API, 50 req/s general)
- Security headers (HSTS, CSP, X-Frame-Options)
- CORS configuration

✅ **Reliability**
- Health checks for all services
- Auto-restart on failure
- Graceful shutdown
- Resource limits

✅ **Monitoring**
- Structured logging
- Log rotation (14-day retention)
- Health check endpoint
- Container metrics

✅ **Operations**
- Automated backups (PostgreSQL, Redis, volumes)
- Restore scripts
- Rolling updates
- Scaling support

### Directory Structure

```
docker/
├── nginx/              # Reverse proxy configuration
│   ├── nginx.conf      # Main Nginx config
│   ├── conf.d/         # Server blocks
│   └── ssl/            # SSL certificates
├── redis/              # Redis production config
├── logging/            # Logging configuration
└── scripts/            # Deployment automation
    ├── deploy.sh/bat   # Initial deployment
    ├── update.sh       # Rolling updates
    ├── backup.sh/bat   # Automated backups
    ├── restore.sh      # Restore from backup
    └── health-check.sh # Service health checks
```

## Operations

### Monitoring

```bash
# View all logs
docker-compose -f docker-compose.prod.yml logs -f

# Service status
docker-compose -f docker-compose.prod.yml ps

# Health checks
bash docker/scripts/health-check.sh

# Resource usage
docker stats
```

### Backup & Restore

```bash
# Create backup
bash docker/scripts/backup.sh

# Restore from backup
bash docker/scripts/restore.sh 20240115_103045

# Automated daily backups (cron)
0 2 * * * /path/to/docker/scripts/backup.sh
```

### Updates

```bash
# Rolling update (zero-downtime)
bash docker/scripts/update.sh
```

### Scaling

```bash
# Scale backend
docker-compose -f docker-compose.prod.yml up -d --scale backend=3

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale worker=5
```

## Development Workflows

### Running Services Individually

```bash
# Backend + dependencies
docker-compose up backend postgres redis

# Frontend only
docker-compose up frontend
```

### Volumes

- `redis_data`: Redis persistence
- `postgres_data`: Database persistence  
- `./checkpoints`: Protein predictor checkpoints
- `./pdb_cache`: PDB file cache
- `./logs`: Application logs (production)

### Rebuilding

```bash
# After code changes
docker-compose up --build

# Clean rebuild
docker-compose down
docker system prune
docker-compose up --build
```

## Configuration

### Environment Variables

**Development** (`.env`):
- `DEBUG=true`
- `LOG_LEVEL=DEBUG`
- Standard ports (3000, 8000)

**Production** (`.env.production`):
- `ENVIRONMENT=production`
- `DEBUG=false`
- `POSTGRES_PASSWORD` (required)
- `SECRET_KEY` (required)
- `REDIS_PASSWORD` (required)
- SSL certificates
- Resource limits

### SSL/TLS Setup

**Development (Self-Signed)**:
```bash
bash docker/nginx/ssl/generate-self-signed-cert.sh
```

**Production (Let's Encrypt)**:
```bash
certbot certonly --standalone -d yourdomain.com
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem docker/nginx/ssl/cert.pem
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem docker/nginx/ssl/key.pem
```

## Troubleshooting

### Common Issues

**Port conflicts**: Check ports 80, 443, 5432, 6379
```bash
netstat -tuln | grep -E '80|443|5432|6379'
```

**Permission issues**: Adjust volume permissions on Linux
```bash
chmod -R 755 checkpoints/ pdb_cache/ logs/
```

**Build failures**: Clear Docker cache
```bash
docker system prune -a
```

### Service Debugging

**Check logs**:
```bash
docker-compose -f docker-compose.prod.yml logs backend
```

**Access container**:
```bash
docker-compose -f docker-compose.prod.yml exec backend bash
```

**Database connection**:
```bash
docker-compose -f docker-compose.prod.yml exec postgres psql -U postgres
```

**Redis connection**:
```bash
docker-compose -f docker-compose.prod.yml exec redis redis-cli
```

### Health Checks

```bash
# All services
bash docker/scripts/health-check.sh

# Individual checks
curl -k https://localhost/api/health
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping
docker-compose -f docker-compose.prod.yml exec postgres pg_isready
```

## Production Checklist

Before deploying:

- [ ] Configure `.env.production` with production values
- [ ] Obtain SSL/TLS certificates
- [ ] Set strong passwords (32+ characters)
- [ ] Configure CORS and allowed origins
- [ ] Set up automated backups
- [ ] Configure DNS records
- [ ] Test backup/restore procedures
- [ ] Enable firewall rules
- [ ] Set up monitoring alerts
- [ ] Document custom configuration

## Documentation

- **Full Production Guide**: See sections above and inline comments
- **Nginx Config**: `nginx/nginx.conf`, `nginx/conf.d/app.conf`
- **Redis Config**: `redis/redis.conf`
- **Logging**: `logging/README.md`
- **SSL Setup**: `nginx/ssl/README.md`

## Support

1. Check logs: `docker-compose -f docker-compose.prod.yml logs`
2. Review troubleshooting section above
3. See `docs/` for additional documentation
4. Create GitHub issue for bugs