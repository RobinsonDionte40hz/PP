# Docker Setup Guide

This directory contains the Docker configuration for the Protein Prediction Platform frontend interface.

## Prerequisites

- Docker 24.0+
- Docker Compose 2.20+

## Quick Start

1. **Clone and navigate to the project root**:
   ```bash
   cd /path/to/PP
   ```

2. **Copy environment file**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your configuration values.

3. **Build and start all services**:
   ```bash
   docker-compose up --build
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Services

- **frontend**: React application served by Nginx
- **backend**: FastAPI application
- **worker**: Celery worker for background tasks
- **redis**: Redis for caching and Celery broker
- **postgres**: PostgreSQL database (optional)

## Development

For development, you can run services individually:

```bash
# Start only backend and database
docker-compose up backend postgres redis

# Start only frontend
docker-compose up frontend
```

## Volumes

- `redis_data`: Redis persistence
- `postgres_data`: Database persistence
- `./checkpoints`: PP system checkpoints
- `./visualization_output`: PP visualization files
- `./pdb_cache`: PDB file cache

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 3000, 8000, 6379, 5432 are available
2. **Permission issues**: On Linux, you may need to adjust file permissions for volumes
3. **Build failures**: Clear Docker cache with `docker system prune`

### Logs

View logs for all services:
```bash
docker-compose logs
```

View logs for specific service:
```bash
docker-compose logs backend
```

### Rebuilding

After code changes:
```bash
docker-compose up --build
```

## Production Deployment

For production, use the production compose file:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

Note: Production configuration includes SSL, health checks, and optimized settings.