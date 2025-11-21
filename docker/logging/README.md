# Logging Configuration

Production logging configuration for the Protein Predictor application.

## Overview

This directory contains logging configurations for:
- Application logs (Python/FastAPI)
- Celery worker logs
- Nginx access and error logs
- Log rotation policies

## Files

- `logging.conf` - Python logging configuration (rotating file handlers)
- `logrotate.conf` - Log rotation configuration (14-day retention)

## Log Locations

Inside containers:
```
/var/log/app/app.log         # Application logs
/var/log/app/error.log       # Error logs only
/var/log/celery/worker.log   # Celery worker logs
/var/log/nginx/access.log    # Nginx access logs
/var/log/nginx/error.log     # Nginx error logs
```

On host (via volumes):
```
./logs/app/
./logs/celery/
./logs/nginx/
```

## Log Rotation

Logs are automatically rotated:
- **Frequency**: Daily
- **Retention**: 14 days
- **Compression**: Gzip after 1 day
- **Max Size**: 100MB per log file

## Usage

### View Logs

```bash
# Application logs
docker-compose -f docker-compose.prod.yml logs -f backend

# Celery worker logs
docker-compose -f docker-compose.prod.yml logs -f worker

# Nginx logs
docker-compose -f docker-compose.prod.yml logs -f nginx

# All logs
docker-compose -f docker-compose.prod.yml logs -f
```

### Access Log Files

```bash
# From host
tail -f ./logs/app/app.log
tail -f ./logs/nginx/access.log

# Inside container
docker-compose -f docker-compose.prod.yml exec backend tail -f /var/log/app/app.log
```

### Manual Log Rotation

```bash
# Rotate logs manually
logrotate -f docker/logging/logrotate.conf
```

## Log Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: Confirmation that things are working as expected
- **WARNING**: Indication of something unexpected
- **ERROR**: Serious problem that needs attention
- **CRITICAL**: Very serious error that may prevent the application from running

## Configuration in Code

To use this logging configuration in your Python code:

```python
import logging.config

# Load configuration
logging.config.fileConfig('docker/logging/logging.conf')

# Get logger
logger = logging.getLogger('app')

# Use logger
logger.info("Application started")
logger.error("An error occurred", exc_info=True)
```

## Environment Variables

Set log level via environment variable:

```bash
# In .env.production
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Monitoring Integration

Logs can be integrated with monitoring tools:

### Prometheus + Loki

```yaml
# docker-compose.prod.yml
services:
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    volumes:
      - ./logs:/var/log:ro

  promtail:
    image: grafana/promtail:latest
    volumes:
      - ./logs:/var/log:ro
      - ./docker/logging/promtail-config.yml:/etc/promtail/config.yml
```

### ELK Stack

```yaml
# docker-compose.prod.yml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    
  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    volumes:
      - ./logs:/var/log:ro
      
  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
```

## Log Format

### Application Logs
```
2024-01-15 10:30:45 - app - INFO - main:start:42 - Application started successfully
```

### Nginx Access Logs
```
192.168.1.100 - - [15/Jan/2024:10:30:45 +0000] "GET /api/prediction HTTP/1.1" 200 1234
```

### Celery Logs
```
[2024-01-15 10:30:45,123: INFO/MainProcess] Task app.tasks.predict_structure[abc-123] received
```

## Best Practices

1. **Structured Logging**: Use JSON format for easier parsing
2. **Correlation IDs**: Include request IDs in logs for tracing
3. **Sensitive Data**: Never log passwords, tokens, or personal data
4. **Performance**: Use appropriate log levels to avoid excessive logging
5. **Retention**: Archive old logs to external storage if needed

## Troubleshooting

### Logs Not Appearing

1. Check log directory permissions:
   ```bash
   ls -la logs/
   ```

2. Check logging configuration:
   ```bash
   docker-compose -f docker-compose.prod.yml exec backend python -c "import logging.config; logging.config.fileConfig('docker/logging/logging.conf')"
   ```

3. Check container logs:
   ```bash
   docker-compose -f docker-compose.prod.yml logs backend
   ```

### Disk Space Issues

1. Check log sizes:
   ```bash
   du -sh logs/*
   ```

2. Manually rotate logs:
   ```bash
   logrotate -f docker/logging/logrotate.conf
   ```

3. Adjust retention period in `logrotate.conf`

## Security

- Logs may contain sensitive information
- Restrict access to log directories
- Use encrypted volumes for log storage in production
- Implement log aggregation with access controls
- Regularly audit log access
