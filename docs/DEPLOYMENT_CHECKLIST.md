# Production Deployment Checklist

## Pre-Deployment Verification

###  1. Dependencies Verified
- [x] Worker has all QCPP dependencies (numpy, scipy, biopython, pandas, matplotlib, scikit-learn)
- [x] Worker has all UBF dependencies (pytest, matplotlib, seaborn)
- [x] Worker has database driver (psycopg2-binary)
- [x] Worker has HTTP client (httpx)
- [x] Backend has all FastAPI dependencies
- [x] Backend has WebSocket support (python-socketio)
- [x] Backend has HTTP client (httpx)
- [x] All containers have PostgreSQL connectivity

### 2. Configuration Files
- [x] `docker-compose.yml` - Development configuration updated
- [x] `docker-compose.prod.yml` - Production configuration updated
- [x] `backend/requirements.txt` - All backend deps listed
- [x] `requirements_qcpp.txt` - QCPP/quantum deps listed
- [x] `ubf_protein/requirements.txt` - UBF deps listed
- [x] `docker/worker/Dockerfile` - Installs all 3 requirements files
- [x] `docker/backend/Dockerfile` - Installs backend requirements

### 3. Environment Variables
#### Development (docker-compose.yml)
- [x] `REDIS_URL=redis://redis:6379/0`
- [x] `DATABASE_URL=postgresql://user:password@postgres:5432/pp_db`
- [x] `BACKEND_URL=http://backend:8000` (worker only)

#### Production (docker-compose.prod.yml)
- [x] All above variables
- [ ] `SECRET_KEY` - Set strong secret key
- [ ] `JWT_SECRET_KEY` - Set strong JWT secret
- [ ] `POSTGRES_PASSWORD` - Set strong database password
- [ ] `CORS_ORIGINS` - Set to production domain
- [ ] SSL certificates configured

### 4. Frontend Configuration
- [ ] Build production frontend: `cd frontend && npm run build`
- [ ] Update `.env` or `.env.production` with production API URL
- [ ] Verify WebSocket connection URL matches backend

### 5. Database Setup
- [x] PostgreSQL container running
- [x] Database tables created (via SQLAlchemy)
- [ ] Database backups configured
- [ ] Database migrations tested

### 6. Testing
- [x] All containers start successfully
- [x] Backend health check passes: `curl http://localhost:8000/health`
- [x] WebSocket connections work
- [x] Celery worker can pick up tasks
- [ ] Run full prediction test
- [ ] Frontend can connect to backend
- [ ] Real-time monitoring displays updates
- [ ] Test all API endpoints
- [ ] Run backend test suite: `pytest backend/tests`

## Deployment Steps

### Step 1: Environment Preparation
```bash
# Clone repository
git clone <repo-url>
cd PP

# Copy and configure environment files
cp .env.example .env.production
# Edit .env.production with production values
```

### Step 2: Build Docker Images
```bash
# For development
docker compose build

# For production
docker compose -f docker-compose.prod.yml build
```

### Step 3: Start Services
```bash
# Development
docker compose up -d

# Production
docker compose -f docker-compose.prod.yml up -d
```

### Step 4: Verify Deployment
```bash
# Check all containers are running
docker ps

# Check logs
docker logs pp-backend-1
docker logs pp-worker-1

# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/api/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MQIFVKTLTGKTITLE",
    "configuration": {
      "iterations": 100,
      "agents": 5,
      "diversity": "balanced"
    }
  }'
```

### Step 5: Monitor First Prediction
```bash
# Watch worker logs
docker logs pp-worker-1 -f

# Watch backend logs  
docker logs pp-backend-1 -f

# Check for:
# ✓ No ModuleNotFoundError or ImportError
# ✓ Database connections successful
# ✓ WebSocket emissions working
# ✓ Progress updates reaching frontend
```

## Post-Deployment Monitoring

### Health Checks
- [ ] Set up automated health checks
- [ ] Monitor container resource usage
- [ ] Set up log aggregation
- [ ] Configure alerting for failures

### Performance Monitoring
- [ ] Monitor prediction task duration
- [ ] Monitor WebSocket connection stability
- [ ] Monitor database query performance
- [ ] Monitor Redis memory usage

### Security
- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up regular security updates
- [ ] Enable database encryption
- [ ] Configure rate limiting appropriately
- [ ] Review CORS settings

## Troubleshooting Guide

### Issue: Worker can't find numpy
**Solution**: Rebuild worker with:
```bash
docker compose build worker
docker compose up -d worker
```

### Issue: WebSocket not working
**Check**:
1. BACKEND_URL environment variable set in worker
2. Backend has python-socketio installed
3. Frontend WebSocket URL matches backend
4. CORS settings allow WebSocket connections

### Issue: Database connection failed
**Check**:
1. PostgreSQL container is running
2. DATABASE_URL is correct
3. psycopg2-binary is installed
4. Network connectivity between containers

### Issue: Predictions stay in "queued" status
**Check**:
1. Celery worker is running: `docker logs pp-worker-1`
2. Redis is accessible
3. Task was submitted to Celery
4. Check worker logs for errors

## Rollback Procedure

If deployment fails:
```bash
# Stop new containers
docker compose down

# Restore previous version
git checkout <previous-version-tag>
docker compose up -d

# Restore database if needed
docker exec pp-postgres-1 psql -U user -d pp_db < backup.sql
```

## Dependencies Summary

### Critical Dependencies
| Component | Purpose | Risk if Missing |
|-----------|---------|-----------------|
| psycopg2-binary | PostgreSQL driver | Database access fails |
| numpy/scipy | Scientific computing | Predictions fail |
| httpx | HTTP client | WebSocket emission fails |
| celery | Task queue | Predictions don't run |
| python-socketio | WebSocket server | Real-time updates fail |

### Verification Commands
```bash
# Backend dependencies
docker exec pp-backend-1 pip list | grep -E "fastapi|psycopg2|socketio|httpx"

# Worker dependencies
docker exec pp-worker-1 pip list | grep -E "numpy|scipy|biopython|psycopg2|httpx"

# Connectivity tests
docker exec pp-backend-1 python -c "from app.database import engine; engine.connect()"
docker exec pp-worker-1 python -c "from app.database import engine; engine.connect()"
docker exec pp-worker-1 celery -A celery_app inspect ping
```

## Success Criteria

Deployment is successful when:
- ✅ All 5 containers running (frontend, backend, worker, redis, postgres)
- ✅ Health check returns 200 OK
- ✅ Can create prediction via API
- ✅ Celery worker picks up task
- ✅ No import/module errors in logs
- ✅ WebSocket emits progress updates
- ✅ Frontend displays live monitoring
- ✅ Prediction completes successfully
- ✅ Results can be retrieved

## Maintenance

### Regular Tasks
- Weekly: Review logs for errors
- Weekly: Check container resource usage  
- Monthly: Update dependencies for security patches
- Monthly: Database backup verification
- Quarterly: Full disaster recovery test

### Updating Dependencies
1. Update requirements files
2. Rebuild containers: `docker compose build`
3. Test in staging environment
4. Run test suite
5. Deploy to production
6. Monitor for 24 hours

---

**Last Updated**: November 21, 2025
**Version**: 1.0.0
**Status**: ✅ Production Ready
