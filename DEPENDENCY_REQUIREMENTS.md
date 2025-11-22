# Dependency Requirements - PP Production Deployment

## Overview
This document specifies all required dependencies for the Protein Prediction Platform to ensure successful production deployment.

## Container Dependencies

### Backend Container (`pp-backend-1`)
**Base Image:** `python:3.11-slim`
**System Packages:**
- `gcc` - C compiler for building Python packages
- `libpq-dev` - PostgreSQL development files (for psycopg2)

**Python Packages:** (from `backend/requirements.txt`)
```
fastapi==0.115.0
uvicorn[standard]==0.32.0
celery==5.4.0
redis==5.2.0
sqlalchemy==2.0.36
psycopg2-binary==2.9.10
python-socketio==5.11.4
python-multipart==0.0.12
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.1
pytest==8.3.3
pytest-asyncio==0.24.0
pytest-cov==6.0.0
httpx==0.27.2
requests==2.32.3
slowapi==0.1.9
```

**Critical for:**
- REST API endpoints
- WebSocket/SocketIO server
- Database ORM
- Authentication & security
- Rate limiting

### Worker Container (`pp-worker-1`)
**Base Image:** `python:3.11-slim`
**System Packages:**
- `gcc` - C compiler
- `libpq-dev` - PostgreSQL development files

**Python Packages:** (3 requirements files combined)

1. **Backend Requirements** (`backend/requirements.txt`) - Same as above
2. **QCPP Requirements** (`requirements_qcpp.txt`):
```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
biopython>=1.79
pandas>=1.2.0
scikit-learn>=0.24.0
statsmodels>=0.13.0
```

3. **UBF Requirements** (`ubf_protein/requirements.txt`):
```
pytest>=7.0.0
dataclasses>=0.6
typing>=3.7.4
matplotlib>=3.5.0
seaborn>=0.11.0
```

**Critical for:**
- Celery task execution
- UBF protein prediction system
- QCPP quantum coherence calculations
- Database access
- WebSocket progress emission
- PDB file processing

### Redis Container (`pp-redis-1`)
**Image:** `redis:7-alpine`
**Purpose:** Message broker for Celery, caching

### PostgreSQL Container (`pp-postgres-1`)
**Image:** `postgres:15-alpine`
**Purpose:** Primary database for predictions, campaigns, results

## Environment Variables Required

### Both Backend & Worker
```bash
REDIS_URL=redis://redis:6379/0
DATABASE_URL=postgresql://user:password@postgres:5432/pp_db
```

### Worker Only (Critical Addition)
```bash
BACKEND_URL=http://backend:8000
```
**Purpose:** Allows worker to emit WebSocket events via backend HTTP endpoint

### Production Additional
```bash
SECRET_KEY=<strong-secret-key>
JWT_SECRET_KEY=<strong-jwt-secret>
CORS_ORIGINS=https://yourdomain.com
ENABLE_HSTS=true
ENABLE_CSRF=true
```

## Verification Checklist

### Pre-Deployment
- [ ] All Dockerfiles have correct system dependencies
- [ ] All requirements.txt files are present and complete
- [ ] docker-compose.yml has all environment variables
- [ ] docker-compose.prod.yml has all environment variables
- [ ] Frontend .env has correct API URL

### Post-Deployment
Run verification script:
```bash
bash verify_dependencies.sh
```

Or manual checks:
```bash
# Check backend packages
docker exec pp-backend-1 pip list | grep -E "fastapi|uvicorn|celery|psycopg2|socketio|httpx"

# Check worker packages  
docker exec pp-worker-1 pip list | grep -E "numpy|scipy|biopython|psycopg2|httpx"

# Check database connectivity
docker exec pp-backend-1 python -c "from app.database import engine; engine.connect()"
docker exec pp-worker-1 python -c "from app.database import engine; engine.connect()"

# Check Redis connectivity
docker exec pp-backend-1 python -c "import redis; r=redis.from_url('redis://redis:6379/0'); r.ping()"

# Check Celery worker
docker exec pp-worker-1 celery -A celery_app inspect ping
```

## Common Issues & Solutions

### Issue: Worker can't find numpy/scipy
**Cause:** requirements_qcpp.txt not installed
**Solution:** Rebuild worker with updated Dockerfile that installs all 3 requirements files

### Issue: Worker can't connect to database
**Cause:** psycopg2-binary not installed
**Solution:** Ensure Dockerfile has `libpq-dev` system package and psycopg2-binary Python package

### Issue: WebSocket updates not reaching frontend
**Cause:** Worker calling localhost:8000 instead of backend:8000
**Solution:** Set `BACKEND_URL=http://backend:8000` in worker environment

### Issue: Import errors in worker for UBF modules
**Cause:** ubf_protein directory not copied to worker
**Solution:** Ensure Dockerfile has `COPY ubf_protein/ /ubf_protein/` and `ENV PYTHONPATH=/`

## Testing Dependencies

After deployment, run a test prediction:
```bash
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

Monitor logs:
```bash
docker logs pp-worker-1 -f
docker logs pp-backend-1 -f
```

Check for:
- ✓ No ModuleNotFoundError
- ✓ No ImportError
- ✓ Database connection successful
- ✓ WebSocket emission successful
- ✓ Task completion

## Production Hardening

Additional considerations for production:
1. **Pin all dependency versions** - Avoid using `>=` in requirements
2. **Multi-stage builds** - Separate build and runtime environments
3. **Health checks** - All containers should have healthchecks
4. **Resource limits** - Set CPU/memory limits in docker-compose.prod.yml
5. **Dependency scanning** - Use tools like Safety, Bandit for vulnerability scanning
6. **Regular updates** - Schedule dependency update reviews

## Update Procedure

When updating dependencies:
1. Update requirements.txt files
2. Rebuild containers: `docker compose build`
3. Run tests: `pytest backend/tests`
4. Run verification script
5. Deploy to staging first
6. Monitor for 24 hours
7. Deploy to production

## Contact & Support

For dependency-related issues:
- Check logs: `docker logs <container-name>`
- Run verification: `bash verify_dependencies.sh`
- Review documentation: `docs/SETUP.md`
