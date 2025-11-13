# Troubleshooting Guide - Protein Prediction Platform

This guide covers common issues and their solutions.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Docker Issues](#docker-issues)
- [Backend Issues](#backend-issues)
- [Frontend Issues](#frontend-issues)
- [Prediction Execution Issues](#prediction-execution-issues)
- [Performance Issues](#performance-issues)
- [WebSocket Issues](#websocket-issues)
- [Database Issues](#database-issues)

## Installation Issues

### Python Version Compatibility

**Problem**: BioPython installation fails on Python 3.13+

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement biopython
```

**Solutions**:

1. **Use Python 3.12** (recommended):
   ```bash
   python --version  # Should show 3.12.x
   ```

2. **Install C++ Build Tools** (Windows):
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++"
   - Retry: `pip install biopython`

3. **Use Conda** (alternative):
   ```bash
   conda install -c conda-forge biopython
   ```

### Node.js Version Issues

**Problem**: Frontend won't build or has dependency errors

**Symptoms**:
```
error: Unsupported engine
The engine "node" is incompatible with this module
```

**Solution**: Update Node.js to version 18+
```bash
node --version  # Should be 18.0.0 or higher
```

Download from: https://nodejs.org/

### Permission Denied (Linux/macOS)

**Problem**: Cannot install packages or run commands

**Symptoms**:
```
EACCES: permission denied
```

**Solutions**:

1. **Use virtual environment** (Python):
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Fix npm permissions**:
   ```bash
   mkdir ~/.npm-global
   npm config set prefix '~/.npm-global'
   export PATH=~/.npm-global/bin:$PATH
   ```

3. **Don't use sudo** with pip/npm (creates permission issues)

---

## Docker Issues

### Port Already in Use

**Problem**: Docker containers fail to start

**Symptoms**:
```
Error: bind: address already in use
```

**Solution 1**: Find and stop the conflicting process

Windows:
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

Linux/macOS:
```bash
lsof -i :8000
kill -9 <PID>
```

**Solution 2**: Change ports in `.env`:
```bash
FRONTEND_PORT=3001
BACKEND_PORT=8001
REDIS_PORT=6380
POSTGRES_PORT=5433
```

### Docker Build Fails

**Problem**: Docker image build fails

**Symptoms**:
```
ERROR [builder 5/8] RUN pip install -r requirements.txt
```

**Solutions**:

1. **Clear Docker cache**:
   ```bash
   docker system prune -a
   docker-compose build --no-cache
   ```

2. **Check disk space**:
   ```bash
   docker system df
   ```
   If low, clean up:
   ```bash
   docker system prune --volumes
   ```

3. **Update Docker**:
   ```bash
   docker --version  # Should be 24.0+
   ```

### Container Keeps Restarting

**Problem**: Container enters restart loop

**Symptoms**:
```
Status: Restarting (1) 3 seconds ago
```

**Solution**: Check container logs
```bash
docker-compose logs <service-name>
```

Common causes:
- Missing environment variables
- Database connection failure
- Port conflicts
- Permission issues with volumes

Fix example for backend:
```bash
# Check logs
docker-compose logs backend

# Common fix: ensure Redis is running first
docker-compose up redis
docker-compose up backend
```

### Volume Permission Issues (Linux)

**Problem**: Cannot write to mounted volumes

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied
```

**Solution**: Fix ownership
```bash
sudo chown -R $USER:$USER ./checkpoints ./results ./pdb_cache
chmod -R 755 ./checkpoints ./results ./pdb_cache
```

---

## Backend Issues

### Redis Connection Failed

**Problem**: Cannot connect to Redis

**Symptoms**:
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solutions**:

1. **Check Redis is running**:
   ```bash
   redis-cli ping  # Should return PONG
   ```

2. **Start Redis**:
   
   Windows (Docker):
   ```bash
   docker run -d -p 6379:6379 redis:7-alpine
   ```
   
   macOS:
   ```bash
   brew services start redis
   ```
   
   Linux:
   ```bash
   sudo systemctl start redis-server
   ```

3. **Check connection string** in `.env`:
   ```bash
   REDIS_URL=redis://localhost:6379/0
   ```

4. **Test connection**:
   ```bash
   redis-cli -h localhost -p 6379 ping
   ```

### Celery Worker Not Starting

**Problem**: Celery worker fails to start or crashes

**Symptoms**:
```
ValueError: not enough values to unpack
```

**Solutions**:

1. **Windows users**: Add `--pool=solo` flag
   ```bash
   celery -A app.celery_app:celery_app worker --loglevel=info --pool=solo
   ```

2. **Check Redis connection** (see above)

3. **Verify imports**:
   ```bash
   python -c "from app.celery_app import celery_app"
   ```

4. **Check task registration**:
   ```bash
   celery -A app.celery_app:celery_app inspect registered
   ```

### Import Errors

**Problem**: Module not found errors

**Symptoms**:
```
ModuleNotFoundError: No module named 'app'
```

**Solutions**:

1. **Ensure virtual environment is activated**:
   ```bash
   # Should see (venv) in prompt
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Check Python path**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
   set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
   ```

4. **Run from correct directory** (backend/)

### API Returns 500 Error

**Problem**: Internal server error on API calls

**Symptoms**:
```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "Internal server error"
  }
}
```

**Solutions**:

1. **Check backend logs**:
   ```bash
   # Docker
   docker-compose logs backend
   
   # Direct run
   # Check terminal where uvicorn is running
   ```

2. **Enable debug mode** in `.env`:
   ```bash
   LOG_LEVEL=DEBUG
   APP_ENV=development
   ```

3. **Common causes**:
   - PP system integration error (check file paths)
   - Database connection issue
   - Missing environment variables
   - Celery not running

### PP System Integration Errors

**Problem**: Predictions fail to start or execute

**Symptoms**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'test_protein.py'
```

**Solutions**:

1. **Verify PP files exist**:
   ```bash
   ls test_protein.py
   ls systematic_protein_testing.py
   ls -d ubf_protein/
   ```

2. **Check file paths** in `.env`:
   ```bash
   PP_RESULTS_DIR=./results          # Must be relative to backend/
   PP_CHECKPOINTS_DIR=./checkpoints
   PP_PDB_CACHE_DIR=./pdb_cache
   ```

3. **Create directories**:
   ```bash
   mkdir -p results checkpoints pdb_cache
   ```

4. **Test PP system directly**:
   ```bash
   python test_protein.py --quick
   ```

---

## Frontend Issues

### Build Fails

**Problem**: Frontend build fails with errors

**Symptoms**:
```
✘ [ERROR] Build failed
```

**Solutions**:

1. **Clear node_modules and reinstall**:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **Check Node version**:
   ```bash
   node --version  # Should be 18+
   ```

3. **Clear Vite cache**:
   ```bash
   rm -rf node_modules/.vite
   npm run build
   ```

4. **Check for TypeScript errors**:
   ```bash
   npx tsc --noEmit
   ```

### API Calls Failing (CORS)

**Problem**: API requests blocked by CORS

**Symptoms** (Browser console):
```
Access to XMLHttpRequest blocked by CORS policy
```

**Solutions**:

1. **Check backend CORS settings** in `.env`:
   ```bash
   CORS_ORIGINS=http://localhost:3000,http://localhost:5173
   ```

2. **Add your frontend URL** if different

3. **Restart backend** after changing CORS settings

4. **Verify in browser DevTools**:
   - Network tab → Request headers
   - Should see `Origin: http://localhost:3000`
   - Response should have `Access-Control-Allow-Origin`

### API Calls Failing (Connection Refused)

**Problem**: Cannot connect to backend API

**Symptoms** (Browser console):
```
GET http://localhost:8000/api/health net::ERR_CONNECTION_REFUSED
```

**Solutions**:

1. **Check backend is running**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify API URL** in `frontend/.env`:
   ```bash
   VITE_API_URL=http://localhost:8000
   ```

3. **Check firewall settings** (Windows):
   - Allow uvicorn through firewall

4. **Try different host**:
   ```bash
   # In backend
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### Hot Reload Not Working

**Problem**: Changes don't reflect in browser

**Solutions**:

1. **Hard refresh**: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (macOS)

2. **Clear browser cache**

3. **Restart dev server**:
   ```bash
   npm run dev
   ```

4. **Check file watcher limits** (Linux):
   ```bash
   echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
   sudo sysctl -p
   ```

### TypeScript Errors

**Problem**: TypeScript compilation errors

**Solutions**:

1. **Update types**:
   ```bash
   npm install --save-dev @types/node @types/react @types/react-dom
   ```

2. **Check tsconfig.json** exists

3. **Clear TypeScript cache**:
   ```bash
   rm -rf node_modules/.cache
   ```

4. **Restart TS server** (VS Code):
   - Cmd/Ctrl + Shift + P
   - "TypeScript: Restart TS Server"

---

## Prediction Execution Issues

### Prediction Stuck in Queue

**Problem**: Prediction stays in "queued" status

**Symptoms**: Status doesn't change from "queued" for a long time

**Solutions**:

1. **Check Celery worker is running**:
   ```bash
   # Docker
   docker-compose ps worker
   
   # Direct
   # Look for celery process
   ps aux | grep celery
   ```

2. **Check Celery logs**:
   ```bash
   docker-compose logs worker
   ```

3. **Restart Celery worker**:
   ```bash
   docker-compose restart worker
   ```

4. **Check Redis queue**:
   ```bash
   redis-cli llen celery
   ```

### Prediction Fails Immediately

**Problem**: Prediction fails right after starting

**Symptoms**: Status changes to "failed" within seconds

**Solutions**:

1. **Check task logs**:
   ```bash
   docker-compose logs worker
   ```

2. **Common causes**:
   - Invalid sequence (non-standard amino acids)
   - Missing dependencies (BioPython)
   - File permission issues
   - Insufficient memory

3. **Test sequence manually**:
   ```bash
   python test_protein.py --sequence ACDEFGHIKL
   ```

4. **Check error in prediction details**:
   - View prediction → Check error message

### High RMSD Results

**Problem**: Predictions consistently have high RMSD (>10Å)

**Symptoms**: Results show poor quality

**Solutions**:

1. **Increase iterations**:
   - Try 5000 instead of 1000

2. **Enable quantum refinement**:
   - Check "Enable Quantum Refinement" in form

3. **Increase agents**:
   - Try 50 agents instead of 10

4. **Enable all features**:
   - QCPP integration
   - Mediator agents
   - Geometric targeting

5. **Test with known protein**:
   - Use Ubiquitin (1UBQ) as benchmark
   - Should achieve 7-10Å (research phase)

### Memory Errors

**Problem**: Prediction fails with memory errors

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Solutions**:

1. **Reduce agents**: Use 10 instead of 50

2. **Reduce iterations**: Use 1000 instead of 5000

3. **Smaller sequence**: Test with <100 residues first

4. **Increase Docker memory** (Docker Desktop → Settings → Resources)

5. **Enable swap** (Linux):
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

## Performance Issues

### Slow Predictions

**Problem**: Predictions take very long to complete

**Solutions**:

1. **Use PyPy** for 2-5x speedup:
   ```bash
   pypy3 test_protein.py --sequence ACDEFGH
   ```

2. **Reduce parameters**:
   - Fewer iterations (1000 instead of 5000)
   - Fewer agents (10 instead of 50)

3. **Disable features** for testing:
   - Disable quantum refinement
   - Disable mediator agents

4. **Check system resources**:
   ```bash
   # CPU usage
   top
   
   # Memory usage
   free -h
   ```

5. **Close other applications**

### Slow Frontend

**Problem**: UI is slow or unresponsive

**Solutions**:

1. **Clear browser cache**

2. **Disable browser extensions**

3. **Check browser console** for errors

4. **Reduce chart update frequency**:
   - Settings → Visualization → Chart update interval

5. **Limit active WebSocket connections**:
   - Only monitor one prediction at a time

### Slow API Responses

**Problem**: API calls take too long

**Solutions**:

1. **Enable Redis caching**:
   ```bash
   CACHE_TTL_SECONDS=300
   ```

2. **Check Redis performance**:
   ```bash
   redis-cli --latency
   ```

3. **Optimize database queries** (if using PostgreSQL)

4. **Check network latency**:
   ```bash
   curl -w "@-" -o /dev/null -s http://localhost:8000/health <<< '
   time_namelookup:  %{time_namelookup}\n
   time_connect:  %{time_connect}\n
   time_total:  %{time_total}\n'
   ```

---

## WebSocket Issues

### WebSocket Connection Failed

**Problem**: Real-time updates not working

**Symptoms** (Browser console):
```
WebSocket connection failed
```

**Solutions**:

1. **Check backend is running**

2. **Verify WebSocket URL** in `frontend/.env`:
   ```bash
   VITE_WS_URL=http://localhost:8000
   ```

3. **Check browser console** for specific error

4. **Try different browser**

5. **Check firewall/antivirus** settings

6. **Restart backend**:
   ```bash
   docker-compose restart backend
   ```

### Updates Not Showing

**Problem**: Live monitoring page doesn't update

**Solutions**:

1. **Check WebSocket connection status**:
   - Look for connection indicator in UI

2. **Check browser console** for errors

3. **Refresh page** (Ctrl+R)

4. **Verify prediction is actually running**:
   - Check backend logs
   - Check Celery worker logs

5. **Check room subscription**:
   - Browser DevTools → Network → WS tab
   - Should see "join_prediction" message

---

## Database Issues

### Database Connection Failed

**Problem**: Cannot connect to PostgreSQL

**Symptoms**:
```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solutions**:

1. **Check PostgreSQL is running**:
   ```bash
   docker-compose ps postgres
   ```

2. **Verify connection string** in `.env`:
   ```bash
   DATABASE_URL=postgresql://ppuser:pppassword@localhost:5432/pp_db
   ```

3. **Test connection**:
   ```bash
   psql -h localhost -U ppuser -d pp_db
   ```

4. **Check PostgreSQL logs**:
   ```bash
   docker-compose logs postgres
   ```

5. **Recreate database**:
   ```bash
   docker-compose down -v
   docker-compose up -d postgres
   ```

### Migration Errors

**Problem**: Database migration fails

**Solutions**:

1. **Check Alembic configuration**

2. **Manually run migrations**:
   ```bash
   alembic upgrade head
   ```

3. **Reset database** (development only):
   ```bash
   alembic downgrade base
   alembic upgrade head
   ```

4. **Check migration files** for errors

---

## Getting Additional Help

### Diagnostic Commands

Run these to gather diagnostic information:

```bash
# System info
uname -a                    # OS info
python --version            # Python version
node --version              # Node version
docker --version            # Docker version

# Backend status
curl http://localhost:8000/health

# Service status
docker-compose ps           # All services
redis-cli ping              # Redis
psql -U ppuser -d pp_db -c "SELECT 1"  # PostgreSQL

# Logs
docker-compose logs --tail=100 backend
docker-compose logs --tail=100 worker
docker-compose logs --tail=100 frontend

# Resource usage
docker stats                # Container resources
free -h                     # Memory
df -h                       # Disk space
```

### Reporting Issues

When reporting issues, include:

1. **System information**:
   - OS and version
   - Python version
   - Node version
   - Docker version

2. **Error messages**:
   - Full error output
   - Stack traces
   - Browser console errors

3. **Steps to reproduce**

4. **Expected vs actual behavior**

5. **Configuration**:
   - Relevant environment variables (redact secrets)
   - Docker Compose version
   - Any modifications made

6. **Logs**:
   - Backend logs
   - Worker logs
   - Frontend console errors

### Resources

- [Setup Guide](SETUP.md) - Installation instructions
- [User Guide](USER_GUIDE.md) - How to use the platform
- [Developer Guide](DEVELOPER_GUIDE.md) - Development details
- [API Documentation](API.md) - API reference
- [Environment Variables](ENVIRONMENT_VARIABLES.md) - Configuration
- GitHub Issues: <repository-url>/issues

---

## Common Error Messages

### "Port already in use"
See [Port Already in Use](#port-already-in-use)

### "Connection refused"
See [API Calls Failing (Connection Refused)](#api-calls-failing-connection-refused)

### "CORS policy"
See [API Calls Failing (CORS)](#api-calls-failing-cors)

### "Redis connection failed"
See [Redis Connection Failed](#redis-connection-failed)

### "Module not found"
See [Import Errors](#import-errors)

### "Memory error"
See [Memory Errors](#memory-errors)

### "WebSocket connection failed"
See [WebSocket Connection Failed](#websocket-connection-failed)

---

If your issue isn't covered here, please check the other documentation guides or create a GitHub issue with detailed information.
