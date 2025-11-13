# Backend Setup Guide

## Prerequisites

### 1. Install Redis for Windows

**Option A: Binary Install**
1. Download Redis for Windows: https://github.com/microsoftarchive/redis/releases
2. Extract to `C:\Redis` or any location
3. Add to PATH or note the location

**Option B: Docker (Recommended)**
```bash
docker run -d -p 6379:6379 --name redis redis:alpine
```

**Option C: WSL (Windows Subsystem for Linux)**
```bash
wsl
sudo apt-get update
sudo apt-get install redis-server
redis-server
```

### 2. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

## Running the Services

### Quick Start (All Services)
From project root:
```bash
START_ALL.bat
```

### Manual Start (Individual Services)

#### 1. Start Redis
```bash
cd backend
start_redis.bat
```
Or manually: `redis-server`

#### 2. Start Celery Worker
```bash
cd backend
start_celery.bat
```
Or manually:
```bash
celery -A celery_app worker --loglevel=info --pool=solo
```
Note: Use `--pool=solo` on Windows

#### 3. Start Backend API
```bash
cd backend
start_backend.bat
```
Or manually:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Verify Services

### Check Redis
```bash
redis-cli ping
# Should return: PONG
```

### Check Backend API
Open: http://localhost:8000/health

### Check Celery
Look for "celery@HOSTNAME ready" in the Celery terminal

## Troubleshooting

### Redis Connection Error
- Ensure Redis is running: `redis-cli ping`
- Check firewall settings
- Verify REDIS_URL in `.env` or `config.py`

### Celery Won't Start
- Install eventlet: `pip install eventlet`
- Use solo pool on Windows: `--pool=solo`
- Check Redis connection

### Import Errors
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version: Python 3.8-3.12 recommended
- Virtual environment activated

## Environment Variables

Create `.env` file in `backend/` directory:

```env
APP_ENV=development
SECRET_KEY=your-secret-key-here
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql://user:password@localhost:5432/pp_db
```

## Production Deployment

For production, use:
- Redis with persistence enabled
- Celery with multiple workers
- Gunicorn instead of Uvicorn
- Nginx as reverse proxy
- Supervisor or systemd for process management
