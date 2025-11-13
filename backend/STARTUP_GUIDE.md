# Backend Startup Guide

## 🚀 Quick Start

### Option 1: Full Stack (with Redis & Celery)
For production-like environment with async task processing:

```bash
# 1. Start Redis (required for Celery)
start_redis.bat

# 2. Start Celery Worker (in new terminal)
start_celery.bat

# 3. Start Backend Server (in new terminal)
start_backend.bat
```

### Option 2: Development Mode (without Redis & Celery)
For quick development without background task processing:

```bash
# Just start the backend - predictions will be created in PENDING state
start_backend.bat
```

**Note:** In this mode, predictions will be created but won't process automatically. You'll need to start Redis and Celery later to process them.

---

## 📋 Prerequisites

### Required
- **Python 3.8-3.12** (BioPython wheels available)
- **Virtual Environment** (recommended)

### Optional (for full functionality)
- **Redis** - For async task queue
  - Windows: https://github.com/microsoftarchive/redis/releases
  - Docker: `docker run -d -p 6379:6379 redis:alpine`
- **PostgreSQL** - For production database (development uses SQLite fallback)

---

## 🔧 Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and update:

```bash
copy .env.example .env
```

Edit `.env`:
```env
# Use SQLite for development (no PostgreSQL needed)
DATABASE_URL=sqlite:///./pp_dev.db

# Or use PostgreSQL
# DATABASE_URL=postgresql://user:password@localhost:5432/pp_db

# Redis (only needed if using Celery)
REDIS_URL=redis://localhost:6379/0
```

---

## 🏃 Running the Server

### Method 1: Using Batch Files (Recommended)

```bash
# Start backend with Socket.IO support
start_backend.bat
```

This runs: `uvicorn wsgi:socket_app --reload --host 0.0.0.0 --port 8000`

### Method 2: Direct Command

```bash
# Activate virtual environment first
venv\Scripts\activate

# Run with uvicorn
uvicorn wsgi:socket_app --reload --host 0.0.0.0 --port 8000
```

### Method 3: Python Module

```bash
python -m app.main
```

---

## 🔌 WebSocket/Socket.IO Setup

The backend uses **Socket.IO for real-time updates**. The configuration is:

- **Socket.IO path**: `/socket.io` (standard)
- **CORS**: Allows `localhost:3000` and `localhost:5173`
- **Transports**: WebSocket (primary) + polling (fallback)

### Frontend Connection
The frontend connects via `websocketService.ts`:
```typescript
const socket = io('http://localhost:8000', {
  path: '/socket.io',
  transports: ['websocket', 'polling']
});
```

### Events Available
- `progress_update` - Real-time progress updates
- `metrics_update` - Performance metrics
- `agent_update` - Multi-agent status
- `event_log` - Event logs
- `status_change` - Status changes
- `prediction_complete` - Completion notification
- `prediction_error` - Error notification

---

## 📊 Database Setup

### Development (SQLite - Default)
No setup needed! Database is created automatically.

```env
DATABASE_URL=sqlite:///./pp_dev.db
```

### Production (PostgreSQL)

1. Install PostgreSQL
2. Create database:
   ```sql
   CREATE DATABASE pp_db;
   CREATE USER pp_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE pp_db TO pp_user;
   ```
3. Update `.env`:
   ```env
   DATABASE_URL=postgresql://pp_user:your_password@localhost:5432/pp_db
   ```

---

## 🔥 Redis & Celery (Optional but Recommended)

### Why Redis & Celery?
- **Async processing**: Long-running protein predictions don't block API
- **Real-time updates**: Progress updates via WebSocket while processing
- **Scalability**: Process multiple predictions in parallel

### Redis Installation

#### Option A: Native Windows
1. Download: https://github.com/microsoftarchive/redis/releases
2. Extract to `C:\Redis` or `C:\Program Files\Redis`
3. Run: `redis-server.exe`

#### Option B: Docker
```bash
docker run -d -p 6379:6379 --name redis redis:alpine
```

#### Option C: WSL2
```bash
wsl
sudo apt-get install redis-server
redis-server
```

### Verify Redis
```bash
# Test connection
redis-cli ping
# Should return: PONG
```

### Start Celery Worker
```bash
start_celery.bat
```

This runs: `celery -A app.celery_app:celery_app worker --loglevel=info --pool=solo`

---

## 🧪 Testing

### Test API Endpoints
```bash
# Activate venv
venv\Scripts\activate

# Run tests
pytest tests/ -v
```

### Manual API Testing

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Create Prediction
```bash
curl -X POST http://localhost:8000/api/predictions \
  -H "Content-Type: application/json" \
  -d "{\"sequence\": \"ACDEFGH\", \"configuration\": {\"iterations\": 100}}"
```

#### List Predictions
```bash
curl http://localhost:8000/api/predictions
```

#### Get Prediction Details
```bash
curl http://localhost:8000/api/predictions/{prediction_id}
```

---

## 🐛 Troubleshooting

### Socket.IO Connection Fails

**Symptom:** Frontend shows "WebSocket connection failed"

**Solutions:**
1. Ensure backend is running with `wsgi:socket_app`
   ```bash
   uvicorn wsgi:socket_app --reload
   ```
2. Check CORS settings in `app/main.py`
3. Verify frontend is using correct URL and path:
   ```typescript
   io('http://localhost:8000', { path: '/socket.io' })
   ```

### Predictions Stay in PENDING

**Symptom:** Predictions created but never process

**Solutions:**
1. Start Redis:
   ```bash
   start_redis.bat
   ```
2. Start Celery worker:
   ```bash
   start_celery.bat
   ```
3. Check logs for errors

### Database Connection Failed

**Symptom:** "Database connection failed" in logs

**Solutions:**
1. **For development**: Use SQLite (automatic fallback)
   ```env
   DATABASE_URL=sqlite:///./pp_dev.db
   ```
2. **For PostgreSQL**: Verify credentials and service is running
   ```bash
   # Check PostgreSQL status
   pg_isready
   ```

### Import Errors

**Symptom:** "ModuleNotFoundError" or import errors

**Solutions:**
1. Activate virtual environment:
   ```bash
   venv\Scripts\activate
   ```
2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Check Python version (3.8-3.12 required)

### BioPython Installation Fails (Python 3.13+)

**Symptom:** BioPython won't install on Python 3.13

**Solutions:**
1. Use Python 3.12 (recommended)
2. Install C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
3. Or use Conda: `conda install -c conda-forge biopython`

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── api/              # API endpoints
│   │   ├── predictions.py
│   │   ├── campaigns.py
│   │   └── results.py
│   ├── models/           # Database models
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic
│   ├── tasks/            # Celery tasks
│   ├── websocket/        # Socket.IO configuration
│   ├── config.py         # Configuration
│   ├── database.py       # Database setup
│   └── main.py           # FastAPI app
├── tests/                # Test suite
├── wsgi.py              # ASGI entry point (with Socket.IO)
├── celery_app.py        # Celery configuration
├── requirements.txt      # Python dependencies
├── .env.example         # Environment template
└── start_*.bat          # Startup scripts
```

---

## 🎯 Next Steps

1. **Start Basic Backend**: `start_backend.bat`
2. **Test API**: Visit http://localhost:8000/docs (Swagger UI)
3. **Add Redis + Celery**: For production-like setup
4. **Configure Frontend**: Point to `http://localhost:8000`

---

## 📞 Support

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Logs**: Check console output for errors

---

## 🔑 Key URLs

- **API Base**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health**: http://localhost:8000/health
- **Socket.IO**: ws://localhost:8000/socket.io

---

## ✅ Success Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] `.env` file configured
- [ ] Backend starts without errors
- [ ] Health check returns 200
- [ ] Can create predictions via API
- [ ] (Optional) Redis running
- [ ] (Optional) Celery worker running
- [ ] Frontend connects to WebSocket

**You're ready to go! 🚀**
