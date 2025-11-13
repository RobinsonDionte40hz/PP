# Running the Full Application Stack

## Quick Start

### All-in-One (Windows)
```bash
START_ALL.bat
```

### Step-by-Step

**1. Start Redis**
```bash
docker run -d -p 6379:6379 --name redis redis:alpine
```

**2. Start Celery Worker**
```bash
cd backend
celery -A celery_app worker --loglevel=info --pool=solo
```

**3. Start Backend API**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

**4. Start Frontend**
```bash
cd frontend
npm run dev
```

## Access Points

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

## Recent Fixes Applied

✅ **CORS Issue** - Added `http://localhost:5173` to allowed origins  
✅ **API Response Format** - Fixed paginated predictions endpoint  
✅ **Type Imports** - Fixed component import paths  
✅ **Request Schema** - Aligned frontend data with backend expectations  
✅ **Celery Setup** - Proper error handling for task queue  

## Current Requirements

To create predictions, you **must** have:
1. ✅ Backend API running (port 8000)
2. ✅ Frontend running (port 5173)
3. ⚠️ **Redis running** (port 6379) - **REQUIRED**
4. ⚠️ **Celery worker running** - **REQUIRED**

## Troubleshooting

### "Task queue unavailable" Error

This means Redis or Celery worker is not running:

```bash
# Check Redis
docker ps | grep redis
# or
redis-cli ping

# Start Celery worker in backend directory
cd backend
celery -A celery_app worker --loglevel=info --pool=solo
```

### CORS Errors

Backend must be restarted after CORS configuration changes.

### Import Errors

Check that all components exist:
- `backend/celery_app.py` ✅
- `backend/app/tasks/` ✅
- `frontend/src/components/dashboard/` ✅

## Architecture

```
┌─────────────┐
│  Frontend   │ (React + Vite)
│   :5173     │
└──────┬──────┘
       │ HTTP/WebSocket
       ▼
┌─────────────┐
│  Backend    │ (FastAPI)
│   :8000     │
└──────┬──────┘
       │ Task Queue
       ▼
┌─────────────┐      ┌─────────────┐
│   Celery    │─────▶│   Redis     │
│   Worker    │      │   :6379     │
└─────────────┘      └─────────────┘
       │
       ▼
┌─────────────┐
│ UBF Protein │ (Prediction Engine)
│   System    │
└─────────────┘
```

See `backend/SETUP_GUIDE.md` for detailed setup instructions.
