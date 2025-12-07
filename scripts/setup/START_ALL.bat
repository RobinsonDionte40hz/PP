@echo off
REM Start all services for Protein Prediction Platform

echo ========================================
echo Protein Prediction Platform
echo Starting All Services
echo ========================================
echo.

REM Start Redis
echo [1/4] Starting Redis...
start "Redis Server" cmd /k "cd backend && start_redis.bat"
timeout /t 3 /nobreak >nul

REM Start Celery Worker
echo [2/4] Starting Celery Worker...
start "Celery Worker" cmd /k "cd backend && start_celery.bat"
timeout /t 3 /nobreak >nul

REM Start Backend API
echo [3/4] Starting Backend API...
start "Backend API" cmd /k "cd backend && start_backend.bat"
timeout /t 5 /nobreak >nul

REM Start Frontend
echo [4/4] Starting Frontend...
start "Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo All services starting...
echo.
echo Backend API:  http://localhost:8000
echo Frontend:     http://localhost:5173
echo Redis:        localhost:6379
echo.
echo Press any key to stop all services...
pause >nul

REM Cleanup - this won't work well for started processes, but informational
echo.
echo To stop services, close the individual terminal windows
echo or use Task Manager to stop:
echo - node.exe (Frontend)
echo - python.exe (Backend, Celery)
echo - redis-server.exe (Redis)
