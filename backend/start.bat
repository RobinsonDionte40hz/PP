@echo off
REM Simple startup check and launcher for Protein Prediction Platform

echo ========================================
echo  Protein Prediction Platform Backend
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    if not exist ".venv\Scripts\activate.bat" (
        if not exist "myvenv\Scripts\activate.bat" (
            echo [ERROR] No virtual environment found!
            echo.
            echo Please create one first:
            echo    python -m venv venv
            echo    venv\Scripts\activate
            echo    pip install -r requirements.txt
            echo.
            pause
            exit /b 1
        )
    )
)

REM Activate virtual environment
echo [1/4] Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "myvenv\Scripts\activate.bat" (
    call myvenv\Scripts\activate.bat
)
echo      ✓ Virtual environment activated
echo.

REM Check if .env exists
echo [2/4] Checking configuration...
if not exist ".env" (
    echo      ⚠ Warning: .env file not found
    echo      Creating from .env.example...
    copy .env.example .env
    echo      ✓ Created .env file - please review it
) else (
    echo      ✓ Configuration found
)
echo.

REM Check Redis (optional)
echo [3/4] Checking optional services...
redis-cli ping >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo      ✓ Redis is running - async processing enabled
    set REDIS_STATUS=RUNNING
) else (
    echo      ⚠ Redis not running - predictions will be pending
    echo        Start with: start_redis.bat
    set REDIS_STATUS=NOT_RUNNING
)
echo.

REM Start the backend
echo [4/4] Starting backend server...
echo.
echo ========================================
echo  Server Configuration
echo ========================================
echo  • URL: http://localhost:8000
echo  • Docs: http://localhost:8000/docs
echo  • Socket.IO: ws://localhost:8000/socket.io
echo  • Redis: %REDIS_STATUS%
echo ========================================
echo.

if "%REDIS_STATUS%"=="NOT_RUNNING" (
    echo [INFO] Running in development mode:
    echo        - API endpoints work normally
    echo        - Predictions created but not processed
    echo        - Start Redis + Celery for full functionality
    echo.
)

echo Starting Uvicorn server...
echo.

uvicorn wsgi:socket_app --reload --host 0.0.0.0 --port 8000

pause
