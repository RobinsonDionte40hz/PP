@echo off
REM Start FastAPI backend server

echo Starting FastAPI backend...
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "myvenv\Scripts\activate.bat" (
    call myvenv\Scripts\activate.bat
)

REM Start Uvicorn server with Socket.IO wrapped app
uvicorn wsgi:socket_app --reload --host 0.0.0.0 --port 8000

pause
