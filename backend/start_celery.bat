@echo off
REM Start Celery worker for Windows

echo Starting Celery worker...
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "myvenv\Scripts\activate.bat" (
    call myvenv\Scripts\activate.bat
)

REM Start Celery worker
celery -A celery_app worker --loglevel=info --pool=solo

pause
