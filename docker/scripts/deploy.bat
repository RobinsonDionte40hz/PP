@echo off
REM Production deployment script for Protein Predictor (Windows)

setlocal enabledelayedexpansion

echo =========================================
echo   Protein Predictor - Production Deploy
echo =========================================
echo.

REM Configuration
set COMPOSE_FILE=docker-compose.prod.yml
set ENV_FILE=.env.production

REM Check prerequisites
echo [STEP] Checking prerequisites...

where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    exit /b 1
)

where docker-compose >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Compose is not installed.
    exit /b 1
)

docker --version
docker-compose --version

REM Check environment file
if not exist "%ENV_FILE%" (
    echo [ERROR] Environment file not found: %ENV_FILE%
    echo [ERROR] Please copy .env.production.example to .env.production and configure it.
    exit /b 1
)

echo [INFO] Environment configuration found

REM Check SSL certificates
echo [STEP] Checking SSL certificates...

if not exist "docker\nginx\ssl\cert.pem" (
    echo [WARN] SSL certificates not found
    echo [INFO] Generating self-signed certificates...
    call docker\nginx\ssl\generate-self-signed-cert.bat
)

echo [INFO] SSL certificates found

REM Create required directories
echo [STEP] Creating required directories...

if not exist "logs\app" mkdir logs\app
if not exist "logs\nginx" mkdir logs\nginx
if not exist "logs\celery" mkdir logs\celery
if not exist "backups" mkdir backups
if not exist "pdb_cache" mkdir pdb_cache
if not exist "checkpoints" mkdir checkpoints
if not exist "prediction_results" mkdir prediction_results

echo [INFO] Directories created

REM Pull latest images
echo [STEP] Pulling Docker images...
docker-compose -f %COMPOSE_FILE% pull

REM Build custom images
echo [STEP] Building application images...
docker-compose -f %COMPOSE_FILE% build --no-cache

REM Stop existing containers (if any)
docker-compose -f %COMPOSE_FILE% ps -q >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [STEP] Stopping existing containers...
    docker-compose -f %COMPOSE_FILE% down
)

REM Start services
echo [STEP] Starting services...
docker-compose -f %COMPOSE_FILE% up -d

REM Wait for services to be healthy
echo [STEP] Waiting for services to become healthy...
timeout /t 30 /nobreak >nul

REM Display service status
echo [STEP] Service Status:
docker-compose -f %COMPOSE_FILE% ps

REM Display service URLs
echo.
echo [INFO] Deployment completed successfully!
echo.
echo Services are available at:
echo   - Frontend: https://localhost
echo   - Backend API: https://localhost/api
echo   - API Docs: https://localhost/api/docs
echo.
echo To view logs:
echo   docker-compose -f %COMPOSE_FILE% logs -f
echo.
echo To stop services:
echo   docker-compose -f %COMPOSE_FILE% down
echo.

pause
exit /b 0
