@echo off
REM Docker Deployment Test Script
REM Tests complete Docker stack deployment on clean system

echo ========================================
echo PP Docker Deployment Test
echo ========================================
echo.

REM Check Docker availability
echo [1/8] Checking Docker installation...
docker --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH
    exit /b 1
)
echo [OK] Docker found

echo.
echo [2/8] Checking Docker Compose...
docker compose version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker Compose is not available
    exit /b 1
)
echo [OK] Docker Compose found

echo.
echo [3/8] Checking .env file...
if not exist ".env" (
    echo [WARN] .env file not found, copying from .env.example
    copy .env.example .env
)
echo [OK] Environment configuration exists

echo.
echo [4/8] Stopping existing containers...
docker compose down >nul 2>&1
echo [OK] Clean slate ready

echo.
echo [5/8] Building Docker images...
docker compose build
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker build failed
    exit /b 1
)
echo [OK] Images built successfully

echo.
echo [6/8] Starting containers...
docker compose up -d
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to start containers
    exit /b 1
)
echo [OK] Containers started

echo.
echo [7/8] Waiting for services to be healthy (30 seconds)...
timeout /t 30 /nobreak >nul

echo.
echo [8/8] Checking service health...
docker compose ps

echo.
echo ========================================
echo Testing individual services...
echo ========================================
echo.

REM Test Redis
echo [Redis] Testing connection...
docker compose exec -T redis redis-cli ping >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Redis is responding
) else (
    echo [WARN] Redis health check failed
)

REM Test PostgreSQL
echo [PostgreSQL] Testing connection...
docker compose exec -T postgres pg_isready >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] PostgreSQL is responding
) else (
    echo [WARN] PostgreSQL health check failed
)

REM Test Backend
echo [Backend] Testing health endpoint...
curl -f http://localhost:8000/health >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Backend health check passed
) else (
    echo [WARN] Backend health check failed
)

REM Test Frontend (if running on port 3000)
echo [Frontend] Testing availability...
curl -f http://localhost:3000 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] Frontend is accessible
) else (
    echo [WARN] Frontend not accessible (may not be built yet)
)

echo.
echo ========================================
echo Deployment Test Summary
echo ========================================
echo.
echo Run 'docker compose ps' to see container status
echo Run 'docker compose logs -f' to see live logs
echo Run 'docker compose down' to stop all services
echo.
echo To run E2E tests, execute:
echo   python tests\e2e_test.py --url http://localhost:8000
echo.

pause
