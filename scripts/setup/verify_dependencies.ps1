# Dependency Verification Script for PP Production Deployment (PowerShell)
# Verifies all required packages are installed in each container

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "PP Dependency Verification" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

function Check-Container {
    param(
        [string]$Container,
        [string]$Package
    )
    
    $result = docker exec $Container pip list 2>$null | Select-String $Package
    if ($result) {
        Write-Host "✓ " -ForegroundColor Green -NoNewline
        Write-Host "$Container`: $Package installed"
        return $true
    } else {
        Write-Host "✗ " -ForegroundColor Red -NoNewline
        Write-Host "$Container`: $Package NOT installed"
        return $false
    }
}

Write-Host "Checking Backend Container..." -ForegroundColor Yellow
Write-Host "------------------------------"
Check-Container "pp-backend-1" "fastapi"
Check-Container "pp-backend-1" "uvicorn"
Check-Container "pp-backend-1" "celery"
Check-Container "pp-backend-1" "redis"
Check-Container "pp-backend-1" "sqlalchemy"
Check-Container "pp-backend-1" "psycopg2-binary"
Check-Container "pp-backend-1" "python-socketio"
Check-Container "pp-backend-1" "httpx"
Write-Host ""

Write-Host "Checking Worker Container..." -ForegroundColor Yellow
Write-Host "-----------------------------"
Check-Container "pp-worker-1" "celery"
Check-Container "pp-worker-1" "redis"
Check-Container "pp-worker-1" "sqlalchemy"
Check-Container "pp-worker-1" "psycopg2-binary"
Check-Container "pp-worker-1" "httpx"
Check-Container "pp-worker-1" "numpy"
Check-Container "pp-worker-1" "scipy"
Check-Container "pp-worker-1" "biopython"
Check-Container "pp-worker-1" "matplotlib"
Check-Container "pp-worker-1" "pandas"
Write-Host ""

Write-Host "Checking Database Connectivity..." -ForegroundColor Yellow
Write-Host "----------------------------------"
docker exec pp-backend-1 python -c "from app.database import engine; engine.connect()" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Backend can connect to PostgreSQL" -ForegroundColor Green
} else {
    Write-Host "✗ Backend CANNOT connect to PostgreSQL" -ForegroundColor Red
}

docker exec pp-worker-1 python -c "from app.database import engine; engine.connect()" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Worker can connect to PostgreSQL" -ForegroundColor Green
} else {
    Write-Host "✗ Worker CANNOT connect to PostgreSQL" -ForegroundColor Red
}
Write-Host ""

Write-Host "Checking Redis Connectivity..." -ForegroundColor Yellow
Write-Host "-------------------------------"
docker exec pp-backend-1 python -c "import redis; r=redis.from_url('redis://redis:6379/0'); r.ping()" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Backend can connect to Redis" -ForegroundColor Green
} else {
    Write-Host "✗ Backend CANNOT connect to Redis" -ForegroundColor Red
}

docker exec pp-worker-1 python -c "import redis; r=redis.from_url('redis://redis:6379/0'); r.ping()" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Worker can connect to Redis" -ForegroundColor Green
} else {
    Write-Host "✗ Worker CANNOT connect to Redis" -ForegroundColor Red
}
Write-Host ""

Write-Host "Checking Celery Worker Status..." -ForegroundColor Yellow
Write-Host "---------------------------------"
$celeryOutput = docker exec pp-worker-1 celery -A celery_app inspect ping 2>&1
if ($celeryOutput -match "pong") {
    Write-Host "✓ Celery worker is responsive" -ForegroundColor Green
} else {
    Write-Host "⚠ Celery worker may not be running" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "Checking Container Health..." -ForegroundColor Yellow
Write-Host "-----------------------------"
$containers = @("pp-backend-1", "pp-worker-1", "pp-redis-1", "pp-postgres-1")
foreach ($container in $containers) {
    $running = docker inspect -f "{{.State.Running}}" $container 2>&1
    if ($running -eq "true") {
        Write-Host "✓ $container is running" -ForegroundColor Green
    } else {
        Write-Host "✗ $container is NOT running" -ForegroundColor Red
    }
}
Write-Host ""

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Verification Complete" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

