@echo off
REM Backup script for Protein Predictor (Windows)

setlocal enabledelayedexpansion

REM Configuration
set BACKUP_DIR=backups
set COMPOSE_FILE=docker-compose.prod.yml

REM Create timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%

REM Create backup directory
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

echo [INFO] Starting backup at %TIMESTAMP%

REM Backup PostgreSQL database
echo [INFO] Backing up PostgreSQL database...
docker-compose -f %COMPOSE_FILE% exec -T postgres pg_dumpall -U postgres > "%BACKUP_DIR%\postgres_%TIMESTAMP%.sql"

if %ERRORLEVEL% EQU 0 (
    echo [INFO] PostgreSQL backup completed: postgres_%TIMESTAMP%.sql
) else (
    echo [ERROR] PostgreSQL backup failed
    exit /b 1
)

REM Backup Redis data
echo [INFO] Backing up Redis data...
docker-compose -f %COMPOSE_FILE% exec -T redis redis-cli BGSAVE
timeout /t 5 /nobreak >nul

for /f %%i in ('docker-compose -f %COMPOSE_FILE% ps -q redis') do set REDIS_CONTAINER=%%i
docker cp %REDIS_CONTAINER%:/data/dump.rdb "%BACKUP_DIR%\redis_%TIMESTAMP%.rdb"

if %ERRORLEVEL% EQU 0 (
    echo [INFO] Redis backup completed: redis_%TIMESTAMP%.rdb
) else (
    echo [WARN] Redis backup failed (non-critical)
)

REM Backup PDB cache
if exist "pdb_cache" (
    echo [INFO] Backing up PDB cache...
    tar -czf "%BACKUP_DIR%\pdb_cache_%TIMESTAMP%.tar.gz" pdb_cache
    echo [INFO] PDB cache backup completed
)

REM Backup checkpoints
if exist "checkpoints" (
    echo [INFO] Backing up checkpoints...
    tar -czf "%BACKUP_DIR%\checkpoints_%TIMESTAMP%.tar.gz" checkpoints
    echo [INFO] Checkpoints backup completed
)

REM Backup configuration files
echo [INFO] Backing up configuration files...
tar -czf "%BACKUP_DIR%\config_%TIMESTAMP%.tar.gz" .env.production docker-compose.prod.yml docker\nginx docker\redis docker\logging

echo [INFO] Configuration backup completed

echo.
echo [INFO] Backup completed successfully at %date% %time%
echo.

dir /B "%BACKUP_DIR%\*%TIMESTAMP%*"

pause
exit /b 0
