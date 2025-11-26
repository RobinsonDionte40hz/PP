@echo off
REM Database migration script for Windows

echo ========================================
echo Database Migration Tool
echo ========================================
echo.

cd /d "%~dp0"

if "%1"=="" goto usage
if "%1"=="upgrade" goto upgrade
if "%1"=="downgrade" goto downgrade
if "%1"=="current" goto current
if "%1"=="history" goto history
goto usage

:upgrade
echo Applying migrations...
alembic upgrade head
if %errorlevel% neq 0 (
    echo Migration failed!
    exit /b 1
)
echo Migration completed successfully!
goto end

:downgrade
if "%2"=="" (
    echo Rolling back last migration...
    alembic downgrade -1
) else (
    echo Rolling back to: %2
    alembic downgrade %2
)
if %errorlevel% neq 0 (
    echo Rollback failed!
    exit /b 1
)
echo Rollback completed successfully!
goto end

:current
echo Current database version:
alembic current
goto end

:history
echo Migration history:
alembic history
goto end

:usage
echo Usage: migrate.bat [command] [options]
echo.
echo Commands:
echo   upgrade          Apply all pending migrations
echo   downgrade [rev]  Rollback to specified revision (or -1 for last)
echo   current          Show current database version
echo   history          Show migration history
echo.
echo Examples:
echo   migrate.bat upgrade
echo   migrate.bat downgrade
echo   migrate.bat downgrade base
echo   migrate.bat current
echo   migrate.bat history
goto end

:end
