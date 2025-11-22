@echo off
REM Setup Master Accounts for WeFold
REM This script will:
REM 1. Run database migration to add role column
REM 2. Create admin and developer test accounts

echo ============================================================
echo WeFold Master Account Setup
echo ============================================================
echo.

echo Step 1: Running database migration...
python migrate_add_role.py
if %ERRORLEVEL% NEQ 0 (
    echo Migration failed! Please check the error above.
    pause
    exit /b 1
)

echo.
echo Step 2: Creating master accounts...
python setup_master_accounts.py
if %ERRORLEVEL% NEQ 0 (
    echo Account creation failed! Please check the error above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo You can now log in with:
echo.
echo   Admin Account:
echo     Username: admin
echo     Password: Admin@2025!
echo.
echo   Developer Account:
echo     Username: developer
echo     Password: Dev@2025!
echo.
echo WARNING: Change these passwords in production!
echo ============================================================
echo.
pause
