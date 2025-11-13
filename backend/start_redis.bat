@echo off
REM Start Redis for Windows
REM Download Redis from: https://github.com/microsoftarchive/redis/releases

echo Starting Redis server...
echo.
echo If Redis is not installed, download from:
echo https://github.com/microsoftarchive/redis/releases
echo.

REM Try to find Redis in common locations
if exist "C:\Program Files\Redis\redis-server.exe" (
    start "Redis Server" "C:\Program Files\Redis\redis-server.exe"
    echo Redis started on localhost:6379
) else if exist "C:\Redis\redis-server.exe" (
    start "Redis Server" "C:\Redis\redis-server.exe"
    echo Redis started on localhost:6379
) else if exist "%USERPROFILE%\Redis\redis-server.exe" (
    start "Redis Server" "%USERPROFILE%\Redis\redis-server.exe"
    echo Redis started on localhost:6379
) else (
    echo Redis not found in common locations.
    echo.
    echo Please install Redis for Windows:
    echo 1. Download from: https://github.com/microsoftarchive/redis/releases
    echo 2. Extract to C:\Redis or add to PATH
    echo 3. Run redis-server.exe
    echo.
    echo Alternative: Use Docker:
    echo    docker run -d -p 6379:6379 redis:alpine
    pause
)
