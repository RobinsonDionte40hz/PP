@echo off
REM Run database migrations for user authentication
echo Running database migrations...
cd backend
python -m app.migrations.create_users_table
echo.
echo Migration complete!
pause
