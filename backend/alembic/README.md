# Database Migrations

This directory contains Alembic database migrations for the session-based storage system.

## Setup

1. Install Alembic (if not already installed):
```bash
pip install alembic
```

2. Initialize the database (first time only):
```bash
cd backend
alembic upgrade head
```

## Running Migrations

### Apply all pending migrations
```bash
alembic upgrade head
```

### Rollback last migration
```bash
alembic downgrade -1
```

### Rollback all migrations
```bash
alembic downgrade base
```

### View migration history
```bash
alembic history
```

### View current database version
```bash
alembic current
```

## Creating New Migrations

### Auto-generate migration from model changes
```bash
alembic revision --autogenerate -m "description of changes"
```

### Create empty migration file
```bash
alembic revision -m "description of changes"
```

## Migration 001: Add Work Sessions

This migration adds the session-based storage system:

### Tables Created
- **work_sessions**: Stores user work sessions for organizing predictions
  - `id` (String, PK): Unique session identifier
  - `user_id` (String, FK to users.key_id): Owner of the session
  - `name` (String): User-provided session name
  - `created_at` (DateTime): Session creation timestamp
  - `updated_at` (DateTime): Last update timestamp
  - `last_active_at` (DateTime): Last activity timestamp

- **shared_exports**: Stores public share links for sessions
  - `share_id` (String, PK): Unique share identifier
  - `session_id` (String, FK to work_sessions.id): Session being shared
  - `created_at` (DateTime): Share creation timestamp
  - `expires_at` (DateTime): Share expiration timestamp
  - `access_count` (Integer): Number of times accessed
  - `last_accessed_at` (DateTime): Last access timestamp

### Columns Added
- **predictions.session_id** (String, FK to work_sessions.id, nullable): Links prediction to session

### Indexes Created
- work_sessions: id, user_id, (user_id, last_active_at), (user_id, created_at)
- shared_exports: share_id, session_id, (session_id, expires_at), expires_at
- predictions: session_id, (session_id, created_at), (session_id, status)

### Foreign Key Relationships
- work_sessions.user_id → users.key_id (CASCADE DELETE)
- shared_exports.session_id → work_sessions.id (CASCADE DELETE)
- predictions.session_id → work_sessions.id (CASCADE DELETE)

## Testing Migrations

### Test upgrade
```bash
# Apply migration
alembic upgrade head

# Verify tables exist
python -c "from app.database import engine; from sqlalchemy import inspect; print(inspect(engine).get_table_names())"
```

### Test downgrade
```bash
# Rollback migration
alembic downgrade -1

# Verify tables removed
python -c "from app.database import engine; from sqlalchemy import inspect; print(inspect(engine).get_table_names())"
```

### Test complete cycle
```bash
# Start fresh
alembic downgrade base

# Apply all migrations
alembic upgrade head

# Verify schema
python verify_schema.py
```

## Troubleshooting

### "No such table" errors
Make sure you've run migrations:
```bash
alembic upgrade head
```

### "Constraint already exists" errors
Check current database version:
```bash
alembic current
```

If out of sync, you may need to stamp the database:
```bash
alembic stamp head
```

### Migration conflicts
If you have local changes conflicting with migrations:
```bash
# Rollback to base
alembic downgrade base

# Delete database file (development only!)
rm pp_dev.db

# Reapply migrations
alembic upgrade head
```

## Notes

- Migrations are applied sequentially based on revision IDs
- Always test migrations in development before applying to production
- Backup your database before running migrations in production
- The `session_id` column in predictions is nullable for backward compatibility
