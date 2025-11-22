# Database Migrations

This directory contains database migration scripts for the PP application.

## Running Migrations

### Automatic (Windows)
From the project root, run:
```bash
run_migration.bat
```

### Manual
From the project root, run:
```bash
cd backend
python -m app.migrations.create_users_table
```

## Migration Scripts

### `create_users_table.py`
Creates the users table for authentication with the following schema:

**Columns:**
- `key_id` (String(36), PRIMARY KEY) - UUID v4 identifier
- `username` (String(150), UNIQUE, NOT NULL) - User's unique username
- `email` (String(255), UNIQUE, NULLABLE) - User's email address
- `password_hash` (String(255), NOT NULL) - Bcrypt hashed password
- `is_active` (Boolean, NOT NULL, DEFAULT TRUE) - Account active status
- `created_at` (DateTime, NOT NULL) - Account creation timestamp
- `updated_at` (DateTime, NOT NULL) - Last update timestamp
- `last_login` (DateTime, NULLABLE) - Last login timestamp

**Indexes:**
- Primary key index on `key_id`
- Unique index on `username`
- Unique index on `email`
- Composite index on `(username, is_active)` for efficient active user lookups
- Composite index on `(email, is_active)` for efficient email-based queries

**Features:**
- Checks if table exists before creating (idempotent)
- Automatically creates all indexes
- Provides detailed logging of migration progress
- Safe to run multiple times

## Database Schema Evolution

When adding new tables or modifying existing ones:

1. Create a new migration script in this directory (e.g., `add_session_table.py`)
2. Follow the pattern in `create_users_table.py`
3. Always check if changes already exist (idempotent migrations)
4. Update this README with the new migration details
5. Test the migration on a development database first

## Best Practices

- **Always backup** production databases before running migrations
- **Test migrations** on development/staging environments first
- **Make migrations idempotent** (safe to run multiple times)
- **Use transactions** for complex multi-step migrations
- **Log everything** to help with debugging
- **Document changes** in this README
