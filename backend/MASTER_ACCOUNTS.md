# EmergentFolds Master Accounts Setup

This directory contains scripts to set up master accounts for testing and administration.

## Quick Setup

Run the following command in the `backend` directory:

```bash
setup_master.bat
```

This will:
1. Add the `role` column to the users table (if not exists)
2. Create admin and developer accounts

## Master Credentials

### Admin Account
- **Username:** `admin`
- **Password:** `Admin@2025!`
- **Role:** `admin`
- **Email:** `admin@emergentfolds.local`
- **Permissions:** Full system access, user management, system configuration

### Developer Account
- **Username:** `developer`
- **Password:** `Dev@2025!`
- **Role:** `developer`
- **Email:** `dev@emergentfolds.local`
- **Permissions:** Testing features, API access, debugging tools

## Manual Setup

If you prefer to run the scripts individually:

### 1. Run Database Migration
```bash
python migrate_add_role.py
```

This adds the `role` column to the users table.

### 2. Create Master Accounts
```bash
python setup_master_accounts.py
```

This creates the admin and developer accounts.

## Role System

The system now supports three roles:

- **`user`** - Standard user with basic access (default)
- **`developer`** - Developer with extended access for testing
- **`admin`** - Administrator with full system access

## Security Notes

⚠️ **IMPORTANT:** These are test accounts with well-known passwords. 

**Production Security Checklist:**
- [ ] Change all master account passwords
- [ ] Use strong, unique passwords
- [ ] Enable 2FA/MFA if available
- [ ] Restrict admin access to trusted IPs
- [ ] Regularly audit admin/developer account usage
- [ ] Consider disabling these accounts in production

## Checking Existing Accounts

To verify the accounts were created, check your database:

```bash
# Using SQLite CLI
sqlite3 app.db "SELECT username, role, email FROM users WHERE role IN ('admin', 'developer');"
```

## Resetting Master Accounts

If you need to reset the master accounts:

1. Delete the existing accounts from the database
2. Run `setup_master_accounts.py` again

Or manually update passwords in the database.

## Troubleshooting

### "Role column already exists"
This is normal if you've already run the migration. The script will skip this step.

### "Account already exists"
The accounts have already been created. Use the existing credentials or delete them from the database first.

### Database connection errors
Make sure your database is running and the connection string in `.env` is correct.

## Using Master Accounts

These accounts can be used to:
- Log in to the frontend application
- Test authentication flows
- Access admin-only features (when implemented)
- Debug and troubleshoot issues
- Perform integration testing

## Integration with Frontend

The frontend will automatically recognize user roles and can show/hide features accordingly. Check the user object returned from login:

```javascript
{
  "user": {
    "username": "admin",
    "role": "admin",
    "email": "admin@emergentfolds.local",
    ...
  }
}
```
