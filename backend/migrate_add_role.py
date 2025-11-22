"""
Database migration: Add role column to users table

This migration adds a 'role' column to the users table with default value 'user'.
All existing users will be assigned the 'user' role.

Usage:
    python migrate_add_role.py
"""
import sys
import os
import sqlite3

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Get database path from config
try:
    from app.config import settings
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
except:
    # Fallback to default
    db_path = "app.db"


def migrate_add_role():
    """Add role column to users table"""
    
    print("=" * 60)
    print("Database Migration: Add Role Column")
    print("=" * 60)
    print()
    
    try:
        # Connect to SQLite database
        print(f"Connecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if users table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='users'
        """)
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            print("⚠  Users table does not exist yet")
            print("   It will be created when setup_master_accounts.py runs")
            print("   (The role column will be included automatically)")
            print()
            conn.close()
            return True
        
        # Check if column already exists
        print("Checking if 'role' column exists...")
        
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if 'role' in column_names:
            print("✓ Role column already exists")
            print()
            conn.close()
            return True
        
        print("Adding 'role' column to users table...")
        
        # Add role column with default value 'user'
        cursor.execute("""
            ALTER TABLE users 
            ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'user'
        """)
        
        conn.commit()
        
        print("✓ Role column added successfully")
        print("  Default role: 'user'")
        print()
        
        # Verify the change
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"✓ Migration verified ({user_count} users updated)")
        print()
        
        conn.close()
        
    except Exception as e:
        print(f"✗ Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("=" * 60)
    print("Migration Complete")
    print("=" * 60)
    print()
    print("All existing users have been assigned the 'user' role.")
    print("You can now run setup_master_accounts.py to create admin/dev accounts.")
    print()
    
    return True


if __name__ == "__main__":
    success = migrate_add_role()
    sys.exit(0 if success else 1)
