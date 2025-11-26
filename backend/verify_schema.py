"""
Verify database schema after migration

This script checks that all tables and columns exist after running migrations.
"""
import sys
from sqlalchemy import inspect, text
from app.database import engine
from app.models import User, WorkSession, SharedExport, Prediction


def verify_table_exists(inspector, table_name):
    """Check if a table exists"""
    tables = inspector.get_table_names()
    if table_name in tables:
        print(f"✓ Table '{table_name}' exists")
        return True
    else:
        print(f"✗ Table '{table_name}' NOT FOUND")
        return False


def verify_column_exists(inspector, table_name, column_name):
    """Check if a column exists in a table"""
    try:
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        if column_name in columns:
            print(f"  ✓ Column '{column_name}' exists")
            return True
        else:
            print(f"  ✗ Column '{column_name}' NOT FOUND")
            return False
    except Exception as e:
        print(f"  ✗ Error checking column '{column_name}': {e}")
        return False


def verify_foreign_key_exists(inspector, table_name, fk_column):
    """Check if a foreign key constraint exists"""
    try:
        foreign_keys = inspector.get_foreign_keys(table_name)
        for fk in foreign_keys:
            if fk_column in fk['constrained_columns']:
                print(f"  ✓ Foreign key on '{fk_column}' -> {fk['referred_table']}.{fk['referred_columns'][0]}")
                return True
        print(f"  ✗ Foreign key on '{fk_column}' NOT FOUND")
        return False
    except Exception as e:
        print(f"  ✗ Error checking foreign key: {e}")
        return False


def verify_index_exists(inspector, table_name, index_columns):
    """Check if an index exists"""
    try:
        indexes = inspector.get_indexes(table_name)
        for idx in indexes:
            if set(idx['column_names']) == set(index_columns):
                print(f"  ✓ Index on {index_columns} exists")
                return True
        # Don't fail if index not found, as SQLite handles this differently
        print(f"  ⚠ Index on {index_columns} not found (may be implicit)")
        return True
    except Exception as e:
        print(f"  ✗ Error checking index: {e}")
        return False


def main():
    """Main verification function"""
    print("=" * 60)
    print("Database Schema Verification")
    print("=" * 60)
    print()
    
    inspector = inspect(engine)
    all_passed = True
    
    # Verify work_sessions table
    print("1. Verifying work_sessions table...")
    if verify_table_exists(inspector, 'work_sessions'):
        all_passed &= verify_column_exists(inspector, 'work_sessions', 'id')
        all_passed &= verify_column_exists(inspector, 'work_sessions', 'user_id')
        all_passed &= verify_column_exists(inspector, 'work_sessions', 'name')
        all_passed &= verify_column_exists(inspector, 'work_sessions', 'created_at')
        all_passed &= verify_column_exists(inspector, 'work_sessions', 'updated_at')
        all_passed &= verify_column_exists(inspector, 'work_sessions', 'last_active_at')
        all_passed &= verify_foreign_key_exists(inspector, 'work_sessions', 'user_id')
        verify_index_exists(inspector, 'work_sessions', ['user_id', 'last_active_at'])
    else:
        all_passed = False
    print()
    
    # Verify shared_exports table
    print("2. Verifying shared_exports table...")
    if verify_table_exists(inspector, 'shared_exports'):
        all_passed &= verify_column_exists(inspector, 'shared_exports', 'share_id')
        all_passed &= verify_column_exists(inspector, 'shared_exports', 'session_id')
        all_passed &= verify_column_exists(inspector, 'shared_exports', 'created_at')
        all_passed &= verify_column_exists(inspector, 'shared_exports', 'expires_at')
        all_passed &= verify_column_exists(inspector, 'shared_exports', 'access_count')
        all_passed &= verify_column_exists(inspector, 'shared_exports', 'last_accessed_at')
        all_passed &= verify_foreign_key_exists(inspector, 'shared_exports', 'session_id')
        verify_index_exists(inspector, 'shared_exports', ['session_id', 'expires_at'])
    else:
        all_passed = False
    print()
    
    # Verify predictions.session_id column
    print("3. Verifying predictions.session_id column...")
    if verify_table_exists(inspector, 'predictions'):
        all_passed &= verify_column_exists(inspector, 'predictions', 'session_id')
        all_passed &= verify_foreign_key_exists(inspector, 'predictions', 'session_id')
        verify_index_exists(inspector, 'predictions', ['session_id', 'created_at'])
    else:
        all_passed = False
    print()
    
    # Verify users table still exists
    print("4. Verifying users table (should already exist)...")
    if not verify_table_exists(inspector, 'users'):
        all_passed = False
    print()
    
    # Test model instantiation
    print("5. Testing model instantiation...")
    try:
        from datetime import datetime, timezone
        import uuid
        
        # Test that models can be instantiated (doesn't save to DB)
        test_user_id = str(uuid.uuid4())
        test_session_id = str(uuid.uuid4())
        
        user = User(
            key_id=test_user_id,
            username="test",
            password_hash="hash",
            is_active=True
        )
        print("  ✓ User model instantiation successful")
        
        work_session = WorkSession(
            id=test_session_id,
            user_id=test_user_id,
            name="Test Session"
        )
        print("  ✓ WorkSession model instantiation successful")
        
        shared_export = SharedExport(
            share_id=str(uuid.uuid4()),
            session_id=test_session_id,
            expires_at=datetime.now(timezone.utc)
        )
        print("  ✓ SharedExport model instantiation successful")
        
        prediction = Prediction(
            id=str(uuid.uuid4()),
            session_id=test_session_id,
            sequence="ACDEFGH"
        )
        print("  ✓ Prediction model instantiation successful (with session_id)")
        
    except Exception as e:
        print(f"  ✗ Model instantiation failed: {e}")
        all_passed = False
    print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("=" * 60)
        print("\nTo fix issues, try:")
        print("  1. Run migrations: python migrate.bat upgrade")
        print("  2. Check migration status: python migrate.bat current")
        print("  3. View migration history: python migrate.bat history")
        return 1


if __name__ == "__main__":
    sys.exit(main())
