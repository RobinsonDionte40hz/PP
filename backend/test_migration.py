"""
Test database migration upgrade and downgrade

This script tests that the migration can be applied and rolled back successfully.
"""
import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print(f"✓ {description} - SUCCESS")
        return True
    else:
        print(f"✗ {description} - FAILED (exit code {result.returncode})")
        return False


def main():
    """Main test function"""
    # Change to backend directory
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    print("=" * 60)
    print("Database Migration Test Suite")
    print("=" * 60)
    
    # Backup existing database if it exists
    db_file = Path("pp_dev.db")
    backup_file = Path("pp_dev.db.backup")
    
    if db_file.exists():
        print(f"\nBacking up existing database to {backup_file}")
        import shutil
        shutil.copy(db_file, backup_file)
    
    all_passed = True
    
    # Test 1: Check Alembic is installed
    print("\n1. Checking Alembic installation...")
    result = subprocess.run(["alembic", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Alembic is installed: {result.stdout.strip()}")
    else:
        print("✗ Alembic is not installed")
        print("Please install: pip install alembic")
        return 1
    
    # Test 2: Check current database state
    all_passed &= run_command(
        ["alembic", "current"],
        "2. Check current database version"
    )
    
    # Test 3: Downgrade to base (clean state)
    all_passed &= run_command(
        ["alembic", "downgrade", "base"],
        "3. Downgrade to base (clean slate)"
    )
    
    # Test 4: Upgrade to head (apply migration)
    all_passed &= run_command(
        ["alembic", "upgrade", "head"],
        "4. Upgrade to head (apply migration 001)"
    )
    
    # Test 5: Verify schema
    print("\n5. Verifying database schema...")
    result = subprocess.run([sys.executable, "verify_schema.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("✓ Schema verification - SUCCESS")
    else:
        print("✗ Schema verification - FAILED")
        all_passed = False
    
    # Test 6: Check migration history
    all_passed &= run_command(
        ["alembic", "history"],
        "6. Check migration history"
    )
    
    # Test 7: Test downgrade (rollback)
    all_passed &= run_command(
        ["alembic", "downgrade", "-1"],
        "7. Test downgrade (rollback migration)"
    )
    
    # Test 8: Re-apply migration
    all_passed &= run_command(
        ["alembic", "upgrade", "head"],
        "8. Re-apply migration (test idempotency)"
    )
    
    # Test 9: Run model property tests
    print("\n9. Running property tests for WorkSession model...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_work_session_model.py", "-v"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("✓ Property tests - SUCCESS")
    else:
        print("✗ Property tests - FAILED")
        all_passed = False
    
    # Restore backup if requested
    if backup_file.exists():
        response = input("\nRestore database backup? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.copy(backup_file, db_file)
            print(f"✓ Database restored from {backup_file}")
            backup_file.unlink()
        else:
            print(f"Backup kept at {backup_file}")
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nMigration is working correctly!")
        print("\nNext steps:")
        print("  1. Implement FileStorageService (Task 2)")
        print("  2. Implement WorkSessionService (Task 3)")
        print("  3. Create API schemas (Task 4)")
        print("  4. Implement API endpoints (Tasks 5-7)")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease review the errors above and fix before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
