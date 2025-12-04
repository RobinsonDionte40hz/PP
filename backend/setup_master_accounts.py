"""
Setup script for creating admin and developer test accounts.
Run this script to initialize master accounts for testing and administration.

Master Accounts:
- admin / Admin@2025!
- developer / Dev@2025!

Usage:
    python setup_master_accounts.py
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import Session
from app.database import get_db, engine
from app.models.user import User
from app.services.auth_service import AuthService


def setup_master_accounts():
    """Create master admin and developer accounts"""
    
    print("=" * 60)
    print("EmergentFolds Master Account Setup")
    print("=" * 60)
    print()
    
    # Get database session
    db = next(get_db())
    
    try:
        # Create tables if they don't exist
        from app.database import Base
        Base.metadata.create_all(bind=engine)
        print("✓ Database tables verified")
        print()
        
        # Master accounts configuration
        master_accounts = [
            {
                "username": "admin",
                "password": "Admin@2025!",
                "email": "admin@emergentfolds.local",
                "role": "admin",
                "description": "Administrator Account"
            },
            {
                "username": "developer",
                "password": "Dev@2025!",
                "email": "dev@emergentfolds.local",
                "role": "developer",
                "description": "Developer Account"
            }
        ]
        
        created_accounts = []
        
        for account in master_accounts:
            # Check if account already exists
            existing = AuthService.get_user_by_username(db, account["username"])
            
            if existing:
                print(f"⚠  {account['description']} already exists")
                print(f"   Username: {account['username']}")
                print(f"   Role: {existing.role}")
                print()
                continue
            
            # Create the account
            success, message, user = AuthService.register_user(
                db=db,
                username=account["username"],
                password=account["password"],
                email=account["email"],
                role=account["role"]
            )
            
            if success:
                created_accounts.append(account)
                print(f"✓ {account['description']} created successfully")
                print(f"   Username: {account['username']}")
                print(f"   Password: {account['password']}")
                print(f"   Email: {account['email']}")
                print(f"   Role: {account['role']}")
                print()
            else:
                print(f"✗ Failed to create {account['description']}")
                print(f"   Error: {message}")
                print()
        
        print("=" * 60)
        print("Setup Complete")
        print("=" * 60)
        print()
        
        if created_accounts:
            print("New Accounts Created:")
            for account in created_accounts:
                print(f"  • {account['username']} / {account['password']} ({account['role']})")
            print()
            print("⚠  IMPORTANT: Change these passwords in production!")
        else:
            print("No new accounts created (all already exist)")
        
        print()
        print("Master Access Credentials:")
        print("─" * 60)
        print("Admin Account:")
        print("  Username: admin")
        print("  Password: Admin@2025!")
        print("  Role: admin")
        print()
        print("Developer Account:")
        print("  Username: developer")
        print("  Password: Dev@2025!")
        print("  Role: developer")
        print("─" * 60)
        print()
        
    except Exception as e:
        print(f"✗ Error during setup: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    setup_master_accounts()
