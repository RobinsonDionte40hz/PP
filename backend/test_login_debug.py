import asyncio
import sys
sys.path.insert(0, ".")

from app.database import SessionLocal
from app.services.auth_service import AuthService

async def test_login():
    db = SessionLocal()
    try:
        # Register a user
        print("Registering user...")
        success, message, user = AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        print(f"Registration: success={success}, message={message}, user={user}")
        
        # Try to login
        print("\nAttempting login...")
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        print(f"Login: success={success}, message={message}, data={data}")
        
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_login())
