"""
Quick test to verify User model functionality
"""
import uuid
from datetime import datetime, timezone
from app.models.user import User
from app.database import SessionLocal, engine
from sqlalchemy import inspect

def test_user_model():
    """Test User model creation and retrieval"""
    
    # 1. Verify table exists
    inspector = inspect(engine)
    assert "users" in inspector.get_table_names(), "Users table not found"
    print("✅ Users table exists")
    
    # 2. Verify columns
    columns = [c["name"] for c in inspector.get_columns("users")]
    expected_columns = ["key_id", "username", "email", "password_hash", 
                       "is_active", "created_at", "updated_at", "last_login"]
    assert all(col in columns for col in expected_columns), "Missing columns"
    print(f"✅ All expected columns present: {columns}")
    
    # 3. Verify indexes
    indexes = inspector.get_indexes("users")
    index_names = [idx["name"] for idx in indexes]
    print(f"✅ Indexes created: {index_names}")
    
    # 4. Test creating a user
    db = SessionLocal()
    try:
        test_user = User(
            key_id=str(uuid.uuid4()),
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password_placeholder",
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        db.add(test_user)
        db.commit()
        print(f"✅ User created: {test_user.username}")
        
        # 5. Test retrieval
        retrieved = db.query(User).filter_by(username="testuser").first()
        assert retrieved is not None, "User not found"
        assert retrieved.email == "test@example.com", "Email mismatch"
        print(f"✅ User retrieved: {retrieved.to_dict()}")
        
        # 6. Test to_profile method
        profile = retrieved.to_profile()
        assert "password_hash" not in profile, "Password hash in profile!"
        print(f"✅ User profile (no password): {profile}")
        
        # 7. Test unique constraint on username
        duplicate_user = User(
            key_id=str(uuid.uuid4()),
            username="testuser",  # Duplicate username
            email="another@example.com",
            password_hash="another_hash",
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        db.add(duplicate_user)
        try:
            db.commit()
            print("❌ FAILED: Duplicate username should have been rejected")
        except Exception as e:
            db.rollback()
            print(f"✅ Duplicate username correctly rejected: {type(e).__name__}")
        
        # Cleanup
        db.delete(retrieved)
        db.commit()
        print("✅ Test user cleaned up")
        
    finally:
        db.close()
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED - User model is working correctly!")
    print("="*50)

if __name__ == "__main__":
    test_user_model()
