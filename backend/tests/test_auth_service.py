"""
Tests for user registration service
"""
import pytest
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.user import User
from app.services.auth_service import AuthService


# Create test database
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db():
    """Create a fresh database for each test"""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


class TestUserRegistration:
    """Tests for user registration functionality"""
    
    def test_register_user_success(self, db):
        """Test successful user registration"""
        success, message, user = AuthService.register_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            email="test@example.com"
        )
        
        assert success is True
        assert message == "User registered successfully"
        assert user is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert len(user.key_id) == 36  # UUID length
        assert user.password_hash != "TestPass123!"  # Password should be hashed
        
    def test_register_user_without_email(self, db):
        """Test registration without email"""
        success, message, user = AuthService.register_user(
            db=db,
            username="testuser",
            password="TestPass123!"
        )
        
        assert success is True
        assert user is not None
        assert user.email is None
        
    def test_register_duplicate_username(self, db):
        """Test that duplicate username is rejected"""
        # Register first user
        AuthService.register_user(db, "testuser", "TestPass123!", "test1@example.com")
        
        # Try to register with same username
        success, message, user = AuthService.register_user(
            db, "testuser", "DifferentPass123!", "test2@example.com"
        )
        
        assert success is False
        assert "Username already exists" in message
        assert user is None
        
    def test_register_duplicate_email(self, db):
        """Test that duplicate email is rejected"""
        # Register first user
        AuthService.register_user(db, "testuser1", "TestPass123!", "test@example.com")
        
        # Try to register with same email
        success, message, user = AuthService.register_user(
            db, "testuser2", "TestPass123!", "test@example.com"
        )
        
        assert success is False
        assert "Email already exists" in message
        assert user is None
        
    def test_register_empty_username(self, db):
        """Test that empty username is rejected"""
        success, message, user = AuthService.register_user(
            db, "", "TestPass123!"
        )
        
        assert success is False
        assert "Username cannot be empty" in message
        assert user is None
        
    def test_register_empty_password(self, db):
        """Test that empty password is rejected"""
        success, message, user = AuthService.register_user(
            db, "testuser", ""
        )
        
        assert success is False
        assert "Password cannot be empty" in message
        assert user is None
        
    def test_register_weak_password(self, db):
        """Test that weak password is rejected"""
        success, message, user = AuthService.register_user(
            db, "testuser", "weak"
        )
        
        assert success is False
        assert user is None
        # Should have multiple validation errors
        assert "at least 8 characters" in message.lower()
        
    def test_register_password_no_uppercase(self, db):
        """Test password without uppercase is rejected"""
        success, message, user = AuthService.register_user(
            db, "testuser", "testpass123!"
        )
        
        assert success is False
        assert "uppercase" in message.lower()
        
    def test_register_password_no_digit(self, db):
        """Test password without digit is rejected"""
        success, message, user = AuthService.register_user(
            db, "testuser", "TestPassword!"
        )
        
        assert success is False
        assert "digit" in message.lower()
        
    def test_register_password_no_special(self, db):
        """Test password without special character is rejected"""
        success, message, user = AuthService.register_user(
            db, "testuser", "TestPassword123"
        )
        
        assert success is False
        assert "special character" in message.lower()
        
    def test_uuid_uniqueness(self, db):
        """Test that each user gets a unique UUID"""
        success1, _, user1 = AuthService.register_user(
            db, "user1", "TestPass123!", "user1@example.com"
        )
        success2, _, user2 = AuthService.register_user(
            db, "user2", "TestPass123!", "user2@example.com"
        )
        
        assert success1 and success2
        assert user1 is not None
        assert user2 is not None
        assert user1.key_id != user2.key_id
        
        # Verify UUID format
        try:
            uuid.UUID(user1.key_id)
            uuid.UUID(user2.key_id)
        except ValueError:
            pytest.fail("Invalid UUID format")


class TestUserQueries:
    """Tests for user query functions"""
    
    def test_get_user_by_username(self, db):
        """Test finding user by username"""
        # Create a user
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        
        # Find by username
        user = AuthService.get_user_by_username(db, "testuser")
        assert user is not None
        assert user.username == "testuser"
        
        # Try non-existent user
        user = AuthService.get_user_by_username(db, "nonexistent")
        assert user is None
        
    def test_get_user_by_email(self, db):
        """Test finding user by email"""
        # Create a user
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        
        # Find by email
        user = AuthService.get_user_by_email(db, "test@example.com")
        assert user is not None
        assert user.email == "test@example.com"
        
        # Try non-existent email
        user = AuthService.get_user_by_email(db, "nonexistent@example.com")
        assert user is None
        
    def test_get_user_by_id(self, db):
        """Test finding user by key_id"""
        # Create a user
        _, _, created_user = AuthService.register_user(
            db, "testuser", "TestPass123!", "test@example.com"
        )
        assert created_user is not None
        
        # Find by key_id
        user = AuthService.get_user_by_id(db, created_user.key_id)
        assert user is not None
        assert user.key_id == created_user.key_id
        
        # Try non-existent key_id
        fake_uuid = str(uuid.uuid4())
        user = AuthService.get_user_by_id(db, fake_uuid)
        assert user is None


class TestSecurityProperties:
    """Property-based tests for security requirements"""
    
    def test_password_not_stored_plaintext(self, db):
        """Property: Passwords must never be stored in plaintext"""
        password = "TestPass123!"
        success, _, user = AuthService.register_user(
            db, "testuser", password, "test@example.com"
        )
        
        assert success
        assert user is not None
        assert user.password_hash != password
        assert password not in user.password_hash
        
    def test_user_profile_excludes_password(self, db):
        """Property: User profile must never include password hash"""
        success, _, user = AuthService.register_user(
            db, "testuser", "TestPass123!", "test@example.com"
        )
        
        assert success
        assert user is not None
        profile = user.to_profile()
        
        # Profile should not contain password_hash
        assert "password_hash" not in profile
        assert "password" not in profile
        
        # Profile should contain public info
        assert "key_id" in profile
        assert "username" in profile
        assert "email" in profile
        assert "created_at" in profile


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
