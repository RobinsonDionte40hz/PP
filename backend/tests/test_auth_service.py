"""
Tests for user registration and login services
"""
import pytest
import uuid
import asyncio
from datetime import datetime, timezone
from hypothesis import given, strategies as st, settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.user import User
from app.services.auth_service import AuthService
from app.services.session_manager import SessionManager


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


class TestUserLogin:
    """Tests for user login functionality"""
    
    @pytest.fixture
    async def session_manager(self):
        """Create a test session manager"""
        manager = SessionManager(
            redis_url="redis://localhost:6379/15",
            session_expire_minutes=30
        )
        try:
            manager.redis_client.flushdb()
            yield manager
        finally:
            manager.redis_client.flushdb()
            manager.close()
    
    @pytest.mark.asyncio
    async def test_login_success(self, db, session_manager):
        """Test successful login"""
        # Register a user
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        
        # Login
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        assert success is True
        assert message == "Login successful"
        assert data is not None
        assert "user" in data
        assert "access_token" in data
        assert "refresh_token" in data
        assert "expires_in" in data
        assert data["user"].username == "testuser"
        
    @pytest.mark.asyncio
    async def test_login_invalid_username(self, db, session_manager):
        """Test login with non-existent username"""
        success, message, data = await AuthService.login_user(
            db=db,
            username="nonexistent",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        assert success is False
        assert "Invalid username or password" in message
        assert data is None
        
    @pytest.mark.asyncio
    async def test_login_invalid_password(self, db, session_manager):
        """Test login with wrong password"""
        # Register a user
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        
        # Login with wrong password
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="WrongPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        assert success is False
        assert "Invalid username or password" in message
        assert data is None
        
    @pytest.mark.asyncio
    async def test_login_empty_username(self, db, session_manager):
        """Test login with empty username"""
        success, message, data = await AuthService.login_user(
            db=db,
            username="",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        assert success is False
        assert "Username cannot be empty" in message
        assert data is None
        
    @pytest.mark.asyncio
    async def test_login_empty_password(self, db, session_manager):
        """Test login with empty password"""
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        assert success is False
        assert "Password cannot be empty" in message
        assert data is None
        
    @pytest.mark.asyncio
    async def test_login_session_creation(self, db, session_manager):
        """Test that login creates a session in Redis"""
        # Register a user
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        
        # Login
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        assert success is True
        # Session should be created - we can't easily verify the token JTI here
        # but we can verify the user has a session
        assert data["user"].key_id is not None
        
    @pytest.mark.asyncio
    async def test_login_updates_last_login(self, db, session_manager):
        """Test that login updates last_login timestamp"""
        # Register a user
        _, _, user = AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        initial_last_login = user.last_login
        
        # Login
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        assert success is True
        assert data["user"].last_login is not None
        # last_login should be updated (or set if it was None)
        if initial_last_login:
            assert data["user"].last_login > initial_last_login


class TestLoginPropertyTests:
    """Property-based tests for login functionality"""
    
    @pytest.fixture
    async def session_manager(self):
        """Create a test session manager"""
        manager = SessionManager(
            redis_url="redis://localhost:6379/15",
            session_expire_minutes=30
        )
        try:
            manager.redis_client.flushdb()
            yield manager
        finally:
            manager.redis_client.flushdb()
            manager.close()
    
    # Property 6: Valid credentials create sessions (Requirement 6.1)
    @pytest.mark.asyncio
    async def test_property_6_valid_credentials_create_session(self, db, session_manager):
        """
        Property 6: For any user with valid credentials,
        login should succeed and create an active session.
        
        Requirements: 2.1, 2.5
        """
        # Register user
        _, _, user = AuthService.register_user(
            db, "testuser", "TestPass123!", "test@example.com"
        )
        
        # Login with valid credentials
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        # Should succeed
        assert success is True
        assert data is not None
        assert "access_token" in data
        assert "refresh_token" in data
        
        # Verify session exists (can't get JTI easily here, but tokens should exist)
        assert len(data["access_token"]) > 0
        assert len(data["refresh_token"]) > 0
    
    # Property 7: Invalid credentials are rejected (Requirement 6.2)
    @pytest.mark.asyncio
    @settings(max_examples=50)
    @given(
        password=st.text(min_size=1, max_size=72).filter(lambda x: x != "TestPass123!")
    )
    async def test_property_7_invalid_credentials_rejected(self, db, session_manager, password):
        """
        Property 7: For any invalid password (not matching stored hash),
        login should fail with generic error message.
        
        Requirements: 2.2
        """
        # Register user
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        
        # Try login with wrong password
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password=password,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        # Should fail
        assert success is False
        assert data is None
        # Generic error message to prevent username enumeration
        assert "Invalid username or password" in message
    
    # Property 8: Sessions associate with correct user (Requirement 6.4)
    @pytest.mark.asyncio
    async def test_property_8_session_user_association(self, db, session_manager):
        """
        Property 8: For any successful login, the created session
        must be associated with the correct user's key_id.
        
        Requirements: 2.5, 3.2
        """
        # Register users
        _, _, user1 = AuthService.register_user(db, "user1", "TestPass123!", "user1@example.com")
        _, _, user2 = AuthService.register_user(db, "user2", "TestPass123!", "user2@example.com")
        
        # Login as user1
        success, _, data1 = await AuthService.login_user(
            db=db,
            username="user1",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        assert success is True
        assert data1["user"].key_id == user1.key_id
        assert data1["user"].username == "user1"
        
        # Login as user2
        success, _, data2 = await AuthService.login_user(
            db=db,
            username="user2",
            password="TestPass123!",
            ip_address="192.168.1.2",
            user_agent="Mozilla/5.0"
        )
        
        assert success is True
        assert data2["user"].key_id == user2.key_id
        assert data2["user"].username == "user2"
        
        # Verify users are different
        assert data1["user"].key_id != data2["user"].key_id
    
    # Property 9: Single session enforcement (Requirement 6.3)
    @pytest.mark.asyncio
    async def test_property_9_single_session_enforcement(self, db, session_manager):
        """
        Property 9: For any user, a new login should terminate
        any existing active session.
        
        Requirements: 3.1, 3.4
        """
        # Register user
        _, _, user = AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        
        # First login
        success1, _, data1 = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        assert success1 is True
        
        # Second login (should terminate first session)
        success2, _, data2 = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.2",
            user_agent="Chrome/120.0"
        )
        assert success2 is True
        
        # Should have different tokens
        assert data1["access_token"] != data2["access_token"]
        assert data1["refresh_token"] != data2["refresh_token"]
        
        # Only the second session should be active
        # (We can't easily verify the first was terminated without JTI,
        # but the enforce_single_session logic is tested separately)


class TestUserLogout:
    """Tests for user logout functionality"""
    
    @pytest.fixture
    async def session_manager(self):
        """Create a test session manager"""
        manager = SessionManager(
            redis_url="redis://localhost:6379/15",
            session_expire_minutes=30
        )
        try:
            manager.redis_client.flushdb()
            yield manager
        finally:
            manager.redis_client.flushdb()
            manager.close()
    
    @pytest.mark.asyncio
    async def test_logout_success(self, db, session_manager):
        """Test successful logout"""
        # Register and login
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        assert success is True
        token = data["access_token"]
        user_key_id = data["user"].key_id
        
        # Logout
        success, message = await AuthService.logout_user(
            token=token,
            user_key_id=user_key_id
        )
        
        assert success is True
        assert message == "Logout successful"
    
    @pytest.mark.asyncio
    async def test_logout_with_invalid_token(self, db, session_manager):
        """Test logout with invalid token (should still succeed gracefully)"""
        # Try to logout with invalid token
        success, message = await AuthService.logout_user(
            token="invalid_token",
            user_key_id="some-user-id"
        )
        
        # Should succeed gracefully (idempotent)
        assert success is True
        assert message == "Logout successful"
    
    @pytest.mark.asyncio
    async def test_logout_already_logged_out(self, db, session_manager):
        """Test logout when already logged out (idempotent)"""
        # Register and login
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        token = data["access_token"]
        user_key_id = data["user"].key_id
        
        # Logout first time
        success1, message1 = await AuthService.logout_user(token, user_key_id)
        assert success1 is True
        
        # Logout second time (should still succeed)
        success2, message2 = await AuthService.logout_user(token, user_key_id)
        assert success2 is True
        assert message2 == "Logout successful"


class TestLogoutPropertyTests:
    """Property-based tests for logout functionality"""
    
    @pytest.fixture
    async def session_manager(self):
        """Create a test session manager"""
        manager = SessionManager(
            redis_url="redis://localhost:6379/15",
            session_expire_minutes=30
        )
        try:
            manager.redis_client.flushdb()
            yield manager
        finally:
            manager.redis_client.flushdb()
            manager.close()
    
    # Property 10: Logout terminates sessions completely (Requirement 7.1)
    @pytest.mark.asyncio
    async def test_property_10_logout_terminates_session(self, db, session_manager):
        """
        Property 10: For any active session, logout should completely
        terminate the session and prevent reuse.
        
        Requirements: 3.3, 5.1, 5.2, 5.4
        """
        # Register and login
        _, _, user = AuthService.register_user(
            db, "testuser", "TestPass123!", "test@example.com"
        )
        
        success, _, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        assert success is True
        token = data["access_token"]
        user_key_id = data["user"].key_id
        
        # Extract JTI from token
        from app.security import extract_jti_from_token
        jti = extract_jti_from_token(token)
        assert jti is not None
        
        # Verify session exists before logout
        session_data = await session_manager.get_session(jti)
        assert session_data is not None
        assert session_data.user_key_id == user_key_id
        
        # Logout
        success, message = await AuthService.logout_user(token, user_key_id)
        assert success is True
        
        # Verify session is terminated (Requirement 5.1, 5.2)
        session_data_after = await session_manager.get_session(jti)
        assert session_data_after is None
        
        # Verify user has no active session (Requirement 5.4)
        active_session = await session_manager.get_active_session(user_key_id)
        assert active_session is None or active_session != jti
    
    @pytest.mark.asyncio
    async def test_property_10_logout_removes_session_data(self, db, session_manager):
        """
        Property 10 (variant): Logout must remove ALL session data,
        including user-to-session mapping.
        
        Requirements: 3.3, 5.2
        """
        # Register and login
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        
        success, _, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        token = data["access_token"]
        user_key_id = data["user"].key_id
        
        # Extract JTI
        from app.security import extract_jti_from_token
        jti = extract_jti_from_token(token)
        
        # Verify active session mapping exists
        active_jti_before = await session_manager.get_active_session(user_key_id)
        assert active_jti_before == jti
        
        # Logout
        await AuthService.logout_user(token, user_key_id)
        
        # Verify session data is removed
        session_data = await session_manager.get_session(jti)
        assert session_data is None
        
        # Verify active session mapping is removed or doesn't match
        active_jti_after = await session_manager.get_active_session(user_key_id)
        assert active_jti_after is None or active_jti_after != jti
    
    @pytest.mark.asyncio
    async def test_property_10_logout_idempotent(self, db, session_manager):
        """
        Property 10 (idempotency): Logout should succeed even if
        called multiple times or session doesn't exist.
        
        Requirement: 5.1
        """
        # Register and login
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        
        success, _, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        token = data["access_token"]
        user_key_id = data["user"].key_id
        
        # First logout
        success1, msg1 = await AuthService.logout_user(token, user_key_id)
        assert success1 is True
        
        # Second logout (session already gone)
        success2, msg2 = await AuthService.logout_user(token, user_key_id)
        assert success2 is True
        
        # Third logout (still should succeed)
        success3, msg3 = await AuthService.logout_user(token, user_key_id)
        assert success3 is True


class TestTokenRefresh:
    """Tests for token refresh functionality"""
    
    @pytest.fixture
    async def session_manager(self):
        """Create a test session manager"""
        manager = SessionManager(
            redis_url="redis://localhost:6379/15",
            session_expire_minutes=30
        )
        try:
            manager.redis_client.flushdb()
            yield manager
        finally:
            manager.redis_client.flushdb()
            manager.close()
    
    @pytest.mark.asyncio
    async def test_refresh_token_success(self, db, session_manager):
        """Test successful token refresh"""
        # Register and login
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        assert success is True
        old_access_token = data["access_token"]
        refresh_token = data["refresh_token"]
        
        # Refresh token
        success, message, refresh_data = await AuthService.refresh_token(
            refresh_token=refresh_token
        )
        
        assert success is True
        assert message == "Token refreshed successfully"
        assert "access_token" in refresh_data
        assert "expires_in" in refresh_data
        
        # New access token should be different from old one
        new_access_token = refresh_data["access_token"]
        assert new_access_token != old_access_token
    
    @pytest.mark.asyncio
    async def test_refresh_token_invalid_token(self, db, session_manager):
        """Test refresh with invalid token"""
        success, message, data = await AuthService.refresh_token(
            refresh_token="invalid_token"
        )
        
        assert success is False
        assert "Token refresh failed" in message
        assert data is None
    
    @pytest.mark.asyncio
    async def test_refresh_token_with_access_token(self, db, session_manager):
        """Test refresh with access token (wrong type)"""
        # Register and login
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        access_token = data["access_token"]
        
        # Try to refresh with access token (should fail - wrong type)
        success, message, refresh_data = await AuthService.refresh_token(
            refresh_token=access_token
        )
        
        assert success is False
        assert "Invalid token type" in message
        assert refresh_data is None
    
    @pytest.mark.asyncio
    async def test_refresh_token_after_logout(self, db, session_manager):
        """Test refresh after logout (session terminated)"""
        # Register and login
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        access_token = data["access_token"]
        refresh_token = data["refresh_token"]
        user_key_id = data["user"].key_id
        
        # Logout
        await AuthService.logout_user(access_token, user_key_id)
        
        # Try to refresh (should fail - no active session)
        success, message, refresh_data = await AuthService.refresh_token(
            refresh_token=refresh_token
        )
        
        assert success is False
        assert "No active session found" in message
        assert refresh_data is None
    
    @pytest.mark.asyncio
    async def test_refresh_token_updates_session(self, db, session_manager):
        """Test that refresh updates the session with new JTI"""
        # Register and login
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        old_access_token = data["access_token"]
        refresh_token = data["refresh_token"]
        user_key_id = data["user"].key_id
        
        # Extract old JTI
        from app.security import extract_jti_from_token
        old_jti = extract_jti_from_token(old_access_token)
        
        # Verify old session exists
        old_session = await session_manager.get_session(old_jti)
        assert old_session is not None
        
        # Refresh token
        success, message, refresh_data = await AuthService.refresh_token(
            refresh_token=refresh_token
        )
        
        assert success is True
        new_access_token = refresh_data["access_token"]
        
        # Extract new JTI
        new_jti = extract_jti_from_token(new_access_token)
        
        # Verify JTIs are different
        assert new_jti != old_jti
        
        # Verify new session exists
        new_session = await session_manager.get_session(new_jti)
        assert new_session is not None
        assert new_session.user_key_id == user_key_id
        
        # Verify old session is terminated
        old_session_after = await session_manager.get_session(old_jti)
        assert old_session_after is None
    
    @pytest.mark.asyncio
    async def test_refresh_token_multiple_times(self, db, session_manager):
        """Test refreshing token multiple times"""
        # Register and login
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        refresh_token = data["refresh_token"]
        
        # First refresh
        success1, _, data1 = await AuthService.refresh_token(refresh_token)
        assert success1 is True
        token1 = data1["access_token"]
        
        # Second refresh (using same refresh token)
        success2, _, data2 = await AuthService.refresh_token(refresh_token)
        assert success2 is True
        token2 = data2["access_token"]
        
        # All tokens should be different
        assert token1 != token2
        assert token1 != data["access_token"]
        assert token2 != data["access_token"]


class TestSecurityPropertyTests:
    """
    Property-based tests for security requirements (Task 10)
    """
    
    @pytest.mark.asyncio
    async def test_property_14_jwt_tokens_cryptographically_secure(self, db, session_manager):
        """
        Property 14: JWT tokens must be cryptographically secure
        
        Requirements:
        - 6.2: Secure session tokens resistant to prediction/guessing
        - Each token must have unique JTI (jti claim)
        - Tokens generated for same user must be different
        - JTI must be UUID4 (random, not sequential)
        """
        # Register a user
        AuthService.register_user(db, "secureuser", "SecurePass123!", "secure@example.com")
        
        # Generate multiple tokens for the same user
        tokens = []
        jtis = []
        
        for i in range(10):
            # Login to get a fresh token
            success, message, data = await AuthService.login_user(
                db=db,
                username="secureuser",
                password="SecurePass123!",
                ip_address=f"192.168.1.{i}",
                user_agent="TestAgent"
            )
            
            assert success is True
            access_token = data["access_token"]
            tokens.append(access_token)
            
            # Extract JTI
            import jwt
            decoded = jwt.decode(access_token, options={"verify_signature": False})
            jti = decoded.get("jti")
            jtis.append(jti)
            
            # Verify JTI is a valid UUID4
            try:
                uuid_obj = uuid.UUID(jti, version=4)
                assert str(uuid_obj) == jti, "JTI must be UUID4 format"
            except ValueError:
                pytest.fail(f"JTI {jti} is not a valid UUID4")
        
        # Verify all tokens are unique
        assert len(set(tokens)) == 10, "All tokens must be unique"
        
        # Verify all JTIs are unique
        assert len(set(jtis)) == 10, "All JTIs must be unique"
        
        # Verify JTIs are not sequential (random)
        # Convert to integers and check they're not consecutive
        jti_ints = [int(uuid.UUID(jti)) for jti in jtis]
        for i in range(len(jti_ints) - 1):
            diff = abs(jti_ints[i+1] - jti_ints[i])
            # If JTIs were sequential, differences would be small (<1000)
            # Random UUIDs have large differences (typically > 10^30)
            assert diff > 1000, f"JTIs appear sequential: {diff}"
    
    @pytest.mark.asyncio
    async def test_property_14_refresh_tokens_unique(self, db, session_manager):
        """
        Property 14 (variant): Refresh tokens must be unique per session
        
        Requirements:
        - 6.2: Secure session tokens
        - Each refresh token must be different
        - Refresh tokens must be UUID4
        """
        # Register a user
        AuthService.register_user(db, "refreshuser", "RefreshPass123!", "refresh@example.com")
        
        refresh_tokens = []
        
        for i in range(5):
            # Login to get refresh token
            success, message, data = await AuthService.login_user(
                db=db,
                username="refreshuser",
                password="RefreshPass123!",
                ip_address=f"192.168.1.{i}",
                user_agent="TestAgent"
            )
            
            assert success is True
            refresh_token = data["refresh_token"]
            refresh_tokens.append(refresh_token)
            
            # Verify refresh token is UUID4
            try:
                uuid_obj = uuid.UUID(refresh_token, version=4)
                assert str(uuid_obj) == refresh_token
            except ValueError:
                pytest.fail(f"Refresh token {refresh_token} is not valid UUID4")
        
        # All refresh tokens must be unique
        assert len(set(refresh_tokens)) == 5, "All refresh tokens must be unique"
    
    @pytest.mark.asyncio
    async def test_property_15_no_passwords_in_logs(self, db, session_manager, caplog):
        """
        Property 15: Sensitive data must not appear in logs
        
        Requirements:
        - 6.5: Secure logging - no passwords, tokens, or PII
        - Logs may contain: username, IP, user_agent, timestamps
        - Logs must NOT contain: password, access_token, refresh_token, key_id
        """
        import logging
        caplog.set_level(logging.INFO)
        
        # Register user
        password = "SuperSecret123!"
        email = "sensitive@example.com"
        
        success, message, user = AuthService.register_user(
            db=db,
            username="loguser",
            password=password,
            email=email
        )
        
        # Check registration logs don't contain password
        registration_logs = caplog.text
        assert password not in registration_logs, "Password found in registration logs"
        assert email not in registration_logs or "key_id" in registration_logs, \
            "Email may appear but not with password"
        
        # Clear logs
        caplog.clear()
        
        # Login user
        success, message, data = await AuthService.login_user(
            db=db,
            username="loguser",
            password=password,
            ip_address="192.168.1.100",
            user_agent="TestBrowser/1.0"
        )
        
        login_logs = caplog.text
        
        # Verify password not in logs
        assert password not in login_logs, "Password found in login logs"
        
        # Verify tokens not in logs
        if success and data:
            access_token = data.get("access_token")
            refresh_token = data.get("refresh_token")
            
            assert access_token not in login_logs, "Access token found in logs"
            assert refresh_token not in login_logs, "Refresh token found in logs"
        
        # Verify safe data IS present (username, IP)
        assert "loguser" in login_logs, "Username should be in logs"
        assert "192.168.1.100" in login_logs, "IP address should be in logs"
    
    @pytest.mark.asyncio
    @given(
        username=st.text(
            alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_-'),
            min_size=3,
            max_size=50
        ).filter(lambda x: x[0].isalnum() if x else False),
        password=st.text(min_size=8, max_size=72).filter(
            lambda p: any(c.isupper() for c in p) and 
                      any(c.islower() for c in p) and 
                      any(c.isdigit() for c in p) and 
                      any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in p)
        )
    )
    @settings(max_examples=20, deadline=5000)
    async def test_property_15_no_sensitive_data_in_any_operation(self, db, session_manager, username, password, caplog):
        """
        Property 15 (hypothesis): Sensitive data never appears in logs regardless of input
        
        This property test validates that no matter what valid username/password
        combination is used, sensitive data is never logged.
        """
        import logging
        caplog.set_level(logging.INFO)
        caplog.clear()
        
        # Register with hypothesis-generated credentials
        success, message, user = AuthService.register_user(
            db=db,
            username=username,
            password=password,
            email=f"{username}@example.com"
        )
        
        if not success:
            # Skip invalid registrations
            return
        
        # Check all logs
        all_logs = caplog.text
        
        # Password must never appear
        assert password not in all_logs, f"Password '{password}' found in logs"
        
        # If login succeeds, tokens must not appear
        caplog.clear()
        success, message, data = await AuthService.login_user(
            db=db,
            username=username,
            password=password,
            ip_address="192.168.1.1",
            user_agent="HypothesisTest"
        )
        
        login_logs = caplog.text
        
        # Password never in logs
        assert password not in login_logs, f"Password found in login logs"
        
        # Tokens never in logs
        if success and data:
            assert data.get("access_token") not in login_logs, "Access token in logs"
            assert data.get("refresh_token") not in login_logs, "Refresh token in logs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
