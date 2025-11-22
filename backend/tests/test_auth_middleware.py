"""
Tests for authentication middleware (Task 8)

Tests Property 13 from design.md:
- Property 13: Protected resource access control
"""
import pytest
import asyncio
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.middleware import AuthenticationMiddleware
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


@pytest.fixture
async def session_manager():
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


@pytest.fixture
def test_app():
    """Create a test FastAPI app with authentication middleware"""
    app = FastAPI()
    
    # Add authentication middleware for protected routes
    app.add_middleware(
        AuthenticationMiddleware,
        protected_paths=["/api/protected"]
    )
    
    @app.get("/api/protected/resource")
    async def protected_resource(request: Request):
        """Protected endpoint that requires authentication"""
        # User info should be attached by middleware
        user = getattr(request.state, 'user', None)
        if user:
            return {"message": "Access granted", "user": user}
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized"}
        )
    
    @app.get("/api/public/resource")
    async def public_resource():
        """Public endpoint that doesn't require authentication"""
        return {"message": "Public access"}
    
    @app.get("/health")
    async def health():
        """Health check endpoint (public)"""
        return {"status": "healthy"}
    
    return app


class TestAuthenticationMiddleware:
    """Tests for authentication middleware functionality"""
    
    @pytest.mark.asyncio
    async def test_public_path_no_auth_required(self, test_app):
        """Test that public paths don't require authentication"""
        client = TestClient(test_app)
        
        # Public endpoint should be accessible without auth
        response = client.get("/api/public/resource")
        assert response.status_code == 200
        assert response.json()["message"] == "Public access"
        
        # Health check should be accessible
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_protected_path_requires_auth(self, test_app):
        """Test that protected paths require authentication"""
        client = TestClient(test_app)
        
        # Protected endpoint should require auth
        response = client.get("/api/protected/resource")
        assert response.status_code == 401
        assert "Authentication required" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_protected_path_with_invalid_token(self, test_app):
        """Test protected path with invalid token"""
        client = TestClient(test_app)
        
        # Try with invalid token
        response = client.get(
            "/api/protected/resource",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401
        assert "Could not validate credentials" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_protected_path_with_missing_bearer(self, test_app):
        """Test protected path with token missing Bearer prefix"""
        client = TestClient(test_app)
        
        # Try with token without Bearer prefix
        response = client.get(
            "/api/protected/resource",
            headers={"Authorization": "some_token"}
        )
        assert response.status_code == 401


class TestAuthenticationPropertyTests:
    """Property-based tests for authentication middleware"""
    
    # Property 13: Protected resource access control (Task 8.1)
    @pytest.mark.asyncio
    async def test_property_13_access_control_authenticated(self, db, session_manager, test_app):
        """
        Property 13: For any user with valid credentials and active session,
        access to protected resources should be granted.
        
        Requirements: 4.2, 4.5, 5.5
        """
        # Register and login user
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
        
        # Access protected resource with valid token
        client = TestClient(test_app)
        response = client.get(
            "/api/protected/resource",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Should succeed (Requirement 4.2, 4.5)
        assert response.status_code == 200
        assert response.json()["message"] == "Access granted"
        assert "user" in response.json()
        assert response.json()["user"]["username"] == "testuser"
    
    @pytest.mark.asyncio
    async def test_property_13_access_control_no_session(self, db, session_manager, test_app):
        """
        Property 13: For any user without an active session,
        access to protected resources should be denied.
        
        Requirements: 4.2, 5.5
        """
        # Register and login user
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
        
        # Logout to terminate session
        await AuthService.logout_user(token, user_key_id)
        
        # Try to access protected resource after logout (Requirement 5.5)
        client = TestClient(test_app)
        response = client.get(
            "/api/protected/resource",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Should fail with 401
        assert response.status_code == 401
        assert "Session not found or expired" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_property_13_access_control_unauthenticated(self, test_app):
        """
        Property 13: For any request without authentication credentials,
        access to protected resources should be denied.
        
        Requirement: 4.2
        """
        client = TestClient(test_app)
        
        # Try to access protected resource without token
        response = client.get("/api/protected/resource")
        
        # Should fail with 401
        assert response.status_code == 401
        assert "Authentication required" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_property_13_session_validation_on_each_request(self, db, session_manager, test_app):
        """
        Property 13: For any protected resource request, the middleware
        must validate the session exists in Redis before granting access.
        
        Requirement: 4.2, 4.5
        """
        # Register and login user
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        success, message, data = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        token = data["access_token"]
        
        client = TestClient(test_app)
        
        # First request should succeed
        response1 = client.get(
            "/api/protected/resource",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response1.status_code == 200
        
        # Manually delete session from Redis (simulate session expiration)
        from app.security import extract_jti_from_token
        jti = extract_jti_from_token(token)
        await session_manager.terminate_session(jti)
        
        # Second request should fail (session no longer exists)
        response2 = client.get(
            "/api/protected/resource",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response2.status_code == 401
        assert "Session not found or expired" in response2.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_property_13_single_session_enforcement_in_middleware(self, db, session_manager, test_app):
        """
        Property 13: The middleware should enforce single-session-per-user
        by validating that the token matches the active session.
        
        Requirement: 4.5
        """
        # Register and login user first time
        AuthService.register_user(db, "testuser", "TestPass123!", "test@example.com")
        success1, _, data1 = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        token1 = data1["access_token"]
        
        client = TestClient(test_app)
        
        # First token should work
        response1 = client.get(
            "/api/protected/resource",
            headers={"Authorization": f"Bearer {token1}"}
        )
        assert response1.status_code == 200
        
        # Login again (should terminate first session)
        success2, _, data2 = await AuthService.login_user(
            db=db,
            username="testuser",
            password="TestPass123!",
            ip_address="192.168.1.2",
            user_agent="Chrome/120.0"
        )
        
        token2 = data2["access_token"]
        
        # First token should no longer work (old session terminated)
        response2 = client.get(
            "/api/protected/resource",
            headers={"Authorization": f"Bearer {token1}"}
        )
        assert response2.status_code == 401
        
        # Second token should work
        response3 = client.get(
            "/api/protected/resource",
            headers={"Authorization": f"Bearer {token2}"}
        )
        assert response3.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
