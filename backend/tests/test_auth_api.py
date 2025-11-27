"""
Tests for authentication API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.models.user import User  # Import to register model with Base

# Create test database (file-based to share between connections)
TEST_DATABASE_URL = "sqlite:///./test_auth.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Ensure User model is registered with Base before creating tables
_ = User

# Create tables before anything else
Base.metadata.create_all(bind=engine)

# Override the dependency
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Clear database for each test"""
    # Tables are already created, just need to clear them
    with TestingSessionLocal() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
    yield
    # Cleanup after all tests
    

@pytest.fixture(scope="module", autouse=True)
def cleanup_test_db():
    """Remove test database file after all tests"""
    yield
    # Dispose of all connections before removing file
    engine.dispose()
    import os
    import time
    if os.path.exists("./test_auth.db"):
        time.sleep(0.1)  # Brief delay to ensure file handles released
        try:
            os.remove("./test_auth.db")
        except PermissionError:
            pass  # File in use, will be cleaned up manually


class TestRegistrationEndpoint:
    """Tests for /api/auth/register endpoint"""
    
    def test_register_success(self):
        """Test successful registration"""
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "TestPass123!",
                "email": "test@example.com"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["message"] == "User registered successfully"
        assert data["user"]["username"] == "testuser"
        assert data["user"]["email"] == "test@example.com"
        assert "key_id" in data["user"]
        assert "created_at" in data["user"]
        assert "password" not in data["user"]
        assert "password_hash" not in data["user"]
        
    def test_register_without_email(self):
        """Test registration without email"""
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        username = f"testuser_{unique_id}"
        
        response = client.post(
            "/api/auth/register",
            json={
                "username": username,
                "password": "TestPass123!"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["user"]["email"] is None
        
    def test_register_duplicate_username(self):
        """Test duplicate username returns 409"""
        # Register first user
        client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "TestPass123!",
                "email": "test1@example.com"
            }
        )
        
        # Try to register with same username
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "DifferentPass123!",
                "email": "test2@example.com"
            }
        )
        
        assert response.status_code == 409
        assert "Username already exists" in response.json()["detail"]
        
    def test_register_duplicate_email(self):
        """Test duplicate email returns 409"""
        # Register first user
        client.post(
            "/api/auth/register",
            json={
                "username": "testuser1",
                "password": "TestPass123!",
                "email": "test@example.com"
            }
        )
        
        # Try to register with same email
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser2",
                "password": "TestPass123!",
                "email": "test@example.com"
            }
        )
        
        assert response.status_code == 409
        assert "Email already exists" in response.json()["detail"]
        
    def test_register_invalid_username_too_short(self):
        """Test username too short returns 422"""
        response = client.post(
            "/api/auth/register",
            json={
                "username": "ab",  # Less than 3 chars
                "password": "TestPass123!",
                "email": "test@example.com"
            }
        )
        
        assert response.status_code == 422
        
    def test_register_invalid_username_special_char_start(self):
        """Test username starting with special char returns 422"""
        response = client.post(
            "/api/auth/register",
            json={
                "username": "_testuser",
                "password": "TestPass123!",
                "email": "test@example.com"
            }
        )
        
        assert response.status_code == 422
        assert "must start with a letter or number" in response.text.lower()
        
    def test_register_invalid_email(self):
        """Test invalid email returns 422"""
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "TestPass123!",
                "email": "not-an-email"
            }
        )
        
        assert response.status_code == 422
        
    def test_register_weak_password(self):
        """Test weak password returns 422 (validation error from pydantic)"""
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "weak",
                "email": "test@example.com"
            }
        )
        
        # Pydantic validation catches this before service layer
        assert response.status_code == 422
        
    def test_register_password_no_uppercase(self):
        """Test password without uppercase returns 400"""
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "testpass123!",
                "email": "test@example.com"
            }
        )
        
        assert response.status_code == 400
        assert "uppercase" in response.json()["detail"].lower()
        
    def test_register_password_no_special(self):
        """Test password without special char returns 400"""
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "password": "TestPass123",
                "email": "test@example.com"
            }
        )
        
        assert response.status_code == 400
        assert "special character" in response.json()["detail"].lower()
        
    def test_register_missing_username(self):
        """Test missing username returns 422"""
        response = client.post(
            "/api/auth/register",
            json={
                "password": "TestPass123!",
                "email": "test@example.com"
            }
        )
        
        assert response.status_code == 422
        
    def test_register_missing_password(self):
        """Test missing password returns 422"""
        response = client.post(
            "/api/auth/register",
            json={
                "username": "testuser",
                "email": "test@example.com"
            }
        )
        
        assert response.status_code == 422


class TestHealthEndpoint:
    """Tests for /api/auth/health endpoint"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/auth/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "authentication"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
