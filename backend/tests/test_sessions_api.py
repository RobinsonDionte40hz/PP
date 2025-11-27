"""
Tests for work sessions API endpoints
"""
import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Disable rate limiting for tests
os.environ["TESTING"] = "true"

from app.main import app
from app.database import Base, get_db
from app.models.user import User
from app.models.work_session import WorkSession
from app.models.prediction import Prediction
from app.services.auth_service import AuthService

# Create test database
TEST_DATABASE_URL = "sqlite:///./test_sessions_api.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Ensure models are registered
_ = User
_ = WorkSession
_ = Prediction

# Create tables
Base.metadata.create_all(bind=engine)

# Override the dependency
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Clear database before and after each test (except users table)"""
    # Clear before test - use raw text to force execution
    with TestingSessionLocal() as db:
        # Clear all tables except users (to preserve authenticated_user fixture)
        # Check if table exists before deleting
        try:
            db.execute(text("DELETE FROM shared_exports"))
        except:
            pass
        try:
            db.execute(text("DELETE FROM predictions"))
        except:
            pass
        try:
            db.execute(text("DELETE FROM work_sessions"))
        except:
            pass
        db.commit()
        db.expunge_all()  # Clear SQLAlchemy session cache
        db.close()  # Explicitly close to flush connection
    
    # Flush Redis before each test
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        # Flush only session keys, not user keys
        for key in r.scan_iter("session:*"):
            r.delete(key)
    except Exception:
        pass
    
    yield
    
    # Clear after test to ensure complete isolation
    with TestingSessionLocal() as db:
        try:
            db.execute(text("DELETE FROM shared_exports"))
        except:
            pass
        try:
            db.execute(text("DELETE FROM predictions"))
        except:
            pass
        try:
            db.execute(text("DELETE FROM work_sessions"))
        except:
            pass
        db.commit()
        db.expunge_all()  # Clear SQLAlchemy session cache
        db.close()  # Explicitly close


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_db():
    """Remove test database file after all tests"""
    yield
    engine.dispose()
    import os
    import time
    if os.path.exists("./test_sessions_api.db"):
        time.sleep(0.1)
        try:
            os.remove("./test_sessions_api.db")
        except PermissionError:
            pass


@pytest.fixture(scope="function")
def authenticated_user():
    """Create a test user and return authentication token (fresh for each test)"""
    # Clear Redis first to avoid rate limiting
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.flushdb()
    except Exception:
        pass
    
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    username = f"testuser_{unique_id}"
    email = f"test_{unique_id}@example.com"
    
    with TestingSessionLocal() as db:
        # Register user with unique credentials
        success, message, user = AuthService.register_user(
            db=db,
            username=username,
            password="TestPass123!",
            email=email
        )
        
        if not success:
            raise Exception(f"Failed to register test user: {message}")
    
    # Login to get token
    response = client.post(
        "/api/auth/login",
        json={
            "username": username,
            "password": "TestPass123!"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    return {
        "user_id": data["user"]["key_id"],
        "token": data["tokens"]["access_token"],
        "headers": {"Authorization": f"Bearer {data['tokens']['access_token']}"}
    }


@pytest.fixture(scope="function")
def second_authenticated_user():
    """Create a second test user for cross-user access tests (fresh for each test)"""
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    username = f"testuser2_{unique_id}"
    email = f"test2_{unique_id}@example.com"
    
    with TestingSessionLocal() as db:
        # Register second user with unique credentials
        success, message, user = AuthService.register_user(
            db=db,
            username=username,
            password="TestPass123!",
            email=email
        )
        
        if not success:
            raise Exception(f"Failed to register second test user: {message}")
    
    response = client.post(
        "/api/auth/login",
        json={
            "username": username,
            "password": "TestPass123!"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    return {
        "user_id": data["user"]["key_id"],
        "token": data["tokens"]["access_token"],
        "headers": {"Authorization": f"Bearer {data['tokens']['access_token']}"}
    }


class TestListSessionsEndpoint:
    """Tests for GET /api/sessions endpoint"""
    
    def test_list_sessions_requires_auth(self):
        """Test that listing sessions requires authentication"""
        response = client.get("/api/sessions")
        assert response.status_code == 401
    
    def test_list_sessions_empty(self, authenticated_user):
        """Test listing sessions when user has none"""
        response = client.get(
            "/api/sessions",
            headers=authenticated_user["headers"]
        )
        assert response.status_code == 200
        data = response.json()
        
        assert data["sessions"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["page_size"] == 20
    
    def test_list_sessions_with_data(self, authenticated_user):
        """Test listing sessions with data"""
        # Create sessions
        for i in range(3):
            client.post(
                "/api/sessions",
                headers=authenticated_user["headers"],
                json={"name": f"Test Session {i}"}
            )
        
        # List sessions
        response = client.get(
            "/api/sessions",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["sessions"]) == 3
        assert data["total"] == 3
        assert all("id" in s for s in data["sessions"])
        assert all("name" in s for s in data["sessions"])
        assert all("prediction_count" in s for s in data["sessions"])
    
    def test_list_sessions_pagination(self, authenticated_user):
        """Test pagination of session list"""
        # Get initial count (may have leftover sessions from other tests)
        response = client.get(
            "/api/sessions",
            headers=authenticated_user["headers"]
        )
        initial_total = response.json()["total"]
        
        # Create 5 new sessions
        for i in range(5):
            client.post(
                "/api/sessions",
                headers=authenticated_user["headers"],
                json={"name": f"Pagination Test Session {i}"}
            )
        
        expected_total = initial_total + 5
        
        # Get first page
        response = client.get(
            "/api/sessions?page=1&page_size=2",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["sessions"]) == 2
        assert data["total"] == expected_total
        assert data["page"] == 1
        assert data["page_size"] == 2
        
        # Get second page
        response = client.get(
            "/api/sessions?page=2&page_size=2",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["sessions"]) == 2
        assert data["total"] == expected_total
        assert data["page"] == 2
    
    def test_list_sessions_user_isolation(self, authenticated_user, second_authenticated_user):
        """Test that users only see their own sessions"""
        # Get initial counts for both users
        response1 = client.get(
            "/api/sessions",
            headers=authenticated_user["headers"]
        )
        initial_user1_total = response1.json()["total"]
        
        response2 = client.get(
            "/api/sessions",
            headers=second_authenticated_user["headers"]
        )
        initial_user2_total = response2.json()["total"]
        
        # User 1 creates session
        response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "User 1 Isolation Test Session"}
        )
        user1_session_id = response.json()["id"]
        
        # User 2 creates session
        response = client.post(
            "/api/sessions",
            headers=second_authenticated_user["headers"],
            json={"name": "User 2 Isolation Test Session"}
        )
        user2_session_id = response.json()["id"]
        
        # User 1 should see their initial sessions plus 1 new session
        response = client.get(
            "/api/sessions",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total"] == initial_user1_total + 1
        # Verify the new session is present
        session_ids = [s["id"] for s in data["sessions"]]
        assert user1_session_id in session_ids
        assert user2_session_id not in session_ids  # User 2's session should NOT appear


class TestCreateSessionEndpoint:
    """Tests for POST /api/sessions endpoint"""
    
    def test_create_session_requires_auth(self):
        """Test that creating sessions requires authentication"""
        response = client.post(
            "/api/sessions",
            json={"name": "Test Session"}
        )
        assert response.status_code == 401
    
    def test_create_session_success(self, authenticated_user):
        """Test successful session creation"""
        response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "My Test Project"}
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["name"] == "My Test Project"
        assert data["user_id"] == authenticated_user["user_id"]
        assert "id" in data
        assert "created_at" in data
        assert data["prediction_count"] == 0
        assert data["total_size"] == 0
    
    def test_create_session_validates_name(self, authenticated_user):
        """Test that session name validation works"""
        # Empty name
        response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": ""}
        )
        assert response.status_code == 422
        
        # Too long name
        response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "A" * 256}
        )
        assert response.status_code == 422
    
    def test_create_session_trims_whitespace(self, authenticated_user):
        """Test that session names are trimmed"""
        response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "  Trimmed Name  "}
        )
        
        assert response.status_code == 201
        assert response.json()["name"] == "Trimmed Name"


class TestGetSessionEndpoint:
    """Tests for GET /api/sessions/{id} endpoint"""
    
    def test_get_session_requires_auth(self, authenticated_user):
        """Test that getting session requires authentication"""
        # Create session first
        create_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Test Session"}
        )
        session_id = create_response.json()["id"]
        
        # Try to get without auth
        response = client.get(f"/api/sessions/{session_id}")
        assert response.status_code == 401
    
    def test_get_session_success(self, authenticated_user):
        """Test successful session retrieval"""
        # Create session
        create_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Test Session"}
        )
        session_id = create_response.json()["id"]
        
        # Get session
        response = client.get(
            f"/api/sessions/{session_id}",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == session_id
        assert data["name"] == "Test Session"
        assert data["user_id"] == authenticated_user["user_id"]
        assert "prediction_count" in data
        assert "total_size" in data
    
    def test_get_session_not_found(self, authenticated_user):
        """Test getting non-existent session"""
        response = client.get(
            "/api/sessions/nonexistent-id",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 404
    
    def test_get_session_cross_user_denied(self, authenticated_user, second_authenticated_user):
        """Test that users cannot access other users' sessions"""
        # User 1 creates session
        create_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "User 1 Session"}
        )
        session_id = create_response.json()["id"]
        
        # User 2 tries to access it
        response = client.get(
            f"/api/sessions/{session_id}",
            headers=second_authenticated_user["headers"]
        )
        
        assert response.status_code == 404


class TestUpdateSessionEndpoint:
    """Tests for PUT /api/sessions/{id} endpoint"""
    
    def test_update_session_requires_auth(self, authenticated_user):
        """Test that updating session requires authentication"""
        # Create session
        create_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Test Session"}
        )
        session_id = create_response.json()["id"]
        
        # Try to update without auth
        response = client.put(
            f"/api/sessions/{session_id}",
            json={"name": "Updated Name"}
        )
        assert response.status_code == 401
    
    def test_update_session_success(self, authenticated_user):
        """Test successful session update"""
        # Create session
        create_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Original Name"}
        )
        session_id = create_response.json()["id"]
        
        # Update session
        response = client.put(
            f"/api/sessions/{session_id}",
            headers=authenticated_user["headers"],
            json={"name": "Updated Name"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == session_id
        assert data["name"] == "Updated Name"
    
    def test_update_session_validates_name(self, authenticated_user):
        """Test name validation on update"""
        # Create session
        create_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Test Session"}
        )
        session_id = create_response.json()["id"]
        
        # Try empty name
        response = client.put(
            f"/api/sessions/{session_id}",
            headers=authenticated_user["headers"],
            json={"name": ""}
        )
        assert response.status_code == 422
    
    def test_update_session_cross_user_denied(self, authenticated_user, second_authenticated_user):
        """Test that users cannot update other users' sessions"""
        # User 1 creates session
        create_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "User 1 Session"}
        )
        session_id = create_response.json()["id"]
        
        # User 2 tries to update it
        response = client.put(
            f"/api/sessions/{session_id}",
            headers=second_authenticated_user["headers"],
            json={"name": "Hacked Name"}
        )
        
        assert response.status_code == 404


class TestDeleteSessionEndpoint:
    """Tests for DELETE /api/sessions/{id} endpoint"""
    
    def test_delete_session_requires_auth(self, authenticated_user):
        """Test that deleting session requires authentication"""
        # Create session
        create_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Test Session"}
        )
        session_id = create_response.json()["id"]
        
        # Try to delete without auth
        response = client.delete(f"/api/sessions/{session_id}")
        assert response.status_code == 401
    
    def test_delete_session_success(self, authenticated_user):
        """Test successful session deletion"""
        # Create session
        create_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Test Session"}
        )
        session_id = create_response.json()["id"]
        
        # Delete session
        response = client.delete(
            f"/api/sessions/{session_id}",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 204
        
        # Verify session is gone
        get_response = client.get(
            f"/api/sessions/{session_id}",
            headers=authenticated_user["headers"]
        )
        assert get_response.status_code == 404
    
    def test_delete_session_not_found(self, authenticated_user):
        """Test deleting non-existent session"""
        response = client.delete(
            "/api/sessions/nonexistent-id",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 404
    
    def test_delete_session_cross_user_denied(self, authenticated_user, second_authenticated_user):
        """Test that users cannot delete other users' sessions"""
        # User 1 creates session
        create_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "User 1 Session"}
        )
        session_id = create_response.json()["id"]
        
        # User 2 tries to delete it
        response = client.delete(
            f"/api/sessions/{session_id}",
            headers=second_authenticated_user["headers"]
        )
        
        assert response.status_code == 404
        
        # Verify session still exists for user 1
        get_response = client.get(
            f"/api/sessions/{session_id}",
            headers=authenticated_user["headers"]
        )
        assert get_response.status_code == 200


class TestAuthenticationEnforcement:
    """Tests for authentication enforcement on all endpoints"""
    
    def test_all_endpoints_require_auth(self):
        """Test that all session endpoints require authentication"""
        endpoints = [
            ("GET", "/api/sessions"),
            ("POST", "/api/sessions"),
            ("GET", "/api/sessions/test-id"),
            ("PUT", "/api/sessions/test-id"),
            ("DELETE", "/api/sessions/test-id"),
            ("GET", "/api/sessions/test-id/predictions"),
            ("POST", "/api/sessions/test-id/predictions"),
        ]
        
        for method, endpoint in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                if "predictions" in endpoint:
                    response = client.post(endpoint, json={"sequence": "ACDEFG"})
                else:
                    response = client.post(endpoint, json={"name": "Test"})
            elif method == "PUT":
                response = client.put(endpoint, json={"name": "Test"})
            elif method == "DELETE":
                response = client.delete(endpoint)
            
            assert response.status_code == 401, f"{method} {endpoint} should require auth"


class TestListSessionPredictionsEndpoint:
    """Tests for GET /api/sessions/{id}/predictions endpoint"""
    
    def test_list_predictions_requires_auth(self, authenticated_user):
        """Test that listing predictions requires authentication"""
        # Create session
        create_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Test Session"}
        )
        session_id = create_response.json()["id"]
        
        # Try to list predictions without auth
        response = client.get(f"/api/sessions/{session_id}/predictions")
        assert response.status_code == 401
    
    def test_list_predictions_empty(self, authenticated_user):
        """Test listing predictions when session has none"""
        # Create session
        create_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Empty Session"}
        )
        session_id = create_response.json()["id"]
        
        # List predictions
        response = client.get(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["predictions"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["page_size"] == 20
    
    def test_list_predictions_with_data(self, authenticated_user):
        """Test listing predictions with data"""
        # Create session
        session_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Test Session"}
        )
        session_id = session_response.json()["id"]
        
        # Create predictions in session
        sequences = ["ACDEFGH", "ACDEFGHI", "ACDEFGHIK"]
        for seq in sequences:
            pred_response = client.post(
                f"/api/sessions/{session_id}/predictions",
                headers=authenticated_user["headers"],
                json={"sequence": seq}
            )
            assert pred_response.status_code == 201, f"Failed to create prediction: {pred_response.json()}"
        
        # List predictions
        response = client.get(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["predictions"]) == 3
        assert data["total"] == 3
        assert all("id" in p for p in data["predictions"])
        assert all("sequence" in p for p in data["predictions"])
        assert all("status" in p for p in data["predictions"])
    
    def test_list_predictions_pagination(self, authenticated_user):
        """Test pagination of prediction list"""
        # Create session
        session_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Pagination Test Session"}
        )
        session_id = session_response.json()["id"]
        
        # Create 5 predictions
        sequences = ["ACDEFGH", "ACDEFGHI", "ACDEFGHIK", "ACDEFGHIKL", "ACDEFGHIKLM"]
        for seq in sequences:
            pred_response = client.post(
                f"/api/sessions/{session_id}/predictions",
                headers=authenticated_user["headers"],
                json={"sequence": seq}
            )
            assert pred_response.status_code == 201, f"Failed to create prediction: {pred_response.json()}"
        
        # Get first page
        response = client.get(
            f"/api/sessions/{session_id}/predictions?page=1&page_size=2",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["predictions"]) == 2
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["page_size"] == 2
        
        # Get second page
        response = client.get(
            f"/api/sessions/{session_id}/predictions?page=2&page_size=2",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["predictions"]) == 2
        assert data["total"] == 5
        assert data["page"] == 2
    
    def test_list_predictions_session_not_found(self, authenticated_user):
        """Test listing predictions for non-existent session"""
        response = client.get(
            "/api/sessions/nonexistent-id/predictions",
            headers=authenticated_user["headers"]
        )
        
        assert response.status_code == 404
    
    def test_list_predictions_cross_user_denied(self, authenticated_user, second_authenticated_user):
        """Test that users cannot list predictions from other users' sessions"""
        # User 1 creates session
        session_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "User 1 Session"}
        )
        session_id = session_response.json()["id"]
        
        # User 2 tries to list predictions
        response = client.get(
            f"/api/sessions/{session_id}/predictions",
            headers=second_authenticated_user["headers"]
        )
        
        assert response.status_code == 404


class TestCreateSessionPredictionEndpoint:
    """Tests for POST /api/sessions/{id}/predictions endpoint"""
    
    def test_create_prediction_requires_auth(self, authenticated_user):
        """Test that creating prediction requires authentication"""
        # Create session
        session_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Test Session"}
        )
        session_id = session_response.json()["id"]
        
        # Try to create prediction without auth
        response = client.post(
            f"/api/sessions/{session_id}/predictions",
            json={"sequence": "ACDEFGH"}
        )
        assert response.status_code == 401
    
    def test_create_prediction_success(self, authenticated_user):
        """Test successful prediction creation in session"""
        # Create session
        session_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Test Session"}
        )
        session_id = session_response.json()["id"]
        
        # Create prediction
        response = client.post(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "ACDEFGHIKLMNPQRSTVWY"}
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert "id" in data
        assert data["sequence"] == "ACDEFGHIKLMNPQRSTVWY"
        assert data["status"] == "pending"
        assert "configuration" in data
        assert "created_at" in data
    
    def test_create_prediction_with_configuration(self, authenticated_user):
        """Test creating prediction with custom configuration"""
        # Create session
        session_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Test Session"}
        )
        session_id = session_response.json()["id"]
        
        # Create prediction with config
        response = client.post(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"],
            json={
                "sequence": "ACDEFGHIKLM",
                "configuration": {
                    "iterations": 500,
                    "agents": 5,
                    "diversity": "balanced"
                }
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["configuration"]["iterations"] == 500
        assert data["configuration"]["agents"] == 5
        assert data["configuration"]["diversity"] == "balanced"
    
    def test_create_prediction_validates_sequence(self, authenticated_user):
        """Test that sequence validation works"""
        # Create session
        session_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Test Session"}
        )
        session_id = session_response.json()["id"]
        
        # Try invalid sequence (too short)
        response = client.post(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "AC"}
        )
        assert response.status_code == 422
        
        # Try invalid sequence (invalid characters)
        response = client.post(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "ACDEFG123"}
        )
        assert response.status_code == 422
    
    def test_create_prediction_session_not_found(self, authenticated_user):
        """Test creating prediction in non-existent session"""
        response = client.post(
            "/api/sessions/nonexistent-id/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "ACDEFGH"}
        )
        
        assert response.status_code == 404
    
    def test_create_prediction_cross_user_denied(self, authenticated_user, second_authenticated_user):
        """Test that users cannot create predictions in other users' sessions"""
        # User 1 creates session
        session_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "User 1 Session"}
        )
        session_id = session_response.json()["id"]
        
        # User 2 tries to create prediction
        response = client.post(
            f"/api/sessions/{session_id}/predictions",
            headers=second_authenticated_user["headers"],
            json={"sequence": "ACDEFGH"}
        )
        
        assert response.status_code == 404
    
    def test_create_prediction_updates_session_activity(self, authenticated_user):
        """Test that creating prediction updates session activity timestamp"""
        # Create session
        session_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Activity Test Session"}
        )
        session_id = session_response.json()["id"]
        initial_activity = session_response.json()["last_active_at"]
        
        # Wait a moment
        import time
        time.sleep(0.1)
        
        # Create prediction
        client.post(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "ACDEFGH"}
        )
        
        # Check session was updated
        session_response = client.get(
            f"/api/sessions/{session_id}",
            headers=authenticated_user["headers"]
        )
        
        assert session_response.status_code == 200
        new_activity = session_response.json()["last_active_at"]
        
        # Activity timestamp should be updated
        assert new_activity > initial_activity
        
        # Prediction count should be 1
        assert session_response.json()["prediction_count"] == 1
    
    def test_prediction_linked_to_session(self, authenticated_user):
        """Test that created prediction is properly linked to session"""
        # Create session
        session_response = client.post(
            "/api/sessions",
            headers=authenticated_user["headers"],
            json={"name": "Link Test Session"}
        )
        session_id = session_response.json()["id"]
        
        # Create prediction
        pred_response = client.post(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "ACDEFGH"}
        )
        prediction_id = pred_response.json()["id"]
        
        # List predictions in session
        list_response = client.get(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"]
        )
        
        assert list_response.status_code == 200
        predictions = list_response.json()["predictions"]
        
        # Verify prediction is in the list
        prediction_ids = [p["id"] for p in predictions]
        assert prediction_id in prediction_ids
