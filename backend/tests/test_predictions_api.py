"""
Tests for prediction API endpoints
"""
import pytest
import os
import uuid
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Disable rate limiting for tests
os.environ["TESTING"] = "true"

from app.main import app
from app.database import Base, get_db
from app.models.user import User
from app.models.work_session import WorkSession
from app.models.prediction import Prediction, PredictionStatus
from app.services.auth_service import AuthService
from app.services.work_session_service import work_session_service

# Create test database
TEST_DATABASE_URL = "sqlite:///./test_predictions_api.db"
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
    with TestingSessionLocal() as db:
        try:
            db.execute(text("DELETE FROM predictions"))
        except:
            pass
        try:
            db.execute(text("DELETE FROM work_sessions"))
        except:
            pass
        db.commit()
        db.expunge_all()
        db.close()
    
    # Flush Redis before each test
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        for key in r.scan_iter("session:*"):
            r.delete(key)
    except Exception:
        pass
    
    yield
    
    with TestingSessionLocal() as db:
        try:
            db.execute(text("DELETE FROM predictions"))
        except:
            pass
        try:
            db.execute(text("DELETE FROM work_sessions"))
        except:
            pass
        db.commit()
        db.expunge_all()
        db.close()


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_db():
    """Remove test database file after all tests"""
    yield
    engine.dispose()
    import time
    if os.path.exists("./test_predictions_api.db"):
        time.sleep(0.1)
        try:
            os.remove("./test_predictions_api.db")
        except PermissionError:
            pass


@pytest.fixture(scope="function")
def authenticated_user():
    """Create a test user with a work session and return authentication token"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.flushdb()
    except Exception:
        pass
    
    unique_id = str(uuid.uuid4())[:8]
    username = f"testuser_{unique_id}"
    email = f"test_{unique_id}@example.com"
    
    with TestingSessionLocal() as db:
        success, message, user = AuthService.register_user(
            db=db,
            username=username,
            password="TestPass123!",
            email=email
        )
        
        if not success or user is None:
            raise Exception(f"Failed to register test user: {message}")
        
        user_id = user.key_id
    
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
    
    # Create a work session for this user to associate predictions with
    headers = {"Authorization": f"Bearer {data['tokens']['access_token']}"}
    session_response = client.post(
        "/api/sessions",
        headers=headers,
        json={"name": "Test Session"}
    )
    
    assert session_response.status_code == 201
    session_id = session_response.json()["id"]
    
    return {
        "user_id": data["user"]["key_id"],
        "token": data["tokens"]["access_token"],
        "headers": headers,
        "session_id": session_id
    }


@pytest.fixture(scope="function")
def second_authenticated_user():
    """Create a second test user for cross-user access tests"""
    unique_id = str(uuid.uuid4())[:8]
    username = f"testuser2_{unique_id}"
    email = f"test2_{unique_id}@example.com"
    
    with TestingSessionLocal() as db:
        success, message, user = AuthService.register_user(
            db=db,
            username=username,
            password="TestPass123!",
            email=email
        )
        
        if not success or user is None:
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
    
    headers = {"Authorization": f"Bearer {data['tokens']['access_token']}"}
    session_response = client.post(
        "/api/sessions",
        headers=headers,
        json={"name": "Second User Session"}
    )
    
    assert session_response.status_code == 201
    session_id = session_response.json()["id"]
    
    return {
        "user_id": data["user"]["key_id"],
        "token": data["tokens"]["access_token"],
        "headers": headers,
        "session_id": session_id
    }


class TestPredictionAPI:
    """Test prediction API endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_list_predictions_requires_auth(self):
        """Test that listing predictions requires authentication"""
        response = client.get("/api/predictions")
        assert response.status_code == 401
    
    def test_create_prediction_requires_auth(self):
        """Test that creating prediction requires authentication"""
        data = {"sequence": "MQIFVKTLTGKTITLEVEPSD"}
        response = client.post("/api/predictions", json=data)
        assert response.status_code == 401
    
    def test_list_predictions_empty(self, authenticated_user):
        """Test listing predictions when none exist for user"""
        response = client.get("/api/predictions", headers=authenticated_user["headers"])
        assert response.status_code == 200
        result = response.json()
        assert "predictions" in result
        assert result["total"] == 0
    
    def test_create_prediction_via_session(self, authenticated_user):
        """Test creating a prediction through session endpoint"""
        session_id = authenticated_user["session_id"]
        data = {
            "sequence": "MQIFVKTLTGKTITLEVEPSD",
            "configuration": {
                "iterations": 500,
                "agents": 5,
                "diversity": "balanced"
            }
        }
        
        response = client.post(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"],
            json=data
        )
        assert response.status_code == 201
        
        result = response.json()
        assert result["sequence"] == data["sequence"]
        assert result["session_id"] == session_id
        assert "id" in result
        assert result["id"].startswith("pred_")
    
    def test_list_predictions_with_data(self, authenticated_user):
        """Test listing predictions returns user's predictions"""
        session_id = authenticated_user["session_id"]
        
        # Create predictions through session
        sequences = ["ACDEFGH", "MKTAYVG", "GHIKLMN"]
        for seq in sequences:
            response = client.post(
                f"/api/sessions/{session_id}/predictions",
                headers=authenticated_user["headers"],
                json={"sequence": seq}
            )
            assert response.status_code == 201
        
        # List predictions
        response = client.get("/api/predictions", headers=authenticated_user["headers"])
        assert response.status_code == 200
        
        result = response.json()
        assert result["total"] == 3
        assert len(result["predictions"]) == 3
    
    def test_predictions_isolated_between_users(self, authenticated_user, second_authenticated_user):
        """Test that users only see their own predictions"""
        # User 1 creates predictions
        client.post(
            f"/api/sessions/{authenticated_user['session_id']}/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "ACDEFGH"}
        )
        client.post(
            f"/api/sessions/{authenticated_user['session_id']}/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "MKTAYVG"}
        )
        
        # User 2 creates prediction
        client.post(
            f"/api/sessions/{second_authenticated_user['session_id']}/predictions",
            headers=second_authenticated_user["headers"],
            json={"sequence": "GHIKLMN"}
        )
        
        # User 1 should only see their 2 predictions
        response1 = client.get("/api/predictions", headers=authenticated_user["headers"])
        assert response1.status_code == 200
        assert response1.json()["total"] == 2
        
        # User 2 should only see their 1 prediction
        response2 = client.get("/api/predictions", headers=second_authenticated_user["headers"])
        assert response2.status_code == 200
        assert response2.json()["total"] == 1
    
    def test_get_prediction_requires_ownership(self, authenticated_user, second_authenticated_user):
        """Test that users cannot access other users' predictions"""
        # User 1 creates prediction
        create_response = client.post(
            f"/api/sessions/{authenticated_user['session_id']}/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "ACDEFGH"}
        )
        prediction_id = create_response.json()["id"]
        
        # User 1 can access their own prediction
        response1 = client.get(
            f"/api/predictions/{prediction_id}",
            headers=authenticated_user["headers"]
        )
        assert response1.status_code == 200
        
        # User 2 cannot access User 1's prediction
        response2 = client.get(
            f"/api/predictions/{prediction_id}",
            headers=second_authenticated_user["headers"]
        )
        assert response2.status_code == 404
    
    def test_delete_prediction_requires_ownership(self, authenticated_user, second_authenticated_user):
        """Test that users cannot delete other users' predictions"""
        # User 1 creates prediction
        create_response = client.post(
            f"/api/sessions/{authenticated_user['session_id']}/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "ACDEFGH"}
        )
        prediction_id = create_response.json()["id"]
        
        # User 2 cannot delete User 1's prediction
        response = client.delete(
            f"/api/predictions/{prediction_id}",
            headers=second_authenticated_user["headers"]
        )
        assert response.status_code == 404
        
        # User 1 can delete their own prediction
        response = client.delete(
            f"/api/predictions/{prediction_id}",
            headers=authenticated_user["headers"]
        )
        assert response.status_code == 204
    
    def test_list_predictions_with_status_filter(self, authenticated_user):
        """Test listing predictions with status filter"""
        session_id = authenticated_user["session_id"]
        
        # Create a prediction
        client.post(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "ACDEFGH"}
        )
        
        # Filter by pending status
        response = client.get(
            f"/api/predictions?status={PredictionStatus.PENDING.value}",
            headers=authenticated_user["headers"]
        )
        assert response.status_code == 200
        result = response.json()
        for pred in result["predictions"]:
            assert pred["status"] in [PredictionStatus.PENDING.value, PredictionStatus.QUEUED.value]
    
    def test_pause_prediction_requires_ownership(self, authenticated_user, second_authenticated_user):
        """Test that users cannot pause other users' predictions"""
        session_id = authenticated_user["session_id"]
        
        # Create prediction and set to running
        create_response = client.post(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "ACDEFGHIKL"}
        )
        prediction_id = create_response.json()["id"]
        
        # Manually set to running
        from app.services.prediction_service import prediction_service
        from app.schemas.prediction import PredictionUpdateSchema
        prediction_service.update_prediction(
            prediction_id,
            PredictionUpdateSchema(status=PredictionStatus.RUNNING)
        )
        
        # User 2 cannot pause User 1's prediction
        response = client.post(
            f"/api/predictions/{prediction_id}/pause",
            headers=second_authenticated_user["headers"]
        )
        assert response.status_code == 400  # Not found returns 400 from service
        
        # User 1 can pause their own prediction
        response = client.post(
            f"/api/predictions/{prediction_id}/pause",
            headers=authenticated_user["headers"]
        )
        assert response.status_code == 200
        assert response.json()["status"] == PredictionStatus.PAUSED.value
    
    def test_stop_prediction_requires_ownership(self, authenticated_user, second_authenticated_user):
        """Test that users cannot stop other users' predictions"""
        session_id = authenticated_user["session_id"]
        
        # Create prediction and set to running
        create_response = client.post(
            f"/api/sessions/{session_id}/predictions",
            headers=authenticated_user["headers"],
            json={"sequence": "ACDEFGHIKL"}
        )
        prediction_id = create_response.json()["id"]
        
        from app.services.prediction_service import prediction_service
        from app.schemas.prediction import PredictionUpdateSchema
        prediction_service.update_prediction(
            prediction_id,
            PredictionUpdateSchema(status=PredictionStatus.RUNNING)
        )
        
        # User 2 cannot stop User 1's prediction
        response = client.post(
            f"/api/predictions/{prediction_id}/stop",
            headers=second_authenticated_user["headers"]
        )
        assert response.status_code == 400
        
        # User 1 can stop their own prediction
        response = client.post(
            f"/api/predictions/{prediction_id}/stop",
            headers=authenticated_user["headers"]
        )
        assert response.status_code == 200
        assert response.json()["status"] == PredictionStatus.STOPPED.value
    
    def test_get_prediction_not_found(self, authenticated_user):
        """Test getting non-existent prediction"""
        response = client.get(
            "/api/predictions/pred_nonexistent",
            headers=authenticated_user["headers"]
        )
        assert response.status_code == 404
    
    def test_list_predictions_pagination(self, authenticated_user):
        """Test prediction list pagination"""
        session_id = authenticated_user["session_id"]
        
        # Create multiple predictions
        for i in range(5):
            client.post(
                f"/api/sessions/{session_id}/predictions",
                headers=authenticated_user["headers"],
                json={"sequence": "ACDEFGHIKLMNPQRSTVWY"}
            )
        
        # Test page_size parameter
        response = client.get(
            "/api/predictions?page_size=3",
            headers=authenticated_user["headers"]
        )
        assert response.status_code == 200
        result = response.json()
        assert len(result["predictions"]) == 3
        assert result["total"] == 5
        assert result["page_size"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
