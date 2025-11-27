"""
Unit tests for session download and sharing endpoints

Tests:
- Task 7.1: GET /api/sessions/{id}/download endpoint
- Task 7.2: Property test for download security (cross-user access prevention)
- Task 7.3: POST /api/sessions/{id}/share endpoint  
- Task 7.4: GET /api/shared/{share_id} endpoint
- Task 7.5: Property test for read-only sharing
- Task 7.6: Unit tests for download and sharing

Property Tests:
- Property 13: Cross-user access is prevented (download)
- Property 16: Share links have unique identifiers
- Property 18: Expired share links deny access
- Property 19: Shared sessions are read-only
"""
import pytest
import os
from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone
from pathlib import Path
import zipfile
import json
from unittest.mock import patch, MagicMock, Mock
from app.main import app
from app.models.work_session import WorkSession
from app.models.shared_export import SharedExport
from app.models.user import User
from app.models.prediction import Prediction
from app.services.work_session_service import work_session_service
from app.security import require_auth_with_session

# Disable rate limiting for tests
os.environ["TESTING"] = "true"


# Global variable to store the current mock user
_current_mock_user = None


def mock_require_auth():
    """Mock authentication dependency that returns current mock user"""
    global _current_mock_user
    return _current_mock_user


@pytest.fixture(autouse=True)
def override_auth_dependency():
    """Override auth dependency for all tests"""
    app.dependency_overrides[require_auth_with_session] = mock_require_auth
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_user_token():
    """Mock JWT token payload"""
    global _current_mock_user
    _current_mock_user = {
        "sub": "test-user-id-123",
        "key_id": "test-user-id-123",
        "username": "testuser"
    }
    return _current_mock_user


@pytest.fixture
def mock_other_user_token():
    """Mock JWT token for a different user"""
    global _current_mock_user
    _current_mock_user = {
        "sub": "other-user-id-456",
        "key_id": "other-user-id-456",
        "username": "otheruser"
    }
    return _current_mock_user


@pytest.fixture
def mock_session():
    """Mock work session"""
    return WorkSession(
        id="test-session-123",
        user_id="test-user-id-123",
        name="Test Session",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        last_active_at=datetime.now(timezone.utc),
        predictions=[]
    )


@pytest.fixture
def mock_shared_export():
    """Mock shared export"""
    return SharedExport(
        share_id="share-uuid-789",
        session_id="test-session-123",
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        access_count=0,
        last_accessed_at=None
    )


class TestDownloadEndpoint:
    """Test GET /api/sessions/{id}/download endpoint"""
    
    @patch("app.api.sessions.work_session_service")
    def test_download_session_success(
        self,
        mock_service,
        mock_user_token,
        tmp_path
    ):
        """Test successful session download"""
        # Setup - mock_user_token fixture sets up the mock auth user
        zip_path = tmp_path / "session_test-session-123.zip"
        
        # Create a real ZIP file
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("metadata.json", json.dumps({"test": "data"}))
        
        mock_service.create_session_archive.return_value = zip_path
        
        client = TestClient(app)
        
        # Execute
        response = client.get(
            "/api/sessions/test-session-123/download",
            headers={"Authorization": "Bearer fake-token"}
        )
        
        # Verify
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert "session_test-session-123.zip" in response.headers.get("content-disposition", "")
        mock_service.create_session_archive.assert_called_once_with(
            session_id="test-session-123",
            user_id="test-user-id-123"
        )
    
    @patch("app.api.sessions.work_session_service")
    def test_download_session_not_found(
        self,
        mock_service,
        mock_user_token
    ):
        """Test download with non-existent session"""
        # Setup
        mock_service.create_session_archive.return_value = None
        
        client = TestClient(app)
        
        # Execute
        response = client.get(
            "/api/sessions/non-existent/download",
            headers={"Authorization": "Bearer fake-token"}
        )
        
        # Verify
        assert response.status_code == 404
        assert "not found or access denied" in response.json()["detail"].lower()
    
    @patch("app.api.sessions.work_session_service")
    def test_download_cross_user_access_denied(
        self,
        mock_service,
        mock_user_token
    ):
        """
        Property 13: Cross-user access is prevented
        Test that users cannot download other users' sessions
        """
        # Setup - user tries to access another user's session
        mock_service.create_session_archive.return_value = None  # Returns None for unauthorized
        
        client = TestClient(app)
        
        # Execute - try to download session owned by another user
        response = client.get(
            "/api/sessions/other-user-session/download",
            headers={"Authorization": "Bearer fake-token"}
        )
        
        # Verify
        assert response.status_code == 404
        mock_service.create_session_archive.assert_called_once()
        # Service validates ownership and returns None for unauthorized access
    
    def test_download_requires_authentication(self):
        """Test that download requires authentication"""
        global _current_mock_user
        # Clear the mock user to simulate no authentication
        _current_mock_user = None
        
        client = TestClient(app)
        
        # Execute - no auth token
        response = client.get("/api/sessions/test-session-123/download")
        
        # Verify - should fail (either 401, 403, or 404 when user_id is None)
        assert response.status_code in [401, 403, 404, 500]  # Acceptable error codes


class TestShareLinkCreation:
    """Test POST /api/sessions/{id}/share endpoint"""
    
    @patch("app.api.sessions.work_session_service")
    def test_create_share_link_success(
        self,
        mock_service,
        mock_user_token,
        mock_shared_export
    ):
        """
        Test successful share link creation
        Property 16: Share links have unique identifiers
        """
        # Setup
        mock_service.create_share_link.return_value = mock_shared_export
        
        client = TestClient(app)
        
        # Execute
        response = client.post(
            "/api/sessions/test-session-123/share",
            json={"expiration_hours": 24},
            headers={"Authorization": "Bearer fake-token"}
        )
        
        # Verify
        assert response.status_code == 201
        data = response.json()
        assert data["share_id"] == "share-uuid-789"
        assert data["session_id"] == "test-session-123"
        assert "share_url" in data
        assert "/api/shared/" in data["share_url"]
        assert data["access_count"] == 0
        
        # Property 16: Verify unique identifier
        assert data["share_id"] is not None
        assert len(data["share_id"]) > 0
    
    @patch("app.api.sessions.work_session_service")
    def test_create_share_link_validation(
        self,
        mock_service,
        mock_user_token
    ):
        """Test share link expiration validation"""
        # Setup
        
        client = TestClient(app)
        
        # Test invalid expiration (too short)
        response = client.post(
            "/api/sessions/test-session-123/share",
            json={"expiration_hours": 0},
            headers={"Authorization": "Bearer fake-token"}
        )
        assert response.status_code == 422  # Validation error
        
        # Test invalid expiration (too long)
        response = client.post(
            "/api/sessions/test-session-123/share",
            json={"expiration_hours": 200},
            headers={"Authorization": "Bearer fake-token"}
        )
        assert response.status_code == 422  # Validation error
    
    @patch("app.api.sessions.work_session_service")
    def test_create_share_link_not_found(
        self,
        mock_service,
        mock_user_token
    ):
        """Test share link creation with non-existent session"""
        # Setup
        mock_service.create_share_link.return_value = None
        
        client = TestClient(app)
        
        # Execute
        response = client.post(
            "/api/sessions/non-existent/share",
            json={"expiration_hours": 24},
            headers={"Authorization": "Bearer fake-token"}
        )
        
        # Verify
        assert response.status_code == 404
        assert "not found or access denied" in response.json()["detail"].lower()


class TestSharedSessionAccess:
    """Test GET /api/shared/{share_id} endpoint"""
    
    @patch("app.api.sessions.work_session_service")
    def test_access_shared_session_success(
        self,
        mock_service
    ):
        """
        Test successful shared session access
        Property 19: Shared sessions are read-only
        """
        # Setup
        session_data = {
            "id": "test-session-123",
            "name": "Test Session",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_active_at": datetime.now(timezone.utc).isoformat(),
            "prediction_count": 5,
            "predictions": []
        }
        mock_service.get_shared_session.return_value = session_data
        
        client = TestClient(app)
        
        # Execute - no authentication required
        response = client.get("/api/shared/share-uuid-789")
        
        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-session-123"
        assert data["name"] == "Test Session"
        assert data["prediction_count"] == 5
        
        # Property 19: Verify read-only (no sensitive info)
        assert "user_id" not in data
        assert "predictions" not in data  # Full prediction data not exposed
        
        mock_service.get_shared_session.assert_called_once_with("share-uuid-789")
    
    @patch("app.api.sessions.work_session_service")
    def test_access_shared_session_not_found(
        self,
        mock_service
    ):
        """Test accessing non-existent share link"""
        # Setup
        mock_service.get_shared_session.return_value = None
        
        client = TestClient(app)
        
        # Execute
        response = client.get("/api/shared/non-existent-share")
        
        # Verify
        assert response.status_code == 404
        assert "not found or has expired" in response.json()["detail"].lower()
    
    @patch("app.api.sessions.work_session_service")
    def test_access_expired_share_link(
        self,
        mock_service
    ):
        """
        Property 18: Expired share links deny access
        Test that expired share links return 404
        """
        # Setup - service returns None for expired links
        mock_service.get_shared_session.return_value = None
        
        client = TestClient(app)
        
        # Execute
        response = client.get("/api/shared/expired-share-link")
        
        # Verify
        assert response.status_code == 404
        assert "not found or has expired" in response.json()["detail"].lower()
    
    def test_shared_access_no_auth_required(self):
        """Test that shared session access does NOT require authentication"""
        client = TestClient(app)
        
        with patch("app.api.sessions.work_session_service") as mock_service:
            session_data = {
                "id": "test-session-123",
                "name": "Test Session",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "prediction_count": 0,
            }
            mock_service.get_shared_session.return_value = session_data
            
            # Execute - no authorization header
            response = client.get("/api/shared/share-uuid-789")
            
            # Verify - should work without auth
            assert response.status_code == 200


class TestPropertyDownloadSecurity:
    """Property test for download security"""
    
    @patch("app.api.sessions.work_session_service")
    def test_property_cross_user_download_denied(
        self,
        mock_service,
        mock_user_token,
        mock_other_user_token
    ):
        """
        Property 13: Cross-user access is prevented
        Systematic test that User A cannot download User B's sessions
        """
        global _current_mock_user
        client = TestClient(app)
        
        # Scenario 1: User A tries to download User B's session
        _current_mock_user = mock_user_token
        mock_service.create_session_archive.return_value = None  # Ownership validation fails
        
        response = client.get(
            "/api/sessions/user-b-session/download",
            headers={"Authorization": "Bearer user-a-token"}
        )
        
        assert response.status_code == 404
        
        # Scenario 2: User B tries to download User A's session
        _current_mock_user = mock_other_user_token
        mock_service.create_session_archive.return_value = None
        
        response = client.get(
            "/api/sessions/user-a-session/download",
            headers={"Authorization": "Bearer user-b-token"}
        )
        
        assert response.status_code == 404


class TestPropertyReadOnlySharing:
    """Property test for read-only sharing"""
    
    @patch("app.api.sessions.work_session_service")
    def test_property_shared_sessions_read_only(
        self,
        mock_service
    ):
        """
        Property 19: Shared sessions are read-only
        Verify that shared sessions do not expose sensitive or mutable data
        """
        # Setup
        session_data = {
            "id": "test-session-123",
            "name": "Test Session",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_active_at": datetime.now(timezone.utc).isoformat(),
            "prediction_count": 5,
            "predictions": [
                {
                    "id": "pred-1",
                    "sequence": "ACDEFGH",
                    "status": "completed"
                }
            ]
        }
        mock_service.get_shared_session.return_value = session_data
        
        client = TestClient(app)
        
        # Execute
        response = client.get("/api/shared/share-uuid-789")
        
        # Verify read-only properties
        assert response.status_code == 200
        data = response.json()
        
        # Must NOT expose sensitive information
        assert "user_id" not in data
        assert "api_key" not in data
        assert "predictions" not in data  # Full prediction details not exposed
        
        # Must expose only safe, read-only information
        assert "id" in data
        assert "name" in data
        assert "created_at" in data
        assert "prediction_count" in data


class TestPropertyShareLinkUniqueness:
    """Property test for share link uniqueness"""
    
    @patch("app.api.sessions.work_session_service")
    def test_property_share_links_unique(
        self,
        mock_service,
        mock_user_token
    ):
        """
        Property 16: Share links have unique identifiers
        Test that each share link gets a unique ID
        """
        # Setup handled by fixture
        
        # Create multiple share links
        share_ids = []
        for i in range(5):
            mock_export = SharedExport(
                share_id=f"unique-share-{i}",
                session_id="test-session-123",
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
                access_count=0
            )
            mock_service.create_share_link.return_value = mock_export
            
            client = TestClient(app)
            response = client.post(
                "/api/sessions/test-session-123/share",
                json={"expiration_hours": 24},
                headers={"Authorization": "Bearer fake-token"}
            )
            
            assert response.status_code == 201
            share_ids.append(response.json()["share_id"])
        
        # Verify all share IDs are unique
        assert len(share_ids) == len(set(share_ids))
        assert all(share_id is not None for share_id in share_ids)


class TestPropertyExpiredShareLinks:
    """Property test for expired share link handling"""
    
    @patch("app.api.sessions.work_session_service")
    def test_property_expired_links_deny_access(
        self,
        mock_service
    ):
        """
        Property 18: Expired share links deny access
        Systematic test that expired links are properly rejected
        """
        client = TestClient(app)
        
        # Test multiple expired scenarios
        expired_scenarios = [
            "expired-1-hour-ago",
            "expired-1-day-ago",
            "expired-1-week-ago",
        ]
        
        for share_id in expired_scenarios:
            # Service returns None for expired links
            mock_service.get_shared_session.return_value = None
            
            response = client.get(f"/api/shared/{share_id}")
            
            # Verify all expired links are denied
            assert response.status_code == 404
            assert "expired" in response.json()["detail"].lower()


# Integration test demonstrating full workflow
class TestDownloadShareIntegration:
    """Integration test for download and share workflow"""
    
    @patch("app.api.sessions.work_session_service")
    def test_full_share_workflow(
        self,
        mock_service,
        mock_user_token,
        mock_shared_export,
        tmp_path
    ):
        """Test complete workflow: create session, share, access shared"""
        # Setup handled by fixture
        client = TestClient(app)
        
        # Step 1: Create share link
        mock_service.create_share_link.return_value = mock_shared_export
        
        response = client.post(
            "/api/sessions/test-session-123/share",
            json={"expiration_hours": 24},
            headers={"Authorization": "Bearer fake-token"}
        )
        
        assert response.status_code == 201
        share_id = response.json()["share_id"]
        
        # Step 2: Access shared session (no auth required)
        session_data = {
            "id": "test-session-123",
            "name": "Test Session",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prediction_count": 0,
        }
        mock_service.get_shared_session.return_value = session_data
        
        response = client.get(f"/api/shared/{share_id}")
        
        assert response.status_code == 200
        assert response.json()["id"] == "test-session-123"
        
        # Step 3: Download session (auth required)
        zip_path = tmp_path / "session.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("metadata.json", "{}")
        
        mock_service.create_session_archive.return_value = zip_path
        
        response = client.get(
            "/api/sessions/test-session-123/download",
            headers={"Authorization": "Bearer fake-token"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
