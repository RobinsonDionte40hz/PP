"""
Unit and integration tests for session cleanup service

Tests:
- Task 9.1: Cleanup service identifies and deletes expired sessions
- Task 9.2: Property test for complete cleanup (Property 23)
- Task 9.4: Unit tests for cleanup operations

Property Tests:
- Property 23: Cleanup removes all session data (Requirements 7.3, 7.4)
"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock, patch
import shutil

from app.services.session_cleanup_service import SessionCleanupService, get_cleanup_service
from app.models.work_session import WorkSession
from app.models.shared_export import SharedExport
from app.database import get_db


@pytest.fixture
def cleanup_service():
    """Create cleanup service with temporary storage"""
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_storage = Mock()
        mock_storage.get_session_directory = Mock(
            return_value=Path(temp_dir) / "user_123" / "sessions" / "session_456"
        )
        service = SessionCleanupService(file_storage=mock_storage)
        yield service


@pytest.fixture
def db_session():
    """Create database session for tests"""
    db = next(get_db())
    yield db
    db.close()


class TestIdentifyExpiredSessions:
    """Test identifying expired sessions based on retention period"""
    
    def test_identifies_expired_sessions(self, cleanup_service, db_session):
        """Test that expired sessions are correctly identified"""
        # Create test sessions with different activity dates
        now = datetime.now(timezone.utc)
        
        # Recently active (should NOT be expired)
        active_session = WorkSession(
            id="session-active",
            user_id="user-123",
            name="Active Session",
            created_at=now - timedelta(days=10),
            last_active_at=now - timedelta(days=5),
        )
        
        # Old inactive (should be expired)
        expired_session = WorkSession(
            id="session-expired",
            user_id="user-123",
            name="Expired Session",
            created_at=now - timedelta(days=200),
            last_active_at=now - timedelta(days=100),
        )
        
        db_session.add(active_session)
        db_session.add(expired_session)
        db_session.commit()
        
        # Identify expired (90 day retention)
        expired = cleanup_service.identify_expired_sessions(
            retention_days=90,
            db=db_session
        )
        
        # Verify only expired session found
        assert len(expired) == 1
        assert expired[0]["id"] == "session-expired"
        assert expired[0]["user_id"] == "user-123"
        assert expired[0]["days_inactive"] >= 100
        
        # Cleanup
        db_session.delete(active_session)
        db_session.delete(expired_session)
        db_session.commit()
    
    def test_retention_period_cutoff(self, cleanup_service, db_session):
        """Test that retention period cutoff is correctly applied"""
        now = datetime.now(timezone.utc)
        
        # Session exactly at cutoff (90 days ago)
        cutoff_session = WorkSession(
            id="session-cutoff",
            user_id="user-123",
            name="Cutoff Session",
            created_at=now - timedelta(days=100),
            last_active_at=now - timedelta(days=90),
        )
        
        # Session just before cutoff (89 days ago, should NOT expire)
        recent_session = WorkSession(
            id="session-recent",
            user_id="user-123",
            name="Recent Session",
            created_at=now - timedelta(days=100),
            last_active_at=now - timedelta(days=89),
        )
        
        db_session.add(cutoff_session)
        db_session.add(recent_session)
        db_session.commit()
        
        # Identify expired (90 day retention)
        expired = cleanup_service.identify_expired_sessions(
            retention_days=90,
            db=db_session
        )
        
        # Only session at/past cutoff should be expired
        expired_ids = [s["id"] for s in expired]
        assert "session-cutoff" in expired_ids
        assert "session-recent" not in expired_ids
        
        # Cleanup
        db_session.delete(cutoff_session)
        db_session.delete(recent_session)
        db_session.commit()
    
    def test_no_expired_sessions(self, cleanup_service, db_session):
        """Test when no sessions are expired"""
        now = datetime.now(timezone.utc)
        
        # All sessions recently active
        for i in range(3):
            session = WorkSession(
                id=f"session-{i}",
                user_id="user-123",
                name=f"Session {i}",
                created_at=now - timedelta(days=10),
                last_active_at=now - timedelta(days=5),
            )
            db_session.add(session)
        
        db_session.commit()
        
        # Identify expired
        expired = cleanup_service.identify_expired_sessions(
            retention_days=90,
            db=db_session
        )
        
        # No sessions should be expired
        assert len(expired) == 0
        
        # Cleanup
        for i in range(3):
            session = db_session.query(WorkSession).filter_by(id=f"session-{i}").first()
            if session:
                db_session.delete(session)
        db_session.commit()


class TestDeleteExpiredSessions:
    """Test deleting expired sessions with DB and file cleanup"""
    
    def test_delete_expired_sessions(self, cleanup_service, db_session):
        """Test complete deletion of expired sessions"""
        now = datetime.now(timezone.utc)
        
        # Create expired session with shared export
        expired_session = WorkSession(
            id="session-delete-test",
            user_id="user-456",
            name="To Delete",
            created_at=now - timedelta(days=200),
            last_active_at=now - timedelta(days=100),
        )
        
        share = SharedExport(
            share_id="share-123",
            session_id="session-delete-test",
            expires_at=now + timedelta(hours=24),
        )
        
        db_session.add(expired_session)
        db_session.add(share)
        db_session.commit()
        
        # Delete expired sessions
        stats = cleanup_service.delete_expired_sessions(
            retention_days=90,
            dry_run=False,
            db=db_session
        )
        
        # Verify statistics
        assert stats["sessions_identified"] == 1
        assert stats["sessions_deleted"] == 1
        assert stats["dry_run"] is False
        
        # Verify session deleted from DB
        session = db_session.query(WorkSession).filter_by(
            id="session-delete-test"
        ).first()
        assert session is None
        
        # Verify shared export deleted
        share = db_session.query(SharedExport).filter_by(
            share_id="share-123"
        ).first()
        assert share is None
    
    def test_dry_run_mode(self, cleanup_service, db_session):
        """Test dry run mode doesn't delete anything"""
        now = datetime.now(timezone.utc)
        
        expired_session = WorkSession(
            id="session-dry-run",
            user_id="user-789",
            name="Dry Run Test",
            created_at=now - timedelta(days=200),
            last_active_at=now - timedelta(days=100),
        )
        
        db_session.add(expired_session)
        db_session.commit()
        
        # Dry run
        stats = cleanup_service.delete_expired_sessions(
            retention_days=90,
            dry_run=True,
            db=db_session
        )
        
        # Verify stats show identified but not deleted
        assert stats["sessions_identified"] == 1
        assert stats["sessions_deleted"] == 0
        assert stats["dry_run"] is True
        
        # Verify session still exists
        session = db_session.query(WorkSession).filter_by(
            id="session-dry-run"
        ).first()
        assert session is not None
        
        # Cleanup
        db_session.delete(session)
        db_session.commit()
    
    def test_directory_deletion(self, db_session):
        """Test that session directories are deleted from filesystem"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create session directory structure
            user_dir = Path(temp_dir) / "user-dir-test"
            session_dir = user_dir / "sessions" / "session-dir-test"
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Create some files in session
            (session_dir / "results.json").write_text('{"data": "test"}')
            (session_dir / "checkpoint.pkl").write_text("checkpoint data")
            
            assert session_dir.exists()
            assert len(list(session_dir.iterdir())) == 2
            
            # Setup cleanup service with real paths
            mock_storage = Mock()
            mock_storage.get_session_directory = Mock(return_value=session_dir)
            cleanup_service = SessionCleanupService(file_storage=mock_storage)
            
            # Create expired session
            now = datetime.now(timezone.utc)
            expired_session = WorkSession(
                id="session-dir-test",
                user_id="user-dir-test",
                name="Directory Test",
                created_at=now - timedelta(days=200),
                last_active_at=now - timedelta(days=100),
            )
            
            db_session.add(expired_session)
            db_session.commit()
            
            # Delete expired sessions
            stats = cleanup_service.delete_expired_sessions(
                retention_days=90,
                dry_run=False,
                db=db_session
            )
            
            # Verify directory deleted
            assert stats["directories_deleted"] == 1
            assert not session_dir.exists()


class TestPropertyCleanupCompleteness:
    """Property 23: Cleanup removes all session data"""
    
    def test_property_cleanup_removes_all_data(self, db_session):
        """
        Property 23: Verify cleanup removes ALL session data
        
        This validates Requirements 7.3 and 7.4:
        - Database records (session, predictions, shares) deleted
        - File system directories removed
        - No orphaned data remains
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            now = datetime.now(timezone.utc)
            session_dir = Path(temp_dir) / "user-prop23" / "sessions" / "session-prop23"
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "data.json").write_text('{"test": "data"}')
            
            mock_storage = Mock()
            mock_storage.get_session_directory = Mock(return_value=session_dir)
            cleanup_service = SessionCleanupService(file_storage=mock_storage)
            
            # Create session with shared export
            expired_session = WorkSession(
                id="session-prop23",
                user_id="user-prop23",
                name="Property Test",
                created_at=now - timedelta(days=200),
                last_active_at=now - timedelta(days=100),
            )
            
            share = SharedExport(
                share_id="share-prop23",
                session_id="session-prop23",
                expires_at=now + timedelta(hours=24),
            )
            
            db_session.add(expired_session)
            db_session.add(share)
            db_session.commit()
            
            # Verify data exists before cleanup
            assert session_dir.exists()
            assert db_session.query(WorkSession).filter_by(
                id="session-prop23"
            ).first() is not None
            assert db_session.query(SharedExport).filter_by(
                share_id="share-prop23"
            ).first() is not None
            
            # Execute cleanup
            stats = cleanup_service.delete_expired_sessions(
                retention_days=90,
                dry_run=False,
                db=db_session
            )
            
            # Property 23 Assertion: ALL data removed
            # 1. Session record deleted
            assert db_session.query(WorkSession).filter_by(
                id="session-prop23"
            ).first() is None
            
            # 2. Shared export deleted
            assert db_session.query(SharedExport).filter_by(
                share_id="share-prop23"
            ).first() is None
            
            # 3. Directory deleted
            assert not session_dir.exists()
            
            # 4. Statistics confirm complete cleanup
            assert stats["sessions_deleted"] == 1
            assert stats["directories_deleted"] == 1
            assert len(stats["errors"]) == 0


class TestCleanupExpiredShares:
    """Test cleanup of expired share links (separate from sessions)"""
    
    def test_cleanup_expired_shares_only(self, cleanup_service, db_session):
        """Test that only expired shares are deleted, not sessions"""
        now = datetime.now(timezone.utc)
        
        # Create session (should NOT be deleted)
        session = WorkSession(
            id="session-share-test",
            user_id="user-share",
            name="Share Test",
            created_at=now - timedelta(days=10),
            last_active_at=now - timedelta(days=5),
        )
        
        # Expired share
        expired_share = SharedExport(
            share_id="share-expired",
            session_id="session-share-test",
            expires_at=now - timedelta(hours=1),  # Expired 1 hour ago
        )
        
        # Active share
        active_share = SharedExport(
            share_id="share-active",
            session_id="session-share-test",
            expires_at=now + timedelta(hours=24),  # Expires in 24 hours
        )
        
        db_session.add(session)
        db_session.add(expired_share)
        db_session.add(active_share)
        db_session.commit()
        
        # Cleanup expired shares
        stats = cleanup_service.cleanup_expired_shares(db=db_session)
        
        # Verify only expired share deleted
        assert stats["shares_deleted"] == 1
        
        # Session still exists
        assert db_session.query(WorkSession).filter_by(
            id="session-share-test"
        ).first() is not None
        
        # Expired share deleted
        assert db_session.query(SharedExport).filter_by(
            share_id="share-expired"
        ).first() is None
        
        # Active share still exists
        assert db_session.query(SharedExport).filter_by(
            share_id="share-active"
        ).first() is not None
        
        # Cleanup
        db_session.delete(session)
        db_session.delete(active_share)
        db_session.commit()


class TestCleanupServiceSingleton:
    """Test singleton pattern for cleanup service"""
    
    def test_get_cleanup_service_singleton(self):
        """Test that get_cleanup_service returns same instance"""
        service1 = get_cleanup_service()
        service2 = get_cleanup_service()
        
        assert service1 is service2


class TestErrorHandling:
    """Test error handling in cleanup operations"""
    
    def test_handles_file_deletion_errors(self, db_session):
        """Test that file deletion errors are logged but don't stop cleanup"""
        # Clean up any leftover sessions from previous tests
        db_session.query(WorkSession).delete()
        db_session.commit()
        
        now = datetime.now(timezone.utc)
        
        # Mock storage that raises error
        mock_storage = Mock()
        mock_storage.get_session_directory = Mock(side_effect=Exception("File error"))
        cleanup_service = SessionCleanupService(file_storage=mock_storage)
        
        # Create expired session
        expired_session = WorkSession(
            id="session-error-test",
            user_id="user-error",
            name="Error Test",
            created_at=now - timedelta(days=200),
            last_active_at=now - timedelta(days=100),
        )
        
        db_session.add(expired_session)
        db_session.commit()
        
        # Cleanup should continue despite error
        stats = cleanup_service.delete_expired_sessions(
            retention_days=90,
            dry_run=False,
            db=db_session
        )
        
        # Session deleted from DB despite file error
        assert stats["sessions_deleted"] == 1
        assert len(stats["errors"]) > 0
        
        # Verify session deleted
        session = db_session.query(WorkSession).filter_by(
            id="session-error-test"
        ).first()
        assert session is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
