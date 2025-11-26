"""
Property tests for WorkSession model

Tests requirement 1.1: Session creation generates unique identifiers
"""
import pytest  # type: ignore[import-not-found]
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.models import User, WorkSession
from app.database import Base, engine
import uuid


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test"""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    yield session
    
    # Cleanup
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user(db_session: Session):
    """Create a test user"""
    user = User(
        key_id=str(uuid.uuid4()),
        username=f"testuser_{uuid.uuid4().hex[:8]}",
        email=f"test_{uuid.uuid4().hex[:8]}@example.com",
        password_hash="hashed_password",
        is_active=True,
        role="user"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


class TestWorkSessionModel:
    """Test WorkSession model properties"""
    
    def test_session_creation_generates_unique_identifiers(self, db_session: Session, test_user: User):
        """
        Property 1: Session creation generates unique identifiers
        Validates: Requirements 1.1
        
        Tests that each created session has a unique ID and is properly
        associated with the correct user.
        """
        # Create multiple sessions
        session_ids = []
        for i in range(10):
            session_id = str(uuid.uuid4())
            work_session = WorkSession(
                id=session_id,
                user_id=test_user.key_id,
                name=f"Test Session {i}"
            )
            db_session.add(work_session)
            session_ids.append(session_id)
        
        db_session.commit()
        
        # Verify all sessions have unique IDs
        assert len(session_ids) == len(set(session_ids)), "Session IDs are not unique"
        
        # Verify all sessions are associated with the user
        user_sessions = db_session.query(WorkSession).filter(
            WorkSession.user_id == test_user.key_id  # type: ignore[arg-type]
        ).all()
        
        assert len(user_sessions) == 10, "Not all sessions were created"
        
        for session in user_sessions:
            assert session.id in session_ids, "Session ID mismatch"
            assert session.user_id == test_user.key_id, "User ID mismatch"  # type: ignore[comparison-overlap]
    
    def test_session_name_is_persisted(self, db_session: Session, test_user: User):
        """
        Property 2: Session names are persisted correctly
        Validates: Requirements 1.2
        
        Tests that session names are stored and retrieved correctly.
        """
        session_id = str(uuid.uuid4())
        session_name = "My Protein Research Project"
        
        # Create session with name
        work_session = WorkSession(
            id=session_id,
            user_id=test_user.key_id,
            name=session_name
        )
        db_session.add(work_session)
        db_session.commit()
        
        # Retrieve and verify
        retrieved = db_session.query(WorkSession).filter(
            WorkSession.id == session_id  # type: ignore[arg-type]
        ).first()
        
        assert retrieved is not None, "Session not found"
        assert retrieved.name == session_name, "Session name mismatch"  # type: ignore[comparison-overlap]
    
    def test_timestamps_are_set_on_creation(self, db_session: Session, test_user: User):
        """
        Tests that created_at and last_active_at timestamps are set automatically
        Validates: Requirements 1.4
        """
        work_session = WorkSession(
            id=str(uuid.uuid4()),
            user_id=test_user.key_id,
            name="Timestamp Test Session"
        )
        db_session.add(work_session)
        db_session.commit()
        db_session.refresh(work_session)
        
        assert work_session.created_at is not None, "created_at not set"
        assert work_session.last_active_at is not None, "last_active_at not set"
        assert work_session.updated_at is not None, "updated_at not set"
        
        # Verify timestamps are datetime objects
        assert isinstance(work_session.created_at, datetime), "created_at is not datetime"
        assert isinstance(work_session.last_active_at, datetime), "last_active_at is not datetime"
        assert isinstance(work_session.updated_at, datetime), "updated_at is not datetime"
    
    def test_user_relationship(self, db_session: Session, test_user: User):
        """
        Tests that the relationship between WorkSession and User works correctly
        """
        work_session = WorkSession(
            id=str(uuid.uuid4()),
            user_id=test_user.key_id,
            name="Relationship Test"
        )
        db_session.add(work_session)
        db_session.commit()
        db_session.refresh(work_session)
        
        # Test relationship from session to user
        assert work_session.user is not None, "User relationship not working"
        assert work_session.user.key_id == test_user.key_id
        
        # Test relationship from user to sessions
        db_session.refresh(test_user)
        assert len(test_user.work_sessions) == 1
        assert test_user.work_sessions[0].id == work_session.id
    
    def test_to_dict_method(self, db_session: Session, test_user: User):
        """
        Tests that the to_dict method returns correct dictionary representation
        """
        session_id = str(uuid.uuid4())
        session_name = "Dict Test Session"
        
        work_session = WorkSession(
            id=session_id,
            user_id=test_user.key_id,
            name=session_name
        )
        db_session.add(work_session)
        db_session.commit()
        db_session.refresh(work_session)
        
        session_dict = work_session.to_dict()
        
        assert session_dict["id"] == session_id
        assert session_dict["user_id"] == test_user.key_id
        assert session_dict["name"] == session_name
        assert "created_at" in session_dict
        assert "updated_at" in session_dict
        assert "last_active_at" in session_dict
    
    def test_shared_export_is_expired(self, db_session: Session, test_user: User):
        """
        Tests that the SharedExport is_expired method works correctly
        """
        from app.models import SharedExport
        from datetime import timedelta
        
        # Create a work session first
        work_session = WorkSession(
            id=str(uuid.uuid4()),
            user_id=test_user.key_id,
            name="Test Session"
        )
        db_session.add(work_session)
        db_session.commit()
        
        # Test expired share
        expired_share = SharedExport(
            share_id=str(uuid.uuid4()),
            session_id=work_session.id,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        db_session.add(expired_share)
        db_session.commit()
        db_session.refresh(expired_share)
        
        assert expired_share.is_expired() is True, "Share should be expired"
        
        # Test valid share
        valid_share = SharedExport(
            share_id=str(uuid.uuid4()),
            session_id=work_session.id,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        db_session.add(valid_share)
        db_session.commit()
        db_session.refresh(valid_share)
        
        assert valid_share.is_expired() is False, "Share should not be expired"
