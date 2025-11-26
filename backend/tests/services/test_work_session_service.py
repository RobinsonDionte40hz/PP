"""
Unit tests for WorkSessionService
"""
import pytest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import shutil

from app.services.work_session_service import WorkSessionService
from app.services.file_storage_service import FileStorageService
import app.models  # Import all models to register them with Base
from app.models.work_session import WorkSession
from app.models.shared_export import SharedExport
from app.models.prediction import Prediction, PredictionStatus
from app.models.user import User
from app.database import SessionLocal, Base, engine


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test"""
    # Create all tables (includes users, work_sessions, shared_exports, predictions)
    Base.metadata.create_all(bind=engine)
    
    session = SessionLocal()
    yield session
    
    # Cleanup
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def temp_storage():
    """Create temporary storage directory"""
    temp_dir = Path(tempfile.mkdtemp())
    file_storage = FileStorageService(base_path=str(temp_dir))
    
    yield file_storage
    
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def work_session_service(temp_storage, db_session):
    """Create WorkSessionService with temporary storage and test database session"""
    return WorkSessionService(file_storage=temp_storage, db=db_session)


@pytest.fixture
def sample_user_id():
    """Generate a sample user ID"""
    return str(uuid.uuid4())


# ========== Session CRUD Tests ==========

def test_create_session_success(work_session_service, sample_user_id, temp_storage):
    """Test creating a new work session"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Test Session"
    )
    
    # Verify database record
    assert session.id is not None
    assert session.user_id == sample_user_id
    assert session.name == "Test Session"
    assert session.created_at is not None
    assert session.last_active_at is not None
    
    # Verify file system directory exists
    session_dir = temp_storage.get_session_directory(sample_user_id, session.id)
    assert session_dir.exists()
    assert session_dir.is_dir()


def test_create_session_empty_name(work_session_service, sample_user_id):
    """Test creating session with empty name raises ValueError"""
    with pytest.raises(ValueError, match="Session name cannot be empty"):
        work_session_service.create_session(
            user_id=sample_user_id,
            name=""
        )


def test_create_session_name_too_long(work_session_service, sample_user_id):
    """Test creating session with name > 255 chars raises ValueError"""
    long_name = "A" * 256
    with pytest.raises(ValueError, match="Session name cannot exceed 255 characters"):
        work_session_service.create_session(
            user_id=sample_user_id,
            name=long_name
        )


def test_create_session_strips_whitespace(work_session_service, sample_user_id):
    """Test that session names are stripped of leading/trailing whitespace"""
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="  Test Session  "
    )
    
    assert session.name == "Test Session"


def test_get_session_success(work_session_service, sample_user_id):
    """Test retrieving an existing session"""
    # Create session
    created = work_session_service.create_session(
        user_id=sample_user_id,
        name="Test Session"
    )
    
    # Retrieve session
    retrieved = work_session_service.get_session(
        session_id=created.id,
        user_id=sample_user_id
    )
    
    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.name == "Test Session"


def test_get_session_wrong_user(work_session_service, sample_user_id):
    """Test that users cannot access other users' sessions"""
    # Create session for user A
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="User A Session"
    )
    
    # Try to access as user B
    other_user_id = str(uuid.uuid4())
    retrieved = work_session_service.get_session(
        session_id=session.id,
        user_id=other_user_id
    )
    
    assert retrieved is None


def test_get_session_nonexistent(work_session_service, sample_user_id):
    """Test retrieving non-existent session returns None"""
    retrieved = work_session_service.get_session(
        session_id=str(uuid.uuid4()),
        user_id=sample_user_id
    )
    
    assert retrieved is None


def test_list_sessions_empty(work_session_service, sample_user_id):
    """Test listing sessions when user has none"""
    sessions, total = work_session_service.list_sessions(user_id=sample_user_id)
    
    assert sessions == []
    assert total == 0


def test_list_sessions_multiple(work_session_service, sample_user_id):
    """Test listing multiple sessions for a user"""
    # Create 3 sessions
    for i in range(3):
        work_session_service.create_session(
            user_id=sample_user_id,
            name=f"Session {i+1}"
        )
    
    # List sessions
    sessions, total = work_session_service.list_sessions(user_id=sample_user_id)
    
    assert len(sessions) == 3
    assert total == 3
    
    # Verify all belong to the user
    for session in sessions:
        assert session.user_id == sample_user_id


def test_list_sessions_user_isolation(work_session_service):
    """Test that list_sessions only returns sessions for the specified user"""
    user_a = str(uuid.uuid4())
    user_b = str(uuid.uuid4())
    
    # Create sessions for user A
    work_session_service.create_session(user_id=user_a, name="User A Session 1")
    work_session_service.create_session(user_id=user_a, name="User A Session 2")
    
    # Create sessions for user B
    work_session_service.create_session(user_id=user_b, name="User B Session 1")
    
    # List for user A
    sessions_a, total_a = work_session_service.list_sessions(user_id=user_a)
    assert len(sessions_a) == 2
    assert total_a == 2
    
    # List for user B
    sessions_b, total_b = work_session_service.list_sessions(user_id=user_b)
    assert len(sessions_b) == 1
    assert total_b == 1


def test_list_sessions_pagination(work_session_service, sample_user_id):
    """Test pagination in list_sessions"""
    # Create 5 sessions
    for i in range(5):
        work_session_service.create_session(
            user_id=sample_user_id,
            name=f"Session {i+1}"
        )
    
    # Get page 1 (2 items)
    page1, total = work_session_service.list_sessions(
        user_id=sample_user_id,
        page=1,
        page_size=2
    )
    assert len(page1) == 2
    assert total == 5
    
    # Get page 2 (2 items)
    page2, total = work_session_service.list_sessions(
        user_id=sample_user_id,
        page=2,
        page_size=2
    )
    assert len(page2) == 2
    assert total == 5
    
    # Get page 3 (1 item)
    page3, total = work_session_service.list_sessions(
        user_id=sample_user_id,
        page=3,
        page_size=2
    )
    assert len(page3) == 1
    assert total == 5


def test_update_session_success(work_session_service, sample_user_id):
    """Test updating a session name"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Old Name"
    )
    
    # Store original updated_at
    original_updated_at = session.updated_at
    
    # Sleep to ensure timestamp changes (SQLite datetime has limited precision)
    import time
    time.sleep(0.1)
    
    # Update name
    updated = work_session_service.update_session(
        session_id=session.id,
        user_id=sample_user_id,
        name="New Name"
    )
    
    assert updated is not None
    assert updated.name == "New Name"
    assert updated.updated_at > original_updated_at


def test_update_session_wrong_user(work_session_service, sample_user_id):
    """Test that users cannot update other users' sessions"""
    # Create session for user A
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="User A Session"
    )
    
    # Try to update as user B
    other_user_id = str(uuid.uuid4())
    updated = work_session_service.update_session(
        session_id=session.id,
        user_id=other_user_id,
        name="Hacked Name"
    )
    
    assert updated is None


def test_update_session_empty_name(work_session_service, sample_user_id):
    """Test updating session with empty name raises ValueError"""
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Valid Name"
    )
    
    with pytest.raises(ValueError, match="Session name cannot be empty"):
        work_session_service.update_session(
            session_id=session.id,
            user_id=sample_user_id,
            name=""
        )


def test_delete_session_success(work_session_service, sample_user_id, temp_storage):
    """Test deleting a session"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="To Delete"
    )
    
    session_dir = temp_storage.get_session_directory(sample_user_id, session.id)
    assert session_dir.exists()
    
    # Delete session
    result = work_session_service.delete_session(
        session_id=session.id,
        user_id=sample_user_id
    )
    
    assert result is True
    
    # Verify database record is gone
    retrieved = work_session_service.get_session(
        session_id=session.id,
        user_id=sample_user_id
    )
    assert retrieved is None
    
    # Verify directory is gone
    assert not session_dir.exists()


def test_delete_session_wrong_user(work_session_service, sample_user_id):
    """Test that users cannot delete other users' sessions"""
    # Create session for user A
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="User A Session"
    )
    
    # Try to delete as user B
    other_user_id = str(uuid.uuid4())
    result = work_session_service.delete_session(
        session_id=session.id,
        user_id=other_user_id
    )
    
    assert result is False
    
    # Verify session still exists
    retrieved = work_session_service.get_session(
        session_id=session.id,
        user_id=sample_user_id
    )
    assert retrieved is not None


def test_delete_session_nonexistent(work_session_service, sample_user_id):
    """Test deleting non-existent session returns False"""
    result = work_session_service.delete_session(
        session_id=str(uuid.uuid4()),
        user_id=sample_user_id
    )
    
    assert result is False


# ========== Prediction Operations Tests ==========

def test_create_prediction_in_session(work_session_service, sample_user_id, db_session):
    """Test linking a prediction to a session"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Test Session"
    )
    
    # Create prediction
    prediction = Prediction(
        id=f"pred_{uuid.uuid4().hex[:12]}",
        sequence="ACDEFG",
        status=PredictionStatus.PENDING.value
    )
    db_session.add(prediction)
    db_session.commit()
    
    # Link prediction to session
    result = work_session_service.create_prediction_in_session(
        session_id=session.id,
        user_id=sample_user_id,
        prediction=prediction
    )
    
    assert result is True
    assert prediction.session_id == session.id


def test_create_prediction_in_session_updates_activity(work_session_service, sample_user_id, db_session):
    """Test that linking a prediction updates session activity timestamp"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Test Session"
    )
    original_activity = session.last_active_at
    
    # Create prediction
    prediction = Prediction(
        id=f"pred_{uuid.uuid4().hex[:12]}",
        sequence="ACDEFG",
        status=PredictionStatus.PENDING.value
    )
    db_session.add(prediction)
    db_session.commit()
    
    # Link prediction (after a small delay)
    import time
    time.sleep(0.01)
    
    work_session_service.create_prediction_in_session(
        session_id=session.id,
        user_id=sample_user_id,
        prediction=prediction
    )
    
    # Retrieve session and check timestamp
    updated_session = work_session_service.get_session(session.id, sample_user_id)
    assert updated_session.last_active_at > original_activity


def test_create_prediction_wrong_user(work_session_service, sample_user_id, db_session):
    """Test that predictions cannot be linked to other users' sessions"""
    # Create session for user A
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="User A Session"
    )
    
    # Create prediction
    prediction = Prediction(
        id=f"pred_{uuid.uuid4().hex[:12]}",
        sequence="ACDEFG",
        status=PredictionStatus.PENDING.value
    )
    db_session.add(prediction)
    db_session.commit()
    
    # Try to link as user B
    other_user_id = str(uuid.uuid4())
    result = work_session_service.create_prediction_in_session(
        session_id=session.id,
        user_id=other_user_id,
        prediction=prediction
    )
    
    assert result is False
    assert prediction.session_id is None


def test_get_session_predictions(work_session_service, sample_user_id, db_session):
    """Test retrieving predictions for a session"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Test Session"
    )
    
    # Create and link 3 predictions
    for i in range(3):
        prediction = Prediction(
            id=f"pred_{uuid.uuid4().hex[:12]}",
            sequence=f"ACDEFG{i}",
            status=PredictionStatus.PENDING.value,
            session_id=session.id
        )
        db_session.add(prediction)
    db_session.commit()
    
    # Retrieve predictions
    predictions, total = work_session_service.get_session_predictions(
        session_id=session.id,
        user_id=sample_user_id
    )
    
    assert len(predictions) == 3
    assert total == 3


def test_get_session_predictions_pagination(work_session_service, sample_user_id, db_session):
    """Test pagination when retrieving session predictions"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Test Session"
    )
    
    # Create 5 predictions
    for i in range(5):
        prediction = Prediction(
            id=f"pred_{uuid.uuid4().hex[:12]}",
            sequence=f"ACDEFG{i}",
            status=PredictionStatus.PENDING.value,
            session_id=session.id
        )
        db_session.add(prediction)
    db_session.commit()
    
    # Get page 1 (2 items)
    page1, total = work_session_service.get_session_predictions(
        session_id=session.id,
        user_id=sample_user_id,
        page=1,
        page_size=2
    )
    assert len(page1) == 2
    assert total == 5


def test_update_session_activity(work_session_service, sample_user_id):
    """Test manually updating session activity timestamp"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Test Session"
    )
    original_activity = session.last_active_at
    
    import time
    time.sleep(0.01)
    
    # Update activity
    result = work_session_service.update_session_activity(
        session_id=session.id,
        user_id=sample_user_id
    )
    
    assert result is True
    
    # Verify timestamp changed
    updated_session = work_session_service.get_session(session.id, sample_user_id)
    assert updated_session.last_active_at > original_activity


# ========== Archive Tests ==========

def test_create_session_archive(work_session_service, sample_user_id, db_session, temp_storage):
    """Test creating a ZIP archive of a session"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Archive Test"
    )
    
    # Create a prediction with some files
    prediction_dir = temp_storage.get_prediction_directory(
        sample_user_id, session.id, "pred_test"
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / "results.json").write_text('{"test": "data"}')
    
    # Create archive
    zip_path = work_session_service.create_session_archive(
        session_id=session.id,
        user_id=sample_user_id
    )
    
    assert zip_path is not None
    assert zip_path.exists()
    assert zip_path.suffix == ".zip"
    
    # Cleanup
    if zip_path.exists():
        zip_path.unlink()


def test_create_session_archive_wrong_user(work_session_service, sample_user_id):
    """Test that users cannot create archives of other users' sessions"""
    # Create session for user A
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="User A Session"
    )
    
    # Try to create archive as user B
    other_user_id = str(uuid.uuid4())
    zip_path = work_session_service.create_session_archive(
        session_id=session.id,
        user_id=other_user_id
    )
    
    assert zip_path is None


# ========== Share Link Tests ==========

def test_create_share_link(work_session_service, sample_user_id):
    """Test creating a share link for a session"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Shared Session"
    )
    
    # Get current time before creating share link (naive for SQLite compatibility)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    # Create share link
    share = work_session_service.create_share_link(
        session_id=session.id,
        user_id=sample_user_id,
        expires_in_hours=24
    )
    
    assert share is not None
    assert share.share_id is not None
    assert share.session_id == session.id
    assert share.access_count == 0
    # SQLite returns naive datetime, so compare with naive
    expires_at = share.expires_at.replace(tzinfo=None) if share.expires_at.tzinfo else share.expires_at
    assert expires_at > now


def test_create_share_link_custom_expiration(work_session_service, sample_user_id):
    """Test creating share link with custom expiration"""
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Shared Session"
    )
    
    # Use timezone-naive datetime for SQLite compatibility
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    share = work_session_service.create_share_link(
        session_id=session.id,
        user_id=sample_user_id,
        expires_in_hours=48
    )
    
    # Check expiration is approximately 48 hours from now (SQLite returns naive datetime)
    expected_expiration = now + timedelta(hours=48)
    expires_at = share.expires_at.replace(tzinfo=None) if share.expires_at.tzinfo else share.expires_at
    time_diff = abs((expires_at - expected_expiration).total_seconds())
    assert time_diff < 5  # Within 5 seconds


def test_create_share_link_invalid_expiration(work_session_service, sample_user_id):
    """Test that invalid expiration times raise ValueError"""
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Shared Session"
    )
    
    # Too short
    with pytest.raises(ValueError, match="Expiration time must be at least 1 hour"):
        work_session_service.create_share_link(
            session_id=session.id,
            user_id=sample_user_id,
            expires_in_hours=0
        )
    
    # Too long
    with pytest.raises(ValueError, match="Expiration time cannot exceed 168 hours"):
        work_session_service.create_share_link(
            session_id=session.id,
            user_id=sample_user_id,
            expires_in_hours=169
        )


def test_get_shared_session(work_session_service, sample_user_id, db_session):
    """Test accessing a shared session via share link"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Shared Session"
    )
    
    # Add a prediction
    prediction = Prediction(
        id=f"pred_{uuid.uuid4().hex[:12]}",
        sequence="ACDEFG",
        status=PredictionStatus.COMPLETED.value,
        session_id=session.id
    )
    db_session.add(prediction)
    db_session.commit()
    
    # Create share link
    share = work_session_service.create_share_link(
        session_id=session.id,
        user_id=sample_user_id,
        expires_in_hours=24
    )
    
    # Access shared session
    shared_data = work_session_service.get_shared_session(share.share_id)
    
    assert shared_data is not None
    assert shared_data["id"] == session.id
    assert shared_data["name"] == "Shared Session"
    assert shared_data["prediction_count"] == 1
    assert len(shared_data["predictions"]) == 1
    assert shared_data["shared_link"]["share_id"] == share.share_id


def test_get_shared_session_increments_access_count(work_session_service, sample_user_id):
    """Test that accessing a shared session increments the access count"""
    # Create session and share link
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Shared Session"
    )
    share = work_session_service.create_share_link(
        session_id=session.id,
        user_id=sample_user_id,
        expires_in_hours=24
    )
    
    # Access multiple times
    for i in range(3):
        shared_data = work_session_service.get_shared_session(share.share_id)
        assert shared_data is not None
        assert shared_data["shared_link"]["access_count"] == i + 1


def test_get_shared_session_nonexistent(work_session_service):
    """Test accessing non-existent share link returns None"""
    shared_data = work_session_service.get_shared_session(str(uuid.uuid4()))
    assert shared_data is None


def test_get_shared_session_expired(work_session_service, sample_user_id, db_session):
    """Test that expired share links return None"""
    # Create session and share link
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Shared Session"
    )
    share = work_session_service.create_share_link(
        session_id=session.id,
        user_id=sample_user_id,
        expires_in_hours=1
    )
    
    # Manually expire the link
    share_obj = db_session.query(SharedExport).filter(
        SharedExport.share_id == share.share_id
    ).first()
    share_obj.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    db_session.commit()
    
    # Try to access
    shared_data = work_session_service.get_shared_session(share.share_id)
    assert shared_data is None


def test_cleanup_expired_shares(work_session_service, sample_user_id, db_session):
    """Test cleanup of expired share links"""
    # Create 2 sessions with share links
    session1 = work_session_service.create_session(
        user_id=sample_user_id,
        name="Session 1"
    )
    share1 = work_session_service.create_share_link(
        session_id=session1.id,
        user_id=sample_user_id,
        expires_in_hours=24
    )
    
    session2 = work_session_service.create_session(
        user_id=sample_user_id,
        name="Session 2"
    )
    share2 = work_session_service.create_share_link(
        session_id=session2.id,
        user_id=sample_user_id,
        expires_in_hours=24
    )
    
    # Expire share1
    share1_obj = db_session.query(SharedExport).filter(
        SharedExport.share_id == share1.share_id
    ).first()
    share1_obj.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    db_session.commit()
    
    # Run cleanup
    count = work_session_service.cleanup_expired_shares()
    
    assert count == 1
    
    # Verify share1 is gone, share2 remains
    assert work_session_service.get_shared_session(share1.share_id) is None
    assert work_session_service.get_shared_session(share2.share_id) is not None


# ========== Utility Tests ==========

def test_get_session_size(work_session_service, sample_user_id, temp_storage):
    """Test calculating session size"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Size Test"
    )
    
    # Create some files
    prediction_dir = temp_storage.get_prediction_directory(
        sample_user_id, session.id, "pred_test"
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / "file1.txt").write_text("A" * 1000)  # 1000 bytes
    (prediction_dir / "file2.txt").write_text("B" * 2000)  # 2000 bytes
    
    # Get size
    size = work_session_service.get_session_size(session.id, sample_user_id)
    
    assert size == 3000


def test_get_session_with_stats(work_session_service, sample_user_id, db_session, temp_storage):
    """Test getting session with statistics"""
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Stats Test"
    )
    
    # Add predictions
    for i in range(2):
        prediction = Prediction(
            id=f"pred_{uuid.uuid4().hex[:12]}",
            sequence=f"ACDEFG{i}",
            status=PredictionStatus.PENDING.value,
            session_id=session.id
        )
        db_session.add(prediction)
    db_session.commit()
    
    # Create some files
    prediction_dir = temp_storage.get_prediction_directory(
        sample_user_id, session.id, "pred_test"
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (prediction_dir / "file.txt").write_text("A" * 500)
    
    # Get stats
    stats = work_session_service.get_session_with_stats(session.id, sample_user_id)
    
    assert stats is not None
    assert stats["id"] == session.id
    assert stats["prediction_count"] == 2
    assert stats["total_size_bytes"] == 500


# ========== Property Tests ==========

def test_property_session_names_persisted(work_session_service, sample_user_id):
    """
    Property 2: Session names are persisted correctly
    Validates: Requirements 1.2
    """
    # Create session with specific name
    original_name = "Test Session With Special Chars: 測試"
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name=original_name
    )
    
    # Retrieve and verify
    retrieved = work_session_service.get_session(session.id, sample_user_id)
    assert retrieved.name == original_name


def test_property_user_isolation(work_session_service):
    """
    Property 3: User isolation in session queries
    Validates: Requirements 1.3, 6.3
    """
    user_a = str(uuid.uuid4())
    user_b = str(uuid.uuid4())
    
    # Create sessions for each user
    session_a = work_session_service.create_session(user_id=user_a, name="User A")
    session_b = work_session_service.create_session(user_id=user_b, name="User B")
    
    # User A can only see their session
    retrieved_a = work_session_service.get_session(session_a.id, user_a)
    assert retrieved_a is not None
    
    retrieved_b_as_a = work_session_service.get_session(session_b.id, user_a)
    assert retrieved_b_as_a is None
    
    # User B can only see their session
    retrieved_b = work_session_service.get_session(session_b.id, user_b)
    assert retrieved_b is not None
    
    retrieved_a_as_b = work_session_service.get_session(session_a.id, user_b)
    assert retrieved_a_as_b is None


def test_property_session_activity_updates(work_session_service, sample_user_id):
    """
    Property 5: Session activity updates timestamp
    Validates: Requirements 1.5, 10.5
    """
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Activity Test"
    )
    original_activity = session.last_active_at
    
    import time
    time.sleep(0.01)
    
    # Update activity
    work_session_service.update_session_activity(session.id, sample_user_id)
    
    # Verify timestamp increased
    updated = work_session_service.get_session(session.id, sample_user_id)
    assert updated.last_active_at > original_activity


def test_property_share_links_unique(work_session_service, sample_user_id):
    """
    Property 16: Share links have unique identifiers
    Validates: Requirements 5.1
    """
    # Create session
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Share Test"
    )
    
    # Create multiple share links
    share1 = work_session_service.create_share_link(
        session_id=session.id,
        user_id=sample_user_id,
        expires_in_hours=24
    )
    share2 = work_session_service.create_share_link(
        session_id=session.id,
        user_id=sample_user_id,
        expires_in_hours=24
    )
    
    # Verify unique IDs
    assert share1.share_id != share2.share_id


def test_property_expired_shares_deny_access(work_session_service, sample_user_id, db_session):
    """
    Property 18: Expired share links deny access
    Validates: Requirements 5.4
    """
    # Create session and share
    session = work_session_service.create_session(
        user_id=sample_user_id,
        name="Expiration Test"
    )
    share = work_session_service.create_share_link(
        session_id=session.id,
        user_id=sample_user_id,
        expires_in_hours=1
    )
    
    # Manually expire
    share_obj = db_session.query(SharedExport).filter(
        SharedExport.share_id == share.share_id
    ).first()
    share_obj.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    db_session.commit()
    
    # Verify access denied
    shared_data = work_session_service.get_shared_session(share.share_id)
    assert shared_data is None
