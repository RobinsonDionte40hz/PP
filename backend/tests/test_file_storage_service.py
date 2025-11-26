"""
Property-based and unit tests for FileStorageService (Task 2).

Tests Properties from session-based-storage design.md:
- Property 20: User directories are isolated
- Property 10: All prediction artifacts are created
- Property 14: ZIP archives contain all session data
"""
import pytest
import tempfile
import shutil
import zipfile
import json
from pathlib import Path
from datetime import datetime
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from app.services.file_storage_service import FileStorageService


# ==================== Custom Strategies ====================

@st.composite
def user_id_strategy(draw):
    """Generate valid user IDs (UUID format)"""
    import uuid
    return str(uuid.uuid4())


@st.composite
def session_id_strategy(draw):
    """Generate valid session IDs (UUID format)"""
    import uuid
    return str(uuid.uuid4())


@st.composite
def prediction_id_strategy(draw):
    """Generate valid prediction IDs"""
    import uuid
    return f"pred_{uuid.uuid4().hex[:12]}"


@st.composite
def session_name_strategy(draw):
    """Generate valid session names (1-255 chars)"""
    length = draw(st.integers(min_value=1, max_value=255))
    return draw(st.text(
        alphabet=st.characters(blacklist_categories=('Cs', 'Cc')),
        min_size=length,
        max_size=length
    ))


# ==================== Fixtures ====================

@pytest.fixture
def temp_storage():
    """Create temporary storage directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def file_storage(temp_storage):
    """Create FileStorageService with temporary directory"""
    return FileStorageService(base_path=temp_storage)


@pytest.fixture
def sample_artifacts():
    """Sample prediction artifacts for testing"""
    return {
        'results': {
            'energy': -123.45,
            'rmsd': 2.5,
            'convergence': True
        },
        'trajectory': {
            'steps': [
                {'energy': -100.0, 'iteration': 1},
                {'energy': -110.0, 'iteration': 2},
                {'energy': -123.45, 'iteration': 3}
            ]
        },
        'structure': "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\nEND",
        'visualization': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00'
    }


# ==================== Property 20: User directories are isolated ====================
# Feature: session-based-storage, Property 20: User directories are isolated

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    user1_id=user_id_strategy(),  # type: ignore[call-arg]
    user2_id=user_id_strategy(),  # type: ignore[call-arg]
    session_id=session_id_strategy(),  # type: ignore[call-arg]
)
def test_property_20_user_directories_isolated(
    file_storage,
    user1_id: str,
    user2_id: str,
    session_id: str
):
    """
    Property 20: For any two different users, their session data should be stored
    in separate directories under user_data/{user_id_1}/ and user_data/{user_id_2}/.
    
    Validates: Requirements 6.1
    """
    # Ensure users are different
    assume(user1_id != user2_id)
    
    # Get user directories
    user1_dir = file_storage.get_user_directory(user1_id)
    user2_dir = file_storage.get_user_directory(user2_id)
    
    # Verify directories are different
    assert user1_dir != user2_dir, "User directories must be distinct"
    
    # Verify each directory is under its respective user ID
    assert str(user1_dir).endswith(user1_id), f"User1 directory should contain user1 ID"
    assert str(user2_dir).endswith(user2_id), f"User2 directory should contain user2 ID"
    
    # Get session directories for both users
    session1_dir = file_storage.get_session_directory(user1_id, session_id)
    session2_dir = file_storage.get_session_directory(user2_id, session_id)
    
    # Verify session directories are isolated
    assert session1_dir != session2_dir, "Session directories must be isolated by user"
    assert user1_id in str(session1_dir), "Session1 directory should be under user1"
    assert user2_id in str(session2_dir), "Session2 directory should be under user2"
    
    # Verify no path traversal between users
    assert not str(session1_dir).startswith(str(user2_dir))
    assert not str(session2_dir).startswith(str(user1_dir))


# ==================== Property 10: All prediction artifacts are created ====================
# Feature: session-based-storage, Property 10: All prediction artifacts are created

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    user_id=user_id_strategy(),  # type: ignore[call-arg]
    session_id=session_id_strategy(),  # type: ignore[call-arg]
    prediction_id=prediction_id_strategy(),  # type: ignore[call-arg]
)
def test_property_10_all_artifacts_created(
    file_storage,
    sample_artifacts,
    user_id: str,
    session_id: str,
    prediction_id: str
):
    """
    Property 10: For any completed prediction, the files results.json, trajectory.json,
    structure.pdb, and visualization.png should all exist in the prediction directory.
    
    Validates: Requirements 2.5
    """
    # Create session directory first
    file_storage.create_session_directory(user_id, session_id)
    
    # Save prediction artifacts
    success = file_storage.save_prediction_artifacts(
        user_id, session_id, prediction_id, sample_artifacts
    )
    
    assert success, "Artifact saving should succeed"
    
    # Get prediction directory
    pred_dir = file_storage.get_prediction_directory(user_id, session_id, prediction_id)
    
    # Verify all required files exist
    required_files = ['results.json', 'trajectory.json', 'structure.pdb', 'visualization.png']
    for filename in required_files:
        file_path = pred_dir / filename
        assert file_path.exists(), f"Required artifact {filename} must exist"
        assert file_path.is_file(), f"{filename} must be a file"
        assert file_path.stat().st_size > 0, f"{filename} must not be empty"
    
    # Verify JSON files are valid
    results_data = json.loads((pred_dir / 'results.json').read_text())
    assert results_data == sample_artifacts['results']
    
    trajectory_data = json.loads((pred_dir / 'trajectory.json').read_text())
    assert trajectory_data == sample_artifacts['trajectory']
    
    # Verify PDB file content
    structure_content = (pred_dir / 'structure.pdb').read_text()
    assert structure_content == sample_artifacts['structure']
    
    # Verify binary visualization file
    viz_content = (pred_dir / 'visualization.png').read_bytes()
    assert viz_content == sample_artifacts['visualization']


# ==================== Property 14: ZIP archives contain all session data ====================
# Feature: session-based-storage, Property 14: ZIP archives contain all session data

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    user_id=user_id_strategy(),  # type: ignore[call-arg]
    session_id=session_id_strategy(),  # type: ignore[call-arg]
    num_predictions=st.integers(min_value=1, max_value=5),
)
def test_property_14_zip_contains_all_data(
    file_storage,
    sample_artifacts,
    user_id: str,
    session_id: str,
    num_predictions: int
):
    """
    Property 14: For any session with predictions, creating a download archive
    should produce a ZIP file containing metadata.json and all prediction
    subdirectories with their artifacts.
    
    Validates: Requirements 4.1, 4.2, 4.3
    """
    # Create session directory
    session_dir = file_storage.create_session_directory(user_id, session_id)
    
    # Create multiple predictions
    prediction_ids = [f"pred_{i}_{session_id[:8]}" for i in range(num_predictions)]
    
    for pred_id in prediction_ids:
        file_storage.save_prediction_artifacts(
            user_id, session_id, pred_id, sample_artifacts
        )
    
    # Create metadata file
    file_storage.create_session_metadata(
        user_id=user_id,
        session_id=session_id,
        session_name="Test Session",
        created_at=datetime.utcnow(),
        last_active_at=datetime.utcnow(),
        predictions=[{"id": pid, "status": "completed"} for pid in prediction_ids]
    )
    
    # Create ZIP archive
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        zip_path = Path(tmp.name)
    
    try:
        result_path = file_storage.create_zip_archive(user_id, session_id, zip_path)
        assert result_path == zip_path
        assert zip_path.exists()
        assert zip_path.stat().st_size > 0
        
        # Verify ZIP contents
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zip_files = zipf.namelist()
            
            # Verify metadata.json is included
            assert 'metadata.json' in zip_files, "metadata.json must be in archive"
            
            # Verify all prediction directories are included
            for pred_id in prediction_ids:
                # Check for prediction artifacts
                assert any(pred_id in f for f in zip_files), f"Prediction {pred_id} must be in archive"
                
                # Verify all required files for each prediction
                required_files = ['results.json', 'trajectory.json', 'structure.pdb', 'visualization.png']
                for filename in required_files:
                    expected_path = f"{pred_id}/{filename}"
                    assert expected_path in zip_files, f"{expected_path} must be in archive"
            
            # Verify we can extract and read metadata
            metadata_content = zipf.read('metadata.json')
            metadata = json.loads(metadata_content)
            assert metadata['session_id'] == session_id
            assert len(metadata['predictions']) == num_predictions
            
    finally:
        # Cleanup
        if zip_path.exists():
            zip_path.unlink()


# ==================== Unit Tests ====================

class TestFileStorageService:
    """Unit tests for FileStorageService"""
    
    def test_initialization(self, temp_storage):
        """Test service initialization creates base directory"""
        service = FileStorageService(base_path=temp_storage)
        assert service.base_path.exists()
        assert service.base_path.is_dir()
    
    def test_get_user_directory(self, file_storage):
        """Test getting user directory path"""
        user_id = "test-user-123"
        user_dir = file_storage.get_user_directory(user_id)
        assert user_id in str(user_dir)
        assert str(user_dir).startswith(str(file_storage.base_path))
    
    def test_get_session_directory(self, file_storage):
        """Test getting session directory path"""
        user_id = "test-user-123"
        session_id = "test-session-456"
        session_dir = file_storage.get_session_directory(user_id, session_id)
        
        assert user_id in str(session_dir)
        assert session_id in str(session_dir)
        assert "sessions" in str(session_dir)
    
    def test_get_prediction_directory(self, file_storage):
        """Test getting prediction directory path"""
        user_id = "test-user-123"
        session_id = "test-session-456"
        prediction_id = "pred_789"
        
        pred_dir = file_storage.get_prediction_directory(user_id, session_id, prediction_id)
        
        assert user_id in str(pred_dir)
        assert session_id in str(pred_dir)
        assert prediction_id in str(pred_dir)
    
    def test_create_session_directory(self, file_storage):
        """Test creating session directory"""
        user_id = "test-user-123"
        session_id = "test-session-456"
        
        session_dir = file_storage.create_session_directory(user_id, session_id)
        
        assert session_dir.exists()
        assert session_dir.is_dir()
    
    def test_create_session_directory_idempotent(self, file_storage):
        """Test creating same directory twice (race condition handling)"""
        user_id = "test-user-123"
        session_id = "test-session-456"
        
        # Create twice - should not raise error
        dir1 = file_storage.create_session_directory(user_id, session_id)
        dir2 = file_storage.create_session_directory(user_id, session_id)
        
        assert dir1 == dir2
        assert dir1.exists()
    
    def test_delete_session_directory(self, file_storage):
        """Test deleting session directory"""
        user_id = "test-user-123"
        session_id = "test-session-456"
        
        # Create then delete
        session_dir = file_storage.create_session_directory(user_id, session_id)
        assert session_dir.exists()
        
        success = file_storage.delete_session_directory(user_id, session_id)
        
        assert success
        assert not session_dir.exists()
    
    def test_delete_nonexistent_directory(self, file_storage):
        """Test deleting non-existent directory"""
        user_id = "test-user-123"
        session_id = "nonexistent-session"
        
        success = file_storage.delete_session_directory(user_id, session_id)
        
        assert not success
    
    def test_delete_directory_security_check(self, temp_storage, monkeypatch):
        """Test security check prevents deleting outside base path"""
        # Create service with specific base path
        service = FileStorageService(base_path=temp_storage)
        
        # Create a directory outside the base path
        outside_dir = Path(tempfile.mkdtemp())
        
        try:
            # Mock get_session_directory to return a path outside base_path
            def mock_get_session_directory(user_id, session_id):
                return outside_dir
            
            monkeypatch.setattr(service, 'get_session_directory', mock_get_session_directory)
            
            # This should fail security check
            with pytest.raises(ValueError, match="Invalid directory path"):
                service.delete_session_directory("user", "session")
                
        finally:
            if outside_dir.exists():
                shutil.rmtree(outside_dir)
    
    def test_save_prediction_artifacts_success(self, file_storage, sample_artifacts):
        """Test saving prediction artifacts successfully"""
        user_id = "test-user-123"
        session_id = "test-session-456"
        prediction_id = "pred_789"
        
        # Create session first
        file_storage.create_session_directory(user_id, session_id)
        
        success = file_storage.save_prediction_artifacts(
            user_id, session_id, prediction_id, sample_artifacts
        )
        
        assert success
        
        # Verify files exist
        pred_dir = file_storage.get_prediction_directory(user_id, session_id, prediction_id)
        assert (pred_dir / 'results.json').exists()
        assert (pred_dir / 'trajectory.json').exists()
        assert (pred_dir / 'structure.pdb').exists()
        assert (pred_dir / 'visualization.png').exists()
    
    def test_save_artifacts_partial_data(self, file_storage):
        """Test saving artifacts with missing optional files"""
        user_id = "test-user-123"
        session_id = "test-session-456"
        prediction_id = "pred_789"
        
        file_storage.create_session_directory(user_id, session_id)
        
        # Only provide results
        partial_artifacts = {
            'results': {'energy': -100.0}
        }
        
        success = file_storage.save_prediction_artifacts(
            user_id, session_id, prediction_id, partial_artifacts
        )
        
        assert success
        
        pred_dir = file_storage.get_prediction_directory(user_id, session_id, prediction_id)
        assert (pred_dir / 'results.json').exists()
    
    def test_save_artifacts_atomic_writes(self, file_storage, sample_artifacts):
        """Test that artifact writes are atomic"""
        user_id = "test-user-123"
        session_id = "test-session-456"
        prediction_id = "pred_789"
        
        file_storage.create_session_directory(user_id, session_id)
        
        # Save artifacts
        file_storage.save_prediction_artifacts(
            user_id, session_id, prediction_id, sample_artifacts
        )
        
        pred_dir = file_storage.get_prediction_directory(user_id, session_id, prediction_id)
        
        # Verify no .tmp files left behind
        tmp_files = list(pred_dir.glob('*.tmp'))
        assert len(tmp_files) == 0, "No temporary files should remain"
    
    def test_get_session_size(self, file_storage, sample_artifacts):
        """Test calculating session size"""
        user_id = "test-user-123"
        session_id = "test-session-456"
        
        file_storage.create_session_directory(user_id, session_id)
        
        # Initially empty
        size = file_storage.get_session_size(user_id, session_id)
        assert size == 0
        
        # Add prediction
        file_storage.save_prediction_artifacts(
            user_id, session_id, "pred_1", sample_artifacts
        )
        
        # Size should be positive
        size = file_storage.get_session_size(user_id, session_id)
        assert size > 0
    
    def test_get_session_size_nonexistent(self, file_storage):
        """Test getting size of non-existent session"""
        size = file_storage.get_session_size("fake-user", "fake-session")
        assert size == 0
    
    def test_create_session_metadata(self, file_storage):
        """Test creating session metadata file"""
        user_id = "test-user-123"
        session_id = "test-session-456"
        
        file_storage.create_session_directory(user_id, session_id)
        
        success = file_storage.create_session_metadata(
            user_id=user_id,
            session_id=session_id,
            session_name="Test Session",
            created_at=datetime(2025, 11, 26, 10, 0, 0),
            last_active_at=datetime(2025, 11, 26, 15, 0, 0),
            predictions=[{"id": "pred_1", "status": "completed"}]
        )
        
        assert success
        
        # Verify metadata file exists and is valid
        session_dir = file_storage.get_session_directory(user_id, session_id)
        metadata_path = session_dir / "metadata.json"
        
        assert metadata_path.exists()
        
        metadata = json.loads(metadata_path.read_text())
        assert metadata['session_id'] == session_id
        assert metadata['name'] == "Test Session"
        assert metadata['user_id'] == user_id
        assert len(metadata['predictions']) == 1
    
    def test_create_zip_archive(self, file_storage, sample_artifacts):
        """Test creating ZIP archive"""
        user_id = "test-user-123"
        session_id = "test-session-456"
        
        # Setup session with data
        file_storage.create_session_directory(user_id, session_id)
        file_storage.save_prediction_artifacts(
            user_id, session_id, "pred_1", sample_artifacts
        )
        
        # Create archive
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            zip_path = Path(tmp.name)
        
        try:
            result_path = file_storage.create_zip_archive(user_id, session_id, zip_path)
            
            assert result_path.exists()
            assert zipfile.is_zipfile(result_path)
            
            # Verify can open and read
            with zipfile.ZipFile(result_path, 'r') as zipf:
                files = zipf.namelist()
                assert len(files) > 0
                
        finally:
            if zip_path.exists():
                zip_path.unlink()
    
    def test_create_zip_archive_nonexistent_session(self, file_storage):
        """Test creating archive for non-existent session"""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            zip_path = Path(tmp.name)
        
        try:
            with pytest.raises(FileNotFoundError):
                file_storage.create_zip_archive("fake-user", "fake-session", zip_path)
        finally:
            if zip_path.exists():
                zip_path.unlink()
    
    def test_create_zip_archive_cleanup_on_failure(self, file_storage, monkeypatch):
        """Test that partial ZIP is cleaned up on failure"""
        user_id = "test-user-123"
        session_id = "test-session-456"
        
        file_storage.create_session_directory(user_id, session_id)
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            zip_path = Path(tmp.name)
        
        # Force failure by making session directory unreadable (mock)
        def mock_rglob(*args, **kwargs):
            raise OSError("Simulated failure")
        
        session_dir = file_storage.get_session_directory(user_id, session_id)
        monkeypatch.setattr(session_dir.__class__, 'rglob', mock_rglob)
        
        try:
            with pytest.raises(OSError):
                file_storage.create_zip_archive(user_id, session_id, zip_path)
            
            # Verify partial ZIP was cleaned up
            assert not zip_path.exists()
        finally:
            if zip_path.exists():
                zip_path.unlink()
