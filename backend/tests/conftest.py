"""
Pytest configuration and shared fixtures
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture(scope="function", autouse=True)
def reset_services():
    """Reset service state between tests"""
    from app.services.prediction_service import prediction_service
    from app.services.campaign_service import campaign_service
    
    # Clear in-memory storage
    prediction_service._predictions = {}
    campaign_service._campaigns = {}
    
    yield
    
    # Cleanup after test
    prediction_service._predictions = {}
    campaign_service._campaigns = {}


@pytest.fixture(autouse=True)
def mock_celery_task():
    """Mock Celery task execution for API tests"""
    with patch('app.tasks.run_prediction.delay') as mock_delay:
        # Mock task with an ID
        mock_task = Mock()
        mock_task.id = 'test-task-123'
        mock_delay.return_value = mock_task
        yield mock_delay


@pytest.fixture
def sample_sequence():
    """Sample protein sequence for testing"""
    return "MQIFVKTLTGKTITLEVEPSDTIENVK"


@pytest.fixture
def valid_prediction_config():
    """Valid prediction configuration"""
    return {
        "iterations": 1000,
        "agents": 10,
        "diversity": "balanced",
        "enable_checkpointing": True,
        "checkpoint_interval": 50
    }


@pytest.fixture
def valid_campaign_config():
    """Valid campaign configuration"""
    return {
        "iterations_per_phase": 500,
        "agents": 5,
        "quality_thresholds": {
            "rmsd": 5.0,
            "energy": -50.0
        }
    }


@pytest.fixture
def mock_prediction_results():
    """Mock prediction results data"""
    return {
        "best_energy": -123.45,
        "best_rmsd": 2.5,
        "conformations_explored": 1000,
        "runtime_seconds": 120.5,
        "convergence_achieved": True
    }


@pytest.fixture
def mock_conformation():
    """Mock protein conformation"""
    return {
        "energy": -100.0,
        "rmsd_to_native": 3.5,
        "atoms": [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 1.5, "y": 0.0, "z": 0.0},
            {"x": 3.0, "y": 0.0, "z": 0.0}
        ]
    }


# Async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
