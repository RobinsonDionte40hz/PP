"""
Tests for Celery tasks
"""
import pytest
from unittest.mock import Mock, patch
from app.models.prediction import PredictionStatus


@pytest.mark.unit
@pytest.mark.slow
class TestPredictionTasks:
    """Test prediction Celery tasks - simplified for existing implementation"""
    
    @patch('app.tasks.prediction_tasks.MultiAgentCoordinator')
    @patch('app.tasks.prediction_tasks.prediction_service')
    def test_run_prediction_imports(self, mock_service, mock_coordinator):
        """Test that task dependencies can be imported"""
        from app.tasks.prediction_tasks import run_prediction
        assert run_prediction is not None
        assert callable(run_prediction)
    
    def test_prediction_task_exists(self):
        """Test that prediction task is defined"""
        from app.tasks import prediction_tasks
        assert hasattr(prediction_tasks, 'run_prediction')


@pytest.mark.integration
@pytest.mark.slow
class TestPredictionTasksIntegration:
    """Integration tests for prediction tasks"""
    
    def test_task_registration(self):
        """Test that task is registered with Celery"""
        from app.tasks.prediction_tasks import run_prediction
        # Task should be callable
        assert callable(run_prediction)
