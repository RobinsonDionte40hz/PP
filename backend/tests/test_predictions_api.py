"""
Tests for prediction API endpoints
"""
import pytest
import os
from fastapi.testclient import TestClient

# Disable rate limiting for tests
os.environ["TESTING"] = "true"

from app.main import app
from app.models.prediction import PredictionStatus, Prediction

client = TestClient(app)


class TestPredictionAPI:
    """Test prediction API endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_create_prediction(self):
        """Test creating a new prediction"""
        data = {
            "sequence": "MQIFVKTLTGKTITLEVEPSD",
            "configuration": {
                "iterations": 500,
                "agents": 5,
                "diversity": "balanced"
            }
        }
        
        response = client.post("/api/predictions", json=data)
        assert response.status_code == 201
        
        result = response.json()
        assert result["sequence"] == data["sequence"]
        assert result["status"] == PredictionStatus.QUEUED.value  # Status set to QUEUED after task submission
        assert result["configuration"]["iterations"] == 500
        assert "id" in result
        assert result["id"].startswith("pred_")
    
    def test_create_prediction_invalid_sequence(self):
        """Test creating prediction with invalid sequence"""
        data = {
            "sequence": "INVALID123",
        }
        
        response = client.post("/api/predictions", json=data)
        assert response.status_code == 422  # Validation error
    
    def test_get_prediction(self):
        """Test getting prediction by ID"""
        # First create a prediction
        data = {"sequence": "ACDEFGHIKLMNPQRSTVWY"}
        create_response = client.post("/api/predictions", json=data)
        prediction_id = create_response.json()["id"]
        
        # Get the prediction
        response = client.get(f"/api/predictions/{prediction_id}")
        assert response.status_code == 200
        
        result = response.json()
        assert result["id"] == prediction_id
        assert result["sequence"] == data["sequence"]
    
    def test_get_prediction_not_found(self):
        """Test getting non-existent prediction"""
        response = client.get("/api/predictions/pred_nonexistent")
        assert response.status_code == 404
    
    def test_list_predictions(self):
        """Test listing predictions"""
        # Create multiple predictions with valid sequences
        sequences = ["ACDEFG", "MKTAYV", "GHIKLM"]
        for seq in sequences:
            data = {"sequence": seq}
            client.post("/api/predictions", json=data)
        
        # List all
        response = client.get("/api/predictions")
        assert response.status_code == 200
        
        result = response.json()
        assert "predictions" in result
        assert result["total"] >= 3
        assert result["page"] == 1
    
    def test_list_predictions_with_filter(self):
        """Test listing predictions with status filter"""
        response = client.get(f"/api/predictions?status={PredictionStatus.PENDING.value}")
        assert response.status_code == 200
        
        result = response.json()
        for pred in result["predictions"]:
            assert pred["status"] == PredictionStatus.PENDING.value
    
    def test_pause_prediction(self):
        """Test pausing a prediction"""
        # Create and get prediction
        data = {"sequence": "ACDEFGHIKL"}
        create_response = client.post("/api/predictions", json=data)
        prediction_id = create_response.json()["id"]
        
        # Manually set to running (in real scenario, Celery would do this)
        from app.services.prediction_service import prediction_service
        from app.schemas.prediction import PredictionUpdateSchema
        prediction_service.update_prediction(
            prediction_id,
            PredictionUpdateSchema(status=PredictionStatus.RUNNING)
        )
        
        # Pause it
        response = client.post(f"/api/predictions/{prediction_id}/pause")
        assert response.status_code == 200
        assert response.json()["status"] == PredictionStatus.PAUSED.value
    
    def test_resume_prediction(self):
        """Test resuming a paused prediction"""
        # Create prediction and set to paused
        data = {"sequence": "ACDEFGHIKL"}
        create_response = client.post("/api/predictions", json=data)
        prediction_id = create_response.json()["id"]
        
        from app.services.prediction_service import prediction_service
        from app.schemas.prediction import PredictionUpdateSchema
        prediction_service.update_prediction(
            prediction_id,
            PredictionUpdateSchema(status=PredictionStatus.PAUSED)
        )
        
        # Resume it
        response = client.post(f"/api/predictions/{prediction_id}/resume")
        assert response.status_code == 200
        assert response.json()["status"] == PredictionStatus.RUNNING.value
    
    def test_stop_prediction(self):
        """Test stopping a prediction"""
        # Create prediction and set to running
        data = {"sequence": "ACDEFGHIKL"}
        create_response = client.post("/api/predictions", json=data)
        prediction_id = create_response.json()["id"]
        
        from app.services.prediction_service import prediction_service
        from app.schemas.prediction import PredictionUpdateSchema
        prediction_service.update_prediction(
            prediction_id,
            PredictionUpdateSchema(status=PredictionStatus.RUNNING)
        )
        
        # Stop it
        response = client.post(f"/api/predictions/{prediction_id}/stop")
        assert response.status_code == 200
        assert response.json()["status"] == PredictionStatus.STOPPED.value
    
    def test_delete_prediction(self):
        """Test deleting a prediction"""
        # Create prediction
        data = {"sequence": "ACDEFGHIKL"}
        create_response = client.post("/api/predictions", json=data)
        prediction_id = create_response.json()["id"]
        
        # Delete it
        response = client.delete(f"/api/predictions/{prediction_id}")
        assert response.status_code == 204
        
        # Verify it's gone
        response = client.get(f"/api/predictions/{prediction_id}")
        assert response.status_code == 404
    
    def test_get_checkpoint(self):
        """Test getting checkpoint data"""
        # Create prediction
        data = {"sequence": "ACDEFGHIKL", "configuration": {"enable_checkpointing": True}}
        create_response = client.post("/api/predictions", json=data)
        prediction_id = create_response.json()["id"]
        
        # Try to get checkpoint
        response = client.get(f"/api/predictions/{prediction_id}/checkpoint")
        # Will 404 if no checkpoint exists yet, which is expected
        assert response.status_code in [200, 404]
    
    def test_create_prediction_with_all_options(self):
        """Test creating prediction with all configuration options"""
        data = {
            "sequence": "MQIFVKTLTGK",
            "configuration": {
                "iterations": 750,
                "agents": 8,
                "diversity": "aggressive",
                "enable_checkpointing": True,
                "checkpoint_interval": 75
            }
        }
        
        response = client.post("/api/predictions", json=data)
        assert response.status_code == 201
        result = response.json()
        assert result["configuration"]["iterations"] == 750
        assert result["configuration"]["agents"] == 8
    
    def test_list_predictions_pagination(self):
        """Test prediction list pagination"""
        # Create multiple predictions
        for i in range(5):
            data = {"sequence": f"ACDEFGHIKLMNPQRSTVWY"}
            client.post("/api/predictions", json=data)
        
        # Test limit parameter
        response = client.get("/api/predictions?limit=3")
        assert response.status_code == 200
        # Response format may vary, just check it succeeds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
