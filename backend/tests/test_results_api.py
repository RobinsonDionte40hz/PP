"""
Tests for results API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.prediction import PredictionStatus
import json
import tempfile
from pathlib import Path

client = TestClient(app)


class TestResultsAPI:
    """Test results API endpoints"""
    
    @pytest.fixture
    def prediction_with_results(self):
        """Create a prediction with mock results"""
        from app.services.prediction_service import prediction_service
        from app.schemas.prediction import PredictionCreateSchema, PredictionUpdateSchema
        
        # Create prediction
        data = PredictionCreateSchema(sequence="ACDEFGHIKL")
        prediction = prediction_service.create_prediction(data)
        
        # Create mock results directory and file
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            
            # Create results.json
            results_data = {
                "prediction_id": prediction.id,
                "sequence": "ACDEFGHIKL",
                "best_energy": -123.45,
                "best_rmsd": 2.5,
                "conformations_explored": 1000
            }
            result_file = result_dir / "results.json"
            with open(result_file, 'w') as f:
                json.dump(results_data, f)
            
            # Create mock PDB file
            pdb_file = result_dir / "structure.pdb"
            with open(pdb_file, 'w') as f:
                f.write("HEADER    TEST PROTEIN\n")
                f.write("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
                f.write("END\n")
            
            # Update prediction with results
            prediction_service.update_prediction(
                prediction.id,
                PredictionUpdateSchema(
                    status=PredictionStatus.COMPLETED,
                    result_path=str(result_dir)
                )
            )
            
            yield prediction.id
    
    def test_get_results_not_found(self):
        """Test getting results for non-existent prediction"""
        response = client.get("/api/results/pred_nonexistent")
        assert response.status_code == 404
    
    def test_get_results_not_ready(self):
        """Test getting results when not yet available"""
        from app.services.prediction_service import prediction_service
        from app.schemas.prediction import PredictionCreateSchema
        
        # Create prediction without results
        data = PredictionCreateSchema(sequence="ACDEFG")
        prediction = prediction_service.create_prediction(data)
        
        response = client.get(f"/api/results/{prediction.id}")
        assert response.status_code == 404
        assert "not available" in response.json()["detail"].lower()
    
    def test_get_structure_not_found(self):
        """Test getting structure for non-existent prediction"""
        response = client.get("/api/results/pred_nonexistent/structure")
        assert response.status_code == 404
    
    def test_get_trajectory(self):
        """Test getting trajectory data"""
        from app.services.prediction_service import prediction_service
        from app.schemas.prediction import PredictionCreateSchema
        
        prediction = prediction_service.create_prediction(
            PredictionCreateSchema(sequence="ACDEFG")
        )
        
        response = client.get(f"/api/results/{prediction.id}/trajectory")
        # Currently returns placeholder
        assert response.status_code == 200
    
    def test_get_metrics(self):
        """Test getting detailed metrics"""
        from app.services.prediction_service import prediction_service
        from app.schemas.prediction import PredictionCreateSchema, PredictionUpdateSchema
        
        prediction = prediction_service.create_prediction(
            PredictionCreateSchema(sequence="ACDEFG")
        )
        
        # Add some metrics
        prediction_service.update_prediction(
            prediction.id,
            PredictionUpdateSchema(
                metrics={
                    "best_energy": -100.0,
                    "best_rmsd": 3.5
                }
            )
        )
        
        response = client.get(f"/api/results/{prediction.id}/metrics")
        assert response.status_code == 200
        assert "metrics" in response.json()
    
    def test_export_results_invalid_format(self):
        """Test export with invalid format"""
        from app.services.prediction_service import prediction_service
        from app.schemas.prediction import PredictionCreateSchema
        
        prediction = prediction_service.create_prediction(
            PredictionCreateSchema(sequence="ACDEFG")
        )
        
        response = client.get(f"/api/results/{prediction.id}/export?format=invalid")
        assert response.status_code == 400
        assert "Invalid format" in response.json()["detail"]
    
    def test_export_results_json(self):
        """Test export in JSON format"""
        from app.services.prediction_service import prediction_service
        from app.schemas.prediction import PredictionCreateSchema
        
        prediction = prediction_service.create_prediction(
            PredictionCreateSchema(sequence="ACDEFG")
        )
        
        response = client.get(f"/api/results/{prediction.id}/export?format=json")
        assert response.status_code == 200
    
    def test_compare_results_insufficient(self):
        """Test comparison with too few predictions"""
        response = client.post(
            "/api/results/compare",
            params={"prediction_ids": ["pred_1"]}
        )
        assert response.status_code == 400
        assert "at least 2" in response.json()["detail"].lower()
    
    def test_compare_results_too_many(self):
        """Test comparison with too many predictions"""
        prediction_ids = [f"pred_{i}" for i in range(15)]
        response = client.post(
            "/api/results/compare",
            params={"prediction_ids": prediction_ids}
        )
        assert response.status_code == 400
        assert "more than 10" in response.json()["detail"].lower()
    
    def test_compare_results(self):
        """Test comparing multiple predictions"""
        from app.services.prediction_service import prediction_service
        from app.schemas.prediction import PredictionCreateSchema, PredictionUpdateSchema
        
        # Create two predictions
        pred1 = prediction_service.create_prediction(
            PredictionCreateSchema(sequence="ACDEFG")
        )
        pred2 = prediction_service.create_prediction(
            PredictionCreateSchema(sequence="ACDEFGH")
        )
        
        # Add metrics
        for pred in [pred1, pred2]:
            prediction_service.update_prediction(
                pred.id,
                PredictionUpdateSchema(
                    status=PredictionStatus.COMPLETED,
                    metrics={"best_energy": -100.0}
                )
            )
        
        response = client.post(
            "/api/results/compare",
            params={"prediction_ids": [pred1.id, pred2.id]}
        )
        assert response.status_code == 200
        assert "predictions" in response.json()
        assert len(response.json()["predictions"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
