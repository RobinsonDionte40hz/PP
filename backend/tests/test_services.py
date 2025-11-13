"""
Tests for service layer
"""
import pytest
from app.services.prediction_service import PredictionService
from app.services.campaign_service import CampaignService
from app.schemas.prediction import PredictionCreateSchema, PredictionUpdateSchema
from app.schemas.campaign import CampaignCreateSchema
from app.models.prediction import PredictionStatus
from app.models.campaign import CampaignStatus


class TestPredictionService:
    """Test prediction service logic"""
    
    @pytest.fixture
    def service(self):
        """Create fresh service instance for each test"""
        return PredictionService()
    
    def test_create_prediction(self, service):
        """Test creating a prediction"""
        data = PredictionCreateSchema(
            sequence="ACDEFGHIKL",
            configuration={
                "iterations": 1000,
                "agents": 10
            }
        )
        
        prediction = service.create_prediction(data)
        
        assert prediction.id.startswith("pred_")
        assert prediction.sequence == "ACDEFGHIKL"
        assert prediction.status == PredictionStatus.PENDING
        assert prediction.configuration["iterations"] == 1000
        assert prediction.total_iterations == 1000
    
    def test_get_prediction(self, service):
        """Test retrieving a prediction"""
        # Create
        data = PredictionCreateSchema(sequence="ACDEFG")
        prediction = service.create_prediction(data)
        
        # Retrieve
        retrieved = service.get_prediction(prediction.id)
        
        assert retrieved is not None
        assert retrieved.id == prediction.id
        assert retrieved.sequence == "ACDEFG"
    
    def test_get_nonexistent_prediction(self, service):
        """Test retrieving non-existent prediction"""
        result = service.get_prediction("pred_nonexistent")
        assert result is None
    
    def test_list_predictions(self, service):
        """Test listing predictions"""
        # Create multiple - use different valid sequences
        sequences = ["ACDEFG", "MKTAYV", "GHIKLM", "PQRST", "VWAYF"]
        for seq in sequences:
            data = PredictionCreateSchema(sequence=seq)
            service.create_prediction(data)
        
        # List
        predictions, total = service.list_predictions(page=1, page_size=3)
        
        assert len(predictions) == 3
        assert total == 5
    
    def test_list_predictions_with_filter(self, service):
        """Test listing with status filter"""
        # Create some predictions with different sequences
        sequences = ["ACDEFG", "MKTAYV", "GHIKLM"]
        for i, seq in enumerate(sequences):
            data = PredictionCreateSchema(sequence=seq)
            pred = service.create_prediction(data)
            
            # Set one to running
            if i == 1:
                service.update_prediction(
                    pred.id,
                    PredictionUpdateSchema(status=PredictionStatus.RUNNING)
                )
        
        # Filter by running
        predictions, total = service.list_predictions(
            status=PredictionStatus.RUNNING
        )
        
        assert total == 1
        assert predictions[0].status == PredictionStatus.RUNNING
    
    def test_update_prediction(self, service):
        """Test updating a prediction"""
        # Create
        data = PredictionCreateSchema(sequence="ACDEFG")
        prediction = service.create_prediction(data)
        
        # Update
        updated = service.update_prediction(
            prediction.id,
            PredictionUpdateSchema(
                status=PredictionStatus.RUNNING,
                current_iteration=100,
                progress_percentage=10.0
            )
        )
        
        assert updated.status == PredictionStatus.RUNNING
        assert updated.current_iteration == 100
        assert updated.progress_percentage == 10.0
        assert updated.started_at is not None
    
    def test_delete_prediction(self, service):
        """Test deleting a prediction"""
        # Create
        data = PredictionCreateSchema(sequence="ACDEFG")
        prediction = service.create_prediction(data)
        
        # Delete
        success = service.delete_prediction(prediction.id)
        assert success
        
        # Verify gone
        retrieved = service.get_prediction(prediction.id)
        assert retrieved is None
    
    def test_pause_prediction(self, service):
        """Test pausing a running prediction"""
        # Create and set to running
        data = PredictionCreateSchema(sequence="ACDEFG")
        prediction = service.create_prediction(data)
        service.update_prediction(
            prediction.id,
            PredictionUpdateSchema(status=PredictionStatus.RUNNING)
        )
        
        # Pause
        paused = service.pause_prediction(prediction.id)
        
        assert paused is not None
        assert paused.status == PredictionStatus.PAUSED
    
    def test_pause_invalid_status(self, service):
        """Test pausing a non-running prediction"""
        # Create pending prediction
        data = PredictionCreateSchema(sequence="ACDEFG")
        prediction = service.create_prediction(data)
        
        # Try to pause
        result = service.pause_prediction(prediction.id)
        
        assert result is None  # Cannot pause pending
    
    def test_resume_prediction(self, service):
        """Test resuming a paused prediction"""
        # Create and set to paused
        data = PredictionCreateSchema(sequence="ACDEFG")
        prediction = service.create_prediction(data)
        service.update_prediction(
            prediction.id,
            PredictionUpdateSchema(status=PredictionStatus.PAUSED)
        )
        
        # Resume
        resumed = service.resume_prediction(prediction.id)
        
        assert resumed is not None
        assert resumed.status == PredictionStatus.RUNNING
    
    def test_stop_prediction(self, service):
        """Test stopping a running prediction"""
        # Create and set to running
        data = PredictionCreateSchema(sequence="ACDEFG")
        prediction = service.create_prediction(data)
        service.update_prediction(
            prediction.id,
            PredictionUpdateSchema(status=PredictionStatus.RUNNING)
        )
        
        # Stop
        stopped = service.stop_prediction(prediction.id)
        
        assert stopped is not None
        assert stopped.status == PredictionStatus.STOPPED
        assert stopped.completed_at is not None


class TestCampaignService:
    """Test campaign service logic"""
    
    @pytest.fixture
    def service(self):
        """Create fresh service instance for each test"""
        return CampaignService()
    
    def test_create_campaign(self, service):
        """Test creating a campaign"""
        data = CampaignCreateSchema(
            name="Test Campaign",
            protein_ids=["1UBQ", "1CRN", "2MR9"]
        )
        
        campaign = service.create_campaign(data)
        
        assert campaign.id.startswith("camp_")
        assert campaign.name == "Test Campaign"
        assert len(campaign.protein_ids) == 3
        assert campaign.status == CampaignStatus.PENDING
        assert campaign.total_phases == 4
        assert len(campaign.phases) == 4
    
    def test_get_campaign(self, service):
        """Test retrieving a campaign"""
        # Create
        data = CampaignCreateSchema(
            name="Test Campaign",
            protein_ids=["1UBQ"]
        )
        campaign = service.create_campaign(data)
        
        # Retrieve
        retrieved = service.get_campaign(campaign.id)
        
        assert retrieved is not None
        assert retrieved.id == campaign.id
        assert retrieved.name == "Test Campaign"
    
    def test_list_campaigns(self, service):
        """Test listing campaigns"""
        # Create multiple
        for i in range(5):
            data = CampaignCreateSchema(
                name=f"Campaign {i}",
                protein_ids=["1UBQ"]
            )
            service.create_campaign(data)
        
        # List
        campaigns, total = service.list_campaigns(page=1, page_size=3)
        
        assert len(campaigns) == 3
        assert total == 5
    
    def test_update_campaign_status(self, service):
        """Test updating campaign status"""
        # Create
        data = CampaignCreateSchema(
            name="Test Campaign",
            protein_ids=["1UBQ"]
        )
        campaign = service.create_campaign(data)
        
        # Update
        updated = service.update_campaign_status(
            campaign.id,
            CampaignStatus.RUNNING
        )
        
        assert updated.status == CampaignStatus.RUNNING
        assert updated.started_at is not None
    
    def test_delete_campaign(self, service):
        """Test deleting a campaign"""
        # Create
        data = CampaignCreateSchema(
            name="Test Campaign",
            protein_ids=["1UBQ"]
        )
        campaign = service.create_campaign(data)
        
        # Delete
        success = service.delete_campaign(campaign.id)
        assert success
        
        # Verify gone
        retrieved = service.get_campaign(campaign.id)
        assert retrieved is None
    
    def test_get_phase_details(self, service):
        """Test getting phase details"""
        # Create
        data = CampaignCreateSchema(
            name="Test Campaign",
            protein_ids=["1UBQ"]
        )
        campaign = service.create_campaign(data)
        
        # Get phase
        phase = service.get_phase_details(campaign.id, 1)
        
        assert phase is not None
        assert phase["phase_number"] == 1
        assert "status" in phase
    
    def test_update_phase(self, service):
        """Test updating phase details"""
        # Create
        data = CampaignCreateSchema(
            name="Test Campaign",
            protein_ids=["1UBQ"]
        )
        campaign = service.create_campaign(data)
        
        # Update phase
        updated_phase = service.update_phase(
            campaign.id,
            1,
            {"proteins_tested": 5, "proteins_passed": 3}
        )
        
        assert updated_phase["proteins_tested"] == 5
        assert updated_phase["proteins_passed"] == 3
    
    def test_resume_campaign(self, service):
        """Test resuming a paused campaign"""
        # Create
        data = CampaignCreateSchema(
            name="Test Campaign",
            protein_ids=["1UBQ"]
        )
        campaign = service.create_campaign(data)
        
        # Set to paused
        service.update_campaign_status(campaign.id, CampaignStatus.PAUSED)
        
        # Resume
        resumed = service.resume_campaign(campaign.id)
        
        assert resumed is not None
        assert resumed.status == CampaignStatus.RUNNING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
