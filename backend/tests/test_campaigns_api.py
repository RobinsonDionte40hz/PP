"""
Tests for campaign API endpoints
"""
import pytest
import os
from fastapi.testclient import TestClient

# Disable rate limiting for tests
os.environ["TESTING"] = "true"

from app.main import app
from app.models.campaign import CampaignStatus
from app.models.prediction import Prediction

client = TestClient(app)


class TestCampaignAPI:
    """Test campaign API endpoints"""
    
    def test_create_campaign(self):
        """Test creating a new campaign"""
        data = {
            "name": "Test Campaign",
            "protein_ids": ["1UBQ", "1CRN", "2MR9"],
            "configuration": {
                "iterations_per_phase": 500,
                "agents": 5,
                "quality_thresholds": {
                    "rmsd": 5.0,
                    "energy": -50.0
                }
            }
        }
        
        response = client.post("/api/campaigns", json=data)
        assert response.status_code == 201
        
        result = response.json()
        assert result["name"] == data["name"]
        assert len(result["protein_ids"]) == 3
        assert result["status"] == CampaignStatus.PENDING.value
        assert result["total_phases"] == 4
        assert "id" in result
        assert result["id"].startswith("camp_")
    
    def test_create_campaign_no_proteins(self):
        """Test creating campaign without proteins"""
        data = {
            "name": "Empty Campaign",
            "protein_ids": []
        }
        
        response = client.post("/api/campaigns", json=data)
        assert response.status_code == 422  # Validation error
    
    def test_get_campaign(self):
        """Test getting campaign by ID"""
        # First create a campaign
        data = {
            "name": "Test Campaign",
            "protein_ids": ["1UBQ"]
        }
        create_response = client.post("/api/campaigns", json=data)
        campaign_id = create_response.json()["id"]
        
        # Get the campaign
        response = client.get(f"/api/campaigns/{campaign_id}")
        assert response.status_code == 200
        
        result = response.json()
        assert result["id"] == campaign_id
        assert result["name"] == data["name"]
    
    def test_get_campaign_not_found(self):
        """Test getting non-existent campaign"""
        response = client.get("/api/campaigns/camp_nonexistent")
        assert response.status_code == 404
    
    def test_list_campaigns(self):
        """Test listing campaigns"""
        # Create multiple campaigns
        for i in range(3):
            data = {
                "name": f"Campaign {i}",
                "protein_ids": [f"PROT{i}"]
            }
            client.post("/api/campaigns", json=data)
        
        # List all
        response = client.get("/api/campaigns")
        assert response.status_code == 200
        
        result = response.json()
        assert "campaigns" in result
        assert result["total"] >= 3
        assert result["page"] == 1
    
    def test_list_campaigns_with_filter(self):
        """Test listing campaigns with status filter"""
        response = client.get(f"/api/campaigns?status={CampaignStatus.PENDING.value}")
        assert response.status_code == 200
        
        result = response.json()
        for campaign in result["campaigns"]:
            assert campaign["status"] == CampaignStatus.PENDING.value
    
    def test_delete_campaign(self):
        """Test deleting a campaign"""
        # Create campaign
        data = {
            "name": "Test Campaign",
            "protein_ids": ["1UBQ"]
        }
        create_response = client.post("/api/campaigns", json=data)
        campaign_id = create_response.json()["id"]
        
        # Delete it
        response = client.delete(f"/api/campaigns/{campaign_id}")
        assert response.status_code == 204
        
        # Verify it's gone
        response = client.get(f"/api/campaigns/{campaign_id}")
        assert response.status_code == 404
    
    def test_resume_campaign(self):
        """Test resuming a paused campaign"""
        # Create campaign
        data = {
            "name": "Test Campaign",
            "protein_ids": ["1UBQ"]
        }
        create_response = client.post("/api/campaigns", json=data)
        campaign_id = create_response.json()["id"]
        
        # Manually set to paused
        from app.services.campaign_service import campaign_service
        campaign_service.update_campaign_status(campaign_id, CampaignStatus.PAUSED)
        
        # Resume it
        response = client.post(f"/api/campaigns/{campaign_id}/resume")
        assert response.status_code == 200
        assert response.json()["status"] == CampaignStatus.RUNNING.value
    
    def test_get_campaign_statistics(self):
        """Test getting campaign statistics"""
        # Create campaign
        data = {
            "name": "Test Campaign",
            "protein_ids": ["1UBQ"]
        }
        create_response = client.post("/api/campaigns", json=data)
        campaign_id = create_response.json()["id"]
        
        # Get statistics
        response = client.get(f"/api/campaigns/{campaign_id}/statistics")
        assert response.status_code == 200
        
        result = response.json()
        assert "total_proteins" in result
        assert "proteins_completed" in result
        assert "success_rate" in result
    
    def test_get_phase_details(self):
        """Test getting phase details"""
        # Create campaign
        data = {
            "name": "Test Campaign",
            "protein_ids": ["1UBQ"]
        }
        create_response = client.post("/api/campaigns", json=data)
        campaign_id = create_response.json()["id"]
        
        # Get phase 1 details
        response = client.get(f"/api/campaigns/{campaign_id}/phase/1")
        assert response.status_code == 200
        
        result = response.json()
        assert result["phase_number"] == 1
        assert "status" in result
    
    def test_get_invalid_phase(self):
        """Test getting invalid phase number"""
        # Create campaign
        data = {
            "name": "Test Campaign",
            "protein_ids": ["1UBQ"]
        }
        create_response = client.post("/api/campaigns", json=data)
        campaign_id = create_response.json()["id"]
        
        # Try to get phase 99 - phase numbers are validated by FastAPI path constraints
        # A phase that doesn't exist will return 422 (Unprocessable Entity) if invalid format
        # or 404 if valid format but phase not found - we'll test with invalid (0)
        response = client.get(f"/api/campaigns/{campaign_id}/phase/0")
        # Invalid phase number returns 422 due to Pydantic validation
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
