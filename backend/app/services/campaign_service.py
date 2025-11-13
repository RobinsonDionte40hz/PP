"""
Campaign service - business logic for campaigns
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from app.models.campaign import Campaign, CampaignStatus, PhaseStatus
from app.schemas.campaign import CampaignCreateSchema
import logging

logger = logging.getLogger(__name__)


class CampaignService:
    """Service for managing campaigns"""
    
    def __init__(self):
        self._campaigns: Dict[str, Campaign] = {}
    
    def create_campaign(self, data: CampaignCreateSchema) -> Campaign:
        """Create a new campaign"""
        campaign_id = f"camp_{uuid.uuid4().hex[:12]}"
        
        config = data.configuration.model_dump() if data.configuration else {}
        
        # Initialize phases
        phases = [
            {
                "phase_number": i,
                "status": PhaseStatus.PENDING.value,
                "proteins_tested": 0,
                "proteins_passed": 0,
                "proteins_failed": 0,
                "results": []
            }
            for i in range(1, 5)  # 4 phases
        ]
        
        campaign = Campaign(
            id=campaign_id,
            name=data.name,
            protein_ids=data.protein_ids,
            status=CampaignStatus.PENDING,
            configuration=config,
            phases=phases,
            total_phases=4,
            statistics={
                "total_proteins": len(data.protein_ids),
                "proteins_completed": 0,
                "proteins_failed": 0,
                "success_rate": 0.0
            }
        )
        
        self._campaigns[campaign_id] = campaign
        logger.info(f"Created campaign {campaign_id} with {len(data.protein_ids)} proteins")
        
        return campaign
    
    def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Get campaign by ID"""
        return self._campaigns.get(campaign_id)
    
    def list_campaigns(
        self,
        status: Optional[CampaignStatus] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[Campaign], int]:
        """List campaigns with optional filtering and pagination"""
        campaigns = list(self._campaigns.values())
        
        if status:
            campaigns = [c for c in campaigns if c.status == status]
        
        campaigns.sort(key=lambda c: c.created_at, reverse=True)
        
        total = len(campaigns)
        start = (page - 1) * page_size
        end = start + page_size
        campaigns = campaigns[start:end]
        
        return campaigns, total
    
    def update_campaign_status(self, campaign_id: str, status: CampaignStatus) -> Optional[Campaign]:
        """Update campaign status"""
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return None
        
        campaign.status = status
        campaign.updated_at = datetime.utcnow()
        
        if status == CampaignStatus.RUNNING and not campaign.started_at:
            campaign.started_at = datetime.utcnow()
        elif status in [CampaignStatus.COMPLETED, CampaignStatus.FAILED, CampaignStatus.STOPPED]:
            campaign.completed_at = datetime.utcnow()
        
        logger.info(f"Updated campaign {campaign_id} status to {status.value}")
        return campaign
    
    def delete_campaign(self, campaign_id: str) -> bool:
        """Delete campaign"""
        if campaign_id in self._campaigns:
            del self._campaigns[campaign_id]
            logger.info(f"Deleted campaign {campaign_id}")
            return True
        return False
    
    def get_phase_details(self, campaign_id: str, phase_num: int) -> Optional[Dict[str, Any]]:
        """Get details for a specific phase"""
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return None
        
        if phase_num < 1 or phase_num > campaign.total_phases:
            return None
        
        for phase in campaign.phases:
            if phase.get("phase_number") == phase_num:
                return phase
        
        return None
    
    def update_phase(
        self,
        campaign_id: str,
        phase_num: int,
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update phase details"""
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return None
        
        for phase in campaign.phases:
            if phase.get("phase_number") == phase_num:
                phase.update(updates)
                campaign.updated_at = datetime.utcnow()
                logger.info(f"Updated campaign {campaign_id} phase {phase_num}")
                return phase
        
        return None
    
    def update_statistics(self, campaign_id: str, statistics: Dict[str, Any]) -> Optional[Campaign]:
        """Update campaign statistics"""
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return None
        
        campaign.statistics.update(statistics)
        campaign.updated_at = datetime.utcnow()
        
        logger.info(f"Updated campaign {campaign_id} statistics")
        return campaign
    
    def resume_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Resume a paused campaign"""
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return None
        
        if campaign.status != CampaignStatus.PAUSED:
            logger.warning(f"Cannot resume campaign {campaign_id} - status is {campaign.status}")
            return None
        
        campaign.status = CampaignStatus.RUNNING
        campaign.updated_at = datetime.utcnow()
        
        logger.info(f"Resumed campaign {campaign_id}")
        return campaign


# Global service instance
campaign_service = CampaignService()
