"""
Campaign API endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks
from typing import Optional
from app.schemas.campaign import (
    CampaignCreateSchema,
    CampaignResponseSchema,
    CampaignListResponseSchema,
    CampaignStatisticsSchema,
    PhaseResponseSchema,
)
from app.models.campaign import CampaignStatus
from app.services.campaign_service import campaign_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=CampaignResponseSchema,
    status_code=201,
    summary="Create campaign",
    description="Create a new multi-protein testing campaign"
)
async def create_campaign(
    data: CampaignCreateSchema,
    background_tasks: BackgroundTasks,
):
    """
    Create a new campaign to test multiple proteins.
    
    The campaign will run in phases with quality gates.
    """
    try:
        campaign = campaign_service.create_campaign(data)
        
        # TODO: Queue Celery task for campaign execution
        # background_tasks.add_task(queue_campaign_task, campaign.id)
        
        return CampaignResponseSchema(**campaign.to_dict())
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating campaign: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "",
    response_model=CampaignListResponseSchema,
    summary="List campaigns",
    description="Get list of campaigns with optional filtering"
)
async def list_campaigns(
    status: Optional[CampaignStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """List campaigns with pagination and optional status filter."""
    try:
        campaigns, total = campaign_service.list_campaigns(
            status=status,
            page=page,
            page_size=page_size
        )
        
        return CampaignListResponseSchema(
            campaigns=[CampaignResponseSchema(**c.to_dict()) for c in campaigns],
            total=total,
            page=page,
            page_size=page_size
        )
    
    except Exception as e:
        logger.error(f"Error listing campaigns: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{campaign_id}",
    response_model=CampaignResponseSchema,
    summary="Get campaign details",
    description="Get detailed information about a specific campaign"
)
async def get_campaign(
    campaign_id: str = Path(..., description="Campaign ID")
):
    """Get detailed information about a specific campaign."""
    campaign = campaign_service.get_campaign(campaign_id)
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return CampaignResponseSchema(**campaign.to_dict())


@router.delete(
    "/{campaign_id}",
    status_code=204,
    summary="Delete campaign",
    description="Delete a campaign and its associated data"
)
async def delete_campaign(
    campaign_id: str = Path(..., description="Campaign ID")
):
    """Delete a campaign and all associated data."""
    success = campaign_service.delete_campaign(campaign_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return None


@router.post(
    "/{campaign_id}/resume",
    response_model=CampaignResponseSchema,
    summary="Resume campaign",
    description="Resume a paused campaign"
)
async def resume_campaign(
    campaign_id: str = Path(..., description="Campaign ID")
):
    """Resume a paused campaign from the last checkpoint."""
    campaign = campaign_service.resume_campaign(campaign_id)
    
    if not campaign:
        raise HTTPException(
            status_code=400,
            detail="Cannot resume campaign - not found or not in paused state"
        )
    
    # TODO: Queue Celery task to resume campaign
    
    return CampaignResponseSchema(**campaign.to_dict())


@router.get(
    "/{campaign_id}/statistics",
    response_model=CampaignStatisticsSchema,
    summary="Get campaign statistics",
    description="Get statistical analysis of campaign results"
)
async def get_campaign_statistics(
    campaign_id: str = Path(..., description="Campaign ID")
):
    """Get detailed statistics for a campaign."""
    campaign = campaign_service.get_campaign(campaign_id)
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return CampaignStatisticsSchema(**campaign.statistics)


@router.get(
    "/{campaign_id}/phase/{phase_num}",
    summary="Get phase details",
    description="Get detailed information about a specific phase"
)
async def get_phase_details(
    campaign_id: str = Path(..., description="Campaign ID"),
    phase_num: int = Path(..., ge=1, le=4, description="Phase number (1-4)")
):
    """Get detailed information about a specific campaign phase."""
    phase = campaign_service.get_phase_details(campaign_id, phase_num)
    
    if not phase:
        raise HTTPException(status_code=404, detail="Phase not found")
    
    return phase
