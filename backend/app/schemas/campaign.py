"""
Campaign schemas for request/response validation
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from app.models.campaign import CampaignStatus, PhaseStatus


class CampaignConfigurationSchema(BaseModel):
    """Configuration for campaign"""
    iterations_per_phase: int = Field(default=1000, ge=100, le=10000)
    agents: int = Field(default=10, ge=1, le=100)
    quality_thresholds: Dict[str, float] = Field(
        default={"rmsd": 5.0, "energy": -50.0},
        description="Quality thresholds for progression"
    )
    enable_checkpointing: bool = Field(default=True)
    checkpoint_interval: int = Field(default=50, ge=10, le=1000)


class CampaignCreateSchema(BaseModel):
    """Schema for creating a new campaign"""
    name: str = Field(..., min_length=1, max_length=200)
    protein_ids: List[str] = Field(..., min_items=1, max_items=50, description="List of PDB IDs or sequences")
    configuration: Optional[CampaignConfigurationSchema] = None


class PhaseResponseSchema(BaseModel):
    """Schema for phase information"""
    phase_number: int
    status: PhaseStatus
    proteins_tested: int = 0
    proteins_passed: int = 0
    proteins_failed: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: List[Dict[str, Any]] = []


class CampaignStatisticsSchema(BaseModel):
    """Campaign statistics"""
    total_proteins: int = 0
    proteins_completed: int = 0
    proteins_failed: int = 0
    average_rmsd: Optional[float] = None
    average_energy: Optional[float] = None
    success_rate: float = 0.0
    total_runtime_seconds: float = 0.0


class CampaignResponseSchema(BaseModel):
    """Schema for campaign response"""
    id: str
    name: str
    protein_ids: List[str]
    status: CampaignStatus
    configuration: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_phase: int = 1
    total_phases: int = 4
    phases: List[Dict[str, Any]] = []
    statistics: Dict[str, Any] = {}
    checkpoint_path: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class CampaignListResponseSchema(BaseModel):
    """Schema for list of campaigns"""
    campaigns: List[CampaignResponseSchema]
    total: int
    page: int
    page_size: int
