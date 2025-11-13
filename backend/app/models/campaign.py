"""
Campaign database model
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class CampaignStatus(str, Enum):
    """Status of a campaign"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class PhaseStatus(str, Enum):
    """Status of a campaign phase"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Campaign:
    """Campaign model for testing multiple proteins"""
    
    def __init__(
        self,
        id: str,
        name: str,
        protein_ids: List[str],
        status: CampaignStatus = CampaignStatus.PENDING,
        configuration: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        current_phase: int = 1,
        total_phases: int = 4,
        phases: Optional[List[Dict[str, Any]]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        self.id = id
        self.name = name
        self.protein_ids = protein_ids
        self.status = status
        self.configuration = configuration or {}
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.started_at = started_at
        self.completed_at = completed_at
        self.current_phase = current_phase
        self.total_phases = total_phases
        self.phases = phases or []
        self.statistics = statistics or {}
        self.checkpoint_path = checkpoint_path
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "protein_ids": self.protein_ids,
            "status": self.status.value,
            "configuration": self.configuration,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_phase": self.current_phase,
            "total_phases": self.total_phases,
            "phases": self.phases,
            "statistics": self.statistics,
            "checkpoint_path": self.checkpoint_path,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Campaign":
        """Create from dictionary"""
        for date_field in ["created_at", "updated_at", "started_at", "completed_at"]:
            if data.get(date_field) and isinstance(data[date_field], str):
                data[date_field] = datetime.fromisoformat(data[date_field])
        
        if "status" in data and isinstance(data["status"], str):
            data["status"] = CampaignStatus(data["status"])
        
        return cls(**data)
