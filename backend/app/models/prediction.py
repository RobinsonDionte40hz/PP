"""
Prediction database model
"""
from datetime import datetime, timezone
from typing import Optional
from enum import Enum
from sqlalchemy import Column, String, DateTime, Integer, Float, Text, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

from app.database import Base


class PredictionStatus(str, Enum):
    """Status of a prediction"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Prediction(Base):
    """
    Prediction SQLAlchemy model
    """
    __tablename__ = "predictions"

    id = Column(String, primary_key=True, index=True)
    
    # Foreign key to WorkSession model (nullable for backward compatibility)
    session_id = Column(String(36), ForeignKey("work_sessions.id", ondelete="CASCADE"), nullable=True, index=True)
    
    sequence = Column(Text, nullable=False)
    status = Column(String, nullable=False, default=PredictionStatus.PENDING.value)
    configuration = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    task_id = Column(String, nullable=True)
    checkpoint_path = Column(String, nullable=True)
    result_path = Column(String, nullable=True)
    current_iteration = Column(Integer, nullable=False, default=0)
    total_iterations = Column(Integer, nullable=False, default=0)
    progress_percentage = Column(Float, nullable=False, default=0.0)
    metrics = Column(JSON, nullable=False, default=dict)
    
    # Relationships
    work_session = relationship("WorkSession", back_populates="predictions")
    
    # Additional indexes for performance
    __table_args__ = (
        Index('idx_session_created', 'session_id', 'created_at'),
        Index('idx_session_status', 'session_id', 'status'),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        # Extract commonly-used metrics for top-level access (frontend convenience)
        best_energy = self.metrics.get("best_energy") if self.metrics else None
        best_rmsd = self.metrics.get("best_rmsd") if self.metrics else None
        
        return {
            "id": self.id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "status": self.status,
            "configuration": self.configuration,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "task_id": self.task_id,
            "checkpoint_path": self.checkpoint_path,
            "result_path": self.result_path,
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "progress_percentage": self.progress_percentage,
            "metrics": self.metrics,
            # Top-level convenience fields (extracted from metrics)
            "best_energy": best_energy,
            "best_rmsd": best_rmsd,
        }
