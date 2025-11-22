"""
Prediction database model
"""
from datetime import datetime, timezone
from typing import Optional
from enum import Enum
from sqlalchemy import Column, String, DateTime, Integer, Float, Text, JSON
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

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
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
        }
