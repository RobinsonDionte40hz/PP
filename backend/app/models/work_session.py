"""
Work Session database model for organizing predictions
"""
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import Column, String, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from app.database import Base


class WorkSession(Base):
    """
    WorkSession SQLAlchemy model for grouping related predictions.
    
    Note: This represents a "Work Session" for organizing predictions,
    which is distinct from the authentication session (JWT tokens in Redis).
    """
    __tablename__ = "work_sessions"

    # Primary key - UUID v4 generated externally
    id = Column(String(36), primary_key=True, index=True)
    
    # Foreign key to User model
    user_id = Column(String(36), ForeignKey("users.key_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Session metadata
    name = Column(String(255), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_active_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="work_sessions")
    predictions = relationship(
        "Prediction",
        back_populates="work_session",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    shared_exports = relationship(
        "SharedExport",
        back_populates="work_session",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    # Additional indexes for performance
    __table_args__ = (
        Index('idx_user_last_active', 'user_id', 'last_active_at'),
        Index('idx_user_created', 'user_id', 'created_at'),
    )

    def to_dict(self, include_predictions: bool = False) -> dict:
        """
        Convert to dictionary
        
        Args:
            include_predictions: Whether to include related predictions
        """
        result = {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "created_at": self.created_at.isoformat() if self.created_at is not None else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at is not None else None,
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at is not None else None,
        }
        
        if include_predictions:
            result["predictions"] = [pred.to_dict() for pred in self.predictions]
            result["prediction_count"] = len(self.predictions)
        
        return result

    def __repr__(self) -> str:
        return f"<WorkSession(id={self.id}, name={self.name}, user_id={self.user_id})>"
