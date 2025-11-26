"""
Shared Export database model for public session sharing
"""
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Index
from sqlalchemy.orm import relationship
from app.database import Base


class SharedExport(Base):
    """
    SharedExport SQLAlchemy model for public share links to work sessions.
    
    Allows read-only access to a work session via a public URL without authentication.
    """
    __tablename__ = "shared_exports"

    # Primary key - unique share identifier (UUID v4 or random string)
    share_id = Column(String(36), primary_key=True, index=True)
    
    # Foreign key to WorkSession model
    session_id = Column(String(36), ForeignKey("work_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=False)
    
    # Access tracking
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed_at = Column(DateTime, nullable=True)

    # Relationships
    work_session = relationship("WorkSession", back_populates="shared_exports")

    # Additional indexes for performance
    __table_args__ = (
        Index('idx_session_expires', 'session_id', 'expires_at'),
        Index('idx_expires', 'expires_at'),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "share_id": self.share_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at.isoformat() if self.last_accessed_at else None,
        }

    def is_expired(self) -> bool:
        """Check if this share link has expired"""
        return datetime.now(timezone.utc) > self.expires_at

    def __repr__(self) -> str:
        return f"<SharedExport(share_id={self.share_id}, session_id={self.session_id}, expires_at={self.expires_at})>"
