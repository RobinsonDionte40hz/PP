"""
User database model
"""
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Column, String, DateTime, Boolean, Index
from app.database import Base


class User(Base):
    """
    User SQLAlchemy model for authentication
    """
    __tablename__ = "users"

    # Primary key - UUID v4 generated externally
    key_id = Column(String(36), primary_key=True, index=True)
    
    # Authentication fields
    username = Column(String(150), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Account status
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Role management (user, developer, admin)
    role = Column(String(20), nullable=False, default='user')
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime, nullable=True)

    # Additional indexes for performance
    __table_args__ = (
        Index('idx_username_active', 'username', 'is_active'),
        Index('idx_email_active', 'email', 'is_active'),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary (excludes password_hash)"""
        return {
            "key_id": self.key_id,
            "username": self.username,
            "email": self.email,
            "is_active": self.is_active,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }

    def to_profile(self) -> dict:
        """Convert to user profile (public-facing data only)"""
        return {
            "key_id": self.key_id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return f"<User(key_id={self.key_id}, username={self.username})>"
