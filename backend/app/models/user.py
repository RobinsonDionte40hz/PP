"""
User database model
"""
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Column, String, DateTime, Boolean, Index, Integer
from sqlalchemy.orm import relationship
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
    
    # Quota tracking
    daily_prediction_count = Column(Integer, nullable=False, default=0)
    monthly_prediction_count = Column(Integer, nullable=False, default=0)
    daily_quota_reset_at = Column(DateTime(timezone=True), nullable=True)
    monthly_quota_reset_at = Column(DateTime(timezone=True), nullable=True)
    
    # Tier settings (free, pro, enterprise)
    account_tier = Column(String(20), nullable=False, default='free')
    daily_prediction_limit = Column(Integer, nullable=False, default=20)
    monthly_prediction_limit = Column(Integer, nullable=False, default=100)
    
    # Email verification
    email_verified = Column(Boolean, nullable=False, default=False)
    email_verification_token = Column(String(64), nullable=True, index=True)
    email_verification_sent_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime, nullable=True)

    # Relationships
    work_sessions = relationship(
        "WorkSession",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

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
            "email_verified": self.email_verified,
            "account_tier": self.account_tier,
            "daily_prediction_count": self.daily_prediction_count,
            "monthly_prediction_count": self.monthly_prediction_count,
            "daily_prediction_limit": self.daily_prediction_limit,
            "monthly_prediction_limit": self.monthly_prediction_limit,
            "daily_quota_reset_at": self.daily_quota_reset_at.isoformat() if self.daily_quota_reset_at else None,
            "monthly_quota_reset_at": self.monthly_quota_reset_at.isoformat() if self.monthly_quota_reset_at else None,
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
            "email_verified": self.email_verified,
            "account_tier": self.account_tier,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    def to_quota_info(self) -> dict:
        """Get quota information for the user"""
        return {
            "account_tier": self.account_tier,
            "daily": {
                "used": self.daily_prediction_count,
                "limit": self.daily_prediction_limit,
                "remaining": max(0, self.daily_prediction_limit - self.daily_prediction_count),
                "reset_at": self.daily_quota_reset_at.isoformat() if self.daily_quota_reset_at else None,
            },
            "monthly": {
                "used": self.monthly_prediction_count,
                "limit": self.monthly_prediction_limit,
                "remaining": max(0, self.monthly_prediction_limit - self.monthly_prediction_count),
                "reset_at": self.monthly_quota_reset_at.isoformat() if self.monthly_quota_reset_at else None,
            },
        }

    def __repr__(self) -> str:
        return f"<User(key_id={self.key_id}, username={self.username})>"
