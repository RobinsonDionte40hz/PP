"""
Session data models and schemas
"""
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class SessionData:
    """
    Session data stored in Redis.
    Immutable data class for session information.
    """
    user_key_id: str
    username: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str

    def to_dict(self) -> dict:
        """Convert to dictionary for Redis storage"""
        return {
            "user_key_id": self.user_key_id,
            "username": self.username,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionData":
        """Create SessionData from dictionary"""
        return cls(
            user_key_id=data["user_key_id"],
            username=data["username"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            ip_address=data["ip_address"],
            user_agent=data["user_agent"],
        )


class SessionResponse(BaseModel):
    """Response model for session information"""
    user_key_id: str = Field(..., description="User's unique identifier")
    username: str = Field(..., description="Username")
    created_at: datetime = Field(..., description="Session creation timestamp")
    expires_at: datetime = Field(..., description="Session expiration timestamp")
    ip_address: str = Field(..., description="IP address of session")

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_key_id": "550e8400-e29b-41d4-a716-446655440000",
                "username": "john_doe",
                "created_at": "2025-11-22T10:30:00Z",
                "expires_at": "2025-11-22T11:00:00Z",
                "ip_address": "192.168.1.1",
            }
        }
    }
