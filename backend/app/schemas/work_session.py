"""
Work Session schemas for request/response validation
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class WorkSessionCreateSchema(BaseModel):
    """Schema for creating a new work session"""
    name: str = Field(..., min_length=1, max_length=255, description="Session name (1-255 characters)")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate session name"""
        # Strip whitespace
        v = v.strip()
        
        # Check if empty after stripping
        if not v:
            raise ValueError("Session name cannot be empty or only whitespace")
        
        # Check length after stripping
        if len(v) > 255:
            raise ValueError("Session name cannot exceed 255 characters")
        
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Antibody Design Project"
                },
                {
                    "name": "Enzyme Stability Tests - Nov 2025"
                }
            ]
        }
    }


class WorkSessionUpdateSchema(BaseModel):
    """Schema for updating an existing work session"""
    name: str = Field(..., min_length=1, max_length=255, description="New session name (1-255 characters)")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate session name"""
        # Strip whitespace
        v = v.strip()
        
        # Check if empty after stripping
        if not v:
            raise ValueError("Session name cannot be empty or only whitespace")
        
        # Check length after stripping
        if len(v) > 255:
            raise ValueError("Session name cannot exceed 255 characters")
        
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Updated Project Name"
                }
            ]
        }
    }


class WorkSessionResponseSchema(BaseModel):
    """Schema for work session response"""
    id: str = Field(..., description="Session unique identifier (UUID)")
    user_id: str = Field(..., description="Owner's user ID (key_id from User model)")
    name: str = Field(..., description="Session name")
    created_at: datetime = Field(..., description="Session creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_active_at: datetime = Field(..., description="Last activity timestamp")
    prediction_count: Optional[int] = Field(default=0, description="Number of predictions in this session")
    total_size: Optional[int] = Field(default=0, description="Total size of session data in bytes")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "user_id": "660e8400-e29b-41d4-a716-446655440111",
                    "name": "Antibody Design Project",
                    "created_at": "2025-11-20T10:00:00Z",
                    "updated_at": "2025-11-25T15:30:00Z",
                    "last_active_at": "2025-11-26T09:15:00Z",
                    "prediction_count": 12,
                    "total_size": 52428800
                }
            ]
        }
    }


class WorkSessionListResponseSchema(BaseModel):
    """Schema for paginated list of work sessions"""
    sessions: List[WorkSessionResponseSchema] = Field(..., description="List of work sessions")
    total: int = Field(..., description="Total number of sessions")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Number of items per page")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sessions": [
                        {
                            "id": "550e8400-e29b-41d4-a716-446655440000",
                            "user_id": "660e8400-e29b-41d4-a716-446655440111",
                            "name": "Antibody Design Project",
                            "created_at": "2025-11-20T10:00:00Z",
                            "updated_at": "2025-11-25T15:30:00Z",
                            "last_active_at": "2025-11-26T09:15:00Z",
                            "prediction_count": 12,
                            "total_size": 52428800
                        }
                    ],
                    "total": 1,
                    "page": 1,
                    "page_size": 20
                }
            ]
        }
    }


class ShareLinkCreateSchema(BaseModel):
    """Schema for creating a share link"""
    expiration_hours: int = Field(
        ...,
        ge=1,
        le=168,
        description="Share link expiration time in hours (1-168, i.e., 1 hour to 7 days)"
    )
    
    @field_validator("expiration_hours")
    @classmethod
    def validate_expiration(cls, v: int) -> int:
        """Validate expiration hours"""
        if v < 1:
            raise ValueError("Expiration hours must be at least 1")
        if v > 168:
            raise ValueError("Expiration hours cannot exceed 168 (7 days)")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "expiration_hours": 24
                },
                {
                    "expiration_hours": 168
                }
            ]
        }
    }


class ShareLinkResponseSchema(BaseModel):
    """Schema for share link response"""
    share_id: str = Field(..., description="Unique share identifier (UUID)")
    session_id: str = Field(..., description="Associated session ID")
    share_url: str = Field(..., description="Public share URL")
    created_at: datetime = Field(..., description="Share link creation timestamp")
    expires_at: datetime = Field(..., description="Share link expiration timestamp")
    access_count: int = Field(default=0, description="Number of times this link has been accessed")
    last_accessed_at: Optional[datetime] = Field(default=None, description="Last access timestamp")
    
    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "share_id": "770e8400-e29b-41d4-a716-446655440222",
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "share_url": "https://api.example.com/api/shared/770e8400-e29b-41d4-a716-446655440222",
                    "created_at": "2025-11-26T10:00:00Z",
                    "expires_at": "2025-11-27T10:00:00Z",
                    "access_count": 0,
                    "last_accessed_at": None
                }
            ]
        }
    }


class SharedSessionResponseSchema(BaseModel):
    """Schema for accessing a shared session (read-only)"""
    id: str = Field(..., description="Session ID")
    name: str = Field(..., description="Session name")
    created_at: datetime = Field(..., description="Session creation timestamp")
    prediction_count: int = Field(default=0, description="Number of predictions in this session")
    # Note: Omits user_id and other sensitive information for public access
    
    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "name": "Antibody Design Project",
                    "created_at": "2025-11-20T10:00:00Z",
                    "prediction_count": 12
                }
            ]
        }
    }
