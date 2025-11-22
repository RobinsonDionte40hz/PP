"""
Authentication-related Pydantic schemas
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import re


class UserRegisterRequest(BaseModel):
    """Request schema for user registration"""
    username: str = Field(..., min_length=3, max_length=50, description="Username (3-50 characters)")
    email: Optional[str] = Field(None, description="Email address (optional)")
    password: str = Field(..., min_length=8, max_length=72, description="Password (8-72 characters)")
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format"""
        # Username must start with letter or number
        if not v[0].isalnum():
            raise ValueError("Username must start with a letter or number")
        
        # Username can only contain alphanumeric, underscore, and hyphen
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens")
        
        return v.strip()
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Validate email format"""
        if v is None or v == "":
            return None
        
        # Simple email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email format")
        
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "username": "john_doe",
                    "email": "john@example.com",
                    "password": "SecurePass123!"
                }
            ]
        }
    }


class UserResponse(BaseModel):
    """Response schema for user data"""
    key_id: str = Field(..., description="User unique identifier (UUID)")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="Email address")
    created_at: str = Field(..., description="Account creation timestamp (ISO format)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "key_id": "550e8400-e29b-41d4-a716-446655440000",
                    "username": "john_doe",
                    "email": "john@example.com",
                    "created_at": "2025-11-22T10:30:00"
                }
            ]
        }
    }


class UserRegisterResponse(BaseModel):
    """Response schema for successful registration"""
    message: str = Field(..., description="Success message")
    user: UserResponse = Field(..., description="Created user data")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "User registered successfully",
                    "user": {
                        "key_id": "550e8400-e29b-41d4-a716-446655440000",
                        "username": "john_doe",
                        "email": "john@example.com",
                        "created_at": "2025-11-22T10:30:00"
                    }
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    detail: str = Field(..., description="Error message")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": "Username already exists"
                }
            ]
        }
    }
