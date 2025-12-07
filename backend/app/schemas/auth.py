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
    captcha_token: Optional[str] = Field(None, description="CAPTCHA verification token (required when CAPTCHA is enabled)")
    
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
                    "password": "SecurePass123!",
                    "captcha_token": "03AGdBq24..."
                }
            ]
        }
    }


class UserResponse(BaseModel):
    """Response schema for user data"""
    key_id: str = Field(..., description="User unique identifier (UUID)")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="Email address")
    role: str = Field(default="user", description="User role (user, developer, admin)")
    created_at: str = Field(..., description="Account creation timestamp (ISO format)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "key_id": "550e8400-e29b-41d4-a716-446655440000",
                    "username": "john_doe",
                    "email": "john@example.com",
                    "role": "user",
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


class UserLoginRequest(BaseModel):
    """Request schema for user login"""
    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "username": "john_doe",
                    "password": "SecurePass123!"
                }
            ]
        }
    }


class TokenResponse(BaseModel):
    """Response schema for JWT tokens"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer",
                    "expires_in": 1800
                }
            ]
        }
    }


class UserLoginResponse(BaseModel):
    """Response schema for successful login"""
    message: str = Field(..., description="Success message")
    user: UserResponse = Field(..., description="User profile data")
    tokens: TokenResponse = Field(..., description="Authentication tokens")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Login successful",
                    "user": {
                        "key_id": "550e8400-e29b-41d4-a716-446655440000",
                        "username": "john_doe",
                        "email": "john@example.com",
                        "created_at": "2025-11-22T10:30:00"
                    },
                    "tokens": {
                        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                        "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                        "token_type": "bearer",
                        "expires_in": 1800
                    }
                }
            ]
        }
    }


class LogoutResponse(BaseModel):
    """Response schema for logout"""
    message: str = Field(..., description="Success message")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Logout successful"
                }
            ]
        }
    }


class TokenRefreshRequest(BaseModel):
    """Request schema for token refresh"""
    refresh_token: str = Field(..., description="JWT refresh token")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
                }
            ]
        }
    }


class TokenRefreshResponse(BaseModel):
    """Response schema for token refresh"""
    access_token: str = Field(..., description="New JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer",
                    "expires_in": 1800
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


# Email Verification Schemas

class SendVerificationRequest(BaseModel):
    """Request schema for sending verification email"""
    force_resend: bool = Field(default=False, description="Force resend even if recently sent")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "force_resend": False
                }
            ]
        }
    }


class SendVerificationResponse(BaseModel):
    """Response schema for send verification endpoint"""
    message: str = Field(..., description="Status message")
    success: bool = Field(..., description="Whether email was sent")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Verification email sent",
                    "success": True
                }
            ]
        }
    }


class VerifyEmailRequest(BaseModel):
    """Request schema for email verification"""
    token: str = Field(..., description="Verification token from email")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "token": "abc123def456..."
                }
            ]
        }
    }


class VerifyEmailResponse(BaseModel):
    """Response schema for email verification"""
    message: str = Field(..., description="Status message")
    success: bool = Field(..., description="Whether verification succeeded")
    user: Optional[UserResponse] = Field(None, description="User data if verification succeeded")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Email verified successfully",
                    "success": True,
                    "user": {
                        "key_id": "550e8400-e29b-41d4-a716-446655440000",
                        "username": "john_doe",
                        "email": "john@example.com",
                        "role": "user",
                        "created_at": "2025-11-22T10:30:00"
                    }
                }
            ]
        }
    }


class VerificationStatusResponse(BaseModel):
    """Response schema for verification status"""
    email: Optional[str] = Field(None, description="User's email address")
    email_verified: bool = Field(..., description="Whether email is verified")
    verification_required: bool = Field(..., description="Whether verification is required")
    can_resend: bool = Field(..., description="Whether user can request new verification email")
    token_expires_at: Optional[str] = Field(None, description="When current verification token expires")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "email": "john@example.com",
                    "email_verified": False,
                    "verification_required": True,
                    "can_resend": True,
                    "token_expires_at": "2025-12-08T10:30:00+00:00"
                }
            ]
        }
    }
