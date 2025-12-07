"""
User API endpoints

Endpoints for user account management including:
- Quota information
- Account settings
"""
from fastapi import APIRouter, HTTPException, Depends, Request, status
from typing import Dict, Any
from pydantic import BaseModel
from datetime import datetime
from app.security import require_auth_with_session
from app.services.quota_service import quota_service
from app.database import get_db
from app.models.user import User
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging
import os

logger = logging.getLogger(__name__)

# Initialize rate limiter (disabled in testing)
IS_TESTING = os.getenv("TESTING", "false").lower() == "true"
limiter = Limiter(key_func=get_remote_address, enabled=not IS_TESTING)

router = APIRouter()


# ========== Pydantic Response Schemas ==========

class QuotaInfoResponse(BaseModel):
    """Response schema for quota information"""
    account_tier: str
    daily: Dict[str, Any]
    monthly: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "account_tier": "free",
                "daily": {
                    "used": 5,
                    "limit": 20,
                    "remaining": 15,
                    "reset_at": "2025-12-07T00:00:00Z"
                },
                "monthly": {
                    "used": 25,
                    "limit": 100,
                    "remaining": 75,
                    "reset_at": "2025-12-01T00:00:00Z"
                }
            }
        }


class UserProfileResponse(BaseModel):
    """Response schema for user profile"""
    key_id: str
    username: str
    email: str | None
    role: str
    account_tier: str
    created_at: str | None
    quota: QuotaInfoResponse


# ========== Endpoints ==========

@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get current user profile",
    description="Get the authenticated user's profile including quota information"
)
@limiter.limit("30/minute")
async def get_current_user_profile(
    request: Request,
    user: Dict[str, Any] = Depends(require_auth_with_session)
):
    """
    Get the current authenticated user's profile.
    
    Returns user information and quota status.
    """
    user_id = user.get("sub") or user.get("key_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token"
        )
    
    db = next(get_db())
    try:
        db_user = db.query(User).filter(User.key_id == user_id).first()
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        quota_info = quota_service.get_user_quota(user_id)
        
        return UserProfileResponse(
            key_id=db_user.key_id,
            username=db_user.username,
            email=db_user.email,
            role=db_user.role,
            account_tier=db_user.account_tier,
            created_at=db_user.created_at.isoformat() if db_user.created_at else None,
            quota=QuotaInfoResponse(**quota_info)
        )
    finally:
        db.close()


@router.get(
    "/me/quota",
    response_model=QuotaInfoResponse,
    summary="Get current user quota",
    description="Get the authenticated user's prediction quota status"
)
@limiter.limit("60/minute")  # Allow frequent checks for UI updates
async def get_current_user_quota(
    request: Request,
    user: Dict[str, Any] = Depends(require_auth_with_session)
):
    """
    Get the current authenticated user's quota information.
    
    Returns:
    - account_tier: User's subscription tier (free, pro, enterprise)
    - daily: Daily quota info (used, limit, remaining, reset_at)
    - monthly: Monthly quota info (used, limit, remaining, reset_at)
    
    Use this endpoint to display quota status in the UI and determine
    if the user can create more predictions.
    """
    user_id = user.get("sub") or user.get("key_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token"
        )
    
    quota_info = quota_service.get_user_quota(user_id)
    
    if not quota_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return QuotaInfoResponse(**quota_info)


@router.get(
    "/me/quota/check",
    summary="Check if user has quota remaining",
    description="Quick check to see if user can create a prediction"
)
@limiter.limit("120/minute")  # Allow very frequent checks
async def check_quota_available(
    request: Request,
    user: Dict[str, Any] = Depends(require_auth_with_session)
):
    """
    Quick check to see if the user can create a new prediction.
    
    Returns:
    - can_create: Boolean indicating if user has quota remaining
    - message: Human-readable message about quota status
    """
    user_id = user.get("sub") or user.get("key_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token"
        )
    
    has_quota, error_message = quota_service.check_quota(user_id)
    
    if has_quota:
        quota_info = quota_service.get_user_quota(user_id)
        return {
            "can_create": True,
            "message": f"You have {quota_info['daily']['remaining']} predictions remaining today",
            "quota": quota_info
        }
    else:
        quota_info = quota_service.get_user_quota(user_id)
        return {
            "can_create": False,
            "message": error_message,
            "quota": quota_info
        }
